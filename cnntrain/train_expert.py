"""cnntrain: 2.4GHz ISM 専門家 CNN の学習（実データのみ・案A / 用途3クラス）。

汎用 sim CNN（`runs/m2_5`・方式軸5クラス）とは **別物** で、用途3クラス
（ble-adv / wifi-24 / spurious）を **`method=human` の実データだけ** で新規学習する。

設計（指示書_専門家CNN学習.md）:
  * 合成データを混ぜない（案A / Pattern B 純粋）。教師は human確定ラベルのみ。
    ルール/CNN/LLM 出力は教師にしない（Pattern A 禁止）。
  * 既存 sim 経路（`data.load_split` / `train.run_training`）は **不変**。ここは別入口。
  * 入力は既存 `data.SpecDataset`＋凍結 `spec.render` を **流用**（迂回しない）。
    レコードの label を用途クラスへ差し替える（`dataclasses.replace`）だけで再利用できる。
  * 少数・不均衡（36/23/32）対策として **層化分割** と **逆頻度クラス重み** を新規実装。
  * 既存モデルを上書きしない（**新規 out_dir**、例 `runs/ism24_v1/`）。
  * ルーティング（実運用でバンド別に呼ぶ配管）は今回やらない（別タスク）。
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import classify
import dataset as ds_mod
import spec
from cnntrain import data as cdata
from cnntrain import evaluate
from cnntrain.model import build_model
from cnntrain.train import _epoch_pass, _select_device

# --- SigMF core:label（実データ上の表記）→ 専門家クラス（用途1軸・3クラス）---
#   これ以外（"未識別信号" 等）は学習対象外（除外）。写像は1箇所に集約。
#   spurious の正準表記は classify.SPURIOUS（"スプリアス(HackRF内部)"）を単一の真実にする。
LABEL_MAP: dict[str, str] = {
    "BLE/Bluetooth (adv?)": "ble-adv",
    "WiFi (2.4GHz, 20/40MHz)": "wifi-24",
    classify.SPURIOUS: "spurious",
}

# クラス順（checkpoint / class_to_idx / 重み / 評価で一貫させる。指示書の想定順に固定）。
EXPERT_CLASSES: list[str] = ["ble-adv", "wifi-24", "spurious"]


def map_label(core_label: str | None) -> str | None:
    """SigMF core:label → 専門家クラス。対象外は None（＝学習から除外）。"""
    return LABEL_MAP.get((core_label or "").strip())


def collect_expert_records(data_dir: str, hw: str = "real"):
    """`method=human` かつ3ラベルの **実データ** Record を集める。

    returns: list[(Record, expert_class)]。
    合成（hw=sim）は hw フィルタで構造的に除外＝案A（Pattern B 純粋）を索引段で保証。
    rule/cnn の annotation は method フィルタで除外＝Pattern A を防ぐ。
    """
    idx = ds_mod.load_index(data_dir)
    out = []
    for r in idx.query(hw=hw):
        if (r.method or "") != "human":       # human確定のみ（ルール/CNN出力を教師にしない）
            continue
        cls = map_label(r.label)
        if cls is None:                       # 3クラス外（未識別信号 等）は除外
            continue
        out.append((r, cls))
    return out


def stratified_split(items, val_ratio: float = 0.2, seed: int = 0):
    """(Record, cls) を **クラス層化** で train/val に分割する（少数データ対策）。

    各クラスを独立に val_ratio で分け、**val に全クラスが最低1件**入るよう保証する
    （最小クラスでも 1 件は val へ）。同時に train も最低1件残す。seed 固定で決定的。
    既存 `dataset.split`（hw内ランダム・非層化）は使わない。
    """
    by_cls: dict[str, list] = {}
    for rec, cls in items:
        by_cls.setdefault(cls, []).append(rec)
    rng = np.random.default_rng(seed)
    train, val = [], []
    for cls in sorted(by_cls):
        recs = sorted(by_cls[cls], key=lambda r: r.path)   # 安定順
        n = len(recs)
        perm = rng.permutation(n)
        n_val = max(1, int(round(n * val_ratio)))          # val に最低1件
        if n >= 2:
            n_val = min(n_val, n - 1)                       # train にも最低1件残す
        val_idx = set(int(i) for i in perm[:n_val])
        for i, r in enumerate(recs):
            (val if i in val_idx else train).append((r, cls))
    return train, val


def inverse_freq_weights(items, class_names: list[str]) -> list[float]:
    """逆頻度クラス重み: weight[c] = N / (K * count[c])（平均≈1に正規化）。

    count は与えた items（通常は train）での各クラス件数。0 件のクラスは重み 0。
    """
    counts = {c: 0 for c in class_names}
    for _, cls in items:
        if cls in counts:
            counts[cls] += 1
    n = sum(counts.values())
    k = len(class_names)
    return [(n / (k * counts[c])) if counts[c] else 0.0 for c in class_names]


def _make_dataset(items, class_names):
    """(Record, cls) 列 → SpecDataset。label を用途クラスへ差し替えて既存 Dataset を流用。

    spec.render を迂回しない（SpecDataset.__getitem__ がそのまま凍結表現を通す）。
    """
    recs = [replace(r, label=cls) for r, cls in items]
    return cdata.SpecDataset(recs, class_names)


def _per_class_precision_recall(result: dict, class_names: list[str]):
    """confusion（行=真値, 列=予測）から各クラスの precision/recall を出す。"""
    cm = result["confusion"]
    k = len(class_names)
    out = []
    for i in range(k):
        tp = cm[i][i]
        recall_den = sum(cm[i][j] for j in range(k))            # 真値 i の総数
        prec_den = sum(cm[t][i] for t in range(k))              # 予測 i の総数
        recall = (tp / recall_den) if recall_den else 0.0
        prec = (tp / prec_den) if prec_den else 0.0
        out.append(dict(cls=class_names[i], tp=tp, recall=recall,
                        precision=prec, support=recall_den, pred=prec_den))
    return out


def _format_expert_report(result, class_names, meta, weights, counts) -> str:
    """実データ専門家の評価レポート（**正直バナー**・SYNTHETIC-ONLY ではない）。"""
    line = "=" * 72
    L = [line, "  2.4GHz ISM 専門家 CNN 評価（REAL-DATA・用途3クラス）", line,
         "  " + "!" * 68,
         "  !! REAL-DATA-ONLY: human確定した実測 SigMF のみで学習（合成非混合＝案A）。",
         "  !! 教師は method=human のラベルのみ。ルール/CNN/LLM 出力は不使用（Pattern B 純粋）。",
         f"  !! 少数データ: 学習{meta.get('n_train')}件 / 検証{meta.get('n_val')}件。"
         "数値は参考値（誇張しない）。",
         "  " + "!" * 68,
         f"  run        : {meta.get('run_name')}",
         f"  classes    : {', '.join(class_names)}",
         f"  クラス件数 : " + " / ".join(f"{c}={counts.get(c, 0)}" for c in class_names),
         f"  クラス重み : " + " / ".join(f"{c}={w:.3f}" for c, w in zip(class_names, weights)),
         f"  epochs     : {meta.get('epochs')}  batch: {meta.get('batch_size')}  "
         f"lr: {meta.get('lr')}  seed: {meta.get('train_seed')}",
         line, ""]
    acc = result["accuracy"]
    L.append(f"[全体] val accuracy = {acc*100:5.1f}%  ({result['correct']}/{result['n']})")
    L.append("")
    L.append("[クラス別 precision / recall]（support=val の真値件数）")
    for row in _per_class_precision_recall(result, class_names):
        L.append(f"  {row['cls']:10s}  precision={row['precision']*100:5.1f}%  "
                 f"recall={row['recall']*100:5.1f}%  "
                 f"(tp={row['tp']}/support={row['support']}, pred={row['pred']})")
    L.append("")
    L.append("[混同行列]  行=真値(true) / 列=予測(pred)")
    cw = min(max(6, *(len(c) for c in class_names)), 12)
    short = [c[:cw] for c in class_names]
    L.append(" " * (cw + 3) + "".join(f"{s:>{cw+2}}" for s in short))
    cm = result["confusion"]
    for i, c in enumerate(class_names):
        cells = "".join(f"{cm[i][j]:>{cw+2}}" for j in range(len(class_names)))
        L.append(f"  {c[:cw]:<{cw}} |{cells}")
    L.append("")
    L.append("[まとめ] 91件・val 十数件規模。過学習/データ数の限界に留意し、数値は参考値。")
    return "\n".join(L)


def run_expert_training(data_dir: str, out_dir: str, epochs: int = 8,
                        batch_size: int = 16, lr: float = 1e-3, seed: int = 0,
                        val_ratio: float = 0.2, hw: str = "real",
                        run_name: str | None = None, log=print) -> dict:
    """2.4GHz 専門家 CNN を実データで学習し out_dir に checkpoint/レポートを書く。

    既存 `runs/m2_5` は触らない（out_dir は新規指定）。returns: 結果 dict。
    """
    os.makedirs(out_dir, exist_ok=True)
    run_name = run_name or os.path.basename(os.path.normpath(out_dir))

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _select_device()
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    items = collect_expert_records(data_dir, hw=hw)
    if not items:
        raise RuntimeError(
            f"実データ(human・3ラベル)が見つかりません: {data_dir} "
            f"(hw={hw})。captures/ を確認。")
    train_items, val_items = stratified_split(items, val_ratio=val_ratio, seed=seed)
    class_names = list(EXPERT_CLASSES)
    all_counts = {c: 0 for c in class_names}
    for _, cls in items:
        all_counts[cls] += 1
    weights = inverse_freq_weights(train_items, class_names)   # train 頻度で重み

    train_ds = _make_dataset(train_items, class_names)
    val_ds = _make_dataset(val_items, class_names)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0)

    model = build_model(n_classes=len(class_names), in_ch=1).to(device)
    weight_t = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_t)           # 逆頻度重み付き
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())

    log_lines: list[str] = []

    def emit(s):
        log(s)
        log_lines.append(s)

    emit("=" * 72)
    emit("  cnntrain.train_expert — 2.4GHz ISM 専門家 CNN（REAL-DATA・案A）")
    emit("=" * 72)
    emit("  !! 合成非混合（Pattern B 純粋）。教師は method=human のみ。")
    emit(f"  data       : {data_dir}  (hw={hw})")
    emit(f"  classes    : {', '.join(class_names)}")
    emit(f"  全件/クラス: " + " / ".join(f"{c}={all_counts[c]}" for c in class_names))
    emit(f"  train/val  : {len(train_ds)}/{len(val_ds)}  (層化・val_ratio={val_ratio})")
    emit(f"  weights    : " + " / ".join(f"{c}={w:.3f}" for c, w in zip(class_names, weights)))
    emit(f"  epochs     : {epochs}  batch: {batch_size}  lr: {lr}  seed: {seed}")
    dev = "cpu" if device.type == "cpu" else f"cuda ({torch.cuda.get_device_name()})"
    emit(f"  device     : {dev}  torch: {torch.__version__}  params: {n_params}")
    emit("-" * 72)

    history = []
    best = dict(epoch=0, val_acc=-1.0)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = _epoch_pass(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = _epoch_pass(model, val_loader, criterion, None, device)
        dt = time.time() - t0
        history.append(dict(epoch=ep, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc, elapsed_s=dt))
        if va_acc > best["val_acc"]:
            best = dict(epoch=ep, val_acc=va_acc)
        emit(f"  epoch {ep:3d}/{epochs}  train_loss={tr_loss:.4f} acc={tr_acc*100:5.1f}%   "
             f"val_loss={va_loss:.4f} acc={va_acc*100:5.1f}%  [{dt:4.1f}s]")
    emit("-" * 72)
    emit(f"  完了: {time.time()-t0:.1f}s  best val_acc={best['val_acc']*100:.1f}% "
         f"(epoch {best['epoch']})")

    model.to("cpu")
    meta = dict(
        run_name=run_name, rep_version=spec.SIGSCAN_REP_VERSION,
        real_data=True, synthetic_only=False,
        source="human-confirmed real SigMF only (Pattern B pure, no synthetic)",
        classes=class_names, class_counts=all_counts, class_weights=weights,
        train_seed=int(seed), epochs=int(epochs), batch_size=int(batch_size),
        lr=float(lr), val_ratio=float(val_ratio), hw=str(hw),
        n_train=len(train_ds), n_val=len(val_ds), n_params=int(n_params),
        torch_version=str(torch.__version__), data_dir=str(data_dir),
        final_val_acc=float(history[-1]["val_acc"]) if history else 0.0,
        best_val_acc=float(best["val_acc"]), best_epoch=int(best["epoch"]),
    )
    ckpt_path = os.path.join(out_dir, "checkpoint.pt")
    torch.save(dict(state_dict=model.state_dict(), classes=class_names,
                    in_ch=1, meta=meta), ckpt_path)
    emit(f"  チェックポイント: {ckpt_path}")

    with open(os.path.join(out_dir, "train_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    result = evaluate.evaluate_model(model, val_loader, len(class_names))
    report = _format_expert_report(result, class_names, meta, weights, all_counts)
    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(dict(classes=class_names, meta=meta, result=result,
                       per_class=_per_class_precision_recall(result, class_names)),
                  f, indent=2, ensure_ascii=False)
    emit("")
    emit(report)

    return dict(ckpt_path=ckpt_path, classes=class_names, history=history,
                val_accuracy=result["accuracy"], best_val_acc=best["val_acc"],
                n_train=len(train_ds), n_val=len(val_ds), result=result,
                weights=weights, counts=all_counts)


def _force_utf8():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass


def main(argv=None) -> int:
    _force_utf8()
    import argparse
    p = argparse.ArgumentParser(
        prog="cnntrain.train_expert",
        description="2.4GHz ISM 専門家 CNN を実データのみで学習（案A・3クラス）")
    p.add_argument("--data", default="captures/", help="SigMF データ（既定 captures/）")
    p.add_argument("--out", required=True, help="出力先（新規。例 runs/ism24_v1）")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-ratio", type=float, default=0.2, dest="val_ratio")
    p.add_argument("--hw", default="real", help="出所グループ（既定 real）")
    p.add_argument("--name", default=None)
    args = p.parse_args(argv)
    run_expert_training(args.data, args.out, epochs=args.epochs,
                        batch_size=args.batch_size, lr=args.lr, seed=args.seed,
                        val_ratio=args.val_ratio, hw=args.hw, run_name=args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
