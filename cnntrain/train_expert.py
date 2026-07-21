"""cnntrain: 2.4GHz ISM 専門家 CNN の学習（実データのみ・案A / 用途3クラス）。

汎用 sim CNN（`runs/m2_5`・方式軸5クラス）とは **別物** で、用途3クラス
（ble-adv / wifi-24 / spurious）を **`method=human` の実データだけ** で新規学習する。

設計（指示書_専門家CNN学習.md）:
  * 合成データを混ぜない（案A / Pattern B 純粋）。教師は human確定ラベルのみ。
    ルール/CNN/LLM 出力は教師にしない（Pattern A 禁止）。
  * 既存 sim 経路（`data.load_split` / `train.run_training`）は **不変**。ここは別入口。
  * 入力は既存 `data.SpecDataset`＋凍結 `spec.render` を **流用**（迂回しない）。
    レコードの label を用途クラスへ差し替える（`dataclasses.replace`）だけで再利用できる。
  * 少数・不均衡対策として **層化分割** と **逆頻度クラス重み**。
  * 既存モデルを上書きしない（**新規 out_dir**、例 `runs/ism24_v2/`）。
  * ルーティング（実運用でバンド別に呼ぶ配管）は今回やらない（別タスク）。

v2 追加（指示書_専門家CNN再学習.md・データ増量 91→192 件）:
  * **best-val 保存**: 最終 epoch ではなく val accuracy 最良 epoch の重みを保存（`_train_loop`）。
  * **early-stopping**: patience（既定10）で val 停滞時に打ち切り、成果物は best-val の重み。
  * **stratified k-fold**（既定 k=5）: 各 fold を best-val+early-stop で学習し val 評価。
    平均±標準偏差・合算混同行列で「val=十数件で 1件が数%動く」脆さを補強（評価専用）。
  * 最終 checkpoint は (a) 全データの層化 80/20 split で best-val 学習した1本。入口は
    `run_expert_v2`（k-fold 評価 → 最終学習）。
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


def stratified_kfold(items, k: int = 5, seed: int = 0):
    """(Record, cls) を **クラス層化 k-fold** に分ける（評価の信頼性向上）。

    各クラスを独立にシャッフルし、位置 i を fold(i % k) の val へ割り当てる
    （クラスごとに val が均等に散る＝層化）。fold f の train は f 以外の全 val の
    和集合。返り値は k 個の (train, val)。全 val は **重複なく全体を覆う**。
    seed 固定で決定的。クラスサイズ >= k なら各 fold の val に全クラスが最低1件入る。

    stratified_split（単一 train/val）とは別に、k 分割で平均±分散を出すため使う。
    """
    if k < 2:
        raise ValueError(f"k-fold には k>=2 が必要: k={k}")
    by_cls: dict[str, list] = {}
    for rec, cls in items:
        by_cls.setdefault(cls, []).append((rec, cls))
    rng = np.random.default_rng(seed)
    folds_val: list[list] = [[] for _ in range(k)]
    for cls in sorted(by_cls):
        pairs = sorted(by_cls[cls], key=lambda t: t[0].path)   # 安定順
        perm = rng.permutation(len(pairs))
        for i, j in enumerate(perm):
            folds_val[i % k].append(pairs[int(j)])
    splits = []
    for f in range(k):
        val = list(folds_val[f])
        train = [p for g in range(k) if g != f for p in folds_val[g]]
        splits.append((train, val))
    return splits


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


def _snapshot_state(model) -> dict:
    """model の state_dict を CPU クローンで凍結スナップショットする（best-val 保持用）。"""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _train_loop(model, train_loader, val_loader, criterion, optimizer, device,
                epochs: int, patience: int | None = None, emit=lambda s: None):
    """学習ループ本体。**best-val 重みを保持**し、**early-stopping** を行う。

    返り値: (history, best) — best は dict(epoch, val_acc, state)。
      * best["state"] は **val accuracy が最良だった epoch の重み**の CPU スナップショット
        （最終 epoch ではない）。保存/デプロイはこれを使う。
      * patience 指定時、val が patience epoch 連続で更新されなければ停止し、
        best["state"]（停止時点ではなく最良）を最終成果物にする。
      * best["stopped_epoch"] に実際に回った最終 epoch を記録。
    既存 `_epoch_pass`（train.py・凍結の学習1周）をそのまま使う。
    """
    history: list[dict] = []
    best = dict(epoch=0, val_acc=-1.0, state=None, stopped_epoch=epochs)
    since_improve = 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = _epoch_pass(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = _epoch_pass(model, val_loader, criterion, None, device)
        dt = time.time() - t0
        history.append(dict(epoch=ep, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc, elapsed_s=dt))
        improved = va_acc > best["val_acc"]
        if improved:
            best.update(epoch=ep, val_acc=va_acc, state=_snapshot_state(model))
            since_improve = 0
        else:
            since_improve += 1
        emit(f"  epoch {ep:3d}/{epochs}  train_loss={tr_loss:.4f} acc={tr_acc*100:5.1f}%   "
             f"val_loss={va_loss:.4f} acc={va_acc*100:5.1f}%  [{dt:4.1f}s]"
             + ("  *best" if improved else ""))
        best["stopped_epoch"] = ep
        if patience is not None and patience > 0 and since_improve >= patience:
            emit(f"  early-stop: val が {patience} epoch 連続で未改善 → epoch {ep} で停止 "
                 f"(best epoch {best['epoch']} / val {best['val_acc']*100:.1f}%)")
            break
    return history, best


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
    n_total = sum(counts.get(c, 0) for c in class_names)
    L.append(f"[まとめ] {n_total}件・val {meta.get('n_val')}件規模。過学習/データ数の限界に留意し、"
             "単一 val の数値は参考値（信頼できる評価は k-fold 平均±分散を見よ）。")
    return "\n".join(L)


def run_expert_training(data_dir: str, out_dir: str, epochs: int = 8,
                        batch_size: int = 16, lr: float = 1e-3, seed: int = 0,
                        val_ratio: float = 0.2, hw: str = "real",
                        run_name: str | None = None, log=print,
                        patience: int | None = None,
                        class_names: list[str] | None = None,
                        kfold_summary: dict | None = None) -> dict:
    """2.4GHz 専門家 CNN を実データで学習し out_dir に checkpoint/レポートを書く。

    **best-val 保存**（最終 epoch ではなく val accuracy 最良の重みを保存）。
    `patience` 指定で **early-stopping**（既定 None＝止めずに全 epoch 回す）。
    `kfold_summary` を渡すと meta に k-fold 評価サマリを併記する（評価は別関数）。

    既存 `runs/m2_5`・`runs/ism24_v1` は触らない（out_dir は新規指定）。returns: 結果 dict。
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
    class_names = list(class_names) if class_names else list(EXPERT_CLASSES)
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
    emit(f"  patience   : {patience if patience else 'なし(全epoch)'}  保存: best-val")
    dev = "cpu" if device.type == "cpu" else f"cuda ({torch.cuda.get_device_name()})"
    emit(f"  device     : {dev}  torch: {torch.__version__}  params: {n_params}")
    emit("-" * 72)

    t0 = time.time()
    history, best = _train_loop(model, train_loader, val_loader, criterion,
                                optimizer, device, epochs, patience=patience,
                                emit=emit)
    emit("-" * 72)
    emit(f"  完了: {time.time()-t0:.1f}s  best val_acc={best['val_acc']*100:.1f}% "
         f"(epoch {best['epoch']} / 実行 {best['stopped_epoch']} epoch)  保存=best-val")

    # best-val の重みをデプロイ物にする（最終 epoch ではなく最良 epoch の重み）。
    model.to("cpu")
    if best["state"] is not None:
        model.load_state_dict(best["state"])
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
        saved="best-val", patience=(int(patience) if patience else None),
        stopped_epoch=int(best["stopped_epoch"]),
        kfold=kfold_summary,
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


def _format_kfold_report(summary: dict, class_names: list[str]) -> str:
    """k-fold 評価レポート（**平均±標準偏差** と合算混同行列・正直バナー）。"""
    line = "=" * 72
    L = [line, "  2.4GHz ISM 専門家 CNN — k-fold 交差検証（REAL-DATA・用途3クラス）", line,
         "  " + "!" * 68,
         "  !! REAL-DATA-ONLY: human確定 SigMF のみ（合成非混合＝案A / Pattern B 純粋）。",
         "  !! 単一 val の高い数字ではなく **k-fold 平均±標準偏差** で実力を語る。",
         "  " + "!" * 68,
         f"  k          : {summary['k']}   epochs(max): {summary['epochs']}   "
         f"patience: {summary['patience']}   seed: {summary['seed']}",
         f"  全件        : {summary['n_total']}   classes: {', '.join(class_names)}",
         line, ""]
    L.append("[各 fold の val accuracy]（best-val 重みで評価）")
    for r in summary["folds"]:
        L.append(f"  fold {r['fold']}: {r['val_acc']*100:5.1f}%  "
                 f"(n_val={r['n_val']}, best_epoch={r['best_epoch']}, "
                 f"stopped={r['stopped_epoch']})")
    L.append("")
    L.append(f"[平均±標準偏差] val accuracy = {summary['mean_val_acc']*100:.1f}% "
             f"± {summary['std_val_acc']*100:.1f}%   (k={summary['k']}・母集団標準偏差)")
    L.append("")
    L.append("[合算混同行列]  行=真値(true) / 列=予測(pred)  ※全 fold の val を合算")
    cw = min(max(6, *(len(c) for c in class_names)), 12)
    short = [c[:cw] for c in class_names]
    L.append(" " * (cw + 3) + "".join(f"{s:>{cw+2}}" for s in short))
    cm = summary["confusion"]
    for i, c in enumerate(class_names):
        cells = "".join(f"{cm[i][j]:>{cw+2}}" for j in range(len(class_names)))
        L.append(f"  {c[:cw]:<{cw}} |{cells}")
    L.append("")
    L.append("[クラス別 precision / recall]（合算 val 基準）")
    for row in summary["per_class"]:
        L.append(f"  {row['cls']:10s}  precision={row['precision']*100:5.1f}%  "
                 f"recall={row['recall']*100:5.1f}%  "
                 f"(tp={row['tp']}/support={row['support']}, pred={row['pred']})")
    L.append("")
    L.append("[正直な注記] spurious は固定周波数線（2440MHz 等）で分離が容易＝易しいクラス。"
             "難所は ble-adv↔wifi-24 の境界。数値は k-fold 平均で読むこと。")
    return "\n".join(L)


def run_expert_kfold(data_dir: str, out_dir: str, k: int = 5, epochs: int = 40,
                     batch_size: int = 16, lr: float = 1e-3, seed: int = 0,
                     patience: int | None = 10, hw: str = "real",
                     class_names: list[str] | None = None, log=print) -> dict:
    """**stratified k-fold 交差検証**で実力を評価する（checkpoint は残さない・評価専用）。

    各 fold: 層化 train/val に分け、best-val + early-stop で学習し **best-val 重みで val 評価**。
    k 個の val accuracy の平均±標準偏差、**全 fold 合算の混同行列**、クラス別 precision/recall
    を返す（前回の「val=18件で 1件 5.6%」の脆さを平均で補強）。
    fold の一時学習は out_dir 内で完結し、`runs/m2_5`・`runs/ism24_v1` を汚さない。
    """
    os.makedirs(out_dir, exist_ok=True)
    device = _select_device()
    class_names = list(class_names) if class_names else list(EXPERT_CLASSES)
    K = len(class_names)
    items = collect_expert_records(data_dir, hw=hw)
    if not items:
        raise RuntimeError(
            f"実データ(human・3ラベル)が見つかりません: {data_dir} (hw={hw})。")
    splits = stratified_kfold(items, k=k, seed=seed)

    log_lines: list[str] = []

    def emit(s):
        log(s)
        log_lines.append(s)

    emit("=" * 72)
    emit(f"  cnntrain.train_expert — k-fold 交差検証（k={k}・REAL-DATA・案A）")
    emit("=" * 72)
    emit(f"  data: {data_dir} (hw={hw})  全件: {len(items)}  classes: {', '.join(class_names)}")
    emit(f"  epochs(max): {epochs}  patience: {patience}  batch: {batch_size}  lr: {lr}  seed: {seed}")
    emit("-" * 72)

    agg_cm = [[0 for _ in range(K)] for _ in range(K)]
    fold_rows = []
    for f, (train_items, val_items) in enumerate(splits, 1):
        torch.manual_seed(seed + f)          # fold ごとに独立初期化・決定的
        np.random.seed(seed + f)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + f)
        weights = inverse_freq_weights(train_items, class_names)
        train_ds = _make_dataset(train_items, class_names)
        val_ds = _make_dataset(val_items, class_names)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=0)
        model = build_model(n_classes=K, in_ch=1).to(device)
        weight_t = torch.tensor(weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight_t)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        emit(f"[fold {f}/{k}] train/val={len(train_ds)}/{len(val_ds)}  "
             f"weights=" + " ".join(f"{c}:{w:.2f}" for c, w in zip(class_names, weights)))
        _, best = _train_loop(model, train_loader, val_loader, criterion,
                              optimizer, device, epochs, patience=patience,
                              emit=lambda s: None)
        model.to("cpu")
        if best["state"] is not None:
            model.load_state_dict(best["state"])
        result = evaluate.evaluate_model(model, val_loader, K)
        for i in range(K):
            for j in range(K):
                agg_cm[i][j] += result["confusion"][i][j]
        fold_rows.append(dict(fold=f, val_acc=float(result["accuracy"]),
                              n_val=len(val_ds), correct=result["correct"],
                              n=result["n"], best_epoch=int(best["epoch"]),
                              stopped_epoch=int(best["stopped_epoch"]),
                              confusion=result["confusion"]))
        emit(f"  → fold {f}: val_acc={result['accuracy']*100:5.1f}% "
             f"({result['correct']}/{result['n']})  best_epoch={best['epoch']} "
             f"stopped={best['stopped_epoch']}")

    accs = [r["val_acc"] for r in fold_rows]
    mean = float(np.mean(accs)) if accs else 0.0
    std = float(np.std(accs)) if accs else 0.0
    correct = sum(agg_cm[i][i] for i in range(K))
    n_all = sum(agg_cm[i][j] for i in range(K) for j in range(K))
    agg_result = dict(confusion=agg_cm, accuracy=(correct / n_all if n_all else 0.0),
                      n=n_all, correct=correct)
    per_class = _per_class_precision_recall(agg_result, class_names)
    summary = dict(k=int(k), epochs=int(epochs),
                   patience=(int(patience) if patience else None),
                   seed=int(seed), classes=class_names, n_total=len(items),
                   fold_val_acc=accs, mean_val_acc=mean, std_val_acc=std,
                   pooled_val_acc=agg_result["accuracy"], confusion=agg_cm,
                   per_class=per_class, folds=fold_rows)

    emit("-" * 72)
    report = _format_kfold_report(summary, class_names)
    emit("")
    emit(report)
    with open(os.path.join(out_dir, "kfold_report.txt"), "w", encoding="utf-8") as fp:
        fp.write(report + "\n")
    with open(os.path.join(out_dir, "kfold_report.json"), "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "kfold_log.txt"), "w", encoding="utf-8") as fp:
        fp.write("\n".join(log_lines) + "\n")
    return summary


def run_expert_v2(data_dir: str, out_dir: str, k: int = 5, epochs: int = 40,
                  batch_size: int = 16, lr: float = 1e-3, seed: int = 0,
                  val_ratio: float = 0.2, patience: int | None = 10,
                  hw: str = "real", class_names: list[str] | None = None,
                  run_name: str | None = None, log=print) -> dict:
    """再学習の入口: **k-fold 評価** → **最終 checkpoint**（best-val + early-stop）。

    最終 checkpoint の基準は **(a) 全データの層化 80/20 split で best-val 学習した1本**
    （k-fold は評価専用で checkpoint を残さない）。k<2 なら k-fold を省略し最終学習のみ。
    """
    class_names = list(class_names) if class_names else list(EXPERT_CLASSES)
    summary = None
    if k and k >= 2:
        summary = run_expert_kfold(data_dir, out_dir, k=k, epochs=epochs,
                                   batch_size=batch_size, lr=lr, seed=seed,
                                   patience=patience, hw=hw,
                                   class_names=class_names, log=log)
    final = run_expert_training(data_dir, out_dir, epochs=epochs,
                                batch_size=batch_size, lr=lr, seed=seed,
                                val_ratio=val_ratio, hw=hw, run_name=run_name,
                                log=log, patience=patience,
                                class_names=class_names, kfold_summary=summary)
    return dict(kfold=summary, final=final)


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
        description="2.4GHz ISM 専門家 CNN を実データのみで学習（案A・3クラス・"
                    "best-val/early-stop/k-fold）")
    p.add_argument("--data", default="captures/", help="SigMF データ（既定 captures/）")
    p.add_argument("--out", required=True, help="出力先（新規。例 runs/ism24_v2）")
    p.add_argument("--epochs", type=int, default=40, help="最大 epoch（既定 40）")
    p.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-ratio", type=float, default=0.2, dest="val_ratio")
    p.add_argument("--hw", default="real", help="出所グループ（既定 real）")
    p.add_argument("--name", default=None)
    p.add_argument("--kfold", type=int, default=5,
                   help="k-fold 交差検証の k（既定 5・<2 で k-fold 省略）")
    p.add_argument("--early-stop", type=int, default=10, dest="patience",
                   help="early-stopping の patience（既定 10・<=0 で無効）")
    p.add_argument("--best-val", action="store_true",
                   help="best-val 保存を明示（既定で常に best-val 保存）")
    p.add_argument("--classes", default=None,
                   help="クラス（既定 ble-adv,wifi-24,spurious）")
    args = p.parse_args(argv)

    class_names = None
    if args.classes:
        req = [c.strip() for c in args.classes.split(",") if c.strip()]
        unknown = [c for c in req if c not in EXPERT_CLASSES]
        if unknown:
            p.error(f"未知のクラス: {unknown}（既知: {EXPERT_CLASSES}）")
        class_names = req
    patience = args.patience if (args.patience and args.patience > 0) else None

    run_expert_v2(args.data, args.out, k=args.kfold, epochs=args.epochs,
                  batch_size=args.batch_size, lr=args.lr, seed=args.seed,
                  val_ratio=args.val_ratio, patience=patience, hw=args.hw,
                  class_names=class_names, run_name=args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
