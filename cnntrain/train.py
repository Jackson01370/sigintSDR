"""cnntrain (6/6): 学習 CLI + チェックポイント + テキストログ。

`python -m cnntrain.train --data simdata/ --epochs E --out runs/<name>/`

チェックポイント（runs/<name>/checkpoint.pt）に保存するもの（作業指示）:
  * モデル重み (state_dict)
  * クラス名一覧
  * SIGSCAN_REP_VERSION
  * SYNTHETIC-ONLY メタ（合成のみ・ギャップ未測定の明記）
  * 生成シード（データの sigscan:gen_seed を検出）＋学習シード

CPU 前提。小さく速く（火入れであって精度狩りではない）。
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sigmf_io
from cnntrain import classes, data, evaluate
from cnntrain.model import build_model


def _detect_gen_seed(records) -> int | None:
    """データ側 meta の global sigscan:gen_seed を 1 件読み取る（記録用）。"""
    for r in records:
        try:
            _, meta = sigmf_io.read_recording(r.path)
            v = meta.get("global", {}).get("sigscan:gen_seed")
            if v is not None:
                return int(v)
        except Exception:
            continue
    return None


def _select_device(cuda_available: bool | None = None) -> "torch.device":
    """device 自動選択（純関数・テスト可能）: CUDA が使えれば cuda、無ければ cpu。

    cuda_available を明示指定するとその値で分岐する（テスト用）。None なら
    torch.cuda.is_available() を見る。GPU 前提の決め打ちにしない＝CPU 後方互換。
    """
    if cuda_available is None:
        cuda_available = torch.cuda.is_available()
    return torch.device("cuda" if cuda_available else "cpu")


def _epoch_pass(model, loader, criterion, optimizer=None, device=None):
    """1 epoch 分を回す。optimizer 指定時は学習、None なら評価。returns (loss, acc).

    device 未指定なら model の載っている device に合わせる（CPU 後方互換）。
    """
    if device is None:
        device = next(model.parameters()).device
    train = optimizer is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        if train:
            optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        if train:
            loss.backward()
            optimizer.step()
        loss_sum += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += x.size(0)
    n = max(1, total)
    return loss_sum / n, correct / n


def run_training(data_dir: str, out_dir: str, epochs: int = 8,
                 batch_size: int = 16, lr: float = 1e-3, seed: int = 0,
                 val_ratio: float = 0.2, run_name: str | None = None,
                 log=print) -> dict:
    """学習を実行し、チェックポイント・ログ・評価レポートを out_dir に書く。

    returns: dict(ckpt_path, report_txt, report_json, classes, history,
                  val_accuracy, n_train, n_val)
    """
    os.makedirs(out_dir, exist_ok=True)
    run_name = run_name or os.path.basename(os.path.normpath(out_dir))

    # 再現性。
    torch.manual_seed(seed)
    np.random.seed(seed)

    # device 自動選択: CUDA が使えれば GPU、無ければ従来どおり CPU（後方互換）。
    device = _select_device()
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)   # GPU 側 RNG（Dropout 等）も同一 seed で固定

    train_recs, val_recs, class_names = data.load_split(
        data_dir, val_ratio=val_ratio, seed=seed)
    gen_seed = _detect_gen_seed(train_recs + val_recs)

    train_ds = data.SpecDataset(train_recs, class_names)
    val_ds = data.SpecDataset(val_recs, class_names)
    # num_workers=0: Windows + CPU 火入れではプロセス生成オーバヘッドを避ける。
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0)

    model = build_model(n_classes=len(class_names), in_ch=1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_params = sum(p.numel() for p in model.parameters())

    # --- ログヘッダ（バナー）---
    log_path = os.path.join(out_dir, "train_log.txt")
    log_lines: list[str] = []

    def emit(s: str):
        log(s)
        log_lines.append(s)

    emit("=" * 72)
    emit("  cnntrain.train — 学習（SYNTHETIC-ONLY）")
    emit("=" * 72)
    for line in classes.SYNTHETIC_ONLY_LINES:
        emit("  !! " + line)
    emit("-" * 72)
    emit(f"  run        : {run_name}")
    emit(f"  data       : {data_dir}")
    emit(f"  classes    : {', '.join(class_names)}")
    emit(f"  train/val  : {len(train_ds)}/{len(val_ds)}  (val_ratio={val_ratio})")
    emit(f"  epochs     : {epochs}   batch: {batch_size}   lr: {lr}")
    emit(f"  seed       : train={seed}  gen={gen_seed}")
    emit(f"  rep_version: {classes.REP_VERSION}   params: {n_params}")
    dev_str = ("cpu" if device.type == "cpu"
               else f"cuda:{torch.cuda.current_device()} "
                    f"({torch.cuda.get_device_name()})")
    emit(f"  device     : {dev_str}   torch: {torch.__version__}")
    emit("-" * 72)

    history: list[dict] = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = _epoch_pass(model, train_loader, criterion, optimizer,
                                      device)
        va_loss, va_acc = _epoch_pass(model, val_loader, criterion, None, device)
        dt = time.time() - t0
        history.append(dict(epoch=ep, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc, elapsed_s=dt))
        emit(f"  epoch {ep:3d}/{epochs}  "
             f"train_loss={tr_loss:.4f} acc={tr_acc*100:5.1f}%   "
             f"val_loss={va_loss:.4f} acc={va_acc*100:5.1f}%   "
             f"[{dt:5.1f}s]")
    train_secs = time.time() - t0
    emit("-" * 72)
    emit(f"  学習完了: {train_secs:.1f}s")

    # checkpoint は CPU 化して保存する。GPU で学習しても、保存物は CPU テンソルの
    # state_dict とし、CPU 推論（--cnn 収集・既存テスト）でそのまま読めるようにする
    # （読込側 infer.load_checkpoint は map_location="cpu" で二重に安全）。保存する
    # dict のキー・メタは一切変えない。以降の評価もこの CPU モデル＋CPU val_loader
    # で走るため evaluate 経路は無改修（device 最適化は将来課題）。
    model.to("cpu")

    # --- チェックポイント保存 ---
    meta = dict(
        run_name=run_name,
        rep_version=classes.REP_VERSION,
        synthetic_only=True,
        synthetic_only_note=classes.SYNTHETIC_ONLY_TAG,
        synthetic_only_lines=classes.SYNTHETIC_ONLY_LINES,
        train_seed=int(seed),
        gen_seed=(int(gen_seed) if gen_seed is not None else None),
        epochs=int(epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        n_train=len(train_ds),
        n_val=len(val_ds),
        n_params=int(n_params),
        torch_version=str(torch.__version__),
        data_dir=str(data_dir),
        final_val_acc=float(history[-1]["val_acc"]) if history else 0.0,
    )
    ckpt_path = os.path.join(out_dir, "checkpoint.pt")
    torch.save(dict(state_dict=model.state_dict(), classes=class_names,
                    in_ch=1, meta=meta), ckpt_path)
    emit(f"  チェックポイント: {ckpt_path}")

    # ログ書き出し（人間向け成果物は UTF-8 固定）。
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    # history を JSON でも残す。
    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    # --- 評価レポート（混同行列 + バナー）---
    result = evaluate.evaluate_model(model, val_loader, len(class_names))
    report_meta = dict(meta)
    txt_path, json_path = evaluate.write_report(out_dir, result, class_names,
                                                report_meta)
    emit(f"  評価レポート    : {txt_path}")
    emit("")
    emit(evaluate.format_report(result, class_names, report_meta))

    return dict(ckpt_path=ckpt_path, report_txt=txt_path, report_json=json_path,
                classes=class_names, history=history,
                val_accuracy=result["accuracy"],
                n_train=len(train_ds), n_val=len(val_ds))


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
        prog="cnntrain.train",
        description="合成 SigMF で軽量 CNN を学習（CPU・火入れ・SYNTHETIC-ONLY）")
    p.add_argument("--data", required=True, help="SigMF データディレクトリ")
    p.add_argument("--out", required=True, help="出力先 runs/<name>/")
    p.add_argument("--epochs", type=int, default=8, help="エポック数（既定 8）")
    p.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    p.add_argument("--lr", type=float, default=1e-3, help="学習率（既定 1e-3）")
    p.add_argument("--seed", type=int, default=0, help="学習/分割シード（既定 0）")
    p.add_argument("--val-ratio", type=float, default=0.2, dest="val_ratio")
    p.add_argument("--name", default=None, help="run 名（既定: out のベース名）")
    args = p.parse_args(argv)

    run_training(args.data, args.out, epochs=args.epochs,
                 batch_size=args.batch_size, lr=args.lr, seed=args.seed,
                 val_ratio=args.val_ratio, run_name=args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
