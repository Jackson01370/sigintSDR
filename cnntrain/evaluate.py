"""cnntrain (4/6): 評価（val accuracy + 混同行列）とレポート（バナー必須）。

レポート冒頭に **SYNTHETIC-ONLY バナー**（eval-harness の流儀）を必ず出す。
text と JSON の両方を書ける。
"""
from __future__ import annotations

import json
import sys

import torch

from cnntrain import classes


@torch.no_grad()
def evaluate_model(model, loader, n_classes: int) -> dict:
    """val ローダで accuracy と混同行列を計算する。

    returns: dict(accuracy, n, correct, confusion[n_classes][n_classes],
                  per_class_total, per_class_correct)
      confusion[t][p] = 真値 t を p と予測した件数。
    """
    model.eval()
    cm = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    correct = 0
    total = 0
    for x, y in loader:
        logits = model(x)
        pred = logits.argmax(dim=1)
        for t, p in zip(y.tolist(), pred.tolist()):
            cm[t][p] += 1
            total += 1
            if t == p:
                correct += 1
    per_total = [sum(cm[t]) for t in range(n_classes)]
    per_correct = [cm[t][t] for t in range(n_classes)]
    acc = (correct / total) if total else 0.0
    return dict(accuracy=acc, n=total, correct=correct, confusion=cm,
                per_class_total=per_total, per_class_correct=per_correct)


def _banner_lines(meta: dict) -> list[str]:
    line = "=" * 72
    out = [line, "  cnntrain 評価レポート", line,
           "  " + "!" * 68]
    for s in classes.SYNTHETIC_ONLY_LINES:
        out.append("  !! " + s)
    out.append("  " + "!" * 68)
    if meta:
        out.append(f"  run      : {meta.get('run_name', '-')}")
        out.append(f"  rep_ver  : {meta.get('rep_version', '-')}   "
                   f"gen_seed: {meta.get('gen_seed', '-')}   "
                   f"train_seed: {meta.get('train_seed', '-')}")
        out.append(f"  epochs   : {meta.get('epochs', '-')}   "
                   f"train/val: {meta.get('n_train', '-')}/{meta.get('n_val', '-')}")
    out.append(line)
    return out


def format_report(result: dict, class_names: list[str], meta: dict | None = None) -> str:
    """評価結果を人間可読テキストにする（冒頭にバナー）。"""
    meta = meta or {}
    lines = _banner_lines(meta)
    acc = result["accuracy"]
    lines.append("")
    lines.append(f"[全体] val accuracy = {acc*100:5.1f}%  "
                 f"({result['correct']}/{result['n']})")

    # クラス別 recall
    lines.append("")
    lines.append("[クラス別 recall]")
    for i, c in enumerate(class_names):
        tot = result["per_class_total"][i]
        cor = result["per_class_correct"][i]
        r = (cor / tot * 100) if tot else 0.0
        lines.append(f"  {c:18s}  {r:5.1f}%  ({cor}/{tot})   {classes.look_of(c)}")

    # 混同行列（行=真値, 列=予測）
    lines.append("")
    lines.append("[混同行列]  行 = 真値(true) / 列 = 予測(pred)")
    cw = max(6, *(len(c) for c in class_names))
    cw = min(cw, 16)
    short = [c[:cw] for c in class_names]
    header = " " * (cw + 3) + "".join(f"{s:>{cw+2}}" for s in short)
    lines.append(header)
    cm = result["confusion"]
    for i, c in enumerate(class_names):
        cells = "".join(f"{cm[i][j]:>{cw+2}}" for j in range(len(class_names)))
        lines.append(f"  {c[:cw]:<{cw}} |{cells}")

    lines.append("")
    lines.append("[まとめ]")
    lines.append("  * 火入れの成否は『動くこと』。精度は参考値（合成のみ・ギャップ未測定）。")
    # 不健全チェック: 全予測が同一クラスに潰れていないか。
    pred_cols = [sum(cm[i][j] for i in range(len(class_names)))
                 for j in range(len(class_names))]
    nonzero = [j for j, v in enumerate(pred_cols) if v > 0]
    if len(nonzero) <= 1 and result["n"] > 0:
        only = class_names[nonzero[0]] if nonzero else "(なし)"
        lines.append(f"  * !! 不健全: 全サンプルが単一クラス '{only}' に予測されている。")
        lines.append("       原因の見立て: epoch/データ不足・学習率・クラス潰れ等を疑う。")
    return "\n".join(lines)


def write_report(out_dir: str, result: dict, class_names: list[str],
                 meta: dict | None = None) -> tuple[str, str]:
    """report.txt と report.json を out_dir に書く。returns (txt_path, json_path)。"""
    import os
    os.makedirs(out_dir, exist_ok=True)
    txt = format_report(result, class_names, meta)
    txt_path = os.path.join(out_dir, "report.txt")
    # 凍結契約に倣い meta はロケール既定で書く流儀だが、レポートは UTF-8 固定で
    # 可読性を優先（往復契約の対象外の人間向け成果物のため）。
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt + "\n")

    payload = dict(
        synthetic_only=True,
        synthetic_only_note=classes.SYNTHETIC_ONLY_TAG,
        synthetic_only_lines=classes.SYNTHETIC_ONLY_LINES,
        classes=class_names,
        meta=meta or {},
        accuracy=result["accuracy"],
        n=result["n"], correct=result["correct"],
        confusion=result["confusion"],
        per_class_total=result["per_class_total"],
        per_class_correct=result["per_class_correct"],
    )
    json_path = os.path.join(out_dir, "report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return txt_path, json_path


def _force_utf8():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass


def main(argv=None) -> int:
    """既存チェックポイント + データで評価レポートを再生成する CLI。"""
    _force_utf8()
    import argparse
    from torch.utils.data import DataLoader
    from cnntrain import data, infer

    p = argparse.ArgumentParser(
        prog="cnntrain.evaluate",
        description="チェックポイントを val で評価し混同行列レポートを出す（合成限定）")
    p.add_argument("--data", required=True, help="SigMF データディレクトリ")
    p.add_argument("--ckpt", required=True, help="チェックポイント(.pt)")
    p.add_argument("--val-ratio", type=float, default=0.2, dest="val_ratio")
    p.add_argument("--seed", type=int, default=0, help="分割シード（学習と揃える）")
    p.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    p.add_argument("--out", default=None, help="レポート出力先（既定: 表示のみ）")
    args = p.parse_args(argv)

    ckpt = infer.load_checkpoint(args.ckpt)
    class_names = ckpt.classes
    _, val_recs, _ = data.load_split(args.data, val_ratio=args.val_ratio,
                                     seed=args.seed)
    val_ds = data.SpecDataset(val_recs, class_names)
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0)
    result = evaluate_model(ckpt.model, loader, len(class_names))
    meta = dict(ckpt.meta)
    meta["n_val"] = len(val_ds)
    print(format_report(result, class_names, meta))
    if args.out:
        write_report(args.out, result, class_names, meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
