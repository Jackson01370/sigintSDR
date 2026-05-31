"""低信頼アノテーションの人手レビュー導線。

弱教師ラベル（`method=='rule'` かつ `confidence<0.5`）を列挙し、対話で正しい
label に直して `.sigmf-meta` に書き戻す。書き戻し時に `sigscan:method` を
`'human'` に更新し、`sigscan:confidence` を 1.0（人手＝確定）に上げる。元の
ルールラベルは provenance として comment に残す。

生IQ（`.sigmf-data`）には一切触れない。meta JSON のみを最小限書き換える。
依存は numpy 不要・標準ライブラリと classify/config のみ。

CLI:
    python3 review.py captures/                 # 対話レビュー
    python3 review.py captures/ --list          # 対象を列挙のみ（書き換えなし）
    python3 -m dataset review captures/         # 同じ導線（dataset サブコマンド）
"""
from __future__ import annotations
import glob
import json
import os
import sys

from config import BAND_PLAN
from classify import SIGNAL_DB, UNKNOWN, NOISE

REVIEW_METHOD = "human"


def _force_utf8():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass


def candidate_labels() -> list[str]:
    """再ラベルの選択肢（信号DB由来のラベル + バンド名 + 統一ラベル）。"""
    labels: list[str] = []
    for entry in SIGNAL_DB:           # (band_sub, bw_cond, label, conf, note)
        labels.append(entry[2])
    for b in BAND_PLAN:
        labels.append(b.name)
    labels += [UNKNOWN, NOISE]
    # 重複を順序保持で除去
    seen: set[str] = set()
    uniq: list[str] = []
    for x in labels:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def find_low_confidence(dirpath: str, conf_max: float = 0.5) -> list[dict]:
    """method=='rule' かつ confidence<conf_max のアノテーションを列挙。

    returns: [{meta_path, ann_index, meta, ann, center, bw, label, confidence,
               snr_db, hw, datetime}, ...]
    """
    items: list[dict] = []
    for meta_path in sorted(glob.glob(os.path.join(dirpath, "*.sigmf-meta"))):
        try:
            # sigmf_io と同じ（ロケール既定）エンコーディングで開き往復互換を保つ。
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception as e:   # noqa: BLE001
            print(f"[review] skip {meta_path}: {e}", file=sys.stderr)
            continue
        g = meta.get("global", {})
        caps = meta.get("captures") or [{}]
        cap0 = caps[0]
        for i, ann in enumerate(meta.get("annotations") or []):
            method = ann.get("sigscan:method")
            conf = ann.get("sigscan:confidence")
            if method != "rule" or conf is None:
                continue
            if float(conf) >= conf_max:
                continue
            lo = ann.get("core:freq_lower_edge")
            hi = ann.get("core:freq_upper_edge")
            bw = float(hi - lo) if (lo is not None and hi is not None) else 0.0
            items.append(dict(
                meta_path=meta_path, ann_index=i, meta=meta, ann=ann,
                center=float(cap0.get("core:frequency", 0.0)), bw=bw,
                label=ann.get("core:label"), confidence=float(conf),
                snr_db=ann.get("sigscan:snr_db"), hw=g.get("core:hw", ""),
                datetime=cap0.get("core:datetime"),
            ))
    return items


def apply_label(meta_path: str, ann_index: int, new_label: str,
                method: str = REVIEW_METHOD, confidence: float = 1.0) -> dict:
    """meta JSON の指定アノテーションを再ラベルし書き戻す（生IQには触れない）。

    core:label を new_label に、sigscan:method を method('human') に、
    sigscan:confidence を confidence(1.0) に更新。元ラベルは comment に残す。
    sigmf_io.write_recording と同じ整形（indent=2, ensure_ascii=False）で保存。
    returns: 更新後の annotation dict。
    """
    # 読み書きとも sigmf_io（ロケール既定エンコーディング）に合わせ、書き戻した
    # meta を sigmf_io.read_recording が再度読めることを保証する。
    with open(meta_path) as f:
        meta = json.load(f)
    anns = meta.get("annotations") or []
    if not (0 <= ann_index < len(anns)):
        raise IndexError(f"annotation index {ann_index} out of range in {meta_path}")
    ann = anns[ann_index]
    old_label = ann.get("core:label")
    old_method = ann.get("sigscan:method")

    ann["core:label"] = str(new_label)
    ann["sigscan:method"] = method
    ann["sigscan:confidence"] = float(confidence)
    note = f"human-relabeled (was '{old_label}' via {old_method})"
    existing = ann.get("core:comment")
    ann["core:comment"] = f"{existing} | {note}" if existing else note

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return ann


def _print_item(item: dict, print_fn) -> None:
    from dataset import band_for_center, hw_group   # 遅延 import（循環回避）
    band = band_for_center(item["center"]) or "(バンドプラン外)"
    snr = item["snr_db"]
    snr_s = f"{snr:.1f}dB" if snr is not None else "?"
    print_fn(
        f"\n  file   : {os.path.basename(item['meta_path'])}\n"
        f"  freq   : {item['center']/1e6:.3f} MHz  (band: {band}, hw: {hw_group(item['hw'])})\n"
        f"  bw/snr : {item['bw']/1e6:.1f} MHz / {snr_s}\n"
        f"  label  : '{item['label']}'  (confidence={item['confidence']:.2f}, method=rule)"
    )


def run_review(dirpath: str, conf_max: float = 0.5,
               input_fn=input, print_fn=print) -> int:
    """対話で低信頼アノテーションを再ラベルする。

    input_fn / print_fn は差し替え可能（テスト用）。input_fn が空文字や 's' を
    返すとスキップ、'q' で終了。数値で候補ラベル選択、それ以外は自由入力ラベル。
    """
    items = find_low_confidence(dirpath, conf_max=conf_max)
    cands = candidate_labels()

    print_fn(f"低信頼レビュー: method='rule' かつ confidence<{conf_max} のアノテーション "
             f"{len(items)} 件（対象ディレクトリ: {dirpath}）")
    if not items:
        return 0

    print_fn("\n候補ラベル:")
    for i, c in enumerate(cands):
        print_fn(f"  [{i:2d}] {c}")
    print_fn("\n操作: 番号=その候補に確定 / 自由入力=任意ラベル / 空・s=スキップ / q=終了\n")

    changed = 0
    for n, item in enumerate(items, 1):
        print_fn(f"--- [{n}/{len(items)}] ---")
        _print_item(item, print_fn)
        try:
            ans = input_fn("  正しい label を入力 > ")
        except EOFError:
            print_fn("\n入力終了。レビューを中断します。")
            break
        ans = (ans or "").strip()

        if ans.lower() == "q":
            print_fn("レビューを終了します。")
            break
        if ans == "" or ans.lower() == "s":
            print_fn("  → スキップ")
            continue

        if ans.isdigit() and 0 <= int(ans) < len(cands):
            new_label = cands[int(ans)]
        else:
            new_label = ans   # 自由入力ラベル

        apply_label(item["meta_path"], item["ann_index"], new_label)
        changed += 1
        print_fn(f"  → '{item['label']}' を '{new_label}' に修正（method=human）")

    print_fn(f"\n完了: {changed} 件を再ラベルしました。")
    return 0


def _cmd_list(dirpath: str, conf_max: float) -> int:
    items = find_low_confidence(dirpath, conf_max=conf_max)
    print(f"低信頼アノテーション {len(items)} 件 (method='rule', confidence<{conf_max}):")
    for item in items:
        print(f"  {item['center']/1e6:9.3f}MHz  conf={item['confidence']:.2f}  "
              f"'{item['label']}'  <{os.path.basename(item['meta_path'])}>")
    return 0


def main(argv=None) -> int:
    _force_utf8()
    import argparse
    p = argparse.ArgumentParser(prog="review",
                                description="低信頼アノテーションの人手レビュー/再ラベル")
    p.add_argument("dir", help="SigMF を格納したディレクトリ")
    p.add_argument("--conf-max", type=float, default=0.5, dest="conf_max",
                   help="この信頼度未満の rule アノテーションを対象（既定0.5）")
    p.add_argument("--list", action="store_true",
                   help="対象を列挙するだけで書き換えない")
    args = p.parse_args(argv)
    if args.list:
        return _cmd_list(args.dir, args.conf_max)
    return run_review(args.dir, conf_max=args.conf_max)


if __name__ == "__main__":
    raise SystemExit(main())
