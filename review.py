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
    python3 review.py captures/ --verdict C      # CNN 監査で (C)=C-conflict の記録を対象
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

# CNN 監査 (C) = 「ルールと CNN が食い違い、確定を人手に委ねる」verdict の内部値。
# global の sigscan:cnn_verdict に格納される（cnntrain M3 が書き込む）。
C_CONFLICT = "C-conflict"


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


def is_c_conflict(meta: dict) -> bool:
    """global の sigscan:cnn_verdict が 'C-conflict' なら True（純関数・読み取りのみ）。

    キーが無い古い記録・verdict が別値（'A-consistent' 等）は False。
    例外にはしない（安全に除外する）。
    """
    g = meta.get("global") or {}
    return g.get("sigscan:cnn_verdict") == C_CONFLICT


def _candidate_hint(comment) -> str | None:
    """core:comment から監査前の保持候補（"用途=..." 以降）を取り出す。

    構造化キーは無いので comment 文字列の該当部分を返す。marker が無ければ None。
    """
    if not comment:
        return None
    marker = "用途="
    idx = comment.find(marker)
    if idx < 0:
        return None
    return comment[idx:].strip()


def find_c_conflict(dirpath: str) -> list[dict]:
    """global の sigscan:cnn_verdict=='C-conflict' の記録のアノテーションを列挙。

    読み取りのみ（captures/ は書き換えない）。sigmf_io と同じロケール既定
    エンコーディングで meta を開く（UTF-8 決め打ちしない）。global にキーが
    無い古い記録は安全に除外する。

    returns: find_low_confidence と互換の item に CNN 来歴（cnn_class/cnn_conf/
             cnn_verdict）・comment・method（='cnn'）を足した dict のリスト。
    """
    items: list[dict] = []
    for meta_path in sorted(glob.glob(os.path.join(dirpath, "*.sigmf-meta"))):
        try:
            # find_low_confidence と同じ（ロケール既定）エンコーディングで開く。
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception as e:   # noqa: BLE001
            print(f"[review] skip {meta_path}: {e}", file=sys.stderr)
            continue
        if not is_c_conflict(meta):
            continue
        g = meta.get("global", {})
        caps = meta.get("captures") or [{}]
        cap0 = caps[0]
        for i, ann in enumerate(meta.get("annotations") or []):
            lo = ann.get("core:freq_lower_edge")
            hi = ann.get("core:freq_upper_edge")
            bw = float(hi - lo) if (lo is not None and hi is not None) else 0.0
            conf = ann.get("sigscan:confidence")
            items.append(dict(
                meta_path=meta_path, ann_index=i, meta=meta, ann=ann,
                center=float(cap0.get("core:frequency", 0.0)), bw=bw,
                label=ann.get("core:label"),
                confidence=float(conf) if conf is not None else 0.0,
                snr_db=ann.get("sigscan:snr_db"), hw=g.get("core:hw", ""),
                datetime=cap0.get("core:datetime"),
                method=ann.get("sigscan:method"),
                cnn_class=g.get("sigscan:cnn_class"),
                cnn_conf=g.get("sigscan:cnn_conf"),
                cnn_verdict=g.get("sigscan:cnn_verdict"),
                comment=ann.get("core:comment"),
            ))
    return items


def _png_path_for(meta_path: str) -> str | None:
    """captures/_images/<base>.png が存在すればそのパスを返す（無ければ None）。

    内蔵表示はしない（パス表示のみ）。base は meta ファイル名から拡張子を除いたもの。
    """
    base = os.path.basename(meta_path)
    if base.endswith(".sigmf-meta"):
        base = base[: -len(".sigmf-meta")]
    png = os.path.join(os.path.dirname(meta_path), "_images", base + ".png")
    return png if os.path.exists(png) else None


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


def _extra_lines(item: dict) -> str:
    """CNN 来歴・保持候補・PNG パスの追加行を組む。

    後方互換のため、CNN 監査を通っていない（cnn_verdict が無い）既定 rule 経路の
    item では常に空文字を返す（＝従来表示のまま）。取れない値は省略する。
    """
    if not item.get("cnn_verdict"):
        return ""
    lines: list[str] = []
    cls = item.get("cnn_class")
    conf = item.get("cnn_conf")
    conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else str(conf)
    lines.append(f"  cnn    : cnn={cls}@{conf_s} verdict={item['cnn_verdict']}")
    cand = _candidate_hint(item.get("comment"))
    if cand:
        lines.append(f"  候補   : {cand}")
    png = _png_path_for(item["meta_path"])
    if png:
        lines.append(f"  png    : {png}")
    return "\n" + "\n".join(lines)


def _print_item(item: dict, print_fn) -> None:
    from dataset import band_for_center, hw_group   # 遅延 import（循環回避）
    band = band_for_center(item["center"]) or "(バンドプラン外)"
    snr = item["snr_db"]
    snr_s = f"{snr:.1f}dB" if snr is not None else "?"
    method = item.get("method") or "rule"
    print_fn(
        f"\n  file   : {os.path.basename(item['meta_path'])}\n"
        f"  freq   : {item['center']/1e6:.3f} MHz  (band: {band}, hw: {hw_group(item['hw'])})\n"
        f"  bw/snr : {item['bw']/1e6:.1f} MHz / {snr_s}\n"
        f"  label  : '{item['label']}'  (confidence={item['confidence']:.2f}, method={method})"
        + _extra_lines(item)
    )


def run_review(dirpath: str, conf_max: float = 0.5,
               input_fn=input, print_fn=print, verdict: str | None = None) -> int:
    """対話で低信頼アノテーションを再ラベルする。

    input_fn / print_fn は差し替え可能（テスト用）。input_fn が空文字や 's' を
    返すとスキップ、'q' で終了。数値で候補ラベル選択、それ以外は自由入力ラベル。

    verdict=='C' 指定時は対象を rule 低信頼ではなく CNN 監査 (C)=C-conflict の
    記録に切り替える（選別のみ変更。確定＝apply_label のロジックは不変）。
    """
    if verdict == "C":
        items = find_c_conflict(dirpath)
        header = (f"CNN監査(C)レビュー: sigscan:cnn_verdict=='{C_CONFLICT}' の記録 "
                  f"{len(items)} 件（対象ディレクトリ: {dirpath}）")
    else:
        items = find_low_confidence(dirpath, conf_max=conf_max)
        header = (f"低信頼レビュー: method='rule' かつ confidence<{conf_max} の"
                  f"アノテーション {len(items)} 件（対象ディレクトリ: {dirpath}）")
    cands = candidate_labels()

    print_fn(header)
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


def _cmd_list(dirpath: str, conf_max: float, verdict: str | None = None) -> int:
    if verdict == "C":
        items = find_c_conflict(dirpath)
        print(f"CNN監査(C)記録 {len(items)} 件 (sigscan:cnn_verdict=='{C_CONFLICT}'):")
    else:
        items = find_low_confidence(dirpath, conf_max=conf_max)
        print(f"低信頼アノテーション {len(items)} 件 (method='rule', confidence<{conf_max}):")
    for item in items:
        base = os.path.basename(item['meta_path'])
        head = (f"  {item['center']/1e6:9.3f}MHz  conf={item['confidence']:.2f}  "
                f"'{item['label']}'  <{base}>")
        # CNN 来歴・保持候補・PNG パスは C 経路（cnn_verdict あり）でのみ付加。
        # 既定 rule 経路は cnn_verdict を持たないので従来行のまま（後方互換）。
        if item.get("cnn_verdict"):
            conf = item.get("cnn_conf")
            conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else str(conf)
            head += f"  cnn={item.get('cnn_class')}@{conf_s} verdict={item['cnn_verdict']}"
            cand = _candidate_hint(item.get("comment"))
            if cand:
                head += f"  {cand}"
            png = _png_path_for(item['meta_path'])
            if png:
                head += f"  [{png}]"
        print(head)
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
    p.add_argument("--verdict", choices=["C"], default=None,
                   help="CNN監査で (C)=C-conflict になった記録を対象にする"
                        "（rule低信頼の代わり。未指定なら従来どおり）")
    args = p.parse_args(argv)
    if args.list:
        return _cmd_list(args.dir, args.conf_max, verdict=args.verdict)
    return run_review(args.dir, conf_max=args.conf_max, verdict=args.verdict)


if __name__ == "__main__":
    raise SystemExit(main())
