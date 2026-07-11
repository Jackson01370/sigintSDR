"""cnntrain: レビュー提案ツール（サンドボックス）= 人間の ground truth 確定を速く・正確に。

**提案を作るだけ**。ラベルの最終確定(method=human の付与)は人間が別途 review.py で行う。
このツールは SigMF(captures/ 配下)に一切書き込まない（読み取りのみ）。出力は bench/ のみ。

各 capture レコードについて:
  * 客観指標を集める: det_freq / bw / snr / spur_suspect / rule ラベル・confidence、
    在時率 duty（**cnntrain.dutyprobe を import 流用**＝再実装しない）。
  * スプリアス誤確定ガード（原則4）: det_freq が 2400.0±0.1MHz、または spur_suspect=True の行は
    spurious_warn=True とし、cc_class が何であれ recommend=skip を**コードで強制**する。
  * CC(あなた)が PNG を見て付けた視覚分類 cc_class と根拠 cc_rationale を（--verdicts CSV で）併合。
  * recommend は **cc_class 主導・duty 非依存**で決める（decide_recommend・優先順）:
      spurious_warn==True            -> skip（最優先・誤確定ガード・不変）
      cc_class 空/None               -> needs-review（人間に必ず提示。黙って skip しない）
      cc_class=='ble-adv'（warn無し） -> confirm-ble
      それ以外(wifi/spurious/hopping/unclear) -> skip
    duty・duty_inconclusive は分岐に**使わない**（表示専用の補助列）。13ms=inconclusive でも機能する。

出力:
  * bench/<out>/suggestions.csv  … 全列（客観 + cc_class + cc_rationale + recommend）
  * bench/<out>/confirm_sheet.md … 人間の確定チェックリスト（4節 + 正直バナー）

正直な限定:
  * 提案 != 確定。最終承認は人間が PNG を見て review.py で行う。CC は method=human を付与しない。
  * duty は時間占有の測定でありラベルではない。duty も cc_class も CNN 学習入力にしない。

凍結契約: spec / sigmf_io / dutyprobe は **import して読むだけ**（編集しない）。

CLI:
    python -m cnntrain.review_suggest --data captures/ --pattern "2402MHz_1783530*" \
        --out bench/review_suggest_ch37/ [--verdicts bench/review_suggest_ch37/cc_verdicts.csv]
"""
from __future__ import annotations

import csv
import glob
import os
import sys
from dataclasses import dataclass, field

import sigmf_io
from cnntrain import dutyprobe   # duty は import 流用（再実装しない）

# ---------------------------------------------------------------------------
# pre-registered 定数（スプリアス誤確定ガード）
# ---------------------------------------------------------------------------
SPUR_DET_MHZ = 2400.0        # 40MHz クロック高調波（既知内部スプリアス）の周波数
SPUR_DET_TOL_MHZ = 0.1       # det_freq がこの範囲内 → spurious_warn

CC_CLASSES = ("ble-adv", "spurious", "wifi", "hopping", "unclear")

BANNER = (
    "これは CC の提案であって確定ではない。最終承認は人間が PNG を見て review.py で行う。",
    "CC は SigMF を書き換えていない（captures/ は読み取りのみ）。method=human は付与していない。",
    "duty は時間占有の測定でありラベルではない。duty も cc_class も CNN 学習入力にしない。",
    "スプリアス誤確定ガード: det≈2400.0MHz または spur_suspect=True は cc_class に関わらず skip 強制。",
    "recommend は duty ではなく cc_class(視覚)で決定。duty が inconclusive(13ms収録)でも機能する。duty は補助列。",
)


# ---------------------------------------------------------------------------
# データ
# ---------------------------------------------------------------------------
@dataclass
class SuggestRecord:
    record: str
    png: str                 # 対応 PNG パス（無ければ "(なし)"）
    det_freq_mhz: float
    bw_mhz: float
    snr_db: float | None
    rule_label: str
    rule_confidence: float | None
    duty: float
    duty_inconclusive: bool
    spur_suspect: bool
    spurious_warn: bool
    cc_class: str = ""
    cc_rationale: str = ""
    recommend: str = ""
    note: str = ""


def spurious_warn_for(det_freq_mhz: float, spur_suspect: bool) -> bool:
    """原則4: det が 2400.0±0.1MHz、または spur_suspect=True なら警告。"""
    near_spur = abs(float(det_freq_mhz) - SPUR_DET_MHZ) < SPUR_DET_TOL_MHZ
    return bool(near_spur or bool(spur_suspect))


def decide_recommend(spurious_warn: bool, cc_class: str | None) -> str:
    """recommend を **cc_class 主導・duty 非依存**で決める（duty を引数に取らない）。

      1. spurious_warn==True    -> skip（最優先ガード・cc_class 不問・不変）
      2. cc_class 空/None       -> needs-review（人間に必ず提示。黙って skip しない）
      3. cc_class=='ble-adv'    -> confirm-ble
      4. それ以外               -> skip（wifi/spurious/hopping/unclear）

    duty・duty_inconclusive は分岐条件に**一切使わない**（構造的に duty 非依存）。
    13ms 収録で duty=inconclusive でも cc_class='ble-adv' なら confirm-ble を出せる
    （前回の「inconclusive→全skip 空振り」の穴をコードで塞ぐ）。
    """
    if spurious_warn:
        return "skip"                      # cc_class が何であれ skip 強制（最優先）
    if not cc_class or cc_class.strip() == "":
        return "needs-review"              # 未記入は黙って skip せず人間へ回す
    if cc_class == "ble-adv":
        return "confirm-ble"
    return "skip"


# ---------------------------------------------------------------------------
# 客観指標の収集（決定的・読み取りのみ）
# ---------------------------------------------------------------------------
def _first_band(meta: dict):
    for a in meta.get("annotations", []):
        lo = a.get("core:freq_lower_edge")
        hi = a.get("core:freq_upper_edge")
        if lo is not None and hi is not None:
            return a, float(lo), float(hi)
    return None, None, None


def _png_for(data_dir: str, record: str) -> str:
    p = os.path.join(data_dir, "_images", record + ".png")
    return p if os.path.exists(p) else "(なし)"


def collect_one(path_base: str, data_dir: str) -> SuggestRecord | None:
    """SigMF レコード1件の客観指標を集める（読み取りのみ）。band 無しは None。"""
    record = os.path.basename(path_base)
    iq, meta = sigmf_io.read_recording(path_base)
    g = meta.get("global", {})
    caps = meta.get("captures") or [{}]
    center = float(caps[0].get("core:frequency", 0.0))
    rate = float(g.get("core:sample_rate", 20_000_000.0))

    ann, f_lo, f_hi = _first_band(meta)
    if ann is None:
        return None
    det_freq_mhz = (f_lo + f_hi) / 2.0 / 1e6
    bw_mhz = (f_hi - f_lo) / 1e6
    snr_db = ann.get("sigscan:snr_db")
    spur_suspect = bool(ann.get("sigscan:spur_suspect", False))
    rule_label = str(ann.get("core:label", ""))
    rule_conf = ann.get("sigscan:confidence")

    # duty は dutyprobe を流用（再実装しない）。inconclusive も dutyprobe の定数で判定。
    m = dutyprobe.measure_duty(iq, rate, center, f_lo, f_hi)
    duty_inconclusive = (m["snapshot_ms"] < dutyprobe.RES_MIN_SNAPSHOT_MS) or \
                        (m["hop_ms"] > dutyprobe.RES_MAX_HOP_MS)

    warn = spurious_warn_for(det_freq_mhz, spur_suspect)
    return SuggestRecord(
        record=record, png=_png_for(data_dir, record),
        det_freq_mhz=round(det_freq_mhz, 4), bw_mhz=round(bw_mhz, 4),
        snr_db=(round(float(snr_db), 1) if snr_db is not None else None),
        rule_label=rule_label,
        rule_confidence=(round(float(rule_conf), 3) if rule_conf is not None else None),
        duty=round(float(m["duty"]), 4), duty_inconclusive=bool(duty_inconclusive),
        spur_suspect=spur_suspect, spurious_warn=warn,
        # cc_class 未記入の既定は needs-review（spurious なら guard で skip）。
        recommend=decide_recommend(warn, ""),
    )


def collect_objective(data_dir: str, pattern: str = "*") -> list[SuggestRecord]:
    """data_dir 直下（非再帰）の pattern 一致レコードの客観指標を集める（読み取りのみ）。"""
    metas = sorted(glob.glob(os.path.join(data_dir, pattern + ".sigmf-meta")))
    out: list[SuggestRecord] = []
    for mp in metas:
        base = mp[: -len(".sigmf-meta")]
        rec = collect_one(base, data_dir)
        if rec is not None:
            out.append(rec)
    return out


def apply_verdicts(records: list[SuggestRecord],
                   verdicts: dict[str, tuple[str, str]]) -> list[SuggestRecord]:
    """CC の視覚所見(cc_class, cc_rationale)を併合し recommend を再計算（ガード強制）。"""
    for r in records:
        v = verdicts.get(r.record)
        if v is not None:
            cc_class, cc_rationale = v
            r.cc_class = cc_class
            r.cc_rationale = cc_rationale
        r.recommend = decide_recommend(r.spurious_warn, r.cc_class)  # 常にガードを通す
    return records


def read_verdicts_csv(path: str) -> dict[str, tuple[str, str]]:
    """cc_verdicts.csv(record, cc_class, cc_rationale) を読む。無ければ空。"""
    if not path or not os.path.exists(path):
        return {}
    out: dict[str, tuple[str, str]] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rec = (row.get("record") or "").strip()
            if rec:
                out[rec] = ((row.get("cc_class") or "").strip(),
                            (row.get("cc_rationale") or "").strip())
    return out


# ---------------------------------------------------------------------------
# 出力
# ---------------------------------------------------------------------------
CSV_FIELDS = ["record", "png", "det_freq_mhz", "bw_mhz", "snr_db",
              "rule_label", "rule_confidence", "duty", "duty_inconclusive",
              "spur_suspect", "spurious_warn", "cc_class", "cc_rationale",
              "recommend", "note"]


def write_suggestions_csv(out_dir: str, records: list[SuggestRecord]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "suggestions.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        for b in BANNER:
            f.write(f"# {b}\n")
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in records:
            w.writerow({
                "record": r.record, "png": r.png,
                "det_freq_mhz": f"{r.det_freq_mhz:.4f}", "bw_mhz": f"{r.bw_mhz:.4f}",
                "snr_db": ("" if r.snr_db is None else f"{r.snr_db:.1f}"),
                "rule_label": r.rule_label,
                "rule_confidence": ("" if r.rule_confidence is None
                                    else f"{r.rule_confidence:.3f}"),
                "duty": f"{r.duty:.4f}",
                "duty_inconclusive": "True" if r.duty_inconclusive else "False",
                "spur_suspect": "True" if r.spur_suspect else "False",
                "spurious_warn": "True" if r.spurious_warn else "False",
                "cc_class": r.cc_class, "cc_rationale": r.cc_rationale,
                "recommend": r.recommend, "note": r.note,
            })
    return path


def _row(r: SuggestRecord) -> str:
    snr = "" if r.snr_db is None else f"{r.snr_db:.1f}dB"
    inc = " (inconclusive)" if r.duty_inconclusive else ""
    return (f"| {r.record} | {r.det_freq_mhz:.2f} | {r.bw_mhz:.2f} | "
            f"{r.duty:.3f}{inc} | {r.cc_class or '-'} | {r.cc_rationale or '-'} |")


def format_confirm_sheet(records: list[SuggestRecord], data_dir: str,
                         pattern: str) -> str:
    confirm = [r for r in records if r.recommend == "confirm-ble"]
    skip = [r for r in records if r.recommend == "skip"]
    needs = [r for r in records if r.recommend == "needs-review"]
    warns = [r for r in records if r.spurious_warn]

    md: list[str] = []
    md.append(f"# 確定シート: BLE ch37 レビュー提案（サンドボックス） — {pattern}")
    md.append("")
    md.append("> **正直バナー（提案 ≠ 確定）**  ")
    for b in BANNER:
        md.append(f"> - {b}  ")
    md.append("")
    md.append(f"対象: `{data_dir}` の `{pattern}` 一致 {len(records)} 件。"
              "duty は `cnntrain.dutyprobe` を流用（400ms 収録なら inconclusive=False）。"
              "recommend は cc_class(視覚)主導で duty 非依存＝13ms=inconclusive でも機能する。")
    md.append("")
    if needs:
        md.append(f"> ⚠ **視覚分類 未記入 {len(needs)} 件 → needs-review**。"
                  "CC が PNG を確認し cc_class を埋め直すこと（黙って skip にしない）。")
        md.append("")
    md.append("**人間の手順**: `review.py captures/ --pattern \"" + pattern + "\"` で対象だけを列に出し、"
              "下表と照合。confirm-ble の行のみ PNG で BLE adv（離散バースト・2400線と分離）を"
              "確認してから確定（yes）。skip は確定しない。**承認判断は人間が握る（提案の鵜呑み禁止）**。")
    md.append("")

    md.append(f"## ✅ Confirm as BLE ch37（recommend=confirm-ble） — {len(confirm)} 件")
    if confirm:
        md.append("| file | det_freq(MHz) | bw(MHz) | duty | cc_class | rationale |")
        md.append("|---|---|---|---|---|---|")
        md.extend(_row(r) for r in confirm)
    else:
        md.append("(なし)")
    md.append("")

    md.append(f"## ⏭ Skip（確定しない） — {len(skip)} 件")
    if skip:
        md.append("| file | det_freq(MHz) | bw(MHz) | duty | cc_class(理由) | rationale |")
        md.append("|---|---|---|---|---|---|")
        md.extend(_row(r) for r in skip)
    else:
        md.append("(なし)")
    md.append("")

    md.append(f"## 🔎 Needs-review（視覚分類 未記入・要目視） — {len(needs)} 件")
    if needs:
        md.append("CC が cc_class を埋め忘れた行。黙って skip せず人間に必ず提示する（前回の空振り穴の再発防止）。")
        md.append("| file | det_freq(MHz) | bw(MHz) | duty | cc_class | rationale |")
        md.append("|---|---|---|---|---|---|")
        md.extend(_row(r) for r in needs)
    else:
        md.append("(なし) — 全件で cc_class が記入済み。")
    md.append("")

    md.append(f"## ⚠ Warnings（spurious_warn=True・誤確定ガード発火） — {len(warns)} 件")
    if warns:
        md.append("| file | det_freq(MHz) | spur_suspect | 理由 |")
        md.append("|---|---|---|---|")
        for r in warns:
            reason = []
            if abs(r.det_freq_mhz - SPUR_DET_MHZ) < SPUR_DET_TOL_MHZ:
                reason.append("det≈2400.0(40MHz高調波)")
            if r.spur_suspect:
                reason.append("spur_suspect=True")
            md.append(f"| {r.record} | {r.det_freq_mhz:.2f} | "
                      f"{'True' if r.spur_suspect else 'False'} | {' / '.join(reason)} |")
    else:
        md.append("(なし) — 本バッチに 2400.0 スプリアス/spur_suspect の行は無い。")
    md.append("")
    md.append("---")
    md.append("_CC は確定していない・SigMF を書き換えていない・method=human を付与していない。_")
    return "\n".join(md) + "\n"


def write_confirm_sheet(out_dir: str, records: list[SuggestRecord],
                        data_dir: str, pattern: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "confirm_sheet.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(format_confirm_sheet(records, data_dir, pattern))
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
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
        prog="cnntrain.review_suggest",
        description="レビュー提案ツール（サンドボックス・提案のみ・SigMF 非改変）")
    p.add_argument("--data", required=True, help="SigMF データディレクトリ（読み取りのみ）")
    p.add_argument("--out", required=True, help="出力ディレクトリ（bench/ 配下）")
    p.add_argument("--pattern", default="*",
                   help="ベース名 glob 前方一致（例 '2402MHz_1783530*'）。既定 '*'")
    p.add_argument("--verdicts", default=None,
                   help="CC 視覚所見 CSV(record,cc_class,cc_rationale)。"
                        "未指定なら <out>/cc_verdicts.csv があれば併合")
    args = p.parse_args(argv)

    records = collect_objective(args.data, pattern=args.pattern)

    verdicts_path = args.verdicts or os.path.join(args.out, "cc_verdicts.csv")
    verdicts = read_verdicts_csv(verdicts_path)
    apply_verdicts(records, verdicts)

    csv_path = write_suggestions_csv(args.out, records)
    sheet_path = write_confirm_sheet(args.out, records, args.data, args.pattern)

    print("=" * 74)
    print("  cnntrain レビュー提案ツール（SANDBOX・提案のみ）")
    print("=" * 74)
    for b in BANNER:
        print(f"  !! {b}")
    print("=" * 74)
    n_conf = sum(1 for r in records if r.recommend == "confirm-ble")
    n_skip = sum(1 for r in records if r.recommend == "skip")
    n_needs = sum(1 for r in records if r.recommend == "needs-review")
    n_warn = sum(1 for r in records if r.spurious_warn)
    print(f"  対象 {len(records)} 件  confirm-ble={n_conf}  skip={n_skip}  "
          f"needs-review={n_needs}  spurious_warn={n_warn}  "
          f"(verdicts={'有' if verdicts else '無'})")
    if n_needs:
        print(f"  ⚠ 視覚分類 未記入 {n_needs} 件 → needs-review。"
              "CC が PNG を見て cc_class を埋めること（黙って skip にしない）。")
    print(f"  CSV  : {csv_path}")
    print(f"  Sheet: {sheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
