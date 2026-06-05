"""cnntrain (M1.5): 実データ推論プローブ（探針）。**推論のみ／学習しない**。

M1 で Sim 学習した CNN を、実機収集データ（captures/）に **推論だけ** 通し、
ドメインギャップを初観測する。同時に 3 つを観測する:
  1. ドメインギャップ : Sim の方式クラスが実信号にどれだけ通用するか。
  2. 13ms くじ引き問題: 保存IQは約13msの切り取り。まばらなバースト(BLE adv等)は
     窓に写っていない記録がありうる → 「ラベル=BLE なのに 予測=noise-only」の件数
     がこの問題の直接の証拠（画像にバーストが無いなら noise-only は画像として正しい）。
  3. ルールラベル監査  : 高確信度で期待と食い違う予測は、ルールラベル誤りの候補。

軸の違い（最重要）:
  * CNN のクラスは **方式軸**（wideband-ofdm / narrowband-burst / cw-tone /
    pulse-radar / noise-only ＝ 見え方）。
  * 実データのラベルは **用途軸**（WiFi / BLE / Zigbee ＝ 用途）。
  * 両者の照合に「期待対応表（用途→期待方式クラス集合）」を使うが、これは [仮説]。
    照合一致率を **accuracy（精度）と呼ばない**（軸が違う照合。下記 3 因で読む）。

不一致の 3 因分解（レポートの読み方）:
  (a) ドメインギャップ（Sim と実の見え方の差）
  (b) 13ms くじ引き（バーストが窓に写っていない → noise-only）
  (c) ルールラベル自体の誤り（高確信度の食い違い）

入力は必ず凍結 spec.render 経由（infer.classify_iq を使う）。spec.py/sigmf_io.py は
変更しない。captures/ は読み取り専用（移動/改名/上書きしない）。

CLI:
    python -m cnntrain.probe --data captures/ --checkpoint runs/m1/checkpoint.pt \
        --out runs/probe_real/ [--top-n 10]
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field

import sigmf_io
import spec
import dataset as ds_mod
from cnntrain import infer


# ===========================================================================
# (1) 期待対応表（用途軸ラベル → 期待される方式軸クラス集合）  ★ [仮説] ★
# ===========================================================================
# これは「地図上の仮説」である。画像確認と将来の人間レビューで更新する前提。
# 一致は accuracy ではない（用途軸 vs 方式軸の照合）。集合なのは、1 つの用途が
# 複数の見え方を取りうるため（帯域幅・バースト性で分かれる）。
@dataclass(frozen=True)
class ExpectedRow:
    keywords: tuple        # ラベル(小文字)に対する部分一致キーワード（いずれか一致で採用）
    expect: frozenset      # 期待される方式クラス（spec/方式軸）の集合
    rationale: str         # この対応の根拠（[仮説]）


# 事前確認3で captures/ に実在する全ラベルをカバーする:
#   "BLE/Bluetooth (adv?)" / "WiFi (2.4GHz, 20/40MHz)" / "Zigbee/独自2.4G"
EXPECTED_REAL: list[ExpectedRow] = [
    ExpectedRow(
        ("ble", "bluetooth"),
        frozenset({"narrowband-burst"}),
        "[仮説] BLE adv は ~2MHz 以下の狭帯域バースト → narrowband-burst を期待。"
        "ただし保存は約13msの切り取りで、まばらなバーストが窓に未着なら画像は"
        "ノイズのみ → noise-only が出うる（=13msくじ引きの証拠としてカウント）。"),
    ExpectedRow(
        ("wifi",),
        frozenset({"wideband-ofdm"}),
        "[仮説] WiFi 2.4GHz 20/40MHz は OFDM の広帯域ブロック → wideband-ofdm を期待。"
        "実画像が細線寄りなら cw-tone が出る可能性もあり（ラベル監査の観点）。"),
    ExpectedRow(
        ("zigbee", "独自"),
        frozenset({"narrowband-burst", "wideband-ofdm"}),
        "[仮説] Zigbee は ~2MHz O-QPSK（狭め）だが『独自2.4G』が混在しうるため、"
        "narrowband-burst と wideband-ofdm の集合で受ける。"),
]

EXPECTED_DISCLAIMER = (
    "期待対応表は [仮説]（用途軸→方式軸の地図）。画像確認・人間レビューで更新する。"
    "照合の一致は accuracy ではない。")


def match_expected(label: str | None,
                   table: list[ExpectedRow] | None = None) -> tuple[frozenset | None, str]:
    """ラベル → (期待方式クラス集合, 根拠)。未対応は (None, '')（=unmapped）。"""
    table = EXPECTED_REAL if table is None else table
    lab = (label or "").lower()
    for row in table:
        if any(k.lower() in lab for k in row.keywords):
            return row.expect, row.rationale
    return None, ""


# ===========================================================================
# (2) プローブ本体
# ===========================================================================
@dataclass
class ProbeRecord:
    file: str               # ベース名（captures/_images/ の PNG と突き合わせ用）
    label: str              # 実データの用途軸ラベル（ルール由来・弱教師）
    center_mhz: float
    pred: str               # CNN の方式軸予測
    confidence: float
    rule_confidence: float | None
    snr_db: float | None
    expected: list | None   # 期待方式クラス（sorted）。None=unmapped
    matched: bool | None    # pred ∈ expected。None=unmapped
    noise_only: bool        # pred == "noise-only"


@dataclass
class ProbeResult:
    checkpoint: str
    data_dir: str
    classes: list            # CNN の方式クラス（列順）
    n_total: int
    hw_counts: dict
    records: list            # list[ProbeRecord]
    crosstab: dict           # {label: {pred: count}}
    labels: list             # 行ラベル順（ソート）
    per_label: dict          # {label: {...統計...}}
    review_list: list        # 要画像確認（mismatch を確信度降順）
    ckpt_meta: dict = field(default_factory=dict)


def _aggregate(records: list[ProbeRecord], classes: list[str]) -> tuple[dict, list, dict]:
    """クロス表・per-label 統計を作る。returns (crosstab, labels, per_label)。"""
    crosstab: dict[str, dict[str, int]] = {}
    per_label: dict[str, dict] = {}
    labels: list[str] = []
    for r in records:
        if r.label not in crosstab:
            crosstab[r.label] = {}
            labels.append(r.label)
        crosstab[r.label][r.pred] = crosstab[r.label].get(r.pred, 0) + 1
    labels.sort()

    for lab in labels:
        recs = [r for r in records if r.label == lab]
        n = len(recs)
        mapped = [r for r in recs if r.matched is not None]
        matched_n = sum(1 for r in mapped if r.matched)
        noise_n = sum(1 for r in recs if r.noise_only)
        confs = [r.confidence for r in recs]
        unmapped = (len(mapped) == 0)
        exp = next((r.expected for r in recs if r.expected is not None), None)
        per_label[lab] = dict(
            n=n,
            expected=exp,                       # None=unmapped
            unmapped=unmapped,
            expected_match=matched_n,
            expected_match_rate=(matched_n / len(mapped)) if mapped else None,
            noise_only=noise_n,
            noise_only_rate=(noise_n / n) if n else 0.0,
            mean_confidence=(sum(confs) / n) if n else 0.0,
        )
    return crosstab, labels, per_label


def _review_list(records: list[ProbeRecord], top_n: int) -> list[ProbeRecord]:
    """要画像確認: 期待にマップされ かつ 不一致(matched=False) を確信度降順 top_n。"""
    mismatches = [r for r in records if r.matched is False]
    mismatches.sort(key=lambda r: (-r.confidence, r.file))
    return mismatches[:top_n]


def run_probe(data_dir: str, ckpt_path: str, top_n: int = 10,
              table: list[ExpectedRow] | None = None) -> ProbeResult:
    """captures/ 群に **推論のみ** を回し ProbeResult を返す（学習しない）。

    各レコード: read_recording → 凍結 spec.render（infer.classify_iq 内）→ 推論。
    captures/ は読み取りのみ（書き換えない）。
    """
    ck = infer.load_checkpoint(ckpt_path)
    # load_index は top-level の *.sigmf-meta のみ（非再帰）→ _review_pending/ は拾わない。
    ds = ds_mod.load_index(data_dir)
    records: list[ProbeRecord] = []
    hw_counts: dict[str, int] = {}
    for r in ds:
        hw_counts[r.hw_group] = hw_counts.get(r.hw_group, 0) + 1
        iq, meta = sigmf_io.read_recording(r.path)
        rate = float(meta.get("global", {}).get("core:sample_rate",
                                                spec.CAPTURE_RATE_HZ))
        pred, conf = infer.classify_iq(ck, iq, rate)   # spec.render 経由（迂回しない）
        exp, _note = match_expected(r.label, table)
        matched = (pred in exp) if exp is not None else None
        records.append(ProbeRecord(
            file=os.path.basename(r.path),
            label=r.label or "(ラベルなし)",
            center_mhz=r.center / 1e6,
            pred=pred,
            confidence=conf,
            rule_confidence=r.confidence,
            snr_db=r.snr_db,
            expected=sorted(exp) if exp is not None else None,
            matched=matched,
            noise_only=(pred == "noise-only"),
        ))
    crosstab, labels, per_label = _aggregate(records, ck.classes)
    review = _review_list(records, top_n)
    return ProbeResult(
        checkpoint=ckpt_path, data_dir=data_dir, classes=list(ck.classes),
        n_total=len(records), hw_counts=hw_counts, records=records,
        crosstab=crosstab, labels=labels, per_label=per_label,
        review_list=review, ckpt_meta=dict(ck.meta),
    )


# ===========================================================================
# 出力（バナー必須・eval-harness 流儀）
# ===========================================================================
def _banner(res: ProbeResult) -> list[str]:
    line = "=" * 76
    out = [line, "  cnntrain 実データ推論プローブ（REAL-DATA PROBE）", line,
           "  " + "!" * 72,
           "  !! SIM-TRAINED MODEL / REAL-DATA PROBE:",
           "  !!   合成(Sim)のみで学習したモデルで実機データを推論する『探針』。",
           "  !!   実データでの学習・fine-tune は一切していない（推論のみ）。",
           "  !! 期待対応表は [仮説]（用途軸→方式軸の地図）。画像確認で更新する。",
           "  !! 照合の一致は **accuracy（精度）ではない** — 軸が違う照合。",
           "  !!   不一致は 3 因で読む: (a)ドメインギャップ (b)13msくじ引き(noise-only)",
           "  !!   (c)ルールラベル誤り。故障とは限らない＝観測である。",
           "  " + "!" * 72]
    hw = ", ".join(f"{k}={v}" for k, v in sorted(res.hw_counts.items())) or "(なし)"
    out.append(f"  checkpoint : {res.checkpoint}")
    m = res.ckpt_meta
    out.append(f"  model      : SYNTHETIC-ONLY 学習  rep={m.get('rep_version','-')}  "
               f"sim_val_acc={m.get('final_val_acc','-')}  gen_seed={m.get('gen_seed','-')}")
    out.append(f"  data       : {res.data_dir}   records={res.n_total}   hw: {hw}")
    out.append(f"  CNNクラス  : {', '.join(res.classes)}")
    out.append(line)
    return out


def format_report(res: ProbeResult) -> str:
    lines = _banner(res)

    # クロス表（行=実ラベル[用途軸], 列=予測[方式軸]）
    lines.append("")
    lines.append("[クロス表]  行 = 実ラベル(用途軸・ルール弱教師)  ×  列 = CNN予測(方式軸)")
    lines.append("  ※ 軸が違う照合。一致は精度ではない。")
    cls = res.classes
    cw = min(16, max(9, *(len(c) for c in cls)))
    header = " " * 30 + "".join(f"{c[:cw]:>{cw+2}}" for c in cls) + f"{'計':>6}"
    lines.append(header)
    for lab in res.labels:
        row = res.crosstab.get(lab, {})
        cells = "".join(f"{row.get(c, 0):>{cw+2}}" for c in cls)
        lines.append(f"  {lab[:28]:<28}{cells}{sum(row.values()):>6}")
    # 列合計
    tot = {c: sum(res.crosstab[l].get(c, 0) for l in res.labels) for c in cls}
    lines.append(f"  {'(列合計)':<28}"
                 + "".join(f"{tot[c]:>{cw+2}}" for c in cls)
                 + f"{sum(tot.values()):>6}")

    # ラベル別統計
    lines.append("")
    lines.append("[ラベル別]  期待一致率(対応表[仮説]ベース) / noise-only率 / 平均確信度")
    for lab in res.labels:
        s = res.per_label[lab]
        exp = "/".join(s["expected"]) if s["expected"] else "(unmapped)"
        emr = (f"{s['expected_match_rate']*100:5.1f}%"
               if s["expected_match_rate"] is not None else "   -  ")
        lines.append(
            f"  {lab[:28]:<28} n={s['n']:3d}  期待={exp:<28}  "
            f"一致={emr} ({s['expected_match']}/{s['n'] if not s['unmapped'] else 0})  "
            f"noise-only={s['noise_only_rate']*100:5.1f}% ({s['noise_only']}/{s['n']})  "
            f"平均確信度={s['mean_confidence']:.3f}")

    # 期待対応表（根拠つき）を明示
    lines.append("")
    lines.append("[期待対応表 / [仮説] / 根拠]")
    for row in EXPECTED_REAL:
        lines.append(f"  {'|'.join(row.keywords):<16} → {{{', '.join(sorted(row.expect))}}}")
        lines.append(f"      {row.rationale}")

    # 要画像確認リスト
    lines.append("")
    lines.append(f"[要画像確認リスト]  期待と不一致 × 確信度の高い順 上位 {len(res.review_list)} 件")
    lines.append("  ※ captures/_images/ の PNG と突き合わせ、3因(ギャップ/くじ引き/ラベル誤り)を判定。")
    if not res.review_list:
        lines.append("  (該当なし)")
    for r in res.review_list:
        exp = "/".join(r.expected) if r.expected else "-"
        lines.append(
            f"  conf={r.confidence:.3f}  {r.center_mhz:8.1f}MHz  "
            f"ラベル={r.label[:22]:<22} 期待={exp:<28} 予測={r.pred:<16}  {r.file}")

    # 3 因の読み方ガイド（見立ては完了報告で具体化）
    lines.append("")
    lines.append("[3因分解の読み方]")
    lines.append("  (a) ドメインギャップ : 期待方式と違う方式に高確信度で寄る（Sim/実の差）。")
    lines.append("  (b) 13msくじ引き     : ラベル=BLE等 なのに noise-only。窓にバースト未着なら")
    lines.append("                          画像として noise-only は正しい（CNNの故障ではない）。")
    lines.append("  (c) ルールラベル誤り : 高確信度で期待外。中心周波数等が用途と矛盾する候補。")
    return "\n".join(lines)


def write_report(out_dir: str, res: ProbeResult) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    txt = format_report(res)
    txt_path = os.path.join(out_dir, "probe_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt + "\n")

    payload = dict(
        probe="sim-trained model / real-data probe",
        learning="none (inference only)",
        expected_map_is_hypothesis=True,
        match_is_not_accuracy=True,
        disclaimer=EXPECTED_DISCLAIMER,
        checkpoint=res.checkpoint,
        checkpoint_meta=res.ckpt_meta,
        data_dir=res.data_dir,
        n_total=res.n_total,
        hw_counts=res.hw_counts,
        classes=res.classes,
        expected_map=[dict(keywords=list(r.keywords),
                           expect=sorted(r.expect), rationale=r.rationale)
                      for r in EXPECTED_REAL],
        crosstab=res.crosstab,
        labels=res.labels,
        per_label=res.per_label,
        review_list=[dict(file=r.file, label=r.label, center_mhz=r.center_mhz,
                          pred=r.pred, confidence=r.confidence,
                          expected=r.expected, rule_confidence=r.rule_confidence,
                          snr_db=r.snr_db)
                     for r in res.review_list],
        records=[dict(file=r.file, label=r.label, center_mhz=r.center_mhz,
                      pred=r.pred, confidence=r.confidence,
                      expected=r.expected, matched=r.matched,
                      noise_only=r.noise_only, rule_confidence=r.rule_confidence,
                      snr_db=r.snr_db)
                 for r in res.records],
    )
    json_path = os.path.join(out_dir, "probe_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return txt_path, json_path


# ===========================================================================
# CLI
# ===========================================================================
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
        prog="cnntrain.probe",
        description="Sim学習CNNで実データを推論するプローブ（推論のみ・学習しない）")
    p.add_argument("--data", required=True, help="SigMF データディレクトリ（captures/）")
    p.add_argument("--checkpoint", required=True, help="チェックポイント(.pt)")
    p.add_argument("--out", required=True, help="レポート出力先")
    p.add_argument("--top-n", type=int, default=10, dest="top_n",
                   help="要画像確認リストの件数（既定 10）")
    args = p.parse_args(argv)

    res = run_probe(args.data, args.checkpoint, top_n=args.top_n)
    print(format_report(res))
    txt, js = write_report(args.out, res)
    print("")
    print(f"  レポート: {txt}")
    print(f"           {js}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
