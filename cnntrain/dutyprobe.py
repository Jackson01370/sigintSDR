"""cnntrain: 在時率(duty)プローブ = burst/continuous を分ける **決定的審判**。

目的（サンドボックス実験）:
    2401MHz 群の捕獲は persist=1.00（0.5秒窓粒度で毎窓検出）で出ているが、
    persist は 0.5s 窓の粗さのため「高速advビーコン（離散バースト）」と
    「連続エミッタ混獲（連続帯）」を区別できない。その区別に **persist より
    細かい時間分解能** = STFT 行(時間フレーム)単位の在時率(duty)を与える。

測定の定義:
    - IQ を凍結 spec.stft_db（nfft=512 / hop=256）で [freq,time] dB に変換（read-only 流用）。
    - annotation の検出帯域 [f_lo,f_hi] に入る bin の **総和パワー**を各時間フレームで取り、
      capture のノイズ床（帯域外 bin の中央値）× 帯域bin数を基準に **帯域内SNR** を出す。
    - フレームの帯域内SNR ≥ THRESHOLD_DB を「占有」とし、duty = 占有フレーム数 / 全フレーム数。

なぜ「総和(平均パワー)」で「最大(ピーク)」ではないか（pre-registered な設計判断・結果を見る前に固定）:
    帯域が ~30 bin ある場合、純ノイズでも 30 bin の **最大** は中央値床より約 +7〜8dB
    上振れする（順序統計量）。固定閾値 6.0dB では純ノイズが「占有」に化け duty が水増しされる。
    総和(=平均パワー) を per-bin 中央値床と比べると、純ノイズは約 +1.6dB に収まり
    6.0dB を確実に下回る。よって帯域幅に頑健で、ノイズを 6.0dB で確実に棄却できる総和を採る。
    代償（正直に明記）: 広い検出帯域の中の弱い狭帯域信号は平均で薄まり過小計数されうる。
    本群の信号は SNR 19〜25dB と強いため実害は小さいが、これは保守側の既知バイアス。

正直な限定（最重要）:
    * duty は **時間占有の測定**であって「BLE か否か（変調/用途）の判定」ではない。
      burst/continuous という決定軸に客観的な正解軸を与えるだけ。
    * 保存IQスナップショットが短すぎて adv 間隔(20-100ms)の隙間を分解できない場合は
      inconclusive=True（結論不能）。この群は約13msスナップショットのため全件 inconclusive になる。
    * SANDBOX: ground truth ではない・CNN 学習入力にしない・review.py の human確定に流用しない。

凍結契約: spec.py / sigmf_io.py は **import して呼ぶだけ**（編集しない）。captures/ は読み取り専用。

CLI:
    python -m cnntrain.dutyprobe --data captures/ --out bench/cc_vs_human_2401/duty_captures.csv
    python -m cnntrain.dutyprobe --data captures/_review_pending/ --out bench/cc_vs_human_2401/duty_review_pending.csv
"""
from __future__ import annotations

import csv
import glob
import os
import sys
from dataclasses import dataclass

import numpy as np

import sigmf_io
import spec   # STFT を import して read-only で流用（spec.py は編集しない）

# ===========================================================================
# pre-registered 定数（結果に合わせて動かさない）
# ===========================================================================
THRESHOLD_DB = 6.0          # フレーム帯域内SNRがこれ以上なら「占有」

# burst / continuous / ambiguous の切り分け（duty 由来）
DUTY_BURST_MAX = 0.70       # duty < 0.70 → burst
DUTY_CONT_MIN = 0.90        # duty > 0.90 → continuous  （0.70-0.90 は ambiguous＝採点除外）

# 分解能ゲート（IQスナップショットが adv 間隔を分解できるか）
RES_MIN_SNAPSHOT_MS = 300.0  # snapshot_ms < 300 → inconclusive
RES_MAX_HOP_MS = 20.0        # hop_ms > 20 → inconclusive

# ノイズ床推定のフォールバック（帯域外 bin が少なすぎる時）
MIN_OOB_BINS = 16            # 帯域外 bin がこれ未満なら全体低percentile へ退避
NOISE_FALLBACK_PCT = 20.0    # 退避時に使う全体パワーの低percentile

# 正直バナー（出力ヘッダに必ず載せる3種）
BANNER = (
    "duty is time-occupancy, NOT modulation/BLE identification",
    "inconclusive if IQ snapshot too short to resolve adv gaps",
    "SANDBOX — not ground truth, not CNN training input",
)


# ===========================================================================
# 測定本体（純関数・決定的）
# ===========================================================================
@dataclass
class DutyRecord:
    record: str
    label: str
    center_mhz: float
    f_lo_mhz: float
    f_hi_mhz: float
    n_band_bins: int
    duty: float
    referee_label: str      # duty 閾値由来（burst / continuous / ambiguous）
    snapshot_ms: float
    hop_ms: float
    n_rows: int
    inconclusive: bool
    note: str = ""


def referee_from_duty(duty: float) -> str:
    """pre-registered 閾値で duty を burst/continuous/ambiguous に写す。"""
    if duty < DUTY_BURST_MAX:
        return "burst"
    if duty > DUTY_CONT_MIN:
        return "continuous"
    return "ambiguous"


def measure_duty(iq, rate: float, center_hz: float, f_lo: float, f_hi: float,
                 nfft: int = spec.SPEC_NFFT, hop: int = spec.SPEC_HOP,
                 threshold_db: float = THRESHOLD_DB) -> dict:
    """[f_lo,f_hi] 帯域の時間占有率 duty を決定的に測る。

    returns dict(duty, n_rows, n_band_bins, snapshot_ms, hop_ms, occupied_frac...)。
    STFT は凍結 spec.stft_db を流用。パワーは dB→linear に戻して総和/中央値で扱う。
    """
    iq = np.asarray(iq, dtype=np.complex64)
    n_samples = int(iq.size)
    snapshot_ms = n_samples / rate * 1000.0
    hop_ms = hop / rate * 1000.0

    S_db = spec.stft_db(iq, rate=rate, nfft=nfft, hop=hop)   # [freq, time], 絶対dB
    P = np.power(10.0, S_db / 10.0)                          # linear power ∝ |X|^2
    n_freq, n_rows = P.shape

    # bin の絶対周波数（fftshift 済みの並びに一致させる）
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / rate)) + center_hz
    band = (freqs >= f_lo) & (freqs <= f_hi)
    n_band = int(band.sum())
    oob = ~band

    # ノイズ床（capture 単位・per-bin）: 帯域外 bin の中央値。少なすぎれば全体低percentile。
    if int(oob.sum()) >= MIN_OOB_BINS:
        nf_perbin = float(np.median(P[oob, :]))
    else:
        nf_perbin = float(np.percentile(P, NOISE_FALLBACK_PCT))
    nf_perbin = max(nf_perbin, 1e-30)

    if n_band <= 0:
        # 帯域が bin を1つも含まない異常。占有0・note を残す。
        return dict(duty=0.0, n_rows=int(n_rows), n_band_bins=0,
                    snapshot_ms=snapshot_ms, hop_ms=hop_ms,
                    note="band contains no STFT bin")

    # 各時間フレームの帯域内 **総和パワー** を per-bin 床×帯域bin数 と比べて SNR(dB)。
    inband_sum = P[band, :].sum(axis=0)          # [n_rows]（instruction の「総和」）
    noise_ref = nf_perbin * n_band               # 帯域が全部ノイズ床なら期待される総和
    snr_db = 10.0 * np.log10((inband_sum + 1e-30) / noise_ref)
    occupied = snr_db >= threshold_db
    duty = float(np.mean(occupied)) if n_rows > 0 else 0.0

    return dict(duty=duty, n_rows=int(n_rows), n_band_bins=n_band,
                snapshot_ms=snapshot_ms, hop_ms=hop_ms, note="")


def _first_band(meta: dict) -> tuple[float, float, str] | None:
    """annotation から (f_lo, f_hi, label) を取る（最初の帯域付き注釈）。無ければ None。"""
    for a in meta.get("annotations", []):
        lo = a.get("core:freq_lower_edge")
        hi = a.get("core:freq_upper_edge")
        if lo is not None and hi is not None:
            return float(lo), float(hi), str(a.get("core:label", ""))
    return None


def measure_record(path_base: str) -> DutyRecord:
    """SigMF レコード1件を読み（読み取りのみ）duty を測る。"""
    iq, meta = sigmf_io.read_recording(path_base)
    glob_meta = meta.get("global", {})
    rate = float(glob_meta.get("core:sample_rate", spec.CAPTURE_RATE_HZ))
    caps = meta.get("captures", [{}])
    center = float(caps[0].get("core:frequency", 0.0)) if caps else 0.0

    band = _first_band(meta)
    record = os.path.basename(path_base)
    if band is None:
        snapshot_ms = iq.size / rate * 1000.0
        hop_ms = spec.SPEC_HOP / rate * 1000.0
        return DutyRecord(
            record=record, label="", center_mhz=center / 1e6,
            f_lo_mhz=0.0, f_hi_mhz=0.0, n_band_bins=0, duty=0.0,
            referee_label="ambiguous",
            snapshot_ms=snapshot_ms, hop_ms=hop_ms, n_rows=0,
            inconclusive=True, note="no annotation band")

    f_lo, f_hi, label = band
    m = measure_duty(iq, rate, center, f_lo, f_hi)
    inconclusive = (m["snapshot_ms"] < RES_MIN_SNAPSHOT_MS) or \
                   (m["hop_ms"] > RES_MAX_HOP_MS)
    return DutyRecord(
        record=record, label=label, center_mhz=center / 1e6,
        f_lo_mhz=f_lo / 1e6, f_hi_mhz=f_hi / 1e6,
        n_band_bins=m["n_band_bins"], duty=round(m["duty"], 4),
        referee_label=referee_from_duty(m["duty"]),
        snapshot_ms=round(m["snapshot_ms"], 3), hop_ms=round(m["hop_ms"], 5),
        n_rows=m["n_rows"], inconclusive=inconclusive, note=m["note"])


def run_dutyprobe(data_dir: str, pattern: str = "*") -> list[DutyRecord]:
    """data_dir 直下（非再帰）の *.sigmf-meta を走査し duty を測る（読み取りのみ）。"""
    metas = sorted(glob.glob(os.path.join(data_dir, pattern + ".sigmf-meta")))
    out: list[DutyRecord] = []
    for mp in metas:
        base = mp[: -len(".sigmf-meta")]
        out.append(measure_record(base))
    return out


# ===========================================================================
# 出力
# ===========================================================================
CSV_FIELDS = ["record", "label", "center_mhz", "f_lo_mhz", "f_hi_mhz",
              "n_band_bins", "duty", "referee_label", "snapshot_ms",
              "hop_ms", "n_rows", "inconclusive", "note"]


def write_csv(out_path: str, records: list[DutyRecord]) -> None:
    """CSV を書く。先頭に正直バナーを '#' コメント行として載せる（ヘッダにバナー）。"""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        for b in BANNER:
            f.write(f"# {b}\n")
        f.write(f"# THRESHOLD_DB={THRESHOLD_DB} "
                f"burst<{DUTY_BURST_MAX} continuous>{DUTY_CONT_MIN} "
                f"(pre-registered)\n")
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in records:
            w.writerow({
                "record": r.record, "label": r.label,
                "center_mhz": f"{r.center_mhz:.6f}",
                "f_lo_mhz": f"{r.f_lo_mhz:.6f}", "f_hi_mhz": f"{r.f_hi_mhz:.6f}",
                "n_band_bins": r.n_band_bins, "duty": f"{r.duty:.4f}",
                "referee_label": r.referee_label,
                "snapshot_ms": f"{r.snapshot_ms:.3f}", "hop_ms": f"{r.hop_ms:.5f}",
                "n_rows": r.n_rows,
                "inconclusive": "True" if r.inconclusive else "False",
                "note": r.note,
            })


def format_console(data_dir: str, records: list[DutyRecord]) -> str:
    line = "=" * 78
    out = [line, "  cnntrain 在時率(duty)プローブ — DECISIVE REFEREE (SANDBOX)", line,
           "  " + "!" * 74]
    for b in BANNER:
        out.append(f"  !! {b}")
    out.append("  " + "!" * 74)
    out.append(f"  data      : {data_dir}   records={len(records)}")
    out.append(f"  pre-reg   : THRESHOLD_DB={THRESHOLD_DB}  "
               f"burst<{DUTY_BURST_MAX}  continuous>{DUTY_CONT_MIN}  "
               f"inconclusive if snapshot<{RES_MIN_SNAPSHOT_MS}ms or hop>{RES_MAX_HOP_MS}ms")
    out.append(line)
    out.append(f"  {'record':<28}{'duty':>7} {'referee':>11} "
               f"{'snap_ms':>8} {'nrows':>6}  inconc")
    for r in records:
        out.append(f"  {r.record[:28]:<28}{r.duty:>7.3f} {r.referee_label:>11} "
                   f"{r.snapshot_ms:>8.2f} {r.n_rows:>6}  "
                   f"{'INCONCLUSIVE' if r.inconclusive else '-'}")
    n_inc = sum(1 for r in records if r.inconclusive)
    out.append(line)
    out.append(f"  inconclusive: {n_inc}/{len(records)}  "
               f"(duty は測定であってラベルではない／13msスナップショットは adv 隙間を分解できない)")
    return "\n".join(out)


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
        prog="cnntrain.dutyprobe",
        description="在時率(duty)で burst/continuous を分ける決定的審判（サンドボックス・読み取りのみ）")
    p.add_argument("--data", required=True, help="SigMF データディレクトリ（読み取りのみ）")
    p.add_argument("--out", required=True, help="出力 CSV パス")
    p.add_argument("--pattern", default="*",
                   help="ベース名 glob 前方一致（既定 '*'）。例 '2401MHz_*'")
    args = p.parse_args(argv)

    records = run_dutyprobe(args.data, pattern=args.pattern)
    print(format_console(args.data, records))
    write_csv(args.out, records)
    print("")
    print(f"  CSV: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
