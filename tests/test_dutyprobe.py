"""
test_dutyprobe.py — 在時率(duty)審判の **正しさをテストでピン留め**する。

seeded 合成IQで既知 duty を検証（閾値を結果に合わせて後付け調整する不正を不能化）:
  (a) 30%オン/70%オフのバースト列 → duty≈0.3（±0.05）
  (b) 連続トーン           → duty≈1.0
  (c) ノイズのみ           → duty≈0
加えて referee 閾値・分解能ゲート(inconclusive)・正直バナーの機構をロックする。

凍結契約（spec.py / sigmf_io.py）は触らない（import して呼ぶだけ）。既存テストは無改変（追加のみ）。
実 captures は使わない（合成IQと tmp_path の SigMF のみ）。
"""
import numpy as np

from cnntrain import dutyprobe

RATE = 20_000_000.0          # 実群と同じ 20MHz
CENTER = 2_400_000_000.0
FOFF = 2_000_000.0           # 帯域内に置くトーンのベースバンドオフセット(+2MHz)
F_LO = CENTER + 1_500_000.0  # 検出帯域 [+1.5MHz, +2.5MHz]（トーンを含む・約26bin）
F_HI = CENTER + 2_500_000.0
N = 204_800                  # 約10.24ms（≒実群の13msに近い短スナップショット）
SEED = 20260708


def _noise(n, rng, sigma=1.0):
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(
        np.complex64) * sigma


def _tone(n, amp=8.0):
    t = np.arange(n)
    return (amp * np.exp(1j * 2 * np.pi * FOFF * t / RATE)).astype(np.complex64)


def _iq_continuous(rng):
    return _tone(N) + _noise(N, rng)


def _iq_burst(rng, on_frac_segments=(2, 5, 8), n_seg=10):
    """時間を n_seg 等分し、指定セグメントだけトーンON（残りノイズ）。duty≈len(on)/n_seg。"""
    iq = _noise(N, rng)
    tone = _tone(N)
    seg = N // n_seg
    for s in on_frac_segments:
        iq[s * seg:(s + 1) * seg] += tone[s * seg:(s + 1) * seg]
    return iq


def _iq_noise(rng):
    return _noise(N, rng)


# ---------------------------------------------------------------------------
# (1) 既知 duty の検証（合成IQを直接 measure_duty へ）
# ---------------------------------------------------------------------------
def test_duty_burst_30pct():
    rng = np.random.default_rng(SEED)
    m = dutyprobe.measure_duty(_iq_burst(rng), RATE, CENTER, F_LO, F_HI)
    assert m["n_band_bins"] > 0
    assert abs(m["duty"] - 0.30) <= 0.05, m["duty"]
    assert dutyprobe.referee_from_duty(m["duty"]) == "burst"


def test_duty_continuous():
    rng = np.random.default_rng(SEED + 1)
    m = dutyprobe.measure_duty(_iq_continuous(rng), RATE, CENTER, F_LO, F_HI)
    assert m["duty"] >= 0.95, m["duty"]
    assert dutyprobe.referee_from_duty(m["duty"]) == "continuous"


def test_duty_noise_only():
    rng = np.random.default_rng(SEED + 2)
    m = dutyprobe.measure_duty(_iq_noise(rng), RATE, CENTER, F_LO, F_HI)
    assert m["duty"] <= 0.05, m["duty"]        # 固定6.0dBで純ノイズを確実に棄却
    assert dutyprobe.referee_from_duty(m["duty"]) == "burst"   # duty<0.70


def test_burst_different_fraction_tracks_duty():
    """60%オンにすると duty も約0.6へ（占有計数が実際に時間占有を追う証拠）。"""
    rng = np.random.default_rng(SEED + 3)
    iq = _iq_burst(rng, on_frac_segments=(0, 1, 2, 3, 4, 5))   # 6/10 = 0.6
    m = dutyprobe.measure_duty(iq, RATE, CENTER, F_LO, F_HI)
    assert abs(m["duty"] - 0.60) <= 0.06, m["duty"]


# ---------------------------------------------------------------------------
# (2) referee 閾値（pre-registered）— 純関数
# ---------------------------------------------------------------------------
def test_referee_thresholds():
    f = dutyprobe.referee_from_duty
    assert f(0.0) == "burst"
    assert f(0.30) == "burst"
    assert f(0.699) == "burst"
    assert f(0.70) == "ambiguous"      # 境界は ambiguous（採点除外）
    assert f(0.80) == "ambiguous"
    assert f(0.90) == "ambiguous"
    assert f(0.901) == "continuous"
    assert f(1.0) == "continuous"


# ---------------------------------------------------------------------------
# (3) 分解能ゲート inconclusive — end-to-end（tmp の SigMF を測る）
# ---------------------------------------------------------------------------
def _write_sigmf(path_base, iq, rate):
    import sigmf_io
    sigmf_io.write_recording(
        str(path_base), iq, center_hz=CENTER, sample_rate=rate,
        annotations=[{"freq_lower_edge": F_LO, "freq_upper_edge": F_HI,
                      "label": "TEST"}])


def test_measure_record_continuous_inconclusive_short(tmp_path):
    """13ms級の短スナップショット: duty は測れるが inconclusive=True（adv隙間を分解不能）。"""
    rng = np.random.default_rng(SEED + 10)
    base = tmp_path / "cont_short"
    _write_sigmf(base, _iq_continuous(rng), RATE)
    r = dutyprobe.measure_record(str(base))
    assert r.referee_label == "continuous" and r.duty >= 0.95
    assert r.snapshot_ms < dutyprobe.RES_MIN_SNAPSHOT_MS
    assert r.inconclusive is True


def test_measure_record_burst_and_conclusive_when_long(tmp_path):
    """スナップショットを 300ms 以上にすると inconclusive=False（分解能ゲートが開く）。"""
    # rate を下げて同じ N で snapshot_ms を伸ばす（N/rate*1000）。
    long_rate = 400_000.0                        # N/long_rate = 0.512s = 512ms > 300ms
    rng = np.random.default_rng(SEED + 11)
    # この rate 用に帯域・トーンを rate に合わせて作り直す（トーンは Nyquist 内）。
    t = np.arange(N)
    tone = (8.0 * np.exp(1j * 2 * np.pi * 50_000.0 * t / long_rate)).astype(np.complex64)
    iq = (rng.standard_normal(N) + 1j * rng.standard_normal(N)).astype(np.complex64)
    iq += tone
    import sigmf_io
    base = tmp_path / "cont_long"
    sigmf_io.write_recording(
        str(base), iq, center_hz=CENTER, sample_rate=long_rate,
        annotations=[{"freq_lower_edge": CENTER + 25_000.0,
                      "freq_upper_edge": CENTER + 75_000.0, "label": "TEST"}])
    r = dutyprobe.measure_record(str(base))
    assert r.snapshot_ms >= dutyprobe.RES_MIN_SNAPSHOT_MS
    assert r.hop_ms <= dutyprobe.RES_MAX_HOP_MS
    assert r.inconclusive is False               # 分解能十分 → 結論可能


# ---------------------------------------------------------------------------
# (4) 正直バナー・CSV 出力の機構
# ---------------------------------------------------------------------------
def test_banner_present_in_outputs(tmp_path):
    rng = np.random.default_rng(SEED + 20)
    base = tmp_path / "rec0"
    _write_sigmf(base, _iq_noise(rng), RATE)
    recs = [dutyprobe.measure_record(str(base))]

    console = dutyprobe.format_console(str(tmp_path), recs)
    for b in dutyprobe.BANNER:
        assert b in console
    assert "duty is time-occupancy, NOT modulation/BLE identification" in console

    out_csv = tmp_path / "duty.csv"
    dutyprobe.write_csv(str(out_csv), recs)
    text = out_csv.read_text(encoding="utf-8")
    for b in dutyprobe.BANNER:
        assert ("# " + b) in text                 # ヘッダにバナー（'#' コメント）
    # '#' を飛ばせばデータ行が読める。
    data_lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    assert data_lines[0].startswith("record,")
    assert "rec0" in data_lines[1]
