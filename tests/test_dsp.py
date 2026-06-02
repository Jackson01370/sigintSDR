"""
test_dsp.py — 測定継ぎ目(dsp.py)の契約をロックする。

- detect_segments: 帯域制限したパワースペクトルに対し 1 本検出（中心/帯域幅が妥当）
- measure_signal: 合成トーン / 帯域制限ノイズIQ で bw・SNR が妥当な範囲

注: detect_segments は (freqs_hz, power_db) すなわち PSD を入力にとる契約
    （サーベイの sweep_power 出力）。measure_signal は生IQを入力にとる。
"""
import numpy as np
import pytest

import dsp


RATE = 20e6
N = 1 << 16


def _band_limited_noise(half_bw_hz=2e6, seed=0):
    """中心0Hz・±half_bw の帯域制限複素ノイズ（帯域幅 ≒ 2*half_bw）。"""
    rng = np.random.default_rng(seed)
    freq = np.fft.fftfreq(N, d=1.0 / RATE)
    spec = rng.normal(size=N) + 1j * rng.normal(size=N)
    spec[np.abs(freq) > half_bw_hz] = 0.0
    bl = np.fft.ifft(spec)
    bl = bl / np.std(bl)
    bl = bl + 0.01 * (rng.normal(size=N) + 1j * rng.normal(size=N))
    return bl.astype(np.complex64)


def _tone(foff_hz=3e6, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(N) / RATE
    tone = np.exp(2j * np.pi * foff_hz * t)
    return (tone + 0.02 * (rng.normal(size=N) + 1j * rng.normal(size=N))).astype(np.complex64)


def _survey_spectrum(seed=42):
    """サーベイ風パワースペクトル: 平坦床 -100dB に 2.44-2.46GHz(約20MHz)を +30dB。"""
    rng = np.random.default_rng(seed)
    freqs = np.linspace(2.4e9, 2.5e9, 2000)
    power = -100.0 + rng.normal(0, 1.0, freqs.size)
    band = (freqs > 2.44e9) & (freqs < 2.46e9)
    power[band] += 30.0
    return freqs, power


def test_detect_segments_single_band():
    """帯域制限スペクトル(約20MHz) → ちょうど1本検出、中心と帯域幅が妥当。"""
    freqs, power = _survey_spectrum()
    segs = dsp.detect_segments(freqs, power, threshold_db=8.0, min_bw_hz=1e6)

    assert len(segs) == 1
    s = segs[0]
    assert 2.445e9 < s["f_center"] < 2.455e9
    assert 1.5e7 < s["bw_hz"] < 3.0e7
    assert s["f_lo"] < s["f_hi"]
    assert s["snr_db"] > 8.0


def test_detect_segments_min_bw_filters_narrow():
    """min_bw_hz を帯域幅より大きくすると検出ゼロ（しきい値の挙動）。"""
    freqs, power = _survey_spectrum()
    segs = dsp.detect_segments(freqs, power, threshold_db=8.0, min_bw_hz=50e6)
    assert len(segs) == 0


def test_detect_segments_empty_when_flat():
    """信号が無い平坦スペクトルでは検出ゼロ。"""
    rng = np.random.default_rng(7)
    freqs = np.linspace(2.4e9, 2.5e9, 2000)
    power = -100.0 + rng.normal(0, 1.0, freqs.size)
    segs = dsp.detect_segments(freqs, power, threshold_db=8.0, min_bw_hz=1e6)
    assert len(segs) == 0


def test_measure_signal_band_limited():
    """帯域制限ノイズIQ → bw が帯域幅相当、SNR が十分高い。"""
    m = dsp.measure_signal(_band_limited_noise(half_bw_hz=2e6), RATE, 2.437e9)
    assert 2e6 < m["bw_hz"] < 7e6          # 帯域幅 ≒ 4MHz
    assert m["snr_db"] > 15.0
    assert m["peak_db"] > m["noise_floor_db"]
    assert abs(m["center_hz"] - 2.437e9) < 1e6


def test_measure_signal_tone():
    """単一トーンIQ → bw は狭く、SNR が非常に高い。"""
    m = dsp.measure_signal(_tone(foff_hz=3e6), RATE, 2.45e9)
    assert 0.0 <= m["bw_hz"] < 1e6          # トーンは数bin幅
    assert m["snr_db"] > 20.0
    # 中心はトーンのオフセット分だけずれる（center + 3MHz 付近）
    assert abs(m["center_hz"] - 2.453e9) < 1e6


# ---------------------------------------------------------------------------
# DCスパイク（DCオフセット由来の中央スパイク）測定
#   ゼロIF受信機が取得帯域の中央(DC=オフセット0Hz)に出す、本物ではない細い線を
#   1取得IQから測る。中央集中(dc_excess 大)は DC スパイクのみで立ち、中央外の
#   信号・広帯域信号・無信号では立たないことを検証する。
# ---------------------------------------------------------------------------
def _dc_spike(level=0.5, seed=2):
    """中央(DC, オフセット0Hz)固定の定数オフセット + 微小ノイズ = DCスパイク相当。"""
    rng = np.random.default_rng(seed)
    noise = 0.02 * (rng.normal(size=N) + 1j * rng.normal(size=N))
    return (level + noise).astype(np.complex64)


def test_dc_spike_metrics_flags_center_constant():
    """中央(DC)固定の定数オフセットは dc_excess が大（中央だけ突出した細い線）。"""
    m = dsp.dc_spike_metrics(_dc_spike(), RATE)
    assert m["dc_excess_db"] > 30.0
    assert m["dc_peak_db"] > m["side_med_db"]


def test_dc_spike_metrics_ignores_offset_tone():
    """中央からオフセットした細い信号は中央が上がらず dc_excess 小（脇のmedianも動かない）。"""
    assert dsp.dc_spike_metrics(_tone(foff_hz=3e6), RATE)["dc_excess_db"] < 5.0
    # 脇リング内(0.3MHz)に来ても、median は1ビンでは動かないので excess は小さい。
    assert dsp.dc_spike_metrics(_tone(foff_hz=0.3e6), RATE)["dc_excess_db"] < 5.0


def test_dc_spike_metrics_ignores_wideband():
    """帯域を広く埋める信号(WiFi相当)は両脇も同様に上がり dc_excess 小。"""
    assert dsp.dc_spike_metrics(
        _band_limited_noise(half_bw_hz=8e6), RATE)["dc_excess_db"] < 5.0


def test_dc_spike_metrics_ignores_noise():
    """無信号(ノイズのみ)は中央も脇も同程度で dc_excess 小。"""
    rng = np.random.default_rng(5)
    noise = (0.02 * (rng.normal(size=N)
                     + 1j * rng.normal(size=N))).astype(np.complex64)
    assert dsp.dc_spike_metrics(noise, RATE)["dc_excess_db"] < 5.0
