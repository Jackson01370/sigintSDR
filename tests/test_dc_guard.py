"""
test_dc_guard.py — DC残留ガード（--dc-guard-hz / measure_signal_dc_guarded）を固定する。

指示書「1GHz以下の受信対応」Part A のテスト仕様（1〜5）を実装する。ガードは opt-in で、
既定 0 では従来の measure_signal と 1 ビットも変わらないこと（最重要）、有効時はチューナ
中心±HZ を主役候補から外して次に強い本物を拾うこと、判定がチューナ相対であること、除外帯
しか信号が無い場合に例外を出さず検出なしになること、offset併用時に後の実チューナ中心を
基準にすること、を固定する。seam の dsp.measure_signal はシグネチャ不変（test_seams.py）。
"""
import numpy as np
import pytest

import dsp
from config import Config, SDRConfig, ScanConfig
from sdr import SimBackend
from scheduler import HybridScheduler


RATE = 20e6
N = 1 << 16
CENTER = 80.0e6            # 1GHz以下（FM帯）の代表チューナ中心


def _noise(amp=0.01, seed=0):
    rng = np.random.default_rng(seed)
    return (amp * (rng.normal(size=N) + 1j * rng.normal(size=N))).astype(np.complex64)


def _tone(foff_hz, amp, seed=1):
    t = np.arange(N) / RATE
    return (amp * np.exp(2j * np.pi * foff_hz * t)).astype(np.complex64)


def _dc(level):
    """取得帯域中央(オフセット0Hz)固定の定数 = DC残留相当（チューナ中心に張り付く）。"""
    return np.full(N, np.complex64(level), dtype=np.complex64)


def _dc_plus_offset_signal(dc_level=1.0, sig_off_hz=5e6, sig_amp=0.3, seed=2):
    """強い DC残留（中央）+ 弱い本物の信号（+sig_off_hz）+ 微小ノイズ。

    DC の方が強いので、ガード無しでは argmax が中央（DC）を選び検出中心がチューナ中心に
    張り付く。ガード有効で中央を外すと、次に強い本物（+sig_off_hz）が主役になる。
    """
    return (_dc(dc_level) + _tone(sig_off_hz, sig_amp, seed=seed)
            + _noise(0.01, seed=seed + 1)).astype(np.complex64)


# ---------------------------------------------------------------------------
# 1) 既定0で挙動不変（最重要）: guarded(…,0) == measure_signal
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("iq", [
    _dc_plus_offset_signal(),
    _tone(3e6, 1.0),
    _noise(0.02, seed=9),
])
def test_guard_zero_identical_to_measure_signal(iq):
    """dc_guard_hz=0 は seam の measure_signal と完全に同一（1ビットも変わらない）。"""
    base = dsp.measure_signal(iq, RATE, CENTER)
    guarded0 = dsp.measure_signal_dc_guarded(iq, RATE, CENTER, 0.0)
    assert guarded0 == base                      # dict 全キーがビット等価


def test_guard_negative_also_identity():
    """負値も無効扱い（<=0）で measure_signal と同一（防御的に確認）。"""
    iq = _dc_plus_offset_signal()
    assert dsp.measure_signal_dc_guarded(iq, RATE, CENTER, -1.0) == \
        dsp.measure_signal(iq, RATE, CENTER)


# ---------------------------------------------------------------------------
# 2) ガード有効で中心が除外され、次に強い信号が選ばれる
# ---------------------------------------------------------------------------
def test_guard_excludes_center_and_picks_next():
    """DC残留(中央) + 本物(+5MHz)。ガード無し→中央、ガード有り→本物 が主役。"""
    iq = _dc_plus_offset_signal(dc_level=1.0, sig_off_hz=5e6, sig_amp=0.3)

    no_guard = dsp.measure_signal(iq, RATE, CENTER)
    # ガード無し: DC が最強 → 検出中心はチューナ中心付近（DC を掴んでいる）
    assert abs(no_guard["center_hz"] - CENTER) < 0.5e6

    guarded = dsp.measure_signal_dc_guarded(iq, RATE, CENTER, 0.5e6)
    # ガード有り: 中央除外 → 次に強い本物(+5MHz)が選ばれる
    assert abs(guarded["center_hz"] - (CENTER + 5e6)) < 1.0e6


# ---------------------------------------------------------------------------
# 3) チューナ相対であること（中心を動かすと除外帯も一緒に動く・絶対固定ではない）
# ---------------------------------------------------------------------------
def test_guard_is_tuner_relative_not_absolute():
    """同一IQ(DC中央+本物+5MHz)を別々のチューナ中心で測ると、検出中心 - チューナ中心が
    どちらも +5MHz 付近になる（除外帯がチューナ中心に追随する＝相対）。"""
    iq = _dc_plus_offset_signal(dc_level=1.0, sig_off_hz=5e6, sig_amp=0.3)

    m1 = dsp.measure_signal_dc_guarded(iq, RATE, 80e6, 0.5e6)
    m2 = dsp.measure_signal_dc_guarded(iq, RATE, 200e6, 0.5e6)

    rel1 = m1["center_hz"] - 80e6
    rel2 = m2["center_hz"] - 200e6
    assert abs(rel1 - 5e6) < 1.0e6
    assert abs(rel2 - 5e6) < 1.0e6
    # 相対オフセットが中心に依らず一致（絶対周波数固定なら片方ずれる）
    assert abs(rel1 - rel2) < 1.0e6


# ---------------------------------------------------------------------------
# 4) 除外帯しか信号が無い場合、例外を出さず「検出なし」になる
# ---------------------------------------------------------------------------
def test_guard_only_center_signal_is_no_detection():
    """中央の DC だけ（本物なし）。ガード無し→高SNRで検出、ガード有り→検出なし(低SNR)。
    例外を出さないこと。"""
    iq = (_dc(0.5) + _noise(0.02, seed=11)).astype(np.complex64)

    no_guard = dsp.measure_signal(iq, RATE, CENTER)
    assert no_guard["snr_db"] >= 10.0            # DC を検出（高SNR）

    guarded = dsp.measure_signal_dc_guarded(iq, RATE, CENTER, 1.0e6)   # 例外を出さない
    assert guarded["snr_db"] < 10.0              # 中央除外で検出なし相当（品質下限未満）
    assert np.isfinite(guarded["center_hz"])     # 破綻しない


def test_guard_covering_whole_window_does_not_crash():
    """退化設定（ガードが取得帯域全体を覆う）でも例外を出さずフォールバックする。"""
    iq = _dc_plus_offset_signal()
    m = dsp.measure_signal_dc_guarded(iq, RATE, CENTER, RATE)   # ±20MHz > 窓 ±10MHz
    assert np.isfinite(m["snr_db"]) and np.isfinite(m["center_hz"])


# ---------------------------------------------------------------------------
# 5) offset併用時、オフセット適用後の実チューナ中心を基準に除外される
#    （scheduler.dwell が guarded に渡す center は target + offset であること）
# ---------------------------------------------------------------------------
def _guard_cfg(dc_guard_hz=0.5e6, dwell_offset_hz=0.0):
    sdr = SDRConfig(dwell_samples=1 << 14, dc_guard_hz=dc_guard_hz,
                    dwell_offset_hz=dwell_offset_hz)
    scan = ScanConfig(start_hz=76e6, stop_hz=95e6, max_dwell_per_cycle=1)
    return Config(sdr=sdr, scan=scan)


def test_scheduler_routes_to_guarded_with_post_offset_center(monkeypatch):
    """dc_guard_hz>0 のとき scheduler.dwell は measure_signal_dc_guarded を呼び、
    その center はオフセット適用後（target + off）であること（offset併用の基準）。"""
    calls = []
    real = dsp.measure_signal_dc_guarded

    def spy(iq, rate, center_hz, dc_guard_hz):
        calls.append((center_hz, dc_guard_hz))
        return real(iq, rate, center_hz, dc_guard_hz)

    monkeypatch.setattr(dsp, "measure_signal_dc_guarded", spy)

    cfg = _guard_cfg(dc_guard_hz=0.5e6, dwell_offset_hz=4e6)
    be = SimBackend(cfg.sdr, seed=0)
    sched = HybridScheduler(be, cfg, store=None)
    # 狭帯域ターゲット(bw<=8MHz)なのでオフセットが効く → center = 80 + 4 = 84MHz
    sched.dwell({"center": 80e6, "bw": 1e6, "src": "band:test"})

    assert calls, "ガード有効時に measure_signal_dc_guarded が呼ばれていない"
    center_used, guard_used = calls[0]
    assert guard_used == 0.5e6
    assert abs(center_used - 84e6) < 1.0    # オフセット適用後の実チューナ中心が基準


def test_scheduler_default_does_not_call_guarded(monkeypatch):
    """dc_guard_hz=0（既定）では guarded は一切呼ばれない（従来の measure_signal 経路）。"""
    guarded_calls = []
    plain_calls = []
    real_g = dsp.measure_signal_dc_guarded
    real_m = dsp.measure_signal
    monkeypatch.setattr(dsp, "measure_signal_dc_guarded",
                        lambda *a, **k: (guarded_calls.append(1), real_g(*a, **k))[1])
    monkeypatch.setattr(dsp, "measure_signal",
                        lambda *a, **k: (plain_calls.append(1), real_m(*a, **k))[1])

    cfg = _guard_cfg(dc_guard_hz=0.0)
    be = SimBackend(cfg.sdr, seed=0)
    sched = HybridScheduler(be, cfg, store=None)
    sched.dwell({"center": 80e6, "bw": 1e6, "src": "band:test"})

    assert not guarded_calls          # ガード関数は呼ばれない
    assert plain_calls                # 従来 seam を呼ぶ


def test_scheduler_prints_guard_notice_only_when_enabled(capsys):
    """ガード有効時のみ1行告知。既定0では何も出さない（現行出力と完全一致）。"""
    cfg_on = _guard_cfg(dc_guard_hz=0.5e6)
    be = SimBackend(cfg_on.sdr, seed=0)
    HybridScheduler(be, cfg_on, store=None).run(once=True, verbose=True)
    assert "DC残留ガード" in capsys.readouterr().out

    cfg_off = _guard_cfg(dc_guard_hz=0.0)
    be2 = SimBackend(cfg_off.sdr, seed=0)
    HybridScheduler(be2, cfg_off, store=None).run(once=True, verbose=True)
    assert "DC残留ガード" not in capsys.readouterr().out
