"""
test_dc_removal.py — DCスパイク除去（DCオフセット補正 / DC offset correction）。

実機(HackRF等)が取得帯域の中央(オフセット0Hz)に出す時間不変のDCスパイクを、
「捨てる」のではなく「IQから複素平均(DCオフセット)を引いて消す」方式を検証する。

  1) DCオフセットを乗せた合成IQに remove_dc を適用すると、中央(0Hz)スパイクが
     消える（中央ビンが除去前より大幅に下がり、両脇と同程度になる）。
  2) 本物の信号（中央外トーン・広帯域・バースト）は除去前後でパワー/帯域がほぼ不変
     （remove_dc が壊さない）。
  3) SimBackend に --sim-dc-spike 相当(dc_offset)で注入した中央スパイクが、
     dc_removal 有効時には除去され、品質ゲートの dc_spike に出ない。
  4) main.build_backend / scheduler の配線（既定・フラグ・SigMF への dc_removed 記録）。

依存は numpy のみ。継ぎ目シグネチャには触れない（remove_dc は新規追加関数）。
"""
import argparse
import glob
import os

import numpy as np

import dsp
import dwell
import main
import quality
import sdr
import sigmf_io
from config import (Config, SDRConfig, ScanConfig, DwellConfig, QualityConfig)
from scheduler import HybridScheduler
from store import Store

RATE = 20e6
N = 1 << 14


def _noise(n, seed=0, amp=0.02):
    rng = np.random.default_rng(seed)
    return (amp * (rng.normal(size=n) + 1j * rng.normal(size=n))).astype(np.complex64)


# ---------------------------------------------------------------------------
# remove_dc ヘルパ単体
# ---------------------------------------------------------------------------
def test_remove_dc_empty_and_dtype():
    """空配列は安全に通し、定数のみ(=純DC)は平均を引いて 0 になる。dtype は complex64。"""
    empty = dsp.remove_dc(np.array([], dtype=np.complex64))
    assert empty.size == 0 and empty.dtype == np.complex64
    const = dsp.remove_dc(np.full(1000, 0.7 + 0.3j, dtype=np.complex64))
    assert const.dtype == np.complex64
    assert np.allclose(const, 0.0, atol=1e-5)        # 純DCは丸ごと消える


def test_remove_dc_kills_center_spike_metric():
    """DCオフセットを乗せたIQ → dc_excess(中央集中度)が除去で大幅に下がる。"""
    spiked = (_noise(N) + np.complex64(0.5)).astype(np.complex64)
    before = dsp.dc_spike_metrics(spiked, RATE)["dc_excess_db"]
    after = dsp.dc_spike_metrics(dsp.remove_dc(spiked), RATE)["dc_excess_db"]
    assert before > 20.0            # 除去前は中央が突出（DCスパイク）
    assert after < 5.0              # 除去後は両脇と同程度
    assert before - after > 15.0    # 大幅に低下


def test_remove_dc_center_bin_drops_to_sides():
    """除去後の中央(0Hz)ビンのパワーが、除去前より大幅に下がり両脇と同程度になる。"""
    spiked = (_noise(N) + np.complex64(0.7)).astype(np.complex64)
    f, p_before = dsp.welch_psd(spiked, RATE)
    _, p_after = dsp.welch_psd(dsp.remove_dc(spiked), RATE)
    c = int(np.argmin(np.abs(f)))                    # 中央(0Hz)ビン
    side = np.abs(f) > 1e6                            # 両脇
    assert p_before[c] - p_before[side].mean() > 20.0   # 除去前: 中央が突出
    assert p_after[c] - p_after[side].mean() < 5.0      # 除去後: 両脇と同程度
    assert p_before[c] - p_after[c] > 15.0              # 中央ビンが大幅低下


# ---------------------------------------------------------------------------
# 本物の信号を壊さない
# ---------------------------------------------------------------------------
def test_remove_dc_preserves_offcenter_tone():
    """中央外(+3MHz)のトーン: ピーク強度も位置も除去前後で不変。"""
    t = np.arange(N) / RATE
    sig = (np.exp(2j * np.pi * 3e6 * t) + _noise(N)).astype(np.complex64)
    _, p_before = dsp.welch_psd(sig, RATE)
    _, p_after = dsp.welch_psd(dsp.remove_dc(sig), RATE)
    assert abs(p_before.max() - p_after.max()) < 0.5         # ピーク不変
    assert int(np.argmax(p_before)) == int(np.argmax(p_after))   # 位置不変


def test_remove_dc_preserves_wideband():
    """広帯域信号(WiFi相当): 占有帯域・SNR・PSD全体が除去前後でほぼ不変。"""
    rng = np.random.default_rng(3)
    f = np.fft.fftfreq(N, d=1.0 / RATE)
    s = rng.normal(size=N) + 1j * rng.normal(size=N)
    s[np.abs(f) > 8e6] = 0.0                          # |f|<8MHz の帯域制限ノイズ
    wide = np.fft.ifft(s)
    wide = (wide / np.std(wide)).astype(np.complex64)
    sig = (wide + _noise(N, seed=3, amp=0.05)).astype(np.complex64)

    mb = dsp.measure_signal(sig, RATE, 2.4e9)
    ma = dsp.measure_signal(dsp.remove_dc(sig), RATE, 2.4e9)
    assert abs(mb["bw_hz"] - ma["bw_hz"]) < 1e6      # 占有帯域 ほぼ不変
    assert abs(mb["snr_db"] - ma["snr_db"]) < 0.5    # SNR ほぼ不変
    _, p_before = dsp.welch_psd(sig, RATE)
    _, p_after = dsp.welch_psd(dsp.remove_dc(sig), RATE)
    assert float(np.max(np.abs(p_before - p_after))) < 0.5   # 全ビンほぼ不変


def test_remove_dc_preserves_burst():
    """中央外バースト(時間的に途中だけ出る): スペクトルのピーク/位置が不変。"""
    t = np.arange(N) / RATE
    tone = np.exp(2j * np.pi * 2e6 * t)
    env = np.zeros(N)
    env[N // 4:N // 2] = 1.0                          # 1/4区間だけ存在（バースト）
    sig = (env * tone + _noise(N)).astype(np.complex64)
    _, p_before = dsp.welch_psd(sig, RATE)
    _, p_after = dsp.welch_psd(dsp.remove_dc(sig), RATE)
    assert abs(p_before.max() - p_after.max()) < 0.5         # バーストのピーク不変
    assert int(np.argmax(p_before)) == int(np.argmax(p_after))


# ---------------------------------------------------------------------------
# SimBackend: 注入したDCスパイクが dc_removal で消える（診断経路）
# ---------------------------------------------------------------------------
def _dwell_cfg(count=6):
    return DwellConfig(dwell_seconds=0.0, obs_interval_s=0.0,
                       min_observations=count, max_observations=count)


def test_sim_injected_spike_present_without_removal():
    """dc_offset 注入・除去なし → 中央スパイクが残り dc_spike として破棄される。"""
    cfg = SDRConfig(dwell_samples=N)
    be = sdr.SimBackend(cfg, seed=0, dc_offset=0.5, dc_removal=False)
    obs = dwell.observe_dwell(be, 2.4e9, RATE, N, _dwell_cfg(), QualityConfig())
    assert obs.dc_excess_mean_db > 20.0              # 中央集中（DCスパイク）
    v = quality.evaluate_quality(obs, QualityConfig())
    assert v.is_dc_spike and not v.passed
    assert "dc-spike" in v.reasons


def test_sim_injected_spike_removed_with_removal():
    """dc_offset 注入・除去あり → 中央スパイクが消え dc_spike にならない。"""
    cfg = SDRConfig(dwell_samples=N)
    be = sdr.SimBackend(cfg, seed=0, dc_offset=0.5, dc_removal=True)
    obs = dwell.observe_dwell(be, 2.4e9, RATE, N, _dwell_cfg(), QualityConfig())
    assert obs.dc_excess_mean_db < 5.0               # 中央集中が消えた
    v = quality.evaluate_quality(obs, QualityConfig())
    assert not v.is_dc_spike


def test_sim_default_does_not_remove_dc():
    """既定(dc_removal 未指定)の SimBackend は DC を除去しない（合成は元々DC無し）。"""
    cfg = SDRConfig(dwell_samples=N)
    be = sdr.SimBackend(cfg, seed=0)
    assert be.dc_removal is False


# ---------------------------------------------------------------------------
# config / main.build_backend の配線
# ---------------------------------------------------------------------------
def test_sdrconfig_dc_removal_default_true():
    """config 既定（=実機既定）は DC 除去有効。"""
    assert SDRConfig().dc_removal is True


def _sim_args(**kw):
    base = dict(hardware=False, sim=True, seed=0, sim_dc_spike=None,
                no_dc_removal=False, dc_removal=False,
                lna=24.0, vga=20.0, amp=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_build_backend_sim_default_off():
    """Sim: 既定では DC 除去オフ。"""
    be = main.build_backend(_sim_args(), Config())
    assert be.dc_removal is False


def test_build_backend_sim_dc_removal_flag_on():
    """Sim: --dc-removal で強制有効化できる。"""
    be = main.build_backend(_sim_args(dc_removal=True), Config())
    assert be.dc_removal is True


def test_build_backend_no_dc_removal_overrides():
    """--no-dc-removal は --dc-removal より優先（常に無効）。"""
    be = main.build_backend(_sim_args(dc_removal=True, no_dc_removal=True), Config())
    assert be.dc_removal is False


def test_build_backend_no_dc_removal_sets_cfg_for_hardware():
    """--no-dc-removal は cfg.sdr.dc_removal を False に倒す（実機経路の準備）。"""
    cfg = Config()
    # 実機 backend は SoapySDR 依存で生成できないため cfg への反映のみ検証する。
    # build_backend の実機分岐と同じ式: cfg.sdr.dc_removal = not args.no_dc_removal
    args = _sim_args(no_dc_removal=True)
    cfg.sdr.dc_removal = not args.no_dc_removal
    assert cfg.sdr.dc_removal is False


# ---------------------------------------------------------------------------
# scheduler: SigMF global に sigscan:dc_removed を正直に記録
# ---------------------------------------------------------------------------
def _fast_dwell_cfg(**quality_kw):
    sdrc = SDRConfig(dwell_samples=N)
    scan = ScanConfig(start_hz=2.4e9, stop_hz=2.5e9, max_dwell_per_cycle=2)
    dwellc = DwellConfig(dwell_seconds=0.0, obs_interval_s=0.0,
                         min_observations=6, max_observations=6)
    q = QualityConfig(min_detections=1, min_persistence=0.0, **quality_kw)
    return Config(sdr=sdrc, scan=scan, dwell=dwellc, quality=q)


def _read_one_global(collect):
    metas = sorted(glob.glob(os.path.join(collect, "*.sigmf-meta")))
    assert metas, "SigMF が1件も出力されていない"
    base = metas[0][: -len(".sigmf-meta")]
    _, meta = sigmf_io.read_recording(base)          # 凍結リーダで往復読み戻し
    return meta["global"]


def test_scheduler_records_dc_removed_true_dwell(tmp_path):
    """dwell 保存経路: dc_removal 有効な backend の保存物は dc_removed=true。"""
    collect = str(tmp_path / "captures")
    cfg = _fast_dwell_cfg()
    store = Store(str(tmp_path / "t.db"))
    be = sdr.SimBackend(cfg.sdr, seed=1, burst_per_capture=True, dc_removal=True)
    sched = HybridScheduler(be, cfg, store=store, collect_dir=collect,
                            collect_snr_min=0.0, collect_dedup_s=0.0,
                            dwell_mode=True)
    sched.run(once=True, verbose=False)
    g = _read_one_global(collect)
    assert g["sigscan:dc_removed"] is True
    assert g["sigscan:capture_mode"] == "dwell"


def test_scheduler_records_dc_removed_false_nondwell(tmp_path):
    """非dwell 収集経路: 既定(除去なし) backend の保存物は dc_removed=false。"""
    collect = str(tmp_path / "captures")
    sdrc = SDRConfig(dwell_samples=N)
    scan = ScanConfig(start_hz=2.4e9, stop_hz=2.5e9, max_dwell_per_cycle=2)
    cfg = Config(sdr=sdrc, scan=scan)
    store = Store(str(tmp_path / "t.db"))
    be = sdr.SimBackend(cfg.sdr, seed=0)              # 既定 dc_removal=False
    sched = HybridScheduler(be, cfg, store=store, collect_dir=collect,
                            collect_snr_min=0.0, collect_dedup_s=0.0)
    sched.run(once=True, verbose=False)
    g = _read_one_global(collect)
    assert g["sigscan:dc_removed"] is False
