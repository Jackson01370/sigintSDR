"""
test_scheduler.py — スケジューラ本体(scheduler.py)の1サイクルをロックする。

- SimBackend で run(once=True) が例外なく回る
- collect_dir 指定時に SigMF（自動ラベル付き）が出力され、読み戻せて契約を満たす

注: CI を速く保つため帯域とサンプル数を絞った Config を使う（設定値のみ変更）。
"""
import glob
import os

import numpy as np

import sigmf_io
from config import Config, SDRConfig, ScanConfig, DwellConfig, QualityConfig
from sdr import SDRBackend, SimBackend
from store import Store
from scheduler import HybridScheduler


def _fast_cfg():
    """2.4GHz帯のみ・小サンプルで素早く1サイクル回す設定。"""
    sdr = SDRConfig(dwell_samples=1 << 14)
    scan = ScanConfig(start_hz=2.4e9, stop_hz=2.5e9, max_dwell_per_cycle=2)
    return Config(sdr=sdr, scan=scan)


def _make(tmp_path, **kw):
    cfg = _fast_cfg()
    store = Store(str(tmp_path / "t.db"))
    backend = SimBackend(cfg.sdr, seed=0)
    return HybridScheduler(backend, cfg, store=store, **kw), store


def test_run_once_no_exception_sim(tmp_path):
    """SimBackend で run(once=True) が例外なく完走し、store に記録される。"""
    sched, store = _make(tmp_path)
    sched.run(once=True, verbose=False)        # 例外が出ないこと自体が検証
    rows = store.recent(50)
    assert len(rows) >= 1


def test_collect_dir_emits_sigmf(tmp_path):
    """collect_dir 指定で SigMF ペアが出力され、読み戻せて契約を満たす。"""
    collect = str(tmp_path / "captures")
    sched, _ = _make(tmp_path, collect_dir=collect, collect_snr_min=0.0)
    sched.run(once=True, verbose=False)

    data_files = sorted(glob.glob(os.path.join(collect, "*.sigmf-data")))
    meta_files = sorted(glob.glob(os.path.join(collect, "*.sigmf-meta")))
    assert len(data_files) >= 1
    assert len(data_files) == len(meta_files)

    base = data_files[0][: -len(".sigmf-data")]
    iq, meta = sigmf_io.read_recording(base)

    assert iq.dtype == np.complex64
    assert len(iq) > 0
    g = meta["global"]
    assert g["core:datatype"] == "cf32_le"
    # Sim 由来であることが hw に正直に記録される
    assert g["core:hw"].startswith("sigscan-sim")
    assert g["sigscan:rep_version"]            # rep_version 埋め込み
    # 自動ラベル annotation が付いている
    assert len(meta["annotations"]) >= 1
    ann = meta["annotations"][0]
    for key in ("core:freq_lower_edge", "core:freq_upper_edge",
                "sigscan:confidence", "sigscan:method", "sigscan:snr_db"):
        assert key in ann


def test_collect_dir_created_when_missing(tmp_path):
    """存在しない collect_dir はコンストラクタで生成される。"""
    collect = str(tmp_path / "nested" / "captures")
    assert not os.path.isdir(collect)
    _make(tmp_path, collect_dir=collect, collect_snr_min=0.0)
    assert os.path.isdir(collect)


# ===========================================================================
# 滞在観測モード（dwell_mode）
# ===========================================================================
def _fast_dwell_cfg(**quality_kw):
    """2.4GHz帯・短い滞在（スリープなし・固定観測回数）で素早く回す dwell 設定。"""
    sdr = SDRConfig(dwell_samples=1 << 14)
    scan = ScanConfig(start_hz=2.4e9, stop_hz=2.5e9, max_dwell_per_cycle=3)
    dwell = DwellConfig(dwell_seconds=0.0, obs_interval_s=0.0,
                        min_observations=12, max_observations=12)
    quality = QualityConfig(**quality_kw)
    return Config(sdr=sdr, scan=scan, dwell=dwell, quality=quality)


def test_dwell_mode_run_once_sim(tmp_path):
    """dwell_mode で run(once=True) が例外なく完走し、検出ログが残る。"""
    cfg = _fast_dwell_cfg()
    store = Store(str(tmp_path / "t.db"))
    be = SimBackend(cfg.sdr, seed=1, burst_per_capture=True)
    sched = HybridScheduler(be, cfg, store=store, dwell_mode=True)
    sched.run(once=True, verbose=False)
    assert len(store.recent(50)) >= 1


def test_dwell_mode_saves_sigmf_with_quality_meta(tmp_path):
    """合格した滞在観測が SigMF 保存され、annotation に品質メタ(sigscan:)が載る。"""
    collect = str(tmp_path / "captures")
    # ゲートは緩めて確実に1件保存させ、メタ記録の経路を検証する。
    cfg = _fast_dwell_cfg(min_detections=1, min_persistence=0.0)
    store = Store(str(tmp_path / "t.db"))
    be = SimBackend(cfg.sdr, seed=1, burst_per_capture=True)
    sched = HybridScheduler(be, cfg, store=store, collect_dir=collect,
                            collect_snr_min=0.0, collect_dedup_s=0.0,
                            dwell_mode=True)
    sched.run(once=True, verbose=False)

    data_files = sorted(glob.glob(os.path.join(collect, "*.sigmf-data")))
    meta_files = sorted(glob.glob(os.path.join(collect, "*.sigmf-meta")))
    assert len(data_files) >= 1
    assert len(data_files) == len(meta_files)

    base = data_files[0][: -len(".sigmf-data")]
    iq, meta = sigmf_io.read_recording(base)        # 凍結リーダで往復読み戻し可能
    assert iq.dtype == np.complex64 and len(iq) > 0

    g = meta["global"]
    assert g["core:hw"].startswith("sigscan-sim")    # 出所の正直な記録
    assert g["sigscan:capture_mode"] == "dwell"      # 滞在観測である旨を global に記録

    ann = meta["annotations"][0]
    # 既存の根拠メタ
    for key in ("core:freq_lower_edge", "sigscan:confidence",
                "sigscan:method", "sigscan:snr_db"):
        assert key in ann
    # 追加した品質メタ（観測回数・持続率・SNR統計・スプリアス疑い）
    for key in ("sigscan:dwell_obs", "sigscan:dwell_detect", "sigscan:persistence",
                "sigscan:snr_max_db", "sigscan:snr_std_db", "sigscan:spur_suspect",
                "sigscan:quality_pass"):
        assert key in ann
    assert ann["sigscan:quality_pass"] is True
    assert 0.0 <= ann["sigscan:persistence"] <= 1.0


class _TransientBackend(SDRBackend):
    """サーベイは平坦ノイズ、ドウェルは period 回に n_present 回だけ一瞬出る擬似機。

    「一瞬かすっただけの単発」を再現し、厳しめゲートが破棄することの実証に使う。
    """
    def __init__(self, n_present=1, period=12):
        self.n_present = n_present
        self.period = period
        self.k = 0
        self.rng = np.random.default_rng(0)

    def sweep_power(self, start_hz, stop_hz, bin_hz):
        f = np.linspace(start_hz, stop_hz, 64)
        return f, -110.0 + self.rng.normal(0, 1.0, 64)   # 平坦ノイズ（検出帯なし）

    def capture_iq(self, center_hz, rate, n):
        present = (self.k % self.period) < self.n_present
        self.k += 1
        noise = (self.rng.normal(0, 1e-3, n)
                 + 1j * self.rng.normal(0, 1e-3, n)).astype(np.complex64)
        if not present:
            return noise
        t = np.arange(n) / rate
        return (noise + 0.3 * np.exp(2j * np.pi * (rate * 0.1) * t)).astype(np.complex64)


def test_dwell_mode_strict_discards_transient(tmp_path):
    """厳しめゲート: 一瞬かすっただけの単発は保存されない（ログは残る）。"""
    collect = str(tmp_path / "captures")
    sdr = SDRConfig(dwell_samples=1 << 14)
    scan = ScanConfig(start_hz=2.4e9, stop_hz=2.5e9, max_dwell_per_cycle=1)
    dwell = DwellConfig(dwell_seconds=0.0, obs_interval_s=0.0,
                        min_observations=12, max_observations=12)
    cfg = Config(sdr=sdr, scan=scan, dwell=dwell, quality=QualityConfig())
    store = Store(str(tmp_path / "t.db"))
    be = _TransientBackend(n_present=1, period=12)       # 12回中1回だけ出現
    sched = HybridScheduler(be, cfg, store=store, collect_dir=collect,
                            collect_snr_min=0.0, collect_dedup_s=0.0,
                            dwell_mode=True)
    sched.run(once=True, verbose=False)

    assert sched._collected == 0                          # 単発は保存しない
    assert not glob.glob(os.path.join(collect, "*.sigmf-data"))
    assert len(store.recent(50)) >= 1                     # ただし検出ログは残す
