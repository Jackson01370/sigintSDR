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
from config import Config, SDRConfig, ScanConfig
from sdr import SimBackend
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
