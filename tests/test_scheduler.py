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


# ===========================================================================
# 帯域フォーカス（band_focus / --focus）
#   指定 [start, stop] に張り付き、_build_targets の合流点(add)1点で範囲外候補を
#   除外する。既定 OFF（従来挙動）。SimBackend + 完全 BAND_PLAN で検証する。
# ===========================================================================
_FOCUS_LO, _FOCUS_HI = 2.4e9, 2.5e9


def _focus_sched(band_focus, max_dwell=6):
    """完全 BAND_PLAN・範囲 2.4-2.5GHz の scheduler を作る（store なし）。

    bands は Config 既定の完全バンドプラン（GPS/W56 等の範囲外バンドを含む）を
    使い、関所の効きを実バンドプランで検証する。band_focus 以外は既定。
    """
    sdr = SDRConfig(dwell_samples=1 << 14)
    scan = ScanConfig(start_hz=_FOCUS_LO, stop_hz=_FOCUS_HI,
                      band_focus=band_focus, max_dwell_per_cycle=max_dwell)
    cfg = Config(sdr=sdr, scan=scan)
    be = SimBackend(cfg.sdr, seed=0)
    return HybridScheduler(be, cfg)


def test_focus_on_excludes_out_of_band_plan_targets():
    """T1: focus ON で範囲外バンドプラン目標(GPS/W56 等)が targets に出ない。"""
    sched = _focus_sched(band_focus=True)
    sched._segments = []                       # サーベイ検出なし＝巡回のみで埋める
    targets = sched._build_targets()
    assert targets                             # 範囲内バンドで埋まる（空にならない）
    for t in targets:
        assert _FOCUS_LO <= t["center"] <= _FOCUS_HI, t
    # 範囲外バンド由来（GPS 1176MHz / W56 5597MHz 等）は1件も無い
    assert not any(t["center"] < _FOCUS_LO or t["center"] > _FOCUS_HI
                   for t in targets)


def test_focus_on_excludes_survey_spillover():
    """T2: 範囲端の外(2504MHz 相当)のサーベイ由来候補も同じ関所で消える。"""
    sched = _focus_sched(band_focus=True)
    # サーベイ端の食み出しを模した検出帯を注入（--stop 2.5e9 の外側）。
    sched._segments = [dict(f_lo=2.503e9, f_hi=2.505e9, f_center=2.504e9,
                            bw_hz=1e6, peak_db=-40.0, snr_db=20.0)]
    targets = sched._build_targets()
    assert all(_FOCUS_LO <= t["center"] <= _FOCUS_HI for t in targets)
    assert not any(abs(t["center"] - 2.504e9) < 1e6 for t in targets)


def test_focus_off_keeps_out_of_band_targets_regression():
    """T3: focus OFF は従来どおり範囲外バンドプラン目標を含む（回帰ロック）。"""
    sched = _focus_sched(band_focus=False)
    sched._segments = []
    targets = sched._build_targets()
    # バンドプラン巡回は範囲外(例: GPS<2.4GHz)も従来どおり拾う
    assert any(t["center"] < _FOCUS_LO or t["center"] > _FOCUS_HI
               for t in targets), targets
    assert any(t["src"].startswith("band:") for t in targets)


def test_focus_records_band_focus_in_sigmf_global(tmp_path):
    """T4: focus 有効で収集した記録の global に sigscan:band_focus が載る。"""
    collect = str(tmp_path / "captures")
    sdr = SDRConfig(dwell_samples=1 << 14)
    scan = ScanConfig(start_hz=_FOCUS_LO, stop_hz=_FOCUS_HI,
                      band_focus=True, max_dwell_per_cycle=2)
    cfg = Config(sdr=sdr, scan=scan)
    store = Store(str(tmp_path / "t.db"))
    be = SimBackend(cfg.sdr, seed=0)
    sched = HybridScheduler(be, cfg, store=store, collect_dir=collect,
                            collect_snr_min=0.0)
    sched.run(once=True, verbose=False)

    data_files = sorted(glob.glob(os.path.join(collect, "*.sigmf-data")))
    assert len(data_files) >= 1
    base = data_files[0][: -len(".sigmf-data")]
    _, meta = sigmf_io.read_recording(base)
    # 来歴は範囲つき dict で記録（bool ではなく {"start","stop"}）
    assert meta["global"]["sigscan:band_focus"] == {"start": _FOCUS_LO,
                                                     "stop": _FOCUS_HI}


# ===========================================================================
# dwell オフセットチューニング（狙った獲物を DC 位置から避ける）
#   狭帯域ターゲットはチューナーを数MHzずらし、獲物が DC(=0Hz)に乗って dc-spike で
#   構造的に落ちるのを防ぐ。既定 dwell_offset_hz=0 で完全に従来挙動。追加のみ。
# ===========================================================================
def test_dwell_tune_offset_application_conditions():
    """適用条件のユニット: f_tune = center + dwell_tune_offset(...)。

    (a) offset=0 → 0（従来）。(b) 狭帯域(bw<=max) → offset 適用。
    (c) 広帯域(bw>max) → 0（信号端が窓外に出るため不適用）。(d) bw 不明 → 0。
    """
    from scheduler import dwell_tune_offset
    center = 2.402e9
    assert dwell_tune_offset(1.5e6, 0.0, 8e6) == 0.0            # (a)
    assert dwell_tune_offset(1.5e6, 4e6, 8e6) == 4e6           # (b)
    assert center + dwell_tune_offset(1.5e6, 4e6, 8e6) == center + 4e6
    assert dwell_tune_offset(16e6, 4e6, 8e6) == 0.0            # (c) 広帯域は不適用
    assert dwell_tune_offset(None, 4e6, 8e6) == 0.0           # (d) bw 不明は不適用
    assert dwell_tune_offset(8e6, 4e6, 8e6) == 4e6            # 境界 bw==max は適用(<=)


def _offset_sched(tmp_path, offset_hz, F=2.402e9):
    """狭帯域CW(定常)を F に置き、F を狙う detected ターゲット1件を注入した dwell 収集。

    bw=1e6 は narrow_bw(0.7e6) より広いので narrow-steady-spur は不適用＝dc-spike の
    効きだけを裸で観測できる。offset_hz を cfg.sdr.dwell_offset_hz に設定。
    """
    from sdr import _SimSignal
    sdr = SDRConfig(dwell_samples=1 << 14, dwell_offset_hz=offset_hz)
    scan = ScanConfig(start_hz=2.4e9, stop_hz=2.5e9, max_dwell_per_cycle=1)
    dwl = DwellConfig(dwell_seconds=0.0, obs_interval_s=0.0,
                      min_observations=12, max_observations=12)
    cfg = Config(sdr=sdr, scan=scan, dwell=dwl, quality=QualityConfig())
    be = SimBackend(cfg.sdr, seed=0)
    be.signals = [_SimSignal(F, 0.1e6, 30, prob=1.0, kind="cw")]   # 定常・狭帯域CW
    collect = str(tmp_path / "captures")
    sched = HybridScheduler(be, cfg, collect_dir=collect, collect_snr_min=0.0,
                            collect_dedup_s=0.0, dwell_mode=True)
    # detected ターゲット注入: bw=1e6(>narrow_bw) で narrow-steady-spur を外す。
    sched._segments = [dict(f_center=F, bw_hz=1.0e6, snr_db=30.0,
                            f_lo=F - 0.5e6, f_hi=F + 0.5e6, peak_db=-40.0)]
    return sched, collect, F


def test_dwell_offset_zero_drops_dc_spike_prey_sim(tmp_path):
    """offset=0: 狭帯域CWが DC に乗り dc-spike のみで drop（narrow-steady-spur ではない）。

    再現の核心: bw=1e6 で narrow-steady-spur は不適用、CW が DC(0Hz)に定数として乗るため
    dc_excess が大かつ時間不変 → dc-spike 判定（quality.py: dc_excess_mean>=12 かつ
    dc_excess_std<=3）で構造的に落ちる。
    """
    sched, collect, F = _offset_sched(tmp_path, offset_hz=0.0)
    outcomes = sched.dwell_observe_cycle()
    o = outcomes[0]
    assert not o["verdict"].passed
    assert o["verdict"].is_dc_spike
    assert "dc-spike" in o["verdict"].reasons
    assert "narrow-steady-spur" not in o["verdict"].reasons   # bw=1e6 で不適用
    assert not o["saved"]
    assert not glob.glob(os.path.join(collect, "*.sigmf-data"))


def test_dwell_offset_saves_prey_with_correct_abs_freq_sim(tmp_path):
    """offset=4e6: 同じ獲物が保存され、annotation の絶対周波数が真の F と一致（±分解能）。

    ここが崩れると全記録の周波数が静かに壊れるため最重要（絶対周波数の一貫性）。
    チューナーは F+4e6 に合うが、記録される検出中心は Sim 信号の真の周波数 F。
    """
    sched, collect, F = _offset_sched(tmp_path, offset_hz=4e6)
    outcomes = sched.dwell_observe_cycle()
    o = outcomes[0]
    assert o["verdict"].passed and not o["verdict"].is_dc_spike
    assert o["saved"]
    data = sorted(glob.glob(os.path.join(collect, "*.sigmf-data")))
    assert len(data) == 1
    base = data[0][: -len(".sigmf-data")]
    _, meta = sigmf_io.read_recording(base)
    # SigMF captures[0].core:frequency = f_tune（物理IQ中心 = F+4e6）
    assert abs(meta["captures"][0]["core:frequency"] - (F + 4e6)) < 1e3
    # annotation の絶対周波数(検出中心) = Sim 信号の真の周波数 F（±測定分解能）
    ann = meta["annotations"][0]
    det_center = (ann["core:freq_lower_edge"] + ann["core:freq_upper_edge"]) / 2.0
    assert abs(det_center - F) < 0.2e6
    # 来歴: 適用した実効オフセットが global に記録される
    assert meta["global"]["sigscan:dwell_offset_hz"] == 4e6


def test_dwell_offset_default_zero_is_unchanged_behavior_sim(tmp_path):
    """既定(dwell_offset_hz=0): f_tune=center で記録の来歴 offset=0.0（従来挙動の回帰）。

    既定では DC に乗った狭帯域CWは従来どおり dc-spike で落ちる（保存されない）ため、
    ここでは「offset を適用しない同一経路で来歴 0.0」を、救済されるオフセット版との
    対比としてゲート無効化で1件保存し確認する。
    """
    sched, collect, F = _offset_sched(tmp_path, offset_hz=0.0)
    sched.cfg.quality.enabled = False           # ゲート無効化で確実に1件保存し来歴を検証
    sched.dwell_observe_cycle()
    data = sorted(glob.glob(os.path.join(collect, "*.sigmf-data")))
    assert len(data) == 1
    base = data[0][: -len(".sigmf-data")]
    _, meta = sigmf_io.read_recording(base)
    assert meta["global"]["sigscan:dwell_offset_hz"] == 0.0     # 不適用
    # f_tune=center: 物理IQ中心は狙い F のまま（オフセットされていない）
    assert abs(meta["captures"][0]["core:frequency"] - F) < 1e3
