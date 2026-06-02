"""
test_dwell.py — 滞在観測(dwell.observe_dwell)の集計をロックする。

滞在中に複数回 IQ を取得し、出現回数・持続率・検出マージン分布を集計し、最も良く
捉えた瞬間を代表として返すこと。バースト（出たり消えたり）を持続率に反映すること。
取得は継ぎ目 backend.capture_iq を、測定は dsp.measure_signal を使う。
"""
import numpy as np

import dwell
from config import DwellConfig, QualityConfig


class _PatternBackend:
    """present_flags のパターンに従って、強いトーン or ノイズを返す擬似バックエンド。

    capture_iq 呼び出しごとにパターンを1つ進める（バースト的な出現/消失を擬似）。
    """
    def __init__(self, present_flags, seed=0):
        self.present_flags = list(present_flags)
        self.k = 0
        self.rng = np.random.default_rng(seed)

    def capture_iq(self, center_hz, rate, n):
        present = self.present_flags[self.k % len(self.present_flags)]
        self.k += 1
        noise = (self.rng.normal(0, 1e-3, n)
                 + 1j * self.rng.normal(0, 1e-3, n)).astype(np.complex64)
        if not present:
            return noise
        t = np.arange(n) / rate
        tone = 0.5 * np.exp(2j * np.pi * (rate * 0.1) * t)   # 帯域内オフセットの細いトーン
        return (noise + tone).astype(np.complex64)


class _Clock:
    """呼ぶたびに step ずつ進む擬似時計（実時間に依存しないループ検証用）。"""
    def __init__(self, step):
        self.t = 0.0
        self.step = step

    def __call__(self):
        v = self.t
        self.t += self.step
        return v


def _fixed_counts_cfg(count):
    """ちょうど count 回観測する dwell 設定（実時間スリープなし）。"""
    return DwellConfig(dwell_seconds=0.0, obs_interval_s=0.0,
                       min_observations=count, max_observations=count)


RATE = 20e6
N = 1 << 14


def test_observe_counts_and_persistence():
    """持続率 = 検出された割合。出たり消えたりが反映される。"""
    flags = [True, False, True, True, False]            # 5回中3回出現
    be = _PatternBackend(flags * 2)                     # 10観測ぶん
    obs = dwell.observe_dwell(be, 2.4e9, RATE, N,
                              _fixed_counts_cfg(10), QualityConfig(),
                              target_src="detected")
    assert obs.n_obs == 10
    assert obs.n_detect == 6                             # 3/5 × 2周
    assert abs(obs.persistence - 0.6) < 1e-9
    assert obs.snr_max_db >= QualityConfig().detect_snr_db   # 出現時は検出マージン大


def test_always_absent_is_transient():
    """一度も出現しなければ持続率0（後段で単発として破棄される）。"""
    be = _PatternBackend([False])
    obs = dwell.observe_dwell(be, 2.4e9, RATE, N,
                              _fixed_counts_cfg(8), QualityConfig())
    assert obs.n_obs == 8
    assert obs.n_detect == 0
    assert obs.persistence == 0.0


def test_representative_is_a_present_capture():
    """代表(best_iq)は出現した観測（検出マージン最大）から選ばれる。"""
    flags = [False, False, True, False]
    be = _PatternBackend(flags)
    obs = dwell.observe_dwell(be, 2.4e9, RATE, N,
                              _fixed_counts_cfg(4), QualityConfig())
    assert obs.best_iq is not None
    assert obs.best_iq.dtype == np.complex64
    # 代表の検出マージンは出現時の大きな値（=全体の最大）
    assert obs.peak_db_rep > obs.noise_ref_db + 10
    # snr_series は2桁丸め、snr_max_db は未丸めなので丸めて比較
    assert round(obs.snr_max_db, 2) == max(obs.snr_series)


def test_snr_distribution_collected():
    """SNR(検出マージン)の分布が観測ごとに集計される。"""
    be = _PatternBackend([True, False])
    obs = dwell.observe_dwell(be, 2.4e9, RATE, N,
                              _fixed_counts_cfg(6), QualityConfig())
    assert len(obs.snr_series) == 6
    assert len(obs.bw_series) == 6
    assert obs.snr_std_db > 0                       # 出現/消失でばらつく
    assert obs.snr_max_db > obs.snr_mean_db         # 最大 > 平均（分布が取れている）


def test_deadline_with_injected_clock():
    """滞在時間(deadline)で打ち切るが、min_observations は保証する。"""
    be = _PatternBackend([True])
    # 1呼び出しごとに 0.3 進む時計。dwell_seconds=1.0 → 約4観測で deadline 超過。
    clk = _Clock(0.3)
    dcfg = DwellConfig(dwell_seconds=1.0, obs_interval_s=0.0,
                       min_observations=2, max_observations=100)
    obs = dwell.observe_dwell(be, 2.4e9, RATE, N, dcfg, QualityConfig(),
                              time_fn=clk, sleep_fn=lambda s: None)
    # deadline=1.0。time_fn は iter ごとに進み、k>=2 かつ time>=1.0 で停止。
    assert obs.n_obs == 4


def test_min_observations_enforced_past_deadline():
    """deadline を既に過ぎていても min_observations 回は必ず観測する。"""
    be = _PatternBackend([True])
    clk = _Clock(100.0)                  # 即座に deadline を超える時計
    dcfg = DwellConfig(dwell_seconds=1.0, obs_interval_s=0.0,
                       min_observations=5, max_observations=100)
    obs = dwell.observe_dwell(be, 2.4e9, RATE, N, dcfg, QualityConfig(),
                              time_fn=clk, sleep_fn=lambda s: None)
    assert obs.n_obs == 5


def test_noise_floor_override_changes_detection():
    """noise_floor_db を渡すと、その絶対基準で検出マージン(ピーク上昇分)を測る。

    ノイズのみのバックエンドは自己推定基準では未検出だが、極端に低い基準を渡すと
    ピーク上昇分が大きくなり全観測が検出扱いになる（基準が効いている証拠）。
    """
    be1 = _PatternBackend([False])
    self_est = dwell.observe_dwell(be1, 2.4e9, RATE, N, _fixed_counts_cfg(6),
                                   QualityConfig())
    assert self_est.n_detect == 0                    # 自己推定基準ではノイズは未検出

    be2 = _PatternBackend([False])
    forced = dwell.observe_dwell(be2, 2.4e9, RATE, N, _fixed_counts_cfg(6),
                                 QualityConfig(), noise_floor_db=-300.0)
    assert forced.noise_ref_db == -300.0
    assert forced.n_detect == 6                       # 低い基準なら全部「検出」になる
