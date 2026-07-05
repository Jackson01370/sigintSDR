"""
test_dwell.py — 滞在観測(dwell.observe_dwell)の集計をロックする。

滞在中に複数回 IQ を取得し、出現回数・持続率・検出マージン分布を集計し、最も良く
捉えた瞬間を代表として返すこと。バースト（出たり消えたり）を持続率に反映すること。
取得は継ぎ目 backend.capture_iq を、測定は dsp.measure_signal を使う。
"""
import numpy as np

import dwell
import quality
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


# ---------------------------------------------------------------------------
# 案3: bw_median_hz の母集団を「全観測」から「検出された観測」に限定する。
#   BLE 等の間欠バーストでは滞在中の過半が quiet 窓になり、全観測 median だと幅が
#   ~1 FFT ビン(や 0)に潰れて代表性を失う。既存の detected(det_snr>=detect_snr_db)を
#   再利用して検出観測の bw の median を採り、検出 0 件なら全観測にフォールバックする。
# ---------------------------------------------------------------------------
class _BurstBackend:
    """present で ~1.5MHz 幅の帯域制限バースト、absent でノイズのみを返す。

    バーストは検出マージンが高く『検出』に、ノイズ窓は低マージンで『未検出』になる
    （案3 が検出観測だけで median を採ることの検証用）。
    """
    def __init__(self, present_flags, seed=0):
        self.flags = list(present_flags)
        self.k = 0
        self.rng = np.random.default_rng(seed)

    def capture_iq(self, center_hz, rate, n):
        present = self.flags[self.k % len(self.flags)]
        self.k += 1
        noise = (0.01 * (self.rng.normal(size=n)
                         + 1j * self.rng.normal(size=n))).astype(np.complex64)
        if not present:
            return noise                                     # quiet窓(未検出・bw≈0)
        f = np.fft.fftfreq(n, d=1.0 / rate)
        s = self.rng.normal(size=n) + 1j * self.rng.normal(size=n)
        s[np.abs(f - 2e6) > 0.75e6] = 0.0                    # +2MHz中心 ~1.5MHz幅
        b = np.fft.ifft(s)
        b = b / np.std(b)
        return (b + noise).astype(np.complex64)


def test_bw_median_uses_detected_observations():
    """間欠バースト+quietノイズ窓の多観測列で、bw_median が検出観測(バースト)の
    実幅を代表し、quiet 窓の bw≈0 に潰れない（案3）。"""
    flags = [True, False, False, True, False, False, False, False, False, False]  # 2/10
    obs = dwell.observe_dwell(_BurstBackend(flags), 2.433e9, RATE, N,
                              _fixed_counts_cfg(10), QualityConfig())
    assert obs.n_detect == 2                       # バースト2回のみ検出
    assert obs.bw_median_hz > 1e6                  # 検出観測(~1.5MHz)を代表
    # 全観測 median なら quiet 窓(bw≈0)に支配され極小になる（旧挙動との差）。
    assert float(np.median(np.asarray(obs.bw_series))) < 0.5e6


def test_bw_median_fallback_when_no_detection():
    """検出0件(終始 quiet)なら全観測 median にフォールバックする（None/例外にしない）。"""
    obs = dwell.observe_dwell(_BurstBackend([False]), 2.433e9, RATE, N,
                              _fixed_counts_cfg(6), QualityConfig())
    assert obs.n_detect == 0
    assert obs.bw_median_hz == float(np.median(np.asarray(obs.bw_series)))


# ---------------------------------------------------------------------------
# DCスパイク（DCオフセット由来の中央スパイク）指標の集計
#   各取得IQから中央集中度(dc_excess)を測り、平均(中央集中)と std(時間不変性)を
#   集計する。これを quality.py が dc_spike 破棄に使う。
# ---------------------------------------------------------------------------
class _DcBackend:
    """kind に従った IQ を返す擬似バックエンド（DCスパイク指標の検証用）。

    kind: 'spike'  中央(DC)固定の定数オフセット = DCスパイク
          'offset' 中央外(+3MHz)のトーン
          'wide'   帯域を広く埋める信号(WiFi相当)
    present_flags が与えられれば、そのパターンで present/absent を切り替えて
    バースト（時間変動）を擬似する（None なら常時 present）。
    """
    def __init__(self, kind, present_flags=None, seed=0):
        self.kind = kind
        self.flags = present_flags
        self.k = 0
        self.rng = np.random.default_rng(seed)

    def capture_iq(self, center_hz, rate, n):
        present = True if self.flags is None else self.flags[self.k % len(self.flags)]
        self.k += 1
        noise = (0.02 * (self.rng.normal(size=n)
                         + 1j * self.rng.normal(size=n))).astype(np.complex64)
        if not present:
            return noise
        if self.kind == "spike":
            return (0.5 + noise).astype(np.complex64)        # 中央(DC)定数
        if self.kind == "offset":
            t = np.arange(n) / rate
            return (np.exp(2j * np.pi * 3e6 * t) + noise).astype(np.complex64)
        if self.kind == "wide":
            f = np.fft.fftfreq(n, d=1.0 / rate)
            s = self.rng.normal(size=n) + 1j * self.rng.normal(size=n)
            s[np.abs(f) > 8e6] = 0.0
            b = np.fft.ifft(s)
            b = b / np.std(b)
            return (b + noise).astype(np.complex64)
        return noise


def test_dc_excess_high_and_steady_for_center_constant():
    """中央定数(DCスパイク) → dc_excess の平均が大、観測間 std が小（時間不変）。"""
    be = _DcBackend("spike")
    obs = dwell.observe_dwell(be, 2.4e9, RATE, N, _fixed_counts_cfg(6),
                              QualityConfig())
    assert obs.dc_excess_mean_db > 20.0
    assert obs.dc_excess_std_db < 3.0
    assert len(obs.dc_excess_series) == 6


def test_dc_excess_low_for_offset_and_wideband():
    """中央外トーン・広帯域信号 → 中央集中せず dc_excess の平均が小。"""
    off = dwell.observe_dwell(_DcBackend("offset"), 2.4e9, RATE, N,
                              _fixed_counts_cfg(6), QualityConfig())
    wide = dwell.observe_dwell(_DcBackend("wide"), 2.4e9, RATE, N,
                               _fixed_counts_cfg(6), QualityConfig())
    assert off.dc_excess_mean_db < 5.0
    assert wide.dc_excess_mean_db < 5.0


def test_dc_excess_varies_for_time_varying_center_burst():
    """中央に出ても時間変動するバースト → dc_excess の観測間 std が大（時間不変でない）。"""
    be = _DcBackend("spike", present_flags=[True, False])    # 中央に出たり消えたり
    obs = dwell.observe_dwell(be, 2.4e9, RATE, N, _fixed_counts_cfg(8),
                              QualityConfig())
    assert obs.dc_excess_std_db > 5.0


def test_dc_spike_rejected_end_to_end():
    """観測→集計→品質ゲートを通し、中央定数のDCスパイクが dc_spike で破棄される。"""
    be = _DcBackend("spike")
    obs = dwell.observe_dwell(be, 2.4e9, RATE, N, _fixed_counts_cfg(6),
                              QualityConfig())
    v = quality.evaluate_quality(obs, QualityConfig())
    assert v.is_dc_spike
    assert not v.passed
    assert "dc-spike" in v.reasons
