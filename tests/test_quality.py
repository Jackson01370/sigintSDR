"""
test_quality.py — 品質ゲート(quality.py)の足切りロジックをロックする。

「量より質」: 保存するのは持続性があり、極細スプリアスでも等間隔コムスプリアスでも
ないものだけ。ただし幅だけで切らず「同一強度で居座る」かどうかと併用し、バースト性の
ある正規の狭帯域信号(BLE等)を誤って捨てないことを検証する。
"""
import dwell
import quality
from config import QualityConfig


def _obs(**kw):
    """テスト用の DwellObservation を既定値から組み立てる。"""
    d = dict(
        center_hz=2.437e9, target_src="detected", n_obs=20, n_detect=15,
        persistence=0.75, snr_max_db=30.0, snr_mean_db=18.0, snr_std_db=10.0,
        snr_detect_mean_db=25.0, bw_rep_hz=20e6, bw_median_hz=20e6,
        occupied_frac_rep=0.6, peak_db_rep=-70.0, noise_ref_db=-110.0,
        best={"center_hz": 2.437e9, "bw_hz": 20e6, "snr_db": 30.0},
        best_iq=None, snr_series=[], bw_series=[],
    )
    d.update(kw)
    return dwell.DwellObservation(**d)


def test_persistent_wideband_passes():
    """持続性があり広帯域でばらつきのある信号は合格（破棄理由なし）。"""
    v = quality.evaluate_quality(_obs(), QualityConfig())
    assert v.passed
    assert v.reasons == []
    assert not v.is_spur_suspect


def test_transient_oneoff_rejected():
    """一瞬かすっただけの単発（検出回数不足・低持続率）は破棄。"""
    obs = _obs(n_detect=1, persistence=0.05)
    v = quality.evaluate_quality(obs, QualityConfig())
    assert not v.passed
    assert any("transient" in r for r in v.reasons)
    assert any("low-persistence" in r for r in v.reasons)


def test_narrow_steady_spur_rejected():
    """極細かつ同一強度で居座る(低分散・高持続)山はスプリアスとして破棄。"""
    obs = _obs(bw_rep_hz=0.3e6, snr_std_db=0.2, persistence=1.0, n_detect=20)
    v = quality.evaluate_quality(obs, QualityConfig())
    assert v.is_spur_suspect
    assert not v.passed
    assert "narrow-steady-spur" in v.reasons


def test_narrow_bursty_signal_kept():
    """正規の狭帯域信号(BLE等)=細いがバースト性があり強度がばらつく → 残す。

    幅だけで切らず、持続性(steady でない)と併用して判定することの検証。
    """
    obs = _obs(bw_rep_hz=0.3e6, snr_std_db=15.0, persistence=0.5, n_detect=10)
    v = quality.evaluate_quality(obs, QualityConfig())
    assert not v.is_spur_suspect          # 居座りスプリアスではない
    assert v.passed
    assert v.reasons == []


def test_bw_override_controls_narrow_decision():
    """bw_hz オーバーライドが「極細」判定を支配する（サーベイ実測bwを使える）。"""
    qcfg = QualityConfig()
    # 実測 bw=0（measure_signal が当てにならないケース）でも、広いサーベイbwを渡せば
    # 居座りでも spur 扱いしない。
    obs = _obs(bw_rep_hz=0.0, snr_std_db=0.2, persistence=1.0, n_detect=20)
    v_wide = quality.evaluate_quality(obs, qcfg, bw_hz=20e6)
    assert not v_wide.is_spur_suspect and v_wide.passed
    # 逆に細いサーベイbwを渡せば spur 判定。
    v_narrow = quality.evaluate_quality(obs, qcfg, bw_hz=0.3e6)
    assert v_narrow.is_spur_suspect and not v_narrow.passed


def test_gate_disabled_passes_everything():
    """enabled=False ならゲート無効化で何でも合格（フラグ計算は維持）。"""
    qcfg = QualityConfig(enabled=False)
    obs = _obs(n_detect=0, persistence=0.0, bw_rep_hz=0.1e6, snr_std_db=0.0)
    v = quality.evaluate_quality(obs, qcfg)
    assert v.passed
    assert v.reasons == []


def test_thresholds_are_configurable():
    """しきい値が config で調整できる（緩めれば単発も通る）。"""
    obs = _obs(n_detect=2, persistence=0.10)
    strict = quality.evaluate_quality(obs, QualityConfig())
    assert not strict.passed
    loose = quality.evaluate_quality(
        obs, QualityConfig(min_detections=1, min_persistence=0.05))
    assert loose.passed


# ---------------------------------------------------------------------------
# 等間隔・同一強度コムスプリアス（クロスターゲット）
# ---------------------------------------------------------------------------
def test_comb_spurs_equal_spacing_same_power_flagged():
    """等間隔(1MHz)・同一強度の細いピーク列はコムスプリアスとして全て検出。"""
    qcfg = QualityConfig()
    obs = [
        _obs(center_hz=2400e6 + i * 1e6, bw_rep_hz=0.3e6, peak_db_rep=-50.0)
        for i in range(4)
    ]
    # 等間隔列から外れた広帯域・別強度のものは候補にすらならない/ランに入らない
    obs.append(_obs(center_hz=2410e6, bw_rep_hz=20e6, peak_db_rep=-50.0))   # 広帯域
    obs.append(_obs(center_hz=2500e6, bw_rep_hz=0.3e6, peak_db_rep=-90.0))   # 孤立・別強度
    flags = quality.flag_comb_spurs(obs, qcfg)
    assert flags[:4] == [True, True, True, True]   # 4本の等間隔コム
    assert flags[4] is False                        # 広帯域は対象外
    assert flags[5] is False                        # 孤立ピークはラン未満


def test_comb_unequal_spacing_not_flagged():
    """間隔が不揃いなら（受信機コムではない）誤検出しない。"""
    qcfg = QualityConfig()
    centers = [2400e6, 2401e6, 2403e6]   # 間隔 1MHz, 2MHz（不揃い）
    obs = [_obs(center_hz=c, bw_rep_hz=0.3e6, peak_db_rep=-50.0) for c in centers]
    flags = quality.flag_comb_spurs(obs, qcfg)
    assert flags == [False, False, False]


def test_comb_varying_power_not_flagged():
    """等間隔でも強度が揃っていなければコム(同一強度で居座る)とはみなさない。"""
    qcfg = QualityConfig()
    powers = [-50.0, -60.0, -70.0, -80.0]   # 強度がバラバラ
    obs = [_obs(center_hz=2400e6 + i * 1e6, bw_rep_hz=0.3e6, peak_db_rep=p)
           for i, p in enumerate(powers)]
    flags = quality.flag_comb_spurs(obs, qcfg)
    assert flags == [False, False, False, False]


def test_comb_spur_reason_in_verdict():
    """comb_spur=True を渡すと破棄理由に comb-spur が入る。"""
    v = quality.evaluate_quality(_obs(), QualityConfig(), comb_spur=True)
    assert not v.passed
    assert v.is_comb_spur
    assert "comb-spur" in v.reasons


def test_quality_annotation_meta_keys():
    """annotation 用品質メタが sigscan: 名前空間で必要キーを持つ。"""
    obs = _obs()
    v = quality.evaluate_quality(obs, QualityConfig())
    meta = quality.quality_annotation_meta(obs, v)
    for key in ("sigscan:dwell_obs", "sigscan:dwell_detect", "sigscan:persistence",
                "sigscan:snr_max_db", "sigscan:snr_mean_db", "sigscan:snr_std_db",
                "sigscan:spur_suspect", "sigscan:quality_pass"):
        assert key in meta
    assert all(k.startswith("sigscan:") for k in meta)
