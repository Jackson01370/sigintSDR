"""
test_cnntrain_m2_5.py — M2.5 の simgen 形状再アラインをロックする。

T9 : narrowband-burst が短く疎ら（1〜3発・各 ~3%）。帯域は 1〜2.5MHz 維持。
T10: wideband-ofdm がほぼ全幅・縦縞 3〜6%・2モード（少数+沈黙 / クラスタ）で、
     pulse-radar の厳密周期と時間構造で分離。**pulse-radar 定義は不変**であること。
T11: 中心外スプリアス線が全クラスに無相関で入り、強度が較正域内（DC 注入は不変）。

凍結契約（spec.py / sigmf_io.py）は触らない。既存テストは無改変（追加のみ）。
実データ（captures/）は使わない。数十秒以内。
"""
import os
import glob
import tempfile

import numpy as np

import sigmf_io
import dsp
from cnntrain import simgen, classes


# ---------------------------------------------------------------------------
# T9: narrowband-burst の形状（短く 1〜3 発）
# ---------------------------------------------------------------------------
def test_t9_narrowband_short_and_sparse():
    """バースト長は窓の ~3%（0.025〜0.05）、本数 1〜3。帯域は 1〜2.5MHz 維持。"""
    N = simgen.N
    rng = np.random.default_rng(90)
    counts = []
    for _ in range(300):
        _, starts, lengths = simgen._short_sparse_burst_envelope(N, rng)
        counts.append(len(starts))
        for l in lengths:
            assert 0.024 <= l / N <= 0.051           # 短い（実測 ~0.3-0.6ms/13ms）
    assert min(counts) >= 1 and max(counts) <= 3     # 1〜3 発・疎ら
    # 全体としても 1 発だけに偏っていない（疎らさの担保）。
    assert max(counts) >= 2

    # 帯域は維持（1〜2.5MHz）。
    for _ in range(50):
        _, info = simgen._gen_narrowband_burst(rng)
        assert 1.0e6 <= info["bw"] <= 2.5e6


# ---------------------------------------------------------------------------
# T10: wideband-ofdm 全幅 + 2モード、pulse-radar との分離・pulse 不変
# ---------------------------------------------------------------------------
def _interval_cv(starts) -> float:
    if len(starts) < 2:
        return 0.0
    d = np.diff(starts)
    return float(np.std(d) / (np.mean(d) + 1e-9))


def test_t10_wideband_fullwidth_two_modes_and_separation():
    N = simgen.N
    rate = simgen.RATE
    rng = np.random.default_rng(100)

    # ほぼ全幅（実 WiFi は画面上端〜下端）。
    bws = [simgen._gen_wideband_ofdm(rng)[1]["bw"] / rate for _ in range(60)]
    assert min(bws) >= 0.88 and max(bws) <= 1.0

    # 縦縞幅 3〜6%、2モード（クラスタ span<0.5 と 拡散 span>0.55 の両方が出る）。
    spans = []
    for _ in range(200):
        _, starts, lengths = simgen._irregular_burst_envelope(N, rng)
        for l in lengths:
            assert 0.029 <= l / N <= 0.061
        if len(starts) >= 1:
            span = (max(starts) + max(lengths) - min(starts)) / N
            spans.append(span)
    spans = np.array(spans)
    assert (spans < 0.5).sum() > 0       # 密集クラスタ・モード(ii) 由来
    assert (spans > 0.55).sum() > 0      # 少数+沈黙・モード(i) 由来

    # 時間構造で pulse-radar と分離（wideband 非周期 / pulse 厳密周期）。
    wb_cv = [_interval_cv(simgen._irregular_burst_envelope(N, rng)[1]) for _ in range(80)]
    pl_cv = [_interval_cv(simgen._periodic_pulse_envelope(N, rng)[1]) for _ in range(80)]
    assert max(pl_cv) < 0.05
    assert np.mean(wb_cv) > 0.15
    assert np.mean(wb_cv) > np.mean(pl_cv) + 0.1


def test_t10_pulse_radar_definition_unchanged():
    """pulse-radar は厳密周期・短パルス・一様幅のまま（M2.5 で変更しない）。"""
    N = simgen.N
    rng = np.random.default_rng(101)
    for _ in range(60):
        _, starts, lengths = simgen._periodic_pulse_envelope(N, rng)
        assert len(starts) >= 5                       # 多数の短パルス
        # 厳密周期: 間隔が一定（CV≈0）。
        assert _interval_cv(starts) < 0.02
        # 一様幅: すべて同じ長さ。
        assert len(set(lengths)) == 1
        # 短パルス: 1 本は窓の数% 未満（wideband の 3〜6% より短い側）。
        assert lengths[0] / N < 0.05


# ---------------------------------------------------------------------------
# T11: 中心外スプリアス注入（全クラス無相関・較正域・DC 不変）
# ---------------------------------------------------------------------------
def _offcenter_prominence(iq, rate) -> float:
    f, p = dsp.welch_psd(iq, rate, nperseg=1024)
    af = np.abs(f)
    W = 6
    best = -1e9
    for i in np.flatnonzero(af > 0.5e6):
        lo = max(0, i - W); hi = min(p.size, i + W + 1)
        loc = np.concatenate([p[lo:i], p[i + 1:hi]])
        if loc.size:
            best = max(best, p[i] - np.median(loc))
    return best


def test_t11_spurious_injection_calibrated_and_offcenter():
    """注入後、中心外に細線が立ち、prominence が実測較正域に入る。"""
    rate = simgen.RATE
    rng = np.random.default_rng(110)
    proms = []
    for _ in range(200):
        iq, _ = simgen._gen_noise_only(rng)          # 信号なしでスプリアスを分離
        iq2, injected, amp = simgen._inject_spurious_line(iq, rate, rng)
        if injected:
            proms.append(_offcenter_prominence(iq2, rate))
    assert len(proms) > 0
    proms = np.array(proms)
    # 実測 off-center prominence は ~1.7..10.7dB。余裕を持たせた較正域に収まる。
    assert proms.min() >= 1.0 and proms.max() <= 12.0
    assert 2.0 <= np.median(proms) <= 6.0


def test_t11_spurious_is_class_uncorrelated_and_dc_preserved():
    """スプリアス注入率がクラス無相関、かつ DC 注入(M2)が壊れていない。"""
    d = tempfile.mkdtemp(prefix="m25spur_")
    simgen.generate(d, per_class=30, seed=11)
    spur_rates = {}
    dc_rates = {}
    for cls in classes.CLASSES:
        metas = glob.glob(os.path.join(d, f"{cls}_*.sigmf-meta"))
        sflags = []
        dflags = []
        for mp in metas:
            _, meta = sigmf_io.read_recording(mp[:-len(".sigmf-meta")])
            g = meta["global"]
            sflags.append(bool(g.get("sigscan:spur_injected")))
            dflags.append(bool(g.get("sigscan:dc_injected")))
        spur_rates[cls] = sum(sflags) / len(sflags)
        dc_rates[cls] = sum(dflags) / len(dflags)
    # スプリアスは全クラスで起き、率が設定値(0.3)周辺で揃う（クラス無相関）。
    assert all(0.1 <= r <= 0.55 for r in spur_rates.values()), spur_rates
    assert (max(spur_rates.values()) - min(spur_rates.values())) <= 0.4, spur_rates
    # DC 注入(M2)は引き続き全クラスで高率（壊れていない）。
    assert all(r >= 0.5 for r in dc_rates.values()), dc_rates
