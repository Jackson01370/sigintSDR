"""
test_spec.py — 正準スペクトログラム表現(spec.py)の契約をロックする。

- spec.render(iq, rate) の出力 shape == (IMG_FREQ, IMG_TIME) / dtype float32 /
  値域 [0,1]（ランダムIQ・無音IQ両方で）
- spec.spec_summary() を snapshots/spec_summary.json に固定し一致を検証
  （= 表現仕様が無断で変わったらここで落ちる）

制約: 契約ファイル(spec.py)のロジックは変更しない（テストするだけ）。
"""
import json
import os

import numpy as np
import pytest

import spec


SNAPSHOT = os.path.join(os.path.dirname(__file__), "snapshots", "spec_summary.json")


def _random_iq(n=8192, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)


def _silent_iq(n=8192):
    return np.zeros(n, dtype=np.complex64)


@pytest.mark.parametrize("factory", [_random_iq, _silent_iq], ids=["random", "silent"])
def test_render_shape_dtype_range(factory):
    """render は [IMG_FREQ, IMG_TIME] float32 in [0,1] を返す。"""
    img = spec.render(factory(), spec.CAPTURE_RATE_HZ)

    assert img.shape == (spec.IMG_FREQ, spec.IMG_TIME)
    assert img.dtype == np.float32
    assert np.all(np.isfinite(img))           # NaN/Inf 混入なし
    assert float(img.min()) >= 0.0
    assert float(img.max()) <= 1.0


def test_render_default_rate_matches_capture_rate():
    """rate 省略時は契約レート(CAPTURE_RATE_HZ)と同一結果。"""
    iq = _random_iq()
    assert np.array_equal(spec.render(iq), spec.render(iq, spec.CAPTURE_RATE_HZ))


def test_render_empty_iq_is_safe():
    """空IQでも落ちず、ゼロ画像 [IMG_FREQ, IMG_TIME] float32 を返す。"""
    img = spec.render(np.array([], dtype=np.complex64))
    assert img.shape == (spec.IMG_FREQ, spec.IMG_TIME)
    assert img.dtype == np.float32
    assert float(img.max()) == 0.0


def test_spec_summary_matches_snapshot():
    """spec_summary() を凍結スナップショットと突き合わせ（表現仕様の無断変更検知）。"""
    with open(SNAPSHOT, encoding="utf-8") as f:
        expected = json.load(f)

    actual = spec.spec_summary()

    assert actual == expected, (
        "spec_summary() がスナップショットと不一致。表現仕様(spec.py)を変えたなら、"
        "SigMF 生IQから再レンダの上で snapshots/spec_summary.json を更新すること。"
    )


def test_spec_summary_reflects_module_constants():
    """サマリ各値が spec.py の定数と一致（サマリ自身の自己整合）。"""
    s = spec.spec_summary()
    assert s["rate_hz"] == spec.CAPTURE_RATE_HZ
    assert s["nfft"] == spec.SPEC_NFFT
    assert s["hop"] == spec.SPEC_HOP
    assert s["img"] == [spec.IMG_FREQ, spec.IMG_TIME]
    assert s["dyn_range_db"] == spec.DB_DYN_RANGE
    assert s["version"] == spec.SIGSCAN_REP_VERSION
