"""
test_classify.py — 分類継ぎ目(classify.py)の契約をロックする。

- 代表的な measurement で rule_based のラベルとバンド対応が期待どおり
  (2.437GHz/20MHz, 3.55GHz/100MHz, 2.402GHz/2MHz)
- cnn/llm 未実装時に classify がルール結果へ劣化すること

契約: classify(measurement, bands, spectrogram_db=None, png_path=None, ...) -> ClassResult。
measurement は dict {center_hz, bw_hz, snr_db}。
"""
import numpy as np
import pytest

import classify
from classify import rule_based, classify as do_classify, ClassResult
from config import BAND_PLAN


# (center_hz, bw_hz, 期待ラベル, 期待バンド名)
REPRESENTATIVE = [
    (2.437e9, 20e6, "WiFi (2.4GHz, 20/40MHz)", "ISM 2.4G (WiFi/BT)"),
    (3.55e9, 100e6, "5G NR (n77/n78 3.5G)", "5G NR n77/n78 3.5G"),
    (2.402e9, 2e6, "BLE/Bluetooth (adv?)", "ISM 2.4G (WiFi/BT)"),
]


def _m(center, bw, snr=30.0):
    return {"center_hz": center, "bw_hz": bw, "snr_db": snr}


@pytest.mark.parametrize("center,bw,label,band", REPRESENTATIVE)
def test_rule_based_label_and_band(center, bw, label, band):
    """周波数×帯域幅 → 信号DBラベル / バンド対応 が期待どおり。"""
    r = rule_based(_m(center, bw), BAND_PLAN)
    assert r.label == label
    assert r.method == "rule"
    assert 0.0 < r.confidence <= 0.85
    # バンド対応: center がそのバンドにマッチする
    matched = classify._match_band(center, BAND_PLAN)
    assert matched is not None and matched.name == band


def test_rule_based_outside_band_plan_is_unknown():
    """バンドプラン外(1GHz未満)は UNKNOWN・低信頼。"""
    r = rule_based(_m(900e6, 1e6, snr=10.0), BAND_PLAN)
    assert r.label == classify.UNKNOWN
    assert r.method == "rule"
    assert r.confidence < 0.5


@pytest.mark.parametrize("center,bw,label,band", REPRESENTATIVE)
def test_classify_degrades_to_rule_when_hooks_unimplemented(center, bw, label, band):
    """CNN/LLM 未実装時、classify は rule_based と同一結果へ劣化する。"""
    m = _m(center, bw)
    expected = rule_based(m, BAND_PLAN)
    actual = do_classify(m, BAND_PLAN)        # spectrogram_db=None, png_path=None

    assert actual.method == "rule"
    assert actual.label == expected.label == label
    assert actual.confidence == expected.confidence


def test_cnn_hook_returns_none_and_classify_still_degrades():
    """cnn_classify は未実装で None を返す → spectrogram を渡しても rule のまま。"""
    dummy_spec = np.zeros((256, 256), dtype=np.float32)
    assert classify.cnn_classify(dummy_spec) is None

    m = _m(2.437e9, 20e6)
    r = do_classify(m, BAND_PLAN, spectrogram_db=dummy_spec)
    assert r.method == "rule"
    assert r.label == "WiFi (2.4GHz, 20/40MHz)"


def test_classify_returns_classresult_type():
    """戻り値が ClassResult であること。"""
    r = do_classify(_m(2.437e9, 20e6), BAND_PLAN)
    assert isinstance(r, ClassResult)
