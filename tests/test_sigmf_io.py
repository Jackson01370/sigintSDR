"""
test_sigmf_io.py — SigMF 交換形式(sigmf_io.py)の契約をロックする。

- write_recording -> read_recording の往復で IQ(complex64) が一致
- meta に core:datatype=='cf32_le' / core:sample_rate / core:hw が保持される
- annotation_from_result が freq_lower/upper_edge と confidence/method/snr_db を持ち、
  write_recording を通すと core:freq_*_edge / sigscan:confidence/method/snr_db になる

制約: 契約ファイル(sigmf_io.py)のロジックは変更しない（テストするだけ）。
"""
import numpy as np

import sigmf_io
from classify import ClassResult


def _make_iq(n=4096, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)


def test_write_read_roundtrip_iq(tmp_path):
    """生IQ(complex64) が往復で完全一致すること。"""
    iq = _make_iq()
    base = str(tmp_path / "cap0")

    meta = sigmf_io.write_recording(base, iq, center_hz=2.437e9, sample_rate=20e6,
                                    hw="unit-test")
    assert isinstance(meta, dict)

    iq_back, meta_back = sigmf_io.read_recording(base)
    assert iq_back.dtype == np.complex64
    assert iq_back.shape == iq.shape
    assert np.array_equal(iq_back, iq)


def test_meta_global_block_contract(tmp_path):
    """global ブロックの契約フィールドが保持されること。"""
    iq = _make_iq()
    base = str(tmp_path / "cap1")
    sigmf_io.write_recording(base, iq, 2.437e9, 20e6, hw="HackRF One")

    _, meta = sigmf_io.read_recording(base)
    g = meta["global"]

    assert g["core:datatype"] == "cf32_le"            # complex64 LE 固定
    assert g["core:sample_rate"] == 20e6
    assert isinstance(g["core:sample_rate"], float)
    assert g["core:hw"] == "HackRF One"               # 出所の正直な記録
    assert g["core:version"] == sigmf_io.SIGMF_VERSION


def test_extra_global_is_preserved(tmp_path):
    """独自 global 拡張(sigscan:rep_version 等)が保持される。"""
    iq = _make_iq(n=1024)
    base = str(tmp_path / "cap2")
    sigmf_io.write_recording(
        base, iq, 3.55e9, 20e6, hw="x",
        extra_global={"sigscan:rep_version": "1.0", "sigscan:target_src": "self"})
    _, meta = sigmf_io.read_recording(base)
    g = meta["global"]
    assert g["sigscan:rep_version"] == "1.0"
    assert g["sigscan:target_src"] == "self"


def test_captures_block_contract(tmp_path):
    """captures[0] に中心周波数・sample_start が入る。"""
    iq = _make_iq(n=2048)
    base = str(tmp_path / "cap3")
    sigmf_io.write_recording(base, iq, 3.55e9, 20e6, hw="x")
    _, meta = sigmf_io.read_recording(base)
    cap = meta["captures"][0]
    assert cap["core:frequency"] == 3.55e9
    assert cap["core:sample_start"] == 0
    assert "core:datetime" in cap


def test_annotation_from_result_fields():
    """annotation_from_result が周波数エッジ(絶対Hz)と分類根拠フィールドを持つ。"""
    center, bw = 2.437e9, 20e6
    result = ClassResult("WiFi (2.4GHz, 20/40MHz)", 0.78, "rule", notes="OFDM")
    measurement = {"center_hz": center, "bw_hz": bw, "snr_db": 40.0}

    ann = sigmf_io.annotation_from_result(measurement, result)

    # 周波数エッジ（絶対Hz）
    assert ann["freq_lower_edge"] == center - bw / 2.0
    assert ann["freq_upper_edge"] == center + bw / 2.0
    assert ann["freq_lower_edge"] < ann["freq_upper_edge"]
    # ラベリング根拠
    assert ann["confidence"] == 0.78
    assert ann["method"] == "rule"
    assert ann["snr_db"] == 40.0
    assert ann["label"] == "WiFi (2.4GHz, 20/40MHz)"


def test_annotation_becomes_sigmf_namespaced_after_write(tmp_path):
    """write を通すと core:freq_*_edge / sigscan:confidence/method/snr_db になる。"""
    iq = _make_iq(n=1024)
    base = str(tmp_path / "cap4")
    result = ClassResult("WiFi", 0.78, "rule", notes="n")
    ann = sigmf_io.annotation_from_result(
        {"center_hz": 2.437e9, "bw_hz": 20e6, "snr_db": 40.0}, result)
    sigmf_io.write_recording(base, iq, 2.437e9, 20e6, annotations=[ann], hw="x")

    _, meta = sigmf_io.read_recording(base)
    assert len(meta["annotations"]) == 1
    a = meta["annotations"][0]
    for key in ("core:freq_lower_edge", "core:freq_upper_edge",
                "sigscan:confidence", "sigscan:method", "sigscan:snr_db"):
        assert key in a
    assert a["core:freq_lower_edge"] == 2.437e9 - 20e6 / 2.0
    assert a["sigscan:method"] == "rule"
