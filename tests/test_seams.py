"""
test_seams.py — 6つの安定した継ぎ目(seam)のシグネチャを固定する。

CONTRACT.md §3「安定した継ぎ目」を inspect.signature でロックする。
引数名(順序込み)が無断で変わったらここで落ちる。

タスク必須: spec.render / classify.classify / sigmf_io.write_recording。
6継ぎ目を網羅する。
"""
import inspect

import classify
import dsp
import sigmf_io
import spec
from sdr import SDRBackend
from store import Store


def _params(func):
    return list(inspect.signature(func).parameters)


# (callable, 期待する引数名の並び) — 実装の現状を凍結する
SEAM_SIGNATURES = [
    # 表現
    (spec.render, ["iq", "rate"]),
    # 測定
    (dsp.detect_segments,
     ["freqs_hz", "power_db", "threshold_db", "min_bw_hz", "smooth_hz", "merge_gap_hz"]),
    (dsp.measure_signal, ["iq", "rate", "center_hz"]),
    # 分類
    (classify.classify,
     ["measurement", "bands", "spectrogram_db", "png_path", "cnn_threshold"]),
    (classify.rule_based, ["measurement", "bands"]),
    # 交換
    (sigmf_io.write_recording,
     ["path_base", "iq", "center_hz", "sample_rate", "annotations",
      "hw", "recorder", "description", "extra_global"]),
    (sigmf_io.read_recording, ["path_base"]),
    (sigmf_io.annotation_from_result, ["measurement", "result"]),
    # 取得（Sim/HackRF 抽象基底）
    (SDRBackend.sweep_power, ["self", "start_hz", "stop_hz", "bin_hz"]),
    (SDRBackend.capture_iq, ["self", "center_hz", "rate", "n"]),
]


def test_seam_signatures_frozen():
    """各継ぎ目関数の引数名（順序込み）が契約どおり固定であること。"""
    mismatches = []
    for func, expected in SEAM_SIGNATURES:
        actual = _params(func)
        if actual != expected:
            name = getattr(func, "__qualname__", func.__name__)
            mismatches.append(f"{func.__module__}.{name}: expected {expected}, got {actual}")
    assert not mismatches, "seam signature changed:\n" + "\n".join(mismatches)


def test_required_three_seams_explicit():
    """タスク必須の3継ぎ目を明示的にロック（意図の固定）。"""
    assert _params(spec.render) == ["iq", "rate"]
    assert _params(classify.classify) == [
        "measurement", "bands", "spectrogram_db", "png_path", "cnn_threshold"]
    assert _params(sigmf_io.write_recording)[:4] == [
        "path_base", "iq", "center_hz", "sample_rate"]


def test_class_result_fields_frozen():
    """分類継ぎ目の戻り値 ClassResult のフィールドを固定。"""
    fields = list(classify.ClassResult.__dataclass_fields__)
    assert fields == ["label", "confidence", "method", "notes", "candidates"]


def test_store_seam_methods():
    """蓄積継ぎ目 store.Store が log / recent / close を提供する。"""
    for m in ("log", "recent", "close"):
        assert callable(getattr(Store, m, None))
