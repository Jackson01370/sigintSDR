"""
test_view_captures.py — view_captures のラベル表示（一覧行・PNG タイトル文字列）。

書き込み側 sigmf_io.write_recording は用途ラベルを annotation の core:label に、
CNN 監査来歴(M3)を global の sigscan:cnn_verdict / cnn_class / cnn_conf に置く。
ここでは _meta_summary → label_text の純関数経路が「実キー名」で正しく拾うことと、
render_one の継ぎ目（シグネチャ。cnntrain.spotlight が再利用）が不変であることを
固定する。追加のみ（既存テストは変更しない）。
"""
import inspect
import os

import numpy as np
import pytest

import view_captures


def _meta(label=None, cnn_global=None):
    """実記録と同じキー配置の最小メタを作る（書き込み側 sigmf_io と同じ置き場所）。"""
    g = {"core:sample_rate": 20_000_000.0, "core:hw": "HackRF One"}
    if cnn_global:
        g.update(cnn_global)
    ann = {"core:freq_lower_edge": 2.419e9, "core:freq_upper_edge": 2.433e9}
    if label is not None:
        ann["core:label"] = label
    return {"global": g, "captures": [], "annotations": [ann]}


def test_label_text_with_cnn_verdict():
    """(i) 用途ラベル + CNN verdict あり → 両方併記（実機 M3 C 記録と同じ値）。"""
    meta = _meta(label="未識別信号", cnn_global={
        "sigscan:cnn_verdict": "C-conflict",
        "sigscan:cnn_class": "noise-only",
        "sigscan:cnn_conf": 0.469,
    })
    info = view_captures._meta_summary(meta)
    assert view_captures.label_text(info) == \
        "未識別信号  [CNN:C-conflict cnn=noise-only@0.47]"


def test_label_text_label_only_old_record():
    """(ii) ラベルのみ（CNN 来歴なしの古い記録）→ ラベルだけ・エラーにしない。"""
    info = view_captures._meta_summary(_meta(label="BLE/Bluetooth (adv?)"))
    text = view_captures.label_text(info)
    assert text == "BLE/Bluetooth (adv?)"
    assert "[CNN:" not in text


def test_label_text_no_label_fallback():
    """(iii) ラベル無し → 従来どおり "(no label)" に穏当フォールバック。"""
    info = view_captures._meta_summary(_meta())
    assert view_captures.label_text(info) == "(no label)"


def test_render_one_signature_unchanged():
    """render_one の継ぎ目: cnntrain.spotlight が (base, out_path) 位置引数で呼ぶ。"""
    assert list(inspect.signature(view_captures.render_one).parameters) == \
        ["base", "out_path", "flatten_dc"]


def test_render_one_writes_png_with_label(tmp_path):
    """write→read→PNG の実経路スモーク: ラベル+CNN来歴が要約に載り PNG が出る。"""
    pytest.importorskip("matplotlib")
    import sigmf_io
    rng = np.random.default_rng(0)
    iq = (rng.standard_normal(8192) + 1j * rng.standard_normal(8192)).astype(np.complex64)
    base = str(tmp_path / "rec")
    sigmf_io.write_recording(
        base, iq, center_hz=2.433e9, sample_rate=20e6,
        annotations=[dict(freq_lower_edge=2.426e9, freq_upper_edge=2.440e9,
                          label="未識別信号", confidence=0.53, method="cnn")],
        extra_global={"sigscan:cnn_verdict": "C-conflict",
                      "sigscan:cnn_class": "noise-only",
                      "sigscan:cnn_conf": 0.469})
    out = str(tmp_path / "rec.png")
    info = view_captures.render_one(base, out)
    assert os.path.exists(out) and os.path.getsize(out) > 0
    assert info["label"] == "未識別信号"
    assert info["cnn_verdict"] == "C-conflict"
    assert view_captures.label_text(info) == \
        "未識別信号  [CNN:C-conflict cnn=noise-only@0.47]"
