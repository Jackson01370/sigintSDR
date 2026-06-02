"""
test_eval.py — eval-harness（外部モデル配線）の M1 契約をロックする。

検証対象:
  * adapters が spec.render の正準画像 / 生 IQ を各 family の入力形に写すこと
    （サイズ・チャネル）。リサイズ/再正規化が adapters に吸収されていること。
  * reference stand-in がロードでき、1 枚の spec.render テンソルに推論が通り、
    is_stand_in=True を立てること（torch 不要で配線が通る）。
  * torch 未導入環境では実モデルローダが ModelUnavailable を明示送出すること
    （握りつぶさない）。torch がある環境ではこのチェックを skip。
  * run_report が sim 収集物に対して動き、synthetic-only / stand-in を明示する。

これは eval-harness 自身の配線テスト（凍結契約 spec.py/sigmf_io.py は触らない）。
"""
import numpy as np
import pytest

import sigmf_io
import spec
from eval import adapters, loaders
from eval.loaders import ModelUnavailable
from eval import report as eval_report


def _torch_present() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _make_iq(n=1 << 15, seed=3):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)


# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------
def test_canonical_goes_through_spec_render():
    """adapters.canonical は spec.render と同一の正準テンソルを返す。"""
    iq = _make_iq()
    a = adapters.canonical(iq, spec.CAPTURE_RATE_HZ)
    b = spec.render(iq, spec.CAPTURE_RATE_HZ)
    assert a.shape == (spec.IMG_FREQ, spec.IMG_TIME)
    assert a.dtype == np.float32
    assert np.array_equal(a, b)


def test_segmentation_adapter_shape_and_channels():
    """正準 1ch → セグメンテーション入力 [3,256,256]（ImageNet 正規化）。"""
    canon = spec.render(_make_iq(), spec.CAPTURE_RATE_HZ)
    x = adapters.to_segmentation_input(canon)
    assert x.shape == (3, 256, 256)
    assert x.dtype == np.float32
    # ImageNet 正規化で [0,1] を外れる（負値が出る）こと
    assert x.min() < 0.0


def test_segmentation_adapter_resizes():
    """異サイズ要求でも adapter がリサイズを吸収する。"""
    canon = spec.render(_make_iq(), spec.CAPTURE_RATE_HZ)
    x = adapters.to_segmentation_input(canon, size=(128, 200), normalize="none")
    assert x.shape == (3, 128, 200)
    assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0   # none なら [0,1]


def test_narrowband_adapter_shape():
    """生 IQ → 狭帯域入力 [2,4096]（実部/虚部, 単位エネルギー正規化）。"""
    x = adapters.to_narrowband_input(_make_iq(), n=4096)
    assert x.shape == (2, 4096)
    assert x.dtype == np.float32
    rms = float(np.sqrt(np.mean(x[0] ** 2 + x[1] ** 2)))
    assert 0.5 < rms < 2.0          # 単位エネルギー付近


def test_wideband_adapter_shape():
    canon = spec.render(_make_iq(), spec.CAPTURE_RATE_HZ)
    x = adapters.to_wideband_input(canon, size=(512, 512))
    assert x.shape == (2, 512, 512)


def test_adapt_for_dispatch_by_family():
    """adapt_for が ModelSpec.family で正しいアダプタを選ぶ。"""
    iq = _make_iq()
    seg_spec = loaders.load_reference_standin("qoherent-segmentation").spec
    nb_spec = loaders.load_reference_standin("torchsig-narrowband").spec
    assert adapters.adapt_for(seg_spec, iq=iq).shape == (3, 256, 256)
    assert adapters.adapt_for(nb_spec, iq=iq).shape == (2, 4096)


# ---------------------------------------------------------------------------
# reference stand-in: 1 枚のテンソルに推論が通る
# ---------------------------------------------------------------------------
def test_standin_inference_on_one_render_tensor():
    """spec.render テンソル1枚に stand-in 推論が通り、is_stand_in を立てる。"""
    canon = spec.render(_make_iq(), spec.CAPTURE_RATE_HZ)
    m = loaders.load_reference_standin("qoherent-segmentation")
    out = m.predict(adapters.to_segmentation_input(canon))
    assert m.is_stand_in is True
    assert out["is_stand_in"] is True
    assert out["pred_name"] in loaders.STANDIN_CLASSES
    assert out["pred_index"] in (0, 1, 2)


def test_standin_narrowband_path():
    m = loaders.load_reference_standin("torchsig-narrowband")
    out = m.predict(adapters.to_narrowband_input(_make_iq()))
    assert out["is_stand_in"] is True
    assert out["pred_name"] in loaders.STANDIN_CLASSES


# ---------------------------------------------------------------------------
# 実モデルローダ: 依存欠如を握りつぶさない
# ---------------------------------------------------------------------------
@pytest.mark.skipif(_torch_present(), reason="torch があるので欠如チェックは不可")
def test_real_loaders_raise_when_torch_missing():
    """torch 未導入なら実モデルローダは ModelUnavailable を明示送出する。"""
    for fn in (loaders.load_torchsig_narrowband,
               loaders.load_torchsig_wideband,
               loaders.load_qoherent_segmentation):
        with pytest.raises(ModelUnavailable):
            fn()


def test_load_model_standin_no_fallback_flag():
    """allow_standin=False で実モデル要求は ModelUnavailable（黙って退避しない）。"""
    if _torch_present():
        pytest.skip("torch があるとロード経路が変わる")
    with pytest.raises(ModelUnavailable):
        loaders.load_model("qoherent-segmentation", allow_standin=False)


def test_load_model_explicit_fallback():
    """allow_standin=True なら stand-in に明示退避し、名前に痕跡を残す。"""
    if _torch_present():
        pytest.skip("torch があると実体ロードされる")
    m = loaders.load_model("qoherent-segmentation", allow_standin=True)
    assert m.is_stand_in is True
    assert "fallback" in m.spec.name


# ---------------------------------------------------------------------------
# report: sim に対して動き、合成限定/stand-in を明示
# ---------------------------------------------------------------------------
def _write_sim_capture(base, center_hz, label):
    iq = _make_iq()
    ann = [dict(freq_lower_edge=center_hz - 5e6, freq_upper_edge=center_hz + 5e6,
                label=label, confidence=0.6, method="rule", snr_db=20.0)]
    sigmf_io.write_recording(base, iq, center_hz, spec.CAPTURE_RATE_HZ,
                             annotations=ann, hw="sigscan-sim (synthetic)")


def test_run_report_on_sim_dir(tmp_path):
    """run_report が sim ディレクトリで動き、対応表と stand-in 明示を返す。"""
    _write_sim_capture(str(tmp_path / "2140MHz_0"), 2.140e9, "LTE/UMTS DL")
    _write_sim_capture(str(tmp_path / "3550MHz_1"), 3.550e9, "5G NR n78")

    rep = eval_report.run_report(str(tmp_path), model_name="reference-standin")
    assert rep.n_total == 2
    assert rep.n_inferred == 2
    assert rep.is_stand_in is True
    assert rep.hw_counts.get("sim") == 2          # 出所が sim として集計される
    assert rep.crosstab                            # 対応表が空でない


def test_report_banner_states_synthetic_and_standin(tmp_path, capsys):
    """print_report が synthetic-only と stand-in のバナーを必ず出す。"""
    _write_sim_capture(str(tmp_path / "2140MHz_0"), 2.140e9, "LTE/UMTS DL")
    rep = eval_report.run_report(str(tmp_path), model_name="reference-standin")
    eval_report.print_report(rep)
    text = capsys.readouterr().out
    assert "SYNTHETIC-ONLY" in text
    assert "STAND-IN" in text
    assert "未測定" in text                         # ドメインギャップ未測定の明示
