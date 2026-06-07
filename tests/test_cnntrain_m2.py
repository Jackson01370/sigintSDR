"""
test_cnntrain_m2.py — M2 の simgen 実測アライン + スポットライトをロックする。

T5: DC 残留注入が中心(0Hz)に出る・強度が較正レンジ内・全クラスに無相関で入る。
T6: cw-tone のオフセットが |off| > 0.5MHz を守る（注入DC線と分離）。
T7: wideband-ofdm の非周期性 vs pulse-radar の周期性（生成統計の分離）。
T8: スポットライト機構（合成代役で タイプ1/2 抽出・top-N 上限・レポート生成・
    元データ無変更）。

凍結契約（spec.py / sigmf_io.py）は触らない。既存テストは無改変（追加のみ）。
実データ（captures/）は使わない。数十秒以内。
"""
import os

import numpy as np
import pytest

import sigmf_io
import spec
import dsp
from cnntrain import simgen, classes


def _torch_present() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


HAS_TORCH = _torch_present()
requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch 未導入")


# ---------------------------------------------------------------------------
# T5: DC 残留注入（中心・較正レンジ・クラス無相関）
# ---------------------------------------------------------------------------
def test_t5_dc_injection_center_and_calibrated_range():
    """注入後の IQ を spec.render すると中心列が持ち上がり、dc_excess が実測レンジ内。"""
    rate = simgen.RATE
    rng = np.random.default_rng(5)
    excesses = []
    center_gains = []
    n_inj = 0
    for _ in range(120):
        iq, _ = simgen._gen_noise_only(rng)          # 信号なしで DC 効果を分離
        iq2, injected, amp = simgen._inject_dc_residual(iq, rate, rng)
        if not injected:
            continue
        n_inj += 1
        # dc_excess_db が実測の妥当域（[0.5, 11]dB ＝実測 min..max に余裕）に入る。
        exc = dsp.dc_spike_metrics(iq2, rate)["dc_excess_db"]
        excesses.append(exc)
        # spec.render の中心周波数行が全体平均より明るい（中心に線が出る）。
        img = spec.render(iq2, rate)
        center_gains.append(float(img[spec.IMG_FREQ // 2].mean()) - float(img.mean()))
    assert n_inj > 0
    excesses = np.array(excesses)
    # 較正レンジ（実測 dc_excess は min=-1.25..max=10.77dB。余裕を持たせた範囲）に収まる。
    assert excesses.min() >= -2.0 and excesses.max() <= 12.0
    # 中央値は実測（中央値~1.4, 10-90pct[1.0,4.25]）の妥当域。
    assert 0.8 <= np.median(excesses) <= 5.0
    # 中心行が平均より明るい（DC 線の存在）。
    assert np.median(center_gains) > 0.0


def test_t5_dc_injection_is_class_uncorrelated():
    """DC 注入率がクラスに依らずほぼ一定（識別の手掛かりにならない）。"""
    import glob
    import tempfile
    d = tempfile.mkdtemp(prefix="m2dc_")
    simgen.generate(d, per_class=30, seed=5)
    rates = {}
    for cls in classes.CLASSES:
        metas = glob.glob(os.path.join(d, f"{cls}_*.sigmf-meta"))
        flags = []
        for mp in metas:
            _, meta = sigmf_io.read_recording(mp[:-len(".sigmf-meta")])
            flags.append(bool(meta["global"].get("sigscan:dc_injected")))
        rates[cls] = sum(flags) / len(flags)
    # 全クラスで注入が起きており、率が設定値(0.8)の周辺に揃う（クラス無相関）。
    assert all(0.5 <= r <= 0.98 for r in rates.values()), rates
    assert (max(rates.values()) - min(rates.values())) <= 0.35, rates


# ---------------------------------------------------------------------------
# T6: cw-tone のオフセット制約
# ---------------------------------------------------------------------------
def test_t6_cw_tone_offset_excludes_center():
    """cw-tone の信号は中心から |off| > 0.5MHz に置かれる。"""
    rng = np.random.default_rng(6)
    for _ in range(200):
        _, info = simgen._gen_cw_tone(rng)
        assert abs(info["off"]) > 0.5e6


# ---------------------------------------------------------------------------
# T7: wideband-ofdm 非周期 vs pulse-radar 周期
# ---------------------------------------------------------------------------
def _interval_cv(starts) -> float:
    if len(starts) < 2:
        return 0.0
    d = np.diff(starts)
    return float(np.std(d) / (np.mean(d) + 1e-9))


def test_t7_wideband_aperiodic_vs_pulse_periodic():
    """pulse-radar は周期的(間隔CV≈0)、wideband-ofdm は非周期(間隔CV高)。"""
    rng = np.random.default_rng(7)
    wb_cv, pl_cv = [], []
    for _ in range(60):
        _, s_wb, _ = simgen._irregular_burst_envelope(simgen.N, rng)
        _, s_pl, _ = simgen._periodic_pulse_envelope(simgen.N, rng)
        wb_cv.append(_interval_cv(s_wb))
        pl_cv.append(_interval_cv(s_pl))
    # pulse は厳密周期 → CV はほぼ 0。
    assert max(pl_cv) < 0.05
    # wideband は非周期 → 平均 CV が明確に大きい。
    assert np.mean(wb_cv) > 0.15
    assert np.mean(wb_cv) > np.mean(pl_cv) + 0.1


# ---------------------------------------------------------------------------
# T8: スポットライト機構
# ---------------------------------------------------------------------------
@requires_torch
def test_t8_spotlight_selection_and_topn():
    """select_spotlight が タイプ1(自白)/タイプ2(監査) を正しく分け、top-N で切る。"""
    from cnntrain import spotlight, probe
    R = probe.ProbeRecord
    recs = [
        # タイプ1: rule conf 低い / ラベル空
        R("low1", "WiFi", 2437.0, "wideband-ofdm", 0.7, 0.30, 20.0,
          ["wideband-ofdm"], True, False),
        R("empty", "", 5000.0, "cw-tone", 0.8, None, None, None, None, False),
        # タイプ2: 不一致 かつ CNN 高確信
        R("audit1", "WiFi", 2462.0, "pulse-radar", 0.97, 0.62, 35.0,
          ["wideband-ofdm"], False, False),
        R("audit2", "BLE", 2402.0, "cw-tone", 0.92, 0.62, 28.0,
          ["narrowband-burst"], False, False),
        # 不一致だが CNN 低確信 → タイプ2 に入らない
        R("lowconf", "BLE", 2402.0, "cw-tone", 0.55, 0.62, 10.0,
          ["narrowband-burst"], False, False),
        # 一致 → どちらにも入らない（rule conf も十分）
        R("ok", "WiFi", 2437.0, "wideband-ofdm", 0.99, 0.80, 35.0,
          ["wideband-ofdm"], True, False),
    ]
    sel = spotlight.select_spotlight(recs, type1_conf_thr=0.5,
                                     type2_cnn_thr=0.9, top_n=15)
    t1 = {r.file for r in sel["type1"]}
    t2 = {r.file for r in sel["type2"]}
    assert t1 == {"low1", "empty"}
    assert t2 == {"audit1", "audit2"}
    # タイプ2 は確信度降順。
    assert [r.file for r in sel["type2"]] == ["audit1", "audit2"]
    # top-N 上限。
    sel2 = spotlight.select_spotlight(recs, type1_conf_thr=0.5,
                                      type2_cnn_thr=0.9, top_n=1)
    assert len(sel2["type1"]) == 1 and len(sel2["type2"]) == 1


@pytest.fixture(scope="module")
def sim_and_ckpt_m2(tmp_path_factory):
    from cnntrain import train
    d = tmp_path_factory.mktemp("m2_sim")
    data_dir = str(d / "simdata")
    run_dir = str(d / "run")
    simgen.generate(data_dir, per_class=4, seed=13)
    res = train.run_training(data_dir, run_dir, epochs=1, batch_size=8,
                             seed=13, val_ratio=0.25, log=lambda s: None)
    return data_dir, res["ckpt_path"]


@requires_torch
def test_t8_spotlight_end_to_end_and_data_unchanged(sim_and_ckpt_m2, tmp_path):
    """run_spotlight が end-to-end で動き、レポートが出て、元データが不変。"""
    from cnntrain import spotlight
    data_dir, ckpt = sim_and_ckpt_m2

    # 実行前のデータ署名（パス・サイズ・mtime）を採取。
    import glob
    before = {}
    for f in sorted(glob.glob(os.path.join(data_dir, "*.sigmf-*"))):
        st = os.stat(f)
        before[f] = (st.st_size, st.st_mtime_ns)

    out = str(tmp_path / "_spotlight")
    # 合成ラベル(方式軸)で タイプ2 を出しやすい [仮説] 表（narrowband を未対応に）。
    table = [
        spotlight.probe.ExpectedRow(("cw-tone",), frozenset({"cw-tone"}), "t"),
        spotlight.probe.ExpectedRow(("wideband-ofdm",),
                                    frozenset({"wideband-ofdm"}), "t"),
        spotlight.probe.ExpectedRow(("noise-only",), frozenset({"noise-only"}), "t"),
        spotlight.probe.ExpectedRow(("pulse-radar",), frozenset({"pulse-radar"}), "t"),
    ]
    pres = spotlight.probe.run_probe(data_dir, ckpt, top_n=20, table=table)
    summary = spotlight.run_spotlight(
        data_dir, ckpt, out, type1_conf_thr=0.5, type2_cnn_thr=0.0,  # 全不一致を拾う
        top_n=3, render=True, probe_result=pres)

    # レポートが生成され、理由が載る。
    assert os.path.exists(summary["report_txt"])
    assert os.path.exists(summary["report_json"])
    with open(summary["report_txt"], encoding="utf-8") as f:
        txt = f.read()
    assert "発見の漏斗" in txt
    assert "自動隔離" in txt
    # top-N=3 の上限が効く。
    assert summary["n_type2"] <= 3 and summary["n_type1"] <= 3
    # PNG が出力先に生成される（matplotlib 前提・本環境では有効）。
    pngs = glob.glob(os.path.join(out, "*.png"))
    assert len(pngs) >= 1

    # 元データ（*.sigmf-*）が一切変更されていない。
    after = {}
    for f in sorted(glob.glob(os.path.join(data_dir, "*.sigmf-*"))):
        st = os.stat(f)
        after[f] = (st.st_size, st.st_mtime_ns)
    assert before == after
