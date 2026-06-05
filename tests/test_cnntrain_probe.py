"""
test_cnntrain_probe.py — 実データ推論プローブ（M1.5）の **機構** をロックする。

実 captures は使わない（CI で持てない）。simgen の合成データを実データの代役にし、
プローブの機構を検証する:
  * 期待対応表 match_expected（用途軸ラベル → 方式クラス集合 / unmapped）。
  * クロス表集計・ラベル別統計（期待一致率・noise-only 率・平均確信度）。
  * 要画像確認リスト（不一致 × 確信度降順 top_n）。
  * run_probe の end-to-end（read→spec.render→推論→集計→レポート生成、バナー）。

凍結契約（spec.py / sigmf_io.py）は触らない。既存テストは無改変（追加のみ）。
プローブは **推論のみ**（このテストでも学習・fine-tune はしない）。数十秒以内。
"""
import os

import pytest


def _torch_present() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


HAS_TORCH = _torch_present()
# probe は infer 経由で torch を必要とする。torch 無し環境ではモジュールごと skip。
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch 未導入（プローブはスキップ）")

if HAS_TORCH:
    from cnntrain import probe, simgen, train


# ---------------------------------------------------------------------------
# 期待対応表（純関数・モデル不要）
# ---------------------------------------------------------------------------
def test_match_expected_real_labels():
    """captures/ の実ラベル 3 種が期待集合に対応し、未知ラベルは unmapped。"""
    exp, note = probe.match_expected("BLE/Bluetooth (adv?)")
    assert exp == frozenset({"narrowband-burst"}) and note
    exp, _ = probe.match_expected("WiFi (2.4GHz, 20/40MHz)")
    assert exp == frozenset({"wideband-ofdm"})
    exp, _ = probe.match_expected("Zigbee/独自2.4G")
    assert exp == frozenset({"narrowband-burst", "wideband-ofdm"})
    # 対応表にないラベルは (None, '') = unmapped。
    exp, note = probe.match_expected("Totally Unknown Mode")
    assert exp is None and note == ""
    assert probe.match_expected(None) == (None, "")


def test_aggregate_crosstab_and_per_label():
    """ハンドメイドの ProbeRecord でクロス表・統計・unmapped を検証。"""
    R = probe.ProbeRecord
    recs = [
        # BLE: 1 件期待一致(narrowband-burst)、2 件 noise-only(=くじ引き)
        R("ble_0", "BLE", 2402.0, "narrowband-burst", 0.9, 0.6, 20.0,
          ["narrowband-burst"], True, False),
        R("ble_1", "BLE", 2402.0, "noise-only", 0.7, 0.6, 5.0,
          ["narrowband-burst"], False, True),
        R("ble_2", "BLE", 2402.0, "noise-only", 0.8, 0.6, 5.0,
          ["narrowband-burst"], False, True),
        # WiFi: 1 件一致
        R("wifi_0", "WiFi", 2437.0, "wideband-ofdm", 0.95, 0.7, 35.0,
          ["wideband-ofdm"], True, False),
        # 未対応ラベル
        R("x_0", "MysteryLabel", 5000.0, "cw-tone", 0.5, None, None,
          None, None, False),
    ]
    classes = ["cw-tone", "narrowband-burst", "noise-only", "pulse-radar",
               "wideband-ofdm"]
    crosstab, labels, per_label = probe._aggregate(recs, classes)

    assert labels == ["BLE", "MysteryLabel", "WiFi"]
    assert crosstab["BLE"] == {"narrowband-burst": 1, "noise-only": 2}
    assert crosstab["WiFi"] == {"wideband-ofdm": 1}

    ble = per_label["BLE"]
    assert ble["n"] == 3
    assert ble["expected_match"] == 1
    assert abs(ble["expected_match_rate"] - 1 / 3) < 1e-9
    assert ble["noise_only"] == 2
    assert abs(ble["noise_only_rate"] - 2 / 3) < 1e-9
    assert ble["unmapped"] is False

    wifi = per_label["WiFi"]
    assert wifi["expected_match_rate"] == 1.0
    assert wifi["noise_only_rate"] == 0.0

    myst = per_label["MysteryLabel"]
    assert myst["unmapped"] is True
    assert myst["expected_match_rate"] is None     # 一致率は未定義（accuracy ではない）


def test_review_list_orders_by_confidence_desc():
    """要画像確認: 不一致(matched=False)のみ・確信度降順・top_n で切る。"""
    R = probe.ProbeRecord
    recs = [
        R("a", "BLE", 2402.0, "wideband-ofdm", 0.6, None, None,
          ["narrowband-burst"], False, False),
        R("b", "BLE", 2402.0, "wideband-ofdm", 0.99, None, None,
          ["narrowband-burst"], False, False),
        R("ok", "BLE", 2402.0, "narrowband-burst", 0.95, None, None,
          ["narrowband-burst"], True, False),     # 一致は出さない
        R("u", "X", 1000.0, "cw-tone", 1.0, None, None,
          None, None, False),                     # unmapped は不一致ではない
    ]
    top = probe._review_list(recs, top_n=10)
    assert [r.file for r in top] == ["b", "a"]    # 確信度降順、一致/unmapped 除外
    assert probe._review_list(recs, top_n=1)[0].file == "b"


# ---------------------------------------------------------------------------
# end-to-end 機構（合成を実データ代役に）
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sim_and_ckpt(tmp_path_factory):
    """極小の合成データ + 1epoch チェックポイントを 1 度だけ用意（使い回し）。"""
    d = tmp_path_factory.mktemp("probe_sim")
    data_dir = str(d / "simdata")
    run_dir = str(d / "run")
    simgen.generate(data_dir, per_class=4, seed=11)
    res = train.run_training(data_dir, run_dir, epochs=1, batch_size=8,
                             seed=11, val_ratio=0.25, log=lambda s: None)
    return data_dir, res["ckpt_path"]


def test_run_probe_end_to_end_mechanism(sim_and_ckpt):
    """run_probe が合成データ(実データ代役)で集計・unmapped・レポート生成まで通る。"""
    data_dir, ckpt = sim_and_ckpt

    # 合成ラベル(方式軸)を用途軸ラベルに見立てた [仮説] 表。
    # narrowband-burst を **わざと未対応** にして unmapped 経路を検証する。
    table = [
        probe.ExpectedRow(("cw-tone",), frozenset({"cw-tone"}), "test"),
        probe.ExpectedRow(("wideband-ofdm",), frozenset({"wideband-ofdm"}), "test"),
        probe.ExpectedRow(("noise-only",), frozenset({"noise-only"}), "test"),
        probe.ExpectedRow(("pulse-radar",), frozenset({"pulse-radar"}), "test"),
    ]
    res = probe.run_probe(data_dir, ckpt, top_n=5, table=table)

    # 全 20 件(5クラス×4)を推論。
    assert res.n_total == 20
    assert res.hw_counts.get("sim") == 20      # 合成は hw_group=sim
    # クロス表の行は合成 5 ラベル。
    assert set(res.labels) == {"cw-tone", "narrowband-burst", "noise-only",
                               "pulse-radar", "wideband-ofdm"}
    # narrowband-burst は unmapped（表に入れていない）→ 一致率 None。
    nb = res.per_label["narrowband-burst"]
    assert nb["unmapped"] is True and nb["expected_match_rate"] is None
    # マップ済みラベルは一致率が数値で出る（accuracy ではないが率として定義される）。
    assert res.per_label["cw-tone"]["expected_match_rate"] is not None
    # クロス表の総数は全件。
    assert sum(sum(row.values()) for row in res.crosstab.values()) == 20
    # 要画像確認リストは全て不一致(matched=False)。
    assert all(r.matched is False for r in res.review_list)

    # レポート生成（バナー必須要素）。
    out = os.path.join(os.path.dirname(ckpt), "probe_out")
    txt_path, json_path = probe.write_report(out, res)
    assert os.path.exists(txt_path) and os.path.exists(json_path)
    with open(txt_path, encoding="utf-8") as f:
        txt = f.read()
    assert "REAL-DATA PROBE" in txt
    assert "[仮説]" in txt
    assert "accuracy" in txt          # 「accuracy ではない」明記
    assert "noise-only" in txt

    import json
    with open(json_path, encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["match_is_not_accuracy"] is True
    assert payload["learning"] == "none (inference only)"
    assert len(payload["records"]) == 20
