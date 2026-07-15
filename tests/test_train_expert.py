"""test_train_expert.py — 2.4GHz ISM 専門家 CNN 学習（実データのみ・3クラス）を固定する。

指示書_専門家CNN学習.md のテスト要件:
  1. ラベル写像: 3ラベルが正しく写像・対象外(未識別信号)除外・method!=human 除外。
  2. 層化分割: 少数・不均衡でも val に全クラスが最低1件入る（seed 固定で決定的）。
  3. クラス重み: 逆頻度重みが期待通り。
  4. 既存不変: 専門家経路は別入口（hw="real" 既定）で sim 経路を触らない。
  5. スモーク学習: 小さな実データで1 epoch 完走し checkpoint(classes 付き)保存→load_checkpoint 可。

captures/ は触らない（tmp のみ）。runs/m2_5 も触らない（out は tmp）。
"""
import numpy as np
import pytest

import classify
import sigmf_io
from cnntrain import train_expert as te
from cnntrain import infer


def _write_real(dirpath, name, label, *, method="human", hw="HackRF One",
                center=2.44e9, n=4096, seed=0):
    """実データ相当の小さな SigMF を書く（hw=HackRF One・任意 method）。"""
    rng = np.random.default_rng(seed)
    iq = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    base = str(dirpath / name)
    sigmf_io.write_recording(
        base, iq, center, 20e6,
        annotations=[dict(freq_lower_edge=center - 6e5, freq_upper_edge=center + 6e5,
                          label=label, method=method, confidence=1.0)],
        hw=hw)
    return base


# 1. ラベル写像
def test_map_label():
    assert te.map_label("BLE/Bluetooth (adv?)") == "ble-adv"
    assert te.map_label("WiFi (2.4GHz, 20/40MHz)") == "wifi-24"
    assert te.map_label(classify.SPURIOUS) == "spurious"          # 単一の真実
    assert te.map_label("未識別信号") is None                      # 対象外は除外
    assert te.map_label("") is None and te.map_label(None) is None
    assert te.EXPERT_CLASSES == ["ble-adv", "wifi-24", "spurious"]


# 1b. 収集: human かつ3ラベルのみ（rule/cnn・対象外・sim を除外）
def test_collect_only_human_3class(tmp_path):
    _write_real(tmp_path, "a_ble", "BLE/Bluetooth (adv?)", method="human")
    _write_real(tmp_path, "b_wifi", "WiFi (2.4GHz, 20/40MHz)", method="human")
    _write_real(tmp_path, "c_spur", classify.SPURIOUS, method="human")
    _write_real(tmp_path, "d_rule", "BLE/Bluetooth (adv?)", method="rule")   # 除外(method)
    _write_real(tmp_path, "e_unk", "未識別信号", method="human")              # 除外(対象外ラベル)
    _write_real(tmp_path, "f_sim", "BLE/Bluetooth (adv?)", method="human",
                hw="sigscan-sim (synthetic)")                                # 除外(合成)
    items = te.collect_expert_records(str(tmp_path), hw="real")
    got = sorted(cls for _, cls in items)
    assert got == ["ble-adv", "spurious", "wifi-24"]     # 3件だけ（human・実データ・3ラベル）


# 2. 層化分割: val に全クラスが最低1件
def test_stratified_split_covers_all_classes():
    # 不均衡（ble 10 / wifi 3 / spurious 6）を模した (record擬似, cls)。
    class _R:
        def __init__(self, p): self.path = p
    items = ([(_R(f"ble{i}"), "ble-adv") for i in range(10)]
             + [(_R(f"wf{i}"), "wifi-24") for i in range(3)]
             + [(_R(f"sp{i}"), "spurious") for i in range(6)])
    train, val = te.stratified_split(items, val_ratio=0.2, seed=0)
    val_classes = {cls for _, cls in val}
    assert val_classes == {"ble-adv", "wifi-24", "spurious"}   # 全クラスが val に居る
    # 各クラス train/val とも最低1件（最小クラス wifi=3 でも壊れない）。
    for cls in ("ble-adv", "wifi-24", "spurious"):
        assert sum(1 for _, c in train if c == cls) >= 1
        assert sum(1 for _, c in val if c == cls) >= 1
    assert len(train) + len(val) == len(items)                 # 取りこぼしなし
    # 決定的: 同 seed で同じ分割。
    train2, val2 = te.stratified_split(items, val_ratio=0.2, seed=0)
    assert [r.path for r, _ in val] == [r.path for r, _ in val2]


# 3. クラス重み（逆頻度・平均≈1）
def test_inverse_freq_weights():
    class _R:
        def __init__(self, p): self.path = p
    items = ([(_R(f"a{i}"), "ble-adv") for i in range(36)]
             + [(_R(f"b{i}"), "wifi-24") for i in range(23)]
             + [(_R(f"c{i}"), "spurious") for i in range(32)])
    w = te.inverse_freq_weights(items, te.EXPERT_CLASSES)
    n, k = 91, 3
    assert w[0] == pytest.approx(n / (k * 36))    # ble
    assert w[1] == pytest.approx(n / (k * 23))    # wifi（最小＝最大重み）
    assert w[2] == pytest.approx(n / (k * 32))    # spurious
    assert w[1] > w[2] > w[0]                      # 少数クラスほど重い
    # 0件クラスは重み0（ゼロ割回避）。
    assert te.inverse_freq_weights([], ["x"]) == [0.0]


# 4. 既存不変: 専門家収集は既定 hw="real"（sim 経路に触れない）
def test_expert_defaults_to_real(tmp_path):
    _write_real(tmp_path, "sim_ble", "BLE/Bluetooth (adv?)", method="human",
                hw="sigscan-sim (synthetic)")
    # 既定 real なので sim は拾わない＝合成非混合（案A）を索引段で保証。
    assert te.collect_expert_records(str(tmp_path)) == []


# 5. スモーク学習: 1 epoch 完走・checkpoint(classes 付き)保存・load 可
def test_expert_training_smoke(tmp_path):
    data = tmp_path / "caps"; data.mkdir()
    for i in range(4):
        _write_real(data, f"ble{i}", "BLE/Bluetooth (adv?)", method="human", seed=i)
        _write_real(data, f"wf{i}", "WiFi (2.4GHz, 20/40MHz)", method="human", seed=100 + i)
        _write_real(data, f"sp{i}", classify.SPURIOUS, method="human", seed=200 + i)
    out = tmp_path / "runs" / "ism24_test"
    res = te.run_expert_training(str(data), str(out), epochs=1, batch_size=4,
                                 seed=0, val_ratio=0.25, log=lambda *a, **k: None)
    assert res["classes"] == ["ble-adv", "wifi-24", "spurious"]
    ckpt_path = out / "checkpoint.pt"
    assert ckpt_path.exists()
    # load_checkpoint（weights_only）で読めて classes が復元される。
    ck = infer.load_checkpoint(str(ckpt_path))
    assert ck.classes == ["ble-adv", "wifi-24", "spurious"]
    assert ck.meta.get("real_data") is True and ck.meta.get("synthetic_only") is False
    # 学習成果物が出ている。
    assert (out / "report.txt").exists() and (out / "history.json").exists()
