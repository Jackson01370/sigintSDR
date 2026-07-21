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
import torch
import torch.nn as nn

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


# ==========================================================================
# v2 追加: best-val 保存 / early-stopping / k-fold（指示書_専門家CNN再学習.md）
# ==========================================================================

class _Tiny(nn.Module):
    """state_dict に単一パラメータ w を持つだけの極小モデル（重み追跡用）。"""
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x


def _fake_epoch_factory(val_schedule):
    """`_epoch_pass` の差し替え: train パスで w←epoch 番号、val パスで schedule を返す。

    これで「best epoch の重み」を確実に識別できる（w の値＝そのときの epoch）。
    """
    state = {"epoch": 0}

    def fake(model, loader, criterion, optimizer=None, device=None):
        if optimizer is not None:                 # train パス（epoch 開始）
            state["epoch"] += 1
            with torch.no_grad():
                for p in model.parameters():
                    p.fill_(float(state["epoch"]))
            return 0.1, 1.0
        return 0.2, val_schedule[state["epoch"] - 1]   # val パス
    return fake


# v2-1. best-val 保存: 最終 epoch でなく最良 epoch の重みが保存される。
def test_train_loop_saves_best_not_last(monkeypatch):
    model = _Tiny()
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    val = [0.5, 0.9, 0.6, 0.55]                    # 最良=epoch2, 最終=epoch4
    monkeypatch.setattr(te, "_epoch_pass", _fake_epoch_factory(val))
    history, best = te._train_loop(model, None, None, None, opt,
                                   torch.device("cpu"), epochs=4)
    assert best["epoch"] == 2 and best["val_acc"] == pytest.approx(0.9)
    assert best["stopped_epoch"] == 4              # early-stop なしで全 epoch 実行
    assert len(history) == 4
    # best["state"] は epoch2 の重み(=2.0)。現モデルは epoch4(=4.0)＝ best≠last。
    assert float(best["state"]["w"]) == pytest.approx(2.0)
    assert float(next(iter(model.parameters()))) == pytest.approx(4.0)


# v2-2. early-stopping: patience 超過で停止し、保存重みは停止 epoch でなく best。
def test_train_loop_early_stops_and_keeps_best(monkeypatch):
    model = _Tiny()
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    val = [0.9, 0.8, 0.8, 0.8, 0.8, 0.8]           # epoch1 最良、その後停滞
    monkeypatch.setattr(te, "_epoch_pass", _fake_epoch_factory(val))
    history, best = te._train_loop(model, None, None, None, opt,
                                   torch.device("cpu"), epochs=6, patience=2)
    # epoch1 best → epoch2 未改善(1) → epoch3 未改善(2>=patience) → epoch3 停止。
    assert best["stopped_epoch"] == 3 and len(history) == 3
    assert best["epoch"] == 1
    # 保存重みは停止 epoch(3) ではなく best(1) の重み(=1.0)。
    assert float(best["state"]["w"]) == pytest.approx(1.0)


# v2-3. stratified k-fold: 各 fold が全クラスを train/val に含み、val が全体を重複なく覆う。
def test_stratified_kfold_covers_and_partitions():
    class _R:
        def __init__(self, p): self.path = p
    items = ([(_R(f"ble{i}"), "ble-adv") for i in range(9)]
             + [(_R(f"wf{i}"), "wifi-24") for i in range(6)]
             + [(_R(f"sp{i}"), "spurious") for i in range(6)])
    k = 3
    splits = te.stratified_kfold(items, k=k, seed=0)
    assert len(splits) == k
    all_val = []
    for train, val in splits:
        assert {c for _, c in val} == {"ble-adv", "wifi-24", "spurious"}
        assert {c for _, c in train} == {"ble-adv", "wifi-24", "spurious"}
        assert len(train) + len(val) == len(items)          # 各 fold は全体を覆う
        all_val += [r.path for r, _ in val]
    assert sorted(all_val) == sorted(r.path for r, _ in items)   # val は全体を覆う
    assert len(all_val) == len(set(all_val))                     # val は重複なし
    # 決定的（同 seed で同分割）。
    s2 = te.stratified_kfold(items, k=k, seed=0)
    assert ([[r.path for r, _ in v] for _, v in splits]
            == [[r.path for r, _ in v] for _, v in s2])
    with pytest.raises(ValueError):                              # k<2 はエラー
        te.stratified_kfold(items, k=1)


# v2-4. クラス重み: 増量件数(60/93/39)での逆頻度重みが期待通り。
def test_inverse_freq_weights_increased_counts():
    class _R:
        def __init__(self, p): self.path = p
    items = ([(_R(f"a{i}"), "ble-adv") for i in range(60)]
             + [(_R(f"b{i}"), "wifi-24") for i in range(93)]
             + [(_R(f"c{i}"), "spurious") for i in range(39)])
    w = te.inverse_freq_weights(items, te.EXPERT_CLASSES)
    n, k = 192, 3
    assert w[0] == pytest.approx(n / (k * 60))    # ble
    assert w[1] == pytest.approx(n / (k * 93))    # wifi（最多＝最軽）
    assert w[2] == pytest.approx(n / (k * 39))    # spurious（最少＝最重）
    assert w[2] > w[0] > w[1]                      # 少数クラスほど重い


# v2-5. スモーク: run_expert_v2 が k-fold 評価＋最終 best-val checkpoint を作る。
def test_expert_v2_smoke_kfold_and_final(tmp_path):
    data = tmp_path / "caps"; data.mkdir()
    for i in range(6):
        _write_real(data, f"ble{i}", "BLE/Bluetooth (adv?)", method="human", seed=i)
        _write_real(data, f"wf{i}", "WiFi (2.4GHz, 20/40MHz)", method="human", seed=100 + i)
        _write_real(data, f"sp{i}", classify.SPURIOUS, method="human", seed=200 + i)
    out = tmp_path / "runs" / "ism24_v2_test"
    res = te.run_expert_v2(str(data), str(out), k=2, epochs=1, batch_size=4,
                           seed=0, val_ratio=0.25, patience=None,
                           log=lambda *a, **k: None)
    # k-fold サマリ（平均±分散・合算混同行列）。
    kf = res["kfold"]
    assert kf is not None and len(kf["fold_val_acc"]) == 2
    assert "mean_val_acc" in kf and "std_val_acc" in kf
    assert len(kf["confusion"]) == 3 and len(kf["confusion"][0]) == 3
    assert (out / "kfold_report.json").exists() and (out / "kfold_report.txt").exists()
    # 最終 checkpoint: best-val 保存・k-fold サマリ入り meta。
    ckpt = out / "checkpoint.pt"
    assert ckpt.exists()
    ck = infer.load_checkpoint(str(ckpt))
    assert ck.classes == ["ble-adv", "wifi-24", "spurious"]
    assert ck.meta.get("saved") == "best-val"
    assert ck.meta.get("best_epoch") is not None
    assert ck.meta.get("kfold") is not None            # meta に k-fold サマリ併記
    assert ck.meta.get("real_data") is True and ck.meta.get("synthetic_only") is False
