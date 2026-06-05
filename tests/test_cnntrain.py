"""
test_cnntrain.py — CNN 学習パイプライン（M1 火入れ）の経路をロックする。

検証対象（作業指示 T1〜T4）:
  * T1: simgen が指定クラス・件数の SigMF を作り、真実ラベルが記録される。
  * T2: 読み込み（SigMF → 凍結 spec.render → テンソル）の形状・値域[0,1]・
        ラベル対応が正しい。
  * T3: 極小データの 1 エポック・スモーク学習が完走し、チェックポイントが
        保存・再読込できる（rep_version / SYNTHETIC-ONLY メタ / seed つき）。
  * T4: 推論ヘルパが保存済みチェックポイントで (クラス, 確信度∈[0,1]) を返す。

凍結契約（spec.py / sigmf_io.py）は触らない。既存テストにも手を入れない（追加のみ）。
全体で 1〜2 分以内に収まる極小規模。
"""
import os

import numpy as np
import pytest

import sigmf_io
import spec
from cnntrain import classes, simgen


def _torch_present() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


# torch を要するテスト（T2〜T4）は torch 無し環境では skip（eval テストと同じ流儀）。
requires_torch = pytest.mark.skipif(not _torch_present(),
                                    reason="torch 未導入（CPU 学習はスキップ）")

PER_CLASS = 6
GEN_SEED = 7


@pytest.fixture(scope="module")
def simdata(tmp_path_factory):
    """極小の合成データセットを 1 度だけ生成して使い回す（高速化）。"""
    d = tmp_path_factory.mktemp("simdata")
    simgen.generate(str(d), per_class=PER_CLASS, seed=GEN_SEED)
    return str(d)


# ---------------------------------------------------------------------------
# T1: simgen が真実ラベル付き SigMF を作る
# ---------------------------------------------------------------------------
def test_t1_simgen_creates_balanced_sigmf_with_truth(simdata):
    import glob
    metas = sorted(glob.glob(os.path.join(simdata, "*.sigmf-meta")))
    datas = sorted(glob.glob(os.path.join(simdata, "*.sigmf-data")))
    # 5 クラス × PER_CLASS、data/meta が対で揃う。
    assert len(metas) == len(classes.CLASSES) * PER_CLASS
    assert len(datas) == len(metas)

    seen: dict[str, int] = {}
    for mp in metas:
        base = mp[: -len(".sigmf-meta")]
        iq, meta = sigmf_io.read_recording(base)
        g = meta["global"]
        # 出所の正直な表記（合成）。
        assert g["core:hw"] == "sigscan-sim (synthetic)"
        # 真実ラベルは global の sigscan:true_class に冗長記録される。
        true_cls = g.get("sigscan:true_class")
        assert true_cls in classes.CLASSES
        # ファイル名のクラス接頭辞とも一致。
        assert os.path.basename(base).startswith(true_cls)
        # annotation の core:label も真実ラベル（method=sim-truth＝ルール由来でない）。
        ann = meta["annotations"][0]
        assert ann["core:label"] == true_cls
        assert ann["sigscan:method"] == "sim-truth"
        assert ann["sigscan:confidence"] == 1.0
        # rep_version / SYNTHETIC-ONLY タグ / 生成シードが載る。
        assert g["sigscan:rep_version"] == spec.SIGSCAN_REP_VERSION
        assert "synthetic" in g["sigscan:synthetic_only"].lower()
        assert int(g["sigscan:gen_seed"]) == GEN_SEED
        # IQ は complex64 で十分な長さ（spec.render に通る）。
        assert iq.dtype == np.complex64 and iq.size >= spec.SPEC_NFFT
        seen[true_cls] = seen.get(true_cls, 0) + 1

    # クラス均衡。
    assert set(seen) == set(classes.CLASSES)
    assert all(v == PER_CLASS for v in seen.values())


def test_t1_simgen_is_deterministic(tmp_path):
    """同じシードは同じ IQ を生む（再現可能）。"""
    a = tmp_path / "a"
    b = tmp_path / "b"
    simgen.generate(str(a), per_class=2, seed=123)
    simgen.generate(str(b), per_class=2, seed=123)
    iq_a, _ = sigmf_io.read_recording(str(a / "cw-tone_0000"))
    iq_b, _ = sigmf_io.read_recording(str(b / "cw-tone_0000"))
    assert np.array_equal(iq_a, iq_b)


# ---------------------------------------------------------------------------
# T2: SigMF → spec.render → テンソル（形状・値域・ラベル対応）
# ---------------------------------------------------------------------------
@requires_torch
def test_t2_loading_shape_range_and_labels(simdata):
    import torch
    from cnntrain import data

    train_recs, val_recs, class_names = data.load_split(
        simdata, val_ratio=0.25, seed=GEN_SEED)
    # クラス名は正準 5 クラス（dataset.split の core:label ソート順）。
    assert class_names == list(classes.CLASSES)
    # 分割は全件を覆い、hw 別（全 sim）で 80/20 目安。
    assert len(train_recs) + len(val_recs) == len(classes.CLASSES) * PER_CLASS
    assert len(val_recs) > 0 and len(train_recs) > 0

    ds = data.SpecDataset(train_recs, class_names)
    assert len(ds) == len(train_recs)
    c2i = ds.class_to_idx
    for i in range(len(ds)):
        x, y = ds[i]
        # 形状 [1,256,256]・float32。
        assert tuple(x.shape) == (1, spec.IMG_FREQ, spec.IMG_TIME)
        assert x.dtype == torch.float32
        # 値域 [0,1]（spec.render の正準域。迂回・再正規化なし）。
        assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0
        # ラベル対応: テンソルの y は record.label のインデックスと一致。
        assert y == c2i[ds.records[i].label]


@requires_torch
def test_t2_render_matches_spec_render(simdata):
    """SpecDataset の画像は凍結 spec.render の出力そのもの（迂回していない）。"""
    import torch
    from cnntrain import data

    train_recs, _, class_names = data.load_split(simdata, val_ratio=0.25,
                                                 seed=GEN_SEED)
    ds = data.SpecDataset(train_recs, class_names)
    r = ds.records[0]
    iq, meta = sigmf_io.read_recording(r.path)
    rate = float(meta["global"]["core:sample_rate"])
    expected = spec.render(iq, rate)
    x, _ = ds[0]
    assert np.array_equal(x.squeeze(0).numpy(), expected)


# ---------------------------------------------------------------------------
# T3: 1 エポック・スモーク学習 → チェックポイント保存/再読込
# ---------------------------------------------------------------------------
@requires_torch
def test_t3_smoke_train_and_checkpoint(simdata, tmp_path):
    from cnntrain import train, infer

    out = tmp_path / "run"
    res = train.run_training(simdata, str(out), epochs=1, batch_size=8,
                             seed=GEN_SEED, val_ratio=0.25, log=lambda s: None)

    # 成果物が揃う。
    assert os.path.exists(res["ckpt_path"])
    assert os.path.exists(os.path.join(str(out), "train_log.txt"))
    assert os.path.exists(res["report_txt"])
    assert os.path.exists(res["report_json"])

    # レポート冒頭に SYNTHETIC-ONLY バナー。
    with open(res["report_txt"], encoding="utf-8") as f:
        txt = f.read()
    assert "SYNTHETIC-ONLY" in txt
    assert "混同行列" in txt

    # チェックポイント再読込 → クラス・メタが復元。
    ck = infer.load_checkpoint(res["ckpt_path"])
    assert ck.classes == list(classes.CLASSES)
    assert ck.meta["synthetic_only"] is True
    assert ck.meta["rep_version"] == spec.SIGSCAN_REP_VERSION
    assert int(ck.meta["train_seed"]) == GEN_SEED
    assert int(ck.meta["gen_seed"]) == GEN_SEED  # データ側 meta から検出
    # val_accuracy は [0,1]（火入れなので値は問わない）。
    assert 0.0 <= res["val_accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# T4: 推論ヘルパ（クラス, 確信度∈[0,1]）
# ---------------------------------------------------------------------------
@requires_torch
def test_t4_inference_helper(simdata, tmp_path):
    from cnntrain import train, infer

    out = tmp_path / "run4"
    res = train.run_training(simdata, str(out), epochs=1, batch_size=8,
                             seed=GEN_SEED, val_ratio=0.25, log=lambda s: None)
    ck = infer.load_checkpoint(res["ckpt_path"])

    # 正準画像（spec.render の出力）を 1 枚作って推論。
    iq, meta = sigmf_io.read_recording(
        os.path.join(simdata, "cw-tone_0000"))
    img = spec.render(iq, float(meta["global"]["core:sample_rate"]))

    cls, conf = infer.classify_image(ck, img)
    assert cls in classes.CLASSES
    assert 0.0 <= conf <= 1.0

    # top-k も確信度降順で返る。
    topk = infer.classify_image_topk(ck, img, k=3)
    assert len(topk) == 3
    assert all(c in classes.CLASSES for c, _ in topk)
    confs = [p for _, p in topk]
    assert confs == sorted(confs, reverse=True)
    assert topk[0][0] == cls

    # 生 IQ 経由の便宜ラッパも同じ結果（spec.render を必ず通す）。
    cls2, conf2 = infer.classify_iq(ck, iq,
                                    float(meta["global"]["core:sample_rate"]))
    assert cls2 == cls and abs(conf2 - conf) < 1e-5

    # 形状不正は弾く（迂回しないことの担保）。
    with pytest.raises(ValueError):
        infer.classify_image(ck, np.zeros((128, 128), dtype=np.float32))
