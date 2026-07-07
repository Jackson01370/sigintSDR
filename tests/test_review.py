"""
test_review.py — review.py の (C)=C-conflict 選別と提示の追加分を固定する。

対象は「CNN 監査で (C) になった記録」を review.py が --verdict C で選別・提示できること。
確定ロジック（apply_label）とメタ書き戻しは一切変更していないので、ここでは
  (i)  sigscan:cnn_verdict=='C-conflict' を拾う／'A-consistent'・verdict無しを除外する
  (ii) 既定挙動（method='rule' かつ confidence<conf_max）が無変更であること（回帰）
  (iii) 選別・提示で captures/ を書き換えない／apply_label を呼ばないこと
を検証する。追加のみ（既存テストは変更しない）。

meta の読み書きは review.py と同様ロケール既定エンコーディングで行う（UTF-8 決め打ち
しない）。テスト用 meta も同じ open() で書き round-trip 互換を担保する。
"""
import json
import os

import review


def _write_meta(path, *, label, center, cnn=None, method="rule", confidence=0.2,
                comment=None, lo=None, hi=None):
    """実記録と同じキー配置の最小 meta を書く（cnn は global の CNN 来歴 dict）。"""
    g = {
        "core:datatype": "cf32_le",
        "core:sample_rate": 20_000_000.0,
        "core:hw": "HackRF One",
    }
    if cnn:
        g.update(cnn)
    ann = {
        "core:sample_start": 0,
        "core:sample_count": 262144,
        "core:label": label,
        "sigscan:method": method,
        "sigscan:confidence": confidence,
        "sigscan:snr_db": 30.0,
    }
    if lo is not None:
        ann["core:freq_lower_edge"] = lo
    if hi is not None:
        ann["core:freq_upper_edge"] = hi
    if comment is not None:
        ann["core:comment"] = comment
    meta = {
        "global": g,
        "captures": [{"core:sample_start": 0, "core:frequency": center,
                      "core:datetime": "2026-07-07T00:00:00Z"}],
        "annotations": [ann],
    }
    # review.py と同じロケール既定エンコーディングで書く（読みと round-trip）。
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def _c_global(cnn_class="cw-tone", cnn_conf=0.61, verdict="C-conflict"):
    return {
        "sigscan:cnn_class": cnn_class,
        "sigscan:cnn_conf": cnn_conf,
        "sigscan:cnn_verdict": verdict,
        "sigscan:cnn_checkpoint": "checkpoint.pt",
    }


# --- (i) 純関数: is_c_conflict --------------------------------------------

def test_is_c_conflict_true():
    meta = {"global": {"sigscan:cnn_verdict": "C-conflict"}}
    assert review.is_c_conflict(meta) is True


def test_is_c_conflict_rejects_other_verdicts():
    for v in ("A-consistent", "unmapped", None, ""):
        meta = {"global": {"sigscan:cnn_verdict": v}} if v is not None else {"global": {}}
        assert review.is_c_conflict(meta) is False, v


def test_is_c_conflict_missing_global_safe():
    # global ごと欠けても例外にせず False。
    assert review.is_c_conflict({}) is False
    assert review.is_c_conflict({"global": None}) is False


# --- find_c_conflict: 実データ相当のディレクトリで選別 ----------------------

def test_find_c_conflict_selects_only_c(tmp_path):
    d = str(tmp_path)
    _write_meta(os.path.join(d, "c.sigmf-meta"), label="未識別信号",
                center=2.489e9, method="cnn", confidence=0.39,
                cnn=_c_global(cnn_class="cw-tone", cnn_conf=0.61),
                comment="ISM 2.4G: 狭帯域 [CNN監査:C-conflict cw-tone@0.61] → 用途=Unknown(候補つき)",
                lo=2.488e9, hi=2.490e9)
    _write_meta(os.path.join(d, "a.sigmf-meta"), label="Wi-Fi/BT",
                center=2.434e9, method="cnn", confidence=0.65,
                cnn=_c_global(cnn_class="wideband-ofdm", cnn_conf=0.94,
                              verdict="A-consistent"))
    _write_meta(os.path.join(d, "r.sigmf-meta"), label="something",
                center=2.402e9, method="rule", confidence=0.62)  # verdict なし

    items = review.find_c_conflict(d)
    assert len(items) == 1
    it = items[0]
    assert os.path.basename(it["meta_path"]) == "c.sigmf-meta"
    # CNN 来歴が global から取れている。
    assert it["cnn_verdict"] == "C-conflict"
    assert it["cnn_class"] == "cw-tone"
    assert it["cnn_conf"] == 0.61
    assert it["method"] == "cnn"
    # 日本語ラベル・comment がロケール既定エンコーディングで正しく往復している。
    assert it["label"] == "未識別信号"
    assert "用途=Unknown(候補つき)" in it["comment"]
    assert it["ann_index"] == 0


def test_find_c_conflict_empty_when_none(tmp_path):
    d = str(tmp_path)
    _write_meta(os.path.join(d, "r.sigmf-meta"), label="x", center=2.4e9,
                method="rule", confidence=0.2)
    assert review.find_c_conflict(d) == []


# --- (ii) 既定挙動（rule & conf<conf_max）が無変更であることの回帰 ----------

def test_find_low_confidence_unchanged(tmp_path):
    d = str(tmp_path)
    # rule & conf<0.5 → 拾う
    _write_meta(os.path.join(d, "lo.sigmf-meta"), label="lo", center=2.4e9,
                method="rule", confidence=0.2, lo=2.399e9, hi=2.401e9)
    # rule だが conf>=0.5 → 除外
    _write_meta(os.path.join(d, "hi.sigmf-meta"), label="hi", center=2.41e9,
                method="rule", confidence=0.62)
    # (C) 記録は method='cnn' なので既定選別からは除外される
    _write_meta(os.path.join(d, "c.sigmf-meta"), label="未識別信号", center=2.489e9,
                method="cnn", confidence=0.39, cnn=_c_global())

    items = review.find_low_confidence(d, conf_max=0.5)
    assert len(items) == 1
    assert os.path.basename(items[0]["meta_path"]) == "lo.sigmf-meta"
    assert items[0]["confidence"] == 0.2


# --- 提示ヘルパ ------------------------------------------------------------

def test_candidate_hint():
    c = "ISM 2.4G (WiFi/BT): 狭帯域・ホッピング [CNN監査:C-conflict cw-tone@0.61] → 用途=Unknown(候補つき)"
    assert review._candidate_hint(c) == "用途=Unknown(候補つき)"
    assert review._candidate_hint(None) is None
    assert review._candidate_hint("用途キーなしのコメント") is None


def test_extra_lines_backcompat_for_rule_item():
    # 既定 rule 経路の item（cnn_verdict なし）は追加行なし＝従来表示のまま。
    rule_item = {"meta_path": "x.sigmf-meta"}
    assert review._extra_lines(rule_item) == ""


def test_extra_lines_for_c_item():
    c_item = {
        "meta_path": "x.sigmf-meta",
        "cnn_class": "cw-tone", "cnn_conf": 0.61, "cnn_verdict": "C-conflict",
        "comment": "... → 用途=Unknown(候補つき)",
    }
    out = review._extra_lines(c_item)
    assert "cnn=cw-tone@0.61 verdict=C-conflict" in out
    assert "用途=Unknown(候補つき)" in out


# --- (iii) 選別・提示で captures を書き換えない／apply_label を呼ばない ------

def _snapshot(d):
    """dir 内の *.sigmf-meta の生バイトを控える（書き換え検出用）。"""
    snap = {}
    for name in sorted(os.listdir(d)):
        if name.endswith(".sigmf-meta"):
            with open(os.path.join(d, name), "rb") as f:
                snap[name] = f.read()
    return snap


def test_cmd_list_c_does_not_write(tmp_path, capsys):
    d = str(tmp_path)
    _write_meta(os.path.join(d, "c.sigmf-meta"), label="未識別信号", center=2.489e9,
                method="cnn", confidence=0.39, cnn=_c_global(),
                comment="... → 用途=Unknown(候補つき)", lo=2.488e9, hi=2.490e9)
    _write_meta(os.path.join(d, "r.sigmf-meta"), label="x", center=2.4e9,
                method="rule", confidence=0.2)
    before = _snapshot(d)

    rc = review._cmd_list(d, conf_max=0.5, verdict="C")
    assert rc == 0
    out = capsys.readouterr().out
    assert "c.sigmf-meta" in out
    assert "C-conflict" in out
    assert "r.sigmf-meta" not in out          # 既定 rule 記録は C 列挙に出ない
    assert _snapshot(d) == before             # 書き換えていない


def test_run_review_c_skip_does_not_apply(tmp_path, monkeypatch):
    d = str(tmp_path)
    _write_meta(os.path.join(d, "c.sigmf-meta"), label="未識別信号", center=2.489e9,
                method="cnn", confidence=0.39, cnn=_c_global(),
                comment="... → 用途=Unknown(候補つき)", lo=2.488e9, hi=2.490e9)
    before = _snapshot(d)

    # apply_label が呼ばれたら即失敗（選別・提示・スキップでは絶対に呼ばれない）。
    def _boom(*a, **k):
        raise AssertionError("apply_label must not be called during C review skip")
    monkeypatch.setattr(review, "apply_label", _boom)

    printed = []
    # 全アイテムをスキップ（'s'）してから終了。
    rc = review.run_review(d, verdict="C", input_fn=lambda _p: "s",
                           print_fn=printed.append)
    assert rc == 0
    joined = "\n".join(printed)
    assert "C-conflict" in joined            # (C) レビューのヘッダが出ている
    assert _snapshot(d) == before            # 書き換えていない
