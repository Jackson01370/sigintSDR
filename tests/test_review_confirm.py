"""
test_review_confirm.py — 確定フロー3点の機構をロックする（追加のみ・既存無変更）。

  (1) --include-human 選別: human 確定済みが既定で対象外／include_human=True で対象に入る。
  (2) 履歴: 再確定で sigscan:relabel_history に {from,to,at} が append され元ラベルが残る。
  (3) ○×UI 摩擦（最重要）: ble-adv/wifi は y 提示・確定できる。unclear / spurious_warn /
      needs-review(未記入) は y を出さずラベル選択強制（Pattern A 化＝素通り 防止）。
  (4) 後方互換: --suggest / --include-human 未指定は従来の対象選択・ディスパッチ。
  (5) cc_class → ラベル写像。
加えて review_suggest の confirm_sheet PNG 列 / --auto-classify タスクリストを固定。

確定は run_suggest_review 経由（apply_fn 差し替えで検証）。captures/ は触らない（tmp のみ）。
"""
import csv
import json
import os

import review
from cnntrain import review_suggest as rs


def _wm(dirpath, name, *, label="BLE/Bluetooth (adv?)", method="rule",
        confidence=0.2, lo=2.4795e9, hi=2.4807e9):
    ann = {
        "core:sample_start": 0, "core:sample_count": 262144,
        "core:label": label, "sigscan:method": method,
        "sigscan:confidence": confidence, "sigscan:snr_db": 25.0,
        "core:freq_lower_edge": lo, "core:freq_upper_edge": hi,
        "sigscan:persistence": 0.44,
    }
    meta = {
        "global": {"core:datatype": "cf32_le", "core:sample_rate": 20e6,
                   "core:hw": "HackRF One"},
        "captures": [{"core:sample_start": 0, "core:frequency": (lo + hi) / 2,
                      "core:datetime": "2026-07-12T00:00:00Z"}],
        "annotations": [ann],
    }
    p = os.path.join(dirpath, name + ".sigmf-meta")
    with open(p, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return p


def _ann(path):
    # review.py と同じロケール既定エンコーディングで読む（meta は utf-8 決め打ちしない）。
    with open(path) as f:
        return json.load(f)["annotations"][0]


# ---------------------------------------------------------------------------
# (5) cc_class → ラベル写像
# ---------------------------------------------------------------------------
def test_cc_class_to_label():
    assert review.cc_class_to_label("ble-adv") == "BLE/Bluetooth (adv?)"
    assert review.cc_class_to_label("wifi") == "WiFi (2.4GHz, 20/40MHz)"
    for none_case in ("hopping", "spurious", "unclear", "", None):
        assert review.cc_class_to_label(none_case) is None, none_case


# ---------------------------------------------------------------------------
# (1) --include-human 選別
# ---------------------------------------------------------------------------
def test_include_human_selection(tmp_path):
    d = str(tmp_path)
    _wm(d, "rule_lo", method="rule", confidence=0.2)
    _wm(d, "human_1", method="human", confidence=1.0)
    # 既定: rule のみ（human 除外＝従来挙動）
    items = review.find_low_confidence(d, conf_max=0.5)
    assert sorted(os.path.basename(i["meta_path"]) for i in items) == \
        ["rule_lo.sigmf-meta"]
    # include_human=True: human も conf に関わらず含む（訂正対象）
    items2 = review.find_low_confidence(d, conf_max=0.5, include_human=True)
    assert sorted(os.path.basename(i["meta_path"]) for i in items2) == \
        ["human_1.sigmf-meta", "rule_lo.sigmf-meta"]
    hi = [i for i in items2 if "human_1" in i["meta_path"]][0]
    assert hi["method"] == "human" and hi["confidence"] == 1.0


def test_include_human_default_excludes_human(tmp_path):
    """回帰: human は既定で対象外（rule<conf のみ拾う）。"""
    d = str(tmp_path)
    _wm(d, "human_only", method="human", confidence=1.0)
    assert review.find_low_confidence(d, conf_max=0.5) == []


# ---------------------------------------------------------------------------
# (2) 履歴（relabel_history）
# ---------------------------------------------------------------------------
def test_apply_label_records_history(tmp_path):
    d = str(tmp_path)
    p = _wm(d, "rec", label="BLE/Bluetooth (adv?)", method="human", confidence=1.0)
    review.apply_label(p, 0, "WiFi (2.4GHz, 20/40MHz)", record_history=True,
                       at="2026-07-12T00:00:00.000000Z")
    ann = _ann(p)
    assert ann["core:label"] == "WiFi (2.4GHz, 20/40MHz)"
    h = ann["sigscan:relabel_history"]
    assert len(h) == 1
    assert h[0]["from"] == "BLE/Bluetooth (adv?)"
    assert h[0]["to"] == "WiFi (2.4GHz, 20/40MHz)"
    assert h[0]["from_method"] == "human"
    assert h[0]["at"] == "2026-07-12T00:00:00.000000Z"
    # 2回目の訂正 → 履歴2件・元(初回)も残る（黙って上書きしない）
    review.apply_label(p, 0, "未識別信号", record_history=True,
                       at="2026-07-12T01:00:00.000000Z")
    h2 = _ann(p)["sigscan:relabel_history"]
    assert len(h2) == 2
    assert h2[0]["from"] == "BLE/Bluetooth (adv?)"     # 元ラベルは失われない
    assert h2[1]["from"] == "WiFi (2.4GHz, 20/40MHz)"


def test_apply_label_no_history_by_default(tmp_path):
    """record_history 省略時は履歴キーを足さない＝既存挙動不変（後方互換）。"""
    d = str(tmp_path)
    p = _wm(d, "rec", label="x", method="rule", confidence=0.2)
    review.apply_label(p, 0, "y")
    ann = _ann(p)
    assert "sigscan:relabel_history" not in ann
    assert ann["core:label"] == "y" and ann["sigscan:method"] == "human"
    assert "human-relabeled (was 'x' via rule)" in ann["core:comment"]


# ---------------------------------------------------------------------------
# (3) ○×UI の摩擦（安全弁）
# ---------------------------------------------------------------------------
_SUGGEST_FIELDS = ["record", "png", "det_freq_mhz", "bw_mhz", "snr_db",
                   "rule_label", "rule_confidence", "duty", "duty_inconclusive",
                   "spur_suspect", "spurious_warn", "cc_class", "cc_rationale",
                   "recommend", "note"]


def _make_suggest(tmp_path, rows):
    caps = tmp_path / "caps"; caps.mkdir(exist_ok=True)
    out = tmp_path / "out"; out.mkdir(exist_ok=True)
    sc = out / "suggestions.csv"
    with open(sc, "w", encoding="utf-8", newline="") as f:
        f.write("# banner\n")
        w = csv.DictWriter(f, fieldnames=_SUGGEST_FIELDS)
        w.writeheader()
        for r in rows:
            row = {k: "" for k in _SUGGEST_FIELDS}
            row.update({k: v for k, v in r.items() if k in _SUGGEST_FIELDS})
            w.writerow(row)
            _wm(str(caps), r["record"],
                label=r.get("_cur_label", "BLE/Bluetooth (adv?)"),
                method=r.get("_cur_method", "rule"),
                confidence=float(r.get("_cur_conf", 0.62)))
    return str(caps), str(sc)


def _run_suggest(tmp_path, rows, answers, include_human=False):
    caps, sc = _make_suggest(tmp_path, rows)
    prompts, printed, calls = [], [], []
    it = iter(answers)

    def input_fn(prompt):
        prompts.append(prompt)
        try:
            return next(it)
        except StopIteration:
            return "q"

    def apply_fn(meta_path, ann_index, new_label, **kw):
        calls.append(dict(record=os.path.basename(meta_path),
                          new_label=new_label, kw=kw))

    review.run_suggest_review(caps, sc, input_fn=input_fn,
                              print_fn=printed.append,
                              include_human=include_human, apply_fn=apply_fn)
    return prompts, "\n".join(printed), calls


def test_suggest_ui_ble_offers_y_and_confirms(tmp_path):
    rows = [dict(record="a", cc_class="ble-adv", recommend="confirm-ble",
                 spurious_warn="False", det_freq_mhz="2480.00", bw_mhz="1.20",
                 png="captures/_images/a.png")]
    prompts, out, calls = _run_suggest(tmp_path, rows, ["y"])
    assert any("y=確定" in p for p in prompts)              # y が提示される
    assert len(calls) == 1
    assert calls[0]["new_label"] == "BLE/Bluetooth (adv?)"   # 提案ラベルで確定
    assert calls[0]["kw"].get("record_history") is True      # 履歴を残す
    assert "captures/_images/a.png" in out                   # PNG を必ず表示


def test_suggest_ui_wifi_offers_y(tmp_path):
    """wifi も写像ありなので y 可（_11 の能動再ラベル用途）。"""
    rows = [dict(record="w", cc_class="wifi", recommend="skip",
                 spurious_warn="False")]
    prompts, out, calls = _run_suggest(tmp_path, rows, ["y"])
    assert any("y=確定" in p for p in prompts)
    assert calls[0]["new_label"] == "WiFi (2.4GHz, 20/40MHz)"


def test_suggest_ui_unclear_blocks_y(tmp_path):
    rows = [dict(record="u", cc_class="unclear", recommend="skip",
                 spurious_warn="False")]
    prompts, out, calls = _run_suggest(tmp_path, rows, ["s"])
    assert not any("y=確定" in p for p in prompts)          # y を出さない
    assert "unclear" in out and "提案確定(y)は不可" in out
    assert calls == []                                       # 確定していない


def test_suggest_ui_spurious_blocks_y(tmp_path):
    rows = [dict(record="s1", cc_class="ble-adv", recommend="skip",
                 spurious_warn="True")]
    prompts, out, calls = _run_suggest(tmp_path, rows, ["s"])
    assert not any("y=確定" in p for p in prompts)
    assert "スプリアス警告" in out
    assert calls == []


def test_suggest_ui_needs_review_blocks_y(tmp_path):
    rows = [dict(record="nr", cc_class="", recommend="needs-review",
                 spurious_warn="False")]
    prompts, out, calls = _run_suggest(tmp_path, rows, ["s"])
    assert not any("y=確定" in p for p in prompts)
    assert "未記入" in out
    assert calls == []


def test_suggest_ui_n_shows_label_list(tmp_path):
    """y 可のケースでも n を押せばラベル一覧が出て人間が選ぶ（従来の選択）。"""
    rows = [dict(record="a", cc_class="ble-adv", recommend="confirm-ble",
                 spurious_warn="False")]
    prompts, out, calls = _run_suggest(tmp_path, rows, ["n", "0"])
    assert any("y=確定" in p for p in prompts)              # 最初に y/n は出た
    assert "候補ラベル" in out
    assert len(calls) == 1                                   # 人間選択で確定
    assert calls[0]["kw"].get("record_history") is True


def test_suggest_ui_confirmed_human_gated_by_include(tmp_path):
    """既定は確定済み(human)を提示せずスキップ／include_human=True で訂正対象として提示。"""
    rows = [dict(record="hh", cc_class="ble-adv", recommend="confirm-ble",
                 spurious_warn="False", _cur_method="human", _cur_conf="1.0",
                 _cur_label="BLE/Bluetooth (adv?)")]
    _, out, calls = _run_suggest(tmp_path, rows, [], include_human=False)
    assert "確定済み(human)" in out and calls == []
    prompts2, out2, calls2 = _run_suggest(tmp_path, rows, ["y"], include_human=True)
    assert any("y=確定" in p for p in prompts2)
    assert calls2[0]["new_label"] == "BLE/Bluetooth (adv?)"
    assert "訂正対象" in out2


# ---------------------------------------------------------------------------
# (4) 後方互換: main dispatch
# ---------------------------------------------------------------------------
def test_main_dispatch_backcompat(tmp_path, monkeypatch):
    seen = {}

    def fake_cmd_list(dirpath, conf_max, verdict=None, pattern=None,
                      include_human=False):
        seen.clear(); seen.update(mode="list", include_human=include_human)
        return 0

    def fake_run_review(dirpath, conf_max=0.5, input_fn=input, print_fn=print,
                        verdict=None, pattern=None, include_human=False):
        seen.clear(); seen.update(mode="review", include_human=include_human)
        return 0

    def fake_suggest(dirpath, suggest_csv, input_fn=input, print_fn=print,
                     include_human=False, apply_fn=None):
        seen.clear()
        seen.update(mode="suggest", csv=suggest_csv, include_human=include_human)
        return 0

    monkeypatch.setattr(review, "_cmd_list", fake_cmd_list)
    monkeypatch.setattr(review, "run_review", fake_run_review)
    monkeypatch.setattr(review, "run_suggest_review", fake_suggest)

    review.main([str(tmp_path)])                              # 従来: run_review
    assert seen == {"mode": "review", "include_human": False}
    review.main([str(tmp_path), "--list"])                    # 従来: list
    assert seen == {"mode": "list", "include_human": False}
    review.main([str(tmp_path), "--include-human"])           # 訂正経路
    assert seen["mode"] == "review" and seen["include_human"] is True
    review.main([str(tmp_path), "--suggest", "x.csv"])        # ○×UI
    assert seen["mode"] == "suggest" and seen["csv"] == "x.csv"
    review.main([str(tmp_path), "--suggest", "x.csv", "--list"])  # --list 優先(読み取り)
    assert seen["mode"] == "list"
    review.main([str(tmp_path), "--list", "--include-human"])     # 訂正経路の列挙
    assert seen["mode"] == "list" and seen["include_human"] is True


# ---------------------------------------------------------------------------
# review_suggest: confirm_sheet PNG 列 / --auto-classify タスクリスト
# ---------------------------------------------------------------------------
def _suggest_record(**kw):
    base = dict(record="a", png="captures/_images/a.png", det_freq_mhz=2480.0,
                bw_mhz=1.2, snr_db=25.0, rule_label="BLE", rule_confidence=0.62,
                duty=0.03, duty_inconclusive=True, spur_suspect=False,
                spurious_warn=False)
    base.update(kw)
    return rs.SuggestRecord(**base)


def test_confirm_sheet_has_png_column():
    r = _suggest_record(cc_class="ble-adv", cc_rationale="離散バースト",
                        recommend="confirm-ble")
    sheet = rs.format_confirm_sheet([r], "captures/", "*")
    assert "| PNG |" in sheet                              # ヘッダに PNG 列
    assert "captures/_images/a.png" in sheet               # 行に PNG パス


def test_auto_classify_tasklist(tmp_path):
    r = _suggest_record()
    path = rs.write_classify_tasklist(str(tmp_path), [r], "captures/", "*")
    txt = open(path, encoding="utf-8").read()
    assert "captures/_images/a.png" in txt                 # PNG パス
    assert "record,cc_class,cc_rationale" in txt           # cc_verdicts テンプレ
    assert "a,," in txt
