"""test_contact_sheet.py — 確定レビューのコンタクトシート（表示補助のみ）。

ロックする契約（追加のみ・既存無変更）:
  C1 キャプション: `[番号] cc_class`（提案なしは (提案なし)）。純関数で組み立て検証。
  C2 番号一致: sheet_entries の index が提示順（batch→rest）＝○×UI の一括候補/個別確認
     の番号と一致（マッピングの単体検証）。
  C3 生成: entries → contact_sheet.png が生成される（既存 PNG を並べるだけ）。
  C4 PNG 欠損耐性: 一部 PNG が無くてもプレースホルダで完走する。
  C5 オプション性・非侵襲: --open-sheet 無しなら生成・オープンが呼ばれない。
     ヘッドレス（interactive=False）では開かない（os.startfile 相当が呼ばれない）。
  C6 確定不変: シートの有無で○×UI の確定（apply_fn 呼び出し）が変わらない。
  C7 captures 非書込: シート出力は bench/ 配下。captures/ を指定しても bench/ へ退避。

captures/ は触らない（tmp のみ）。matplotlib は build 時のみ。
"""
import csv
import json
import os

import matplotlib
import review
import contact_sheet

matplotlib.use("Agg")
import matplotlib.pyplot as _plt  # noqa: E402


_SUGGEST_FIELDS = ["record", "cc_class", "cc_rationale", "spurious_warn",
                   "recommend", "png", "det_freq_mhz", "bw_mhz", "duty",
                   "duty_inconclusive"]


def _real_png(path):
    """imshow 可能な本物の小さな PNG を書く（欠損耐性テストと区別するため）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig = _plt.figure(figsize=(1.4, 1.4))
    _plt.plot([0, 1], [0, 1])
    fig.savefig(path)
    _plt.close(fig)


def _wm(dirpath, name, *, label="BLE/Bluetooth (adv?)", method="rule",
        confidence=0.62, lo=2.4795e9, hi=2.4807e9):
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


def _make_env(tmp_path, rows, images=None):
    """caps（meta＋任意で実 PNG）と suggestions.csv を作る。returns (caps, sc)。"""
    images = set(images or ())
    caps = tmp_path / "caps"; caps.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "out"; out.mkdir(parents=True, exist_ok=True)
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
            if r["record"] in images:
                _real_png(str(caps / "_images" / (r["record"] + ".png")))
    return str(caps), str(sc)


# ===========================================================================
# C1 / C2: キャプション・番号一致（純関数）
# ===========================================================================
def test_caption_format():
    assert contact_sheet.caption_for(1, "wifi") == "[1] wifi"
    assert contact_sheet.caption_for(7, "unclear") == "[7] unclear"
    assert contact_sheet.caption_for(3, "") == "[3] (提案なし)"
    assert contact_sheet.caption_for(4, None) == "[4] (提案なし)"


def test_sheet_entries_index_matches_presentation_order():
    # allow_y（一括候補）2件 + allow_y=False（個別）2件。提示順は batch→rest。
    ctxs = [
        dict(record="w", cc_class="wifi", png=None, allow_y=True, has_suggestion=True),
        dict(record="b", cc_class="ble-adv", png=None, allow_y=True, has_suggestion=True),
        dict(record="u", cc_class="unclear", png=None, allow_y=False, has_suggestion=True),
        dict(record="n", cc_class="", png=None, allow_y=False, has_suggestion=False),
    ]
    batch = [c for c in ctxs if c["allow_y"]]
    rest = [c for c in ctxs if not c["allow_y"]]
    entries = contact_sheet.sheet_entries(batch + rest)
    # index は 1-based で提示順に一致。
    assert [e["index"] for e in entries] == [1, 2, 3, 4]
    # batch 部分（[1],[2]）＝ _batch_confirm の一括候補番号と対応。
    assert entries[0]["caption"] == "[1] wifi" and entries[0]["record"] == "w"
    assert entries[1]["caption"] == "[2] ble-adv" and entries[1]["record"] == "b"
    # 各キャプションに index と cc_class が含まれる（提案なしは (提案なし)）。
    assert entries[2]["caption"] == "[3] unclear"
    assert entries[3]["caption"] == "[4] (提案なし)"
    # allow_y は保持（枠色ヒント）。
    assert [e["allow_y"] for e in entries] == [True, True, False, False]


# ===========================================================================
# C3 / C4: 生成・PNG 欠損耐性
# ===========================================================================
def test_build_generates_file(tmp_path):
    png = str(tmp_path / "img" / "a.png")
    _real_png(png)
    entries = contact_sheet.sheet_entries([
        dict(record="a", cc_class="wifi", png=png, allow_y=True, has_suggestion=True),
    ])
    out = str(tmp_path / "bench" / "contact_sheet.png")
    got = contact_sheet.build_contact_sheet(entries, out)
    assert got == out and os.path.isfile(out) and os.path.getsize(out) > 0


def test_build_tolerates_missing_png(tmp_path):
    real = str(tmp_path / "img" / "a.png")
    _real_png(real)
    entries = contact_sheet.sheet_entries([
        dict(record="a", cc_class="wifi", png=real, allow_y=True, has_suggestion=True),
        dict(record="miss", cc_class="ble-adv",
             png=str(tmp_path / "img" / "nope.png"), allow_y=True, has_suggestion=True),
        dict(record="none", cc_class="", png=None, allow_y=False, has_suggestion=False),
    ])
    out = str(tmp_path / "bench" / "sheet.png")
    got = contact_sheet.build_contact_sheet(entries, out)      # 欠損でも落ちない
    assert got == out and os.path.isfile(out)


def test_build_zero_entries_returns_none(tmp_path):
    assert contact_sheet.build_contact_sheet([], str(tmp_path / "x.png")) is None


# ===========================================================================
# C5: オプション性・非侵襲（--open-sheet 無し / ヘッドレス）
# ===========================================================================
def _run(tmp_path, rows, answers, *, open_sheet=False, interactive=None,
         images=None):
    caps, sc = _make_env(tmp_path, rows, images=images)
    calls, opened, printed = [], [], []
    it = iter(answers)

    def input_fn(_p):
        try:
            return next(it)
        except StopIteration:
            return "q"

    def apply_fn(meta_path, ann_index, new_label, **kw):
        calls.append(dict(record=os.path.basename(meta_path), new_label=new_label))

    review.run_suggest_review(
        caps, sc, input_fn=input_fn, print_fn=printed.append, apply_fn=apply_fn,
        batch_confirm=True, open_sheet=open_sheet, interactive=interactive,
        open_fn=lambda p: opened.append(p))
    return calls, opened, "\n".join(printed)


def test_open_sheet_off_does_not_open(tmp_path):
    rows = [dict(record="a", cc_class="ble-adv", recommend="confirm-ble")]
    calls, opened, _ = _run(tmp_path, rows, ["n"], open_sheet=False,
                            interactive=True, images=["a"])
    assert opened == []                                # 生成/オープンなし（既定）


def test_headless_does_not_open(tmp_path):
    rows = [dict(record="a", cc_class="ble-adv", recommend="confirm-ble")]
    calls, opened, out = _run(tmp_path, rows, ["n"], open_sheet=True,
                              interactive=False, images=["a"])
    assert opened == []                                # ヘッドレスでは開かない
    assert "非対話" in out


def test_open_sheet_interactive_opens_under_bench(tmp_path):
    rows = [dict(record="a", cc_class="ble-adv", recommend="confirm-ble"),
            dict(record="b", cc_class="wifi", recommend="confirm-wifi")]
    calls, opened, _ = _run(tmp_path, rows, ["n"], open_sheet=True,
                            interactive=True, images=["a", "b"])
    assert len(opened) == 1                            # 1枚を自動オープン
    sheet = opened[0]
    assert os.path.isfile(sheet)
    # C7: 出力は bench/（suggestions.csv と同じ out/ ディレクトリ）で captures/ ではない。
    assert not review._under_captures(sheet)
    assert os.path.dirname(sheet).endswith(os.path.join("out"))
    assert calls == []                                 # n＝中止・確定していない


# ===========================================================================
# C6: 確定不変（シートの有無で apply_fn 呼び出しが変わらない）
# ===========================================================================
def test_confirmation_identical_with_and_without_sheet(tmp_path):
    rows = [dict(record="a", cc_class="ble-adv", recommend="confirm-ble"),
            dict(record="b", cc_class="wifi", recommend="confirm-wifi")]
    # y＝一括確定。シート無し。
    calls_off, opened_off, _ = _run(tmp_path / "off", rows, ["y"],
                                    open_sheet=False, interactive=True,
                                    images=["a", "b"])
    # y＝一括確定。シート有り（対話）。
    calls_on, opened_on, _ = _run(tmp_path / "on", rows, ["y"],
                                  open_sheet=True, interactive=True,
                                  images=["a", "b"])
    key = lambda cs: sorted((c["record"], c["new_label"]) for c in cs)
    assert key(calls_off) == key(calls_on)             # 確定内容が完全一致
    assert len(calls_on) == 2                          # 2件確定（一括）
    assert opened_off == [] and len(opened_on) == 1    # シートは表示のみ差分


# ===========================================================================
# C7: captures/ 配下への出力は bench/ へ退避
# ===========================================================================
def test_sheet_out_under_captures_is_redirected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("captures", exist_ok=True)
    ctxs = [dict(record="a", cc_class="wifi", png=None, allow_y=True,
                 has_suggestion=True)]
    opened = []
    printed = []
    # captures/ 配下を sheet_out に渡す → bench/ へ退避されること。
    review._maybe_open_contact_sheet(
        ctxs, os.path.join("captures", "contact_sheet.png"),
        printed.append, open_fn=opened.append, interactive=True)
    assert len(opened) == 1
    assert not review._under_captures(opened[0])        # captures/ には出ない
    assert "bench" in opened[0]
