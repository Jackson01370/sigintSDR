"""
test_review_suggest.py — レビュー提案ツールと review.py --pattern の機構をロックする。

提案ツール(cnntrain.review_suggest):
  (1) spurious_warn ロジック（det≈2400 / spur_suspect）。
  (2) recommend 制約（spurious_warn=True は cc_class 不問で skip / ble-adv→confirm-ble / 他→skip）。
  (3) CSV/シート生成: 合成 meta で期待列が揃い、入力 SigMF を書き換えない（出力は out のみ）。
  (4) duty は cnntrain.dutyprobe を呼ぶ純粋連携（モックで差し替え可能＝再実装していない）。

review.py --pattern（追加のみ）:
  (5) pattern がファイル名一致集合だけを対象化し、非一致を除外。未指定時は従来どおり。
      main の実効 conf_max 解決（pattern 単独→信頼度無視 / 併用→AND / 無指定→0.5）。

凍結契約（spec/sigmf_io/dutyprobe）は import して読むだけ。既存テストは無改変（追加のみ）。
captures/ には一切書き込まない（tmp_path と bench 相当の out のみ）。
"""
import csv
import json
import os

import numpy as np
import pytest

import review
import sigmf_io
from cnntrain import review_suggest as rs


# ---------------------------------------------------------------------------
# (1) spurious_warn
# ---------------------------------------------------------------------------
def test_spurious_warn_logic():
    assert rs.spurious_warn_for(2400.05, False) is True     # det≈2400（40MHz高調波）
    assert rs.spurious_warn_for(2399.95, False) is True
    assert rs.spurious_warn_for(2402.0, False) is False     # BLE・spur なし
    assert rs.spurious_warn_for(2402.0, True) is True       # spur_suspect=True
    assert rs.spurious_warn_for(2400.2, False) is False     # 0.2MHz 離れ → 範囲外


# ---------------------------------------------------------------------------
# (2) recommend 制約（誤確定ガード）
# ---------------------------------------------------------------------------
def test_recommend_constraint():
    for c in rs.CC_CLASSES:                     # spurious_warn=True は必ず skip
        assert rs.recommend_for(True, c) == "skip", c
    assert rs.recommend_for(False, "ble-adv") == "confirm-ble"
    assert rs.recommend_for(False, "wifi") == "skip"
    assert rs.recommend_for(False, "spurious") == "skip"
    assert rs.recommend_for(False, "hopping") == "skip"
    assert rs.recommend_for(False, "") == "skip"


def test_apply_verdicts_enforces_guard(tmp_path):
    """cc_verdicts で ble-adv でも spurious_warn=True なら recommend=skip に矯正される。"""
    r = rs.SuggestRecord(
        record="x", png="(なし)", det_freq_mhz=2400.0, bw_mhz=0.2, snr_db=20.0,
        rule_label="未識別", rule_confidence=0.4, duty=1.0, duty_inconclusive=False,
        spur_suspect=True, spurious_warn=True)
    rs.apply_verdicts([r], {"x": ("ble-adv", "誤って ble と付けた")})
    assert r.cc_class == "ble-adv"
    assert r.recommend == "skip"                # ガードが confirm-ble を握り潰す


# ---------------------------------------------------------------------------
# 合成 SigMF ヘルパ
# ---------------------------------------------------------------------------
def _write(dirpath, name, *, center, f_lo, f_hi, label, conf, spur=False,
           n=60_000, seed=0):
    rng = np.random.default_rng(seed)
    iq = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    base = os.path.join(dirpath, name)
    sigmf_io.write_recording(
        base, iq, center_hz=center, sample_rate=20e6,
        annotations=[{"freq_lower_edge": f_lo, "freq_upper_edge": f_hi,
                      "label": label, "confidence": conf, "method": "rule",
                      "snr_db": 20.0}])
    if spur:   # spur_suspect は write_recording が扱わないので meta に直接足す
        with open(base + ".sigmf-meta") as f:
            m = json.load(f)
        m["annotations"][0]["sigscan:spur_suspect"] = True
        with open(base + ".sigmf-meta", "w") as f:
            json.dump(m, f, indent=2, ensure_ascii=False)
    return base


def _snapshot(d):
    snap = {}
    for name in sorted(os.listdir(d)):
        if name.endswith((".sigmf-meta", ".sigmf-data")):
            with open(os.path.join(d, name), "rb") as f:
                snap[name] = f.read()
    return snap


# ---------------------------------------------------------------------------
# (3) CSV/シート生成・入力 SigMF 非改変
# ---------------------------------------------------------------------------
def test_collect_and_write_no_input_mutation(tmp_path):
    data = tmp_path / "caps"
    data.mkdir()
    _write(str(data), "2402MHz_1_0", center=2402e6, f_lo=2401.4e6, f_hi=2402.5e6,
           label="BLE/Bluetooth (adv?)", conf=0.62, seed=1)
    _write(str(data), "2400MHz_2_0", center=2400e6, f_lo=2399.94e6, f_hi=2400.06e6,
           label="未識別信号", conf=0.4, spur=True, seed=2)
    before = _snapshot(str(data))

    recs = rs.collect_objective(str(data), pattern="*")
    assert len(recs) == 2
    by = {r.record: r for r in recs}
    assert by["2402MHz_1_0"].spurious_warn is False       # BLE・spur なし
    assert by["2400MHz_2_0"].spurious_warn is True        # det≈2400 かつ spur_suspect
    # 客観列が埋まっている。
    assert by["2402MHz_1_0"].rule_label == "BLE/Bluetooth (adv?)"
    assert by["2402MHz_1_0"].bw_mhz > 1.0
    assert abs(by["2400MHz_2_0"].det_freq_mhz - 2400.0) < 1e-6

    out = tmp_path / "out"
    rs.apply_verdicts(recs, {"2402MHz_1_0": ("ble-adv", "離散バースト")})
    csv_path = rs.write_suggestions_csv(str(out), recs)
    sheet_path = rs.write_confirm_sheet(str(out), recs, str(data), "*")

    # 期待列がすべて存在。
    with open(csv_path, encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    header = next(csv.reader(lines))
    for col in rs.CSV_FIELDS:
        assert col in header
    # 誤確定ガード: spurious 行は cc_class 未記入でも skip。ble は confirm-ble。
    assert by["2402MHz_1_0"].recommend == "confirm-ble"
    assert by["2400MHz_2_0"].recommend == "skip"

    # シートに正直バナー4種が入っている。
    sheet = open(sheet_path, encoding="utf-8").read()
    for b in rs.BANNER:
        assert b in sheet

    # 入力 SigMF は一切書き換わっていない（読み取りのみ）。
    assert _snapshot(str(data)) == before
    # 出力は out ディレクトリのみ（data には suggestions/confirm が生成されない）。
    assert not os.path.exists(os.path.join(str(data), "suggestions.csv"))
    assert os.path.exists(csv_path) and os.path.exists(sheet_path)


def test_collect_respects_pattern(tmp_path):
    data = tmp_path / "caps"
    data.mkdir()
    _write(str(data), "2402MHz_A_0", center=2402e6, f_lo=2401.4e6, f_hi=2402.5e6,
           label="BLE", conf=0.62, seed=3)
    _write(str(data), "5805MHz_B_0", center=5805e6, f_lo=5804e6, f_hi=5806e6,
           label="ETC", conf=0.5, seed=4)
    recs = rs.collect_objective(str(data), pattern="2402MHz_*")
    assert [r.record for r in recs] == ["2402MHz_A_0"]     # 非一致は除外


# ---------------------------------------------------------------------------
# (4) duty は dutyprobe を呼ぶ純粋連携（モック可＝再実装していない）
# ---------------------------------------------------------------------------
def test_duty_delegates_to_dutyprobe(tmp_path, monkeypatch):
    data = tmp_path / "caps"
    data.mkdir()
    base = _write(str(data), "2402MHz_M_0", center=2402e6, f_lo=2401.4e6,
                  f_hi=2402.5e6, label="BLE", conf=0.62, seed=5)

    calls = {"n": 0}

    def fake_measure_duty(iq, rate, center, f_lo, f_hi, **kw):
        calls["n"] += 1
        return {"duty": 0.4242, "snapshot_ms": 500.0, "hop_ms": 0.0128,
                "n_rows": 10, "n_band_bins": 20, "note": ""}

    # review_suggest が参照する dutyprobe.measure_duty を差し替える。
    monkeypatch.setattr(rs.dutyprobe, "measure_duty", fake_measure_duty)
    rec = rs.collect_one(base, str(data))
    assert calls["n"] == 1                       # dutyprobe を必ず呼ぶ
    assert rec.duty == 0.4242                     # 返り値をそのまま使う（再実装しない）
    assert rec.duty_inconclusive is False         # snapshot 500ms>=300 → conclusive


# ---------------------------------------------------------------------------
# (5) review.py --pattern（追加のみ）
# ---------------------------------------------------------------------------
def _meta(dirpath, name, *, conf, method="rule", center=2402e6):
    """find_low_confidence が読む最小 meta（IQ 不要）。"""
    meta = {
        "global": {"core:datatype": "cf32_le", "core:sample_rate": 20e6,
                   "core:hw": "HackRF One"},
        "captures": [{"core:sample_start": 0, "core:frequency": center,
                      "core:datetime": "2026-07-08T00:00:00Z"}],
        "annotations": [{"core:sample_start": 0, "core:sample_count": 1000,
                         "core:label": name, "sigscan:method": method,
                         "sigscan:confidence": conf, "sigscan:snr_db": 20.0,
                         "core:freq_lower_edge": center - 5e5,
                         "core:freq_upper_edge": center + 5e5}]}
    with open(os.path.join(dirpath, name + ".sigmf-meta"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def test_review_pattern_filters_matching_only(tmp_path):
    d = str(tmp_path)
    _meta(d, "2402MHz_a", conf=0.62)
    _meta(d, "2402MHz_b", conf=0.62)
    _meta(d, "5805MHz_x", conf=0.62)
    # pattern 単独想定（conf_max=inf）: 一致 rule 全件、非一致は除外。
    items = review.find_low_confidence(d, conf_max=float("inf"),
                                       pattern="2402MHz_*")
    names = sorted(os.path.basename(i["meta_path"]) for i in items)
    assert names == ["2402MHz_a.sigmf-meta", "2402MHz_b.sigmf-meta"]


def test_review_pattern_none_is_unchanged(tmp_path):
    d = str(tmp_path)
    _meta(d, "lo", conf=0.2)          # rule & conf<0.5 → 拾う
    _meta(d, "hi", conf=0.62)         # rule & conf>=0.5 → 除外（従来どおり）
    items = review.find_low_confidence(d, conf_max=0.5)     # pattern 未指定
    assert [os.path.basename(i["meta_path"]) for i in items] == ["lo.sigmf-meta"]


def test_main_conf_max_resolution(tmp_path, monkeypatch):
    """main の実効 conf_max: 無指定→0.5 / pattern単独→inf / 併用→明示値。"""
    captured = {}

    def fake_cmd_list(dirpath, conf_max, verdict=None, pattern=None):
        captured.clear()
        captured.update(conf_max=conf_max, pattern=pattern, verdict=verdict)
        return 0

    monkeypatch.setattr(review, "_cmd_list", fake_cmd_list)

    review.main([str(tmp_path), "--list"])
    assert captured == {"conf_max": 0.5, "pattern": None, "verdict": None}

    review.main([str(tmp_path), "--list", "--pattern", "2402MHz_*"])
    assert captured["conf_max"] == float("inf")
    assert captured["pattern"] == "2402MHz_*"

    review.main([str(tmp_path), "--list", "--conf-max", "0.7",
                 "--pattern", "2402MHz_*"])
    assert captured["conf_max"] == 0.7            # 併用時は明示 conf_max（AND）
    assert captured["pattern"] == "2402MHz_*"
