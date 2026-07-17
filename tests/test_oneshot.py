"""
test_oneshot.py — 収集〜確定ワンコマンド化の4点（追加のみ・既存無変更）をロックする。

1. バッチタグ命名: scheduler.collect_record_name / validate_collect_tag と、
   SimBackend 実収集で *tag* 選択できること。未指定は従来命名（後方互換）。
2. 指示文ブロック: review_suggest.format_cc_instruction_block が件数・タスクリスト・
   正しい併合コマンド（実 pattern/out/verdicts）・§5 基準を実値で含み、プレースホルダを残さない。
3. PNGパス: review._png_path_for が絶対パス、_build_suggest_ctx がローカル実解決を優先し
   （CSV の "(なし)"/相対に上書きされない）、画像不在時は明示メッセージ（"(なし)" 素通し無し）。

凍結契約（spec/sigmf_io）は import して読むだけ。captures/ には一切書かない（tmp_path のみ）。
LLM/API は不使用（指示文は定型文の文字列組立）。
"""
import fnmatch
import glob
import os

import pytest

import review
import scheduler
from cnntrain import review_suggest as rs


# ---------------------------------------------------------------------------
# 1. バッチタグ命名
# ---------------------------------------------------------------------------
def test_collect_record_name_backward_compat():
    """tag=None/空 は従来命名（タグ挿入なし＝完全後方互換）。"""
    assert scheduler.collect_record_name(2408.1e6, 3, None, ts_ms=1784090014856) \
        == "2408MHz_1784090014856_3"
    assert scheduler.collect_record_name(2408.1e6, 3, "", ts_ms=1784090014856) \
        == "2408MHz_1784090014856_3"


def test_collect_record_name_with_tag_selectable():
    """tag 指定で周波数直後にタグが入り、*tag* で一意に選べる。"""
    name = scheduler.collect_record_name(2408.1e6, 3, "wifi_aug2", ts_ms=1784090014856)
    assert name == "2408MHz_wifi_aug2_1784090014856_3"
    assert fnmatch.fnmatch(name, "*wifi_aug2*")


def test_validate_collect_tag():
    """英数・ハイフン・アンダースコアのみ許可。空は None、不正は ValueError。"""
    assert scheduler.validate_collect_tag(None) is None
    assert scheduler.validate_collect_tag("") is None
    assert scheduler.validate_collect_tag("  ") is None
    assert scheduler.validate_collect_tag("wifi-aug_2") == "wifi-aug_2"
    assert scheduler.validate_collect_tag("  keep  ") == "keep"      # strip
    for bad in ["bad tag", "a/b", "x*y", "日本語", "a.b"]:
        with pytest.raises(ValueError):
            scheduler.validate_collect_tag(bad)


def _fast_scheduler(tmp_path, **kw):
    from config import Config, SDRConfig, ScanConfig
    from sdr import SimBackend
    from store import Store
    sdr = SDRConfig(dwell_samples=1 << 14)
    scan = ScanConfig(start_hz=2.4e9, stop_hz=2.5e9, max_dwell_per_cycle=2)
    cfg = Config(sdr=sdr, scan=scan)
    store = Store(str(tmp_path / "t.db"))
    return scheduler.HybridScheduler(
        SimBackend(cfg.sdr, seed=0), cfg, store=store, **kw), store


def test_collect_tag_end_to_end_sim(tmp_path):
    """SimBackend 実収集で、全 SigMF ファイル名にタグが入り *tag* で選べる。"""
    collect = str(tmp_path / "captures")
    sched, store = _fast_scheduler(
        tmp_path, collect_dir=collect, collect_snr_min=0.0, collect_tag="smoke")
    try:
        sched.run(once=True, verbose=False)
    finally:
        store.close()
    allm = glob.glob(os.path.join(collect, "*.sigmf-meta"))
    tagged = glob.glob(os.path.join(collect, "*smoke*.sigmf-meta"))
    assert allm, "収集が1件も無い（前提が崩れている）"
    assert len(tagged) == len(allm), "全収集ファイルにタグが入るべき"


def test_collect_tag_default_none_backward_compat(tmp_path):
    """collect_tag 未指定でも従来どおり収集でき、ファイル名にタグ区間が無い。"""
    collect = str(tmp_path / "captures")
    sched, store = _fast_scheduler(
        tmp_path, collect_dir=collect, collect_snr_min=0.0)
    try:
        sched.run(once=True, verbose=False)
    finally:
        store.close()
    metas = glob.glob(os.path.join(collect, "*.sigmf-meta"))
    assert metas, "収集が1件も無い"
    for m in metas:
        # <freq>MHz_<ts>_<n> の3セグメント（タグ無し）であること
        base = os.path.basename(m)[: -len(".sigmf-meta")]
        assert base.split("_")[0].endswith("MHz")


def test_scheduler_rejects_bad_tag(tmp_path):
    """不正タグはスケジューラ構築時に ValueError（黙って通さない）。"""
    with pytest.raises(ValueError):
        _fast_scheduler(tmp_path, collect_dir=str(tmp_path / "c"),
                        collect_tag="bad tag")


# ---------------------------------------------------------------------------
# 2. コピペ用 CC 指示文ブロック（LLM 不使用の定型文）
# ---------------------------------------------------------------------------
def _mk_records(n):
    return [rs.SuggestRecord(
        record=f"2408MHz_1784{i:03d}_x", png="(なし)", det_freq_mhz=2412.0,
        bw_mhz=4.0, snr_db=16.0, rule_label="Zigbee", rule_confidence=0.5,
        duty=0.06, duty_inconclusive=True, spur_suspect=False,
        spurious_warn=False) for i in range(n)]


def test_cc_instruction_block_real_values_no_placeholder():
    recs = _mk_records(6)
    block = rs.format_cc_instruction_block(
        "bench/foo/", recs, "captures/", "2408MHz_1784*")
    # 件数（実値）
    assert "6 件" in block
    # タスクリスト・verdicts の実パス
    assert "classify_tasklist.md" in block
    assert "cc_verdicts.csv" in block
    # 正しい併合コマンド（実 pattern/out/verdicts）
    assert "cnntrain.review_suggest" in block
    assert '--pattern "2408MHz_1784*"' in block
    assert "--out bench/foo/" in block
    assert "--verdicts" in block
    # §5 の判断基準（主要クラス）
    for cls in ("ble-adv", "wifi", "spurious", "unclear"):
        assert cls in block
    # スプリアス誤確定ガードへの言及
    assert "skip" in block
    # プレースホルダを残さない
    for ph in ("<out>", "<tag>", "<pattern>", "<data>", "<n>"):
        assert ph not in block


def test_cc_instruction_block_count_tracks_records():
    assert "3 件" in rs.format_cc_instruction_block(
        "bench/b/", _mk_records(3), "captures/", "p*")


# ---------------------------------------------------------------------------
# 3. PNGパス表示（絶対パス・ローカル優先・明示フォールバック）
# ---------------------------------------------------------------------------
def _make_meta_with_png(tmp_path, base, with_png=True):
    if with_png:
        (tmp_path / "_images").mkdir(exist_ok=True)
        (tmp_path / "_images" / (base + ".png")).write_bytes(b"\x89PNG")
    meta = tmp_path / (base + ".sigmf-meta")
    meta.write_text("{}", encoding="utf-8")
    return str(meta)


def test_png_path_absolute_when_exists(tmp_path):
    base = "2408MHz_1784090014856_34"
    meta = _make_meta_with_png(tmp_path, base, with_png=True)
    p = review._png_path_for(meta)
    assert p is not None
    assert os.path.isabs(p)
    assert p.endswith(base + ".png")


def test_png_path_none_when_missing(tmp_path):
    meta = _make_meta_with_png(tmp_path, "x", with_png=False)
    assert review._png_path_for(meta) is None


def _item_for(meta):
    return {"meta_path": meta, "ann_index": 0, "label": "x", "method": "rule",
            "confidence": 0.4, "center": 2412e6, "bw": 4e6, "snr_db": 16, "ann": {}}


def test_build_ctx_png_local_first_over_stale_csv(tmp_path):
    """CSV の png 列が "(なし)" でも、ローカル実在画像（絶対パス）が優先される。"""
    base = "recLocal"
    meta = _make_meta_with_png(tmp_path, base, with_png=True)
    suggest = {base: {"record": base, "cc_class": "", "png": "(なし)",
                      "recommend": "needs-review"}}
    ctx = review._build_suggest_ctx(_item_for(meta), suggest)
    assert os.path.isabs(ctx["png"])
    assert ctx["png"] != "(なし)"
    assert ctx["png"].endswith(base + ".png")


def test_build_ctx_png_explicit_fallback_when_missing(tmp_path):
    """画像が無ければ "(なし)" を素通しせず、次操作を促す明示メッセージにする。"""
    meta = _make_meta_with_png(tmp_path, "recNo", with_png=False)
    ctx = review._build_suggest_ctx(_item_for(meta), {})
    assert ctx["png"] == "(画像未生成: view_captures.py 実行)"
    assert ctx["png"] != "(なし)"
