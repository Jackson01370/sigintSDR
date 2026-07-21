"""
test_cnntrain_m3.py — M3: CNN を 3段分類器の 2段目（監査役）として classify へ接続。

ロックする契約:
  T12  整合チェック audit() の A/B/C/unmapped と確信度調整（純関数・torch不要）
       + classify 側の AuditDecision 反映（_apply_cnn_decision・torch不要）。
  T13  OFF 時 classify は rule_based と完全同一・コンテキスト非漏洩・OFF 経路は
       torch を import しない（禁止事項3）。
  T14  ON 時 end-to-end: 滞在観測で保存 → SigMF global に CNN 来歴が載る（torch要）。
  T15  --cnn 有効 + チェックポイント不在 → 明示エラー（torch不要：解決はinfer前）。

凍結契約（spec.py / sigmf_io.py / classify.classify シグネチャ / ClassResult）は
触らない。既存テストは無改変（本ファイルは追加のみ）。
"""
import glob
import os
import subprocess
import sys

import numpy as np
import pytest

import classify
from classify import rule_based, ClassResult
from config import (Config, SDRConfig, ScanConfig, DwellConfig, QualityConfig,
                    CNNConfig, BAND_PLAN)
from cnntrain import audit
from cnntrain.audit import (AuditDecision, VERDICT_A, VERDICT_B, VERDICT_C,
                            VERDICT_UNMAPPED, UNKNOWN_THRESHOLD)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _torch_present() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


HAS_TORCH = _torch_present()


def _m(center, bw, snr=30.0):
    return {"center_hz": center, "bw_hz": bw, "snr_db": snr}


# ===========================================================================
# T12: 整合チェック audit()（純関数・torch 不要）
# ===========================================================================
def test_audit_A_consistent_bonus_and_cap():
    """(A) CNN ∈ 期待集合 → +0.10（上限 0.95）。ラベル維持・Unknown化なし。"""
    # WiFi → 期待 {wideband-ofdm}。CNN=wideband-ofdm は整合。
    d = audit.audit("WiFi (2.4GHz, 20/40MHz)", 0.78, "wideband-ofdm", 0.9,
                    center_hz=2.437e9)
    assert d.verdict == VERDICT_A
    assert d.conf_after == pytest.approx(0.88)      # 0.78 + 0.10
    assert d.to_unknown is False
    # 上限 0.95: 0.90 + 0.10 = 1.00 → 0.95 にクリップ。
    d2 = audit.audit("WiFi", 0.90, "wideband-ofdm", 0.5)
    assert d2.conf_after == pytest.approx(0.95)


def test_audit_B_context_explained_penalty_and_floor():
    """(B) 期待外だが文脈で説明（2.4G×WiFi×pulse-radar）→ −0.05（下限0.05）。維持。"""
    d = audit.audit("WiFi (2.4GHz, 20/40MHz)", 0.78, "pulse-radar", 0.8,
                    center_hz=2.437e9)
    assert d.verdict == VERDICT_B
    assert d.conf_after == pytest.approx(0.73)      # 0.78 - 0.05
    assert d.to_unknown is False
    assert d.rationale                              # 文脈の根拠が載る
    # 下限 0.05: 0.06 - 0.05 = 0.01 → 0.05 にクランプ。
    d2 = audit.audit("WiFi", 0.06, "pulse-radar", 0.8, center_hz=2.45e9)
    assert d2.verdict == VERDICT_B
    assert d2.conf_after == pytest.approx(0.05)


def test_audit_B_is_frequency_gated_to_24ghz():
    """文脈(B)は周波数で限定: 5GHz の WiFi×pulse-radar は文脈不成立 → (C) に落ちる。"""
    d = audit.audit("WiFi (5GHz, 20-160MHz)", 0.78, "pulse-radar", 0.30,
                    center_hz=5.5e9)
    assert d.verdict == VERDICT_C                   # 2.4G 文脈に当たらない
    # center_hz=None でも文脈は成立しない（周波数文脈が必須）。
    d2 = audit.audit("WiFi (2.4GHz, 20/40MHz)", 0.78, "pulse-radar", 0.30,
                     center_hz=None)
    assert d2.verdict == VERDICT_C


def test_audit_C_conflict_min_formula_and_unknown_boundary():
    """(C) 説明不能 → min(rule,1-cnn)。0.7未満で Unknown化（境界は厳密 <0.7）。"""
    # narrowband-burst は WiFi 期待{wideband-ofdm}に無く、文脈規則も無い → (C)。
    # rule=0.78, cnn=0.20 → min(0.78, 0.80)=0.78 ≥0.7 → 維持。
    d = audit.audit("WiFi (2.4GHz, 20/40MHz)", 0.78, "narrowband-burst", 0.20,
                    center_hz=2.437e9)
    assert d.verdict == VERDICT_C
    assert d.conf_after == pytest.approx(0.78)
    assert d.to_unknown is False
    # rule=0.78, cnn=0.35 → min(0.78,0.65)=0.65 <0.7 → Unknown化。
    d2 = audit.audit("WiFi (2.4GHz, 20/40MHz)", 0.78, "narrowband-burst", 0.35,
                     center_hz=2.437e9)
    assert d2.conf_after == pytest.approx(0.65)
    assert d2.to_unknown is True
    # 境界 ちょうど 0.70 → Unknown化しない（< のみ）。
    d3 = audit.audit("WiFi", 0.70, "narrowband-burst", 0.30)
    assert d3.conf_after == pytest.approx(0.70)
    assert d3.to_unknown is False
    # 0.69 → Unknown化する。
    d4 = audit.audit("WiFi", 0.71, "narrowband-burst", 0.31)
    assert d4.conf_after == pytest.approx(0.69)
    assert d4.to_unknown is True


def test_audit_unmapped_is_noop():
    """期待対応表に無い用途 → unmapped。確信度・期待は変えない（所見のみ）。"""
    d = audit.audit("GPS/QZSS L1 C/A", 0.80, "wideband-ofdm", 0.9, center_hz=1.575e9)
    assert d.verdict == VERDICT_UNMAPPED
    assert d.conf_after == pytest.approx(0.80)      # 不変
    assert d.to_unknown is False
    assert d.expected is None
    assert d.cnn_class == "wideband-ofdm"           # 所見は保持


def test_apply_cnn_decision_into_classresult():
    """classify 側: AuditDecision を ClassResult に反映（純粋・torch不要）。"""
    base = ClassResult("WiFi (2.4GHz, 20/40MHz)", 0.78, "rule", "ISM 2.4G: OFDM",
                       ["WiFi (2.4GHz, 20/40MHz)", "Zigbee/独自2.4G"])

    # (A): ラベル維持・確信度更新・method=cnn・notes に監査タグ。
    da = AuditDecision(VERDICT_A, 0.88, False, "ok", ["wideband-ofdm"],
                       "wideband-ofdm", 0.9)
    ra = classify._apply_cnn_decision(base, da)
    assert ra.label == base.label and ra.confidence == pytest.approx(0.88)
    assert ra.method == "cnn" and "CNN監査:A-consistent" in ra.notes

    # (C) Unknown化: ラベル=UNKNOWN・元ラベルを候補先頭に温存・method=cnn。
    dc = AuditDecision(VERDICT_C, 0.65, True, "conflict",
                       ["wideband-ofdm"], "narrowband-burst", 0.35)
    rc = classify._apply_cnn_decision(base, dc)
    assert rc.label == classify.UNKNOWN
    assert base.label in rc.candidates              # 元ラベルは候補に残る（人間判断へ）
    assert rc.method == "cnn" and "Unknown" in rc.notes

    # unmapped: ラベル・確信度・method すべて不変（所見だけ notes に付く）。
    du = AuditDecision(VERDICT_UNMAPPED, 0.80, False, "noop", None,
                       "wideband-ofdm", 0.9)
    ru = classify._apply_cnn_decision(base, du)
    assert (ru.label, ru.confidence, ru.method) == (base.label, base.confidence,
                                                    base.method)


# ===========================================================================
# T13: OFF 時の従来挙動同一・非漏洩・torch を引かない
# ===========================================================================
_REPR = [
    (2.437e9, 20e6, "WiFi (2.4GHz, 20/40MHz)"),
    (3.55e9, 100e6, "5G NR (n77/n78 3.5G)"),
    (2.402e9, 2e6, "BLE/Bluetooth (adv?)"),
]


@pytest.mark.parametrize("center,bw,label", _REPR)
def test_classify_off_equals_rule_based(center, bw, label):
    """CNN コンテキスト未設定なら classify は rule_based と全フィールド同一。"""
    classify.clear_cnn_context()                    # 念のため OFF を保証
    m = _m(center, bw)
    exp = rule_based(m, BAND_PLAN)
    act = classify.classify(m, BAND_PLAN)
    assert act.label == exp.label == label
    assert act.confidence == exp.confidence
    assert act.method == "rule"
    assert act.notes == exp.notes
    assert act.candidates == exp.candidates


def test_cnn_context_does_not_leak_after_clear():
    """set→clear 後は OFF 挙動に戻る（次信号へ状態を漏らさない）。"""
    classify.set_cnn_context(None)                  # None セットも OFF と同義
    m = _m(2.437e9, 20e6)
    assert classify.classify(m, BAND_PLAN).method == "rule"
    classify.clear_cnn_context()
    assert classify.classify(m, BAND_PLAN).method == "rule"


def test_off_path_does_not_import_torch():
    """OFF 経路で classify を使っても torch / cnntrain.infer を import しない。"""
    code = (
        "import sys\n"
        "import classify\n"
        "from config import BAND_PLAN\n"
        "m={'center_hz':2.437e9,'bw_hz':20e6,'snr_db':30.0}\n"
        "r=classify.classify(m, BAND_PLAN)\n"
        "assert r.method=='rule', r.method\n"
        "assert 'torch' not in sys.modules, 'torch imported on OFF path'\n"
        "assert 'cnntrain.infer' not in sys.modules, 'infer imported on OFF path'\n"
        "print('OK')\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=REPO_ROOT)
    assert res.returncode == 0, res.stderr
    assert "OK" in res.stdout


# ===========================================================================
# T15: --cnn 有効 + チェックポイント不在 → 明示エラー（torch 不要）
# ===========================================================================
def _min_cfg(**cnn_kw):
    sdr = SDRConfig(dwell_samples=1 << 14)
    scan = ScanConfig(start_hz=2.4e9, stop_hz=2.5e9, max_dwell_per_cycle=1)
    return Config(sdr=sdr, scan=scan, cnn=CNNConfig(**cnn_kw))


def test_cnn_enabled_missing_checkpoint_raises(tmp_path):
    """フラグ ON かつチェックポイント不在は明示エラー（黙ってスキップしない）。"""
    from sdr import SimBackend
    from scheduler import HybridScheduler
    # 存在しないパス。
    cfg = _min_cfg(enabled=True, checkpoint=str(tmp_path / "does_not_exist"))
    be = SimBackend(cfg.sdr, seed=0)
    with pytest.raises(FileNotFoundError):
        HybridScheduler(be, cfg)
    # ディレクトリは在るが中に checkpoint.pt が無い場合も明示エラー。
    cfg2 = _min_cfg(enabled=True, checkpoint=str(tmp_path))
    be2 = SimBackend(cfg2.sdr, seed=0)
    with pytest.raises(FileNotFoundError):
        HybridScheduler(be2, cfg2)


def test_cnn_disabled_ignores_checkpoint(tmp_path):
    """OFF（既定）なら checkpoint が不在でもスケジューラ構築は通る（挙動不変）。"""
    from sdr import SimBackend
    from scheduler import HybridScheduler
    cfg = _min_cfg(enabled=False, checkpoint=str(tmp_path / "nope"))
    be = SimBackend(cfg.sdr, seed=0)
    sched = HybridScheduler(be, cfg)                # 例外なし
    assert sched._cnn_enabled is False
    assert sched._cnn_ckpt is None


# ===========================================================================
# T14: ON 時 end-to-end（滞在観測で保存 → SigMF global に CNN 来歴）。torch 要。
# ===========================================================================
@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory):
    """極小の合成データ + 1epoch チェックポイント（M2.5 流用せずテストを密閉）。"""
    from cnntrain import simgen, train
    d = tmp_path_factory.mktemp("m3_ckpt")
    data_dir = str(d / "simdata")
    run_dir = str(d / "run")
    simgen.generate(data_dir, per_class=4, seed=11)
    res = train.run_training(data_dir, run_dir, epochs=1, batch_size=8,
                             seed=11, val_ratio=0.25, log=lambda s: None)
    return res["ckpt_path"]


@pytest.mark.skipif(not HAS_TORCH, reason="torch 未導入（CNN e2e はスキップ）")
def test_dwell_e2e_records_cnn_provenance(tmp_path, tiny_ckpt):
    """ON: 滞在観測の保存 SigMF global に sigscan:cnn_* 来歴が載る（調整前後追跡可）。

    【2026-07-21 変更】バンド別 CNN ルーティング（案Y・config.BAND_CNN_ROUTES）導入に
    伴い、本テストの scan 帯 2.4-2.5GHz の検出は **専門家 CNN（runs/ism24_v2・用途3クラス
    ble-adv/wifi-24/spurious）** で監査されるようになった（汎用 runs/m2_5 ではない。
    CNNConfig.checkpoint=tiny_ckpt は汎用として init でロードされるが、2.4GHz ISM 帯では
    ルーティングが専門家を選ぶため使われない）。来歴が記録される検証意図は不変のまま、
    cnn_class が **専門家3クラス**・cnn_checkpoint が **ism24_v2** であることを積極的に
    固定する（＝ルーティングが 2.4GHz を専門家へ正しく回した証拠）。汎用5クラス監査の
    ロジックは T12 群（audit 単体・cnn_classes=None→汎用表）が引き続き緑で担保し、
    空ルート時の汎用フォールバック e2e は test_band_routing.py が担保する。"""
    import sigmf_io
    from sdr import SimBackend
    from scheduler import HybridScheduler
    from store import Store

    collect = str(tmp_path / "captures")
    sdr = SDRConfig(dwell_samples=1 << 14)
    scan = ScanConfig(start_hz=2.4e9, stop_hz=2.5e9, max_dwell_per_cycle=3)
    dwell = DwellConfig(dwell_seconds=0.0, obs_interval_s=0.0,
                        min_observations=12, max_observations=12)
    quality = QualityConfig(min_detections=1, min_persistence=0.0)
    cfg = Config(sdr=sdr, scan=scan, dwell=dwell, quality=quality,
                 cnn=CNNConfig(enabled=True, checkpoint=tiny_ckpt))
    store = Store(str(tmp_path / "t.db"))
    be = SimBackend(cfg.sdr, seed=1, burst_per_capture=True)
    sched = HybridScheduler(be, cfg, store=store, collect_dir=collect,
                            collect_snr_min=0.0, collect_dedup_s=0.0,
                            dwell_mode=True)
    assert sched._cnn_enabled is True and sched._cnn_ckpt is not None
    sched.run(once=True, verbose=False)

    metas = sorted(glob.glob(os.path.join(collect, "*.sigmf-meta")))
    assert len(metas) >= 1, "CNN ON でも保存候補が 1 件は出ること"
    base = metas[0][: -len(".sigmf-meta")]
    iq, meta = sigmf_io.read_recording(base)        # 凍結リーダで往復可能
    assert iq.dtype == np.complex64 and len(iq) > 0

    g = meta["global"]
    # CNN 来歴（global）。来歴の必須キーが揃う（記録意図は不変）。
    for key in ("sigscan:cnn_class", "sigscan:cnn_conf", "sigscan:cnn_verdict",
                "sigscan:cnn_checkpoint", "sigscan:rule_conf_pre"):
        assert key in g, f"{key} が global に無い"
    assert g["sigscan:cnn_verdict"] in (
        VERDICT_A, VERDICT_B, VERDICT_C, VERDICT_UNMAPPED)
    assert 0.0 <= float(g["sigscan:cnn_conf"]) <= 1.0
    assert 0.0 <= float(g["sigscan:rule_conf_pre"]) <= 1.0
    assert g["sigscan:capture_mode"] == "dwell"     # 既存来歴も維持

    # 【ルーティング積極検証】2.4GHz ISM 帯の保存レコードは専門家 CNN で監査される。
    #   cnn_class は専門家3クラス、cnn_checkpoint は "ism24_v2"（＝2.4GHz→専門家へ
    #   ルーティングされた証拠）。本 scan は 2.4-2.5GHz なので保存は全て ISM 帯に載る。
    EXPERT_CLASSES = {"ble-adv", "wifi-24", "spurious"}
    ism_records = 0
    for mp in metas:
        _, mt = sigmf_io.read_recording(mp[: -len(".sigmf-meta")])
        gg = mt["global"]
        fc = mt["captures"][0].get("core:frequency", 0.0)
        if 2400.0e6 <= fc <= 2483.5e6:              # ISM 2.4G バンド内の検出
            ism_records += 1
            assert gg["sigscan:cnn_class"] in EXPERT_CLASSES, gg["sigscan:cnn_class"]
            assert gg["sigscan:cnn_checkpoint"] == "ism24_v2", \
                gg["sigscan:cnn_checkpoint"]
    assert ism_records >= 1, "2.4GHz ISM の保存が 1 件は出て専門家監査されること"
