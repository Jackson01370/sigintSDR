"""test_band_routing.py — バンド別 CNN ルーティング（案Y・専門家は監査のみ）。

ロックする契約:
  R1 ルート選択: 検出 2.4GHz ISM → 専門家 runs/ism24_v2（用途3クラス）/
                 他バンド → 汎用（CNNConfig.checkpoint）。空ルート表なら全て汎用。
  R2 監査マッピング（専門家3クラス ↔ ルールラベル・純関数）:
                 ble-adv↔BLE / wifi-24↔WiFi = A-consistent、食い違い = C-conflict
                 →Unknown化、Zigbee等 = unmapped（所見のみ・確信度不変）。
  R3 専門家は監査のみ: 監査経路は用途ラベルを専門家クラスへ書き換えず、
                 method=human を付与しない（確定は人間の review.py のまま）。
  R4 汎用不変 + 汎用 e2e: 空ルート（BAND_CNN_ROUTES={}）なら 2.4GHz e2e が従来どおり
                 汎用5クラスで監査・記録される（後方互換＆汎用 e2e 経路の保全）。
                 汎用 audit ロジックは test_cnntrain_m3.py T12 群が担保。
  R5 モデルロード: 専門家 checkpoint が classes=['ble-adv','wifi-24','spurious'] で
                 ロードできる（runs/m2_5 を触らない・読むだけ）。

captures/ / runs/m2_5 は触らない（読み取りのみ）。torch 必要な項目は skip 可。
"""
import glob
import os

import pytest

import classify
from config import BAND_PLAN
from cnntrain import audit, expected
from cnntrain.audit import VERDICT_A, VERDICT_C, VERDICT_UNMAPPED

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERT = ["ble-adv", "wifi-24", "spurious"]
GENERAL5 = {"cw-tone", "narrowband-burst", "noise-only", "pulse-radar",
            "wideband-ofdm"}
_EXPERT_CKPT = os.path.join(REPO_ROOT, "runs", "ism24_v2", "checkpoint.pt")


def _torch_present() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


HAS_TORCH = _torch_present()
HAS_EXPERT = os.path.isfile(_EXPERT_CKPT)


# ===========================================================================
# R2: 監査マッピング（専門家3クラス ↔ ルールラベル）— 純関数・torch 不要
# ===========================================================================
def test_expert_audit_A_consistent():
    """ble-adv↔BLE / wifi-24↔WiFi は A-consistent（ラベル維持・Unknown化なし）。"""
    d = audit.audit("BLE/Bluetooth (adv?)", 0.62, "ble-adv", 0.95,
                    center_hz=2.402e9, cnn_classes=EXPERT)
    assert d.verdict == VERDICT_A and d.to_unknown is False
    d2 = audit.audit("WiFi (2.4GHz, 20/40MHz)", 0.78, "wifi-24", 0.9,
                     center_hz=2.437e9, cnn_classes=EXPERT)
    assert d2.verdict == VERDICT_A and d2.to_unknown is False


def test_expert_audit_C_conflict_to_unknown():
    """用途が食い違えば C-conflict → Unknown化（人間の確定へ回す）。"""
    # ルール BLE × 専門家 wifi-24（高確信）→ min(0.62,0.05)=0.05<0.7 → Unknown。
    d = audit.audit("BLE/Bluetooth (adv?)", 0.62, "wifi-24", 0.95,
                    center_hz=2.402e9, cnn_classes=EXPERT)
    assert d.verdict == VERDICT_C and d.to_unknown is True
    # ルール WiFi × 専門家 spurious（HackRF内部）→ Unknown 化して人間へ。
    d2 = audit.audit("WiFi (2.4GHz, 20/40MHz)", 0.78, "spurious", 0.92,
                     center_hz=2.44e9, cnn_classes=EXPERT)
    assert d2.verdict == VERDICT_C and d2.to_unknown is True


def test_expert_audit_unmapped_for_zigbee():
    """専門家対応表に無いルール用途（Zigbee等）は unmapped＝所見のみ・確信度不変。"""
    d = audit.audit("Zigbee/独自2.4G", 0.55, "wifi-24", 0.9,
                    center_hz=2.45e9, cnn_classes=EXPERT)
    assert d.verdict == VERDICT_UNMAPPED
    assert d.conf_after == pytest.approx(0.55) and d.to_unknown is False


def test_expert_audit_has_no_context_B():
    """専門家表は (B) 文脈規則が空 → 食い違いは即 (C)（汎用の pulse-radar 文脈は無関係）。"""
    d = audit.audit("WiFi (2.4GHz, 20/40MHz)", 0.78, "spurious", 0.30,
                    center_hz=2.437e9, cnn_classes=EXPERT)
    assert d.verdict == VERDICT_C            # (B) にならない


def test_tables_selected_by_cnn_vocab():
    """表の選択は checkpoint の語彙で決まる（専門家語彙→専門家表 / 他→汎用表）。"""
    et, cr = expected.tables_for_cnn_classes(EXPERT)
    assert et is expected.EXPECTED_EXPERT_ISM24 and cr == []
    et2, cr2 = expected.tables_for_cnn_classes(None)          # None=汎用
    assert et2 is expected.EXPECTED_REAL and cr2 is expected.CONTEXT_RULES
    et3, _ = expected.tables_for_cnn_classes(sorted(GENERAL5))  # 方式5クラス=汎用
    assert et3 is expected.EXPECTED_REAL


def test_general_audit_unchanged_by_default():
    """cnn_classes 未指定（既定 None）は従来どおり汎用表＝後方互換（回帰ガード）。"""
    d = audit.audit("WiFi (2.4GHz, 20/40MHz)", 0.78, "wideband-ofdm", 0.9,
                    center_hz=2.437e9)
    assert d.verdict == VERDICT_A and d.conf_after == pytest.approx(0.88)


# ===========================================================================
# R3: 専門家は監査のみ（用途ラベルを専門家クラスへ確定しない・method=human なし）
# ===========================================================================
class _FakeCkpt:
    """語彙だけ持つ偽 checkpoint（infer.classify_iq を差し替えて使う）。"""
    classes = ["ble-adv", "wifi-24", "spurious"]


@pytest.mark.skipif(not HAS_TORCH, reason="torch 未導入（infer 経由のため）")
def test_expert_audit_is_advisory_not_confirmation(monkeypatch):
    """専門家監査は所見のみ: ラベルを専門家クラスに書き換えず、method=human も付けない。"""
    from cnntrain import infer
    # 専門家が wifi-24 と言う状況を作る（rule は BLE）。
    monkeypatch.setattr(infer, "classify_iq",
                        lambda ckpt, iq, rate: ("wifi-24", 0.95))
    m = {"center_hz": 2.402e9, "bw_hz": 2e6, "snr_db": 30.0}    # rule → BLE
    ctx = classify.CNNAuditContext(checkpoint=_FakeCkpt(), iq=None, rate=20e6,
                                   center_hz=2.402e9, checkpoint_name="ism24_v2")
    classify.set_cnn_context(ctx)
    try:
        r = classify.classify(m, BAND_PLAN)
    finally:
        classify.clear_cnn_context()
    # 監査専用: method は human でない・ラベルは専門家クラス語彙にならない。
    assert r.method != "human"
    assert r.label not in ("ble-adv", "wifi-24", "spurious")
    # rule=BLE × 専門家=wifi-24 食い違い → Unknown 化（元ラベルは候補に温存）。
    assert r.label == classify.UNKNOWN
    assert "BLE/Bluetooth (adv?)" in r.candidates
    # 来歴に専門家 checkpoint・所見が載る（監査した証拠）。
    assert ctx.provenance["sigscan:cnn_checkpoint"] == "ism24_v2"
    assert ctx.provenance["sigscan:cnn_class"] == "wifi-24"


# ===========================================================================
# R1 / R5: ルート選択・モデルロード（torch + 専門家 checkpoint 要）
# ===========================================================================
@pytest.fixture(scope="module")
def tiny_general_ckpt(tmp_path_factory):
    """密閉の極小・汎用5クラス checkpoint（runs/m2_5 を触らない）。"""
    from cnntrain import simgen, train
    d = tmp_path_factory.mktemp("routing_gen")
    data_dir = str(d / "simdata")
    run_dir = str(d / "gen")
    simgen.generate(data_dir, per_class=4, seed=3)
    res = train.run_training(data_dir, run_dir, epochs=1, batch_size=8,
                             seed=3, val_ratio=0.25, log=lambda s: None)
    return res["ckpt_path"]


def _routing_cfg(general_ckpt):
    from config import Config, SDRConfig, ScanConfig, CNNConfig
    sdr = SDRConfig(dwell_samples=1 << 14)
    scan = ScanConfig(start_hz=2.4e9, stop_hz=2.5e9, max_dwell_per_cycle=1)
    return Config(sdr=sdr, scan=scan,
                  cnn=CNNConfig(enabled=True, checkpoint=general_ckpt))


@pytest.mark.skipif(not HAS_TORCH, reason="torch 未導入")
@pytest.mark.skipif(not HAS_EXPERT, reason="runs/ism24_v2 が無い")
def test_route_selection_24ghz_expert_else_general(tiny_general_ckpt):
    """2.4GHz ISM → 専門家 ism24_v2 / 他バンド(3.5GHz) → 汎用へフォールバック。"""
    from sdr import SimBackend
    from scheduler import HybridScheduler
    cfg = _routing_cfg(tiny_general_ckpt)
    be = SimBackend(cfg.sdr, seed=0)
    sched = HybridScheduler(be, cfg)
    assert "ISM 2.4G (WiFi/BT)" in sched._cnn_routes    # 専門家ルートがロード済み
    # 2.4GHz ISM → 専門家（用途3クラス・表示名 ism24_v2）。
    ck, name = sched._select_cnn_for(2.437e9)
    assert name == "ism24_v2"
    assert list(ck.classes) == ["ble-adv", "wifi-24", "spurious"]
    # 他バンド（3.5GHz 5G NR）→ 汎用（configured checkpoint）へフォールバック。
    ck2, name2 = sched._select_cnn_for(3.55e9)
    assert ck2 is sched._cnn_ckpt and name2 == sched._cnn_ckpt_name
    assert list(ck2.classes) != ["ble-adv", "wifi-24", "spurious"]


@pytest.mark.skipif(not HAS_TORCH, reason="torch 未導入")
def test_empty_routes_all_general(tiny_general_ckpt, monkeypatch):
    """空ルート表（BAND_CNN_ROUTES={}）なら 2.4GHz でも汎用＝後方互換。"""
    import config
    monkeypatch.setattr(config, "BAND_CNN_ROUTES", {})
    from sdr import SimBackend
    from scheduler import HybridScheduler
    cfg = _routing_cfg(tiny_general_ckpt)
    be = SimBackend(cfg.sdr, seed=0)
    sched = HybridScheduler(be, cfg)
    assert sched._cnn_routes == {}                      # ルート未ロード
    ck, name = sched._select_cnn_for(2.437e9)           # 2.4GHz でも汎用
    assert ck is sched._cnn_ckpt and name == sched._cnn_ckpt_name


@pytest.mark.skipif(not HAS_TORCH, reason="torch 未導入")
@pytest.mark.skipif(not HAS_EXPERT, reason="runs/ism24_v2 が無い")
def test_expert_checkpoint_loads_3_classes():
    """専門家 checkpoint が classes=['ble-adv','wifi-24','spurious'] でロードできる。"""
    from cnntrain import infer
    ck = infer.load_checkpoint(_EXPERT_CKPT)             # runs/m2_5 は触らない
    assert ck.classes == ["ble-adv", "wifi-24", "spurious"]
    assert ck.meta.get("real_data") is True             # 実データ専門家である


# ===========================================================================
# R4: 空ルート時は 2.4GHz e2e が従来どおり汎用5クラスで監査・記録（汎用 e2e 保全）
# ===========================================================================
@pytest.mark.skipif(not HAS_TORCH, reason="torch 未導入")
def test_empty_routes_e2e_uses_general(tmp_path, tiny_general_ckpt, monkeypatch):
    """空ルートなら滞在観測 e2e が汎用5クラスで監査・SigMF 来歴に記録（後方互換）。"""
    import config
    import sigmf_io
    from sdr import SimBackend
    from scheduler import HybridScheduler
    from store import Store
    from config import (Config, SDRConfig, ScanConfig, DwellConfig,
                        QualityConfig, CNNConfig)

    monkeypatch.setattr(config, "BAND_CNN_ROUTES", {})   # 空ルート＝汎用フォールバック
    collect = str(tmp_path / "captures")
    sdr = SDRConfig(dwell_samples=1 << 14)
    scan = ScanConfig(start_hz=2.4e9, stop_hz=2.5e9, max_dwell_per_cycle=3)
    dwell = DwellConfig(dwell_seconds=0.0, obs_interval_s=0.0,
                        min_observations=12, max_observations=12)
    quality = QualityConfig(min_detections=1, min_persistence=0.0)
    cfg = Config(sdr=sdr, scan=scan, dwell=dwell, quality=quality,
                 cnn=CNNConfig(enabled=True, checkpoint=tiny_general_ckpt))
    store = Store(str(tmp_path / "t.db"))
    be = SimBackend(cfg.sdr, seed=1, burst_per_capture=True)
    sched = HybridScheduler(be, cfg, store=store, collect_dir=collect,
                            collect_snr_min=0.0, collect_dedup_s=0.0,
                            dwell_mode=True)
    assert sched._cnn_routes == {}
    sched.run(once=True, verbose=False)

    metas = sorted(glob.glob(os.path.join(collect, "*.sigmf-meta")))
    assert len(metas) >= 1
    general_name = sched._cnn_ckpt_name
    for mp in metas:
        _, mt = sigmf_io.read_recording(mp[: -len(".sigmf-meta")])
        g = mt["global"]
        # 汎用5クラスで監査され、汎用 checkpoint 名が来歴に載る（従来挙動）。
        assert g["sigscan:cnn_class"] in GENERAL5, g["sigscan:cnn_class"]
        assert g["sigscan:cnn_checkpoint"] == general_name
