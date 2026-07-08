"""
test_capture_ms.py — 取得スナップショット長ノブ(--capture-ms)の機構をロックする。

worklog(docs/worklog/worklog_2.4GHz_BLE_groundtruth.md) エントリ② の前提ブロッカー:
  既定の取得長 2^18≒13ms では duty 審判が adv 間隔(20-100ms)を分解できず全件
  inconclusive になる。--capture-ms で 300-500ms に延ばせるようにする。

ここでは:
  (1) 変換 config.dwell_samples_for_ms が ms→サンプル数を正しく出す（丸め・非正の拒否）。
  (2) main.py と同じ上書きロジックで cfg.sdr.dwell_samples が差し替わり snapshot が
      inconclusive 閾値(300ms)を越える。
  (3) SimBackend.capture_iq が n を厳守する＝保存される IQ 長が capture-ms に一致する
      （ハード不要の end-to-end 証明。実機 HackRFBackend も _read が buff[:n] を返す）。

凍結契約(spec.py/sigmf_io.py)・6継ぎ目のシグネチャは不変。既存テストは無改変(追加のみ)。
"""
import numpy as np
import pytest

from config import Config, dwell_samples_for_ms
from sdr import SimBackend

RATE = 20e6


# ---------------------------------------------------------------------------
# (1) 変換の正しさ
# ---------------------------------------------------------------------------
def test_ms_to_samples_exact():
    assert dwell_samples_for_ms(300.0, RATE) == 6_000_000
    assert dwell_samples_for_ms(400.0, RATE) == 8_000_000
    assert dwell_samples_for_ms(500.0, RATE) == 10_000_000
    # 既定 2^18 に対応する ms を戻すと 2^18 に一致（往復整合）。
    default_ms = (1 << 18) / RATE * 1000.0        # 13.107...ms
    assert dwell_samples_for_ms(default_ms, RATE) == (1 << 18)


def test_ms_to_samples_rounding_and_rate():
    assert dwell_samples_for_ms(0.05, RATE) == 1000       # 50us*20e6 = 1000
    assert dwell_samples_for_ms(1.0, RATE) == 20_000      # 1ms*20e6
    # rate を変えれば比例（rate をパラメータ化している証拠）。
    assert dwell_samples_for_ms(400.0, 10e6) == 4_000_000
    assert dwell_samples_for_ms(400.0, RATE) == 8_000_000


def test_ms_to_samples_rejects_nonpositive():
    for bad in (0.0, -1.0, -1e-9):
        with pytest.raises(ValueError):
            dwell_samples_for_ms(bad, RATE)


# ---------------------------------------------------------------------------
# (2) config 上書き（main.py と同一ロジック）で inconclusive を外れる
# ---------------------------------------------------------------------------
def test_config_override_crosses_resolution_gate():
    cfg = Config()
    assert cfg.sdr.dwell_samples == (1 << 18)             # 既定=約13ms
    default_snapshot_ms = cfg.sdr.dwell_samples / cfg.sdr.dwell_rate_hz * 1000.0
    assert default_snapshot_ms < 300.0                    # 既定は inconclusive 側

    # main.py の `if args.capture_ms is not None:` と同じ差し替え。
    cfg.sdr.dwell_samples = dwell_samples_for_ms(400.0, cfg.sdr.dwell_rate_hz)
    assert cfg.sdr.dwell_samples == 8_000_000
    snapshot_ms = cfg.sdr.dwell_samples / cfg.sdr.dwell_rate_hz * 1000.0
    assert snapshot_ms >= 300.0                           # 分解能ゲートを越える


# ---------------------------------------------------------------------------
# (3) SimBackend.capture_iq が n を厳守（保存 IQ 長 == capture-ms 由来のサンプル数）
# ---------------------------------------------------------------------------
def test_simbackend_capture_length_matches_capture_ms():
    cfg = Config()
    n = dwell_samples_for_ms(20.0, cfg.sdr.dwell_rate_hz)   # 20ms → 400,000
    assert n == 400_000
    be = SimBackend(cfg.sdr, seed=1)
    # ch37 近傍(帯域内は cw のみ)で取得。長さ・dtype が厳密一致すること。
    iq = be.capture_iq(2.402e9, cfg.sdr.dwell_rate_hz, n)
    assert iq.shape == (n,)
    assert iq.dtype == np.complex64
    # 既定長(13ms)とは明確に異なる長さになっている（ノブが効いている証拠）。
    assert n != (1 << 18)
