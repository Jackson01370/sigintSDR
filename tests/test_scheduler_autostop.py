"""
test_scheduler_autostop.py — 収集の自動停止(--max-records / --max-minutes)を固定する。

指示書_収集自動停止.md のテスト要件 1-5:
  1. --max-records で「ちょうど N 件」停止（N+1 件目を作らない）
  2. --max-minutes で時間到達により停止（無限ループにならない。時刻源を差し替えて決定的に）
  3. 両方指定 → 先に達した方で停止（records 先 / minutes 先 の2ケース）
  4. 後方互換: 未指定なら従来挙動（既定=無制限＝停止しない。既定値で分岐しない）
  5. 半端なファイルを作らない（.sigmf-meta と .sigmf-data が対で揃う）

sim バックエンドのみ（ハードウェア不要）。実時間依存を避けるための決定化:
  - time.sleep を no-op 化して巡回を高速に回す
  - max_minutes 系は time_fn を注入し、経過時間を決定的にする
  - one-shot 収集経路は品質ゲート無し(snr>=collect_snr_min のみ)なので、
    collect_snr_min を十分低くすれば毎ターゲット確実に保存される（ハング無し）
"""
import glob
import os
import time

import pytest

from config import Config, SDRConfig, ScanConfig, DwellConfig, QualityConfig
from sdr import SimBackend
from scheduler import HybridScheduler


def _cfg(max_dwell=6):
    """2.4GHz帯・小サンプルで素早く回す one-shot 収集用 Config。"""
    sdr = SDRConfig(dwell_samples=1 << 14)
    scan = ScanConfig(start_hz=2.4e9, stop_hz=2.5e9, max_dwell_per_cycle=max_dwell)
    return Config(sdr=sdr, scan=scan)


def _sched(tmp_path, **kw):
    """収集先付きスケジューラ。one-shot は毎ターゲット確実に保存させる設定。"""
    cfg = _cfg()
    be = SimBackend(cfg.sdr, seed=0)
    collect = str(tmp_path / "col")
    os.makedirs(collect, exist_ok=True)
    kw.setdefault("collect_snr_min", -100.0)   # one-shot は snr>=下限のみ→毎回保存
    kw.setdefault("collect_dedup_s", 0.0)      # 重複排除OFF→同一帯も毎巡回保存
    sched = HybridScheduler(be, cfg, store=None, collect_dir=collect, **kw)
    return sched, collect


def _count_pair(collect):
    """(meta 件数, data 件数) を返す。半端が無ければ両者は一致する。"""
    metas = glob.glob(os.path.join(collect, "*.sigmf-meta"))
    datas = glob.glob(os.path.join(collect, "*.sigmf-data"))
    return len(metas), len(datas)


def _stepping_time():
    """1回目=0.0(run開始), 以降=+1000s を返す決定的な時刻源。"""
    calls = {"n": 0}

    def fake_time():
        calls["n"] += 1
        return 0.0 if calls["n"] == 1 else 1000.0

    return fake_time


@pytest.fixture
def _no_sleep(monkeypatch):
    """巡回間の time.sleep(0.2) を無効化してテストを高速化する。"""
    monkeypatch.setattr(time, "sleep", lambda *a, **k: None)


# 1. --max-records で「ちょうど N 件」停止（N+1 件目を作らない）
def test_max_records_stops_exactly(tmp_path, _no_sleep):
    sched, collect = _sched(tmp_path, max_records=3)
    sched.run(once=False, verbose=False)
    assert sched._collected == 3              # ちょうど3件（4件目を作らない）
    assert sched._stop_reason == "max-records"
    assert _count_pair(collect) == (3, 3)     # 半端なし: meta/data 対で3件


# 2. --max-minutes で時間到達停止（時刻源を注入して決定的に・無限ループにならない）
def test_max_minutes_stops(tmp_path, _no_sleep):
    sched, collect = _sched(tmp_path, max_minutes=0.01, time_fn=_stepping_time())
    sched.run(once=False, verbose=False)      # 時間到達で必ず抜ける
    assert sched._stop_reason == "max-minutes"
    m, d = _count_pair(collect)
    assert m == d                             # 半端なし（対で揃う）


# 3a. 両方指定 → records が先に到達
def test_both_records_first(tmp_path, _no_sleep):
    # minutes は実質到達不能な大値、records=2 が先に達する。
    sched, collect = _sched(tmp_path, max_records=2, max_minutes=1e9)
    sched.run(once=False, verbose=False)
    assert sched._stop_reason == "max-records"
    assert sched._collected == 2
    assert _count_pair(collect) == (2, 2)


# 3b. 両方指定 → minutes が先に到達
def test_both_minutes_first(tmp_path, _no_sleep):
    # records は実質到達不能な大値、minutes=0.01 が先に達する。
    sched, collect = _sched(tmp_path, max_records=10_000, max_minutes=0.01,
                            time_fn=_stepping_time())
    sched.run(once=False, verbose=False)
    assert sched._stop_reason == "max-minutes"
    assert sched._collected < 10_000
    m, d = _count_pair(collect)
    assert m == d


# 4. 後方互換: 未指定なら従来挙動（自動停止は発火しない）
def test_backward_compat_unlimited(tmp_path, _no_sleep):
    sched, _ = _sched(tmp_path)               # max_records/max_minutes 未指定=既定0
    sched.run(once=True, verbose=False)       # once で従来どおり1巡回のみ
    assert sched._stop_reason is None         # 自動停止は発火しない
    assert sched._limit_hit() is None         # 既定は常に None（分岐が no-op）


# 4b. _limit_hit のユニット確認（既定=無制限は件数・時間とも常に None）
def test_limit_hit_defaults_none(tmp_path):
    sched, _ = _sched(tmp_path)
    sched._collected = 999                    # 件数が多くても…
    sched._run_start = 0.0                    # 経過が長くても…
    sched._time_fn = lambda: 1e9
    assert sched._limit_hit() is None         # 0=無制限は発火しない（既定値で分岐しない）


# --- dwell 観測モードでも保存直後の区切りで停止する（別経路の回帰）---
def _dwell_cfg():
    sdr = SDRConfig(dwell_samples=1 << 14)
    scan = ScanConfig(start_hz=2.4e9, stop_hz=2.5e9, max_dwell_per_cycle=3)
    dwell = DwellConfig(dwell_seconds=0.0, obs_interval_s=0.0,
                        min_observations=12, max_observations=12)
    quality = QualityConfig(min_detections=1, min_persistence=0.0)
    return Config(sdr=sdr, scan=scan, dwell=dwell, quality=quality)


def test_max_records_stops_dwell_mode(tmp_path, _no_sleep):
    """dwell_observe_cycle のミッドサイクル break（保存直後の区切り）を固定。"""
    cfg = _dwell_cfg()
    be = SimBackend(cfg.sdr, seed=1, burst_per_capture=True)
    collect = str(tmp_path / "col")
    os.makedirs(collect, exist_ok=True)
    sched = HybridScheduler(be, cfg, store=None, collect_dir=collect,
                            collect_snr_min=-100.0, collect_dedup_s=0.0,
                            dwell_mode=True, max_records=2)
    sched.run(once=False, verbose=False)
    assert sched._collected == 2
    assert sched._stop_reason == "max-records"
    assert _count_pair(collect) == (2, 2)     # 半端なし
