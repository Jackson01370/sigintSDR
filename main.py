#!/usr/bin/env python3
"""sigscan: HackRF による 1〜6GHz 電波自動識別（ハイブリッド・スキャン）。

例:
  python3 main.py --sim --once                 # ハード無しで1サイクル
  python3 main.py --sim                         # ハード無しで連続
  python3 main.py --hardware                    # HackRF 実機
  python3 main.py --hardware --start 2.4e9 --stop 2.5e9   # 2.4GHz帯に限定
"""
from __future__ import annotations
import argparse
import sys

from config import Config
from store import Store
from scheduler import HybridScheduler


def _force_utf8():
    # Windows cp932 等で日本語/Unicode を print してもクラッシュしないように。
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass


def build_backend(args, cfg, dwell_mode: bool = False):
    if args.hardware:
        from sdr import HackRFBackend
        cfg.sdr.lna_gain = args.lna
        cfg.sdr.vga_gain = args.vga
        cfg.sdr.amp_on = args.amp
        return HackRFBackend(cfg.sdr)
    from sdr import SimBackend
    # 滞在観測モードの sim では存在を取得毎に再抽選し、バースト挙動を擬似する
    # （合成なので限定的だが、持続率が 0〜1 で変化し品質ゲートの経路が通る）。
    return SimBackend(cfg.sdr, seed=args.seed, burst_per_capture=dwell_mode)


def main():
    _force_utf8()
    p = argparse.ArgumentParser(description="HackRF 1-6GHz 自動信号識別")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--sim", action="store_true", help="シミュレーション（既定）")
    src.add_argument("--hardware", action="store_true", help="HackRF 実機")

    p.add_argument("--start", type=float, default=None, help="開始周波数(Hz)")
    p.add_argument("--stop", type=float, default=None, help="終了周波数(Hz)")
    p.add_argument("--once", action="store_true", help="1サイクルで終了")
    p.add_argument("--survey-interval", type=float, default=None, help="サーベイ間隔(秒)")
    p.add_argument("--save-spectrograms", action="store_true", help="PNG保存(CNN/LLM前段)")
    p.add_argument("--collect", default=None, metavar="DIR",
                   help="自己収集: 検出信号をSigMF形式でDIRに保存(自動ラベル付き)")
    p.add_argument("--collect-snr", type=float, default=8.0,
                   help="収集するSNR下限dB(既定8)")
    p.add_argument("--collect-dedup-window", type=float, default=30.0,
                   metavar="SEC", help="収集側の近接重複排除の時間窓(秒, 既定30, 0で無効)")
    p.add_argument("--db", default="sigscan.db", help="SQLiteログのパス")

    # --- 滞在観測モード（dwell 観測の長時間化）---
    p.add_argument("--dwell", action="store_true",
                   help="滞在観測モード: 各対象に留まりバーストを待ち受け品質ゲートで選別")
    p.add_argument("--dwell-seconds", type=float, default=None, metavar="SEC",
                   help="各対象帯に滞在する秒数(既定10)。指定すると滞在観測モードを有効化")
    p.add_argument("--obs-interval", type=float, default=None, metavar="SEC",
                   help="滞在中の観測間隔 秒(既定0.5)")
    # --- 品質ゲートのしきい値オーバーライド（既定は config の厳しめ値）---
    p.add_argument("--q-detect-snr", type=float, default=None, metavar="DB",
                   help="1観測で検出とみなすSNR下限dB")
    p.add_argument("--q-min-detections", type=int, default=None, metavar="N",
                   help="滞在中に必要な最低検出回数")
    p.add_argument("--q-min-persistence", type=float, default=None, metavar="R",
                   help="必要な最低持続率(0〜1)")
    p.add_argument("--q-narrow-bw", type=float, default=None, metavar="HZ",
                   help="極細スプリアスとみなす占有帯域幅の上限Hz")
    p.add_argument("--no-quality-gate", action="store_true",
                   help="品質ゲートを無効化（足切りせず全件保存）")

    p.add_argument("--lna", type=float, default=24.0)
    p.add_argument("--vga", type=float, default=20.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=0, help="Sim用シード")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    cfg = Config()
    if args.start is not None:
        cfg.scan.start_hz = args.start
    if args.stop is not None:
        cfg.scan.stop_hz = args.stop
    if args.survey_interval is not None:
        cfg.scan.survey_interval_s = args.survey_interval
    if args.save_spectrograms:
        cfg.scan.save_spectrograms = True

    # 滞在観測モード: --dwell か --dwell-seconds の指定で有効化。
    dwell_mode = bool(args.dwell or args.dwell_seconds is not None)
    if args.dwell_seconds is not None:
        cfg.dwell.dwell_seconds = args.dwell_seconds
    if args.obs_interval is not None:
        cfg.dwell.obs_interval_s = args.obs_interval
    # 品質ゲートのしきい値オーバーライド
    if args.q_detect_snr is not None:
        cfg.quality.detect_snr_db = args.q_detect_snr
    if args.q_min_detections is not None:
        cfg.quality.min_detections = args.q_min_detections
    if args.q_min_persistence is not None:
        cfg.quality.min_persistence = args.q_min_persistence
    if args.q_narrow_bw is not None:
        cfg.quality.narrow_bw_hz = args.q_narrow_bw
    if args.no_quality_gate:
        cfg.quality.enabled = False

    mode = "HackRF実機" if args.hardware else "シミュレーション"
    print(f"sigscan  mode={mode}  "
          f"range={cfg.scan.start_hz/1e9:.2f}-{cfg.scan.stop_hz/1e9:.2f}GHz")

    backend = build_backend(args, cfg, dwell_mode=dwell_mode)
    store = Store(args.db)
    sched = HybridScheduler(backend, cfg, store,
                            collect_dir=args.collect,
                            collect_snr_min=args.collect_snr,
                            collect_dedup_s=args.collect_dedup_window,
                            dwell_mode=dwell_mode)
    if dwell_mode:
        gate = "無効" if not cfg.quality.enabled else "厳しめ"
        print(f"滞在観測モード: 各対象に {cfg.dwell.dwell_seconds:g}s 滞在 / "
              f"観測間隔 {cfg.dwell.obs_interval_s:g}s / 品質ゲート={gate}")
    if args.collect:
        print(f"収集モード: SigMF を {args.collect}/ に保存 (SNR>={args.collect_snr}dB)")
    try:
        sched.run(once=args.once, verbose=not args.quiet)
    finally:
        if args.collect:
            print(f"収集件数: {sched._collected}  "
                  f"(重複スキップ: {sched._skipped_dup})")
        store.close()
        backend.close()


if __name__ == "__main__":
    main()
