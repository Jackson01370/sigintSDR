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

from config import Config
from store import Store
from scheduler import HybridScheduler


def build_backend(args, cfg):
    if args.hardware:
        from sdr import HackRFBackend
        cfg.sdr.lna_gain = args.lna
        cfg.sdr.vga_gain = args.vga
        cfg.sdr.amp_on = args.amp
        return HackRFBackend(cfg.sdr)
    from sdr import SimBackend
    return SimBackend(cfg.sdr, seed=args.seed)


def main():
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
    p.add_argument("--db", default="sigscan.db", help="SQLiteログのパス")

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

    mode = "HackRF実機" if args.hardware else "シミュレーション"
    print(f"sigscan  mode={mode}  "
          f"range={cfg.scan.start_hz/1e9:.2f}-{cfg.scan.stop_hz/1e9:.2f}GHz")

    backend = build_backend(args, cfg)
    store = Store(args.db)
    sched = HybridScheduler(backend, cfg, store,
                            collect_dir=args.collect,
                            collect_snr_min=args.collect_snr)
    if args.collect:
        print(f"収集モード: SigMF を {args.collect}/ に保存 (SNR>={args.collect_snr}dB)")
    try:
        sched.run(once=args.once, verbose=not args.quiet)
    finally:
        if args.collect:
            print(f"収集件数: {sched._collected}")
        store.close()
        backend.close()


if __name__ == "__main__":
    main()
