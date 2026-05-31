"""ハイブリッド・スケジューラ: 単一 SDR を時分割。

サイクル:
  1) survey_interval 経過していれば広域サーベイ → アクティブ帯検出
  2) ターゲット = 検出帯(SNR順) ∪ ホットバンド(優先度で巡回) を構築
  3) 上位 max_dwell_per_cycle 件をドウェル → IQ捕捉 → 測定 → 分類 → ログ
"""
from __future__ import annotations
import os
import time
import itertools

from config import Config
from sdr import SDRBackend
from store import Store
import dsp
import classify
import sigmf_io
import spec


class HybridScheduler:
    def __init__(self, backend: SDRBackend, cfg: Config, store: Store | None = None,
                 collect_dir: str | None = None, collect_snr_min: float = 8.0,
                 collect_dedup_s: float = 30.0):
        self.be = backend
        self.cfg = cfg
        self.store = store
        self.collect_dir = collect_dir
        self.collect_snr_min = collect_snr_min
        # 収集側の重複排除: この秒数の窓で近接周波数を既収集ならスキップ
        # (0 以下で無効化)。_build_targets の近接排除を収集ループにも広げる。
        self.collect_dedup_s = collect_dedup_s
        self._recent_collect: list[dict] = []   # 直近に収集した {center, bw, t}
        self._collected = 0
        self._skipped_dup = 0
        # データ出所を正直に記録（合成と実測を後で混ぜないため）
        self._hw = ("HackRF One" if type(backend).__name__ == "HackRFBackend"
                    else "sigscan-sim (synthetic)")
        self._last_survey = 0.0
        self._segments: list[dict] = []
        # ホットバンドを優先度の重み付きで巡回するためのイテレータ
        weighted = []
        for b in cfg.bands:
            weighted += [b] * max(1, b.priority)
        self._band_cycle = itertools.cycle(weighted) if weighted else None
        if cfg.scan.save_spectrograms:
            os.makedirs(cfg.scan.spectrogram_dir, exist_ok=True)
        if self.collect_dir:
            os.makedirs(self.collect_dir, exist_ok=True)

    # --- サーベイ ---
    def survey(self) -> list[dict]:
        sc = self.cfg.scan
        freqs, power = self.be.sweep_power(sc.start_hz, sc.stop_hz, sc.survey_bin_hz)
        segs = dsp.detect_segments(freqs, power, sc.detect_threshold_db,
                                   sc.min_segment_bw_hz)
        self._segments = segs
        self._last_survey = time.time()
        return segs

    # --- ターゲット構築 ---
    def _build_targets(self) -> list[dict]:
        sc = self.cfg.scan
        targets: list[dict] = []
        seen: list[float] = []

        def add(center, bw, src, snr=0.0):
            for c in seen:
                if abs(c - center) < max(bw, 1e6):   # 近接重複は除外
                    return
            seen.append(center)
            targets.append(dict(center=center, bw=bw, src=src, snr=snr))

        # 1) サーベイ検出帯（SNR順・優先）
        for s in self._segments:
            add(s["f_center"], s["bw_hz"], "detected", s["snr_db"])

        # 2) ホットバンド巡回で埋める
        if self._band_cycle is not None:
            tries = 0
            while len(targets) < sc.max_dwell_per_cycle and tries < 64:
                b = next(self._band_cycle)
                add(b.center, b.width, f"band:{b.name}")
                tries += 1

        return targets[: sc.max_dwell_per_cycle]

    # --- 収集側の重複排除 ---
    def _recently_collected(self, center_hz: float, bw: float) -> bool:
        """短時間窓で近接中心周波数を既に収集済みなら True。

        _build_targets の近接判定（abs(Δ) < max(bw, 1e6)）を収集ループにも
        適用する。サイクルをまたいでも、collect_dedup_s 窓の間は同一帯を
        重複収集しない。窓は新旧どちらの帯域幅も跨がない大きさを採る。
        """
        if self.collect_dedup_s <= 0:
            return False
        now = time.time()
        # 期限切れの記録を間引く
        self._recent_collect = [r for r in self._recent_collect
                                if now - r["t"] <= self.collect_dedup_s]
        for r in self._recent_collect:
            if abs(r["center"] - center_hz) < max(bw, r["bw"], 1e6):
                return True
        return False

    # --- ドウェル ---
    def dwell(self, target: dict) -> tuple[dict, classify.ClassResult]:
        c = self.cfg
        center_hz = target["center"]
        iq = self.be.capture_iq(center_hz, c.sdr.dwell_rate_hz, c.sdr.dwell_samples)
        m = dsp.measure_signal(iq, c.sdr.dwell_rate_hz, center_hz)

        # 検出帯はサーベイ側の帯域幅/SNRを信頼（信号がIBWより広い場合に重要）
        if target.get("src") == "detected":
            if target.get("bw", 0.0) > m["bw_hz"]:
                m["bw_hz"] = float(target["bw"])
            m["snr_db"] = max(m["snr_db"], float(target.get("snr", 0.0)))

        png = None
        spec_db = None
        if c.scan.save_spectrograms and m["snr_db"] >= 6:
            png = os.path.join(c.scan.spectrogram_dir,
                               f"{int(center_hz/1e6)}MHz_{int(time.time())}.png")
            if not dsp.save_spectrogram_png(iq, c.sdr.dwell_rate_hz, center_hz, png):
                png = None
            _, _, spec_db = dsp.spectrogram(iq, c.sdr.dwell_rate_hz)

        result = classify.classify(m, c.bands, spectrogram_db=spec_db, png_path=png)

        # 自己収集: 自動ラベル付きで SigMF 保存（track a の土台）
        if self.collect_dir and m["snr_db"] >= self.collect_snr_min:
            if self._recently_collected(center_hz, m["bw_hz"]):
                # 短時間窓で近接帯を既収集 → 重複としてスキップ（ログは残す）
                self._skipped_dup += 1
            else:
                ann = sigmf_io.annotation_from_result(m, result)
                name = f"{int(round(center_hz/1e6))}MHz_{int(time.time()*1000)}_{self._collected}"
                sigmf_io.write_recording(
                    os.path.join(self.collect_dir, name),
                    iq, center_hz, c.sdr.dwell_rate_hz,
                    annotations=[ann], hw=self._hw,
                    description=f"sigscan auto-collect; rep={spec.SIGSCAN_REP_VERSION}",
                    extra_global={"sigscan:rep_version": spec.SIGSCAN_REP_VERSION,
                                  "sigscan:target_src": target.get("src", "")},
                )
                self._collected += 1
                self._recent_collect.append(
                    dict(center=center_hz, bw=m["bw_hz"], t=time.time()))

        return m, result

    # --- メインループ ---
    def run(self, once: bool = False, verbose: bool = True):
        sc = self.cfg.scan
        try:
            while True:
                if time.time() - self._last_survey >= sc.survey_interval_s:
                    segs = self.survey()
                    if verbose:
                        print(f"\n[survey] active={len(segs)}  "
                              + "  ".join(f"{s['f_center']/1e6:.0f}MHz/"
                                         f"{s['bw_hz']/1e6:.1f}MHz/{s['snr_db']:.0f}dB"
                                         for s in segs[:6]))

                for t in self._build_targets():
                    m, r = self.dwell(t)
                    if self.store:
                        self.store.log(m, r)
                    if verbose and m["snr_db"] >= 5:
                        print(f"  [{t['src']:>16}] "
                              f"{m['center_hz']/1e6:8.2f}MHz  "
                              f"BW={m['bw_hz']/1e6:5.1f}MHz  "
                              f"SNR={m['snr_db']:4.0f}dB  "
                              f"→ {r.label} ({r.confidence:.2f}/{r.method})")

                if once:
                    break
                time.sleep(0.2)
        except KeyboardInterrupt:
            if verbose:
                print("\n停止しました。")
