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
import dwell
import quality


def dwell_tune_offset(bw_hz, offset_hz: float, max_bw_hz: float) -> float:
    """dwell 収集で実際に適用するチューナーオフセット(Hz)を返す（オフセットチューニング）。

    狭帯域ターゲット(bw_hz <= max_bw_hz)のときだけ offset_hz を返し、それ以外
    ——広帯域(bw>max_bw で信号端が窓外に出る)・bw 不明(None)・offset_hz=0(既定)——は
    0.0 を返す（適用判断は「不明なら適用しない」側へ倒す）。呼び出し側は
    f_tune = target_center + dwell_tune_offset(...) を capture_iq / measure_signal /
    記録 center に一貫して渡すこと（絶対周波数の一貫性が本機能の技術的核心）。

    狙った獲物を取得帯域の中央(DC=0Hz)から離し、DC残留線や DC位置の固定スパイク
    (クロック高調波等)との重なりで dc-spike ゲートに構造的に落ちるのを防ぐ。適用判断を
    dwell 収集の全経路で共有する唯一の関門（経路間の不一致を作らない）。
    """
    if offset_hz and bw_hz is not None and float(bw_hz) <= float(max_bw_hz):
        return float(offset_hz)
    return 0.0


class HybridScheduler:
    def __init__(self, backend: SDRBackend, cfg: Config, store: Store | None = None,
                 collect_dir: str | None = None, collect_snr_min: float = 8.0,
                 collect_dedup_s: float = 30.0, dwell_mode: bool = False):
        self.be = backend
        self.cfg = cfg
        self.store = store
        self.collect_dir = collect_dir
        self.collect_snr_min = collect_snr_min
        # 滞在観測モード（既定 off）。on だと各対象に滞在し反復観測 → 品質ゲートで
        # 選別して保存する。off は従来どおり1回ドウェルの収集経路。
        self.dwell_mode = dwell_mode
        # 収集側の重複排除: この秒数の窓で近接周波数を既収集ならスキップ
        # (0 以下で無効化)。_build_targets の近接排除を収集ループにも広げる。
        self.collect_dedup_s = collect_dedup_s
        self._recent_collect: list[dict] = []   # 直近に収集した {center, bw, t}
        self._collected = 0
        self._skipped_dup = 0
        # データ出所を正直に記録（合成と実測を後で混ぜないため）
        self._hw = ("HackRF One" if type(backend).__name__ == "HackRFBackend"
                    else "sigscan-sim (synthetic)")
        # 受信入口で DC オフセット補正(DCスパイク除去)を適用したか。保存する SigMF の
        # global に sigscan:dc_removed として正直に記録する（後で再レンダ/学習時に判別可）。
        self._dc_removed = bool(getattr(backend, "dc_removal", False))
        # 帯域フォーカス: 指定 [start, stop] に張り付き、範囲外候補を _build_targets の
        # 出口で除外する（関所は1点のみ）。収集記録には来歴として SigMF global に
        # sigscan:band_focus を残す（dc_removed と同じ流儀）。既定 OFF で挙動不変。
        self._band_focus = bool(cfg.scan.band_focus)
        # CNN 監査（3段分類器の 2 段目＝監査役）。既定 OFF（挙動不変）。
        #   有効時のみチェックポイントを 1 度だけロードし、滞在観測の保存候補の IQ を
        #   classify のステップ2（CNN 監査）に通す。torch を引く cnntrain.infer は
        #   **有効時のみ遅延 import**（OFF/torch 無し環境を壊さない＝禁止事項3）。
        #   フラグ ON かつチェックポイント不在は明示エラー（黙ってスキップしない）。
        cnn_cfg = getattr(cfg, "cnn", None)
        self._cnn_enabled = bool(cnn_cfg is not None and cnn_cfg.enabled)
        self._cnn_ckpt = None
        self._cnn_ckpt_name = ""
        if self._cnn_enabled:
            ckpt_path = self._resolve_cnn_checkpoint(cnn_cfg.checkpoint)
            from cnntrain import infer        # torch を引く: CNN 有効時のみ
            self._cnn_ckpt = infer.load_checkpoint(ckpt_path)
            self._cnn_ckpt_name = os.path.basename(ckpt_path)
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

    # --- CNN チェックポイント解決 ---
    @staticmethod
    def _resolve_cnn_checkpoint(path: str) -> str:
        """チェックポイントのパスを解決する。

        ディレクトリなら中の checkpoint.pt を補完。解決後に実在しなければ
        **明示エラー**（フラグ ON での不在を黙ってスキップしない＝作業指示）。
        """
        p = path
        if os.path.isdir(p):
            p = os.path.join(p, "checkpoint.pt")
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"CNN分類器(--cnn)が有効ですが、チェックポイントが見つかりません: "
                f"{p}（--cnn-checkpoint で指定。無効化するには --cnn を外す）")
        return p

    # --- フォーカス来歴 ---
    def _focus_global(self) -> dict:
        """フォーカス収集の SigMF global 来歴。

        focus OFF 時は空 dict を返す（global に何も足さない＝記録も挙動も不変）。
        dc_removed と同じく write 側のシグネチャは不変のまま、呼び出し側が
        extra_global に key を足す形（** 展開）で記録する。範囲も素直に残す。
        """
        if not self._band_focus:
            return {}
        return {"sigscan:band_focus": {"start": self.cfg.scan.start_hz,
                                       "stop": self.cfg.scan.stop_hz}}

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
            # 関所: 帯域フォーカス（範囲外除外はこの1点＝候補の合流点に集約）。
            # focus 有効時、中心周波数が [start, stop] 外の候補をここで弾く。これで
            #   a. バンドプラン巡回由来の範囲外目標（例: 2.4GHz指定時の GPS/W56）
            #   b. サーベイ端の食み出し検出（例: --stop 2.5e9 での 2504/2512MHz）
            # の両方が同じ関所で消える。弾いた候補は seen も汚さない。focus OFF 時は
            # 素通り（従来どおり・挙動不変）。バンド巡回(step2)はこの関所で範囲外を
            # 飛ばしつつ枠が埋まるまで範囲内バンドを拾い続けるため、指定帯域に張り付く。
            if self._band_focus and not (sc.start_hz <= center <= sc.stop_hz):
                return
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
        # オフセットチューニング: 狭帯域ターゲットはチューナーを数MHzずらし、獲物を取得帯域
        #   中央(DC)から外す。既定 dwell_offset_hz=0 で off=0 ＝ center_hz=target["center"]
        #   （従来挙動）。f_tune(=center_hz) を capture_iq / measure_signal / 記録 center に
        #   一貫して渡す（絶対周波数の一貫性）。target["center"]（狙い値）自体は書き換えない。
        off = dwell_tune_offset(target.get("bw"), c.sdr.dwell_offset_hz,
                                c.sdr.dwell_offset_max_bw_hz)
        center_hz = float(target["center"]) + off
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
                                  "sigscan:target_src": target.get("src", ""),
                                  "sigscan:dc_removed": self._dc_removed,
                                  "sigscan:dwell_offset_hz": off,
                                  **self._focus_global()},
                )
                self._collected += 1
                self._recent_collect.append(
                    dict(center=center_hz, bw=m["bw_hz"], t=time.time()))

        return m, result

    # --- 滞在観測モード ---
    def _save_dwell(self, obs, m: dict, verdict, result, target: dict,
                    cnn_prov: dict | None = None) -> None:
        """品質ゲートを通った滞在観測を SigMF 保存し、品質メタを annotation に記録。

        cnn_prov: CNN 監査の来歴（sigscan:cnn_* キー）。None なら従来どおり
        （extra_global は不変＝OFF 時の保存出力は完全に同一）。
        """
        ann = sigmf_io.annotation_from_result(m, result)
        name = (f"{int(round(obs.center_hz/1e6))}MHz_"
                f"{int(time.time()*1000)}_{self._collected}")
        base = os.path.join(self.collect_dir, name)
        # 適用したオフセットを来歴として記録（dwell_observe_cycle と同じ helper・同じ入力で
        #   再計算＝決定論的に一致。0.0 は不適用）。旧記録との区別はキーの有無で可能。
        off = dwell_tune_offset(target.get("bw"), self.cfg.sdr.dwell_offset_hz,
                                self.cfg.sdr.dwell_offset_max_bw_hz)
        extra_global = {"sigscan:rep_version": spec.SIGSCAN_REP_VERSION,
                        "sigscan:target_src": target.get("src", ""),
                        "sigscan:capture_mode": "dwell",
                        "sigscan:dc_removed": self._dc_removed,
                        "sigscan:dwell_offset_hz": off,
                        **self._focus_global()}
        if cnn_prov:
            # 凍結 write_recording は extra_global を global にそのまま展開する
            # （dc_removed と同じ流儀）。sigmf_io シグネチャは不変のまま来歴を付与。
            extra_global.update(cnn_prov)
        sigmf_io.write_recording(
            base, obs.best_iq, obs.center_hz, self.cfg.sdr.dwell_rate_hz,
            annotations=[ann], hw=self._hw,
            description=f"sigscan dwell-collect; rep={spec.SIGSCAN_REP_VERSION}",
            extra_global=extra_global,
        )
        # 凍結 write_recording は annotation の任意キーを通さないため、書き出し後に
        # 品質メタ（sigscan:）を annotation へ最小限 patch する（生IQには触れない）。
        quality.add_quality_to_meta(
            base, quality.quality_annotation_meta(obs, verdict))
        self._collected += 1
        self._recent_collect.append(
            dict(center=obs.center_hz, bw=m["bw_hz"], t=time.time()))

    def dwell_observe_cycle(self) -> list[dict]:
        """滞在観測モードの1サイクル。

        各ターゲットに滞在観測 → クロスターゲットのコムスプリアス判定 → 品質ゲート
        → 合格かつ収集条件を満たすものだけ SigMF 保存。検出ログ(store)は全件残す。
        returns: 各ターゲットの {obs, m, result, verdict, saved} のリスト。
        """
        c = self.cfg
        targets = self._build_targets()
        observed: list[tuple[dict, "dwell.DwellObservation"]] = []
        for t in targets:
            # オフセットチューニング（狭帯域のみ・既定 off=0=従来挙動）。f_tune を
            #   observe_dwell に渡すと capture_iq / measure_signal / DwellObservation.center_hz
            #   まで一貫して f_tune を使う（観測ループ側は無改変で絶対周波数の一貫性を得る）。
            #   適用判断は dwell() と同じ helper＝経路間で不一致を作らない。
            f_tune = float(t["center"]) + dwell_tune_offset(
                t.get("bw"), c.sdr.dwell_offset_hz, c.sdr.dwell_offset_max_bw_hz)
            obs = dwell.observe_dwell(
                self.be, f_tune, c.sdr.dwell_rate_hz, c.sdr.dwell_samples,
                c.dwell, c.quality, target_src=t.get("src", ""))
            observed.append((t, obs))

        # 「極細」判定に使う帯域幅: 検出帯はサーベイ実測(detect_segments)が堅牢。
        # 帯域を埋める/CW の信号で measure_signal の bw が当てにならない場合に効く。
        def _bw_eff(t, obs):
            if t.get("src") == "detected" and t.get("bw", 0.0) > 0:
                return float(t["bw"])
            return obs.bw_median_hz
        bw_eff = [_bw_eff(t, obs) for t, obs in observed]

        comb = quality.flag_comb_spurs([o for _, o in observed], c.quality,
                                       bw_list=bw_eff)

        outcomes: list[dict] = []
        for idx, ((t, obs), is_comb) in enumerate(zip(observed, comb)):
            verdict = quality.evaluate_quality(obs, c.quality, comb_spur=is_comb,
                                               bw_hz=bw_eff[idx])

            # 代表測定。検出帯はサーベイ側の帯域幅/SNR を信頼（既存挙動踏襲）。
            m = dict(obs.best)
            if t.get("src") == "detected":
                if t.get("bw", 0.0) > m["bw_hz"]:
                    m["bw_hz"] = float(t["bw"])
                m["snr_db"] = max(m["snr_db"], float(t.get("snr", 0.0)))

            # 保存候補か（品質合格 × SNR 下限 × 収集先あり）。CNN 監査は **保存候補の
            # IQ に対してのみ** 走らせる（サーベイ段では呼ばない＝IQ 無し・重い）。
            is_candidate = (self.collect_dir is not None and verdict.passed
                            and m["snr_db"] >= self.collect_snr_min)
            cnn_prov = None
            if self._cnn_enabled and is_candidate:
                ctx = classify.CNNAuditContext(
                    checkpoint=self._cnn_ckpt, iq=obs.best_iq,
                    rate=c.sdr.dwell_rate_hz, center_hz=obs.center_hz,
                    checkpoint_name=self._cnn_ckpt_name)
                classify.set_cnn_context(ctx)
                try:
                    result = classify.classify(m, c.bands)
                finally:
                    classify.clear_cnn_context()      # 必ず解除（次信号へ漏らさない）
                cnn_prov = ctx.provenance
            else:
                result = classify.classify(m, c.bands)
            if self.store:
                self.store.log(m, result)

            saved = False
            if is_candidate:
                if self._recently_collected(obs.center_hz, m["bw_hz"]):
                    self._skipped_dup += 1
                else:
                    self._save_dwell(obs, m, verdict, result, t,
                                     cnn_prov=cnn_prov)
                    saved = True

            outcomes.append(dict(obs=obs, m=m, result=result,
                                 verdict=verdict, saved=saved))
        return outcomes

    def _print_dwell(self, o: dict) -> None:
        """滞在観測1件の要約を表示（合格は SAVE、破棄は理由を併記）。"""
        obs, m, r, v = o["obs"], o["m"], o["result"], o["verdict"]
        if obs.snr_max_db < 5 and not o["saved"]:
            return  # 何も無かった帯は静かに飛ばす
        if o["saved"]:
            tag = "SAVE"
        elif v.passed:
            tag = "pass"          # 合格だが収集条件外（SNR下限/重複）
        else:
            tag = "drop:" + ",".join(v.reasons)
        print(f"  [{obs.target_src:>16}] "
              f"{obs.center_hz/1e6:8.2f}MHz  "
              f"BW={m['bw_hz']/1e6:5.1f}MHz  "
              f"SNRmax={obs.snr_max_db:4.0f}dB  "
              f"persist={obs.persistence:4.2f}({obs.n_detect}/{obs.n_obs})  "
              f"→ {r.label}  [{tag}]")

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

                if self.dwell_mode:
                    for o in self.dwell_observe_cycle():
                        if verbose:
                            self._print_dwell(o)
                else:
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
