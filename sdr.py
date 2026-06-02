"""SDR バックエンド抽象化。

- SDRBackend: 共通インタフェース
    sweep_power(start, stop, bin_hz) -> (freqs_hz, power_db)   # サーベイ
    capture_iq(center_hz, rate, n)   -> np.complex64[n]        # ドウェル
- SimBackend  : ハード不要。1〜6GHz の電波環境を合成（検証用）。
- HackRFBackend: SoapySDR 経由の実機。SoapySDR を遅延 import。

実機メモ:
  export SOAPY_SDR_PLUGIN_PATH=/usr/local/lib64/SoapySDR/modules0.8-3
  HackRF は 1 プロセスのみ占有可能。
"""
from __future__ import annotations
import time
import numpy as np

from config import SDRConfig
from dsp import welch_psd


class SDRBackend:
    def sweep_power(self, start_hz, stop_hz, bin_hz):
        raise NotImplementedError

    def capture_iq(self, center_hz, rate, n) -> np.ndarray:
        raise NotImplementedError

    def close(self):
        pass


# ===========================================================================
# シミュレーション
# ===========================================================================
class _SimSignal:
    def __init__(self, center, width, power_db, prob=1.0, kind="ofdm"):
        self.center = center
        self.width = width
        self.power_db = power_db
        self.prob = prob          # サーベイ毎に存在する確率（バースト性）
        self.kind = kind          # ofdm / cw / spread / pulse
        self.active = True


# 1〜6GHz の代表的な電波環境（バンドプランと整合）
# power_db = ノイズ床からの相対 dB（正の値ほど強い）
def _default_sim_signals(rng: np.random.Generator) -> list[_SimSignal]:
    return [
        _SimSignal(1575.42e6, 20e6, 9,  prob=1.0, kind="spread"),    # GPS L1（実際は床下。検証用に微弱可視化）
        _SimSignal(1176.45e6, 24e6, 7,  prob=0.9, kind="spread"),    # GPS L5
        _SimSignal(2140.0e6,  40e6, 32, prob=1.0, kind="ofdm"),      # 携帯 B1 DL
        _SimSignal(1842.5e6,  15e6, 28, prob=0.9, kind="ofdm"),      # 携帯 B3 DL
        _SimSignal(3550.0e6, 100e6, 34, prob=0.95, kind="ofdm"),     # 5G n78
        _SimSignal(2437.0e6,  20e6, 35, prob=0.6, kind="ofdm"),      # WiFi ch6（バースト）
        _SimSignal(2402.0e6,   2e6, 18, prob=0.5, kind="cw"),        # BLE adv
        _SimSignal(2480.0e6,   2e6, 17, prob=0.5, kind="cw"),        # BLE adv
        _SimSignal(5180.0e6,  80e6, 33, prob=0.4, kind="ofdm"),      # WiFi 5G ch36（バースト）
        _SimSignal(5805.0e6,   5e6, 16, prob=0.3, kind="ofdm"),      # ETC/DSRC 付近
        _SimSignal(5740.0e6,  18e6, 24, prob=0.25, kind="ofdm"),     # FPVドローン映像（散発）
    ]


class SimBackend(SDRBackend):
    def __init__(self, cfg: SDRConfig, seed: int | None = 0,
                 burst_per_capture: bool = False):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.signals = _default_sim_signals(self.rng)
        self.noise_dbm = -45.0  # 任意基準のノイズ床（相対値）
        # 既定では存在状態の更新はサーベイ毎（=ドウェル中は固定）。滞在観測モードで
        # バースト挙動を擬似したいとき True にすると、capture_iq 毎に存在を再抽選し、
        # 同一帯でも観測ごとに出たり消えたりする（→ 持続率が 0〜1 で変化）。
        self.burst_per_capture = burst_per_capture

    def _refresh_activity(self):
        for s in self.signals:
            s.active = self.rng.random() < s.prob

    def sweep_power(self, start_hz, stop_hz, bin_hz):
        # サーベイ＝環境のスナップショット。ここで存在状態を更新。
        self._refresh_activity()
        n = max(16, int(round((stop_hz - start_hz) / bin_hz)))
        freqs = np.linspace(start_hz, stop_hz, n)
        power = self.noise_dbm + self.rng.normal(0, 1.5, n)
        for s in self.signals:
            if not s.active:
                continue
            # 帯域内をガウシアン状に持ち上げる（ブロックの肩を表現）
            sigma = s.width / 2.5
            bump = s.power_db * np.exp(-0.5 * ((freqs - s.center) / sigma) ** 2)
            # 帯域内はほぼ平坦なブロックにするため矩形成分も加味
            inband = np.abs(freqs - s.center) <= (s.width / 2)
            power = power + bump
            power[inband] = np.maximum(power[inband],
                                       self.noise_dbm + s.power_db
                                       + self.rng.normal(0, 0.8, int(inband.sum())))
        time.sleep(0.002)  # 取得時間の擬似
        return freqs, power

    def capture_iq(self, center_hz, rate, n) -> np.ndarray:
        if self.burst_per_capture:
            # 滞在観測の各取得で存在を再抽選 → バースト的な出現/消失を擬似。
            self._refresh_activity()
        t = np.arange(n) / rate
        # 基準ノイズ（複素ガウシアン）
        amp_noise = 10 ** (self.noise_dbm / 20.0)
        iq = (self.rng.normal(0, amp_noise, n)
              + 1j * self.rng.normal(0, amp_noise, n)).astype(np.complex64)
        half = rate / 2
        for s in self.signals:
            if not s.active:
                continue
            offset = s.center - center_hz
            if abs(offset) > half:   # 取得帯域外
                continue
            a = 10 ** ((self.noise_dbm + s.power_db) / 20.0)
            if s.kind == "cw":
                comp = a * np.exp(2j * np.pi * offset * t)
            else:
                # 帯域制限ノイズ（OFDM/拡散/レーダの代用）
                white = (self.rng.normal(0, 1, n) + 1j * self.rng.normal(0, 1, n))
                spec = np.fft.fftshift(np.fft.fft(white))
                f = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / rate))
                spec[np.abs(f - offset) > s.width / 2] = 0
                bl = np.fft.ifft(np.fft.ifftshift(spec))
                bl = bl / (np.std(bl) + 1e-9) * a
                comp = bl.astype(np.complex64)
            iq = iq + comp.astype(np.complex64)
        time.sleep(0.001)
        return iq


# ===========================================================================
# HackRF 実機（SoapySDR）
# ===========================================================================
class HackRFBackend(SDRBackend):
    def __init__(self, cfg: SDRConfig, device_args: str = "driver=hackrf"):
        self.cfg = cfg
        import SoapySDR                      # 遅延 import
        from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32  # noqa
        self._S = SoapySDR
        self._RX = SOAPY_SDR_RX
        self._CF32 = SOAPY_SDR_CF32
        self.dev = SoapySDR.Device(device_args)
        self.dev.setGainMode(self._RX, 0, False)
        self._apply_gains()

    def _apply_gains(self):
        c = self.cfg
        try:
            self.dev.setGain(self._RX, 0, "LNA", c.lna_gain)
            self.dev.setGain(self._RX, 0, "VGA", c.vga_gain)
            self.dev.setGain(self._RX, 0, "AMP", 14.0 if c.amp_on else 0.0)
        except Exception:
            self.dev.setGain(self._RX, 0, c.lna_gain + c.vga_gain)

    def _read(self, center_hz, rate, n) -> np.ndarray:
        self.dev.setSampleRate(self._RX, 0, rate)
        try:
            self.dev.setBandwidth(self._RX, 0, rate)
        except Exception:
            pass
        self.dev.setFrequency(self._RX, 0, center_hz)
        time.sleep(self.cfg.retune_settle_s)

        st = self.dev.setupStream(self._RX, self._CF32)
        self.dev.activateStream(st)
        buff = np.empty(n, np.complex64)
        got = 0
        chunk = 8192
        tmp = np.empty(chunk, np.complex64)
        # 整定直後の数バッファは捨てる
        for _ in range(2):
            self.dev.readStream(st, [tmp], chunk, timeoutUs=int(1e6))
        while got < n:
            want = min(chunk, n - got)
            sr = self.dev.readStream(st, [tmp], want, timeoutUs=int(1e6))
            num = sr.ret
            if num > 0:
                buff[got:got + num] = tmp[:num]
                got += num
            elif num < 0:
                break
        self.dev.deactivateStream(st)
        self.dev.closeStream(st)
        return buff[:got] if got else buff

    def sweep_power(self, start_hz, stop_hz, bin_hz):
        """リチューン・ループで広帯域スペクトルを合成（ステップ幅=瞬時帯域）。

        高速化したい場合はこのメソッドを hackrf_sweep のサブプロセス実装に
        差し替える（power のみで良いサーベイ用途）。
        """
        rate = self.cfg.dwell_rate_hz
        step = rate * 0.8          # 端の歪みを避けて 80% を採用
        nfft = max(256, int(round(rate / bin_hz)))
        nfft = 1 << int(np.round(np.log2(nfft)))
        n = max(self.cfg.survey_samples, nfft)

        all_f, all_p = [], []
        center = start_hz + step / 2
        while center - step / 2 < stop_hz:
            iq = self._read(center, rate, n)
            f_off, p = welch_psd(iq, rate, nperseg=nfft)
            keep = np.abs(f_off) <= step / 2     # 使う中央帯のみ
            all_f.append(center + f_off[keep])
            all_p.append(p[keep])
            center += step
        if not all_f:
            return np.array([start_hz, stop_hz]), np.array([-120.0, -120.0])
        freqs = np.concatenate(all_f)
        power = np.concatenate(all_p)
        order = np.argsort(freqs)
        return freqs[order], power[order]

    def capture_iq(self, center_hz, rate, n) -> np.ndarray:
        return self._read(center_hz, rate, n)

    def close(self):
        self.dev = None
