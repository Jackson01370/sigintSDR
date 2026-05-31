"""DSP: PSD推定・ノイズ床推定・アクティブ帯検出・IQからの信号測定・スペクトログラム。

依存は numpy のみ（移植性のため scipy には依存しない）。
"""
from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# PSD（Welch法）
# ---------------------------------------------------------------------------
def welch_psd(iq: np.ndarray, rate: float, nperseg: int = 1024,
              overlap: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """複素IQから片側でなく両側（複素信号用）の PSD を dB で返す。

    returns: (freqs_hz_offset, power_db)  freqs は中心からのオフセット。
    """
    iq = np.asarray(iq, dtype=np.complex64)
    if iq.size < nperseg:
        nperseg = 1 << int(np.floor(np.log2(max(iq.size, 8))))
    step = max(1, int(nperseg * (1.0 - overlap)))
    win = np.hanning(nperseg).astype(np.float32)
    win_pow = np.sum(win ** 2)

    acc = np.zeros(nperseg, dtype=np.float64)
    n = 0
    for start in range(0, iq.size - nperseg + 1, step):
        seg = iq[start:start + nperseg] * win
        spec = np.fft.fftshift(np.fft.fft(seg))
        acc += (np.abs(spec) ** 2)
        n += 1
    if n == 0:
        acc = np.ones(nperseg)
        n = 1
    psd = acc / (n * rate * win_pow)
    power_db = 10.0 * np.log10(psd + 1e-20)
    freqs = np.fft.fftshift(np.fft.fftfreq(nperseg, d=1.0 / rate))
    return freqs, power_db


# ---------------------------------------------------------------------------
# ノイズ床
# ---------------------------------------------------------------------------
def noise_floor_db(power_db: np.ndarray) -> float:
    """ロバストなノイズ床推定。下位側の中央値ベース（信号ピークに引きずられにくい）。"""
    p = np.asarray(power_db)
    med = np.median(p)
    below = p[p <= med]
    if below.size == 0:
        return float(med)
    # 中央値 + MAD で床を少し上に置く（純ノイズの揺らぎを跨がない程度）
    mad = np.median(np.abs(below - np.median(below)))
    return float(np.median(below) + 0.5 * mad)


# ---------------------------------------------------------------------------
# アクティブ帯検出（サーベイのパワースペクトルから）
# ---------------------------------------------------------------------------
def detect_segments(freqs_hz: np.ndarray, power_db: np.ndarray,
                    threshold_db: float, min_bw_hz: float,
                    smooth_hz: float = 500e3,
                    merge_gap_hz: float = 1.5e6) -> list[dict]:
    """ノイズ床 + threshold を超える連続区間を「アクティブ帯」として抽出。

    smooth_hz で平滑化し（単一ビンのちらつき除去）、merge_gap_hz 以内の
    近接区間は1つに統合する。freqs_hz は絶対周波数（昇順想定）。
    returns: [{f_lo, f_hi, f_center, bw_hz, peak_db, snr_db}, ...]
    """
    freqs = np.asarray(freqs_hz, dtype=np.float64)
    p = np.asarray(power_db, dtype=np.float64)
    order = np.argsort(freqs)
    freqs, p = freqs[order], p[order]
    bin_hz = float(np.median(np.diff(freqs))) if freqs.size > 1 else 1.0

    # 平滑化（移動平均）してちらつきを抑える
    w = max(1, int(round(smooth_hz / bin_hz)))
    p_s = np.convolve(p, np.ones(w) / w, mode="same") if w > 1 else p

    floor = noise_floor_db(p_s)
    mask = p_s > (floor + threshold_db)
    if not mask.any():
        return []

    idx = np.flatnonzero(mask)
    merge_gap_bins = max(1, int(round(merge_gap_hz / bin_hz)))
    splits = np.flatnonzero(np.diff(idx) > merge_gap_bins)
    groups = np.split(idx, splits + 1)

    segments: list[dict] = []
    for g in groups:
        lo_i, hi_i = int(g[0]), int(g[-1])
        f_lo = freqs[lo_i] - bin_hz / 2
        f_hi = freqs[hi_i] + bin_hz / 2
        bw = f_hi - f_lo
        if bw < min_bw_hz:
            continue
        span = slice(lo_i, hi_i + 1)            # 統合後の連続スパン
        seg_p = p[span]                          # ピーク/重心は生データで
        seg_f = freqs[span]
        peak = float(seg_p.max())
        wts = 10 ** (seg_p / 10.0)
        center = float(np.sum(seg_f * wts) / np.sum(wts))
        segments.append(dict(
            f_lo=float(f_lo), f_hi=float(f_hi), f_center=center,
            bw_hz=float(bw), peak_db=peak, snr_db=float(peak - floor),
        ))
    segments.sort(key=lambda s: s["snr_db"], reverse=True)
    return segments


# ---------------------------------------------------------------------------
# IQ からの信号測定（ドウェル時）
# ---------------------------------------------------------------------------
def measure_signal(iq: np.ndarray, rate: float, center_hz: float) -> dict:
    """捕捉した IQ から占有帯域幅・SNR・中心オフセットを推定。

    信号が取得帯域(IBW)をほぼ埋めている場合は床が取れないため、
    occupied_frac を見て帯域幅を「≥ IBW」として扱う。
    """
    freqs, p = welch_psd(iq, rate)
    floor = float(np.percentile(p, 20))      # ロバストな床（信号占有が高くても効く）
    peak_db = float(p.max())
    snr = peak_db - floor

    mask = p > (floor + 6.0)
    occupied_frac = float(mask.mean())
    if mask.any():
        active = freqs[mask]
        bw = float(active.max() - active.min())
        w = 10 ** (p[mask] / 10.0)
        offset = float(np.sum(active * w) / np.sum(w))
    else:
        bw = 0.0
        offset = 0.0

    if occupied_frac > 0.85:                  # 帯域を埋めている → 少なくとも IBW 幅
        bw = float(rate)

    return dict(
        center_hz=float(center_hz + offset),
        bw_hz=bw,
        snr_db=float(snr),
        peak_db=peak_db,
        noise_floor_db=float(floor),
        occupied_frac=occupied_frac,
    )


# ---------------------------------------------------------------------------
# スペクトログラム（CNN/LLM 連携の前段）
# ---------------------------------------------------------------------------
def spectrogram(iq: np.ndarray, rate: float, nfft: int = 512,
                overlap: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """STFT マグニチュード(dB)。returns (times_s, freqs_hz_offset, S_db[freq, time])."""
    iq = np.asarray(iq, dtype=np.complex64)
    step = max(1, int(nfft * (1.0 - overlap)))
    win = np.hanning(nfft).astype(np.float32)
    cols = []
    times = []
    for start in range(0, iq.size - nfft + 1, step):
        seg = iq[start:start + nfft] * win
        spec = np.fft.fftshift(np.fft.fft(seg))
        cols.append(20.0 * np.log10(np.abs(spec) + 1e-12))
        times.append(start / rate)
    if not cols:
        cols = [np.zeros(nfft)]
        times = [0.0]
    S = np.array(cols).T  # [freq, time]
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / rate))
    return np.array(times), freqs, S


def save_spectrogram_png(iq: np.ndarray, rate: float, center_hz: float,
                         path: str) -> bool:
    """matplotlib があればウォーターフォール PNG を保存。無ければ False。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    times, freqs, S = spectrogram(iq, rate)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=80)
    extent = [(center_hz + freqs[0]) / 1e6, (center_hz + freqs[-1]) / 1e6,
              times[-1] * 1e3, 0]
    ax.imshow(S.T, aspect="auto", cmap="viridis", extent=extent)
    ax.set_xlabel("MHz")
    ax.set_ylabel("ms")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True
