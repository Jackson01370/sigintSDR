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

    ※ この関数は CONTRACT §3 の継ぎ目(seam)。シグネチャ ["iq","rate","center_hz"] は
      test_seams.py がロックしており変更しない。DC残留ガード付きの派生が要るときは
      引数を足さず、下の measure_signal_dc_guarded（本体を共有）を使う。
    """
    return _measure_from_iq(iq, rate, center_hz, dc_guard_hz=0.0)


def measure_signal_dc_guarded(iq: np.ndarray, rate: float, center_hz: float,
                              dc_guard_hz: float) -> dict:
    """measure_signal に DC残留ガードを足した opt-in 版（seam を汚さない兄弟関数）。

    dc_guard_hz>0 のとき、チューナー中心 ± dc_guard_hz(Hz) を「窓の主役」候補から
    除外して測る（除外後の最強＝次に強い本物を拾う）。判定は取得IQの中心相対
    （0Hz=チューナ中心=DC残留の位置）ゆえ構造的にチューナ相対で、絶対周波数固定の
    スプリアスとは別物。dc_guard_hz<=0 は measure_signal と完全に同一（1ビットも
    変わらない）。戻り値の dict 形は measure_signal と同一。
    """
    return _measure_from_iq(iq, rate, center_hz, dc_guard_hz=float(dc_guard_hz))


def _measure_from_iq(iq: np.ndarray, rate: float, center_hz: float,
                     dc_guard_hz: float) -> dict:
    """measure_signal / measure_signal_dc_guarded の共有本体。

    dc_guard_hz=0（既定）では DC残留ガードの分岐に入らず、従来の measure_signal と
    完全に同一の計算経路をたどる（出力はビット等価）。
    """
    freqs, p = welch_psd(iq, rate)
    if dc_guard_hz > 0:
        # DC残留ガード（opt-in）: チューナ中心 ±dc_guard_hz を候補から外す。freqs は
        #   中心相対（0=チューナ中心=DC残留）なので |freqs|<=dc_guard_hz が除外帯。
        #   除外後に残った freqs/p だけでピーク選択・床・SNR・帯域幅・重心を測るので、
        #   DC残留が最強でも次に強い本物が主役として選ばれる。offset併用時は center_hz が
        #   オフセット適用後の実チューナ中心なので、その中心基準で除外される。
        keep = np.abs(freqs) > dc_guard_hz
        if keep.any():                       # ガード外に測れるビンが残る → そこだけで測る
            freqs, p = freqs[keep], p[keep]
        # keep が全 False（ガードが取得帯域全体を覆う退化設定）は除外せず素通り
        #   （例外を出さない）。実運用外の値なので従来測定へフォールバック。
    floor = float(np.percentile(p, 20))      # ロバストな床（信号占有が高くても効く）
    peak_db = float(p.max())
    snr = peak_db - floor

    mask = p > (floor + 6.0)
    occupied_frac = float(mask.mean())
    if mask.any():
        active = freqs[mask]
        # 帯域幅は「活性ビンの最外殻の差(max-min)」ではなく、主ピーク(最大パワーの
        # ビン)を含む連続活性ランの幅とする。max-min だと窓内に離れた2信号がある
        # とき間の不活性ノイズごと跨いで過大化する（占有僅かでも ~10MHz 等）。ピーク
        # から両隣へ mask が True の間だけ伸ばし、不活性ビンで止めることで主ピーク側
        # 1信号の実幅に収める。これは対症療法で、根治(複数信号を各々分離して測る)は
        # 返り値の形を変える設計判断＝task#6。中心(offset)は従来どおり全活性ビンの重心。
        peak_i = int(np.argmax(p))            # mask.any() のとき必ず mask[peak_i]=True
        lo_i = hi_i = peak_i
        while lo_i - 1 >= 0 and mask[lo_i - 1]:
            lo_i -= 1
        while hi_i + 1 < mask.size and mask[hi_i + 1]:
            hi_i += 1
        bw = float(freqs[hi_i] - freqs[lo_i])
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
# DCオフセット除去（DCスパイク除去 / DC offset correction）
# ---------------------------------------------------------------------------
def remove_dc(iq: np.ndarray, rate: float | None = None,
              corner_hz: float = 10e3) -> np.ndarray:
    """DCオフセット/低周波ドリフトを除去し、取得帯域中央のDCスパイクを平坦化する。

    ゼロIF受信機(HackRF等)は信号全体にDCオフセットを乗せ、取得帯域のちょうど中央
    (オフセット0Hz)に本物ではない細い線(DCスパイク)を出す。これを「捨てる」のでは
    なく信号から除く DC offset correction。

    - rate=None: 複素平均(定常DC)のみを各サンプルから引く軽量版。中央外にオフセット
      した信号・帯域内ノイズに対して平均は ~0 なので本物は壊さないが、実機のDCは
      取得中に**時間変動(LO漏れドリフト)**するため、静的平均では中央に残留スパイク
      (+数dB)が残ることがある。
    - rate 指定: 移動平均(窓長 win = rate / corner_hz)を引くハイパスで、時間変動する
      DC にも追従して除去する。コーナー周波数 corner_hz は spec の周波数ビン幅より
      十分細かく取るので、中央にスパイク(突出)もノッチ(へこみ)も残さず平坦になる。
      実機の受信入口(HackRFBackend)はこちらを使う。

    複素平均は complex128 で蓄積してから引く（complex64 のまま大きな配列を足すと
    丸め誤差で引き切れないことがある）。空配列はそのまま返す。
    """
    iq = np.asarray(iq, dtype=np.complex64)
    if iq.size == 0:
        return iq
    if rate is None:
        mean = iq.mean(dtype=np.complex128)
        return (iq - mean).astype(np.complex64)
    # 時間変動DCに追従: 窓長 win の移動平均(局所DC)を引くハイパス。
    win = max(2, int(round(float(rate) / max(1.0, float(corner_hz)))))
    if win >= iq.size:           # 窓が取得長以上の短い取得 → 静的平均にフォールバック
        mean = iq.mean(dtype=np.complex128)
        return (iq - mean).astype(np.complex64)
    x = iq.astype(np.complex128)
    n = x.size
    half = win // 2
    csum = np.concatenate(([0.0 + 0.0j], np.cumsum(x)))   # 累積和で O(n) 移動平均
    i = np.arange(n)
    lo = np.clip(i - half, 0, n)
    hi = np.clip(i - half + win, 0, n)
    ma = (csum[hi] - csum[lo]) / (hi - lo)                # 各点の局所平均(端は窓を縮める)
    return (x - ma).astype(np.complex64)


# ---------------------------------------------------------------------------
# DCスパイク（DCオフセット由来の中央スパイク）測定
# ---------------------------------------------------------------------------
def dc_spike_metrics(iq: np.ndarray, rate: float,
                     dc_band_hz: float = 60e3, side_hz: float = 0.8e6) -> dict:
    """取得帯域の中央(DC, オフセット0Hz)に集中した細いスパイクの強さを測る。

    ゼロIF受信機(HackRF等)は DC オフセットにより、中心周波数のちょうど中央に
    本物の電波ではない細いスパイク(DCスパイク)を出す。これを 1 取得 IQ から測る。

    中央バンド(|offset| <= dc_band_hz)のピーク電力と、その両脇のリング
    (dc_band_hz < |offset| <= side_hz)の電力中央値を比べ、

        dc_excess_db = dc_peak_db - side_med_db

    を返す。中央だけ突出した細い線(DCスパイク)では大きく、帯域を広く埋める信号
    (中央も脇も持ち上がる)・中央からオフセットした信号・無信号(脇と同程度)では
    小さい。「中央集中かつ細い」を 1 指標で捉える(脇まで広がる=細くない なら脇が
    上がって excess が縮む)。時間不変性は観測間で本指標のばらつきを見て別途判定する。

    returns: dict(dc_excess_db, dc_peak_db, side_med_db)
    """
    freqs, p = welch_psd(iq, rate)
    af = np.abs(freqs)
    dc_mask = af <= dc_band_hz
    side_mask = (af > dc_band_hz) & (af <= side_hz)
    if not dc_mask.any() or not side_mask.any():
        # バンド幅が分解能に対して狭すぎる等で測れない → 中立(0)を返す
        return dict(dc_excess_db=0.0, dc_peak_db=float(p.max()),
                    side_med_db=float(np.median(p)))
    dc_peak = float(p[dc_mask].max())
    side_med = float(np.median(p[side_mask]))
    return dict(dc_excess_db=dc_peak - side_med,
                dc_peak_db=dc_peak, side_med_db=side_med)


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
