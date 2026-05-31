"""sigscan データ契約 (1/2): 全コンポーネント共有の正準スペクトログラム表現。

(a)自己収集エンジンと(b)外部モデル評価ハーネスは、必ず本モジュールの
`render()` を通して時間周波数表現を生成すること。ここを単一の真実(single
source of truth)に固定することで、合成・実測・外部データ(WBSig53 等)を
同一スケールで混ぜられる。表現を変えるときは SIGSCAN_REP_VERSION を上げ、
保存済み SigMF の生IQから再レンダする(SigMF は生IQを保持するのでロックされない)。
"""
from __future__ import annotations
import numpy as np

# --- 取得パラメータ（HackRF ドウェルと一致させること） ---
CAPTURE_RATE_HZ = 20_000_000.0      # 瞬時帯域（HackRF 上限付近）

# --- STFT パラメータ ---
SPEC_NFFT = 512
SPEC_HOP = 256                       # 50% オーバーラップ
SPEC_WINDOW = "hann"

# --- 正準画像サイズ（CNN/セグメンテーション入力） ---
IMG_FREQ = 256                       # 周波数軸（高さ）
IMG_TIME = 256                       # 時間軸（幅）

# --- dB 正規化（絶対ゲイン非依存にする） ---
DB_DYN_RANGE = 60.0                  # 床から上 60dB を [0,1] に写像
DB_FLOOR_PCT = 5.0                   # 床推定に使うパーセンタイル

SIGSCAN_REP_VERSION = "1.0"          # 表現仕様バージョン（変更したら再レンダ）


def stft_db(iq, rate: float = CAPTURE_RATE_HZ,
            nfft: int = SPEC_NFFT, hop: int = SPEC_HOP) -> np.ndarray:
    """複素IQ -> dBスペクトログラム [freq, time]（fftshift済み・絶対dB・未正規化）。"""
    iq = np.asarray(iq, dtype=np.complex64)
    if iq.size < nfft:
        iq = np.concatenate([iq, np.zeros(nfft - iq.size, dtype=np.complex64)])
    win = np.hanning(nfft).astype(np.float32)
    cols = []
    for start in range(0, iq.size - nfft + 1, hop):
        seg = iq[start:start + nfft] * win
        spec = np.fft.fftshift(np.fft.fft(seg))
        cols.append(20.0 * np.log10(np.abs(spec) + 1e-12))
    if not cols:
        cols = [np.full(nfft, -240.0, dtype=np.float32)]
    return np.asarray(cols, dtype=np.float32).T   # [freq, time]


def normalize_db(S: np.ndarray, dyn_range: float = DB_DYN_RANGE,
                 floor_pct: float = DB_FLOOR_PCT) -> np.ndarray:
    """床(パーセンタイル)を0、床+dyn_rangeを1にクリップ正規化。絶対ゲイン非依存。"""
    floor = np.percentile(S, floor_pct)
    return np.clip((S - floor) / dyn_range, 0.0, 1.0).astype(np.float32)


def _resize_bilinear(S: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """numpyのみの双線形リサイズ [h,w] -> [out_h,out_w]。"""
    h, w = S.shape
    if (h, w) == (out_h, out_w):
        return S.astype(np.float32)
    ys = np.linspace(0, h - 1, out_h)
    xs = np.linspace(0, w - 1, out_w)
    y0 = np.floor(ys).astype(int); y1 = np.minimum(y0 + 1, h - 1)
    x0 = np.floor(xs).astype(int); x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0)[:, None]; wx = (xs - x0)[None, :]
    top = S[y0][:, x0] * (1 - wx) + S[y0][:, x1] * wx
    bot = S[y1][:, x0] * (1 - wx) + S[y1][:, x1] * wx
    return (top * (1 - wy) + bot * wy).astype(np.float32)


def render(iq, rate: float = CAPTURE_RATE_HZ) -> np.ndarray:
    """正準表現: [IMG_FREQ, IMG_TIME] の float32 in [0,1]。

    収集・評価・学習のすべてが必ずこれを通す。"""
    S = stft_db(iq, rate)
    S = normalize_db(S)
    return _resize_bilinear(S, IMG_FREQ, IMG_TIME)


def spec_summary() -> dict:
    """契約の要約（ログ・メタ埋め込み用）。"""
    return dict(
        rate_hz=CAPTURE_RATE_HZ, nfft=SPEC_NFFT, hop=SPEC_HOP,
        img=[IMG_FREQ, IMG_TIME], dyn_range_db=DB_DYN_RANGE,
        version=SIGSCAN_REP_VERSION,
    )
