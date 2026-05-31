"""LLM Vision 用のスペクトログラム画像レンダ。

- PNG パスが渡された場合: そのファイルを bytes として読み込み、必要なら縮小。
- IQ から作る場合: `spec.render` の正準表現を viridis で PNG 化（matplotlib 要）。
- いずれもアップロード前に長辺 `max_side` 以下に縮小（帯域節約）。

依存:
- matplotlib (任意, IQ→PNG 経路で必要)
- PIL/Pillow (任意, 縮小ができないときは原画像をそのまま返す)
"""
from __future__ import annotations
import io
import os
from typing import Optional

import numpy as np


def _resize_png_bytes(png: bytes, max_side: int) -> bytes:
    """Pillow があれば長辺 max_side まで縮小。失敗時は原本を返す。"""
    if not png:
        return png
    try:
        from PIL import Image
    except Exception:
        return png
    try:
        im = Image.open(io.BytesIO(png))
        im.load()
        w, h = im.size
        if max(w, h) <= max_side:
            return png
        if w >= h:
            nw, nh = max_side, max(1, int(round(h * max_side / w)))
        else:
            nh, nw = max_side, max(1, int(round(w * max_side / h)))
        im = im.convert("RGB").resize((nw, nh), Image.LANCZOS)
        out = io.BytesIO()
        im.save(out, format="PNG", optimize=True)
        return out.getvalue()
    except Exception:
        return png


def load_png(path: str, max_side: int = 768) -> bytes | None:
    """ディスク上の PNG を読み込み、必要なら縮小。"""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError:
        return None
    return _resize_png_bytes(raw, max_side)


def render_iq_to_png(iq: np.ndarray, rate: float, center_hz: float,
                     duration_ms: Optional[float] = None,
                     max_side: int = 768) -> bytes | None:
    """IQ から PNG bytes を生成（matplotlib が無ければ None）。

    `spec.render` の正準 [256,256] 正規化画像を使うため、表現スケールは契約と一致。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    try:
        from spec import render, CAPTURE_RATE_HZ  # noqa: F401
    except Exception:
        return None

    try:
        img = render(np.asarray(iq, dtype=np.complex64), rate=rate)
    except Exception:
        return None

    # 軸ラベル: 中心からの ± rate/2 / 経過時間
    span_mhz = rate / 1e6
    if duration_ms is None and rate > 0:
        duration_ms = (len(iq) / rate) * 1e3
    extent = [(center_hz / 1e6) - span_mhz / 2,
              (center_hz / 1e6) + span_mhz / 2,
              duration_ms or 0.0, 0.0]

    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    ax.imshow(img.T, aspect="auto", cmap="viridis",
              extent=extent, origin="upper", vmin=0.0, vmax=1.0)
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Time [ms]")
    ax.set_title(f"Spectrogram @ {center_hz/1e6:.2f} MHz")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return _resize_png_bytes(buf.getvalue(), max_side)


def ensure_png(*, path: str | None = None,
               iq: np.ndarray | None = None,
               rate: float | None = None,
               center_hz: float | None = None,
               max_side: int = 768) -> bytes | None:
    """PNG bytes を返す。path 優先、無ければ IQ から生成。失敗で None。"""
    if path and os.path.exists(path):
        b = load_png(path, max_side=max_side)
        if b:
            return b
    if iq is not None and rate is not None and center_hz is not None:
        return render_iq_to_png(iq, rate, center_hz, max_side=max_side)
    return None
