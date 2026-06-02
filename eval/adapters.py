"""eval-harness (2/3): 入出力アダプタ。

sigscan の正準表現（`spec.render()` の [256,256] float32 in [0,1]）や SigMF の
生 IQ を、各外部モデルの期待入力（サイズ・チャネル・正規化）へ写す。リサイズと
再正規化は **すべてここで吸収** し、6 継ぎ目のシグネチャ（特に spec.render）は
触らない。

原則（CONTRACT.md §1）:
  * 画像ドメインのモデル（qoherent セグメンテーション / torchsig 広帯域）は
    必ず `spec.render()` の正準画像を起点にする（単一の真実）。
  * IQ ドメインのモデル（torchsig 狭帯域）は spec.render が不可逆な画像表現の
    ため、spec.render と同じ生 IQ を直接使う。この差は明示する。
依存は numpy のみ（torch は要らない。torch テンソル化は report 側 / モデル側で）。
"""
from __future__ import annotations

import numpy as np

import spec as _spec
from spec import render as _render          # 単一の真実（再実装しない）
from spec import _resize_bilinear            # numpy 双線形リサイズを再利用

# ImageNet 正規化（torchvision の DeepLabV3 系の既定前処理に合わせる）
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# 正準表現（必ず spec.render を通す）
# ---------------------------------------------------------------------------
def canonical(iq, rate: float = _spec.CAPTURE_RATE_HZ) -> np.ndarray:
    """生 IQ → 正準スペクトログラム [256,256] f32 in [0,1]（spec.render の薄包み）。"""
    return _render(iq, rate)


# ---------------------------------------------------------------------------
# qoherent セグメンテーション: [3,256,256] RGB（ImageNet 正規化）
# ---------------------------------------------------------------------------
def to_segmentation_input(canonical_img: np.ndarray,
                          size: tuple[int, int] = (256, 256),
                          normalize: str = "imagenet") -> np.ndarray:
    """正準画像 → DeepLabv3 入力 [3,H,W]。

    正準は 1ch [0,1]。3ch に複製し、必要なら size にリサイズ、ImageNet 正規化。
    normalize: "imagenet" | "none"。
    """
    img = np.asarray(canonical_img, dtype=np.float32)
    if img.shape != size:
        img = _resize_bilinear(img, size[0], size[1])
    rgb = np.repeat(img[None, :, :], 3, axis=0)        # [3,H,W] in [0,1]
    if normalize == "imagenet":
        rgb = (rgb - _IMAGENET_MEAN[:, None, None]) / _IMAGENET_STD[:, None, None]
    return rgb.astype(np.float32)


# ---------------------------------------------------------------------------
# torchsig 広帯域: [2,512,512] 複素スペクトログラムの代替（注意: 位相は失う）
# ---------------------------------------------------------------------------
def to_wideband_input(canonical_img: np.ndarray,
                      size: tuple[int, int] = (512, 512)) -> np.ndarray:
    """正準画像 → 広帯域モデル入力 [2,H,W]（mag を複製した近似）。

    本来 WBSig53 は **複素** スペクトログラム（実部/虚部）を要求するが、正準表現は
    正規化マグニチュードのみで位相を持たない。ここでは mag を 2ch に複製した
    近似入力を返す（配線確認用）。実測評価では生 IQ から複素スペクトログラムを
    別途作る必要がある（README の移行手順参照）。
    """
    img = np.asarray(canonical_img, dtype=np.float32)
    if img.shape != size:
        img = _resize_bilinear(img, size[0], size[1])
    return np.repeat(img[None, :, :], 2, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# torchsig 狭帯域: [2,4096]（生 IQ の実部/虚部, 単位エネルギー正規化）
# ---------------------------------------------------------------------------
def to_narrowband_input(iq, n: int = 4096) -> np.ndarray:
    """生 IQ → 狭帯域分類器入力 [2,n]（実部/虚部, 単位エネルギー正規化）。

    NOTE: 狭帯域モデルは「チャネライズ済み・単一信号・複素ベースバンド」を想定する。
    ここではキャプチャ全体の先頭 n サンプルを素朴に切り出すだけで、チャネライズや
    エネルギー検出はしていない（配線確認用）。実測評価では dsp 側で単一信号を
    分離してから渡すべき（README の移行手順参照）。
    """
    iq = np.asarray(iq, dtype=np.complex64).ravel()
    if iq.size < n:
        iq = np.concatenate([iq, np.zeros(n - iq.size, dtype=np.complex64)])
    seg = iq[:n]
    energy = float(np.sqrt(np.mean(np.abs(seg) ** 2)) + 1e-12)
    seg = seg / energy
    return np.stack([seg.real, seg.imag], axis=0).astype(np.float32)   # [2,n]


# ---------------------------------------------------------------------------
# family → adapter の自動選択
# ---------------------------------------------------------------------------
def adapt_for(spec_obj, *, iq=None, rate: float = _spec.CAPTURE_RATE_HZ,
              canonical_img: np.ndarray | None = None) -> np.ndarray:
    """ModelSpec.family に応じて正しいアダプタを選び、入力テンソル(numpy)を返す。

    画像 family は canonical_img（無ければ iq から render）を使う。
    IQ family は生 iq を使う。どちらの素材も無ければ ValueError。
    """
    family = getattr(spec_obj, "family", "")
    needs_image = family in ("qoherent-segmentation", "torchsig-wideband") or \
        getattr(spec_obj, "input_kind", "") in ("spectrogram", "rgb-spectrogram")

    if needs_image:
        if canonical_img is None:
            if iq is None:
                raise ValueError("画像 family には canonical_img か iq が必要")
            canonical_img = canonical(iq, rate)
        if family == "torchsig-wideband":
            return to_wideband_input(canonical_img,
                                     size=tuple(spec_obj.input_size))
        return to_segmentation_input(canonical_img,
                                     size=tuple(spec_obj.input_size))

    # IQ family（狭帯域）
    if iq is None:
        raise ValueError("IQ family には生 iq が必要")
    n = int(spec_obj.input_size[0]) if getattr(spec_obj, "input_size", None) \
        else 4096
    return to_narrowband_input(iq, n=n)
