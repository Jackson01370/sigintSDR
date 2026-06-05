"""cnntrain (5/6): 推論ヘルパ（将来の classify 接続の **準備**。接続はしない）。

チェックポイントを読み込み、正準画像 1 枚 [256,256] float32 [0,1] を
(クラス名, softmax 確信度) に写す。本 M1 では classify.classify への組み込みは
やらない（継ぎ目保護・次マイルストーン）。ここは独立ヘルパに留める。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

import spec
from cnntrain.model import build_model


@dataclass
class Checkpoint:
    model: torch.nn.Module     # eval モードのモデル
    classes: list[str]         # クラス名一覧（インデックス順）
    meta: dict                 # rep_version / synthetic_only / seed 等


def load_checkpoint(path: str) -> Checkpoint:
    """torch.save した辞書チェックポイントを読み、eval 済みモデルを返す。

    保存物は安全な型のみ（state_dict のテンソル + list/str/int/float の meta）
    なので weights_only ローダで読める。
    """
    blob = torch.load(path, map_location="cpu", weights_only=True)
    class_names = list(blob["classes"])
    model = build_model(n_classes=len(class_names),
                        in_ch=int(blob.get("in_ch", 1)))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return Checkpoint(model=model, classes=class_names,
                      meta=dict(blob.get("meta", {})))


def _to_input_tensor(img) -> torch.Tensor:
    """[256,256] float32 [0,1]（spec.render の出力）→ [1,1,256,256] テンソル。

    spec.render の出力をそのまま入れる前提。ここで追加のリサイズ・正規化はしない
    （正準表現を迂回しない＝作業指示の最重要原則）。形状だけ検証する。
    """
    arr = np.asarray(img, dtype=np.float32)
    if arr.shape != (spec.IMG_FREQ, spec.IMG_TIME):
        raise ValueError(
            f"入力画像の形状が不正: {arr.shape}（期待 "
            f"[{spec.IMG_FREQ},{spec.IMG_TIME}]= spec.render の出力）")
    t = torch.from_numpy(np.ascontiguousarray(arr))
    return t.unsqueeze(0).unsqueeze(0)        # [1,1,256,256]


@torch.no_grad()
def classify_image(ckpt: Checkpoint, img) -> tuple[str, float]:
    """正準画像 1 枚 → (クラス名, 確信度 softmax∈[0,1])。

    img は **spec.render(iq, rate)** が返す [256,256] float32 [0,1]。
    """
    x = _to_input_tensor(img)
    probs = F.softmax(ckpt.model(x), dim=1)[0]
    idx = int(torch.argmax(probs).item())
    return ckpt.classes[idx], float(probs[idx].item())


@torch.no_grad()
def classify_image_topk(ckpt: Checkpoint, img, k: int = 3) -> list[tuple[str, float]]:
    """上位 k クラスの (クラス名, 確信度) を確信度降順で返す。"""
    x = _to_input_tensor(img)
    probs = F.softmax(ckpt.model(x), dim=1)[0]
    k = min(k, len(ckpt.classes))
    vals, idxs = torch.topk(probs, k)
    return [(ckpt.classes[int(i)], float(v))
            for v, i in zip(vals.tolist(), idxs.tolist())]


@torch.no_grad()
def classify_iq(ckpt: Checkpoint, iq, rate: float = spec.CAPTURE_RATE_HZ) -> tuple[str, float]:
    """生 IQ → 凍結 spec.render → classify_image の薄い便宜ラッパ。

    将来 classify 段から CNN を呼ぶ際の経路の **準備**（spec.render を必ず通す）。
    本 M1 では classify.classify には接続しない。
    """
    img = spec.render(iq, rate)
    return classify_image(ckpt, img)
