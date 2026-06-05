"""cnntrain (2/6): SigMF → 凍結 spec.render → テンソル + ラベル。

データの索引化・hw 別分割は既存の `dataset.py`（load_index / Dataset.split）を
**再利用**する。これにより将来 real データ（capture-engine の蓄積）へ展開する際も
同じ経路で読める（split は hw を絶対に混ぜないので sim/real の汚染も防げる）。

入力表現は **必ず凍結 spec.render(iq, rate) → [256,256] float32 [0,1]**。
独自の前処理・リサイズ・正規化は足さない（作業指示の最重要原則）。
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

import sigmf_io
import spec
import dataset as ds_mod
from cnntrain import classes


def load_split(data_dir: str, val_ratio: float = 0.2, seed: int = 0):
    """data_dir の合成 SigMF を索引化し train/val に分割する（再現可能）。

    returns: (train_records, val_records, class_names)
      * class_names は core:label（=合成の真実ラベル）のソート済みユニーク。
      * 分割は dataset.Dataset.split（hw 別・seed 固定）を再利用。本データは
        すべて sim なので sim グループ内で 80/20 に割れる。
    """
    idx = ds_mod.load_index(data_dir).query(hw="sim")
    if len(idx) == 0:
        raise RuntimeError(
            f"合成レコードが見つかりません: {data_dir} "
            f"（先に python -m cnntrain.simgen --out {data_dir} を実行）")
    train, val = idx.split(val_ratio=val_ratio, seed=seed)
    class_names = sorted({r.label for r in idx if r.label})
    return list(train), list(val), class_names


class SpecDataset(TorchDataset):
    """SigMF レコード列 → (x[1,256,256] float32 [0,1], y:int) を返す torch Dataset。

    各サンプルは read_recording → spec.render を通すだけ。spec.render の出力を
    そのまま 1ch 画像として使う（迂回・追加正規化なし）。
    """

    def __init__(self, records, class_names: list[str]):
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        # クラス一覧に無いラベル（想定外）は落とす。
        self.records = [r for r in records if r.label in self.class_to_idx]
        self.class_names = list(class_names)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        # read_recording はロケール既定で meta を開く（凍結契約の符号化に一致）。
        iq, meta = sigmf_io.read_recording(r.path)
        rate = float(meta.get("global", {}).get("core:sample_rate",
                                                spec.CAPTURE_RATE_HZ))
        img = spec.render(iq, rate)                  # [256,256] float32 [0,1]
        x = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))
        x = x.unsqueeze(0)                           # [1,256,256]
        y = self.class_to_idx[r.label]
        return x, y


def true_class_of(meta: dict) -> str | None:
    """meta から真実クラスを取り出す（global sigscan:true_class 優先、無ければ
    annotation の core:label）。検証/デバッグ用。"""
    g = meta.get("global", {})
    if g.get("sigscan:true_class"):
        return str(g["sigscan:true_class"])
    for a in meta.get("annotations", []):
        if a.get("core:label"):
            return str(a["core:label"])
    return None
