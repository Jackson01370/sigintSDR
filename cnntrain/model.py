"""cnntrain (3/6): 軽量 CNN（CPU 前提）。

自作の数層 CNN。入力は正準表現 [B,1,256,256]（spec.render 由来）、出力は
クラス logits [B,n_classes]。火入れ用に小さく速く（1 run が CPU で数分規模）。
GPU 前提のコードは置かない。
"""
from __future__ import annotations

import torch.nn as nn


class SmallSpecCNN(nn.Module):
    """4 conv ブロック + 2 層 FC。256→128→64→32→(adaptive 4x4) と段階的に縮小。

    パラメータは ~3万程度に抑え、CPU で軽快に回る規模にしている。
    """

    def __init__(self, n_classes: int, in_ch: int = 1):
        super().__init__()
        self.n_classes = int(n_classes)
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 8, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                   # 256 -> 128
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                   # 128 -> 64
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                   # 64 -> 32
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),                           # -> [B,32,4,4]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, self.n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(n_classes: int, in_ch: int = 1) -> SmallSpecCNN:
    return SmallSpecCNN(n_classes=n_classes, in_ch=in_ch)
