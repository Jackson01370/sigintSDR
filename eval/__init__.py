"""eval-harness: 外部学習済みモデルを sigscan の正準表現に配線する評価ハーネス。

M1 は **配線のみ**（モデルのロードと入出力アダプタ）。実機キャプチャに対する
ドメインギャップ測定は後マイルストーン（CONTRACT.md §4）。sim に対する結果は
必ず "synthetic-vs-synthetic（本当のギャップではない）" と明示される。

公開 API:
    loaders : load_torchsig_narrowband / load_qoherent_segmentation /
              load_reference_standin / load_model / ModelUnavailable
    adapters: canonical / to_segmentation_input / to_narrowband_input / adapt_for
    report  : run_report / print_report
"""
from __future__ import annotations

from . import loaders  # noqa: F401
from . import adapters  # noqa: F401
from .loaders import (  # noqa: F401
    ModelUnavailable,
    ModelSpec,
    LoadedModel,
    load_model,
    available_models,
    load_torchsig_narrowband,
    load_torchsig_wideband,
    load_qoherent_segmentation,
    load_reference_standin,
)

__all__ = [
    "loaders",
    "adapters",
    "ModelUnavailable",
    "ModelSpec",
    "LoadedModel",
    "load_model",
    "available_models",
    "load_torchsig_narrowband",
    "load_torchsig_wideband",
    "load_qoherent_segmentation",
    "load_reference_standin",
]
