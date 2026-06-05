"""cnntrain: CNN（スペクトログラム画像分類器）の学習パイプライン（M1 火入れ）。

3段分類器（ルール → CNN → LLM）のうち CNN 段を **独立モジュール** として
学習・評価・推論ヘルパまで通す。`classify.classify` への接続は次マイルストーン
（継ぎ目保護のため、本 M1 ではやらない）。

設計の固定点（作業指示の最重要原則）:
  * CNN への入力表現は **必ず凍結 `spec.render(iq, rate)` → [256,256] float32 [0,1]**。
    独自の前処理・リサイズ・正規化は足さない（正準表現が単一の真実）。
  * 合成データの教師は **生成時の真実(ground truth)**。ルール分類器の出力は使わない。
  * 出力（レポート・チェックポイント）に **SYNTHETIC-ONLY** を必ず明記。
  * CPU 前提（torch 2.8.0+cpu / torchvision）。小さく速く。

公開 API（サブモジュールを直接 import して使う）:
    cnntrain.simgen   : generate / main      — Sim データ生成 CLI（torch 不要）
    cnntrain.data     : load_split / SpecDataset  — SigMF→spec.render→テンソル
    cnntrain.model    : SmallSpecCNN          — 軽量CNN
    cnntrain.train    : run_training / main   — 学習 CLI + チェックポイント + ログ
    cnntrain.evaluate : evaluate_model / write_report  — 精度+混同行列（バナー付）
    cnntrain.infer    : load_checkpoint / classify_image  — 推論ヘルパ（接続はしない）

注: torch は data/model/train/evaluate/infer のみが必要。simgen と classes は
torch 非依存（合成だけなら torch 無しでも動く）。よって本 __init__ では torch を
import しない（重い依存を import 時に強制しない）。
"""
from __future__ import annotations

from .classes import (  # noqa: F401  (torch 非依存の共有定数)
    CLASSES,
    CLASS_INFO,
    GEN_RATE_HZ,
    GEN_SAMPLES,
    REP_VERSION,
    SYNTHETIC_ONLY_LINES,
    SYNTHETIC_ONLY_TAG,
    look_of,
)

__all__ = [
    "CLASSES",
    "CLASS_INFO",
    "GEN_RATE_HZ",
    "GEN_SAMPLES",
    "REP_VERSION",
    "SYNTHETIC_ONLY_LINES",
    "SYNTHETIC_ONLY_TAG",
    "look_of",
]
