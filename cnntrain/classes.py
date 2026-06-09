"""cnntrain 共有定数（torch 非依存）: クラス定義・SYNTHETIC-ONLY バナー・取得パラメータ。

ここは numpy すら要らない純粋な定数/メタデータ置き場。simgen（合成・torch不要）も
train/infer（torch必要）も同じ真実をここから読む。クラスは **方式軸（見え方）** で
定義する（用途軸ではない）。「CNN は方式（見え方）を学び、用途は周波数等の文脈で
後段が導く」というプロジェクトの確定設計（CONTRACT/作業指示）に沿う。
"""
from __future__ import annotations

import spec  # 凍結契約: 取得レート/REP_VERSION の単一の真実

# 合成 1 サンプルの取得条件（正準表現と一致させる）。
# CAPTURE_RATE_HZ は spec の凍結値（20 MS/s）をそのまま使う。
GEN_RATE_HZ = spec.CAPTURE_RATE_HZ
# 1 サンプルの IQ 長。(N-512)/256+1 ≈ 255 列の STFT になり、spec.render が
# [256,256] にリサイズする。約 3.3 ms（火入れ用に小さく速く）。
GEN_SAMPLES = 65536

# 表現バージョン（チェックポイント・メタに刻む）。
REP_VERSION = spec.SIGSCAN_REP_VERSION

# --- クラス定義（方式軸＝スペクトログラム上の見え方）---
# name: SigMF の core:label / sigscan:true_class に入る正準クラス名。
# look: [256,256] 正準表現（縦=周波数, 横=時間）での「見え方」の根拠。
CLASS_INFO: list[dict] = [
    dict(name="wideband-ofdm",
         look="ほぼ全幅(≈18〜20MHz)・非周期の広帯域縦縞(少数+沈黙/密集クラスタ・実WiFi様)"),
    dict(name="narrowband-burst",
         look="狭帯域(1〜2.5MHz)・短いバースト1〜3発と疎ら(実BLE adv様)"),
    dict(name="cw-tone",
         look="中心外(|off|>0.5MHz)の細い横線=連続波トーン(中心0Hzの細線はDC残留)"),
    dict(name="pulse-radar",
         look="広帯域・厳密に周期的な短い縦縞(パルス列/レーダ風)"),
    dict(name="noise-only",
         look="信号なし・構造のない一様な低レベル背景(ノイズのみ)"),
]

# 正準クラス名一覧（ソート順 = dataset.split が返すクラス順と一致させる）。
CLASSES: list[str] = sorted(c["name"] for c in CLASS_INFO)


def look_of(name: str) -> str:
    """クラス名 → 「見え方」の説明（レポート用）。未知名は空文字。"""
    for c in CLASS_INFO:
        if c["name"] == name:
            return c["look"]
    return ""


# --- SYNTHETIC-ONLY 正直バナー（eval-harness と同じ正直文化）---
# レポート冒頭・チェックポイントメタの双方に入れる。
SYNTHETIC_ONLY_LINES: list[str] = [
    "SYNTHETIC-ONLY: 合成(Sim)データのみで学習・評価。実環境とのギャップは未測定。",
    "ラベルは生成時の真実(ground truth)。ルール分類器の出力は教師に使っていない。",
    "用途はこれが目的ではなく『学習パイプラインが end-to-end で動くこと』の火入れ。",
]

SYNTHETIC_ONLY_TAG = "synthetic-only (sim) / domain gap unmeasured"
