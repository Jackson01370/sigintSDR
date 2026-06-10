"""cnntrain: 期待対応表（用途軸→方式軸）と (B)バンドプラン文脈表の **単一の真実**。

ここは torch 非依存・numpy 非依存の純データ + 純関数の置き場。M1.5 probe と M3 の
classify 監査（cnntrain.audit）が **同じ表をここから import** する（二重定義禁止）。

軸の違い（最重要）:
  * CNN のクラスは **方式軸**（wideband-ofdm / narrowband-burst / cw-tone /
    pulse-radar / noise-only ＝ 見え方）。
  * 実データのラベルは **用途軸**（WiFi / BLE / Zigbee ＝ 用途）。
  * 両者の照合に「期待対応表（用途→期待方式クラス集合）」を使う。これは [仮説]。
    照合一致は **accuracy（精度）ではない**（軸が違う照合）。

probe.py からの移設元: 旧 cnntrain/probe.py の ExpectedRow / EXPECTED_REAL /
EXPECTED_DISCLAIMER / match_expected（挙動はバイト等価に保つ）。probe は本モジュール
から再 import する。infer（torch を引く）には依存しないため classify から安全に読める。
"""
from __future__ import annotations

from dataclasses import dataclass


# ===========================================================================
# (1) 期待対応表（用途軸ラベル → 期待される方式軸クラス集合）  ★ [仮説] ★
# ===========================================================================
# これは「地図上の仮説」である。画像確認と将来の人間レビューで更新する前提。
# 一致は accuracy ではない（用途軸 vs 方式軸の照合）。集合なのは、1 つの用途が
# 複数の見え方を取りうるため（帯域幅・バースト性で分かれる）。
@dataclass(frozen=True)
class ExpectedRow:
    keywords: tuple        # ラベル(小文字)に対する部分一致キーワード（いずれか一致で採用）
    expect: frozenset      # 期待される方式クラス（spec/方式軸）の集合
    rationale: str         # この対応の根拠（[仮説]）


# captures/ に実在する全ラベルをカバーする:
#   "BLE/Bluetooth (adv?)" / "WiFi (2.4GHz, 20/40MHz)" / "Zigbee/独自2.4G"
EXPECTED_REAL: list[ExpectedRow] = [
    ExpectedRow(
        ("ble", "bluetooth"),
        frozenset({"narrowband-burst"}),
        "[仮説] BLE adv は ~2MHz 以下の狭帯域バースト → narrowband-burst を期待。"
        "ただし保存は約13msの切り取りで、まばらなバーストが窓に未着なら画像は"
        "ノイズのみ → noise-only が出うる（=13msくじ引きの証拠としてカウント）。"),
    ExpectedRow(
        ("wifi",),
        frozenset({"wideband-ofdm"}),
        "[仮説] WiFi 2.4GHz 20/40MHz は OFDM の広帯域ブロック → wideband-ofdm を期待。"
        "実画像が細線寄りなら cw-tone が出る可能性もあり（ラベル監査の観点）。"),
    ExpectedRow(
        ("zigbee", "独自"),
        frozenset({"narrowband-burst", "wideband-ofdm"}),
        "[仮説] Zigbee は ~2MHz O-QPSK（狭め）だが『独自2.4G』が混在しうるため、"
        "narrowband-burst と wideband-ofdm の集合で受ける。"),
]

EXPECTED_DISCLAIMER = (
    "期待対応表は [仮説]（用途軸→方式軸の地図）。画像確認・人間レビューで更新する。"
    "照合の一致は accuracy ではない。")


def match_expected(label: str | None,
                   table: list[ExpectedRow] | None = None) -> tuple[frozenset | None, str]:
    """ラベル → (期待方式クラス集合, 根拠)。未対応は (None, '')（=unmapped）。"""
    table = EXPECTED_REAL if table is None else table
    lab = (label or "").lower()
    for row in table:
        if any(k.lower() in lab for k in row.keywords):
            return row.expect, row.rationale
    return None, ""


# ===========================================================================
# (2) バンドプラン文脈表（(B) 説明可能な不整合 用）  ★ データとして分離 ★
# ===========================================================================
# 期待集合に無い CNN 方式でも、バンドプラン上の文脈で「説明がつく」不整合を
# データで列挙する。最小実装は 1 行から開始（拡張しやすく）。
#   起点ケース: 2.4GHz ISM 帯 × ルール=WiFi × CNN=pulse-radar
#               → 周期的な顔をする WiFi（ビーコン/ACK 列・混雑チャネル）。
#   2026-06-10 の実データ確認（画像 + CubicSDR）で「本物の WiFi」と確定した教訓。
#   2.4GHz ISM 帯にレーダはまずいない、という「周波数の文脈」が最終判定に効く。
@dataclass(frozen=True)
class ContextRule:
    freq_lo_hz: float      # この中心周波数範囲で成立（[lo, hi] 内包）
    freq_hi_hz: float
    rule_keywords: tuple   # ルールラベル(小文字)への部分一致キーワード（いずれか一致）
    cnn_class: str         # 説明対象の CNN 方式クラス（厳密一致）
    rationale: str         # 文脈による説明（来歴・notes に残す）


CONTEXT_RULES: list[ContextRule] = [
    ContextRule(
        2400.0e6, 2500.0e6, ("wifi",), "pulse-radar",
        "[文脈] 2.4GHz ISM 帯にレーダは通常不在。ルール=WiFi × CNN=pulse-radar は、"
        "周期的な顔をする WiFi（ビーコン/ACK 列・飽和級の混雑チャネル）と解釈する。"
        "（2026-06-10 の実データ確認で確定）。ラベル維持・確信度を微減。"),
]


def match_context(center_hz: float | None, rule_label: str | None,
                  cnn_class: str,
                  rules: list[ContextRule] | None = None) -> ContextRule | None:
    """(中心周波数, ルールラベル, CNN方式) → 説明可能なら ContextRule、無ければ None。

    周波数文脈が必須（center_hz=None は不一致扱い）。最初に一致した行を返す。
    """
    rules = CONTEXT_RULES if rules is None else rules
    if center_hz is None:
        return None
    lab = (rule_label or "").lower()
    for rule in rules:
        if rule.cnn_class != cnn_class:
            continue
        if not any(k.lower() in lab for k in rule.rule_keywords):
            continue
        if rule.freq_lo_hz <= center_hz <= rule.freq_hi_hz:
            return rule
    return None
