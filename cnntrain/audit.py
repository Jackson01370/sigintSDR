"""cnntrain (M3): ルール × CNN × バンドプラン文脈の整合チェック（監査役）。

3段分類器の 2 段目として接続する CNN は「拒否権」ではなく **「監査役」**である。
ルール判定（用途ラベル＋確信度）と CNN 所見（方式クラス＋確信度）を、期待対応表
（cnntrain.expected に一元化）で突き合わせ、確信度を調整する純関数を提供する。

判定 (A/B/C)（作業指示の合意済み設計。逸脱禁止・係数は最小実装の固定値）:
  (A) 整合         : CNN クラス ∈ 期待集合
                     → 確信度 +0.10（上限 0.95）。ラベル維持。
  (B) 説明可能な不整合: CNN クラス ∉ 期待集合 だが、バンドプラン文脈で説明がつく
                     （例: 2.4GHz ISM × ルール=WiFi × CNN=pulse-radar →周期的WiFi）
                     → ラベル維持、確信度 −0.05（下限 0.05）。
  (C) 説明不能な不整合: 上記以外の不整合
                     → 確信度を min(rule_conf, 1 − cnn_conf) に引き下げ（保守的）。
                     → その結果 0.7 未満なら用途 = Unknown（候補つき）。
  (unmapped)       : 期待対応表に用途が無い（CNN は判定材料を持たない）
                     → 何も調整しない（所見は記録のみ）。

本モジュールは **torch 非依存**（cnntrain.expected の純データに依存するのみ）。
classify から遅延 import される（OFF 運用・torch 無し環境を壊さない）。
係数（+0.10 / −0.05 / min式 / 0.7境界）は **最小実装の固定値**。根拠なき
チューニングは禁止（妥当性検証は将来データで行う）。
"""
from __future__ import annotations

from dataclasses import dataclass

from cnntrain.expected import match_expected, match_context


# --- 係数（最小実装の固定値。根拠なき変更・チューニングループ禁止）---
CONF_BONUS_A = 0.10        # (A) 整合 → 確信度に加算
CONF_CAP_A = 0.95         # (A) の上限
CONF_PENALTY_B = 0.05     # (B) 説明可能な不整合 → 確信度から減算
CONF_FLOOR_B = 0.05       # (B) の下限
UNKNOWN_THRESHOLD = 0.70  # (C) でこの値未満なら用途 = Unknown（候補つき）

# --- verdict 文字列（SigMF 来歴に残す。作業指示の表記に合わせる）---
VERDICT_A = "A-consistent"
VERDICT_B = "B-context-explained"
VERDICT_C = "C-conflict"
VERDICT_UNMAPPED = "unmapped"


@dataclass(frozen=True)
class AuditDecision:
    """整合チェックの結果（純データ。classify が ClassResult へ反映する）。"""
    verdict: str             # VERDICT_A / _B / _C / _UNMAPPED
    conf_after: float        # 調整後の確信度
    to_unknown: bool         # 用途を Unknown（候補つき）に落とすか（(C) かつ <0.7）
    rationale: str           # この判定の根拠（来歴・notes 用）
    expected: list | None    # 期待方式クラス集合（sorted）。None=unmapped
    cnn_class: str           # CNN の方式クラス（所見）
    cnn_conf: float          # CNN の確信度（softmax）


def audit(rule_label: str | None, rule_conf: float,
          cnn_class: str, cnn_conf: float,
          *, center_hz: float | None = None) -> AuditDecision:
    """ルール結果 × CNN 所見 × バンドプラン文脈 → AuditDecision（純関数）。

    引数:
      rule_label : ルール判定の用途ラベル
      rule_conf  : ルール判定の確信度 [0,1]
      cnn_class  : CNN の方式クラス（cnntrain.classes の語彙）
      cnn_conf   : CNN の softmax 確信度 [0,1]
      center_hz  : 中心周波数（(B) 文脈判定に必須。None なら文脈は不一致扱い）
    """
    expect, _exp_note = match_expected(rule_label)

    # (unmapped) 期待対応表に無い用途 → 監査役は黙る（所見は呼び出し側が記録）。
    if expect is None:
        return AuditDecision(
            VERDICT_UNMAPPED, rule_conf, False,
            "期待対応表に未対応の用途ラベル（CNN 所見は記録のみ・確信度は不変）。",
            None, cnn_class, cnn_conf)

    # (A) 整合: CNN クラス ∈ 期待集合 → +0.10（上限 0.95）。
    if cnn_class in expect:
        conf_after = min(rule_conf + CONF_BONUS_A, CONF_CAP_A)
        return AuditDecision(
            VERDICT_A, conf_after, False,
            f"CNN 方式={cnn_class} は期待集合 {sorted(expect)} に整合。確信度 +"
            f"{CONF_BONUS_A:.2f}（上限 {CONF_CAP_A:.2f}）。",
            sorted(expect), cnn_class, cnn_conf)

    # 不整合: バンドプラン文脈で説明できるか（(B)）。
    ctx = match_context(center_hz, rule_label, cnn_class)
    if ctx is not None:
        conf_after = max(rule_conf - CONF_PENALTY_B, CONF_FLOOR_B)
        return AuditDecision(
            VERDICT_B, conf_after, False, ctx.rationale,
            sorted(expect), cnn_class, cnn_conf)

    # (C) 説明不能な不整合 → 保守的に min(rule_conf, 1−cnn_conf)。<0.7 で Unknown。
    conf_after = min(rule_conf, 1.0 - cnn_conf)
    to_unknown = conf_after < UNKNOWN_THRESHOLD
    return AuditDecision(
        VERDICT_C, conf_after, to_unknown,
        f"CNN 方式={cnn_class} は期待集合 {sorted(expect)} と不整合・文脈説明なし。"
        f"確信度を min(rule={rule_conf:.2f}, 1−cnn={1.0 - cnn_conf:.2f})="
        f"{conf_after:.2f} に引き下げ"
        + ("（<0.7 → 用途=Unknown 候補つき）。" if to_unknown else "。"),
        sorted(expect), cnn_class, cnn_conf)
