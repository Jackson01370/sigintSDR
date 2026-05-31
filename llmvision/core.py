"""LLM Vision 段の中核: 画像とバンド文脈を視覚 LLM へ送り `ClassResult` を返す。

- `llm_classify(png_path, measurement=None, bands=None, rule_result=None) -> ClassResult|None`
  `classify.classify()` から低信頼/未知時のみ呼ばれる。
- ネットワーク・APIキー未設定・パース失敗・タイムアウト等いずれの失敗でも
  `None` を返し、上位は **graceful degradation**（ルール段結果を採用）。

設計メモ:
- 「画像 + バンド文脈 + ルール所見 + 参照カタログ」をプロンプトに織り込む。
- 応答は JSON 形式（`prompt.RESPONSE_SCHEMA` 準拠）。
- 送るのは PNG 画像と最小限のメタのみ。**生 IQ は送らない**。
"""
from __future__ import annotations
import os
from typing import Any

from classify import ClassResult, UNKNOWN, NOISE  # type: ignore
from config import Band  # type: ignore

from . import prompt as P
from .client import LLMClient
from .render import ensure_png


# ---------------------------------------------------------------------------
# 文脈構築
# ---------------------------------------------------------------------------
def _match_band(center_hz: float, bands: list[Band] | None) -> Band | None:
    if not bands:
        return None
    best = None
    for b in bands:
        if b.f_lo <= center_hz <= b.f_hi:
            if best is None or b.priority > best.priority:
                best = b
    return best


def build_request_context(measurement: dict | None,
                          bands: list[Band] | None,
                          rule_result: ClassResult | None) -> P.PromptContext:
    """measurement + bands + rule_result から `PromptContext` を組み立てる。"""
    m = measurement or {}
    center = float(m.get("center_hz", 0.0))
    bw = float(m.get("bw_hz", 0.0))
    snr = float(m.get("snr_db", 0.0))
    occ = float(m.get("occupied_frac", 0.0))

    band = _match_band(center, bands)
    band_name = band.name if band else ""
    band_hint = band.hint if band else ""

    try:
        from spec import spec_summary
        rep = spec_summary()
    except Exception:
        rep = None

    return P.PromptContext(
        center_hz=center, bw_hz=bw, snr_db=snr, occupied_frac=occ,
        band_name=band_name, band_hint=band_hint,
        rule_label=rule_result.label if rule_result else "",
        rule_confidence=rule_result.confidence if rule_result else 0.0,
        rule_notes=rule_result.notes if rule_result else "",
        rule_candidates=list(rule_result.candidates) if rule_result else [],
        rep_summary=rep,
    )


# ---------------------------------------------------------------------------
# 応答 → ClassResult
# ---------------------------------------------------------------------------
def _coerce_confidence(v: Any) -> float:
    try:
        c = float(v)
    except (TypeError, ValueError):
        return 0.0
    if c != c:                     # NaN
        return 0.0
    return max(0.0, min(1.0, c))


def _result_from_payload(payload: dict, latency_s: float,
                         provider: str, model: str) -> ClassResult:
    label = str(payload.get("label") or UNKNOWN).strip() or UNKNOWN
    conf = _coerce_confidence(payload.get("confidence"))
    cands_raw = payload.get("candidates") or []
    if isinstance(cands_raw, list):
        candidates = [str(c).strip() for c in cands_raw if str(c).strip()][:6]
    else:
        candidates = []
    notes_bits: list[str] = []
    if payload.get("notes"):
        notes_bits.append(str(payload["notes"]).strip())
    if payload.get("rationale"):
        notes_bits.append("根拠: " + str(payload["rationale"]).strip())
    notes_bits.append(f"[{provider}/{model} {latency_s:.2f}s]")
    notes = " | ".join(b for b in notes_bits if b)

    # ノイズ・未識別判定を統一ラベルに寄せる
    low = label.lower()
    if "noise" in low or "ノイズ" in label or "floor" in low:
        label = NOISE
    elif "unknown" in low or "未識別" in label or "未知" in label:
        label = UNKNOWN

    return ClassResult(label=label, confidence=conf, method="llm",
                       notes=notes, candidates=candidates)


# ---------------------------------------------------------------------------
# 公開エントリポイント
# ---------------------------------------------------------------------------
def llm_classify(spectrogram_png_path: str | None,
                 measurement: dict | None = None,
                 bands: list[Band] | None = None,
                 rule_result: ClassResult | None = None,
                 *,
                 iq=None, rate: float | None = None) -> ClassResult | None:
    """低信頼/未知信号のスペクトログラムを視覚 LLM で識別する。

    Args:
        spectrogram_png_path: 既存 PNG のパス（無くても iq/rate があれば生成）。
        measurement: dsp.measure_signal の戻り（中心周波数・帯域幅・SNR 等）。
        bands: 参照するバンドプラン（config.BAND_PLAN）。
        rule_result: ルール段の結果。文脈としてプロンプトに含める。
        iq, rate: PNG が無い場合に IQ から spec.render 経由でその場生成。

    Returns:
        分類結果（method="llm"）。失敗時は None（graceful degradation）。
    """
    # 1) クライアント解決（環境変数）
    client = LLMClient.from_env()
    if client is None:
        return None

    # 2) 画像準備
    center = float((measurement or {}).get("center_hz", 0.0)) if measurement else 0.0
    png = ensure_png(path=spectrogram_png_path, iq=iq, rate=rate,
                     center_hz=center)
    if not png:
        return None

    # 3) プロンプト
    ctx = build_request_context(measurement, bands, rule_result)
    user_text = P.build_user_text(ctx)

    # 4) 呼び出し
    resp = client.vision_classify(png, P.SYSTEM_PROMPT, user_text)
    if resp is None:
        return None

    # 5) パース → ClassResult
    payload = P.parse_response(resp.text)
    if payload is None:
        if os.environ.get("SIGSCAN_LLM_DEBUG"):
            print("[llmvision] JSON parse failed, raw text head:",
                  (resp.text or "")[:200])
        return None
    return _result_from_payload(payload, resp.latency_s,
                                resp.provider, resp.model)
