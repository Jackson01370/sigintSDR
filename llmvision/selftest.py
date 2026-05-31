"""ネットワーク不要のセルフテスト。

確認項目:
  1. プロバイダ未設定時の `llm_classify()` は None を返す（graceful degradation）。
  2. `parse_response()` がフェンス付き応答からも JSON を抽出できる。
  3. `build_user_text()` がカタログ/バンド/ルール所見を含む文字列を生成。
  4. `classify.classify()` がパッケージ呼び出し後も既存挙動を壊さない。
  5. モック応答からの `ClassResult` 整形（method="llm" になる）。

    python -m llmvision.selftest
"""
from __future__ import annotations
import os
import sys


def _ok(msg: str) -> None:
    print(f"  OK  {msg}")


def _fail(msg: str) -> None:
    print(f"  NG  {msg}")
    raise SystemExit(1)


def test_graceful_no_provider() -> None:
    # 環境変数を一旦退避してプロバイダ無効化
    saved = {k: os.environ.pop(k, None) for k in (
        "SIGSCAN_LLM_PROVIDER", "GEMINI_API_KEY", "GOOGLE_API_KEY",
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY")}
    try:
        from llmvision import available_provider, llm_classify
        if available_provider() is not None:
            _fail("available_provider should be None without keys")
        if llm_classify("nonexistent.png") is not None:
            _fail("llm_classify should return None without provider")
        _ok("graceful degradation when provider/keys absent")
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_parse_response() -> None:
    from llmvision import parse_response
    raw_plain = '{"label":"WiFi","confidence":0.78,"notes":"OFDM ブロック"}'
    raw_fenced = (
        "```json\n"
        '{"label": "5G NR n78", "confidence": 0.62, '
        '"candidates": ["LTE B7"], "notes": "TDD 縞", "rationale": "100MHz 矩形"}\n'
        "```")
    raw_noisy = (
        "判定結果:\n"
        '{"label":"BLE","confidence":0.4}\n'
        "以上です。")
    for name, blob in (("plain", raw_plain), ("fenced", raw_fenced), ("noisy", raw_noisy)):
        obj = parse_response(blob)
        if obj is None or "label" not in obj:
            _fail(f"parse_response failed for {name}")
    if parse_response("これは JSON ではない応答です") is not None:
        _fail("parse_response should reject non-JSON")
    if parse_response("") is not None:
        _fail("parse_response should reject empty")
    _ok("parse_response handles plain / fenced / noisy / invalid")


def test_build_user_text() -> None:
    from llmvision import PromptContext, build_user_text
    ctx = PromptContext(center_hz=2.437e9, bw_hz=20e6, snr_db=22.0,
                        occupied_frac=0.6,
                        band_name="ISM 2.4G (WiFi/BT)",
                        band_hint="WiFi 20/40MHz・BLE・Zigbee",
                        rule_label="WiFi (2.4GHz, 20/40MHz)",
                        rule_confidence=0.40, rule_notes="OFDM",
                        rule_candidates=["BLE", "Zigbee"],
                        rep_summary={"nfft": 512, "hop": 256,
                                     "rate_hz": 20_000_000, "img": [256, 256]})
    text = build_user_text(ctx)
    for needle in ("2437.000 MHz", "ISM 2.4G", "WiFi", "JSON only", "candidates"):
        if needle not in text:
            _fail(f"build_user_text missing: {needle}")
    _ok("build_user_text contains band / rule / catalog / output spec")


def test_classify_orchestrator_unchanged() -> None:
    """ルール段で 0.85 以上なら LLM 段は呼ばれず、シグネチャも保持される。"""
    from config import BAND_PLAN
    from classify import classify, llm_classify
    # 確実に高信頼で決まる例: 2.1 GHz B1 LTE 帯 / 10 MHz / 高 SNR
    measurement = dict(center_hz=2140e6, bw_hz=10e6, snr_db=30.0,
                       peak_db=-30.0, noise_floor_db=-60.0, occupied_frac=0.6)
    r = classify(measurement, BAND_PLAN, spectrogram_db=None, png_path=None)
    if r.method != "rule" or r.confidence < 0.5:
        _fail(f"high-conf rule case fell through: {r}")
    # 後方互換: PNG パスだけで呼べる（プロバイダ無しなので None）
    if llm_classify("nonexistent.png") is not None:
        _fail("legacy llm_classify(path) should still return None gracefully")
    _ok("classify() rule path intact + llm_classify legacy signature OK")


def test_result_payload_mapping() -> None:
    from llmvision.core import _result_from_payload
    from classify import UNKNOWN, NOISE
    r = _result_from_payload(
        {"label": "WiFi (2.4GHz)", "confidence": 0.83,
         "candidates": ["BLE", "Zigbee"],
         "notes": "OFDM 矩形", "rationale": "中央キャリア無し"},
        latency_s=0.42, provider="gemini", model="gemini-2.5-flash")
    assert r.method == "llm" and r.label.startswith("WiFi")
    assert 0.0 <= r.confidence <= 1.0
    assert "BLE" in r.candidates
    # noise / unknown 統一
    rn = _result_from_payload({"label": "Noise floor variation", "confidence": 0.1},
                              0.1, "x", "y")
    ru = _result_from_payload({"label": "unknown signal", "confidence": 0.2},
                              0.1, "x", "y")
    if rn.label != NOISE:
        _fail("noise label not normalized")
    if ru.label != UNKNOWN:
        _fail("unknown label not normalized")
    # 異常値の clamp
    rb = _result_from_payload({"label": "X", "confidence": 9.9}, 0.0, "x", "y")
    if rb.confidence != 1.0:
        _fail("confidence not clamped to 1.0")
    _ok("payload → ClassResult mapping (method=llm, clamping, noise/unknown)")


def main() -> int:
    print("llmvision selftest (offline, no network)")
    test_graceful_no_provider()
    test_parse_response()
    test_build_user_text()
    test_classify_orchestrator_unchanged()
    test_result_payload_mapping()
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
