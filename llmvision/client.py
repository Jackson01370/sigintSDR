"""LLM Vision プロバイダ抽象（Gemini / Anthropic / OpenAI 互換）。

依存追加を避けるため標準ライブラリ (`urllib`) のみで実装。
プロバイダ・モデル・APIキー・タイムアウトは環境変数で切り替える。

| 環境変数                  | 既定                                 | 説明                       |
|---------------------------|--------------------------------------|----------------------------|
| `SIGSCAN_LLM_PROVIDER`    | (空)                                 | gemini / anthropic / openai|
| `SIGSCAN_LLM_MODEL`       | (プロバイダごとの既定)               | モデル名                   |
| `SIGSCAN_LLM_TIMEOUT`     | 30                                   | HTTP タイムアウト秒        |
| `SIGSCAN_LLM_MAX_TOKENS`  | 512                                  | 応答上限トークン           |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` |                            | Gemini 用 API キー         |
| `ANTHROPIC_API_KEY`       |                                      | Anthropic 用 API キー      |
| `OPENAI_API_KEY`          |                                      | OpenAI 用 API キー         |

provider が未設定 / キーが無い場合は `available_provider()` が None を返し、
呼び出し側は graceful degradation で None 応答を返す。
"""
from __future__ import annotations
import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 共通レスポンス型
# ---------------------------------------------------------------------------
@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_s: float
    raw: dict | None = None


# ---------------------------------------------------------------------------
# プロバイダ既定
# ---------------------------------------------------------------------------
DEFAULT_MODELS = {
    "gemini":    "gemini-2.5-flash",
    "anthropic": "claude-haiku-4-5-20251001",
    "openai":    "gpt-4o-mini",
}


def _env(*names: str) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v.strip()
    return None


def available_provider() -> tuple[str, str] | None:
    """環境変数からプロバイダとモデルを解決。利用不可なら None。"""
    prov = (os.environ.get("SIGSCAN_LLM_PROVIDER") or "").strip().lower()
    if not prov:
        # キーから自動推測
        if _env("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            prov = "gemini"
        elif _env("ANTHROPIC_API_KEY"):
            prov = "anthropic"
        elif _env("OPENAI_API_KEY"):
            prov = "openai"
        else:
            return None
    if prov not in DEFAULT_MODELS:
        return None
    if prov == "gemini" and not _env("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        return None
    if prov == "anthropic" and not _env("ANTHROPIC_API_KEY"):
        return None
    if prov == "openai" and not _env("OPENAI_API_KEY"):
        return None
    model = os.environ.get("SIGSCAN_LLM_MODEL") or DEFAULT_MODELS[prov]
    return prov, model


# ---------------------------------------------------------------------------
# HTTP ヘルパ
# ---------------------------------------------------------------------------
def _post_json(url: str, headers: dict, body: dict, timeout: float) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in headers.items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# クライアント本体
# ---------------------------------------------------------------------------
@dataclass
class LLMClient:
    provider: str
    model: str
    timeout_s: float = 30.0
    max_tokens: int = 512
    max_retries: int = 1
    retry_wait_s: float = 1.5

    @classmethod
    def from_env(cls) -> "LLMClient | None":
        sel = available_provider()
        if sel is None:
            return None
        prov, model = sel
        return cls(
            provider=prov,
            model=model,
            timeout_s=float(os.environ.get("SIGSCAN_LLM_TIMEOUT", "30")),
            max_tokens=int(os.environ.get("SIGSCAN_LLM_MAX_TOKENS", "512")),
        )

    # ------- 公開 API -------
    def vision_classify(self, png_bytes: bytes, system: str,
                        user_text: str) -> LLMResponse | None:
        """画像 + テキストを送って文字列応答を取得。失敗時は None。"""
        for attempt in range(self.max_retries + 1):
            try:
                t0 = time.monotonic()
                if self.provider == "gemini":
                    text, raw = self._call_gemini(png_bytes, system, user_text)
                elif self.provider == "anthropic":
                    text, raw = self._call_anthropic(png_bytes, system, user_text)
                elif self.provider == "openai":
                    text, raw = self._call_openai(png_bytes, system, user_text)
                else:
                    return None
                return LLMResponse(text=text, provider=self.provider,
                                   model=self.model,
                                   latency_s=time.monotonic() - t0, raw=raw)
            except urllib.error.HTTPError as e:
                # 429/5xx は1回だけリトライ
                if attempt < self.max_retries and e.code in (429, 500, 502, 503, 504):
                    time.sleep(self.retry_wait_s)
                    continue
                return None
            except (urllib.error.URLError, TimeoutError, ConnectionError,
                    json.JSONDecodeError, KeyError, ValueError):
                if attempt < self.max_retries:
                    time.sleep(self.retry_wait_s)
                    continue
                return None
        return None

    # ------- プロバイダ別実装 -------
    def _call_gemini(self, png: bytes, system: str, user: str
                     ) -> tuple[str, dict]:
        key = _env("GEMINI_API_KEY", "GOOGLE_API_KEY")
        # endpoint オーバーライド (代理サーバ等)
        base = os.environ.get(
            "SIGSCAN_GEMINI_ENDPOINT",
            "https://generativelanguage.googleapis.com/v1beta/models",
        ).rstrip("/")
        url = f"{base}/{self.model}:generateContent?key={key}"
        b64 = base64.b64encode(png).decode("ascii")
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": user},
                    {"inline_data": {"mime_type": "image/png", "data": b64}},
                ],
            }],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": self.max_tokens,
                "responseMimeType": "application/json",
            },
        }
        raw = _post_json(url, {}, body, self.timeout_s)
        cands = raw.get("candidates") or []
        if not cands:
            raise ValueError("gemini: candidates empty")
        parts = (cands[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)
        return text, raw

    def _call_anthropic(self, png: bytes, system: str, user: str
                        ) -> tuple[str, dict]:
        key = _env("ANTHROPIC_API_KEY")
        base = os.environ.get("SIGSCAN_ANTHROPIC_ENDPOINT",
                              "https://api.anthropic.com/v1/messages")
        b64 = base64.b64encode(png).decode("ascii")
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": user},
                ],
            }],
        }
        headers = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        }
        raw = _post_json(base, headers, body, self.timeout_s)
        content = raw.get("content") or []
        if not content:
            raise ValueError("anthropic: content empty")
        text = "".join(c.get("text", "") for c in content if c.get("type") == "text")
        return text, raw

    def _call_openai(self, png: bytes, system: str, user: str
                     ) -> tuple[str, dict]:
        key = _env("OPENAI_API_KEY")
        base = os.environ.get("SIGSCAN_OPENAI_ENDPOINT",
                              "https://api.openai.com/v1/chat/completions")
        b64 = base64.b64encode(png).decode("ascii")
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": user},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ]},
            ],
        }
        headers = {"Authorization": f"Bearer {key}"}
        raw = _post_json(base, headers, body, self.timeout_s)
        choices = raw.get("choices") or []
        if not choices:
            raise ValueError("openai: choices empty")
        msg = choices[0].get("message") or {}
        text = msg.get("content") or ""
        return text, raw
