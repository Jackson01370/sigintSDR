"""sigscan LLM Vision 段。

低信頼/未知のスペクトログラムを視覚 LLM (Gemini/Claude/OpenAI) に送って
信号識別する。`classify.llm_classify()` から呼ばれる前提。

公開 API:
    llm_classify(png_path, measurement=None, bands=None, rule_result=None)
        → ClassResult | None  (失敗時は None で graceful degradation)
"""
from .core import build_request_context, llm_classify
from .client import LLMClient, LLMResponse, available_provider
from .prompt import (
    PromptContext,
    RESPONSE_SCHEMA,
    SIGNAL_CATALOG,
    SYSTEM_PROMPT,
    build_user_text,
    parse_response,
)

__all__ = [
    "llm_classify",
    "build_request_context",
    "LLMClient",
    "LLMResponse",
    "available_provider",
    "PromptContext",
    "SYSTEM_PROMPT",
    "RESPONSE_SCHEMA",
    "SIGNAL_CATALOG",
    "build_user_text",
    "parse_response",
]
