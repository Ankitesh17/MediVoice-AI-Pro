"""LLM adapter factory.

`LLM_PROVIDER` selects the implementation:
- "ollama" (default) — local via Ollama, no key, latency-stable. Needs Ollama
  running (http://localhost:11434).
- "kimi"   — Moonshot Kimi (OpenAI-compatible), needs `MOONSHOT_API_KEY`.
- "claude" — Anthropic SDK, needs `ANTHROPIC_API_KEY`.
"""

from __future__ import annotations

import os

from .base import LLMAdapter, Message

__all__ = [
    "LLMAdapter",
    "Message",
    "create_llm_adapter",
    "DEFAULT_SYSTEM_PROMPT",
    "LANGUAGES",
    "system_prompt_for",
]

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, friendly voice assistant. Keep replies short and "
    "conversational — one or two sentences — since they will be spoken aloud. "
    "Avoid markdown, lists, and code blocks unless explicitly asked."
)

# Languages offered in translate mode (code -> display name for the prompt).
LANGUAGES = {
    "en": "English",
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
}


def system_prompt_for(mode: str, source: str = "ja", target: str = "en") -> str:
    """Pick the system prompt for the requested mode.

    - "talk": the conversational assistant (default).
    - "translate": a simultaneous interpreter that returns ONLY the translation
      of the user's speech from `source` into `target`.
    """
    if mode == "translate":
        src = LANGUAGES.get(source, "the source language")
        tgt = LANGUAGES.get(target, "the target language")
        return (
            f"You are a simultaneous interpreter. Translate the user's {src} "
            f"into natural, fluent {tgt}. Output ONLY the {tgt} translation — "
            "no explanations, no notes, no quotes, no commentary. If the input "
            "is already in {tgt}, repeat it unchanged. /no_think"
        ).replace("{tgt}", tgt)
    return DEFAULT_SYSTEM_PROMPT


def create_llm_adapter(provider: str | None = None) -> LLMAdapter:
    name = (provider or os.getenv("LLM_PROVIDER") or "ollama").lower()
    if name == "ollama":
        from .ollama import OllamaLLM

        return OllamaLLM()
    if name == "kimi":
        from .kimi import KimiLLM

        return KimiLLM()
    if name == "claude":
        from .claude import ClaudeLLM

        return ClaudeLLM()
    raise ValueError(
        f"unknown LLM_PROVIDER: {name!r} (expected 'ollama', 'kimi', or 'claude')"
    )
