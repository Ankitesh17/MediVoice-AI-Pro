"""Local LLM via Ollama's OpenAI-compatible endpoint.

Ollama serves an OpenAI-compatible API at http://localhost:11434/v1, so we reuse
the `openai` async client — same shape as the Kimi adapter, just pointed local.
No API key, no rate limits, fully offline. This is the latency-stable default
when a hosted LLM's response time is unpredictable.

Model is `OLLAMA_MODEL` (default "qwen3:0.6b"). Some local models (e.g. qwen3)
can emit `<think>...</think>` reasoning blocks; we strip those so only the spoken
answer reaches TTS.

`OLLAMA_BASE_URL` overrides the endpoint (default http://localhost:11434/v1).
"""

from __future__ import annotations

import os
import re
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, cast

from .base import LLMAdapter, Message

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

_DEFAULT_BASE_URL = "http://localhost:11434/v1"
_DEFAULT_MODEL = "qwen3:0.6b"
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class OllamaLLM(LLMAdapter):
    name = "ollama"

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        from openai import AsyncOpenAI

        url = base_url or os.getenv("OLLAMA_BASE_URL") or _DEFAULT_BASE_URL
        # Ollama ignores the key but the SDK requires a non-empty one.
        self._client = AsyncOpenAI(api_key="ollama", base_url=url)
        self._model = model or os.getenv("OLLAMA_MODEL") or _DEFAULT_MODEL

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        payload = cast(
            "list[ChatCompletionMessageParam]",
            [{"role": m.role, "content": m.content} for m in messages],
        )
        completion = await self._client.chat.completions.create(
            model=self._model,
            messages=payload,
            stream=True,
        )
        # Strip <think>…</think> blocks. They may span chunk boundaries, so we
        # buffer until a block closes; outside a block, emit deltas as they come.
        in_think = False
        buffer = ""
        async for chunk in completion:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if not delta:
                continue
            buffer += delta
            out, buffer, in_think = _drain(buffer, in_think)
            if out:
                yield out

    async def aclose(self) -> None:
        await self._client.close()


def _drain(buffer: str, in_think: bool) -> tuple[str, str, bool]:
    """Emit any text outside <think> blocks; keep an unclosed tail buffered.

    Returns (emit, remaining_buffer, in_think).
    """
    out = ""
    while buffer:
        if in_think:
            end = buffer.find("</think>")
            if end == -1:
                # Still inside a think block — hold nothing to emit, keep tail
                # only if it could be a partial close tag.
                return out, _keep_partial(buffer, "</think>"), True
            buffer = buffer[end + len("</think>") :]
            in_think = False
        else:
            start = buffer.find("<think>")
            if start == -1:
                # No think start; emit all but a possible partial "<think>" tail.
                keep = _keep_partial(buffer, "<think>")
                out += buffer[: len(buffer) - len(keep)]
                return out, keep, False
            out += buffer[:start]
            buffer = buffer[start + len("<think>") :]
            in_think = True
    return out, buffer, in_think


def _keep_partial(buffer: str, tag: str) -> str:
    """Keep a trailing slice of `buffer` only if it's a prefix of `tag`."""
    for i in range(min(len(tag) - 1, len(buffer)), 0, -1):
        if buffer[-i:] == tag[:i]:
            return buffer[-i:]
    return ""
