"""Claude LLM via the official Anthropic SDK.

Uses `client.messages.stream(...)` and yields text deltas. The system prompt is
passed as the top-level `system` parameter (not a message). Default model is
`claude-opus-4-8`; override with `CLAUDE_MODEL` (e.g. `claude-sonnet-4-6` for
lower latency in a voice loop).

Adaptive thinking is left off here: for short, spoken, conversational replies we
want minimal latency and no reasoning preamble. (On Opus 4.8, `temperature` /
`budget_tokens` are removed — we don't pass them.)

Requires `ANTHROPIC_API_KEY`.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, cast

from .base import LLMAdapter, Message

if TYPE_CHECKING:
    from anthropic.types import MessageParam

_DEFAULT_MODEL = "claude-opus-4-8"


class ClaudeLLM(LLMAdapter):
    name = "claude"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Set it in backend/.env or use LLM_PROVIDER=kimi."
            )
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=key)
        self._model = model or os.getenv("CLAUDE_MODEL") or _DEFAULT_MODEL

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        from anthropic import omit

        # Split the system message out — Anthropic takes it as a top-level param.
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        convo = cast(
            "list[MessageParam]",
            [
                {"role": m.role, "content": m.content}
                for m in messages
                if m.role != "system"
            ],
        )
        async with self._client.messages.stream(
            model=self._model,
            max_tokens=1024,
            system=system or omit,
            messages=convo,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def aclose(self) -> None:
        await self._client.close()
