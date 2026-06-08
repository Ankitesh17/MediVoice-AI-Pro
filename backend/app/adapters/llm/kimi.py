"""Kimi (Moonshot AI) LLM via the OpenAI-compatible endpoint.

Moonshot's API is OpenAI-compatible, so we use the official `openai` async
client pointed at `https://api.moonshot.ai/v1`. Model `kimi-k2.6`. Streaming
chat completions; we yield each content delta as it arrives.

Requires `MOONSHOT_API_KEY`.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, cast

from .base import LLMAdapter, Message

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

_BASE_URL = "https://api.moonshot.ai/v1"
_MODEL = "kimi-k2.6"


class KimiLLM(LLMAdapter):
    name = "kimi"

    def __init__(self, api_key: str | None = None, model: str = _MODEL) -> None:
        key = api_key or os.getenv("MOONSHOT_API_KEY")
        if not key:
            raise RuntimeError(
                "MOONSHOT_API_KEY is not set. Set it in backend/.env or use LLM_PROVIDER=claude."
            )
        # Imported lazily so the package imports without the SDK in use.
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=key, base_url=_BASE_URL)
        self._model = model

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        payload = cast(
            "list[ChatCompletionMessageParam]",
            [{"role": m.role, "content": m.content} for m in messages],
        )
        # NOTE: kimi-k2.6 rejects any temperature other than 1, so we don't set
        # it — the model's default applies. (Verified against the live API.)
        completion = await self._client.chat.completions.create(
            model=self._model,
            messages=payload,
            stream=True,
        )
        async for chunk in completion:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    async def aclose(self) -> None:
        await self._client.close()
