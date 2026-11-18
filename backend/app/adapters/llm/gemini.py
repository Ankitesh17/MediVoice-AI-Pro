
"""Google Gemini LLM using its OpenAI-compatible API."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, cast

from .base import LLMAdapter, Message

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam


_DEFAULT_BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai/"
)
_DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiLLM(LLMAdapter):
    name = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        from openai import AsyncOpenAI

        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is missing. "
                "Set it in backend/.env."
            )

        self._client = AsyncOpenAI(
            api_key=key,
            base_url=_DEFAULT_BASE_URL,
            timeout=30.0,
            max_retries=1,
        )
        self._model = (
            model
            or os.getenv("GEMINI_MODEL")
            or _DEFAULT_MODEL
        )

    async def stream(
        self, messages: list[Message]
    ) -> AsyncIterator[str]:
        payload = cast(
            "list[ChatCompletionMessageParam]",
            [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
        )

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
