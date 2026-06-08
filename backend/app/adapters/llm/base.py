"""LLM adapter boundary.

Every chat LLM provider implements `LLMAdapter`. Callers only see this
interface, so swapping Kimi for Claude (or another model) never leaks past
`app/adapters/llm/`. Same idea as the STT boundary — provider-neutral by design.

Contract:
- input:  the full conversation as a list of `Message`
- output: an async stream of response text deltas (tokens)

Each provider keeps its own SDK inside its own module — Kimi uses the OpenAI
SDK (OpenAI-compatible endpoint), Claude uses the Anthropic SDK. The adapter
interface is what makes them interchangeable, not a shared client.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str


class LLMAdapter(ABC):
    """Streaming chat LLM provider."""

    #: Stable provider id, surfaced to the UI (e.g. "kimi", "claude").
    name: str

    @abstractmethod
    def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        """Consume the conversation, yield response text deltas as they arrive."""
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release provider resources (override if needed)."""
        return None
