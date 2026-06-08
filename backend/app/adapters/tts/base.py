"""TTS adapter boundary.

Every text-to-speech provider implements `TTSAdapter`. Callers only see this
interface, so swapping Inworld for Hume (or a future in-house engine) never
leaks past `app/adapters/tts/`. This is the boundary the eval harness later
uses to A/B providers on one fixed script.

Contract:
- input:  text (one utterance, or a sentence chunk of a streamed reply)
- output: an async stream of 16 kHz mono Int16 PCM `bytes` chunks — the same
          audio contract the STT side and the browser playback use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

# Shared audio contract — matches STT input and browser playback.
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # Int16
CHANNELS = 1


class TTSAdapter(ABC):
    """Streaming text-to-speech provider."""

    #: Stable provider id, surfaced to the UI (e.g. "inworld", "hume").
    name: str

    @abstractmethod
    def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Synthesize `text`, yielding 16 kHz Int16 PCM chunks as they arrive."""
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release provider resources (override if needed)."""
        return None
