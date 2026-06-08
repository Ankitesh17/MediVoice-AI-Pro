"""STT adapter boundary.

Every speech-to-text provider implements `STTAdapter`. The rest of the app only
ever sees this interface, so swapping faster-whisper for Deepgram (or a future
in-house engine) never leaks past `app/adapters/stt/`. This is the concrete
answer to the project's "internalization / provider negotiation" requirement.

Contract:
- input:  an async stream of 16 kHz mono Int16 PCM `bytes` chunks from the client
- output: an async stream of `Transcript` events (partials, then a final per turn)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

# Audio contract shared across adapters and the frontend capture code.
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # bytes per sample (Int16)
CHANNELS = 1


@dataclass(frozen=True, slots=True)
class Transcript:
    """One recognition result.

    `is_final=False` is a partial (the in-progress guess for the current turn);
    `is_final=True` marks the end of a turn's transcription.
    """

    text: str
    is_final: bool


class STTAdapter(ABC):
    """Streaming speech-to-text provider."""

    #: Stable provider id, surfaced to the UI (e.g. "whisper", "deepgram").
    name: str

    @abstractmethod
    def transcribe(self, audio: AsyncIterator[bytes]) -> AsyncIterator[Transcript]:
        """Consume PCM chunks, yield Transcript events as they become available."""
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release provider resources (override if needed)."""
        return None
