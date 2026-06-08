"""Hume Octave 2 TTS — scaffold (lower priority).

The adapter boundary is here so the eval harness can later A/B Hume against
Inworld, but the implementation is intentionally deferred: Hume API access for
this project still needs to be sorted out. Inworld is the shipping provider.

When enabling: Hume exposes a Python/TS SDK and a WebSocket streaming endpoint
that returns audio incrementally; request PCM at 16 kHz to match the pipeline,
yielding Int16 PCM `bytes` chunks like InworldTTS does.

Requires `HUME_API_KEY`.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

from .base import TTSAdapter


class HumeTTS(TTSAdapter):
    name = "hume"

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.getenv("HUME_API_KEY")
        if not key:
            raise RuntimeError(
                "HUME_API_KEY is not set. Set it in backend/.env or use TTS_PROVIDER=inworld."
            )
        self._key = key

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        raise NotImplementedError(
            "HumeTTS is scaffolded but not implemented yet — use TTS_PROVIDER=inworld. "
            "See app/adapters/tts/hume.py for the integration notes."
        )
        # Unreachable, but marks this as an async generator for the type checker.
        if False:  # pragma: no cover
            yield b""
