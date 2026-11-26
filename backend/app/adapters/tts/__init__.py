"""TTS adapter factory.

`TTS_PROVIDER` selects the implementation:
- "piper" (default) — local, offline, key-free (ONNX on CPU).
- "inworld" — cloud, more expressive; needs `INWORLD_API_KEY`.
- "hume" — scaffolded, lower priority; needs `HUME_API_KEY`.

Default is the local engine so the demo runs out of the box with no API keys.
"""

from __future__ import annotations

import os

from .base import SAMPLE_RATE, TTSAdapter

__all__ = ["SAMPLE_RATE", "TTSAdapter", "create_tts_adapter", "TTS_PROVIDERS"]

# Surfaced to the UI so the user can switch providers live (local first).
TTS_PROVIDERS = ["piper", "inworld"]


def preload_tts_model() -> None:
    if (os.getenv("TTS_PROVIDER") or "piper").lower() == "piper":
        create_tts_adapter("piper")


def create_tts_adapter(provider: str | None = None) -> TTSAdapter:
    name = (provider or os.getenv("TTS_PROVIDER") or "piper").lower()
    if name == "inworld":
        from .inworld import InworldTTS

        return InworldTTS()
    if name == "piper":
        from .piper import PiperTTS

        return PiperTTS()
    if name == "hume":
        from .hume import HumeTTS

        return HumeTTS()
    raise ValueError(
        f"unknown TTS_PROVIDER: {name!r} (expected 'inworld', 'piper', or 'hume')"
    )
