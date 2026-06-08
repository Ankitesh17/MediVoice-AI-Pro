"""STT adapter factory.

`STT_PROVIDER` selects the implementation (default: "whisper", which needs no
API key and runs fully locally). "deepgram" requires `DEEPGRAM_API_KEY`.
"""

from __future__ import annotations

import os

from .base import SAMPLE_RATE, STTAdapter, Transcript

__all__ = ["SAMPLE_RATE", "STTAdapter", "Transcript", "create_stt_adapter"]


def create_stt_adapter(provider: str | None = None) -> STTAdapter:
    name = (provider or os.getenv("STT_PROVIDER") or "whisper").lower()
    if name == "whisper":
        from .whisper import FasterWhisperSTT

        return FasterWhisperSTT()
    if name == "deepgram":
        from .deepgram import DeepgramSTT

        return DeepgramSTT()
    raise ValueError(f"unknown STT_PROVIDER: {name!r} (expected 'whisper' or 'deepgram')")
