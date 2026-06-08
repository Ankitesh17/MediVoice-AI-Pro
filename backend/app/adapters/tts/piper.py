"""Local TTS via Piper.

Piper is an ONNX (VITS) TTS that runs on CPU, fully offline — no key, no
network at synthesis time. It yields Int16 PCM at the voice's native sample
rate (commonly 22.05 kHz); we resample to the pipeline's 16 kHz so it slots in
behind the same `TTSAdapter` as Inworld.

The voice model auto-downloads on first use into `~/.local/share/piper-voices`
(a ~60 MB ONNX file). Pick a voice with `PIPER_VOICE` (default
`en_US-lessac-medium`).

This is the local counterpart to the cloud Inworld adapter — the demo can run
TTS entirely on-device, key-free, by setting `TTS_PROVIDER=piper`.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .base import SAMPLE_RATE, TTSAdapter

if TYPE_CHECKING:
    from piper import AudioChunk, PiperVoice

_DEFAULT_VOICE = "en_US-lessac-medium"
_VOICE_DIR = Path.home() / ".local" / "share" / "piper-voices"


class PiperTTS(TTSAdapter):
    name = "piper"

    def __init__(self, voice: str | None = None) -> None:
        from piper import PiperVoice
        from piper.download_voices import download_voice

        name = voice or os.getenv("PIPER_VOICE") or _DEFAULT_VOICE
        _VOICE_DIR.mkdir(parents=True, exist_ok=True)
        model = _VOICE_DIR / f"{name}.onnx"
        if not model.exists():
            # First run: fetch the ONNX voice (and its .json) once.
            download_voice(name, _VOICE_DIR)
        self._voice: PiperVoice = PiperVoice.load(str(model))
        self._native_rate = int(self._voice.config.sample_rate)

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        # Piper is synchronous/CPU-bound — run it off the event loop and hand
        # back resampled 16 kHz PCM chunks.
        chunks = await asyncio.to_thread(self._render, text)
        for pcm in chunks:
            yield pcm

    def _render(self, text: str) -> list[bytes]:
        out: list[bytes] = []
        stream: Iterable[AudioChunk] = self._voice.synthesize(text)
        for chunk in stream:
            pcm16 = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
            out.append(_resample_int16(pcm16, self._native_rate, SAMPLE_RATE))
        return out


def _resample_int16(pcm: np.ndarray, src_rate: int, dst_rate: int) -> bytes:
    """Linear-interpolation resample of Int16 PCM. Good enough for a demo."""
    if src_rate == dst_rate or pcm.size == 0:
        return pcm.astype(np.int16).tobytes()
    duration = pcm.size / src_rate
    dst_len = int(duration * dst_rate)
    if dst_len <= 0:
        return b""
    src_idx = np.linspace(0, pcm.size - 1, dst_len)
    resampled = np.interp(src_idx, np.arange(pcm.size), pcm.astype(np.float32))
    return resampled.astype(np.int16).tobytes()
