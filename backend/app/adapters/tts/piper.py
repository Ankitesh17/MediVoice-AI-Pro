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
from collections.abc import AsyncIterator, Iterable, Iterator
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np

from .base import SAMPLE_RATE, TTSAdapter

if TYPE_CHECKING:
    from piper import AudioChunk, PiperVoice

_DEFAULT_VOICE = "en_US-lessac-medium"
_voice_lock = Lock()
_voices: dict[Path, PiperVoice] = {}
_VOICE_DIR = Path.home() / ".local" / "share" / "piper-voices"


class PiperTTS(TTSAdapter):
    name = "piper"

    def __init__(self, voice: str | None = None) -> None:
        from piper import PiperVoice
        from piper.download_voices import download_voice

        name = voice or os.getenv("PIPER_VOICE") or _DEFAULT_VOICE
        voice_dir = Path(os.getenv("PIPER_VOICE_DIR") or _VOICE_DIR).expanduser()
        voice_dir.mkdir(parents=True, exist_ok=True)
        model = voice_dir / f"{name}.onnx"
        with _voice_lock:
            if model not in _voices:
                if not model.is_file() or not model.with_suffix(".onnx.json").is_file():
                    download_voice(name, voice_dir)
                _voices[model] = PiperVoice.load(str(model))
            self._voice = _voices[model]
        self._native_rate = int(self._voice.config.sample_rate)

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        # Piper yields individual sentences. Advance the iterator in a worker
        # so a long booking response can play its first sentence immediately.
        stream = self._render(text)
        while (pcm := await asyncio.to_thread(next, stream, None)) is not None:
            yield pcm

    def _render(self, text: str) -> Iterator[bytes]:
        stream: Iterable[AudioChunk] = self._voice.synthesize(text)
        for chunk in stream:
            pcm16 = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
            yield _resample_int16(pcm16, self._native_rate, SAMPLE_RATE)


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
