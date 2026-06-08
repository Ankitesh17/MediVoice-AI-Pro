"""Local STT via faster-whisper.

faster-whisper is batch, not natively streaming, so we approximate streaming: an
energy-based VAD groups incoming PCM into utterances and we transcribe each turn
when trailing silence closes it. No API key, runs offline.

**Latency note:** partials (re-transcribing the growing buffer mid-utterance)
are expensive on CPU — each call re-runs the whole buffer, so cost grows with
utterance length and serializes ahead of the final. They are therefore **off by
default**; the turn-final transcribe is one cheap pass that feeds the LLM
immediately. Set `WHISPER_PARTIALS=1` to re-enable live captions, and
`WHISPER_MODEL=tiny` for an even faster (lower-accuracy) model.

For a production streaming pipeline you'd reach for Deepgram (see deepgram.py);
this adapter is the zero-setup default so the demo runs for anyone.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator

import numpy as np

from .base import SAMPLE_RATE, STTAdapter, Transcript

# VAD / segmentation tuning (energy-based; good enough for a demo).
_FRAME_MS = 32
_FRAME_SAMPLES = SAMPLE_RATE * _FRAME_MS // 1000
_SILENCE_RMS = 0.012  # below this is "silence"
_SILENCE_HANG_MS = 600  # trailing silence that closes a turn
_MIN_SPEECH_MS = 300  # ignore blips shorter than this
_PARTIAL_EVERY_MS = 700  # re-transcribe cadence while speaking (when enabled)


class FasterWhisperSTT(STTAdapter):
    name = "whisper"

    def __init__(self, model_size: str | None = None, language: str | None = None) -> None:
        # Imported lazily inside __init__ so the module imports without the model.
        from faster_whisper import WhisperModel

        size = model_size or os.getenv("WHISPER_MODEL") or "base"
        self._model = WhisperModel(size, device="cpu", compute_type="int8")
        self._language = language
        # Partials are off by default — they add CPU cost and latency. Opt in.
        self._emit_partials = os.getenv("WHISPER_PARTIALS", "0") == "1"

    async def transcribe(self, audio: AsyncIterator[bytes]) -> AsyncIterator[Transcript]:
        buf = np.zeros(0, dtype=np.float32)  # current utterance
        leftover = np.zeros(0, dtype=np.float32)  # sub-frame remainder
        silence_ms = 0
        speech_ms = 0
        since_partial_ms = 0
        in_speech = False

        async for chunk in audio:
            samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            leftover = np.concatenate([leftover, samples])

            # Process whole frames; keep the remainder for next time.
            n_frames = len(leftover) // _FRAME_SAMPLES
            for i in range(n_frames):
                frame = leftover[i * _FRAME_SAMPLES : (i + 1) * _FRAME_SAMPLES]
                rms = float(np.sqrt(np.mean(frame * frame)))
                voiced = rms >= _SILENCE_RMS

                if voiced:
                    in_speech = True
                    silence_ms = 0
                    speech_ms += _FRAME_MS
                    buf = np.concatenate([buf, frame])
                    since_partial_ms += _FRAME_MS
                elif in_speech:
                    silence_ms += _FRAME_MS
                    buf = np.concatenate([buf, frame])  # keep trailing silence in buffer

                # Emit a partial periodically while speaking (opt-in only).
                ready_for_partial = (
                    self._emit_partials
                    and in_speech
                    and since_partial_ms >= _PARTIAL_EVERY_MS
                    and speech_ms >= _MIN_SPEECH_MS
                )
                if ready_for_partial:
                    since_partial_ms = 0
                    text = await self._run(buf)
                    if text:
                        yield Transcript(text=text, is_final=False)

                # Trailing silence closes the turn.
                if in_speech and silence_ms >= _SILENCE_HANG_MS:
                    if speech_ms >= _MIN_SPEECH_MS:
                        text = await self._run(buf)
                        if text:
                            yield Transcript(text=text, is_final=True)
                    buf = np.zeros(0, dtype=np.float32)
                    in_speech = False
                    silence_ms = speech_ms = since_partial_ms = 0

            leftover = leftover[n_frames * _FRAME_SAMPLES :]

        # Stream ended mid-utterance: flush a final.
        if in_speech and speech_ms >= _MIN_SPEECH_MS:
            text = await self._run(buf)
            if text:
                yield Transcript(text=text, is_final=True)

    async def _run(self, audio: np.ndarray) -> str:
        """Transcribe a float32 buffer off the event loop."""
        return await asyncio.to_thread(self._transcribe_sync, audio)

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        segments, _ = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=1,
            vad_filter=False,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
