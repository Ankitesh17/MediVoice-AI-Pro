"""Offline, utterance-based speech recognition using faster-whisper."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator

import numpy as np

from .base import SAMPLE_RATE, STTAdapter, Transcript

_FRAME_MS = 32
_FRAME_SAMPLES = SAMPLE_RATE * _FRAME_MS // 1000
_SILENCE_RMS = 0.012
_SILENCE_HANG_MS = 600
_MIN_SPEECH_MS = 300
_PARTIAL_EVERY_MS = 700

# A hint, not a restriction: Whisper can still recognize other phrases.
_DEFAULT_PROMPT = (
    "A patient speaking to a UK GP receptionist about "
    "appointments, prescriptions, or clinic information."
)


class FasterWhisperSTT(STTAdapter):
    name = "whisper"

    def __init__(self, model_size: str | None = None, language: str | None = None) -> None:
        from faster_whisper import WhisperModel

        size = model_size or os.getenv("WHISPER_MODEL") or "base"
        import logging

        logging.getLogger("realtime-voice-agent-demo").info(
            "STT | Loading Whisper model=%s, device=cpu, compute_type=int8",
            size,
        )
        self._model = WhisperModel(size, device="cpu", compute_type="int8")
        # Set WHISPER_LANGUAGE=auto to restore automatic language detection.
        configured_language = (
            language if language is not None else os.getenv("WHISPER_LANGUAGE", "en")
        )
        self._language = (
            None if configured_language.lower() in ("auto", "none", "") else configured_language
        )
        self._initial_prompt = os.getenv("WHISPER_INITIAL_PROMPT", _DEFAULT_PROMPT)
        self._beam_size = max(1, int(os.getenv("WHISPER_BEAM_SIZE", "5")))
        self._emit_partials = os.getenv("WHISPER_PARTIALS", "0") == "1"

    async def transcribe(self, audio: AsyncIterator[bytes]) -> AsyncIterator[Transcript]:
        buf = np.zeros(0, dtype=np.float32)
        leftover = np.zeros(0, dtype=np.float32)
        silence_ms = 0
        speech_ms = 0
        since_partial_ms = 0
        in_speech = False

        async for chunk in audio:
            samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            leftover = np.concatenate([leftover, samples])
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
                    buf = np.concatenate([buf, frame])

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

                if in_speech and silence_ms >= _SILENCE_HANG_MS:
                    if speech_ms >= _MIN_SPEECH_MS:
                        text = await self._run(buf)
                        if text:
                            yield Transcript(text=text, is_final=True)
                    buf = np.zeros(0, dtype=np.float32)
                    in_speech = False
                    silence_ms = speech_ms = since_partial_ms = 0

            leftover = leftover[n_frames * _FRAME_SAMPLES :]

        if in_speech and speech_ms >= _MIN_SPEECH_MS:
            text = await self._run(buf)
            if text:
                yield Transcript(text=text, is_final=True)

    async def _run(self, audio: np.ndarray) -> str:
        import logging
        import time

        start = time.monotonic()
        text = await asyncio.to_thread(self._transcribe_sync, audio)

        logging.getLogger("realtime-voice-agent-demo").info(
            "LATENCY | Whisper transcription: %.0f ms (%.2f s audio)",
            (time.monotonic() - start) * 1000,
            len(audio) / SAMPLE_RATE,
        )

        return text

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        segments, _ = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=self._beam_size,
            initial_prompt=self._initial_prompt,
            condition_on_previous_text=False,
            vad_filter=False,  # Utterance segmentation is already done above.
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
