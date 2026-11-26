"""Offline, utterance-based speech recognition using faster-whisper."""

from __future__ import annotations

import asyncio
import logging
import os
from collections import deque
from collections.abc import AsyncIterator
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np

from .base import SAMPLE_RATE, STTAdapter, Transcript

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

_models: dict[str, WhisperModel] = {}
_model_lock = Lock()

_FRAME_MS = 32
_FRAME_SAMPLES = SAMPLE_RATE * _FRAME_MS // 1000
_MIN_RMS = 0.0035
_START_FRAMES = 2
_PRE_ROLL_FRAMES = 6  # 192 ms
_TRAILING_FRAMES = 5  # 160 ms kept for quiet consonants
_MIN_VOICED_FRAMES = 4  # 128 ms, including short confirmations
_MAX_UTTERANCE_FRAMES = 24_000 // _FRAME_MS
_PARTIAL_EVERY_MS = 700

# A hint, not a restriction: Whisper can still recognize other phrases.
_DEFAULT_PROMPT = (
    "GP appointment, booking reference, repeat prescription, physiotherapy, "
    "asthma clinic, pharmacy, blood pressure."
)


def _get_model(size: str) -> WhisperModel:
    from faster_whisper import WhisperModel

    with _model_lock:
        if size not in _models:
            logging.getLogger("realtime-voice-agent-demo").info(
                "STT | Loading Whisper model=%s, device=cpu, compute_type=int8", size
            )
            _models[size] = WhisperModel(size, device="cpu", compute_type="int8")
        return _models[size]


class FasterWhisperSTT(STTAdapter):
    name = "whisper"

    def __init__(self, model_size: str | None = None, language: str | None = None) -> None:
        size = model_size or os.getenv("WHISPER_MODEL") or "base"
        self._model = _get_model(size)
        # Set WHISPER_LANGUAGE=auto to restore automatic language detection.
        configured_language = (
            language if language is not None else os.getenv("WHISPER_LANGUAGE", "en")
        )
        self._language = (
            None if configured_language.lower() in ("auto", "none", "") else configured_language
        )
        self._initial_prompt = os.getenv("WHISPER_INITIAL_PROMPT", _DEFAULT_PROMPT)
        self._beam_size = max(1, int(os.getenv("WHISPER_BEAM_SIZE", "3")))
        self._emit_partials = os.getenv("WHISPER_PARTIALS", "0") == "1"
        self._vad = os.getenv("WHISPER_VAD", "0") == "1"
        self._normalize = os.getenv("WHISPER_NORMALIZE", "1") == "1"

    async def transcribe(self, audio: AsyncIterator[bytes]) -> AsyncIterator[Transcript]:
        pre_roll: deque[np.ndarray] = deque(maxlen=_PRE_ROLL_FRAMES)
        utterance: list[np.ndarray] = []
        leftover = b""
        noise_floor = 0.001
        voiced_frames = silence_frames = onset = last_partial = 0

        async for chunk in audio:
            # Carry incomplete PCM samples and frames across WebSocket packets.
            samples = leftover + chunk
            frame_bytes = 2 * _FRAME_SAMPLES
            complete = len(samples) // frame_bytes * frame_bytes
            for offset in range(0, complete, frame_bytes):
                frame = (
                    np.frombuffer(samples, dtype="<i2", count=_FRAME_SAMPLES, offset=offset)
                    .astype(np.float32) / 32768.0
                )
                rms = float(np.sqrt(np.mean(frame * frame)))
                threshold = max(_MIN_RMS, min(0.018, noise_floor * 3.0))
                voiced = rms >= threshold * (0.70 if utterance else 1.0)
                if not utterance:
                    pre_roll.append(frame)
                    if rms < 0.018:
                        noise_floor = noise_floor * 0.95 + rms * 0.05
                    onset = onset + 1 if voiced else 0
                    if onset < _START_FRAMES:
                        continue
                    utterance = list(pre_roll)
                    pre_roll.clear()
                    voiced_frames = onset
                    continue

                utterance.append(frame)
                if voiced:
                    voiced_frames += 1
                    silence_frames = 0
                else:
                    silence_frames += 1

                # Leave more room for pauses within a longer request than after
                # a short confirmation. Retain the quiet tail for the decoder.
                hang_frames = 12 if voiced_frames <= 10 else 16
                if silence_frames >= hang_frames or len(utterance) >= _MAX_UTTERANCE_FRAMES:
                    if voiced_frames >= _MIN_VOICED_FRAMES:
                        keep = len(utterance) - max(0, silence_frames - _TRAILING_FRAMES)
                        text, uncertain = await self._run(np.concatenate(utterance[:keep]))
                        if text:
                            yield Transcript(text, is_final=True, uncertain=uncertain)
                    if silence_frames:
                        pre_roll.extend(utterance[-silence_frames:])
                    utterance = []
                    voiced_frames = silence_frames = onset = last_partial = 0
                elif (
                    self._emit_partials
                    and (voiced_frames - last_partial) * _FRAME_MS >= _PARTIAL_EVERY_MS
                ):
                    last_partial = voiced_frames
                    text, _ = await self._run(np.concatenate(utterance))
                    if text:
                        yield Transcript(text, is_final=False)

            leftover = samples[complete:]

        if voiced_frames >= _MIN_VOICED_FRAMES:
            tail = leftover[:len(leftover) // 2 * 2]
            utterance.append(np.frombuffer(tail, dtype="<i2").astype(np.float32) / 32768.0)
            text, uncertain = await self._run(np.concatenate(utterance))
            if text:
                yield Transcript(text, is_final=True, uncertain=uncertain)

    async def _run(self, audio: np.ndarray) -> tuple[str, bool]:
        import time

        start = time.monotonic()
        result = await asyncio.to_thread(self._transcribe_sync, audio)

        logging.getLogger("realtime-voice-agent-demo").info(
            "LATENCY | Whisper transcription: %.0f ms (%.2f s audio)",
            (time.monotonic() - start) * 1000,
            len(audio) / SAMPLE_RATE,
        )

        return result

    def _transcribe_sync(self, audio: np.ndarray) -> tuple[str, bool]:
        if self._normalize:
            peak = float(np.max(np.abs(audio)))
            if 0 < peak < 0.30:
                audio = audio * min(3.0, 0.30 / peak)
        segments, _ = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=self._beam_size,
            initial_prompt=self._initial_prompt,
            condition_on_previous_text=False,
            temperature=0.0,
            vad_filter=self._vad,
            vad_parameters={"min_speech_duration_ms": 100, "speech_pad_ms": 160},
        )
        accepted = [
            seg for seg in segments
            if seg.text.strip() and not (
                seg.no_speech_prob >= 0.80 and seg.avg_logprob < -0.80
            )
        ]
        text = " ".join(seg.text.strip() for seg in accepted).strip()
        uncertain = any(seg.avg_logprob < -0.85 for seg in accepted)
        return text, uncertain
