"""Inworld Realtime TTS.

Streams audio from `POST https://api.inworld.ai/tts/v1/voice:stream`. The
response is NDJSON — one JSON object per line, each carrying a base64-encoded
PCM16 chunk in `result.audioContent`. We request LINEAR16 at 16 kHz so the
audio matches the rest of the pipeline (STT input, browser playback) with no
resampling.

Auth: Inworld uses HTTP Basic with the base64 API key as the credential, i.e.
`Authorization: Basic <INWORLD_API_KEY>`.

Requires `INWORLD_API_KEY`.
"""

from __future__ import annotations

import base64
import json
import os
from collections.abc import AsyncIterator

import httpx

from .base import SAMPLE_RATE, TTSAdapter

_URL = "https://api.inworld.ai/tts/v1/voice:stream"
_DEFAULT_VOICE = "Ashley"
_DEFAULT_MODEL = "inworld-tts-1.5-max"


class InworldTTS(TTSAdapter):
    name = "inworld"

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str | None = None,
        model_id: str = _DEFAULT_MODEL,
    ) -> None:
        key = api_key or os.getenv("INWORLD_API_KEY")
        if not key:
            raise RuntimeError(
                "INWORLD_API_KEY is not set. Set it in backend/.env or use TTS_PROVIDER=hume."
            )
        self._key = key
        self._voice = voice_id or os.getenv("INWORLD_VOICE_ID") or _DEFAULT_VOICE
        self._model = model_id
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=120.0))

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        payload = {
            "text": text,
            "voiceId": self._voice,
            "modelId": self._model,
            "audioConfig": {
                "audioEncoding": "LINEAR16",
                "sampleRateHertz": SAMPLE_RATE,
            },
        }
        headers = {
            "Authorization": f"Basic {self._key}",
            "Content-Type": "application/json",
        }
        async with self._client.stream(
            "POST", _URL, headers=headers, json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                b64 = obj.get("result", {}).get("audioContent")
                if not b64:
                    continue
                pcm = base64.b64decode(b64)
                pcm = _strip_wav_header(pcm)
                if pcm:
                    yield pcm

    async def aclose(self) -> None:
        await self._client.aclose()


def _strip_wav_header(data: bytes) -> bytes:
    """Inworld LINEAR16 chunks are raw PCM, but the first chunk may carry a
    44-byte WAV header. Drop it so we emit a clean PCM stream."""
    if data[:4] == b"RIFF" and len(data) > 44:
        return data[44:]
    return data
