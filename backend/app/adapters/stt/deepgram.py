"""Cloud STT via Deepgram streaming.

True streaming: Deepgram returns interim results (partials) and marks utterance
ends natively, so we don't bolt on our own VAD. Requires `DEEPGRAM_API_KEY`.

Deepgram's live client is callback-based; we bridge its callbacks into an
asyncio.Queue so this adapter exposes the same `AsyncIterator[Transcript]`
contract as the local one. That symmetry is the whole point of the adapter
boundary — callers can't tell which provider is behind it.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator

from .base import SAMPLE_RATE, STTAdapter, Transcript

_SENTINEL = object()


class DeepgramSTT(STTAdapter):
    name = "deepgram"

    def __init__(self, api_key: str | None = None, language: str | None = None) -> None:
        key = api_key or os.getenv("DEEPGRAM_API_KEY")
        if not key:
            raise RuntimeError(
                "DEEPGRAM_API_KEY is not set. Set it in backend/.env or use STT_PROVIDER=whisper."
            )
        self._key = key
        self._language = language

    async def transcribe(self, audio: AsyncIterator[bytes]) -> AsyncIterator[Transcript]:
        # Imported lazily so the package imports without the SDK installed/used.
        from deepgram import (
            AsyncLiveClient,
            LiveOptions,
            LiveTranscriptionEvents,
        )

        queue: asyncio.Queue[object] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        client = AsyncLiveClient(self._key)

        async def on_message(_self: object, result: object, **_: object) -> None:
            alt = result.channel.alternatives[0]  # type: ignore[attr-defined]
            text = (alt.transcript or "").strip()
            if not text:
                return
            is_final = bool(getattr(result, "is_final", False))
            await queue.put(Transcript(text=text, is_final=is_final))

        async def on_close(_self: object, *_: object, **__: object) -> None:
            await queue.put(_SENTINEL)

        client.on(LiveTranscriptionEvents.Transcript, on_message)
        client.on(LiveTranscriptionEvents.Close, on_close)

        options = LiveOptions(
            model="nova-3",
            language=self._language or "multi",
            encoding="linear16",
            sample_rate=SAMPLE_RATE,
            channels=1,
            interim_results=True,
            punctuate=True,
        )
        await client.start(options)

        async def pump() -> None:
            try:
                async for chunk in audio:
                    await client.send(chunk)
            finally:
                await client.finish()

        pump_task = loop.create_task(pump())
        try:
            while True:
                item = await queue.get()
                if item is _SENTINEL:
                    break
                assert isinstance(item, Transcript)
                yield item
        finally:
            pump_task.cancel()
            await client.finish()
