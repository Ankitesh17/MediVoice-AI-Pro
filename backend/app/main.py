"""FastAPI entrypoint for the realtime-voice-agent-demo backend.

Endpoints:
- GET  /health         — liveness
- WS   /ws             — mic PCM in → transcript → LLM reply → spoken audio
- WS   /ws/echo        — audio echo round-trip (wiring smoke test)
- POST /tts/synthesize — one-shot TTS for the eval harness (latency + WAV out)

Providers sit behind adapter boundaries, chosen by env vars (and switchable live
over /ws): STT_PROVIDER, LLM_PROVIDER, TTS_PROVIDER.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import struct
import time
from collections.abc import AsyncIterator
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from .services.booking_flow import BookingFlow

# Load backend/.env so provider keys (MOONSHOT_API_KEY, etc.) are available
# before any adapter reads them. Must run before the adapter imports below
# resolve env vars at construction time.
load_dotenv()

from .adapters.llm import (  # noqa: E402
    LANGUAGES,
    LLMAdapter,
    Message,
    create_llm_adapter,
    system_prompt_for,
)
from .adapters.stt import Transcript, create_stt_adapter  # noqa: E402
from .adapters.tts import (  # noqa: E402
    SAMPLE_RATE,
    TTS_PROVIDERS,
    TTSAdapter,
    create_tts_adapter,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("realtime-voice-agent-demo")

app = FastAPI(title="realtime-voice-agent-demo backend", version="0.1.0")

# Local dev only: the Next.js frontend runs on :3000.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tts/providers")
async def tts_providers() -> dict[str, list[str]]:
    """List TTS providers the eval harness can compare."""
    return {"providers": TTS_PROVIDERS}


@app.post("/tts/synthesize")
async def tts_synthesize(
    payload: Annotated[dict[str, str], Body()],
) -> dict[str, object]:
    """One-shot TTS for the eval harness.

    Synthesize `text` with `provider`, measuring time-to-first-audio and total
    synthesis time. Returns the audio as a base64 WAV (so the browser can play
    it directly) plus the metrics that let you compare providers head-to-head.
    """
    text = (payload.get("text") or "").strip()
    provider = payload.get("provider") or "piper"
    if not text:
        return {"error": "text is required"}
    try:
        tts = create_tts_adapter(provider)
    except Exception as exc:  # missing key / unknown provider
        return {"error": str(exc)}

    t0 = time.monotonic()
    first_audio_s: float | None = None
    chunks: list[bytes] = []
    try:
        async for pcm in tts.synthesize(text):
            if first_audio_s is None:
                first_audio_s = time.monotonic() - t0
            chunks.append(pcm)
    except Exception as exc:
        await tts.aclose()
        return {"error": f"synthesis failed: {exc}"}
    total_s = time.monotonic() - t0
    await tts.aclose()

    pcm = b"".join(chunks)
    audio_s = len(pcm) / 2 / SAMPLE_RATE  # Int16 mono @ SAMPLE_RATE
    wav = _pcm_to_wav(pcm, SAMPLE_RATE)
    return {
        "provider": provider,
        "text": text,
        "metrics": {
            "firstAudioMs": round((first_audio_s or 0.0) * 1000),
            "totalMs": round(total_s * 1000),
            "audioSeconds": round(audio_s, 2),
            "bytes": len(pcm),
        },
        "wavBase64": base64.b64encode(wav).decode(),
    }


def _pcm_to_wav(pcm: bytes, rate: int) -> bytes:
    """Wrap 16-bit mono PCM in a minimal WAV container."""
    buf = io.BytesIO()
    data_len = len(pcm)
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_len))
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_len))
    buf.write(pcm)
    return buf.getvalue()


@app.websocket("/ws/echo")
async def ws_echo(websocket: WebSocket) -> None:
    """Step-1 echo: binary -> audio echo; text -> control echo."""
    await websocket.accept()
    await websocket.send_json({"type": "state", "value": "ready"})
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if (data := message.get("bytes")) is not None:
                await websocket.send_bytes(data)
            elif (text := message.get("text")) is not None:
                await websocket.send_json({"type": "echo", "text": text})
    except WebSocketDisconnect:
        pass


class _Session:
    """Per-connection mutable config + conversation history + live TTS adapter.

    The client can switch modes (prompt) and TTS provider mid-session via a
    `config` message. A mode change rebuilds the system prompt and resets the
    history; a TTS change swaps the adapter (closing the old one).
    """

    def __init__(self, tts: TTSAdapter) -> None:
        self.mode = "talk"  # "talk" | "translate"
        self.source = "ja"
        self.target = "en"
        self.tts = tts
        self.history: list[Message] = []
        self._reset_history()

    def _reset_history(self) -> None:
        prompt = system_prompt_for(self.mode, self.source, self.target)
        self.history = [Message(role="system", content=prompt)]

    def apply_config(self, cfg: dict[str, object]) -> str | None:
        """Apply a config message. Returns an error string if a TTS swap failed
        (the previous provider stays active)."""
        mode = cfg.get("mode")
        if mode in ("talk", "translate"):
            self.mode = mode
        src = cfg.get("source")
        if isinstance(src, str) and src in LANGUAGES:
            self.source = src
        tgt = cfg.get("target")
        if isinstance(tgt, str) and tgt in LANGUAGES:
            self.target = tgt
        self._reset_history()

        tts_name = cfg.get("tts")
        if isinstance(tts_name, str) and tts_name in TTS_PROVIDERS and tts_name != self.tts.name:
            try:
                new_tts = create_tts_adapter(tts_name)
            except Exception as exc:  # e.g. missing key — keep the current one
                return str(exc)
            old, self.tts = self.tts, new_tts
            await_close = old.aclose()  # schedule close without blocking apply
            asyncio.ensure_future(await_close)
        return None

    def config_event(self) -> dict[str, object]:
        return {
            "type": "config",
            "mode": self.mode,
            "source": self.source,
            "target": self.target,
            "tts": self.tts.name,
        }


@app.websocket("/ws")
async def ws_transcribe(websocket: WebSocket) -> None:
    """PCM in → transcript → LLM (chat or translate) → spoken reply."""
    await websocket.accept()
    logger.info("voice client connected")

    try:
        stt = create_stt_adapter()
        llm = create_llm_adapter()
        tts = create_tts_adapter()
    except Exception as exc:  # provider misconfig (e.g. missing key)
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close()
        return

    session = _Session(tts)
    booking_flow = BookingFlow()
    await websocket.send_json(
        {
            "type": "state",
            "value": "ready",
            "stt": stt.name,
            "llm": llm.name,
            "tts": tts.name,
            "languages": LANGUAGES,
            "ttsProviders": TTS_PROVIDERS,
            "mode": session.mode,
            "source": session.source,
            "target": session.target,
        }
    )

    # Bridge inbound WS audio frames into an async iterator for the STT adapter.
    audio_q: asyncio.Queue[bytes | None] = asyncio.Queue()

    async def audio_stream() -> AsyncIterator[bytes]:
        while True:
            chunk = await audio_q.get()
            if chunk is None:
                return
            yield chunk

    # The in-flight response task (LLM + TTS), if the assistant is speaking.
    response_task: asyncio.Task[None] | None = None

    async def cancel_response(reason: str) -> None:
        """Barge-in: stop the assistant mid-reply. Cancels LLM+TTS and tells the
        client to stop playback. Whatever text was already generated stays in
        history (appended by _respond's finally), so context isn't lost."""
        nonlocal response_task
        if response_task is not None and not response_task.done():
            response_task.cancel()
            try:
                await response_task
            except asyncio.CancelledError:
                pass
            await websocket.send_json({"type": "assistant", "phase": "interrupted"})
            logger.info("response interrupted (%s)", reason)
        response_task = None

    async def receive_loop() -> None:
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if (data := message.get("bytes")) is not None:
                    await audio_q.put(data)
                elif (text := message.get("text")) is not None:
                    try:
                        cfg = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    kind = cfg.get("type")
                    if kind == "config":
                        await cancel_response("config-change")
                        err = session.apply_config(cfg)
                        booking_flow.reset()
                        if err:
                            await websocket.send_json(
                                {"type": "error", "message": err}
                            )
                        await websocket.send_json(session.config_event())
                    elif kind == "interrupt":
                        # Client detected the user speaking over the assistant.
                        await cancel_response("client-interrupt")
        except WebSocketDisconnect:
            pass
        finally:
            await audio_q.put(None)  # signal end-of-audio

    recv_task = asyncio.create_task(receive_loop())
    try:
        async for tr in stt.transcribe(audio_stream()):
            # A new final user turn supersedes any in-flight reply (server-side
            # barge-in safety net, in addition to the client interrupt).
            if tr.is_final and tr.text:
                await cancel_response("new-turn")
            await _send_transcript(websocket, tr)
            if tr.is_final and tr.text:
                # Use session.tts (live) so a provider switch mid-session applies.
                response_task = asyncio.create_task(
                    _respond(
                            websocket,
                            llm,
                            session.tts,
                            session.history,
                            tr.text,
                            booking_flow if session.mode == "talk" else None,
                        )
                )
        # Audio stream ended (client stopped sending). Let any in-flight reply
        # finish rather than cutting it off.
        if response_task is not None:
            await asyncio.gather(response_task, return_exceptions=True)
    except WebSocketDisconnect:
        if response_task is not None and not response_task.done():
            response_task.cancel()
    finally:
        recv_task.cancel()
        await stt.aclose()
        await llm.aclose()
        await session.tts.aclose()
        logger.info("voice client disconnected")


async def _send_transcript(websocket: WebSocket, tr: Transcript) -> None:
    await websocket.send_json(
        {"type": "transcript", "text": tr.text, "isFinal": tr.is_final}
    )


_SENTENCE_ENDINGS = ".!?。！？\n"



async def _respond(
    websocket: WebSocket,
    llm: LLMAdapter,
    tts: TTSAdapter,
    history: list[Message],
    user_text: str,
    booking_flow: BookingFlow | None = None,
) -> None:
    """Handle a finalized user turn and stream text and spoken audio.

    Booking requests use the deterministic booking workflow.
    Other conversations use the LLM.
    """

    # First check whether the booking workflow should handle this turn.
    if booking_flow is not None:
        booking_reply = booking_flow.handle(user_text)

        if booking_reply is not None:
            history.append(
                Message(role="user", content=user_text)
            )

            await websocket.send_json(
                {"type": "assistant", "phase": "start"}
            )

            await websocket.send_json(
                {"type": "assistant", "delta": booking_reply}
            )

            history.append(
                Message(role="assistant", content=booking_reply)
            )

            await _speak(websocket, tts, booking_reply)

            await websocket.send_json(
                {"type": "assistant", "phase": "done"}
            )
            return

    
    # Keep appointment operations out of the free-form LLM.
    # BookingFlow is the only component allowed to confirm
    # booking, cancellation, or rescheduling operations.
    if booking_flow is not None:
        history[:] = [
            message for message in history
            if not (
                message.role == "assistant"
                and (
                    "Should I confirm this change?" in message.content
                    or "your original appointment remains unchanged"
                    in message.content.lower()
                )
            )
        ]

    # Normal LLM conversation
    history.append(
        Message(role="user", content=user_text)
    )

    await websocket.send_json(
        {"type": "assistant", "phase": "start"}
    )

    parts: list[str] = []

    try:
        pending = ""

        async for delta in llm.stream(history):
            parts.append(delta)

            await websocket.send_json(
                {"type": "assistant", "delta": delta}
            )

            pending += delta

            sentence, pending = _take_sentence(pending)

            if sentence:
                await _speak(websocket, tts, sentence)

        if pending.strip():
            await _speak(websocket, tts, pending.strip())

        await websocket.send_json(
            {"type": "assistant", "phase": "done"}
        )

    finally:
        # Preserve conversation history, including partial responses
        # if the user interrupts the assistant.
        reply = "".join(parts)

        if reply:
            history.append(
                Message(role="assistant", content=reply)
            )


def _take_sentence(buffer: str) -> tuple[str, str]:
    """Split off the first complete sentence; return (sentence, remainder)."""
    for i, ch in enumerate(buffer):
        if ch in _SENTENCE_ENDINGS:
            return buffer[: i + 1].strip(), buffer[i + 1 :]
    return "", buffer


async def _speak(websocket: WebSocket, tts: TTSAdapter, text: str) -> None:
    """Synthesize one chunk of text and stream its PCM frames to the client.

    The audio is framed by JSON markers so the client knows when a spoken
    segment starts and ends; the PCM itself rides as binary WS frames.
    """
    await websocket.send_json({"type": "audio", "phase": "start", "text": text})
    try:
        async for pcm in tts.synthesize(text):
            await websocket.send_bytes(pcm)
    except Exception as exc:  # TTS failure shouldn't kill the conversation
        logger.warning("TTS failed: %s", exc)
        await websocket.send_json({"type": "audio", "phase": "error", "message": str(exc)})
        return
    await websocket.send_json({"type": "audio", "phase": "end"})
