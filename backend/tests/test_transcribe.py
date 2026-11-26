"""Wiring tests for the /ws endpoint and the adapter factories.

We swap in stub adapters so the tests don't load a Whisper model, hit Deepgram,
or call provider APIs — they verify the WS plumbing (audio in -> transcript ->
streamed reply text -> synthesized audio frames).
"""

import asyncio
import json
from collections.abc import AsyncIterator

import pytest
from fastapi import WebSocket
from fastapi.testclient import TestClient
from starlette.types import Message as ASGIMessage

import app.main as main
from app.adapters.llm import create_llm_adapter
from app.adapters.llm.base import LLMAdapter, Message
from app.adapters.stt import create_stt_adapter
from app.adapters.stt.base import STTAdapter, Transcript
from app.adapters.tts import create_tts_adapter
from app.adapters.tts.base import TTSAdapter


@pytest.fixture(autouse=True)
def stub_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    # WebSocket wiring tests should run without native ML libraries doing inference.
    monkeypatch.setattr(
        main, "retrieve_context",
        lambda _: {"found": True, "context": "Demo clinic reference", "sources": []},
    )


class StubSTT(STTAdapter):
    name = "stub-stt"

    async def transcribe(self, audio: AsyncIterator[bytes]) -> AsyncIterator[Transcript]:
        # Drain one chunk, then emit a partial and a final.
        async for _ in audio:
            yield Transcript(text="hello", is_final=False)
            yield Transcript(text="hello world", is_final=True)
            return


class StubLLM(LLMAdapter):
    name = "stub-llm"

    def __init__(self) -> None:
        self.seen: list[Message] | None = None

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        self.seen = list(messages)  # snapshot — caller mutates history after this
        for token in ["Hi ", "there!"]:
            yield token


class StubTTS(TTSAdapter):
    name = "stub-tts"

    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        self.spoken.append(text)
        yield b"\x01\x02" * 8  # one fake PCM frame


class BlockingLLM(LLMAdapter):
    """Emits one token, then hangs — models an in-flight reply that gets
    cancelled by barge-in."""

    name = "blocking-llm"

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        yield "Once "
        await asyncio.Event().wait()  # never completes; cancelled by barge-in
        yield "upon a time"  # pragma: no cover


class _FakeWS:
    """Minimal WebSocket double that records sent JSON and ignores bytes."""

    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []

    async def send_json(self, payload: dict[str, object]) -> None:
        self.sent.append(payload)

    async def send_bytes(self, data: bytes) -> None:  # pragma: no cover
        pass


def test_ws_full_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_llm = StubLLM()
    stub_tts = StubTTS()
    monkeypatch.setattr(main, "create_stt_adapter", lambda: StubSTT())
    monkeypatch.setattr(main, "create_llm_adapter", lambda: stub_llm)
    monkeypatch.setattr(main, "create_tts_adapter", lambda: stub_tts)
    client = TestClient(main.app)
    with client.websocket_connect("/ws") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "state"
        assert ready["stt"] == "stub-stt"
        assert ready["llm"] == "stub-llm"
        assert ready["tts"] == "stub-tts"

        ws.send_bytes(b"\x00\x00" * 320)  # one 16-bit PCM frame
        assert ws.receive_json() == {"type": "transcript", "text": "hello", "isFinal": False}
        assert ws.receive_json() == {
            "type": "transcript",
            "text": "hello world",
            "isFinal": True,
        }
        # Final transcript triggers an LLM turn streamed as assistant events,
        # and the completed sentence is synthesized to audio.
        assert ws.receive_json() == {"type": "assistant", "phase": "start"}
        assert ws.receive_json() == {"type": "assistant", "delta": "Hi "}
        assert ws.receive_json() == {"type": "assistant", "delta": "there!"}
        # "Hi there!" is a complete sentence -> spoken.
        audio_start = ws.receive_json()
        assert audio_start["type"] == "audio"
        assert audio_start["phase"] == "start"
        assert audio_start["text"] == "Hi there!"
        assert ws.receive_bytes() == b"\x01\x02" * 8
        assert ws.receive_json() == {"type": "audio", "phase": "end"}
        assert ws.receive_json() == {"type": "assistant", "phase": "done"}

    assert stub_llm.seen is not None
    assert stub_llm.seen[-1].content == "hello world"
    assert any("Demo clinic reference" in msg.content for msg in stub_llm.seen)
    assert stub_tts.spoken == ["Hi there!"]


def test_barge_in_cancels_and_saves_partial() -> None:
    """Barge-in cancels an in-flight reply, and whatever the assistant said
    before the cut is still recorded in history (context stays coherent).

    Tested at the task level (not via TestClient, whose single-threaded
    WebSocket can't model a concurrent interrupt mid-reply). The real server
    runs this on a normal event loop; see the README's barge-in E2E note.
    """

    async def scenario() -> tuple[list[dict[str, object]], list[Message]]:
        ws = _FakeWS()
        history: list[Message] = [Message(role="system", content="sys")]
        task = asyncio.create_task(
            main._respond(ws, BlockingLLM(), StubTTS(), history, "tell me a story")
        )
        # Let the reply start and emit its first token, then barge in.
        for _ in range(200):
            if any(m.get("delta") == "Once " for m in ws.sent):
                break
            if task.done():
                await task
                raise AssertionError("Response finished before barge-in")
            await asyncio.sleep(0.01)
        else:
            task.cancel()
            raise AssertionError("Timed out waiting for first response token")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return ws.sent, history

    sent, history = asyncio.run(scenario())

    # Started, streamed one token, never reported "done".
    assert {"type": "assistant", "phase": "start"} in sent
    assert {"type": "assistant", "delta": "Once "} in sent
    assert not any(m.get("phase") == "done" for m in sent)
    # The partial reply is preserved in history.
    assert history[-1].role == "assistant"
    assert history[-1].content == "Once "


def test_config_switches_to_translate_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `config` message switches the system prompt to interpreter mode and
    resets history; the LLM then sees the translate prompt + the user turn."""
    stub_llm = StubLLM()
    monkeypatch.setattr(main, "create_stt_adapter", lambda: StubSTT())
    monkeypatch.setattr(main, "create_llm_adapter", lambda: stub_llm)
    monkeypatch.setattr(main, "create_tts_adapter", lambda: StubTTS())
    client = TestClient(main.app)
    with client.websocket_connect("/ws") as ws:
        ready = ws.receive_json()
        assert ready["mode"] == "talk"
        assert "ja" in ready["languages"] and "en" in ready["languages"]

        ws.send_text('{"type":"config","mode":"translate","source":"ja","target":"en"}')
        cfg = ws.receive_json()
        assert cfg == {
            "type": "config",
            "mode": "translate",
            "source": "ja",
            "target": "en",
            "tts": "stub-tts",
        }

        ws.send_bytes(b"\x00\x00" * 320)
        # Drain until the assistant turn completes. Audio frames interleave as
        # binary, so use the raw receiver and branch on the message kind.
        while True:
            raw = ws.receive()
            if "bytes" in raw and raw["bytes"] is not None:
                continue
            msg = json.loads(raw["text"])
            if msg.get("type") == "assistant" and msg.get("phase") == "done":
                break

    # The interpreter system prompt is active, not the chat one.
    assert stub_llm.seen is not None
    assert "interpreter" in stub_llm.seen[0].content.lower()
    assert stub_llm.seen[-1].content == "hello world"


def test_llm_factory_unknown_provider() -> None:
    with pytest.raises(ValueError, match="unknown LLM_PROVIDER"):
        create_llm_adapter("bogus")


def test_tts_synthesize_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """The eval-harness endpoint returns metrics + a playable WAV."""
    monkeypatch.setattr(main, "create_tts_adapter", lambda provider=None: StubTTS())
    client = TestClient(main.app)
    resp = client.post("/tts/synthesize", json={"text": "hi", "provider": "stub"})
    assert resp.status_code == 200
    body = resp.json()
    assert "error" not in body
    assert body["metrics"]["bytes"] > 0
    assert body["metrics"]["totalMs"] >= 0
    # WAV is base64 and starts with the RIFF header.
    import base64 as _b64

    wav = _b64.b64decode(body["wavBase64"])
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"


def test_tts_synthesize_requires_text() -> None:
    client = TestClient(main.app)
    assert client.post("/tts/synthesize", json={"text": "  "}).json()["error"]


def test_ollama_think_stripping() -> None:
    """The Ollama adapter must strip <think>…</think> reasoning, even split
    across streamed chunks, and leave normal text intact."""
    from app.adapters.llm.ollama import _drain

    # Whole block in one buffer.
    out, rem, in_think = _drain("<think>reasoning</think>Hello", False)
    assert out == "Hello"
    assert rem == ""
    assert in_think is False

    # Block split across chunks: open in one drain, close in the next.
    out1, rem1, t1 = _drain("Hi <think>part", False)
    assert out1 == "Hi "
    assert t1 is True
    out2, rem2, t2 = _drain(rem1 + "ial</think>there", t1)
    assert out2 == "there"
    assert t2 is False

    # Plain text with a partial tag prefix at the tail is held back, not emitted.
    out3, rem3, _ = _drain("done<thi", False)
    assert out3 == "done"
    assert rem3 == "<thi"


def test_tts_factory_unknown_provider() -> None:
    with pytest.raises(ValueError, match="unknown TTS_PROVIDER"):
        create_tts_adapter("bogus")


def test_factory_unknown_provider() -> None:
    with pytest.raises(ValueError, match="unknown STT_PROVIDER"):
        create_stt_adapter("bogus")


def test_factory_default_is_whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    # Don't actually construct the model; just confirm the factory routes to it.
    import app.adapters.stt as stt_pkg

    constructed: list[str] = []

    class FakeWhisper(STTAdapter):
        name = "whisper"

        def __init__(self) -> None:
            constructed.append("whisper")

        async def transcribe(self, audio: AsyncIterator[bytes]) -> AsyncIterator[Transcript]:
            if False:  # pragma: no cover - never iterated
                yield Transcript("", True)

    import app.adapters.stt.whisper as whisper_mod

    monkeypatch.setattr(whisper_mod, "FasterWhisperSTT", FakeWhisper)
    monkeypatch.delenv("STT_PROVIDER", raising=False)
    adapter = stt_pkg.create_stt_adapter()
    assert adapter.name == "whisper"
    assert constructed == ["whisper"]

def test_disconnect_on_ready_closes_initialized_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    closed = []

    def tracked(adapter):
        async def close() -> None:
            closed.append(adapter.name)
        monkeypatch.setattr(adapter, "aclose", close)
        return lambda: adapter

    monkeypatch.setattr(main, "create_stt_adapter", tracked(StubSTT()))
    monkeypatch.setattr(main, "create_llm_adapter", tracked(StubLLM()))
    monkeypatch.setattr(main, "create_tts_adapter", tracked(StubTTS()))

    messages: asyncio.Queue[ASGIMessage] = asyncio.Queue()
    messages.put_nowait({"type": "websocket.connect"})

    async def send(message: ASGIMessage) -> None:
        if message["type"] == "websocket.send":
            assert '"value":"ready"' in message["text"]
            # Starlette maps this transport failure to WebSocketDisconnect(1006).
            raise OSError("peer disconnected")

    ws = WebSocket({"type": "websocket"}, messages.get, send)
    asyncio.run(asyncio.wait_for(main.ws_transcribe(ws), timeout=3))
    assert closed == ["stub-tts", "stub-llm", "stub-stt"]


@pytest.mark.parametrize("disconnect_from_send", [False, True])
def test_disconnect_cancels_active_generation(
    monkeypatch: pytest.MonkeyPatch, disconnect_from_send: bool,
) -> None:
    async def scenario() -> None:
        messages: asyncio.Queue[ASGIMessage] = asyncio.Queue()
        generation_closed = asyncio.Event()
        speech_closed = asyncio.Event()
        messages.put_nowait({"type": "websocket.connect"})
        messages.put_nowait({"type": "websocket.receive", "bytes": b"\0\0"})

        class StreamingLLM(StubLLM):
            async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
                try:
                    while True:
                        yield "Sentence. "
                finally:
                    generation_closed.set()

        class StreamingTTS(StubTTS):
            async def synthesize(self, text: str) -> AsyncIterator[bytes]:
                try:
                    if not disconnect_from_send:
                        messages.put_nowait({"type": "websocket.disconnect", "code": 1006})
                        await asyncio.Event().wait()
                    yield b"\0\0"
                finally:
                    speech_closed.set()

        monkeypatch.setattr(main, "create_stt_adapter", StubSTT)
        monkeypatch.setattr(main, "create_llm_adapter", StreamingLLM)
        monkeypatch.setattr(main, "create_tts_adapter", StreamingTTS)

        async def send(message: ASGIMessage) -> None:
            if message.get("bytes"):
                raise OSError("peer disconnected")

        ws = WebSocket({"type": "websocket"}, messages.get, send)
        await asyncio.wait_for(main.ws_transcribe(ws), timeout=3)
        await asyncio.wait_for(generation_closed.wait(), timeout=3)
        await asyncio.wait_for(speech_closed.wait(), timeout=3)

    asyncio.run(scenario())


def test_disconnect_during_provider_load_skips_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        from threading import Event

        loading = asyncio.Event()
        disconnected = asyncio.Event()
        release = Event()
        loop = asyncio.get_running_loop()
        sent = []
        stt_closed = asyncio.Event()

        class SlowSTT(StubSTT):
            async def aclose(self) -> None:
                stt_closed.set()

        def load() -> SlowSTT:
            loop.call_soon_threadsafe(loading.set)
            assert release.wait(timeout=3)
            return SlowSTT()

        async def receive() -> ASGIMessage:
            if not sent:
                return {"type": "websocket.connect"}
            await loading.wait()
            disconnected.set()
            return {"type": "websocket.disconnect", "code": 1006}

        async def send(message: ASGIMessage) -> None:
            sent.append(message["type"])

        monkeypatch.setattr(main, "create_stt_adapter", load)
        ws = WebSocket({"type": "websocket"}, receive, send)
        session = asyncio.create_task(main.ws_transcribe(ws))
        try:
            await asyncio.wait_for(disconnected.wait(), timeout=2)
        finally:
            release.set()
        await asyncio.wait_for(session, timeout=2)
        assert stt_closed.is_set()
        assert sent == ["websocket.accept"]

    asyncio.run(scenario())


def test_uncertain_booking_answer_keeps_stage_and_skips_rag(monkeypatch) -> None:
    def unexpected(*_args):
        pytest.fail("An uncertain booking input must not reach retrieval")

    monkeypatch.setattr(main, "retrieve_context", unexpected)
    flow = main.BookingFlow(stage="reschedule_confirm", appointment_id=42)
    ws = _FakeWS()
    llm = StubLLM()
    history = []
    asyncio.run(main._respond(ws, llm, StubTTS(), history, "yes", flow, uncertain=True))

    assert flow.stage == "reschedule_confirm"
    assert flow.appointment_id == 42
    assert llm.seen is None
    assert "repeat" in history[-1].content


def test_greeting_keeps_history_without_rag_or_llm(monkeypatch) -> None:
    def unexpected(*_args):
        pytest.fail("A greeting must not reach retrieval")

    monkeypatch.setattr(main, "retrieve_context", unexpected)
    ws = _FakeWS()
    llm = StubLLM()
    history = []
    asyncio.run(main._respond(ws, llm, StubTTS(), history, "Good morning!", main.BookingFlow()))

    assert llm.seen is None
    assert [message.role for message in history] == ["user", "assistant"]
    assert "help" in history[-1].content


def test_sentence_split_leaves_titles_and_decimals_together() -> None:
    assert main._take_sentence("Ask Dr.") == ("", "Ask Dr.")
    assert main._take_sentence("Ask Dr. Shah about the 2.5 value.") == (
        "Ask Dr. Shah about the 2.5 value.", "",
    )


def test_interrupt_is_acknowledged_after_audio_finishes(monkeypatch) -> None:
    async def scenario() -> None:
        messages: asyncio.Queue[ASGIMessage] = asyncio.Queue()
        messages.put_nowait({"type": "websocket.connect"})
        messages.put_nowait({"type": "websocket.receive", "bytes": b"\x00\x00"})
        acknowledged = asyncio.Event()

        class ListeningSTT(StubSTT):
            async def transcribe(self, audio: AsyncIterator[bytes]) -> AsyncIterator[Transcript]:
                async for _ in audio:
                    yield Transcript("hello", True)

        monkeypatch.setattr(main, "create_stt_adapter", ListeningSTT)
        monkeypatch.setattr(main, "create_llm_adapter", StubLLM)
        monkeypatch.setattr(main, "create_tts_adapter", StubTTS)

        async def send(message: ASGIMessage) -> None:
            if message["type"] != "websocket.send" or "text" not in message:
                return
            payload = json.loads(message["text"])
            if payload.get("type") != "assistant":
                return
            if payload.get("phase") == "done":
                messages.put_nowait({"type": "websocket.receive", "text": '{"type":"interrupt"}'})
            if payload.get("phase") == "interrupted":
                acknowledged.set()
                messages.put_nowait({"type": "websocket.disconnect", "code": 1000})

        ws = WebSocket({"type": "websocket"}, messages.get, send)
        await asyncio.wait_for(main.ws_transcribe(ws), timeout=3)
        assert acknowledged.is_set()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("model", "configured", "expected"),
    [
        ("gemini-2.5-flash", None, "none"),
        ("gemini-2.5-flash", "low", "low"),
        ("gemini-2.5-flash", "default", None),
        ("gemini-custom", None, None),
    ],
)
def test_gemini_streaming_reasoning_setting(monkeypatch, model, configured, expected) -> None:
    from types import SimpleNamespace

    from openai import omit

    from app.adapters.llm.gemini import GeminiLLM

    async def run() -> None:
        if configured is None:
            monkeypatch.delenv("GEMINI_REASONING_EFFORT", raising=False)
        else:
            monkeypatch.setenv("GEMINI_REASONING_EFFORT", configured)

        class Completion:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return None

            async def __aiter__(self):
                yield SimpleNamespace(choices=[])
                yield SimpleNamespace(choices=[SimpleNamespace(
                    delta=SimpleNamespace(content="Hello."),
                )])

        async def create(**kwargs):
            assert kwargs["model"] == model and kwargs["stream"] is True
            assert kwargs["reasoning_effort"] == (expected if expected else omit)
            return Completion()

        gemini = GeminiLLM(api_key="unit-test-placeholder", model=model)
        monkeypatch.setattr(gemini._client.chat.completions, "create", create)
        try:
            assert [part async for part in gemini.stream([])] == ["Hello."]
        finally:
            await gemini.aclose()

    asyncio.run(run())
