import asyncio

import numpy as np
import pytest

from app.adapters.stt import whisper
from app.adapters.tts.piper import PiperTTS


@pytest.fixture
def recognizer(monkeypatch):
    monkeypatch.setattr(whisper, "_get_model", lambda _: None)
    monkeypatch.setenv("WHISPER_PARTIALS", "0")
    adapter = whisper.FasterWhisperSTT()
    decoded = []

    async def decode(samples):
        decoded.append(samples)
        return "yes", False

    monkeypatch.setattr(adapter, "_run", decode)
    return adapter, decoded


def _frame(amplitude):
    return np.full(512, amplitude, dtype="<i2")


def _recognize(adapter, audio):
    async def packets():
        # Includes odd-length fragments and partial PCM frames.
        for offset in range(0, len(audio), 877):
            yield audio[offset:offset + 877]

    async def run():
        return [turn async for turn in adapter.transcribe(packets())]

    return asyncio.run(run())


def test_short_turn_keeps_quiet_first_and_last_sounds(recognizer):
    adapter, decoded = recognizer
    signal = np.concatenate(
        [_frame(0)] * 8 + [_frame(65)] + [_frame(400)] * 5
        + [_frame(60)] + [_frame(0)] * 18,
    )
    turns = _recognize(adapter, signal.tobytes())
    assert [turn.text for turn in turns] == ["yes"]
    kept = np.rint(decoded[0] * 32768).astype(np.int16)
    assert kept.size == (6 + 3 + 5) * 512
    assert np.flatnonzero(kept == 65)[-1] < np.flatnonzero(kept == 400)[0]
    assert np.flatnonzero(kept == 60)[0] > np.flatnonzero(kept == 400)[-1]


def test_pause_inside_request_does_not_split_the_turn(recognizer):
    adapter, decoded = recognizer
    frames = [_frame(600)] * 15 + [_frame(0)] * 13 + [_frame(600)] * 8 + [_frame(0)] * 18
    assert len(_recognize(adapter, np.concatenate(frames).tobytes())) == 1
    assert len(decoded) == 1
    assert np.count_nonzero(decoded[0]) == 23 * 512


def test_quiet_room_and_single_spike_never_invoke_decoder(recognizer):
    adapter, decoded = recognizer
    frames = [_frame(29)] * 50 + [_frame(1800)] + [_frame(29)] * 50
    assert _recognize(adapter, np.concatenate(frames).tobytes()) == []
    assert decoded == []


def test_flush_preserves_incomplete_final_pcm_frame(recognizer):
    adapter, decoded = recognizer
    signal = np.concatenate([_frame(500)] * 5 + [_frame(75)[:240]])
    assert len(_recognize(adapter, signal.tobytes())) == 1
    assert decoded[0].size == signal.size


def test_piper_yields_first_sentence_before_rendering_next(monkeypatch):
    piper = PiperTTS.__new__(PiperTTS)
    rendered = []

    def render(_):
        rendered.append(1)
        yield b"first"
        rendered.append(2)
        yield b"second"

    monkeypatch.setattr(piper, "_render", render)

    async def run():
        stream = piper.synthesize("First. Second.")
        assert await anext(stream) == b"first"
        assert rendered == [1]
        await stream.aclose()
        assert rendered == [1]

    asyncio.run(run())
