# Agent Workflow — realtime-voice-agent-demo

## Repository shape

```
realtime-voice-agent-demo/
├── backend/            # Python 3.12 / FastAPI (uv-managed)
│   ├── app/
│   │   ├── main.py     # FastAPI app, WebSocket routes, /tts/synthesize, session state
│   │   └── adapters/   # provider adapters (boundary — keep providers here)
│   │       ├── stt/    # whisper, deepgram
│   │       ├── llm/    # ollama, kimi, claude
│   │       └── tts/    # inworld, piper, hume (scaffold)
│   └── pyproject.toml
├── frontend/           # Next.js 15 / React / TS (npm)
│   └── app/            # main demo page + /bench eval harness
└── docs/
```

## What's built

Each capability produces something runnable on its own:

1. **Skeleton** — mic → WebSocket → echo round-trip (`/ws/echo`).
2. **STT streaming** — mic → transcript in the UI.
3. **LLM response** — transcript → streaming LLM → caption.
4. **TTS streaming** — reply → per-sentence streaming audio playback.
5. **Talk / translate modes** — same pipeline, switchable from the UI.
6. **Barge-in** — interrupt the assistant mid-reply by speaking.
7. **Local + cloud TTS** — Piper (local) and Inworld (cloud), switchable live.
8. **TTS eval harness** — `/bench` compares Inworld vs Piper on one fixed script.

## Conventions

- Providers live behind `backend/app/adapters/`. A provider swap must not change callers.
- The conversation state (IDLE / LISTENING / RESPONDING / INTERRUPT) is handled in
  `main.py`'s WebSocket handler and `_Session` — keep that logic there.
- barge-in must cancel: TTS playback + in-flight TTS + LLM generation, then reset stream state.
- Secrets via `.env` (gitignored). Never commit keys or audio samples.

## Local run

```sh
make install
make dev        # backend :8000, frontend :3000
```
