# AGENTS.md — realtime-voice-agent-demo

## Project

Realtime voice AI assistant demo. Microphone → STT → LLM → TTS → streaming playback,
with barge-in and a talk/translate mode switch. Runs locally (no deploy), and can
run fully offline with local providers.

## Stack

- Frontend: Next.js 15 / React / TypeScript (mic capture, waveform/caption UI, WS client)
- Backend: Python 3.12 / FastAPI (audio I/O, provider adapters, conversation state, WS hub)
- Transport: WebSocket (bidirectional) between frontend and backend
- Providers, each pluggable behind an adapter and switchable at runtime:
  - STT: whisper (local), deepgram (cloud)
  - LLM: ollama (local), kimi, claude
  - TTS: inworld (cloud), piper (local); hume is scaffolded only
- Package mgmt: `uv` (backend), `npm` (frontend)

## Rules

- Keep STT/TTS providers behind the adapter boundary in `backend/app/adapters/`. Provider swaps must not leak past it.
- `barge-in` is a first-class state-machine transition — do not bolt it on. Cancel TTS playback, in-flight TTS, and LLM generation, then reset stream state.
- Do not commit `.env`, API keys, or audio sample data.
- English docs default; Japanese in `*.ja.md`.
- All code must pass `make lint` and `make typecheck` before merge.

## Build & Test

- `make install` — install backend (uv) and frontend (npm) deps
- `make dev` — start backend (:8000) and frontend (:3000)
- `make lint` / `make typecheck` / `make test`

See [docs/agent-workflow.md](docs/agent-workflow.md) for the detailed workflow.
