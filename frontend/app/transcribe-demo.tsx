"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AuraOrb } from "./aura-orb";
import { PlaybackGate } from "./playback-gate";

const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000/ws";
const CONNECTION_TIMEOUT_MS = 20_000;
const TARGET_SAMPLE_RATE = 16000;
const PLAYBACK_TAIL_SECONDS = 0.12;
// Defaults so the controls are usable before connecting; the backend confirms
// the real list on connect.
const DEFAULT_LANGUAGES: Record<string, string> = {
  en: "English",
  ja: "Japanese",
  zh: "Chinese",
  ko: "Korean",
  es: "Spanish",
  fr: "French",
};
const DEFAULT_TTS_PROVIDERS = ["piper", "inworld"];

type Status = "idle" | "connecting" | "live" | "error";
type Mode = "talk" | "translate";

type Turn = { role: "user" | "assistant"; text: string };
type Languages = Record<string, string>;

type ServerMsg =
  | { type: "transcript"; text: string; isFinal: boolean }
  | { type: "assistant"; phase: "start" }
  | { type: "assistant"; delta: string }
  | { type: "assistant"; phase: "done" }
  | { type: "assistant"; phase: "interrupted" }
  | { type: "audio"; phase: "start"; text: string }
  | { type: "audio"; phase: "end" }
  | { type: "audio"; phase: "error"; message: string }
  | { type: "config"; mode: Mode; source: string; target: string; tts: string }
  | {
      type: "state";
      value: string;
      stt?: string;
      llm?: string;
      tts?: string;
      languages?: Languages;
      ttsProviders?: string[];
      mode?: Mode;
      source?: string;
      target?: string;
    }
  | { type: "error"; message: string };

/**
 * Voice-agent demo: capture mic -> 16kHz PCM -> WebSocket -> STT -> LLM -> TTS,
 * with playback. Two modes, switchable live from the UI: "chat" (the assistant
 * converses) and "translate" (the LLM acts as a simultaneous interpreter and
 * the reply is the translation, spoken in the target language).
 */
export function TranscribeDemo() {
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [stt, setStt] = useState<string | null>(null);
  const [llm, setLlm] = useState<string | null>(null);
  const [tts, setTts] = useState<string | null>(null);
  const [level, setLevel] = useState(0);
  const [speaking, setSpeaking] = useState(false);
  const [partial, setPartial] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [languages, setLanguages] = useState<Languages>(DEFAULT_LANGUAGES);
  const [ttsProviders, setTtsProviders] = useState<string[]>(DEFAULT_TTS_PROVIDERS);
  const [mode, setMode] = useState<Mode>("talk");
  const [source, setSource] = useState("ja");
  const [target, setTarget] = useState("en");
  const [ttsProvider, setTtsProvider] = useState("piper");

  // Keep the latest desired config in a ref so the ready handler can send it
  // without being re-created (and without stale closure values).
  const desiredConfigRef = useRef({
    mode: "talk" as Mode,
    source: "ja",
    target: "en",
    tts: "piper",
  });

  const attemptRef = useRef(0);
  const timeoutRef = useRef<number | null>(null);
  const readyRef = useRef(false);
  const wsRef = useRef<WebSocket | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const playHeadRef = useRef(0); // scheduled-playback cursor (seconds)
  const sourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set()); // active TTS playback
  const speakingRef = useRef(false); // mirror of `speaking` for the audio callback
  const waitingForInterruptRef = useRef(false);

  // Barge-in: stop playing the assistant and tell the backend to cancel its
  // in-flight reply. Called when the user starts talking over the assistant.
  const stopPlayback = useCallback(() => {
    for (const src of sourcesRef.current) {
      try {
        src.stop();
      } catch {
        // already stopped
      }
    }
    sourcesRef.current.clear();
    const ctx = audioCtxRef.current;
    if (ctx) playHeadRef.current = ctx.currentTime;
    speakingRef.current = false;
    setSpeaking(false);
  }, []);

  const bargeIn = useCallback(() => {
    stopPlayback();
    const ws = wsRef.current;
    if (readyRef.current && ws?.readyState === WebSocket.OPEN) {
      waitingForInterruptRef.current = true;
      ws.send(JSON.stringify({ type: "interrupt" }));
    }
  }, [stopPlayback]);

  // Apply a control change. Works before *and* after connecting: it updates the
  // desired config (sent to the backend on connect) and, if already connected,
  // pushes it live. Each control passes only the field it changes.
  const applyConfig = useCallback(
    (patch: Partial<{ mode: Mode; source: string; target: string; tts: string }>) => {
      const cfg = { ...desiredConfigRef.current, ...patch };
      desiredConfigRef.current = cfg;
      if (patch.mode !== undefined) setMode(patch.mode);
      if (patch.source !== undefined) setSource(patch.source);
      if (patch.target !== undefined) setTarget(patch.target);
      if (patch.tts !== undefined) setTtsProvider(patch.tts);
      const ws = wsRef.current;
      if (readyRef.current && ws?.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "config", ...cfg }));
        setTurns([]); // a config switch starts a fresh transcript view
      }
    },
    [],
  );

  const stop = useCallback(() => {
    attemptRef.current += 1;
    readyRef.current = false;
    waitingForInterruptRef.current = false;
    if (timeoutRef.current !== null) window.clearTimeout(timeoutRef.current);
    timeoutRef.current = null;
    const ws = wsRef.current;
    wsRef.current = null;
    if (ws && ws.readyState < WebSocket.CLOSING) ws.close();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    sourcesRef.current.clear();
    speakingRef.current = false;
    void audioCtxRef.current?.close();
    audioCtxRef.current = null;
    setStatus("idle");
    setLevel(0);
    setSpeaking(false);
  }, []);

  const start = useCallback(async () => {
    setError(null);
    setPartial("");
    setTurns([]);
    setStt(null);
    setLlm(null);
    setTts(null);
    setStatus("connecting");
    const attempt = ++attemptRef.current;
    const fail = (message: string) => {
      if (attemptRef.current !== attempt) return;
      stop();
      setError(message);
      setStatus("error");
    };
    timeoutRef.current = window.setTimeout(() => {
      fail("Timed out waiting for microphone access or for the voice server to be ready.");
    }, CONNECTION_TIMEOUT_MS);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
      });
      if (attemptRef.current !== attempt) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      streamRef.current = stream;

      const audioCtx = new AudioContext();
      audioCtxRef.current = audioCtx;
      playHeadRef.current = audioCtx.currentTime;
      const inRate = audioCtx.sampleRate;
      const gate = new PlaybackGate();
      let lastSpeechAt: number | null = null;
      let lastTurnSpeechAt: number | null = null;
      let firstAudioPending = false;
      let playbackTailUntil = 0;
      const outputLatency = audioCtx.outputLatency || audioCtx.baseLatency || 0;

      const ws = new WebSocket(WS_URL);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onerror = () => fail("Could not connect to the voice server.");
      ws.onclose = (event) => {
        if (attemptRef.current !== attempt) return;
        if (readyRef.current && event.code === 1000) {
          stop();
        } else {
          fail(`Voice connection closed (code ${event.code}).`);
        }
      };

      ws.onmessage = (ev) => {
        if (attemptRef.current !== attempt || ws.readyState !== WebSocket.OPEN) return;
        // Binary frames are TTS audio (16kHz Int16 PCM) — schedule playback.
        if (typeof ev.data !== "string") {
          if (waitingForInterruptRef.current) return;
          const startAt = playPcm(audioCtx, playHeadRef, sourcesRef.current, ev.data as ArrayBuffer);
          if (startAt === undefined) return;
          playbackTailUntil = playHeadRef.current + outputLatency + PLAYBACK_TAIL_SECONDS;
          if (!speakingRef.current) {
            speakingRef.current = true;
            setSpeaking(true);
          }
          if (firstAudioPending && lastTurnSpeechAt !== null) {
            const delayMs = Math.round((startAt + outputLatency - lastTurnSpeechAt) * 1000);
            console.info("Voice latency | estimated speech end to first audible PCM (ms)", delayMs);
          }
          firstAudioPending = false;
          return;
        }
        const msg = JSON.parse(ev.data) as ServerMsg;
        if (msg.type === "state" && msg.value === "ready" && !readyRef.current) {
          if (timeoutRef.current !== null) window.clearTimeout(timeoutRef.current);
          timeoutRef.current = null;
          readyRef.current = true;
          ws.send(JSON.stringify({ type: "config", ...desiredConfigRef.current }));
          setStatus("live");
          // Display info + available options. Do NOT adopt the server's default
          // mode/source/target here — the user's pre-Start selection wins; the
          // backend confirms it via the `config` echo right after.
          if (msg.stt) setStt(msg.stt);
          if (msg.llm) setLlm(msg.llm);
          if (msg.tts) setTts(msg.tts);
          if (msg.languages) setLanguages(msg.languages);
          if (msg.ttsProviders) setTtsProviders(msg.ttsProviders);
        } else if (msg.type === "config") {
          // Authoritative confirmation of the active config.
          setMode(msg.mode);
          setSource(msg.source);
          setTarget(msg.target);
          setTts(msg.tts);
          setTtsProvider(msg.tts);
          desiredConfigRef.current = {
            mode: msg.mode,
            source: msg.source,
            target: msg.target,
            tts: msg.tts,
          };
        } else if (msg.type === "error") {
          if (readyRef.current) setError(msg.message);
          else fail(msg.message);
        } else if (msg.type === "transcript") {
          if (msg.isFinal) {
            lastTurnSpeechAt = lastSpeechAt;
            lastSpeechAt = null;
            setPartial("");
            setTurns((t) => [...t, { role: "user", text: msg.text }]);
          } else {
            setPartial(msg.text);
          }
        } else if (msg.type === "assistant") {
          if ("phase" in msg && msg.phase === "start") {
            waitingForInterruptRef.current = false;
            firstAudioPending = true;
            setTurns((t) => [...t, { role: "assistant", text: "" }]);
          } else if ("phase" in msg && msg.phase === "interrupted") {
            // Server confirmed the barge-in; mark the turn as cut off.
            waitingForInterruptRef.current = false;
            playbackTailUntil = 0;
            firstAudioPending = false;
            stopPlayback();
            setTurns((t) => {
              const last = t[t.length - 1];
              if (!last || last.role !== "assistant") return t;
              return [...t.slice(0, -1), { ...last, text: last.text + " —" }];
            });
          } else if ("delta" in msg) {
            setTurns((t) => {
              const last = t[t.length - 1];
              if (!last || last.role !== "assistant") return t;
              const updated = { ...last, text: last.text + msg.delta };
              return [...t.slice(0, -1), updated];
            });
          }
        } else if (msg.type === "audio") {
          if (msg.phase === "error") setError(`TTS: ${msg.message}`);
        }
      };

      const source = audioCtx.createMediaStreamSource(stream);
      const processor = audioCtx.createScriptProcessor(2048, 1, 1);
      source.connect(processor);
      processor.connect(audioCtx.destination);
      processor.onaudioprocess = (e) => {
        if (attemptRef.current !== attempt || !readyRef.current) return;
        const input = e.inputBuffer.getChannelData(0);
        let sum = 0;
        for (let i = 0; i < input.length; i++) sum += input[i] * input[i];
        const rms = Math.sqrt(sum / input.length);
        setLevel(Math.min(1, rms * 4));

        const playing = audioCtx.currentTime < playbackTailUntil;
        if (!playing && speakingRef.current) {
          speakingRef.current = false;
          setSpeaking(false);
        }
        if (ws.readyState !== WebSocket.OPEN) return;
        const pcm = downsampleToInt16(input, inRate, TARGET_SAMPLE_RATE);
        const { frames, interrupt } = gate.capture(pcm, rms, playing);
        if (interrupt) {
          playbackTailUntil = 0;
          bargeIn();
        }
        if (frames.length && rms >= 0.0035) lastSpeechAt = audioCtx.currentTime;
        for (const frame of frames) ws.send(frame.buffer as ArrayBuffer);
      };
    } catch (err) {
      fail(err instanceof Error ? err.message : String(err));
    }
  }, [bargeIn, stop, stopPlayback]);

  useEffect(() => () => stop(), [stop]);

  const live = status === "live";

  return (
    <section className="aura-voice-section">
      <div className="aura-voice-hero">
        <AuraOrb
          active={status === "live"}
          speaking={speaking}
          level={level}
        />

        <div className="aura-voice-status">
          <span className={`aura-status-dot aura-status-${status}`} />
          {speaking
            ? "MediVoice is speaking"
            : status === "live"
              ? "Listening to you"
              : status === "connecting"
                ? "Connecting your microphone"
                : status === "error"
                  ? "Connection unavailable"
                  : "Ready when you are"}
        </div>

        <p className="aura-voice-hint">
          {status === "live"
            ? "Speak naturally. I'm here to help."
            : "Start a conversation whenever you're ready."}
        </p>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
        <button
          onClick={live || status === "connecting" ? stop : start}
          style={{
            padding: "12px 24px",
            fontSize: 16,
            borderRadius: 10,
            border: "none",
            cursor: "pointer",
            background: live ? "#e5484d" : "#3b82f6",
            color: "white",
          }}
        >
          {live ? "Stop" : status === "connecting" ? "Connecting…" : "Start mic"}
        </button>
        {stt && (
          <span style={{ fontSize: 13, color: "#9aa0a6" }}>
            STT: <strong style={{ color: "#c5c9cd" }}>{stt}</strong>
          </span>
        )}
        {llm && (
          <span style={{ fontSize: 13, color: "#9aa0a6" }}>
            LLM: <strong style={{ color: "#c5c9cd" }}>{llm}</strong>
          </span>
        )}
        {tts && (
          <span style={{ fontSize: 13, color: "#9aa0a6" }}>
            TTS: <strong style={{ color: "#c5c9cd" }}>{tts}</strong>
          </span>
        )}
        {speaking && (
          <span style={{ fontSize: 13, color: "#22c55e" }}>🔊 speaking…</span>
        )}
      </div>

      <details className="aura-settings">
        <summary className="aura-settings-trigger">
          <span>⚙ Advanced settings</span>
          <span className="aura-settings-chevron">⌄</span>
        </summary>
        <div className="aura-settings-content">
      {/* Controls — usable before AND during a session. Choose before Start;
          change live afterwards. */}
      <div
        style={{
          marginTop: 20,
          display: "flex",
          alignItems: "center",
          gap: 12,
          flexWrap: "wrap",
        }}
      >
        <Segmented<Mode>
          options={[
            { value: "talk", label: "💬 Talk" },
            { value: "translate", label: "🌐 Translate" },
          ]}
          value={mode}
          onChange={(m) => applyConfig({ mode: m })}
        />

        {mode === "translate" && (
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <LangSelect
              value={source}
              languages={languages}
              disabled={false}
              onChange={(v) => applyConfig({ source: v })}
            />
            <span style={{ color: "#6b7280" }}>→</span>
            <LangSelect
              value={target}
              languages={languages}
              disabled={false}
              onChange={(v) => applyConfig({ target: v })}
            />
          </div>
        )}
      </div>

      {/* TTS voice engine — cloud (Inworld) or local (Piper). */}
      <div
        style={{
          marginTop: 12,
          display: "flex",
          alignItems: "center",
          gap: 10,
          flexWrap: "wrap",
        }}
      >
        <span style={{ fontSize: 13, color: "#9aa0a6" }}>Voice:</span>
        <Segmented<string>
          options={ttsProviders.map((p) => ({
            value: p,
            label: p === "piper" ? "Piper (local)" : "Inworld (cloud)",
          }))}
          value={ttsProvider}
          onChange={(p) => applyConfig({ tts: p })}
        />
      </div>

        </div>
      </details>

      <div className="aura-conversation">
        <div className="aura-conversation-header">
          <div className="aura-conversation-title">
            <span className="aura-conversation-icon">✦</span>
            <span>Conversation</span>
          </div>
          <span className="aura-conversation-tag">VOICE SESSION</span>
        </div>

        <div className="aura-conversation-messages" aria-live="polite">
          {turns.length === 0 && !partial && (
            <div className="aura-conversation-empty">
              <div className="aura-empty-symbol">✧</div>
              <h3>Your conversation starts here</h3>
              <p>
                {mode === "translate"
                  ? `Speak in ${languages[source] ?? source}. Your words will be translated into ${languages[target] ?? target}.`
                  : "Tap Start mic and speak naturally. Your conversation will appear here."}
              </p>
            </div>
          )}

          {turns.map((turn, i) => (
            <div
              key={i}
              className={`aura-message aura-message-${turn.role}`}
            >
              {turn.role === "assistant" && (
                <div className="aura-message-avatar" aria-hidden="true">
                  ✦
                </div>
              )}

              <div className="aura-message-body">
                <div className="aura-message-label">
                  {turn.role === "assistant" ? "MediVoice AI" : "You"}
                </div>
                <div className="aura-message-text">
                  {turn.text || (turn.role === "assistant" ? "…" : "")}
                </div>
              </div>
            </div>
          ))}

          {partial && (
            <div className="aura-message aura-message-user aura-message-partial">
              <div className="aura-message-body">
                <div className="aura-message-label">You · Listening</div>
                <div className="aura-message-text">{partial}…</div>
              </div>
            </div>
          )}
        </div>
      </div>

      {error && <p style={{ color: "#f87171", marginTop: 16 }}>⚠ {error}</p>}
    </section>
  );
}

function Segmented<T extends string>({
  options,
  value,
  onChange,
}: {
  options: { value: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <div style={{ display: "inline-flex", borderRadius: 8, overflow: "hidden" }}>
      {options.map((opt) => (
        <button
          key={opt.value}
          onClick={() => onChange(opt.value)}
          style={{
            padding: "8px 16px",
            fontSize: 14,
            border: "none",
            cursor: "pointer",
            background: value === opt.value ? "#3b82f6" : "#1c2128",
            color: value === opt.value ? "white" : "#9aa0a6",
          }}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

function LangSelect({
  value,
  languages,
  disabled,
  onChange,
}: {
  value: string;
  languages: Languages;
  disabled: boolean;
  onChange: (v: string) => void;
}) {
  return (
    <select
      value={value}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value)}
      style={{
        padding: "6px 10px",
        borderRadius: 8,
        background: "#1c2128",
        color: "#e8eaed",
        border: "1px solid #2a313b",
        fontSize: 14,
      }}
    >
      {Object.entries(languages).map(([code, name]) => (
        <option key={code} value={code}>
          {name}
        </option>
      ))}
    </select>
  );
}

/**
 * Schedule a 16kHz Int16 PCM chunk for gap-free playback. We build a 16kHz
 * AudioBuffer (WebAudio resamples to the context rate on output) and start it at
 * the running playhead so consecutive chunks butt up against each other.
 */
function playPcm(
  ctx: AudioContext,
  playHead: { current: number },
  sources: Set<AudioBufferSourceNode>,
  data: ArrayBuffer,
): number | undefined {
  const int16 = new Int16Array(data);
  if (int16.length === 0) return;
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 0x8000;
  const buf = ctx.createBuffer(1, float32.length, TARGET_SAMPLE_RATE);
  buf.copyToChannel(float32, 0);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  // Track the node so barge-in can stop in-flight playback.
  sources.add(src);
  src.onended = () => sources.delete(src);
  const startAt = Math.max(ctx.currentTime, playHead.current);
  src.start(startAt);
  playHead.current = startAt + buf.duration;
  return startAt;
}

/** Linear-interpolation downsample of Float32 [-1,1] audio to Int16 PCM. */
function downsampleToInt16(
  input: Float32Array,
  inRate: number,
  outRate: number,
): Int16Array {
  if (outRate >= inRate) {
    const out = new Int16Array(input.length);
    for (let i = 0; i < input.length; i++) out[i] = clampInt16(input[i]);
    return out;
  }
  const ratio = inRate / outRate;
  const outLen = Math.floor(input.length / ratio);
  const out = new Int16Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio;
    const lo = Math.floor(pos);
    const hi = Math.min(lo + 1, input.length - 1);
    const frac = pos - lo;
    const sample = input[lo] * (1 - frac) + input[hi] * frac;
    out[i] = clampInt16(sample);
  }
  return out;
}

function clampInt16(sample: number): number {
  const s = Math.max(-1, Math.min(1, sample));
  return s < 0 ? s * 0x8000 : s * 0x7fff;
}
