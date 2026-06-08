"use client";

import { useCallback, useEffect, useState } from "react";

// Derive the HTTP base from the WS URL (…/ws -> http://host).
const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000/ws";
const HTTP_BASE = WS_URL.replace(/^ws/, "http").replace(/\/ws$/, "");

const DEFAULT_SCRIPT =
  "Welcome! I'm so glad you're here. Let me show you what this can do — " +
  "it's genuinely exciting.";

type Metrics = {
  firstAudioMs: number;
  totalMs: number;
  audioSeconds: number;
  bytes: number;
};

type Result =
  | { status: "running" }
  | { status: "error"; message: string }
  | { status: "done"; metrics: Metrics; wavUrl: string; notes: string };

/**
 * TTS eval harness: send one fixed script to every provider and lay the results
 * out side by side — measured latency + a play button for the audio + a free
 * notes field for the subjective quality call. The point is to choose a TTS by
 * your own measurement, not the vendor's benchmark.
 */
export function TtsBench() {
  const [script, setScript] = useState(DEFAULT_SCRIPT);
  const [providers, setProviders] = useState<string[]>([]);
  const [results, setResults] = useState<Record<string, Result>>({});
  const [running, setRunning] = useState(false);

  useEffect(() => {
    fetch(`${HTTP_BASE}/tts/providers`)
      .then((r) => r.json())
      .then((d: { providers: string[] }) => setProviders(d.providers))
      .catch(() => setProviders(["inworld", "piper"]));
  }, []);

  const runOne = useCallback(
    async (provider: string, text: string) => {
      setResults((r) => ({ ...r, [provider]: { status: "running" } }));
      try {
        const resp = await fetch(`${HTTP_BASE}/tts/synthesize`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, provider }),
        });
        const body = await resp.json();
        if (body.error) {
          setResults((r) => ({
            ...r,
            [provider]: { status: "error", message: body.error },
          }));
          return;
        }
        const wavUrl = `data:audio/wav;base64,${body.wavBase64}`;
        setResults((r) => ({
          ...r,
          [provider]: {
            status: "done",
            metrics: body.metrics,
            wavUrl,
            notes: "",
          },
        }));
      } catch (e) {
        setResults((r) => ({
          ...r,
          [provider]: {
            status: "error",
            message: e instanceof Error ? e.message : String(e),
          },
        }));
      }
    },
    [],
  );

  const runAll = useCallback(async () => {
    const text = script.trim();
    if (!text || providers.length === 0) return;
    setRunning(true);
    // Run providers concurrently — each is independent.
    await Promise.all(providers.map((p) => runOne(p, text)));
    setRunning(false);
  }, [script, providers, runOne]);

  return (
    <section style={{ marginTop: 28 }}>
      <textarea
        value={script}
        onChange={(e) => setScript(e.target.value)}
        rows={3}
        style={{
          width: "100%",
          boxSizing: "border-box",
          padding: 12,
          borderRadius: 10,
          background: "#11151c",
          color: "#e8eaed",
          border: "1px solid #1c2128",
          fontSize: 15,
          lineHeight: 1.5,
          resize: "vertical",
        }}
      />
      <button
        onClick={runAll}
        disabled={running}
        style={{
          marginTop: 12,
          padding: "10px 22px",
          fontSize: 15,
          borderRadius: 10,
          border: "none",
          cursor: running ? "default" : "pointer",
          background: running ? "#1c2128" : "#3b82f6",
          color: "white",
        }}
      >
        {running ? "Synthesizing…" : "Run all providers"}
      </button>

      <div
        style={{
          marginTop: 24,
          display: "grid",
          gridTemplateColumns: `repeat(${Math.max(providers.length, 1)}, 1fr)`,
          gap: 16,
        }}
      >
        {providers.map((p) => (
          <ProviderCard
            key={p}
            provider={p}
            result={results[p]}
            onNotes={(notes) =>
              setResults((r) => {
                const cur = r[p];
                if (!cur || cur.status !== "done") return r;
                return { ...r, [p]: { ...cur, notes } };
              })
            }
          />
        ))}
      </div>
    </section>
  );
}

function ProviderCard({
  provider,
  result,
  onNotes,
}: {
  provider: string;
  result: Result | undefined;
  onNotes: (notes: string) => void;
}) {
  const isLocal = provider === "piper";
  return (
    <div
      style={{
        padding: 16,
        borderRadius: 12,
        background: "#11151c",
        border: "1px solid #1c2128",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <strong style={{ fontSize: 16 }}>{provider}</strong>
        <span style={{ fontSize: 12, color: "#6b7280" }}>
          {isLocal ? "local · free" : "cloud"}
        </span>
      </div>

      {!result && (
        <p style={{ color: "#6b7280", fontSize: 13 }}>Not run yet.</p>
      )}
      {result?.status === "running" && (
        <p style={{ color: "#9aa0a6", fontSize: 13 }}>Synthesizing…</p>
      )}
      {result?.status === "error" && (
        <p style={{ color: "#f87171", fontSize: 13 }}>⚠ {result.message}</p>
      )}
      {result?.status === "done" && (
        <>
          <dl style={{ fontSize: 13, color: "#c5c9cd", margin: "12px 0" }}>
            <Metric label="First audio" value={`${result.metrics.firstAudioMs} ms`} />
            <Metric label="Total" value={`${result.metrics.totalMs} ms`} />
            <Metric label="Audio length" value={`${result.metrics.audioSeconds}s`} />
          </dl>
          <audio controls src={result.wavUrl} style={{ width: "100%" }} />
          <textarea
            value={result.notes}
            onChange={(e) => onNotes(e.target.value)}
            placeholder="Quality notes (naturalness, emotion…)"
            rows={2}
            style={{
              marginTop: 10,
              width: "100%",
              boxSizing: "border-box",
              padding: 8,
              borderRadius: 8,
              background: "#0b0d12",
              color: "#e8eaed",
              border: "1px solid #1c2128",
              fontSize: 13,
              resize: "vertical",
            }}
          />
        </>
      )}
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "2px 0" }}>
      <dt style={{ color: "#9aa0a6" }}>{label}</dt>
      <dd style={{ margin: 0, fontVariantNumeric: "tabular-nums" }}>{value}</dd>
    </div>
  );
}
