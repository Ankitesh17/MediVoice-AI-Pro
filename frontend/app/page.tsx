import Link from "next/link";

import { TranscribeDemo } from "./transcribe-demo";

export default function Home() {
  return (
    <main
      style={{
        maxWidth: 720,
        margin: "0 auto",
        padding: "48px 24px",
      }}
    >
      <h1 style={{ fontSize: 28, marginBottom: 4 }}>realtime-voice-agent-demo</h1>
      <p style={{ color: "#9aa0a6", marginTop: 0 }}>
        mic &rarr; STT &rarr; LLM &rarr; TTS &rarr; spoken reply &nbsp;·&nbsp;
        talk or live translation &nbsp;·&nbsp;
        <Link href="/bench" style={{ color: "#7dd3fc" }}>
          TTS eval harness →
        </Link>
      </p>
      <TranscribeDemo />
    </main>
  );
}
