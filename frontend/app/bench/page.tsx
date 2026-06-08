import Link from "next/link";

import { TtsBench } from "./tts-bench";

export default function BenchPage() {
  return (
    <main style={{ maxWidth: 860, margin: "0 auto", padding: "48px 24px" }}>
      <h1 style={{ fontSize: 28, marginBottom: 4 }}>TTS eval harness</h1>
      <p style={{ color: "#9aa0a6", marginTop: 0 }}>
        Same script, every provider — compare latency, length, and (by ear)
        quality. Vendor benchmarks are self-serving; measure your own axis.
      </p>
      <TtsBench />
      <p style={{ marginTop: 32 }}>
        <Link href="/" style={{ color: "#7dd3fc" }}>
          ← back to the voice demo
        </Link>
      </p>
    </main>
  );
}
