import type { Metadata } from "next";
import type { ReactNode } from "react";

export const metadata: Metadata = {
  title: "realtime-voice-agent-demo",
  description: "Realtime voice AI assistant — local demo",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body
        style={{
          margin: 0,
          fontFamily: "system-ui, sans-serif",
          background: "#0b0d12",
          color: "#e8eaed",
        }}
      >
        {children}
      </body>
    </html>
  );
}
