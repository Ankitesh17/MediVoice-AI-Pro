import type { NextConfig } from "next";
import { PHASE_PRODUCTION_BUILD } from "next/constants";

function nextConfig(phase: string): NextConfig {
  if (phase === PHASE_PRODUCTION_BUILD && process.env.VERCEL === "1") {
    const value = process.env.NEXT_PUBLIC_WS_URL;
    const url = value ? URL.parse(value) : null;

    if (!url || url.protocol !== "wss:" || url.username || url.password || url.hash) {
      throw new Error("Set NEXT_PUBLIC_WS_URL to a valid wss:// backend URL for Vercel builds.");
    }
  }

  return {
    poweredByHeader: false,
    async headers() {
      return [
        {
          source: "/:path*",
          headers: [
            { key: "X-Content-Type-Options", value: "nosniff" },
            { key: "X-Frame-Options", value: "DENY" },
            { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
            { key: "Permissions-Policy", value: "microphone=(self), camera=(), geolocation=()" },
          ],
        },
      ];
    },
  };
}

export default nextConfig;
