"use client";

type AuraOrbProps = {
  active: boolean;
  speaking: boolean;
  level: number;
};

export function AuraOrb({ active, speaking, level }: AuraOrbProps) {
  const state = speaking ? "speaking" : active ? "listening" : "standby";
  const intensity = active ? Math.min(1, Math.max(0, level)) : 0;

  return (
    <div
      className={`aura-orb-container aura-orb-${state}`}
      role="img"
      aria-label={
        speaking
          ? "AI assistant speaking"
          : active
            ? "Microphone listening"
            : "Voice assistant on standby"
      }
    >
      <div className="aura-orb-outer-ring" />
      <div className="aura-orb-middle-ring" />

      <div
        className="aura-orb-core"
        style={{
          transform: `scale(${1 + intensity * 0.13})`,
        }}
      >
        <div className="aura-orb-shine" />
      </div>

      <div className="aura-orb-shadow" />
    </div>
  );
}
