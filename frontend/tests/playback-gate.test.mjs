import assert from "node:assert/strict";
import test from "node:test";

import { PlaybackGate } from "../app/playback-gate.ts";

const frame = () => new Int16Array(640); // 40 ms

test("rejects residual echo and isolated loud sounds during playback", () => {
  const gate = new PlaybackGate();
  for (const rms of [0.01, 0.08, 0.01, 0.01]) {
    assert.deepEqual(gate.capture(frame(), rms, true), { frames: [], interrupt: false });
  }
});

test("barge-in releases the preceding audio and all speech onset frames", () => {
  const gate = new PlaybackGate();
  const samples = Array.from({ length: 5 }, frame);
  let result;
  for (const [index, rms] of [0.01, 0.06, 0.035, 0.06, 0.06].entries()) {
    result = gate.capture(samples[index], rms, true);
  }
  assert.equal(result.interrupt, true);
  assert.equal(result.frames.length, samples.length);
  result.frames.forEach((sample, index) => assert.equal(sample, samples[index]));
});

test("speech that starts during the playback tail keeps its first frame", () => {
  const gate = new PlaybackGate();
  const [start, rest] = [frame(), frame()];
  assert.equal(gate.capture(start, 0.065, true).frames.length, 0);
  assert.deepEqual(gate.capture(rest, 0.045, false), {
    frames: [start, rest],
    interrupt: false,
  });
});
