// The browser's echo canceller does the first pass. This gate keeps residual
// speaker audio off the STT input while scheduled audio is actually playing.
const BARGE_IN_RMS = 0.045;
const BARGE_IN_HOLD_RMS = 0.025;
const BARGE_IN_SECONDS = 0.14;

export class PlaybackGate {
  private previous: Int16Array | null = null;
  private candidate: Int16Array[] = [];
  private voicedSeconds = 0;

  capture(pcm: Int16Array, rms: number, playing: boolean) {
    if (!playing) {
      const leading = rms >= BARGE_IN_HOLD_RMS && this.candidate.length
        ? this.takeCandidate()
        : [];
      this.reset();
      return { frames: [...leading, pcm], interrupt: false };
    }

    const threshold = this.candidate.length ? BARGE_IN_HOLD_RMS : BARGE_IN_RMS;
    if (rms < threshold) {
      this.reset();
      this.previous = pcm;
      return { frames: [], interrupt: false };
    }

    this.candidate.push(pcm);
    this.voicedSeconds += pcm.length / 16000;
    if (this.voicedSeconds < BARGE_IN_SECONDS) {
      return { frames: [], interrupt: false };
    }

    const frames = this.takeCandidate();
    this.reset();
    return { frames, interrupt: true };
  }

  private takeCandidate() {
    return this.previous ? [this.previous, ...this.candidate] : this.candidate;
  }

  private reset() {
    this.previous = null;
    this.candidate = [];
    this.voicedSeconds = 0;
  }
}
