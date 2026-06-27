#!/usr/bin/env python3
"""Step 3 test: prove the VAD gate blocks non-speech and passes speech.

Checks three things:
  1. Speech (the captured commentary wav) -> VAD finds speech.
  2. Pure silence -> VAD finds nothing.
  3. White noise (stand-in for music/effects) -> VAD finds little/nothing.

    .venv/bin/python scripts/step3_vad.py [path/to/speech.wav]
"""

import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from autotranscript.vad import StreamingVAD, load_vad, speech_segments  # noqa: E402

SPEECH_WAV = sys.argv[1] if len(sys.argv) > 1 else "/tmp/at-test.wav"


def load(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0


def main() -> None:
    vad = load_vad()

    print("1) SPEECH (captured commentary):")
    speech = load(SPEECH_WAV)
    segs = speech_segments(speech, vad)
    print(f"   segments: {[(round(s.start, 2), round(s.end, 2)) for s in segs]}")
    print(f"   verdict: {'PASS (speech detected)' if segs else 'FAIL (missed speech)'}")

    print("\n2) SILENCE (8s of zeros):")
    silence = np.zeros(16000 * 8, np.float32)
    segs = speech_segments(silence, vad)
    print(f"   segments: {[(round(s.start, 2), round(s.end, 2)) for s in segs]}")
    print(f"   verdict: {'PASS (blocked)' if not segs else 'FAIL (hallucinated speech)'}")

    print("\n3) WHITE NOISE (stand-in for music/effects):")
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(16000 * 8) * 0.1).astype(np.float32)
    segs = speech_segments(noise, vad)
    total = sum(s.end - s.start for s in segs)
    print(f"   speech-flagged seconds: {total:.2f} / 8.0")
    print(f"   verdict: {'PASS (mostly blocked)' if total < 1.0 else 'WEAK (leaks noise)'}")

    print("\n4) STREAMING gate on the speech wav (utterance boundaries):")
    sv = StreamingVAD(vad=vad)
    utts = []
    # feed in 100ms chunks to mimic live capture
    step = 1600
    for i in range(0, len(speech), step):
        utts.extend(sv.feed(speech[i:i + step]))
    tail = sv.flush()
    if tail:
        utts.append(tail)
    for u in utts:
        print(f"   utterance {u.start_time:.2f}-{u.end_time:.2f}s "
              f"({len(u.audio) / 16000:.2f}s audio)")
    print(f"   verdict: {'PASS' if utts else 'FAIL (no utterance emitted)'}")


if __name__ == "__main__":
    main()
