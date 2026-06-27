#!/usr/bin/env python3
"""Step 2 smoke test: transcribe a WAV file and print the text + timing.

    .venv/bin/python scripts/step2_transcribe.py [path/to.wav]

Defaults to the file step 1 wrote. Run step 1 first (with speech playing).
"""

import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from autotranscript.engine import Transcriber, pcm16_to_float32  # noqa: E402

DEFAULT_WAV = "/tmp/auto-transcript-step1.wav"


def load_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000, "expected 16 kHz"
        assert w.getnchannels() == 1, "expected mono"
        return pcm16_to_float32(w.readframes(w.getnframes()))


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_WAV
    print(f"Loading {path}")
    audio = load_wav(path)
    secs = len(audio) / 16000
    print(f"Audio: {secs:.1f}s")

    print("Loading base.en (first run downloads ~75 MB) ...")
    t0 = time.time()
    tx = Transcriber("base.en")
    print(f"Model loaded in {time.time() - t0:.1f}s")

    t0 = time.time()
    words = tx.transcribe(audio)
    elapsed = time.time() - t0
    text = "".join(w.text for w in words).strip()

    print(f"\nTranscribed {secs:.1f}s of audio in {elapsed:.2f}s "
          f"(RTF {elapsed / secs:.2f}x; <1.0 = faster than real-time)\n")
    print("TEXT:")
    print(f"  {text or '(nothing)'}")


if __name__ == "__main__":
    main()
