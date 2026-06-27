#!/usr/bin/env python3
"""Step 4: live streaming captions to the TERMINAL (no GUI yet).

LIVE (the real thing) -- pick a source, captions stream as audio plays:
    .venv/bin/python scripts/step4_live.py

OFFLINE test (deterministic, feeds a wav through the full chain):
    .venv/bin/python scripts/step4_live.py --wav /tmp/at-test.wav

Committed words print as they finalize; each finished utterance ends the line.
Ctrl-C to quit.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from autotranscript import capture  # noqa: E402
from autotranscript.streaming import StreamingCaptioner, run_live  # noqa: E402


def _print_event(ev) -> None:
    if ev.committed:
        sys.stdout.write(ev.committed)
        sys.stdout.flush()
    if ev.end_of_utterance:
        sys.stdout.write("\n")
        sys.stdout.flush()


def offline(wav_path: str) -> None:
    import wave

    with wave.open(wav_path, "rb") as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
    print(f"Offline run on {wav_path} ({len(audio)/16000:.1f}s)\n")
    cap = StreamingCaptioner()
    # feed in 100 ms chunks, simulating real-time
    step = 1600
    t0 = time.time()
    for i in range(0, len(audio), step):
        for ev in cap.feed(audio[i:i + step]):
            _print_event(ev)
    for ev in cap.flush():
        _print_event(ev)
    print(f"\n\n[done in {time.time()-t0:.1f}s of compute for "
          f"{len(audio)/16000:.1f}s audio]")


def live() -> None:
    source = capture.pick_source_interactive()
    print("Listening. Play your video. Captions appear below. Ctrl-C to quit.\n")
    run_live(source.name, _print_event)
    print("\nStopped.")


def main() -> None:
    if "--wav" in sys.argv:
        offline(sys.argv[sys.argv.index("--wav") + 1])
    else:
        live()


if __name__ == "__main__":
    main()
