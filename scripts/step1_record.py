#!/usr/bin/env python3
"""Step 1 smoke test: pick a source, record 5s, confirm it's the right audio.

Run with audio PLAYING on your laptop (a YouTube video, music, anything):

    .venv/bin/python scripts/step1_record.py

Then play it back to confirm we captured the right thing:

    paplay /tmp/auto-transcript-step1.wav
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autotranscript import capture  # noqa: E402

OUT = "/tmp/auto-transcript-step1.wav"
SECONDS = 5.0


def main() -> None:
    source = capture.pick_source_interactive()
    print(f"Recording {SECONDS:.0f}s ... (make sure audio is playing now)")
    capture.record_to_wav(source.name, SECONDS, OUT)
    print(f"Saved {OUT}")
    print(f"\nPlay it back to verify:\n    paplay {OUT}")


if __name__ == "__main__":
    main()
