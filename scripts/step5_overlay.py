#!/usr/bin/env python3
"""Step 5 visual test: drive the overlay with fake captions (no audio needed).

    .venv/bin/python scripts/step5_overlay.py

A window with rolling captions should appear near the bottom-center of the
screen. Drag it around. It fades after `linger_sec`. Close the terminal or
Ctrl-C to quit.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt5 import QtCore, QtWidgets  # noqa: E402

from autotranscript.config import Config  # noqa: E402
from autotranscript.overlay import CaptionOverlay  # noqa: E402

SCRIPT = [
    ("commit", "The quick brown fox "),
    ("commit", "jumps over "),
    ("commit", "the lazy dog."),
    ("end", None),
    ("commit", "Drag me anywhere "),
    ("commit", "you like."),
    ("end", None),
    ("commit", "Edit config.yaml for size and color."),
    ("end", None),
]


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    overlay = CaptionOverlay(Config.load())
    overlay.show()

    step = {"i": 0}

    def tick() -> None:
        if step["i"] >= len(SCRIPT):
            return
        kind, payload = SCRIPT[step["i"]]
        step["i"] += 1
        if kind == "commit":
            overlay.committed.emit(payload)
        else:
            overlay.end_line.emit()

    timer = QtCore.QTimer()
    timer.timeout.connect(tick)
    timer.start(700)

    print("Overlay shown. Drag it. It fades after linger_sec. Ctrl-C to quit.")
    app.exec_()


if __name__ == "__main__":
    main()
