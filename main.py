#!/usr/bin/env python3
"""auto-transcript: live local captions for system audio, shown as an overlay.

    .venv/bin/python main.py

Pick your audio source from the menu, then play your video. Drag the caption to
move it. Edit config.yaml for font size / color. Ctrl-C in the terminal to quit.
"""

import signal
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PyQt5 import QtWidgets  # noqa: E402

from autotranscript import capture  # noqa: E402
from autotranscript.config import Config  # noqa: E402
from autotranscript.overlay import CaptionOverlay  # noqa: E402
from autotranscript.streaming import StreamingCaptioner, run_live  # noqa: E402


def main() -> None:
    cfg = Config.load()

    # Source selection happens in the terminal before the GUI starts.
    source = capture.pick_source_interactive()

    app = QtWidgets.QApplication(sys.argv)
    overlay = CaptionOverlay(cfg)
    overlay.show()

    mode = "translating to English" if cfg.task == "translate" else "captioning"
    print(f"{mode.capitalize()} with {cfg.model} "
          f"(language: {cfg.language}). Drag the caption to move it. "
          f"Ctrl-C here to quit.\n")

    from autotranscript.engine import Transcriber
    transcriber = Transcriber(cfg.model, task=cfg.task, language=cfg.language)
    captioner = StreamingCaptioner(transcriber=transcriber, refresh_sec=cfg.refresh_sec)

    stop = threading.Event()

    def on_event(ev) -> None:
        if ev.committed:
            overlay.committed.emit(ev.committed)
        if ev.end_of_utterance:
            overlay.end_line.emit()

    def worker() -> None:
        run_live(source.name, on_event, captioner=captioner)

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    # Ctrl-C in the terminal should quit the Qt app cleanly.
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    # let the Python signal handler run during the Qt loop
    timer = __import__("PyQt5.QtCore", fromlist=["QTimer"]).QTimer()
    timer.start(200)
    timer.timeout.connect(lambda: None)

    app.exec_()
    stop.set()
    print("\nStopped.")


if __name__ == "__main__":
    main()
