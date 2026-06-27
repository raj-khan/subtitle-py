"""X11 caption overlay (PyQt5).

Frameless, translucent, always-on-top window that shows the rolling caption.
Draggable with the mouse (v1 positioning). A semi-transparent box sits behind
the text for readability over bright video. After `linger_sec` of no new text
the caption fades out.

Threading: the capture/transcribe loop runs in a worker thread and pushes text
in via the `committed`/`end_line` Qt signals, so all painting stays on the GUI
thread.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

from .config import Config

# Dragged position lives here, not in config.yaml, so the config stays a clean
# human-owned file. Gitignored (it's per-machine screen state).
POS_FILE = Path(__file__).resolve().parent.parent / ".overlay_pos.json"


class CaptionOverlay(QtWidgets.QWidget):
    # Cross-thread entry points (emit these from the worker thread).
    committed = QtCore.pyqtSignal(str)
    end_line = QtCore.pyqtSignal()

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self._lines: List[str] = [""]      # rolling lines, last is in-progress
        self._opacity = 1.0
        self._drag_offset = None

        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool                # no taskbar entry
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.resize(cfg.width, self._line_height() * cfg.max_lines + 24)
        self._place()

        # Fade timer: after linger_sec idle, fade out the caption.
        self._idle = QtCore.QTimer(self)
        self._idle.setSingleShot(True)
        self._idle.timeout.connect(self._start_fade)

        self._fade = QtCore.QTimer(self)
        self._fade.timeout.connect(self._fade_step)

        self.committed.connect(self._on_committed)
        self.end_line.connect(self._on_end_line)

    # --- geometry ---
    def _line_height(self) -> int:
        return int(self.cfg.font_size * 1.4)

    def _place(self) -> None:
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        saved = self._load_pos()
        if saved is not None:
            self.move(*saved)
        elif self.cfg.pos_x is not None and self.cfg.pos_y is not None:
            self.move(self.cfg.pos_x, self.cfg.pos_y)  # optional manual override
        else:
            x = (screen.width() - self.width()) // 2
            y = int(screen.height() * 0.82) - self.height()
            self.move(x, y)

    @staticmethod
    def _load_pos() -> Optional[Tuple[int, int]]:
        try:
            d = json.loads(POS_FILE.read_text())
            return int(d["x"]), int(d["y"])
        except (OSError, ValueError, KeyError):
            return None

    @staticmethod
    def _save_pos(x: int, y: int) -> None:
        try:
            POS_FILE.write_text(json.dumps({"x": int(x), "y": int(y)}))
        except OSError:
            pass

    # --- caption updates (GUI thread) ---
    def _on_committed(self, text: str) -> None:
        if not text:
            return
        self._lines[-1] += text
        self._reset_idle()
        self.update()

    def _on_end_line(self) -> None:
        if self._lines[-1].strip():
            self._lines.append("")
        while len(self._lines) > self.cfg.max_lines:
            self._lines.pop(0)
        self._reset_idle()
        self.update()

    def _reset_idle(self) -> None:
        self._fade.stop()
        self._opacity = 1.0
        self.setWindowOpacity(1.0)
        self._idle.start(int(self.cfg.linger_sec * 1000))

    def _start_fade(self) -> None:
        self._fade.start(40)  # ~25 fps fade

    def _fade_step(self) -> None:
        self._opacity -= 0.05
        if self._opacity <= 0:
            self._opacity = 0.0
            self._fade.stop()
            self._lines = [""]
            self.update()
        self.setWindowOpacity(max(self._opacity, 0.0))

    # --- painting ---
    def paintEvent(self, _event) -> None:
        visible = [ln for ln in self._lines if ln.strip()]
        if not visible:
            return
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.TextAntialiasing)

        font = QtGui.QFont(self.cfg.font_family, self.cfg.font_size)
        font.setBold(True)
        p.setFont(font)
        metrics = QtGui.QFontMetrics(font)

        line_h = self._line_height()
        pad = 12
        block_h = line_h * len(visible) + pad
        y0 = self.height() - block_h

        # widest line sets the background box width
        text_w = max(metrics.horizontalAdvance(ln) for ln in visible)
        box_w = min(text_w + pad * 2, self.width())
        box_x = (self.width() - box_w) // 2

        if self.cfg.background:
            bg = QtGui.QColor(self.cfg.background_color)
            bg.setAlphaF(self.cfg.background_opacity)
            p.setBrush(bg)
            p.setPen(QtCore.Qt.NoPen)
            p.drawRoundedRect(box_x, y0, box_w, block_h, 10, 10)

        # text with a thin outline for legibility on busy frames
        for i, ln in enumerate(visible):
            ty = y0 + pad // 2 + line_h * (i + 1) - metrics.descent()
            tx = box_x + pad
            path = QtGui.QPainterPath()
            path.addText(float(tx), float(ty), font, ln)
            p.setPen(QtGui.QPen(QtGui.QColor("#000000"), 3))
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawPath(path)
            p.fillPath(path, QtGui.QColor(self.cfg.font_color))

    # --- drag to move ---
    def mousePressEvent(self, e) -> None:
        if e.button() == QtCore.Qt.LeftButton:
            self._drag_offset = e.globalPos() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e) -> None:
        if self._drag_offset is not None:
            self.move(e.globalPos() - self._drag_offset)

    def mouseReleaseEvent(self, _e) -> None:
        if self._drag_offset is not None:
            self._drag_offset = None
            self._save_pos(self.x(), self.y())
