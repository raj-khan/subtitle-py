"""Config loading. Reads config.yaml, falls back to sane defaults."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass
class Config:
    font_size: int = 32
    font_color: str = "#FFFF00"
    font_family: str = "Sans"
    background: bool = True
    background_color: str = "#000000"
    background_opacity: float = 0.55
    max_lines: int = 2
    linger_sec: float = 3.0
    pos_x: Optional[int] = None
    pos_y: Optional[int] = None
    width: int = 900
    model: str = "small.en"
    refresh_sec: float = 1.0
    cpu_threads: int = 4       # cores Whisper may use; lower = gentler on the PC
    max_window_sec: float = 5.0  # cap re-transcribed audio so captions keep up
    task: str = "transcribe"   # "transcribe" | "translate" (any language -> English)
    language: str = "en"       # source hint: "en", "ms", ... or "auto" to detect

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> "Config":
        if not path.exists():
            return cls()
        data = yaml.safe_load(path.read_text()) or {}
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})
