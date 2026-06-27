"""Config loading. Reads config.yaml, falls back to sane defaults."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    model: str = "base.en"
    refresh_sec: float = 1.0

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> "Config":
        if not path.exists():
            return cls()
        data = yaml.safe_load(path.read_text()) or {}
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def with_position(self, x: int, y: int) -> "Config":
        return replace(self, pos_x=x, pos_y=y)

    def save_position(self, x: int, y: int, path: Path = DEFAULT_PATH) -> None:
        """Persist a dragged position back into config.yaml (best effort)."""
        try:
            data = yaml.safe_load(path.read_text()) or {} if path.exists() else {}
            data["pos_x"] = int(x)
            data["pos_y"] = int(y)
            path.write_text(yaml.safe_dump(data, sort_keys=False))
        except OSError:
            pass
