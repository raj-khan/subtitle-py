"""Audio capture.

v1 backend: Linux + PulseAudio, by shelling out to `parec`. `parec` takes a
PulseAudio source name directly and can downmix/resample on the fly, so we ask
it for exactly what Whisper wants: 16 kHz, mono, signed 16-bit little-endian.

This module is the swappable boundary for cross-platform later: a Windows
(WASAPI loopback) or macOS backend would expose the same three things
(`list_monitor_sources`, `record_to_wav`, `stream_pcm`) and nothing upstream
changes.
"""

from __future__ import annotations

import subprocess
import wave
from dataclasses import dataclass
from typing import Iterator, List, Optional

# What we capture at. 16 kHz mono is Whisper's native input, so capturing here
# means zero resampling later.
SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes, s16le
BYTES_PER_SECOND = SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH


@dataclass
class AudioSource:
    """A PulseAudio capture source."""

    name: str          # e.g. alsa_output....monitor
    description: str    # human label
    is_monitor: bool    # True = system output (what you hear); False = a mic
    state: str          # RUNNING / IDLE / SUSPENDED


def list_sources() -> List[AudioSource]:
    """Every PulseAudio source, monitors first."""
    short = subprocess.run(
        ["pactl", "list", "short", "sources"],
        capture_output=True, text=True, check=True,
    ).stdout

    # Map source name -> human description from the verbose listing.
    descriptions = _source_descriptions()

    sources: List[AudioSource] = []
    for line in short.splitlines():
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        name = parts[1]
        state = parts[4]
        sources.append(
            AudioSource(
                name=name,
                description=descriptions.get(name, name),
                is_monitor=name.endswith(".monitor"),
                state=state,
            )
        )

    # Monitors (system audio) first, then mics.
    sources.sort(key=lambda s: (not s.is_monitor, s.name))
    return sources


def list_monitor_sources() -> List[AudioSource]:
    """Only the output monitors, i.e. system audio you can hear."""
    return [s for s in list_sources() if s.is_monitor]


def default_monitor() -> Optional[AudioSource]:
    """Monitor of the current default sink, if we can resolve it."""
    try:
        default_sink = subprocess.run(
            ["pactl", "get-default-sink"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return None
    target = f"{default_sink}.monitor"
    for s in list_monitor_sources():
        if s.name == target:
            return s
    return None


def _source_descriptions() -> dict:
    """Parse `pactl list sources` to map source name -> Description."""
    out = subprocess.run(
        ["pactl", "list", "sources"],
        capture_output=True, text=True, check=True,
    ).stdout
    descriptions: dict = {}
    current_name = None
    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith("Name:"):
            current_name = stripped.split("Name:", 1)[1].strip()
        elif stripped.startswith("Description:") and current_name:
            descriptions[current_name] = stripped.split("Description:", 1)[1].strip()
    return descriptions


def pick_source_interactive() -> AudioSource:
    """Print a numbered menu and return the chosen source.

    Monitors (system audio) are listed first and the active one is the default.
    """
    sources = list_sources()
    if not sources:
        raise RuntimeError("No PulseAudio sources found. Is PulseAudio running?")

    default = default_monitor()
    default_idx = 0
    if default is not None:
        for i, s in enumerate(sources):
            if s.name == default.name:
                default_idx = i
                break

    print("\nPick an audio source to caption:\n")
    for i, s in enumerate(sources):
        kind = "system audio" if s.is_monitor else "microphone"
        star = "  <- active" if s.state in ("RUNNING", "IDLE") and s.is_monitor else ""
        mark = "*" if i == default_idx else " "
        print(f"  {mark}[{i}] ({kind}) {s.description}{star}")
        print(f"        {s.name}")

    print()
    raw = input(f"Number [default {default_idx}]: ").strip()
    idx = default_idx if raw == "" else int(raw)
    if not (0 <= idx < len(sources)):
        raise ValueError(f"Out of range: {idx}")
    chosen = sources[idx]
    print(f"\nUsing: {chosen.description}\n")
    return chosen


def _parec_cmd(source_name: str) -> List[str]:
    return [
        "parec",
        "--device", source_name,
        "--format", "s16le",
        "--rate", str(SAMPLE_RATE),
        "--channels", str(CHANNELS),
        "--raw",
    ]


def stream_pcm(source_name: str, chunk_seconds: float = 0.1) -> Iterator[bytes]:
    """Yield raw s16le PCM chunks from a source, forever.

    Used by the live pipeline (step 4+). Caller is responsible for stopping;
    closing the generator terminates `parec`.
    """
    chunk_bytes = int(BYTES_PER_SECOND * chunk_seconds)
    proc = subprocess.Popen(_parec_cmd(source_name), stdout=subprocess.PIPE)
    assert proc.stdout is not None
    try:
        while True:
            data = proc.stdout.read(chunk_bytes)
            if not data:
                break
            yield data
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()


def record_to_wav(source_name: str, seconds: float, path: str) -> None:
    """Capture `seconds` of audio from `source_name` into a 16 kHz mono WAV."""
    total_bytes = int(BYTES_PER_SECOND * seconds)
    captured = bytearray()
    for chunk in stream_pcm(source_name, chunk_seconds=0.1):
        captured.extend(chunk)
        if len(captured) >= total_bytes:
            break
    captured = captured[:total_bytes]

    with wave.open(path, "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(bytes(captured))
