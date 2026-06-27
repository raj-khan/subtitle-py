"""Voice activity detection (Silero, ONNX).

Sits in front of Whisper so the engine only ever sees real speech. Without this,
music / silence / sound effects make Whisper hallucinate ("thank you for
watching", "no", looping phrases). This is the gate that makes captions blank
out during action scenes instead of inventing text.

Two interfaces:
  - `speech_segments(audio)`  : batch, for offline checks (step 3 test).
  - `StreamingVAD`            : frame-by-frame utterance detector (step 4 live).

Silero wants 16 kHz audio in 512-sample frames (32 ms).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

FRAME_SAMPLES = 512          # silero's required frame size at 16 kHz
SAMPLE_RATE = 16_000


def load_vad():
    return load_silero_vad(onnx=True)


@dataclass
class Segment:
    start: float
    end: float


def speech_segments(audio: np.ndarray, vad=None, threshold: float = 0.5) -> List[Segment]:
    """Batch: list speech regions (seconds) in a float32 16 kHz buffer."""
    vad = vad or load_vad()
    ts = get_speech_timestamps(
        audio, vad, sampling_rate=SAMPLE_RATE,
        threshold=threshold, return_seconds=True,
    )
    return [Segment(t["start"], t["end"]) for t in ts]


@dataclass
class Utterance:
    """A finished span of speech, ready to hand to Whisper."""
    audio: np.ndarray
    start_time: float
    end_time: float


@dataclass
class StreamingVAD:
    """Stateful gate: feed it audio, it emits Utterances when speech ends.

    Hysteresis avoids flapping: open on a high prob, stay open until prob drops
    below the lower bound for `min_silence_ms`. A little pre/post padding keeps
    sentence starts/ends from being clipped (empirical risk #2).
    """

    vad: object = field(default_factory=load_vad)
    start_threshold: float = 0.5
    stop_threshold: float = 0.35
    min_silence_ms: int = 500     # silence this long ends an utterance
    min_speech_ms: int = 200      # ignore blips shorter than this
    pad_ms: int = 150             # keep this much audio around the speech

    def __post_init__(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)      # leftover < one frame
        self._fed_samples = 0                          # total samples seen
        self._triggered = False
        self._speech: List[np.ndarray] = []            # frames in current utterance
        self._silence_run = 0                          # consecutive silence samples
        self._speech_samples = 0
        self._utt_start = 0.0
        self._pre = np.zeros(0, dtype=np.float32)      # rolling pre-roll pad

    @property
    def in_speech(self) -> bool:
        return self._triggered

    @property
    def active_audio(self) -> np.ndarray:
        """Audio of the utterance currently in progress (empty if none)."""
        if self._triggered and self._speech:
            return np.concatenate(self._speech)
        return np.zeros(0, dtype=np.float32)

    def _prob(self, frame: np.ndarray) -> float:
        t = torch.from_numpy(frame)
        return float(self.vad(t, SAMPLE_RATE))

    def feed(self, audio: np.ndarray) -> List[Utterance]:
        """Push a chunk of float32 16 kHz audio. Returns any utterances that just
        completed (usually empty; one when speech -> silence)."""
        out: List[Utterance] = []
        self._buf = np.concatenate([self._buf, audio])
        pad_samples = int(self.pad_ms / 1000 * SAMPLE_RATE)
        min_sil = int(self.min_silence_ms / 1000 * SAMPLE_RATE)
        min_sp = int(self.min_speech_ms / 1000 * SAMPLE_RATE)

        while len(self._buf) >= FRAME_SAMPLES:
            frame = self._buf[:FRAME_SAMPLES]
            self._buf = self._buf[FRAME_SAMPLES:]
            t_now = self._fed_samples / SAMPLE_RATE
            self._fed_samples += FRAME_SAMPLES
            prob = self._prob(frame)

            if not self._triggered:
                # keep a rolling pre-roll so we don't clip the first word
                self._pre = np.concatenate([self._pre, frame])[-pad_samples:]
                if prob >= self.start_threshold:
                    self._triggered = True
                    self._utt_start = max(0.0, t_now - len(self._pre) / SAMPLE_RATE)
                    self._speech = [self._pre.copy(), frame]
                    self._speech_samples = FRAME_SAMPLES
                    self._silence_run = 0
            else:
                self._speech.append(frame)
                if prob >= self.stop_threshold:
                    self._speech_samples += FRAME_SAMPLES
                    self._silence_run = 0
                else:
                    self._silence_run += FRAME_SAMPLES
                    if self._silence_run >= min_sil:
                        utt = self._close(min_sp, pad_samples, t_now)
                        if utt is not None:
                            out.append(utt)
        return out

    def _close(self, min_sp: int, pad_samples: int, t_now: float) -> Optional[Utterance]:
        self._triggered = False
        audio = np.concatenate(self._speech) if self._speech else np.zeros(0, np.float32)
        self._speech = []
        self._pre = np.zeros(0, dtype=np.float32)
        if self._speech_samples < min_sp:
            return None  # too short, discard blip
        return Utterance(
            audio=audio,
            start_time=self._utt_start,
            end_time=t_now,
        )

    def flush(self) -> Optional[Utterance]:
        """Emit any in-progress utterance (e.g. on shutdown)."""
        if self._triggered and self._speech:
            audio = np.concatenate(self._speech)
            self._triggered = False
            self._speech = []
            return Utterance(audio, self._utt_start, self._fed_samples / SAMPLE_RATE)
        return None
