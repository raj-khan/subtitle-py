"""Live streaming captioner: capture -> VAD -> Whisper -> committed words.

The hard part of the whole project. Whisper isn't a streaming model, so to get
near-real-time captions that DON'T flicker we use LocalAgreement-2:

  - While someone is speaking, re-transcribe the growing utterance buffer every
    `refresh_sec`.
  - A word is only "committed" (shown for good) once TWO consecutive
    transcriptions agree on it. Agreement = it's stable, won't be rewritten.
  - When the utterance ends (VAD detects silence), commit whatever's left.

This is the "commit-style, no flicker" decision from the spec: a small extra
delay buys captions that don't twitch while you're trying to read them.

Emits CaptionEvents; the consumer (terminal in step 4, overlay in step 5) just
appends committed text and starts a new line on end_of_utterance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from .engine import Transcriber
from .vad import SAMPLE_RATE, StreamingVAD, load_vad


@dataclass
class CaptionEvent:
    committed: str           # newly finalized words (append these)
    end_of_utterance: bool   # True => start a new caption line after this


def _norm(word: str) -> str:
    """Normalize for agreement comparison (case/punctuation-insensitive)."""
    return re.sub(r"[^\w']", "", word.lower())


def _common_prefix(a: List[str], b: List[str]) -> int:
    """Length of the longest prefix where a and b agree (normalized)."""
    n = 0
    for x, y in zip(a, b):
        if _norm(x) == _norm(y) and _norm(x):
            n += 1
        else:
            break
    return n


class StreamingCaptioner:
    def __init__(
        self,
        transcriber: Optional[Transcriber] = None,
        vad=None,
        refresh_sec: float = 1.0,
    ):
        self.tx = transcriber or Transcriber("base.en")
        self.vad = StreamingVAD(vad=vad or load_vad())
        self.refresh_sec = refresh_sec

        # LocalAgreement state for the current utterance.
        self._prev_words: List[str] = []   # previous hypothesis (full word list)
        self._committed_n = 0              # how many words already emitted
        self._last_refresh_len = 0         # samples at last transcribe

    def feed(self, audio: np.ndarray) -> List[CaptionEvent]:
        """Push float32 16 kHz audio. Return any caption events produced."""
        events: List[CaptionEvent] = []
        completed = self.vad.feed(audio)

        # 1) Mid-utterance refresh: commit stable words for low latency.
        if self.vad.in_speech:
            active = self.vad.active_audio
            if len(active) - self._last_refresh_len >= self.refresh_sec * SAMPLE_RATE:
                self._last_refresh_len = len(active)
                ev = self._refresh(active, end=False)
                if ev:
                    events.append(ev)

        # 2) Finished utterances: commit the remainder, end the line.
        for utt in completed:
            ev = self._refresh(utt.audio, end=True)
            events.append(ev or CaptionEvent("", end_of_utterance=True))
            self._reset_utterance()

        return events

    def _refresh(self, audio: np.ndarray, end: bool) -> Optional[CaptionEvent]:
        words = [w.text for w in self.tx.transcribe(audio)]

        if end:
            # Commit everything past what we've already emitted.
            newly = words[self._committed_n:]
            return CaptionEvent("".join(newly), end_of_utterance=True)

        # LocalAgreement-2: commit the prefix that agrees with last hypothesis.
        agreed = _common_prefix(self._prev_words, words)
        self._prev_words = words
        if agreed > self._committed_n:
            newly = words[self._committed_n:agreed]
            self._committed_n = agreed
            return CaptionEvent("".join(newly), end_of_utterance=False)
        return None

    def _reset_utterance(self) -> None:
        self._prev_words = []
        self._committed_n = 0
        self._last_refresh_len = 0

    def flush(self) -> List[CaptionEvent]:
        """On shutdown, emit any in-progress utterance."""
        tail = self.vad.flush()
        if tail is not None and len(tail.audio):
            ev = self._refresh(tail.audio, end=True)
            self._reset_utterance()
            if ev and ev.committed.strip():
                return [ev]
        return []


def run_live(
    source_name: str,
    on_event: Callable[[CaptionEvent], None],
    captioner: Optional[StreamingCaptioner] = None,
    chunk_sec: float = 0.1,
) -> None:
    """Blocking live loop: stream a PulseAudio source through the captioner.

    Ctrl-C to stop. `on_event` receives each CaptionEvent as it's produced.
    """
    from .capture import stream_pcm
    from .engine import pcm16_to_float32

    cap = captioner or StreamingCaptioner()
    try:
        for pcm in stream_pcm(source_name, chunk_seconds=chunk_sec):
            for ev in cap.feed(pcm16_to_float32(pcm)):
                on_event(ev)
    except KeyboardInterrupt:
        pass
    finally:
        for ev in cap.flush():
            on_event(ev)
