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

import collections
import re
import threading
import time
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
        max_window_sec: float = 8.0,
        context_sec: float = 1.0,
    ):
        self.tx = transcriber or Transcriber("base.en")
        self.vad = StreamingVAD(vad=vad or load_vad())
        self.refresh_sec = refresh_sec
        # Cap how much audio we re-transcribe each refresh. Speech with no pauses
        # (live commentary) would otherwise grow the buffer forever and lag harder
        # the longer it runs. When the window passes max_window_sec we keep only a
        # short context tail of already-committed audio and roll on.
        self.max_window_sec = max_window_sec
        self.context_sec = context_sec

        # LocalAgreement state for the current utterance.
        self._prev_words: List[str] = []   # previous hypothesis (full word list)
        self._committed_n = 0              # how many words already emitted
        self._last_refresh_len = 0         # samples at last transcribe
        self._skip_until = 0.0             # ignore words ending before this (s)
        self._commit_time = 0.0            # end time of last committed word (s)

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
            # Safety valve for pauseless speech: keep the window bounded.
            if len(self.vad.active_audio) >= self.max_window_sec * SAMPLE_RATE:
                self._rollover()

        # 2) Finished utterances: commit the remainder, end the line.
        for utt in completed:
            ev = self._refresh(utt.audio, end=True)
            events.append(ev or CaptionEvent("", end_of_utterance=True))
            self._reset_utterance()

        return events

    def _refresh(self, audio: np.ndarray, end: bool) -> Optional[CaptionEvent]:
        hyp = self.tx.transcribe(audio)
        # After a rollover the window starts with already-committed audio; skip the
        # words that fall inside it so we never emit them twice.
        if self._skip_until > 0.0:
            hyp = [w for w in hyp if w.end > self._skip_until]
        words = [w.text for w in hyp]

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
            self._commit_time = hyp[agreed - 1].end  # window-relative seconds
            return CaptionEvent("".join(newly), end_of_utterance=False)
        return None

    def _rollover(self) -> None:
        """Bound the re-transcribed window during long pauseless speech.

        Keep the uncommitted tail plus `context_sec` of already-committed audio for
        acoustic continuity, drop the rest, and mark the kept committed part so its
        words aren't emitted again. If nothing has committed yet there's no safe
        anchor, so let the window grow a little longer (a hard cap still applies).
        """
        active_len = len(self.vad.active_audio)
        trim_at = max(0.0, self._commit_time - self.context_sec)
        keep = active_len - int(trim_at * SAMPLE_RATE)

        if keep <= 0 or keep >= active_len:
            if active_len < int(1.5 * self.max_window_sec * SAMPLE_RATE):
                return  # nothing committed yet; give it more time to settle
            # Pathological: way over budget with no commit. Hard reset the tail.
            self.vad.trim_active(int(self.context_sec * SAMPLE_RATE))
            self._skip_until = 0.0
            self._commit_time = 0.0
        else:
            self.vad.trim_active(keep)
            self._skip_until = self._commit_time - trim_at
            self._commit_time = self._skip_until

        self._prev_words = []
        self._committed_n = 0
        self._last_refresh_len = len(self.vad.active_audio)

    def _reset_utterance(self) -> None:
        self._prev_words = []
        self._committed_n = 0
        self._last_refresh_len = 0
        self._skip_until = 0.0
        self._commit_time = 0.0

    def reset_live(self) -> None:
        """Drop all in-progress state after stale audio was skipped.

        When the live loop falls behind and jumps to the present, the audio it
        feeds next is not continuous with what came before, so the VAD and the
        current utterance must start fresh (we keep the loaded Silero model).
        """
        self.vad = StreamingVAD(vad=self.vad.vad)
        self._reset_utterance()

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
    max_lag_sec: float = 2.5,
) -> None:
    """Blocking live loop: stream a PulseAudio source through the captioner.

    Capture happens in its own thread so it always runs in real time. The main
    loop drains whatever audio has arrived and transcribes it. If transcription
    can't keep up and the backlog grows past `max_lag_sec`, we DROP the stale
    audio and skip to the present: a live caption that's current and missing a
    few words beats one that drifts minutes behind. `on_event` gets each event.
    """
    from .capture import BYTES_PER_SECOND, stream_pcm
    from .engine import pcm16_to_float32

    cap = captioner or StreamingCaptioner()
    max_lag_bytes = int(BYTES_PER_SECOND * max_lag_sec)

    buf: "collections.deque[bytes]" = collections.deque()
    lock = threading.Lock()
    pending = {"bytes": 0}
    running = threading.Event()
    running.set()

    def reader() -> None:
        try:
            for pcm in stream_pcm(source_name, chunk_seconds=chunk_sec):
                if not running.is_set():
                    break
                with lock:
                    buf.append(pcm)
                    pending["bytes"] += len(pcm)
        finally:
            running.clear()

    th = threading.Thread(target=reader, daemon=True)
    th.start()

    try:
        while running.is_set() or pending["bytes"] > 0:
            with lock:
                # Skip to live: if we're too far behind, drop the oldest audio
                # and keep only the most recent max_lag_sec.
                skipped = False
                while pending["bytes"] > max_lag_bytes and len(buf) > 1:
                    old = buf.popleft()
                    pending["bytes"] -= len(old)
                    skipped = True
                chunks = list(buf)
                buf.clear()
                pending["bytes"] = 0

            if skipped:
                cap.reset_live()
            if not chunks:
                time.sleep(chunk_sec)
                continue

            audio = pcm16_to_float32(b"".join(chunks))
            for ev in cap.feed(audio):
                on_event(ev)
    except KeyboardInterrupt:
        pass
    finally:
        running.clear()
        for ev in cap.flush():
            on_event(ev)
