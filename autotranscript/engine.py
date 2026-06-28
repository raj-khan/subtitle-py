"""Transcription engine: faster-whisper, CPU, int8.

`base.en` is the v1 default (good accuracy / near-real-time on the target i5).
If latency is bad on weaker hardware, drop to `tiny.en`; if accuracy disappoints
and the CPU can take it, try `small.en`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from faster_whisper import WhisperModel

# 16-bit PCM full-scale, for int16 -> float32 [-1, 1] conversion.
INT16_MAX = 32768.0


@dataclass
class Word:
    text: str
    start: float
    end: float


def resolve_model(model_name: str, task: str) -> str:
    """Translation needs a multilingual model; the `.en` models can't do it.

    If someone asks to translate with an English-only model, quietly fall back to
    its multilingual sibling (base.en -> base) and tell them why.
    """
    if task == "translate" and model_name.endswith(".en"):
        fixed = model_name[:-3]
        print(f"note: '{model_name}' is English-only and can't translate; "
              f"using multilingual '{fixed}' instead.")
        return fixed
    return model_name


def pcm16_to_float32(pcm: bytes) -> np.ndarray:
    """Raw s16le mono bytes -> float32 array in [-1, 1], Whisper's input."""
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / INT16_MAX


class Transcriber:
    def __init__(
        self,
        model_name: str = "base.en",
        compute_type: str = "int8",
        task: str = "transcribe",
        language: str = "en",
        cpu_threads: int = 4,
    ):
        # task: "transcribe" (caption as spoken) or "translate" (output English).
        # language: a code like "en"/"ms", or "auto"/None to let Whisper detect.
        # cpu_threads: cap the cores Whisper uses. Left at 0, CTranslate2 grabs ALL
        # cores during each transcription burst and the whole desktop stutters.
        # base.en runs ~0.15x real time, so a few threads keep captions live while
        # leaving the rest of the machine free.
        self.task = task
        self.language = None if language in (None, "auto") else language
        model_name = resolve_model(model_name, task)
        # download_root defaults to ~/.cache/huggingface; fully local after first run.
        self.model = WhisperModel(
            model_name,
            device="cpu",
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )
        self.model_name = model_name

    def transcribe(self, audio: np.ndarray) -> List[Word]:
        """Transcribe (or translate) a float32 16 kHz mono buffer into words.

        `condition_on_previous_text=False` keeps a bad guess from poisoning the
        next chunk (a known cause of runaway repetition loops in streaming use).
        """
        segments, _ = self.model.transcribe(
            audio,
            task=self.task,
            language=self.language,  # None => auto-detect the spoken language
            word_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=False,  # we gate with Silero VAD upstream (step 3)
            beam_size=1,       # greedy: lower latency, fine for streaming
        )
        words: List[Word] = []
        for seg in segments:
            if seg.words:
                for w in seg.words:
                    words.append(Word(text=w.word, start=w.start, end=w.end))
            else:
                words.append(Word(text=seg.text, start=seg.start, end=seg.end))
        return words
