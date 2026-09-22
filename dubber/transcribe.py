"""Speech-to-text with faster-whisper (CTranslate2 Whisper, int8 on CPU).

We use the batched pipeline (≈4x faster than sequential decoding on CPU) with word-level
timestamps, then build our own sentence segments from the words. Whisper's native segments
are either mid-sentence fragments or 30 s chunks; sentences are the right unit for both
translation quality and dubbing timing.
"""

import os
from pathlib import Path

from tqdm import tqdm

from . import log
from .segments import SENTENCE_END, Segment, Word


def transcribe(audio_wav: Path, model_size: str, language: str | None,
               beam_size: int) -> tuple[list[Word], str]:
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    # Using every logical core is slower on hybrid P/E-core laptop CPUs; 8 was the sweet spot.
    threads = min(8, os.cpu_count() or 4)
    log.info(f"loading Whisper '{model_size}' (int8, {threads} threads)")
    model = WhisperModel(model_size, device="auto", compute_type="int8", cpu_threads=threads)

    raw_segments, info = BatchedInferencePipeline(model).transcribe(
        str(audio_wav),
        language=language,
        beam_size=beam_size,
        batch_size=8,
        word_timestamps=True,
    )
    log.info(f"language: {info.language} (p={info.language_probability:.2f}), "
             f"speech to process: {info.duration_after_vad / 60:.1f} min")

    words = []
    with tqdm(total=round(info.duration), unit="s", desc="    transcribing") as bar:
        for s in raw_segments:
            words += [Word(w.start, w.end, w.word) for w in s.words or []]
            bar.update(min(bar.total, round(s.end)) - bar.n)
            log.progress("Transcribe", bar.n, bar.total)
        bar.update(bar.total - bar.n)
        log.progress("Transcribe", bar.total, bar.total)
    return words, info.language


def words_to_sentences(words: list[Word], pause: float = 0.7, max_duration: float = 12.0,
                       soft_max: float = 7.0) -> list[Segment]:
    """Group words into dubbable lines.

    A line ends at sentence punctuation or a pause of `pause` seconds. Lines longer than
    `soft_max` also break at a comma, and nothing exceeds `max_duration`, so the dubbed
    voice never drifts far from the speaker's mouth.
    """
    lines: list[Segment] = []
    current: list[Word] = []

    def flush():
        if current:
            text = "".join(w.text for w in current).strip()
            if text:
                lines.append(Segment(current[0].start, current[-1].end, text))
            current.clear()

    for word in words:
        if current:
            gap = word.start - current[-1].end
            length = word.end - current[0].start
            prev = current[-1].text.rstrip()
            if (prev.endswith(SENTENCE_END) or gap >= pause or length > max_duration
                    or (length > soft_max and prev.endswith((",", ";", ":")))):
                flush()
        current.append(word)
    flush()
    return lines
