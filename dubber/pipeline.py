"""Orchestrates the stages. Each stage writes its result to the work directory, so an
interrupted multi-hour run resumes from the last finished stage instead of starting over."""

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from . import log, media, segments as seglib, synthesize, transcribe, translate

TOTAL_STAGES = 6


@dataclass
class Options:
    source: str
    out_dir: Path
    name: str | None = None
    whisper_model: str = "small"
    language: str | None = None
    beam_size: int = 1
    translator: str = "auto"
    voice: str | None = None
    multi_voice: bool = False
    keep_background: bool = False
    background_gain: float = 0.8


def work_dir_for(opts: Options) -> Path:
    name = opts.name
    if not name:
        match = re.search(r"(?:v=|youtu\.be/|shorts/)([\w-]{11})", opts.source)
        name = match.group(1) if match else hashlib.sha1(opts.source.encode()).hexdigest()[:10]
    path = opts.out_dir / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def run(opts: Options) -> Path:
    t_start = time.perf_counter()
    work = work_dir_for(opts)
    log.info(f"work directory: {work}")

    with log.stage(1, TOTAL_STAGES, "Download"):
        video = media.download(opts.source, work)
        audio_wav = media.extract_audio(video, work / "audio_16k.wav")
        audio16k, sr = sf.read(audio_wav, dtype="float32")
        duration = len(audio16k) / sr
        log.info(f"video: {video.name}, {log.fmt_duration(duration)} long")

    with log.stage(2, TOTAL_STAGES, "Transcribe"):
        words_path = work / "words.json"
        lang_path = work / "language.txt"
        if words_path.exists():
            words = seglib.load(words_path, seglib.Word)
            language = lang_path.read_text().strip()
            log.info(f"cached transcript: {len(words)} words ({language})")
        else:
            words, language = transcribe.transcribe(audio_wav, opts.whisper_model,
                                                    opts.language, opts.beam_size)
            seglib.save(words, words_path)
            lang_path.write_text(language)
        sentences = transcribe.words_to_sentences(words)
        log.info(f"{len(words)} words grouped into {len(sentences)} lines")

    with log.stage(3, TOTAL_STAGES, "Translate"):
        translated_path = work / "translated.json"
        if translated_path.exists():
            sentences = seglib.load(translated_path)
            log.info("cached translation")
        else:
            translate.translate(sentences, language, opts.translator)
            seglib.save(sentences, translated_path)
        seglib.write_srt(sentences, work / "english.srt")
        for s in sentences[:3]:
            log.info(f"  {s.text[:60]!r} → {s.english[:60]!r}")

    with log.stage(4, TOTAL_STAGES, "Synthesize"):
        if opts.voice:
            voices = [opts.voice] * len(sentences)
        else:
            voices = synthesize.choose_voices(audio16k, sentences, opts.multi_voice)
        clips = synthesize.synthesize(sentences, voices, work / "tts_cache", duration)

    with log.stage(5, TOTAL_STAGES, "Align"):
        dub_wav = work / "dub_en.wav"
        timing = synthesize.build_track(clips, duration, dub_wav)

    with log.stage(6, TOTAL_STAGES, "Remix"):
        background = None
        if opts.keep_background:
            log.info("separating music/ambience from vocals with Demucs")
            background = media.separate_background(video, work)
        output = work / f"{video.stem}_dubbed_en.mp4"
        media.mux(video, dub_wav, output, background, opts.background_gain)

    total = time.perf_counter() - t_start
    title_file = work / "title.txt"
    title = title_file.read_text(encoding="utf-8").strip() if title_file.exists() else ""
    report = {
        "source": opts.source,
        "title": title or Path(opts.source).stem,
        "video_duration_s": round(duration, 1),
        "processing_time_s": round(total, 1),
        "processing_time": log.fmt_duration(total),
        "realtime_factor": round(total / duration, 2),
        "source_language": language,
        "whisper_model": opts.whisper_model,
        "translator": opts.translator,
        "voices": sorted(set(voices)),
        "sentences": len(sentences),
        "timing": timing,
        "stage_seconds": {k: round(v, 1) for k, v in log.timings().items()},
    }
    (work / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nDubbed video: {output}")
    print(f"Processed {log.fmt_duration(duration)} of video in {log.fmt_duration(total)} "
          f"({report['realtime_factor']}x realtime)")
    return output
