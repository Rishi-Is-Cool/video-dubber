"""English speech synthesis (edge-tts) and timing alignment against the original speech."""

import asyncio
import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from . import log, media
from .segments import Segment

if sys.platform == "win32":
    # The default Proactor loop logs spurious ConnectionResetErrors when edge-tts closes sockets.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Microsoft's "Multilingual" neural voices are the most expressive free voices edge-tts offers.
VOICES = {"male": "en-US-AndrewMultilingualNeural", "female": "en-US-AvaMultilingualNeural"}

MAX_RATE_BOOST = 30    # at most +30% via the TTS engine's own rate (sounds natural)
MAX_TEMPO = 1.25       # then at most another 1.25x via time-stretching
TTS_TIMEOUT = 20       # seconds per request before retrying
GAP = 0.08             # seconds of breathing room kept before the next line
SR16 = 16_000


# --- Voice selection -------------------------------------------------------------------------

def _voiced_pitches(chunk: np.ndarray) -> list[float]:
    """F0 of clearly voiced 40 ms frames via autocorrelation."""
    frame = int(0.04 * SR16)
    lo, hi = SR16 // 400, SR16 // 70  # search lags for 70–400 Hz
    pitches = []
    for i in range(0, len(chunk) - frame, frame * 2):
        x = chunk[i:i + frame] - chunk[i:i + frame].mean()
        energy = float(np.dot(x, x))
        if energy < 1e-3:
            continue
        ac = np.correlate(x, x, mode="full")[frame - 1:]
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] / energy > 0.5:
            pitches.append(SR16 / lag)
    return pitches


def _gender(pitch: float) -> str:
    # Typical speaking F0: men ~85–155 Hz, women ~165–255 Hz.
    return "female" if pitch >= 165 else "male"


def choose_voices(audio16k: np.ndarray, segments: list[Segment], per_line: bool) -> list[str]:
    """One voice per line. By default the whole video gets one voice matched to the dominant
    speaker's gender; with `per_line`, each line is matched to its own speaker's pitch."""
    per_seg = [_voiced_pitches(audio16k[int(s.start * SR16):int(s.end * SR16)])
               for s in segments]
    all_pitches = [p for ps in per_seg for p in ps]
    median = float(np.median(all_pitches)) if all_pitches else 120.0
    main = _gender(median)
    log.info(f"median speaker pitch ≈ {median:.0f} Hz → {main} voice ({VOICES[main]})")
    if not per_line:
        return [VOICES[main]] * len(segments)

    genders, previous = [], main
    for ps in per_seg:
        if len(ps) >= 5:  # enough voiced frames to judge; otherwise keep the previous speaker
            p = float(np.median(ps))
            # Only switch on a clear signal, so one speaker's intonation doesn't flip voices.
            if p >= 180:
                previous = "female"
            elif p <= 150:
                previous = "male"
        genders.append(previous)
    log.info(f"per-line voices: {genders.count('male')} male, {genders.count('female')} female")
    return [VOICES[g] for g in genders]


# --- TTS -------------------------------------------------------------------------------------

@dataclass
class Clip:
    segment: Segment
    audio: np.ndarray  # mono float32 at media.SAMPLE_RATE, silence trimmed


@dataclass
class TTSJob:
    text: str
    voice: str
    rate: int  # percent, e.g. +15

    def path(self, cache: Path) -> Path:
        key = hashlib.sha1(f"{self.voice}|{self.rate}|{self.text}".encode()).hexdigest()[:16]
        return cache / f"{key}.mp3"


async def _tts_all(jobs: list[TTSJob], cache: Path, desc: str) -> None:
    import edge_tts

    sem = asyncio.Semaphore(8)
    bar = tqdm(total=len(jobs), desc=desc, unit="clip")

    async def one(job: TTSJob):
        path = job.path(cache)
        async with sem:
            for attempt in range(6):
                if path.exists():
                    break
                try:
                    tmp = path.with_suffix(".part")
                    # edge-tts connections occasionally hang for minutes; retry instead.
                    communicate = edge_tts.Communicate(job.text, job.voice,
                                                       rate=f"{job.rate:+d}%")
                    await asyncio.wait_for(communicate.save(str(tmp)), timeout=TTS_TIMEOUT)
                    tmp.replace(path)
                except Exception:
                    await asyncio.sleep(2 ** attempt)
            bar.update(1)

    await asyncio.gather(*(one(j) for j in jobs))
    bar.close()


def _trim_silence(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    voiced = np.flatnonzero(np.abs(audio) > threshold)
    if voiced.size == 0:
        return audio[:0]
    pad = int(0.03 * media.SAMPLE_RATE)
    return audio[max(0, voiced[0] - pad):voiced[-1] + pad]


def _load(path: Path, tempo: float = 1.0) -> np.ndarray:
    if not path.exists():  # TTS failed after retries: leave the line silent rather than crash
        return np.zeros(0, dtype=np.float32)
    return _trim_silence(media.decode(path, tempo))


def _seconds(audio: np.ndarray) -> float:
    return len(audio) / media.SAMPLE_RATE


def synthesize(segments: list[Segment], voices: list[str], cache: Path,
               total_duration: float) -> list[Clip]:
    """Voice every line at natural speed, then fix lines that overrun their slot:
    first by asking the TTS engine to speak faster (sounds natural), then by time-stretching."""
    cache.mkdir(parents=True, exist_ok=True)
    pairs = [(s, v) for s, v in zip(segments, voices) if s.english]
    segs = [s for s, _ in pairs]

    jobs = [TTSJob(s.english, v, 0) for s, v in pairs]
    asyncio.run(_tts_all(jobs, cache, "    synthesizing"))
    with ThreadPoolExecutor(8) as pool:
        audios = list(pool.map(_load, (j.path(cache) for j in jobs)))

    # A line may use the time until the next line starts (pauses included).
    budgets = []
    for i, seg in enumerate(segs):
        next_start = segs[i + 1].start if i + 1 < len(segs) else total_duration
        budgets.append(max(0.3, next_start - seg.start - GAP))

    overrun = [i for i in range(len(segs)) if _seconds(audios[i]) > budgets[i]]
    log.info(f"{len(overrun)}/{len(segs)} lines overrun their slot; re-voicing them faster")
    for i in overrun:
        ratio = _seconds(audios[i]) / budgets[i]
        rate = min(MAX_RATE_BOOST, int(np.ceil((ratio - 1) * 20)) * 5)  # round up to 5%
        jobs[i] = TTSJob(jobs[i].text, jobs[i].voice, rate)
    asyncio.run(_tts_all([jobs[i] for i in overrun], cache, "    re-voicing"))

    def refit(i: int) -> np.ndarray:
        audio = _load(jobs[i].path(cache))
        ratio = _seconds(audio) / budgets[i]
        return _load(jobs[i].path(cache), min(MAX_TEMPO, ratio)) if ratio > 1.01 else audio

    with ThreadPoolExecutor(8) as pool:
        for i, audio in zip(overrun, pool.map(refit, overrun)):
            audios[i] = audio
    still_long = sum(_seconds(audios[i]) > budgets[i] + 0.05 for i in overrun)
    log.info(f"{still_long} line(s) still too long after max speed-up (the next line shifts)")

    return [Clip(s, a) for s, a in zip(segs, audios)]


# --- Timeline --------------------------------------------------------------------------------

def build_track(clips: list[Clip], total_duration: float, out_wav: Path) -> dict:
    """Place each clip at its original start time. If a clip still overruns, the next one is
    pushed back just enough to avoid overlap; the drift is absorbed by the next natural pause."""
    import soundfile as sf

    sr = media.SAMPLE_RATE
    track = np.zeros(int((total_duration + 5) * sr), dtype=np.float32)
    cursor = 0
    drifts = []
    for clip in clips:
        start = max(int(clip.segment.start * sr), cursor)
        drifts.append((start - clip.segment.start * sr) / sr)
        end = start + len(clip.audio)
        if end > len(track):
            track = np.concatenate([track, np.zeros(end - len(track), dtype=np.float32)])
        track[start:end] = clip.audio
        cursor = end + int(GAP * sr)

    track = track[:max(int(total_duration * sr), cursor)]
    peak = float(np.abs(track).max()) or 1.0
    track *= min(1.0, 0.95 / peak)
    sf.write(out_wav, track, sr, subtype="PCM_16")

    drifts = np.array(drifts) if drifts else np.zeros(1)
    stats = {"mean_drift_s": round(float(drifts.mean()), 3),
             "max_drift_s": round(float(drifts.max()), 3),
             "lines_on_time_pct": round(float((drifts < 0.25).mean() * 100), 1)}
    log.info(f"timing: {stats['lines_on_time_pct']}% of lines start within 0.25s of the "
             f"original, max drift {stats['max_drift_s']}s")
    return stats
