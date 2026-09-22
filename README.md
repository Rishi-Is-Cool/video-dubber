# Video Dubber

Turns a YouTube video in any language into an English-dubbed version: same video, English voice,
timed to the original speech. Runs on a laptop CPU with free, open-source tools. No GPU or paid API needed.

```bash
python dub.py "https://www.youtube.com/watch?v=..."
```

Or use the web page: run `python webapp.py` and open <http://127.0.0.1:5000>. Paste a link and
click **Dub video**. The page shows each stage as it runs, then plays the result. You can switch
between the English dub and the original at the same timestamp, and click any transcript line to
jump to it. Earlier videos are listed at the bottom. The page is a thin layer over the same
pipeline, and it processes one video at a time because each stage already uses every CPU core.

Output lands in `output/<video-id>/`:

| File | What it is |
|---|---|
| `source.mp4` | the downloaded original |
| `source_dubbed_en.mp4` | **the dubbed video** |
| `english.srt` | English subtitles (a free by-product) |
| `report.json` | processing time per stage, timing accuracy, settings used |
| `words.json`, `translated.json` | intermediate results, so an interrupted run resumes where it stopped |

## Results

Measured on a laptop: Intel i5-1335U, 16 GB RAM, no GPU, running the default settings.

| Video | Language | Length | Processing time | Lines | On time (±0.25s) | Max drift |
|---|---|---|---|---|---|---|
| [Terra X: Nobelpreis 2025, Quantenphysik](https://www.youtube.com/watch?v=QbyKWvjIhP8) | German | 28m 40s | 1h 03m* | 334 | 99.1% | 0.58s |
| [Cours d'Histoires d'Art : le XIXe siècle](https://www.youtube.com/watch?v=_hKZ7RPK3-Q) | French | 1h 58m 30s | 1h 07m | 1,281 | 99.8% | 0.69s |

\*This run hit edge-tts connections that hung for minutes, and synthesis alone took 47 minutes.
A 20-second request timeout with retry fixed it. On the 2-hour video, synthesis then took 13 minutes.

Stage breakdown for the 2-hour video: download 6m, transcribe 30m, translate 14m,
synthesize 13m, remix 4m.

## Setup

Python 3.10+. ffmpeg is bundled through `imageio-ffmpeg`, so nothing needs a system install.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

The first run downloads the Whisper model (~0.5 GB) and the NLLB model (~1.4 GB).

## Pipeline

```
YouTube URL
   │  yt-dlp                  1. Download   H.264 MP4 + 16 kHz mono WAV for ASR
   ▼
 audio ── faster-whisper ──►  2. Transcribe word-level timestamps → our own sentence lines
   ▼
 lines ── NLLB-200 / Claude ► 3. Translate  meaning-level English per line
   ▼
 English ── edge-tts ───────► 4. Synthesize neural voice matched to the speaker's gender
   ▼
 clips ── timing fitter ────► 5. Align      each line placed at its original start time
   ▼
 dub track ── ffmpeg ───────► 6. Remix      audio replaced, video stream copied (no re-encode)
```

Code layout: `dub.py` (CLI) or `webapp.py` (web page) → `dubber/pipeline.py` (orchestration, caching, report) → one
module per concern: `media.py`, `transcribe.py`, `translate.py`, `synthesize.py`, `segments.py`.

## Design decisions

**Transcription: faster-whisper, batched, with word timestamps.** faster-whisper runs Whisper on
CTranslate2 in int8, roughly 4x faster than the reference implementation on CPU. The batched
pipeline processed audio at about 4-5x realtime on an i5 laptop, compared with about 1.5x for
sequential decoding. It also caught speech that sequential decoding with voice-activity
detection missed. Whisper's own segments are either fragments that end mid-sentence or 30-second
chunks, so I take its **word** timestamps and cut my own lines at sentence punctuation, pauses of
0.7s or more, and at most 12s. Sentences suit both translation (a verb at the end of a German
clause needs the whole clause) and dubbing (a line never drifts far from the speaker's mouth).

**Translation: NLLB-200 1.3B by default, Claude optional.** I first tried Google Translate through
`deep-translator`. Google rate-limited it (HTTP 429) after a few dozen requests, which rules it out
for a 2-hour video. NLLB-200 is Meta's open translation model for 200 languages, including Hindi
and other Indic languages. It runs offline on the same CTranslate2 runtime as Whisper, so it adds
no new heavy dependency. NLLB is trained on single sentences and silently drops text when given
several at once, so each line is split into sentences before translation. With
`ANTHROPIC_API_KEY` set, the pipeline switches to Claude. Claude translates 40 lines at a time
with the previous lines as context, and each line comes with its **time budget and a target word
count**. The English then fits the original timing instead of being sped up afterwards, and Claude
can repair obvious speech-recognition mistakes from context.

**Voice: edge-tts "Multilingual" neural voices.** These are the most natural free voices available.
The speaker's median pitch, estimated by autocorrelation (Men ~85-155 Hz, women ~165-255 Hz),
picks a male or female voice. `--multi-voice` makes that choice per line, for interviews and panels.

**Timing: fit, don't just place.** Each line gets a budget: the time until the next line starts.
If the English runs long:
1. It is re-voiced by the TTS engine at up to +30% rate. This sounds far more natural than
   stretching audio.
2. If it is still long, it is time-stretched with ffmpeg `atempo` (pitch preserved) by up to 1.25x.
3. If it is still long, the next line starts slightly late, and the delay disappears at the next
   pause.

`report.json` records the result: the share of lines that start within 0.25s of the original, and
the maximum drift.

**Remix: the video is never re-encoded.** ffmpeg copies the video stream bit for bit and swaps in
the new AAC track. Muxing takes seconds even for a 2-hour video, with no quality loss.
`--keep-background` separates music and ambience from the original with Demucs and mixes them
under the dub. It is off by default because it is slow on CPU.

**Built for long videos.** Every stage caches its output in the work folder, and TTS clips are
cached by a hash of their text, voice and rate. A crash or network error in minute 90 of a
2-hour run resumes from there, and TTS and network calls retry with backoff.

## Options

```
--whisper-model  small (default) | medium | large-v3 | turbo   bigger = more accurate, slower
--language       de | fr | hi | ...                            skip auto-detection
--translator     auto | nllb | claude                          auto = claude if ANTHROPIC_API_KEY set
--voice          en-US-BrianMultilingualNeural ...             force a specific voice
--multi-voice                                                  per-line male/female voice
--keep-background                                              keep music (pip install demucs)
```

## Limitations and next steps

- **Voice cloning:** edge-tts voices match gender, not the speaker's timbre or energy. Coqui XTTS-v2
  could clone the voice, but on CPU it runs well below realtime, which is impractical for 2 hours.
  With a GPU it would drop into `synthesize.py`.
- **Speaker diarization:** `--multi-voice` separates speakers by pitch, not identity. pyannote
  diarization with a distinct voice per speaker cluster is the proper fix.
- **Lip sync:** timing is aligned per sentence, not per syllable.
