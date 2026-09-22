"""Dub a YouTube video into English.

    python dub.py "https://www.youtube.com/watch?v=..."
"""

import argparse
import sys
from pathlib import Path

from dubber.pipeline import Options, run

# Windows consoles default to a legacy code page that can't print "→" or non-Latin titles.
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    p = argparse.ArgumentParser(description="Dub a YouTube video (any language) into English.")
    p.add_argument("source", nargs="?", help="YouTube URL or path to a local video file")
    p.add_argument("--out", type=Path, default=Path("output"), help="output directory")
    p.add_argument("--name", help="name for the work folder (default: YouTube video id)")
    p.add_argument("--whisper-model", default="small",
                   help="tiny | base | small | medium | large-v3 | turbo (default: small)")
    p.add_argument("--language", help="source language code, e.g. de, fr, hi (default: detect)")
    p.add_argument("--beam-size", type=int, default=1, help="Whisper beam size (higher = slower, slightly more accurate)")
    p.add_argument("--translator", choices=["auto", "claude", "nllb"], default="auto",
                   help="auto = Claude if ANTHROPIC_API_KEY is set, else NLLB (offline)")
    p.add_argument("--voice", help="edge-tts voice, e.g. en-US-BrianMultilingualNeural "
                                   "(default: picked from the speaker's pitch)")
    p.add_argument("--multi-voice", action="store_true",
                   help="pick a male/female voice per line (for interviews, panels)")
    p.add_argument("--keep-background", action="store_true",
                   help="keep music/ambience under the dub (needs `pip install demucs`)")
    args = p.parse_args()

    source = args.source or input("YouTube URL: ").strip()
    run(Options(source=source, out_dir=args.out, name=args.name,
                whisper_model=args.whisper_model, language=args.language,
                beam_size=args.beam_size, translator=args.translator, voice=args.voice,
                multi_voice=args.multi_voice,
                keep_background=args.keep_background))


if __name__ == "__main__":
    main()
