"""Everything that touches ffmpeg or yt-dlp: download, audio extraction, decoding, muxing."""

import shutil
import subprocess
from pathlib import Path

import numpy as np

from . import log

SAMPLE_RATE = 24_000  # edge-tts native rate; the dubbed track is built at this rate


def ffmpeg_exe() -> str:
    """Prefer a system ffmpeg; fall back to the static binary bundled with imageio-ffmpeg."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def run_ffmpeg(args: list[str]) -> bytes:
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{result.stderr.decode(errors='replace')}")
    return result.stdout


def download(source: str, work_dir: Path) -> Path:
    """Download a YouTube URL with yt-dlp (or accept a local file path as-is)."""
    if Path(source).is_file():
        log.info(f"using local file {source}")
        return Path(source)

    existing = list(work_dir.glob("source.*"))
    if existing:
        log.info(f"already downloaded: {existing[0].name}")
        return existing[0]

    import yt_dlp

    last = [""]

    def hook(d):
        if d["status"] == "downloading":
            done = d.get("downloaded_bytes") or 0
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            pct = f"{100 * done / total:3.0f}%" if total else f"{done / 1e6:.0f} MB"
            if pct != last[0]:  # only redraw when the number changes
                last[0] = pct
                print(f"\r    downloading {pct}      ", end="", flush=True)
        elif d["status"] == "finished":
            print(flush=True)

    opts = {
        # Prefer H.264 so the copied video stream plays everywhere (QuickTime, browsers, email).
        "format": ("bv*[vcodec^=avc1][height<=1080]+ba[ext=m4a]/"
                   "bv*[ext=mp4][height<=1080]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b"),
        "merge_output_format": "mp4",
        "outtmpl": str(work_dir / "source.%(ext)s"),
        "ffmpeg_location": ffmpeg_exe(),
        "progress_hooks": [hook],
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(source, download=True)
        log.info(f"title: {info.get('title')}")
    return next(work_dir.glob("source.*"))


def extract_audio(video: Path, out_wav: Path, sample_rate: int = 16_000) -> Path:
    """Mono 16 kHz WAV for Whisper."""
    if not out_wav.exists():
        run_ffmpeg(["-i", str(video), "-vn", "-ac", "1", "-ar", str(sample_rate), str(out_wav)])
    return out_wav


def decode(path: Path, tempo: float = 1.0, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any audio file to mono float32, optionally time-stretched (pitch preserved)."""
    args = ["-i", str(path)]
    if abs(tempo - 1.0) > 1e-3:
        args += ["-af", f"atempo={tempo:.4f}"]  # atempo accepts 0.5–100, we only use ~1.0–1.4
    args += ["-f", "s16le", "-ac", "1", "-ar", str(sample_rate), "-"]
    raw = run_ffmpeg(args)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def separate_background(video: Path, work_dir: Path) -> Path:
    """Strip vocals with Demucs to keep music/ambience under the dub (optional, slow on CPU)."""
    out = work_dir / "htdemucs" / "audio_full" / "no_vocals.wav"
    if out.exists():
        return out
    import sys

    full = work_dir / "audio_full.wav"
    run_ffmpeg(["-i", str(video), "-vn", "-ac", "2", "-ar", "44100", str(full)])
    cmd = [sys.executable, "-m", "demucs", "--two-stems", "vocals", "-n", "htdemucs",
           "-o", str(work_dir), str(full)]
    subprocess.run(cmd, check=True)
    return out


def mux(video: Path, dub_wav: Path, out_path: Path, background: Path | None = None,
        background_gain: float = 1.0) -> None:
    """Replace the audio track. The video stream is copied, never re-encoded."""
    args = ["-i", str(video), "-i", str(dub_wav)]
    if background:
        args += ["-i", str(background), "-filter_complex",
                 f"[2:a]volume={background_gain}[bg];[1:a][bg]amix=inputs=2:duration=first:normalize=0[a]",
                 "-map", "0:v:0", "-map", "[a]"]
    else:
        args += ["-map", "0:v:0", "-map", "1:a:0"]
    args += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
             "-movflags", "+faststart", str(out_path)]
    run_ffmpeg(args)
