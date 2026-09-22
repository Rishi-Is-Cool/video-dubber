"""Data model for transcribed words and dubbable lines, plus (de)serialization."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

SENTENCE_END = (".", "?", "!", "…", "。", "？", "！", "।", "॥")


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str             # source-language transcript
    english: str = ""     # translation

    @property
    def duration(self) -> float:
        return self.end - self.start


def save(items: list, path: Path) -> None:
    path.write_text(json.dumps([asdict(x) for x in items], ensure_ascii=False, indent=1),
                    encoding="utf-8")


def load(path: Path, cls=Segment) -> list:
    return [cls(**d) for d in json.loads(path.read_text(encoding="utf-8"))]


def write_srt(segments: list[Segment], path: Path) -> None:
    def ts(t: float) -> str:
        ms = int(round(t * 1000))
        h, ms = divmod(ms, 3_600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines = []
    for i, seg in enumerate(segments, 1):
        lines += [str(i), f"{ts(seg.start)} --> {ts(seg.end)}", seg.english, ""]
    path.write_text("\n".join(lines), encoding="utf-8")
