"""Terminal progress output and per-stage timing."""

import time
from contextlib import contextmanager

_timings: dict[str, float] = {}


def info(msg: str) -> None:
    print(f"    {msg}", flush=True)


@contextmanager
def stage(number: int, total: int, title: str):
    """Print a stage header, run the block, then print how long it took."""
    print(f"\n[{number}/{total}] {title}", flush=True)
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    _timings[title] = elapsed
    print(f"    done in {fmt_duration(elapsed)}", flush=True)


def timings() -> dict[str, float]:
    return dict(_timings)


def fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"
