"""Terminal progress output and per-stage timing.

Listeners let another front end (the local web demo in webapp.py) mirror the same progress
a terminal user sees, without the pipeline knowing anything about the web layer.
"""

import time
from contextlib import contextmanager
from typing import Callable

Listener = Callable[[str, dict], None]

_timings: dict[str, float] = {}
_listeners: list[Listener] = []
_last_pct: dict[str, int] = {}


def add_listener(fn: Listener) -> None:
    _listeners.append(fn)


def remove_listener(fn: Listener) -> None:
    if fn in _listeners:
        _listeners.remove(fn)


def _emit(kind: str, payload: dict) -> None:
    for fn in list(_listeners):
        try:
            fn(kind, payload)
        except Exception:
            pass  # a broken listener must never break the pipeline


def info(msg: str) -> None:
    print(f"    {msg}", flush=True)
    _emit("info", {"text": msg})


@contextmanager
def stage(number: int, total: int, title: str):
    """Print a stage header, run the block, then print how long it took."""
    print(f"\n[{number}/{total}] {title}", flush=True)
    _last_pct.clear()
    _emit("stage_start", {"number": number, "total": total, "title": title})
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    _timings[title] = elapsed
    print(f"    done in {fmt_duration(elapsed)}", flush=True)
    _emit("stage_done", {"number": number, "title": title, "seconds": round(elapsed, 1)})


def progress(label: str, current: float, total: float) -> None:
    """Throttled progress for long loops (every ~2%). Only goes to listeners: the terminal
    already shows a tqdm bar for the same loop."""
    if total <= 0:
        return
    pct = min(100, int(100 * current / total))
    if pct - _last_pct.get(label, -100) >= 2 or pct == 100:
        _last_pct[label] = pct
        _emit("progress", {"label": label, "pct": pct})


def timings() -> dict[str, float]:
    return dict(_timings)


def fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"
