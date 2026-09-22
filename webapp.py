"""Local web front end for the dubbing pipeline.

    python webapp.py        then open http://127.0.0.1:5000

Paste a YouTube link, watch each stage run, then play the original and the English dub
at the same timestamp. One video is processed at a time, because every stage already
uses all CPU cores.
"""

import json
import re
import sys
import threading
import time
from pathlib import Path

from flask import Flask, Response, abort, jsonify, render_template, request, send_file

from dubber import log, media
from dubber.pipeline import Options, run, work_dir_for

# Windows consoles default to a legacy code page that can't print "→" or non-Latin titles.
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
OUTPUT = ROOT / "output"
ANSI = re.compile(r"\x1b\[[0-9;]*m")

app = Flask(__name__)


class Job:
    """A run in progress. Keeps every event so a page reload can replay the whole run."""

    def __init__(self, run_id: str):
        self.id = run_id
        self.status = "running"
        self.error: str | None = None
        self.events: list[dict] = []
        self.cond = threading.Condition()

    def push(self, kind: str, payload: dict) -> None:
        with self.cond:
            self.events.append({"kind": kind, **payload})
            self.cond.notify_all()


jobs: dict[str, Job] = {}
busy = threading.Lock()


def _work(job: Job, opts: Options) -> None:
    log.add_listener(job.push)
    try:
        run(opts)
        job.status = "done"
    except (Exception, SystemExit) as exc:
        job.status = "error"
        message = ANSI.sub("", str(exc)).strip()
        job.error = re.sub(r"^ERROR:\s*(\[\w+\]\s*[\w-]+:\s*)?", "", message) or type(exc).__name__
        work = OUTPUT / job.id
        if work.is_dir() and not any(work.iterdir()):
            work.rmdir()  # nothing was downloaded; don't leave an empty folder behind
    finally:
        log.remove_listener(job.push)
        job.push("end", {"status": job.status, "error": job.error})
        busy.release()


# --- Finished runs on disk -------------------------------------------------------------------

def _run_dir(run_id: str) -> Path:
    if not re.fullmatch(r"[\w-]+", run_id) or not (OUTPUT / run_id).is_dir():
        abort(404)
    return OUTPUT / run_id


def _report(d: Path) -> dict:
    path = d / "report.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _videos(d: Path) -> tuple[Path | None, Path | None]:
    """(original, dubbed). Also recognises files renamed for submission (*SOURCE*, *DUBBED*)."""
    dubbed = [*d.glob("*_dubbed_en.mp4"), *d.glob("*DUBBED*.mp4")]
    original = [*media._finished_download(d), *d.glob("*SOURCE*.mp4")]
    source = _report(d).get("source", "")
    if not original and source and (ROOT / source).is_file():  # run made from a local file
        original = [ROOT / source]
    return (original[0] if original else None), (dubbed[0] if dubbed else None)


def _summary(d: Path) -> dict:
    report = _report(d)
    title_file = d / "title.txt"
    title = report.get("title") or (
        title_file.read_text(encoding="utf-8").strip() if title_file.exists() else d.name)
    original, dubbed = _videos(d)
    job = jobs.get(d.name)
    return {
        "id": d.name,
        "title": title,
        "status": job.status if job else ("done" if dubbed else "incomplete"),
        "error": job.error if job else None,
        "has_original": original is not None,
        "has_dubbed": dubbed is not None,
        "finished_at": (d / "report.json").stat().st_mtime if report else None,
        "report": report,
    }


# --- Routes ----------------------------------------------------------------------------------

@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/runs")
def list_runs():
    runs = [_summary(d) for d in OUTPUT.iterdir()
            if d.is_dir() and (d / "report.json").exists() and _videos(d)[1]] if OUTPUT.exists() else []
    runs.sort(key=lambda r: r["finished_at"] or 0, reverse=True)
    active = next((j.id for j in jobs.values() if j.status == "running"), None)
    return jsonify(runs=runs, active=active)


@app.post("/api/runs")
def start_run():
    data = request.get_json(silent=True) or {}
    source = (data.get("url") or "").strip()
    if not (re.match(r"https?://", source) or Path(source).is_file()):
        return jsonify(error="That doesn't look like a link. Paste a full YouTube URL."), 400
    if not busy.acquire(blocking=False):
        return jsonify(error="Another video is still being processed. Wait for it to finish."), 409
    try:
        opts = Options(
            source=source,
            out_dir=OUTPUT,
            whisper_model=data.get("whisper_model") or "small",
            language=data.get("language") or None,
            translator=data.get("translator") or "auto",
            voice=data.get("voice") or None,
            multi_voice=bool(data.get("multi_voice")),
            keep_background=bool(data.get("keep_background")),
        )
        job = Job(work_dir_for(opts).name)
        jobs[job.id] = job
        threading.Thread(target=_work, args=(job, opts), daemon=True).start()
    except Exception:
        busy.release()
        raise
    return jsonify(id=job.id)


@app.get("/api/runs/<run_id>")
def get_run(run_id: str):
    return jsonify(_summary(_run_dir(run_id)))


@app.get("/api/runs/<run_id>/events")
def run_events(run_id: str):
    """Server-sent events: replays everything so far, then streams new events live."""
    job = jobs.get(run_id)
    if not job:
        abort(404)

    def stream():
        sent = 0
        while True:
            with job.cond:
                if sent >= len(job.events):
                    job.cond.wait(timeout=15)
                new = job.events[sent:]
            sent += len(new)
            if not new:
                yield ": keep-alive\n\n"
                continue
            for event in new:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event["kind"] == "end":
                    return

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/runs/<run_id>/video/<kind>")
def run_video(run_id: str, kind: str):
    d = _run_dir(run_id)
    original, dubbed = _videos(d)
    path = {"original": original, "dubbed": dubbed}.get(kind)
    if path is None:
        abort(404)
    download = request.args.get("download") == "1"
    name = f"{d.name}_{'english_dub' if kind == 'dubbed' else 'original'}.mp4"
    return send_file(path, mimetype="video/mp4", conditional=True,
                     as_attachment=download, download_name=name)


@app.get("/api/runs/<run_id>/transcript")
def run_transcript(run_id: str):
    path = _run_dir(run_id) / "translated.json"
    if not path.exists():
        return jsonify(lines=[])
    return jsonify(lines=json.loads(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    OUTPUT.mkdir(exist_ok=True)
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)
