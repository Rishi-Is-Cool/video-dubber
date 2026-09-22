const STAGES = ["Download", "Transcribe", "Translate", "Synthesize", "Align", "Remix"];
const STAGE_HINTS = {
  Download: "Fetching the video from YouTube",
  Transcribe: "Speech to text (Whisper)",
  Translate: "Translating lines to English",
  Synthesize: "Generating the English voice",
  Align: "Fitting each line to the original timing",
  Remix: "Replacing the audio track",
};
const LANGUAGES = {
  de: "German", fr: "French", es: "Spanish", it: "Italian", pt: "Portuguese", hi: "Hindi",
  ja: "Japanese", ru: "Russian", zh: "Chinese", ko: "Korean", nl: "Dutch", tr: "Turkish",
  ar: "Arabic", pl: "Polish", en: "English",
};

const $ = (sel) => document.querySelector(sel);
const form = $("#start-form");
const player = $("#player");

let currentRun = null;
let currentTrack = "dubbed";
let transcript = [];
let timer = null;
let source = null;

// ---------- helpers ----------

function fmtDuration(seconds) {
  seconds = Math.round(seconds || 0);
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  return h ? `${h}h ${String(m).padStart(2, "0")}m` : `${m}m ${String(s).padStart(2, "0")}s`;
}

function fmtClock(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function el(tag, attrs = {}, text) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}

async function api(path, options) {
  const res = await fetch(path, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `Request failed (${res.status})`);
  return body;
}

// ---------- starting a run ----------

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const data = Object.fromEntries(new FormData(form));
  data.multi_voice = form.multi_voice.checked;
  data.keep_background = form.keep_background.checked;
  $("#form-error").hidden = true;
  setBusy(true);
  try {
    const { id } = await api("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    watch(id, data.url);
  } catch (err) {
    $("#form-error").textContent = err.message;
    $("#form-error").hidden = false;
    setBusy(false);
  }
});

function setBusy(busy) {
  $("#start-btn").disabled = busy;
  $("#start-btn").textContent = busy ? "Processing…" : "Dub video";
  $("#url").disabled = busy;
}

// ---------- live progress ----------

function renderSteps() {
  const list = $("#steps");
  list.replaceChildren();
  STAGES.forEach((name, i) => {
    const li = el("li", { class: "step", "data-stage": name });
    li.append(
      el("span", { class: "num" }, String(i + 1)),
      el("span", { class: "name" }, name),
      el("span", { class: "detail" }, STAGE_HINTS[name]),
      el("span", { class: "time" }, ""),
    );
    const bar = el("div", { class: "bar" });
    bar.append(el("i"));
    li.append(bar);
    list.append(li);
  });
}

function step(name) {
  return document.querySelector(`.step[data-stage="${name}"]`);
}

function watch(id, label) {
  if (source) source.close();
  renderSteps();
  $("#result").hidden = true;
  player.pause();
  $("#job").hidden = false;
  $("#job-error").hidden = true;
  $("#log").textContent = "";
  $("#job-title").textContent = label || id;
  $("#job").scrollIntoView({ behavior: "smooth", block: "start" });
  setBusy(true);

  const started = Date.now();
  clearInterval(timer);
  timer = setInterval(() => {
    $("#elapsed").textContent = `Elapsed ${fmtDuration((Date.now() - started) / 1000)}`;
  }, 1000);

  let active = null;
  source = new EventSource(`/api/runs/${id}/events`);
  source.onmessage = (msg) => {
    const ev = JSON.parse(msg.data);
    const log = $("#log");

    if (ev.kind === "stage_start") {
      active = ev.title;
      const s = step(active);
      s.classList.add("active");
      s.querySelector(".time").textContent = "running";
      log.textContent += `\n[${ev.number}/${ev.total}] ${ev.title}\n`;
    } else if (ev.kind === "stage_done") {
      const s = step(ev.title);
      s.classList.remove("active");
      s.classList.add("done");
      s.querySelector(".time").textContent = fmtDuration(ev.seconds);
      s.querySelector(".bar > i").style.width = "100%";
    } else if (ev.kind === "progress") {
      const s = step(ev.label);
      if (s) {
        s.querySelector(".bar > i").style.width = `${ev.pct}%`;
        s.querySelector(".time").textContent = `${ev.pct}%`;
      }
    } else if (ev.kind === "info") {
      log.textContent += `    ${ev.text}\n`;
      if (active && !ev.text.startsWith("  ")) step(active).querySelector(".detail").textContent = ev.text;
      if (ev.text.startsWith("title: ")) $("#job-title").textContent = ev.text.slice(7);
    } else if (ev.kind === "end") {
      source.close();
      clearInterval(timer);
      setBusy(false);
      if (ev.status === "done") {
        $("#url").value = "";
        loadRuns().then(() => openRun(id));
      } else {
        if (active) {
          step(active).classList.replace("active", "failed");
          step(active).querySelector(".time").textContent = "failed";
        }
        $("#job-error").textContent = `Failed: ${ev.error}`;
        $("#job-error").hidden = false;
      }
    }
    log.scrollTop = log.scrollHeight;
  };
}

// ---------- results ----------

async function openRun(id) {
  const run = await api(`/api/runs/${id}`);
  currentRun = run;
  const r = run.report || {};
  $("#result").hidden = false;
  $("#result-title").textContent = run.title;

  const lang = LANGUAGES[r.source_language] || r.source_language || "?";
  $("#result-meta").textContent = r.source && r.source.startsWith("http") ? r.source : "";

  const stats = [
    ["Language", `${lang} → English`],
    ["Video length", fmtDuration(r.video_duration_s)],
    ["Processing time", fmtDuration(r.processing_time_s)],
    ["Lines dubbed", (r.sentences || 0).toLocaleString()],
    ["Lines on time (±0.25s)", r.timing ? `${r.timing.lines_on_time_pct}%` : "–"],
  ];
  $("#stats").replaceChildren(...stats.map(([k, v]) => {
    const d = el("div");
    d.append(el("dt", {}, k), el("dd", {}, v));
    return d;
  }));

  $("#dl-dubbed").href = `/api/runs/${id}/video/dubbed?download=1`;
  $("#dl-original").href = `/api/runs/${id}/video/original?download=1`;
  $("#dl-original").hidden = !run.has_original;
  document.querySelector('.seg [data-track="original"]').disabled = !run.has_original;

  setTrack("dubbed", 0);
  loadTranscript(id);
  document.querySelectorAll(".runs tr").forEach((tr) => tr.classList.toggle("selected", tr.dataset.id === id));
  $("#result").scrollIntoView({ behavior: "smooth", block: "start" });
}

function setTrack(track, at) {
  if (!currentRun) return;
  const wasPlaying = !player.paused;
  const time = at ?? player.currentTime;
  currentTrack = track;
  document.querySelectorAll(".seg button").forEach((b) => b.classList.toggle("active", b.dataset.track === track));
  player.src = `/api/runs/${currentRun.id}/video/${track}`;
  player.addEventListener("loadedmetadata", () => {
    player.currentTime = time;
    if (wasPlaying) player.play();
  }, { once: true });
}

document.querySelectorAll(".seg button").forEach((b) => {
  b.addEventListener("click", () => {
    if (b.dataset.track !== currentTrack) setTrack(b.dataset.track);
  });
});

async function loadTranscript(id) {
  const { lines } = await api(`/api/runs/${id}/transcript`);
  transcript = lines;
  const body = $("#transcript");
  body.replaceChildren(...lines.map((line, i) => {
    const tr = el("tr", { "data-i": i });
    tr.append(
      el("td", { class: "t" }, fmtClock(line.start)),
      el("td", { class: "src" }, line.text),
      el("td", {}, line.english),
    );
    tr.addEventListener("click", () => {
      player.currentTime = line.start;
      player.play();
    });
    return tr;
  }));
}

let lastHighlighted = -1;
player.addEventListener("timeupdate", () => {
  const t = player.currentTime;
  let idx = -1;
  for (let i = 0; i < transcript.length && transcript[i].start <= t; i++) idx = i;
  if (idx === lastHighlighted) return;
  const rows = $("#transcript").children;
  if (rows[lastHighlighted]) rows[lastHighlighted].classList.remove("current");
  if (rows[idx]) {
    rows[idx].classList.add("current");
    const wrap = $(".transcript-wrap");
    const top = rows[idx].offsetTop - wrap.clientHeight / 2;
    wrap.scrollTo({ top, behavior: "smooth" });
  }
  lastHighlighted = idx;
});

// ---------- history ----------

async function loadRuns() {
  const { runs, active } = await api("/api/runs");
  const body = $("#runs");
  body.replaceChildren(...runs.map((run) => {
    const r = run.report || {};
    const tr = el("tr", { "data-id": run.id });
    const open = el("button", { class: "link-btn", type: "button" }, "Open");
    open.addEventListener("click", () => openRun(run.id));
    const actions = el("td");
    actions.append(open);
    tr.append(
      el("td", {}, run.title),
      el("td", {}, LANGUAGES[r.source_language] || r.source_language || ""),
      el("td", { class: "num" }, fmtDuration(r.video_duration_s)),
      el("td", { class: "num" }, fmtDuration(r.processing_time_s)),
      actions,
    );
    return tr;
  }));
  $("#runs-empty").hidden = runs.length > 0;
  return active;
}

loadRuns().then((active) => {
  if (active) watch(active);
});
