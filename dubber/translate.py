"""Translation to English.

Two backends:
  * "nllb"   – Meta's NLLB-200 (1.3B, distilled) on CTranslate2, int8 on CPU. Free, offline,
               200 languages including Hindi and other Indic languages. The default.
  * "claude" – context-aware and length-aware: it sees neighbouring lines and each line's
               time budget, so it produces natural spoken English that fits the original
               timing and can fix obvious transcription errors. Needs ANTHROPIC_API_KEY.
"""

import json
import os
import re

from tqdm import tqdm

from . import log
from .segments import Segment

WORDS_PER_SECOND = 2.6  # comfortable English speaking pace used as the length target


def translate(segments: list[Segment], source_lang: str, backend: str) -> None:
    """Fill `seg.english` in place."""
    if backend == "auto":
        backend = "claude" if os.environ.get("ANTHROPIC_API_KEY") else "nllb"
    log.info(f"backend: {backend}")
    if source_lang == "en":
        for seg in segments:
            seg.english = seg.text
        return
    if backend == "claude":
        _translate_claude(segments, source_lang)
    else:
        _translate_nllb(segments, source_lang)


# --- NLLB (free, offline) --------------------------------------------------------------------

NLLB_MODEL = "OpenNMT/nllb-200-distilled-1.3B-ct2-int8"

# Whisper language code -> NLLB (FLORES-200) code, for the languages Whisper handles well.
NLLB_CODES = {
    "de": "deu_Latn", "fr": "fra_Latn", "es": "spa_Latn", "it": "ita_Latn", "pt": "por_Latn",
    "nl": "nld_Latn", "pl": "pol_Latn", "ru": "rus_Cyrl", "uk": "ukr_Cyrl", "tr": "tur_Latn",
    "ar": "arb_Arab", "fa": "pes_Arab", "he": "heb_Hebr", "ja": "jpn_Jpan", "ko": "kor_Hang",
    "zh": "zho_Hans", "vi": "vie_Latn", "id": "ind_Latn", "th": "tha_Thai", "sv": "swe_Latn",
    "da": "dan_Latn", "no": "nob_Latn", "fi": "fin_Latn", "cs": "ces_Latn", "el": "ell_Grek",
    "hu": "hun_Latn", "ro": "ron_Latn", "hi": "hin_Deva", "bn": "ben_Beng", "ta": "tam_Taml",
    "te": "tel_Telu", "mr": "mar_Deva", "gu": "guj_Gujr", "kn": "kan_Knda", "ml": "mal_Mlym",
    "pa": "pan_Guru", "ur": "urd_Arab", "ne": "npi_Deva",
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?।…])\s+")
_LEADING_DASH = re.compile(r"^[-–—]\s*")


def _translate_nllb(segments: list[Segment], source_lang: str) -> None:
    import ctranslate2
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer

    src_code = NLLB_CODES.get(source_lang)
    if not src_code:
        raise SystemExit(f"No NLLB mapping for language '{source_lang}'. Add it to NLLB_CODES "
                         "or use --translator claude.")

    log.info(f"loading {NLLB_MODEL}")
    path = snapshot_download(NLLB_MODEL)
    tokenizer = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
    model = ctranslate2.Translator(path, device="cpu", compute_type="int8",
                                   intra_threads=min(8, os.cpu_count() or 4))

    # NLLB is trained on single sentences and silently drops text from multi-sentence
    # inputs, so translate sentence by sentence and re-join.
    pieces = [(i, s) for i, seg in enumerate(segments)
              for s in _SENTENCE_SPLIT.split(seg.text) if s.strip()]
    outputs: list[list[str]] = [[] for _ in segments]

    chunk = 32
    for start in tqdm(range(0, len(pieces), chunk), desc="    translating", unit="batch"):
        batch = pieces[start:start + chunk]
        source = [[src_code] + tokenizer.encode(text, add_special_tokens=False).tokens + ["</s>"]
                  for _, text in batch]
        results = model.translate_batch(source, target_prefix=[["eng_Latn"]] * len(source),
                                        beam_size=2, max_batch_size=16,
                                        repetition_penalty=1.1, no_repeat_ngram_size=4)
        for (i, _), result in zip(batch, results):
            ids = [tokenizer.token_to_id(t) for t in result.hypotheses[0][1:]]
            text = tokenizer.decode(ids, skip_special_tokens=True).strip()
            outputs[i].append(_LEADING_DASH.sub("", text))  # NLLB mimics subtitle dialogue dashes
        log.progress("Translate", start + len(batch), len(pieces))

    for seg, parts in zip(segments, outputs):
        seg.english = " ".join(parts)


# --- Claude (context + timing aware) ---------------------------------------------------------

BATCH_SIZE = 40
CONTEXT_LINES = 5

SYSTEM_PROMPT = """You are a professional film dubbing translator. You translate transcript lines \
from a video into natural, spoken English that a voice actor will read aloud over the original \
timing.

Rules:
- Translate meaning and tone, not word-for-word. Use idiomatic, conversational English.
- Each line has a time budget in seconds and a target word count. Stay close to the target: \
it is better to condense than to overrun, because long lines must be sped up to fit.
- The transcript comes from speech recognition and may contain mistakes; use context to infer \
what was actually said.
- Keep names, numbers and technical terms accurate.
- Return exactly one translation per input id, in the same order. Never merge or split lines.
- Output only spoken words: no stage directions, brackets or notes."""

SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "english": {"type": "string"}},
                "required": ["id", "english"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["lines"],
    "additionalProperties": False,
}


def _translate_claude(segments: list[Segment], source_lang: str) -> None:
    import anthropic

    client = anthropic.Anthropic(max_retries=5)
    failed: list[Segment] = []
    for start in tqdm(range(0, len(segments), BATCH_SIZE), desc="    translating", unit="batch"):
        batch = segments[start:start + BATCH_SIZE]
        context = segments[max(0, start - CONTEXT_LINES):start]
        try:
            results = _claude_batch(client, batch, context, start, source_lang)
        except Exception as exc:  # never lose a multi-hour run to one bad batch
            log.info(f"Claude batch at line {start} failed ({exc}); will use NLLB for it")
            results = {}
        for i, seg in enumerate(batch, start):
            seg.english = results.get(i, "").strip()
            if not seg.english:
                failed.append(seg)
        log.progress("Translate", start + len(batch), len(segments))
    if failed:
        _translate_nllb(failed, source_lang)


def _claude_batch(client, batch, context, first_id, source_lang) -> dict[int, str]:
    payload = {
        "source_language": source_lang,
        "previous_lines_for_context": [{"text": s.text, "english": s.english} for s in context],
        "lines_to_translate": [
            {"id": i, "text": s.text, "seconds": round(s.duration, 1),
             "target_words": max(1, round(s.duration * WORDS_PER_SECOND))}
            for i, s in enumerate(batch, first_id)
        ],
    }
    response = client.beta.messages.create(
        model="claude-opus-5",
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        output_config={"effort": "medium",
                       "format": {"type": "json_schema", "schema": SCHEMA}},
        # Server-side fallback: if the model declines a batch, it is retried on another model.
        betas=["server-side-fallback-2026-07-01"],
        extra_body={"fallbacks": "default"},
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("request refused")
    text = next(b.text for b in response.content if b.type == "text")
    return {line["id"]: line["english"] for line in json.loads(text)["lines"]}
