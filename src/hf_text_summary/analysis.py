from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Any, Literal

import re

import yake
from transformers import pipeline


DEFAULT_SUMMARY_MODEL = "sshleifer/distilbart-cnn-12-6"
# Smaller + typically faster on CPU than many BART-based MNLI models.
DEFAULT_INTENT_MODEL = "typeform/distilbert-base-uncased-mnli"
DEFAULT_DYNAMIC_INTENT_MODEL = "google/flan-t5-small"


@dataclass(frozen=True)
class IntentPrediction:
    label: str
    score: Optional[float] = None


@dataclass(frozen=True)
class AnalysisResult:
    summary: str
    key_phrases: List[str]
    intent_top: Optional[IntentPrediction]
    intent_top_k: List[IntentPrediction]
    meta: Dict[str, Any]


def _clean_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _device_to_pipeline_arg(device: str) -> int:
    device = (device or "cpu").strip().lower()
    if device in {"cpu", "-1"}:
        return -1
    if device in {"cuda", "gpu", "0"}:
        return 0
    raise ValueError("device must be 'cpu' or 'cuda'")


@lru_cache(maxsize=4)
def _summarization_pipeline(model_name: str, device: str):
    return pipeline(
        task="summarization",
        model=model_name,
        device=_device_to_pipeline_arg(device),
    )


@lru_cache(maxsize=4)
def _intent_pipeline(model_name: str, device: str):
    return pipeline(
        task="zero-shot-classification",
        model=model_name,
        device=_device_to_pipeline_arg(device),
    )


@lru_cache(maxsize=4)
def _intent_generation_pipeline(model_name: str, device: str):
    return pipeline(
        task="text2text-generation",
        model=model_name,
        device=_device_to_pipeline_arg(device),
    )


@lru_cache(maxsize=16)
def _keyword_extractor(language: str, max_ngram_size: int, top_k: int):
    return yake.KeywordExtractor(
        lan=language,
        n=max_ngram_size,
        top=top_k,
        dedupLim=0.9,
        dedupFunc="seqm",
    )


def _iter_paragraphs(text: str) -> Iterable[str]:
    for part in re.split(r"\n\s*\n", text):
        part = part.strip()
        if part:
            yield part


def _chunk_by_tokens(text: str, tokenizer, max_input_tokens: int) -> List[str]:
    paragraphs = list(_iter_paragraphs(text))
    if not paragraphs:
        return []

    def token_lengths(parts: Sequence[str]) -> List[int]:
        if not parts:
            return []
        try:
            enc = tokenizer(
                list(parts),
                add_special_tokens=False,
                return_length=True,
                truncation=False,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            lengths = enc.get("length")
            if isinstance(lengths, list) and len(lengths) == len(parts):
                return [int(x) for x in lengths]
        except Exception:
            pass
        return [len(tokenizer.encode(p, add_special_tokens=False)) for p in parts]

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    para_token_lengths = token_lengths(paragraphs)

    for para, para_tokens in zip(paragraphs, para_token_lengths):

        if para_tokens > max_input_tokens:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
            sent_token_lengths = token_lengths(sentences)
            for sent, sent_tokens in zip(sentences, sent_token_lengths):
                sent = sent.strip()
                if not sent:
                    continue
                if sent_tokens > max_input_tokens:
                    for i in range(0, len(sent), 800):
                        slice_ = sent[i : i + 800].strip()
                        if slice_:
                            chunks.append(slice_)
                    continue

                if current and current_tokens + sent_tokens > max_input_tokens:
                    chunks.append("\n\n".join(current))
                    current = []
                    current_tokens = 0

                current.append(sent)
                current_tokens += sent_tokens
            continue

        if current and current_tokens + para_tokens > max_input_tokens:
            chunks.append("\n\n".join(current))
            current = []
            current_tokens = 0

        current.append(para)
        current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _summarize_one(
    summarizer,
    text: str,
    min_length: int,
    max_length: int,
) -> str:
    out = summarizer(
        text,
        min_length=min_length,
        max_length=max_length,
        do_sample=False,
        truncation=True,
    )
    if not out:
        return ""
    return (out[0].get("summary_text") or "").strip()


def _auto_batch_size(device: str) -> int:
    device = (device or "cpu").strip().lower()
    if device in {"cuda", "gpu", "0"}:
        return 8
    return 2


def summarize_text(
    text: str,
    *,
    model_name: str = DEFAULT_SUMMARY_MODEL,
    device: str = "cpu",
    max_input_tokens: int = 900,
    min_length: int = 40,
    max_length: int = 160,
    refine_final_summary: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """Summarize text using a chunked (map-reduce) approach.

    Returns (summary, meta).
    """

    text = _clean_text(text)
    if not text:
        return "", {"chunks": 0, "model": model_name}

    summarizer = _summarization_pipeline(model_name, device)
    tokenizer = summarizer.tokenizer

    chunks = _chunk_by_tokens(text, tokenizer, max_input_tokens=max_input_tokens)
    if not chunks:
        return "", {"chunks": 0, "model": model_name}

    # Speed: run summarization in batch instead of per-chunk calls.
    try:
        out = summarizer(
            chunks,
            min_length=min_length,
            max_length=max_length,
            do_sample=False,
            truncation=True,
            batch_size=min(len(chunks), _auto_batch_size(device)),
        )
    except TypeError:
        out = summarizer(
            chunks,
            min_length=min_length,
            max_length=max_length,
            do_sample=False,
            truncation=True,
        )

    if isinstance(out, list):
        chunk_summaries = [
            (o.get("summary_text") or "").strip() for o in out if isinstance(o, dict)
        ]
    else:
        chunk_summaries = []
    chunk_summaries = [s for s in chunk_summaries if s]

    if not chunk_summaries:
        return "", {"chunks": len(chunks), "model": model_name}

    if len(chunk_summaries) == 1:
        return chunk_summaries[0], {"chunks": len(chunks), "model": model_name}

    if not refine_final_summary:
        # Fast-path: avoid the extra summarization pass.
        joined = "\n".join(chunk_summaries)
        return joined, {
            "chunks": len(chunks),
            "model": model_name,
            "stage": "map-only",
        }

    combined = "\n".join(chunk_summaries)
    final = _summarize_one(
        summarizer,
        combined,
        min_length=max(20, min_length // 2),
        max_length=max(60, max_length),
    )

    return final or "\n".join(chunk_summaries), {
        "chunks": len(chunks),
        "model": model_name,
        "stage": "map-reduce",
    }


def extract_key_phrases(
    text: str,
    *,
    top_k: int = 10,
    language: str = "en",
    max_ngram_size: int = 3,
) -> List[str]:
    text = _clean_text(text)
    if not text:
        return []

    kw_extractor = _keyword_extractor(language, max_ngram_size, top_k)
    keywords = kw_extractor.extract_keywords(text)

    phrases = [phrase.strip() for phrase, _score in keywords if phrase.strip()]

    seen = set()
    out: List[str] = []
    for p in phrases:
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def detect_intent(
    text: str,
    *,
    candidate_labels: Sequence[str],
    model_name: str = DEFAULT_INTENT_MODEL,
    device: str = "cpu",
    top_k: int = 3,
) -> Tuple[Optional[IntentPrediction], List[IntentPrediction], Dict[str, Any]]:
    text = _clean_text(text)
    labels = [l.strip() for l in candidate_labels if l and l.strip()]
    if not text or not labels:
        return None, [], {"model": model_name}

    classifier = _intent_pipeline(model_name, device)
    res = classifier(text, labels, multi_label=False, truncation=True)

    ranked = list(zip(res.get("labels", []), res.get("scores", [])))
    preds = [IntentPrediction(label=l, score=float(s)) for l, s in ranked[: max(1, top_k)]]
    top = preds[0] if preds else None
    return top, preds, {"model": model_name}


def generate_intent(
    text: str,
    *,
    model_name: str = DEFAULT_DYNAMIC_INTENT_MODEL,
    device: str = "cpu",
    max_new_tokens: int = 16,
) -> Tuple[Optional[IntentPrediction], Dict[str, Any]]:
    """Generate a short intent label directly from text (no candidate labels)."""

    text = _clean_text(text)
    if not text:
        return None, {"model": model_name}

    generator = _intent_generation_pipeline(model_name, device)

    text_snippet = text[:2000]
    prompt = (
        "Return a short intent label (2-6 words) describing what the user wants. "
        "Do not return a sentence.\n\n"
        f"Text:\n{text_snippet}\n\nIntent:"
    )

    out = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        truncation=True,
    )
    generated = ""
    if isinstance(out, list) and out and isinstance(out[0], dict):
        generated = (out[0].get("generated_text") or "").strip()

    generated = generated.strip().strip('"').strip("'")
    generated = re.sub(r"\s+", " ", generated)
    generated = generated.strip(" .,:;\n\t")

    if not generated:
        return None, {"model": model_name}

    return IntentPrediction(label=generated, score=None), {"model": model_name, "mode": "generate"}


def _parse_synopsis_and_keyphrases(output_text: str) -> Tuple[str, List[str]]:
    text = (output_text or "").strip()
    if not text:
        return "", []

    m = re.search(r"(?is)\bSYNOPSIS\s*:\s*(.*?)\n\s*KEYPHRASES\s*:\s*(.*)$", text)
    if not m:
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        return (parts[0] if parts else text), []

    synopsis = (m.group(1) or "").strip()
    key_block = (m.group(2) or "").strip()

    phrases: List[str] = []
    for line in key_block.splitlines():
        line = line.strip()
        line = re.sub(r"^[-*•\d.\)\s]+", "", line).strip()
        if not line:
            continue
        line = re.sub(r"\s+", " ", line)
        phrases.append(line)

    seen = set()
    out: List[str] = []
    for p in phrases:
        k = p.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return synopsis, out


def _word_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = [w.lower() for w in re.findall(r"[A-Za-z0-9']+", text or "")]
    if len(words) < n:
        return set()
    return {tuple(words[i : i + n]) for i in range(0, len(words) - n + 1)}


def _has_high_overlap(generated: str, source: str, *, ngram: int = 6) -> bool:
    g = (generated or "").strip()
    s = (source or "").strip()
    if not g or not s:
        return False

    src = _word_ngrams(s, ngram)
    if not src:
        return False

    gen_words = [w.lower() for w in re.findall(r"[A-Za-z0-9']+", g)]
    if len(gen_words) < ngram:
        return False

    for i in range(0, len(gen_words) - ngram + 1):
        if tuple(gen_words[i : i + ngram]) in src:
            return True
    return False


def generate_synopsis_and_keyphrases(
    text: str,
    *,
    base_summary: str = "",
    top_k: int = 8,
    model_name: str = DEFAULT_DYNAMIC_INTENT_MODEL,
    device: str = "cpu",
    max_new_tokens: int = 180,
) -> Tuple[str, List[str], Dict[str, Any]]:
    """Generate an abstractive synopsis (and optionally keyphrases)."""

    text = _clean_text(text)
    base_summary = (base_summary or "").strip()
    if not text and not base_summary:
        return "", [], {"model": model_name}

    generator = _intent_generation_pipeline(model_name, device)
    requested_k = int(top_k or 0)
    requested_k = max(0, min(requested_k, 20))

    text_snippet = text[:2000]
    summary_snippet = base_summary[:1200] if base_summary else ""

    if requested_k > 0:
        prompt = (
            "You are a helpful analyst. Create a HIGH-LEVEL, ABSTRactive synopsis and keyphrases.\n"
            "CRITICAL: Do not copy or reuse any sentence from the input. Do not quote.\n"
            "CRITICAL: Avoid using 6+ consecutive words that appear in the input. Paraphrase strongly.\n"
            "Write in professional, neutral tone.\n\n"
            "SYNOPSIS requirements:\n"
            "- 2 to 3 short sentences\n"
            "- capture the main situation, impact, and requested action\n"
            "- avoid minor details\n\n"
            "KEYPHRASES requirements:\n"
            f"- exactly {requested_k} short noun-phrases (2-5 words)\n"
            "- no full sentences\n\n"
            "Output EXACTLY in this format:\n"
            "SYNOPSIS: <text>\n"
            "KEYPHRASES:\n"
            "- <phrase>\n\n"
            f"TEXT:\n{text_snippet}\n\n"
        )
        if summary_snippet:
            prompt += f"ROUGH SUMMARY (may be imperfect, paraphrase it):\n{summary_snippet}\n\n"
        prompt += "SYNOPSIS:"
    else:
        prompt = (
            "You are a helpful analyst. Create a HIGH-LEVEL, ABSTRactive synopsis.\n"
            "CRITICAL: Do not copy or reuse any sentence from the input. Do not quote.\n"
            "CRITICAL: Avoid using 6+ consecutive words that appear in the input. Paraphrase strongly.\n"
            "Write in professional, neutral tone.\n\n"
            "SYNOPSIS requirements:\n"
            "- 2 to 3 short sentences\n"
            "- capture the main situation, impact, and requested action\n"
            "- avoid minor details\n\n"
            "Output EXACTLY in this format:\n"
            "SYNOPSIS: <text>\n\n"
            f"TEXT:\n{text_snippet}\n\n"
        )
        if summary_snippet:
            prompt += f"ROUGH SUMMARY (may be imperfect, paraphrase it):\n{summary_snippet}\n\n"
        prompt += "SYNOPSIS:"

    def _generate(prompt_text: str, *, strict: bool) -> str:
        params: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "truncation": True,
            "num_beams": 4,
            "early_stopping": True,
            "no_repeat_ngram_size": 4 if not strict else 6,
            "repetition_penalty": 1.15 if not strict else 1.25,
            "length_penalty": 1.0,
        }
        out_local = generator(prompt_text, **params)
        if isinstance(out_local, list) and out_local and isinstance(out_local[0], dict):
            return (out_local[0].get("generated_text") or "").strip()
        return ""

    generated = _generate(prompt, strict=False)
    synopsis, keyphrases = _parse_synopsis_and_keyphrases(generated)

    retried = False
    if synopsis and _has_high_overlap(synopsis, text, ngram=6):
        retried = True
        strict_prompt = prompt + "\n\nREWRITE: Fully paraphrase. Use different wording than the input."
        generated2 = _generate(strict_prompt, strict=True)
        synopsis2, keyphrases2 = _parse_synopsis_and_keyphrases(generated2)
        if synopsis2 and not _has_high_overlap(synopsis2, text, ngram=6):
            synopsis = synopsis2
            if keyphrases2:
                keyphrases = keyphrases2

    if requested_k > 0 and not keyphrases and text:
        keyphrases = extract_key_phrases(text, top_k=requested_k)

    return synopsis.strip(), (keyphrases[:requested_k] if requested_k > 0 else []), {
        "model": model_name,
        "mode": "generate",
        "requested_keyphrases": requested_k,
        "retried_for_overlap": retried,
    }


def analyze_text(
    text: str,
    *,
    summary_model: str = DEFAULT_SUMMARY_MODEL,
    intent_model: str = DEFAULT_INTENT_MODEL,
    device: str = "cpu",
    summary_min_length: int = 40,
    summary_max_length: int = 160,
    summary_refine_final: bool = True,
    keyphrase_top_k: int = 10,
    enable_intent: bool = True,
    intent_labels: Optional[Sequence[str]] = None,
    intent_top_k: int = 3,
    intent_mode: Literal["auto", "zero-shot", "generate"] = "generate",
) -> AnalysisResult:
    text = _clean_text(text)

    summary, sum_meta = summarize_text(
        text,
        model_name=summary_model,
        device=device,
        min_length=summary_min_length,
        max_length=summary_max_length,
        refine_final_summary=summary_refine_final,
    )

    synopsis, key_phrases, synopsis_meta = generate_synopsis_and_keyphrases(
        text,
        base_summary=summary,
        top_k=keyphrase_top_k,
        model_name=intent_model,
        device=device,
    )
    if synopsis:
        summary = synopsis
        sum_meta = {**(sum_meta or {}), "rewritten": True, "synopsis": synopsis_meta}

    if enable_intent:
        labels = [l.strip() for l in (intent_labels or []) if l and l.strip()]

        mode = intent_mode
        if mode == "auto":
            mode = "zero-shot" if labels else "generate"

        if mode == "zero-shot":
            if labels:
                intent_top, intent_topk, intent_meta = detect_intent(
                    text,
                    candidate_labels=labels,
                    model_name=intent_model,
                    device=device,
                    top_k=intent_top_k,
                )
            else:
                intent_top, intent_topk, intent_meta = (
                    None,
                    [],
                    {"model": intent_model, "skipped": True, "reason": "no_candidate_labels"},
                )
        else:
            intent_top, intent_meta = generate_intent(
                text,
                model_name=intent_model,
                device=device,
            )
            intent_topk = [intent_top] if intent_top else []
    else:
        intent_top, intent_topk, intent_meta = None, [], {"model": intent_model, "skipped": True}

    return AnalysisResult(
        summary=summary,
        key_phrases=key_phrases,
        intent_top=intent_top,
        intent_top_k=intent_topk,
        meta={
            "summary": sum_meta,
            "intent": intent_meta,
            "device": device,
            "chars": len(text),
        },
    )
