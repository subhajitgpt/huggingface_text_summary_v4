from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Any, Literal

import re

import yake



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
    summary_points: List[str]
    key_phrases: List[str]
    intent_top: Optional[IntentPrediction]
    intent_top_k: List[IntentPrediction]
    meta: Dict[str, Any]


def choose_summary_point_count(
    text: str,
    *,
    min_points: int = 5,
    max_points: int = 10,
) -> int:
    """Choose how many summary points to display.

    Heuristic: scale from min_points to max_points as the input gets longer.
    """

    min_points = int(min_points)
    max_points = int(max_points)
    if min_points < 1:
        min_points = 1
    if max_points < min_points:
        max_points = min_points

    words = len(re.findall(r"[A-Za-z0-9']+", text or ""))
    if words <= 200:
        return min_points
    if words >= 1600:
        return max_points

    span = max_points - min_points
    if span <= 0:
        return min_points

    # Add 1 point roughly every ~280 words beyond the first 200.
    steps = min(span, max(0, (words - 200) // 280))
    return min_points + int(steps)


def _iter_candidate_sentences(text: str) -> List[str]:
    text = _clean_text(text)
    if not text:
        return []

    sentences: List[str] = []
    for raw_line in re.split(r"\n+", text):
        line = (raw_line or "").strip()
        if not line:
            continue

        # Preserve bullet/numbered list items as units.
        line = re.sub(r"^\s*([-*•]|\d+[\).])\s+", "", line).strip()
        if not line:
            continue

        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", line) if p.strip()]
        if not parts:
            continue
        sentences.extend(parts)

    out: List[str] = []
    seen = set()
    for s in sentences:
        s = re.sub(r"\s+", " ", s).strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _token_set(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z0-9']+", text or "") if len(w) >= 3}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    union = len(a | b)
    return inter / union if union else 0.0


def generate_summary_points(
    text: str,
    *,
    key_phrases: Optional[Sequence[str]] = None,
    min_points: int = 5,
    max_points: int = 10,
) -> Tuple[List[str], Dict[str, Any]]:
    """Create an extractive bullet-style summary.

    Picks a small set of high-signal sentences from the original text.
    """

    text = _clean_text(text)
    if not text:
        return [], {"target_points": 0, "selected": 0, "mode": "extractive"}

    target = choose_summary_point_count(text, min_points=min_points, max_points=max_points)
    candidates = _iter_candidate_sentences(text)
    if not candidates:
        return [], {"target_points": target, "selected": 0, "mode": "extractive"}

    phrases = [p.strip() for p in (key_phrases or []) if p and p.strip()]
    if not phrases:
        phrases = extract_key_phrases(text, top_k=20)
    phrase_lc = [p.lower() for p in phrases]

    scored: List[Tuple[float, int, str]] = []
    for idx, sent in enumerate(candidates):
        sent_lc = sent.lower()
        hits = sum(1 for p in phrase_lc if p and p in sent_lc)

        word_count = len(re.findall(r"[A-Za-z0-9']+", sent))
        if word_count < 6:
            length_bonus = -1.0
        elif word_count > 40:
            length_bonus = -0.25
        else:
            length_bonus = 0.25

        position_bonus = 0.35 if idx < 3 else (0.15 if idx < 8 else 0.0)
        score = (hits * 1.0) + length_bonus + position_bonus
        scored.append((score, idx, sent))

    scored.sort(key=lambda t: (t[0], -t[1]), reverse=True)

    selected: List[Tuple[int, str]] = []
    selected_sets: List[set[str]] = []
    for score, idx, sent in scored:
        if len(selected) >= target:
            break
        # Avoid picking very low-signal sentences unless we have no choice.
        if score < 0.0 and len(selected) >= max(1, target // 2):
            continue

        tokens = _token_set(sent)
        if any(_jaccard(tokens, prev) >= 0.82 for prev in selected_sets):
            continue

        selected.append((idx, sent))
        selected_sets.append(tokens)

    if not selected:
        # Fallback: return the first few sentences.
        selected = list(enumerate(candidates[:target]))

    selected.sort(key=lambda t: t[0])
    points = [s for _idx, s in selected]
    return points, {"target_points": target, "selected": len(points), "mode": "extractive"}


def _clean_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _normalize_device(device: str) -> str:
    device = (device or "cpu").strip().lower()
    if device in {"cpu", "-1"}:
        return "cpu"
    if device in {"cuda", "gpu", "0"}:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    raise ValueError("device must be 'cpu' or 'cuda'")


def _auto_batch_size(device: str) -> int:
    device = _normalize_device(device)
    return 8 if device == "cuda" else 2


@lru_cache(maxsize=8)
def _seq2seq_components(model_name: str, device: str):
    """Load a seq2seq model + tokenizer for summarization/generation."""

    device = _normalize_device(device)
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(torch.device(device))
    model.eval()
    return tokenizer, model


@lru_cache(maxsize=8)
def _sequence_classifier_components(model_name: str, device: str):
    """Load a sequence classification model + tokenizer (used for NLI/zero-shot)."""

    device = _normalize_device(device)
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(torch.device(device))
    model.eval()
    return tokenizer, model


def _generate_seq2seq_batch(
    inputs: Sequence[str],
    *,
    model_name: str,
    device: str,
    max_source_tokens: int | None = None,
    batch_size: int = 2,
    generation_kwargs: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Run seq2seq generation over a list of inputs with batching."""

    tokenizer, model = _seq2seq_components(model_name, device)
    import torch

    gen_kwargs = dict(generation_kwargs or {})

    outputs: List[str] = []
    for i in range(0, len(inputs), max(1, int(batch_size))):
        batch = [b for b in inputs[i : i + batch_size] if (b or "").strip()]
        if not batch:
            continue

        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_source_tokens,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}

        with torch.no_grad():
            try:
                out_ids = model.generate(**enc, **gen_kwargs)
            except TypeError:
                # Backwards compat for older Transformers: translate *new_tokens to *length.
                compat = dict(gen_kwargs)
                max_new = compat.pop("max_new_tokens", None)
                min_new = compat.pop("min_new_tokens", None)
                if max_new is not None:
                    compat["max_length"] = int(enc["input_ids"].shape[1] + int(max_new))
                if min_new is not None:
                    compat["min_length"] = int(enc["input_ids"].shape[1] + int(min_new))
                out_ids = model.generate(**enc, **compat)

        decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        outputs.extend([(d or "").strip() for d in decoded])

    return outputs


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
    text: str,
    *,
    model_name: str,
    device: str,
    max_source_tokens: int,
    min_new_tokens: int,
    max_new_tokens: int,
) -> str:
    out = _generate_seq2seq_batch(
        [text],
        model_name=model_name,
        device=device,
        max_source_tokens=max_source_tokens,
        batch_size=1,
        generation_kwargs={
            "min_new_tokens": int(min_new_tokens),
            "max_new_tokens": int(max_new_tokens),
            "do_sample": False,
        },
    )
    return (out[0] if out else "").strip()


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

    device = _normalize_device(device)
    tokenizer, _model = _seq2seq_components(model_name, device)

    chunks = _chunk_by_tokens(text, tokenizer, max_input_tokens=max_input_tokens)
    if not chunks:
        return "", {"chunks": 0, "model": model_name}

    chunk_summaries = _generate_seq2seq_batch(
        chunks,
        model_name=model_name,
        device=device,
        max_source_tokens=max_input_tokens,
        batch_size=min(len(chunks), _auto_batch_size(device)),
        generation_kwargs={
            "min_new_tokens": int(min_length),
            "max_new_tokens": int(max_length),
            "do_sample": False,
        },
    )
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
        combined,
        model_name=model_name,
        device=device,
        max_source_tokens=max_input_tokens,
        min_new_tokens=max(10, int(min_length) // 2),
        max_new_tokens=max(60, int(max_length)),
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

    device = _normalize_device(device)
    tokenizer, model = _sequence_classifier_components(model_name, device)
    import torch

    # NLI-style zero-shot: premise is the text, hypothesis is label statement.
    premises = [text] * len(labels)
    hypotheses = [f"This text is about {lbl}." for lbl in labels]
    enc = tokenizer(
        premises,
        hypotheses,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)

    # Try to locate the entailment label id, else fall back to common MNLI convention.
    cfg = getattr(model, "config", None)
    entailment_id: int | None = None
    if cfg is not None:
        label2id = {str(k).lower(): int(v) for k, v in (getattr(cfg, "label2id", {}) or {}).items()}
        if "entailment" in label2id:
            entailment_id = label2id["entailment"]
        else:
            id2label = {int(k): str(v).lower() for k, v in (getattr(cfg, "id2label", {}) or {}).items()}
            for idx, name in id2label.items():
                if "entailment" in name:
                    entailment_id = int(idx)
                    break
            if entailment_id is None and int(getattr(cfg, "num_labels", 0) or 0) == 3:
                entailment_id = 2

    if entailment_id is None:
        entailment_id = min(2, probs.shape[-1] - 1)

    scores = probs[:, int(entailment_id)].detach().cpu().tolist()
    ranked = sorted(zip(labels, scores), key=lambda t: t[1], reverse=True)
    preds = [IntentPrediction(label=l, score=float(s)) for l, s in ranked[: max(1, top_k)]]
    top = preds[0] if preds else None
    return top, preds, {"model": model_name, "mode": "nli"}


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

    device = _normalize_device(device)

    text_snippet = _sample_text_for_prompt(text, max_chars=2000)
    prompt = (
        "Return a short intent label (2-6 words) describing what the user wants. "
        "Do not return a sentence.\n\n"
        f"Text:\n{text_snippet}\n\nIntent:"
    )

    out = _generate_seq2seq_batch(
        [prompt],
        model_name=model_name,
        device=device,
        max_source_tokens=512,
        batch_size=1,
        generation_kwargs={
            "max_new_tokens": int(max_new_tokens),
            "do_sample": False,
            "num_beams": 4,
            "early_stopping": True,
        },
    )
    generated = (out[0] if out else "").strip()

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


def _sample_text_for_prompt(text: str, *, max_chars: int) -> str:
    """Take a representative slice of long text for prompt-based generation."""

    text = (text or "").strip()
    max_chars = int(max_chars)
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text

    # Bias toward the beginning (context) but keep the end (often contains ask/summary).
    head = max(200, int(max_chars * 0.6))
    tail = max(200, max_chars - head)
    return (text[:head].rstrip() + "\n\n…\n\n" + text[-tail:].lstrip()).strip()


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

    device = _normalize_device(device)
    requested_k = int(top_k or 0)
    requested_k = max(0, min(requested_k, 20))

    text_snippet = _sample_text_for_prompt(text, max_chars=2000)
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
            "max_new_tokens": int(max_new_tokens),
            "do_sample": False,
            "num_beams": 4,
            "early_stopping": True,
            "no_repeat_ngram_size": 4 if not strict else 6,
            "repetition_penalty": 1.15 if not strict else 1.25,
            "length_penalty": 1.0,
        }
        out_local = _generate_seq2seq_batch(
            [prompt_text],
            model_name=model_name,
            device=device,
            max_source_tokens=1024,
            batch_size=1,
            generation_kwargs=params,
        )
        return (out_local[0] if out_local else "").strip()

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

    summary_points, points_meta = generate_summary_points(
        text,
        key_phrases=key_phrases,
        min_points=5,
        max_points=10,
    )

    return AnalysisResult(
        summary=summary,
        summary_points=summary_points,
        key_phrases=key_phrases,
        intent_top=intent_top,
        intent_top_k=intent_topk,
        meta={
            "summary": sum_meta,
            "summary_points": points_meta,
            "intent": intent_meta,
            "device": device,
            "chars": len(text),
        },
    )
