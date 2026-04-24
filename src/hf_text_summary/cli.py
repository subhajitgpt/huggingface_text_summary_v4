from __future__ import annotations

import argparse
import sys

from hf_text_summary.analysis import (
    DEFAULT_INTENT_MODEL,
    DEFAULT_SUMMARY_MODEL,
    analyze_text,
)


def _read_text(path: str | None) -> str:
    if not path or path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Summarize text + key phrases + intent")
    ap.add_argument("--file", "-f", default="-", help="Input file (default: stdin)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--summary-model", default=DEFAULT_SUMMARY_MODEL)
    ap.add_argument("--intent-model", default=DEFAULT_INTENT_MODEL)
    ap.add_argument("--min-len", type=int, default=40)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--phrases", type=int, default=10)
    ap.add_argument(
        "--no-refine",
        action="store_true",
        help="Skip the final reduce/refine summarization pass for long inputs (faster).",
    )
    ap.add_argument(
        "--no-intent",
        action="store_true",
        help="Skip intent detection (fastest).",
    )
    ap.add_argument(
        "--intent-mode",
        default="generate",
        choices=["generate", "zero-shot", "auto"],
        help="Intent detection mode. 'generate' creates a short label from text; 'zero-shot' requires --intent-label.",
    )
    ap.add_argument(
        "--intent-label",
        action="append",
        dest="intent_labels",
        help="Repeat to add candidate intent labels",
    )

    args = ap.parse_args(argv)

    text = _read_text(args.file)
    labels = args.intent_labels or []
    enable_intent = not args.no_intent

    result = analyze_text(
        text,
        device=args.device,
        summary_model=args.summary_model,
        intent_model=args.intent_model,
        summary_min_length=args.min_len,
        summary_max_length=args.max_len,
        summary_refine_final=not args.no_refine,
        keyphrase_top_k=args.phrases,
        enable_intent=enable_intent,
        intent_labels=labels,
        intent_mode=args.intent_mode,
    )

    print("SUMMARY\n------")
    print(result.summary)
    print("\nKEY PHRASES\n----------")
    for p in result.key_phrases:
        print(f"- {p}")

    print("\nINTENT\n------")
    if result.intent_top:
        if result.intent_top.score is None:
            print(f"{result.intent_top.label}")
        else:
            print(f"{result.intent_top.label} ({result.intent_top.score:.3f})")
    else:
        print("<none>")

    return 0
