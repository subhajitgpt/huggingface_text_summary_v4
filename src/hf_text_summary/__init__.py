"""hf_text_summary

Production-ready text summarization + key phrase extraction + intent inference.
"""

from .analysis import (
    DEFAULT_INTENT_MODEL,
    DEFAULT_DYNAMIC_INTENT_MODEL,
    DEFAULT_SUMMARY_MODEL,
    AnalysisResult,
    IntentPrediction,
    analyze_text,
    detect_intent,
    generate_intent,
    extract_key_phrases,
    summarize_text,
)

from .text_extract import extract_text_from_bytes, extract_text_from_path, supported_extensions

__all__ = [
    "DEFAULT_INTENT_MODEL",
    "DEFAULT_DYNAMIC_INTENT_MODEL",
    "DEFAULT_SUMMARY_MODEL",
    "IntentPrediction",
    "AnalysisResult",
    "summarize_text",
    "extract_key_phrases",
    "detect_intent",
    "generate_intent",
    "analyze_text",
    "extract_text_from_bytes",
    "extract_text_from_path",
    "supported_extensions",
]
