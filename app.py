from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List

import streamlit as st


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


# If the user runs `python app.py`, Streamlit session runtime won't exist and
# they'll see warnings like: "Session state does not function when running a script
# without `streamlit run`". Relaunch seamlessly.
if __name__ == "__main__" and not _running_in_streamlit():
    raise SystemExit(
        subprocess.call(
            [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]]
        )
    )

try:
    from hf_text_summary import (
        analyze_text,
        DEFAULT_DYNAMIC_INTENT_MODEL,
        DEFAULT_SUMMARY_MODEL,
    )
    from hf_text_summary.text_extract import extract_text_from_bytes
except ModuleNotFoundError:
    # Allows `streamlit run app.py` without requiring an editable install.
    import sys

    sys.path.append(str(Path(__file__).parent / "src"))
    from hf_text_summary import (
        analyze_text,
        DEFAULT_DYNAMIC_INTENT_MODEL,
        DEFAULT_SUMMARY_MODEL,
    )
    from hf_text_summary.text_extract import extract_text_from_bytes


SAMPLE_TEXT = """\
Our mobile app started crashing after yesterday's update. Users report that it closes immediately when opening the Settings screen.\

This seems to happen on Android 14 devices, especially on Pixel phones. We need a quick fix or a rollback plan.\

Can you investigate the logs, identify the root cause, and propose a patch along with an ETA?\
"""
st.set_page_config(page_title="Hugging Face Text Summary", page_icon="📝", layout="wide")

with st.container(border=True):
    st.title("Hugging Face Text Summary")
    st.caption("Paste text, get a summary, and a generated intent label.")

with st.sidebar:
    with st.expander("Models", expanded=True):
        summary_model = st.text_input(
            "Summarization model",
            value=DEFAULT_SUMMARY_MODEL,
            help="Any Hugging Face summarization model compatible with Transformers pipelines.",
        )
        intent_model = st.text_input(
            "Intent model (dynamic)",
            value=DEFAULT_DYNAMIC_INTENT_MODEL,
            help="A text2text model that generates a short intent label from the input.",
        )

    with st.expander("Runtime", expanded=True):
        device_options = ["cpu"] + (["cuda"] if _cuda_available() else [])

        # If a previous run stored 'cuda' in session state on a non-CUDA host,
        # Streamlit will try to restore it. Force it back to 'cpu'.
        if st.session_state.get("Device") == "cuda" and "cuda" not in device_options:
            st.session_state["Device"] = "cpu"

        device = st.selectbox(
            "Device",
            options=device_options,
            index=0,
            help="Use 'cuda' only if a GPU + CUDA Torch is available.",
        )

    with st.expander("Summary", expanded=True):
        summary_min = st.slider("Min length", 10, 200, 40, step=5)
        summary_max = st.slider("Max length", 30, 400, 160, step=10)
        summary_refine_final = st.checkbox(
            "Refine final summary (slower)",
            value=True,
            help="For long text, an extra pass improves coherence but costs time.",
        )

    with st.expander("Intent", expanded=True):
        enable_intent = st.checkbox(
            "Enable intent",
            value=True,
            help="Generates a short intent label from the input.",
        )

    st.caption("Models are downloaded once and cached locally by Transformers.")

run = False

col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    if "input_text" not in st.session_state:
        st.session_state["input_text"] = SAMPLE_TEXT

    if "input_mode" not in st.session_state:
        st.session_state["input_mode"] = "Text"

    with st.container(border=True):
        st.subheader("Input")
        input_mode = st.radio(
            "Input mode",
            options=["Text", "File"],
            horizontal=True,
            label_visibility="collapsed",
            key="input_mode",
        )

        if input_mode == "File":
            st.file_uploader(
                "Upload a file",
                type=["pdf", "docx", "txt", "md", "doc"],
                help="Supported: PDF, DOCX, TXT/MD. Note: scanned/image-only PDFs may have no extractable text unless OCR is enabled (optional). Legacy .doc is not supported; upload .docx or PDF.",
                key="uploaded_file",
            )
            # Keep the text box available as a fallback / preview.
            text = ""
        else:
            text = st.text_area(
                "Text",
                height=280,
                placeholder="Paste text to summarize...",
                label_visibility="collapsed",
                key="input_text",
            )

        run = st.button("Analyze", type="primary", use_container_width=True)

with col_right:
    with st.container(border=True):
        st.subheader("How it works")
        st.markdown(
            "\n".join(
                [
                    "- Long inputs use a chunked (map-reduce) summarization.",
                    "- Intent is generated from the input (no label list needed).",
                    "- First run downloads models into your Hugging Face cache.",
                ]
            )
        )

if run:
    cleaned = (text or "").strip()
    labels: List[str] = []

    # If a file was uploaded, prefer its extracted text.
    if st.session_state.get("input_mode") == "File":
        uploaded_now = st.session_state.get("uploaded_file")
        if uploaded_now is not None:
            try:
                cleaned = extract_text_from_bytes(uploaded_now.name, uploaded_now.getvalue()).strip()
            except Exception as e:
                st.error(f"Failed to read uploaded file: {e}")
                st.stop()
        else:
            st.warning("Please upload a file.")
            st.stop()

    if not cleaned:
        st.warning("Please enter some text." if st.session_state.get("input_mode") != "File" else "The uploaded file contained no extractable text.")
        st.stop()

    if summary_max <= summary_min:
        st.warning("Summary max length must be greater than min length.")
        st.stop()

    if device == "cuda" and not _cuda_available():
        st.warning("CUDA is not available on this host. Falling back to CPU.")
        device = "cpu"

    with st.spinner("Running models..."):
        result = analyze_text(
            cleaned,
            summary_model=summary_model,
            intent_model=intent_model,
            device=device,
            summary_min_length=summary_min,
            summary_max_length=summary_max,
            summary_refine_final=summary_refine_final,
            keyphrase_top_k=0,
            enable_intent=enable_intent,
            intent_labels=labels,
            intent_top_k=1,
            intent_mode="generate",
        )

    st.divider()

    tab_synopsis, tab_meta = st.tabs(["Synopsis", "Metadata"])

    with tab_synopsis:
        if result.summary or getattr(result, "summary_points", None):
            with st.container(border=True):
                st.subheader("Synopsis")
                if getattr(result, "summary_points", None):
                    st.markdown("\n".join([f"- {p}" for p in result.summary_points]))
                else:
                    st.markdown(f"**{result.summary}**")

                st.divider()
                st.subheader("Intent")
                if result.intent_top:
                    st.markdown(f"**{result.intent_top.label}**")
                    if result.intent_top.score is not None:
                        st.metric("Confidence", f"{result.intent_top.score:.3f}")
                else:
                    st.info("No intent prediction.")

                intent_meta = (result.meta or {}).get("intent") if isinstance(result.meta, dict) else None
                if isinstance(intent_meta, dict):
                    model = intent_meta.get("model")
                    mode = intent_meta.get("mode")
                    meta_bits = [f"mode: {mode}" if mode else None, f"model: {model}" if model else None]
                    meta_bits = [b for b in meta_bits if b]
                    if meta_bits:
                        st.caption(" · ".join(meta_bits))

            sum_meta = (result.meta or {}).get("summary") if isinstance(result.meta, dict) else None
            if isinstance(sum_meta, dict):
                model = sum_meta.get("model")
                meta_bits = [f"model: {model}" if model else None]
                meta_bits = [b for b in meta_bits if b]
                if meta_bits:
                    st.caption(" · ".join(meta_bits))
        else:
            st.info("No synopsis produced (input may be too short).")

    with tab_meta:
        st.json(result.meta)

else:
    st.info("Paste some text and click **Analyze**.")
