# Hugging Face Text Summary

A small, deployable Streamlit app that:
- Summarizes text using a Hugging Face Transformers model
- Produces a short list of key phrases/topics
- Infers intent by generating a short label from the text

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip

# Recommended (installs the package from src/)
pip install -e .

# Alternative (if you run the app without installing the package)
# pip install -r requirements.txt

# Optional: enable OCR fallback for scanned/image-only PDFs
pip install -e ".[ocr]"

# Optional dev tooling
pip install -r requirements-dev.txt

streamlit run app.py
```

First run will download the models into your Hugging Face cache.

## CLI usage

```powershell
# After `pip install -e .`
hf-text-summary -f input.txt

# Also supports PDF and DOCX
hf-text-summary -f report.pdf
hf-text-summary -f notes.docx

# from stdin
Get-Content input.txt | hf-text-summary

# (legacy wrapper still works)
python summarize_cli.py -f input.txt
```

## Docker

```powershell
docker build -t hf-text-summary .
docker run --rm -p 8501:8501 hf-text-summary
```

Then open http://localhost:8501

## Notes

- Default summarization model: `sshleifer/distilbart-cnn-12-6` (CPU-friendly)
- Default intent model: `google/flan-t5-small`
- For long text, the summarizer uses a chunked (map-reduce) strategy.

File parsing:
- Supported input extensions: TXT/MD, PDF, DOCX
- Scanned/image-only PDFs may have no extractable text without OCR. If you install the optional `ocr` extra and have the Tesseract binary installed, the app/CLI will attempt OCR as a fallback when normal PDF text extraction yields nothing.
- Legacy Word `.doc` is not supported; export to `.docx` or PDF.

OCR prerequisites:
- Install Tesseract OCR on your system (the `pytesseract` Python package is only a wrapper).
- Windows: install Tesseract and ensure `tesseract.exe` is on `PATH`.
- Linux (Debian/Ubuntu): `apt-get install tesseract-ocr`.

UI theming is configured via `.streamlit/config.toml`.

If deployment environments are resource-constrained, consider switching to smaller models in the sidebar.

## Python install note (Windows)

If your terminal says "Python was not found", install Python 3.10+ from https://www.python.org/downloads/ and ensure it's on PATH.
