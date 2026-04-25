FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf \
    TRANSFORMERS_CACHE=/app/.hf \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

# System deps
# - git: allows installing from VCS dependencies if ever added
# - build-essential/cmake/pkg-config: helps when wheels aren't available (arm, slim)
# - libxml2/libxslt: lxml fallback builds (python-docx dependency chain)
RUN apt-get update \
    ; apt-get install -y --no-install-recommends \
        git \
        build-essential \
        cmake \
        pkg-config \
        libxml2-dev \
        libxslt1-dev \
        zlib1g-dev \
        tesseract-ocr \
        tesseract-ocr-eng \
    ; rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY app.py /app/app.py

RUN pip install --upgrade pip setuptools wheel \
    ; pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu "torch>=2.1" \
    ; pip install --no-cache-dir ".[ocr]" \
    ; python -c "import pypdf, docx; print('deps-ok')"

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app.py"]
