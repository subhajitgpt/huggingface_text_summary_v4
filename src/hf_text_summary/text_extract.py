from __future__ import annotations

from pathlib import Path
from typing import Final

import io
import zipfile
import xml.etree.ElementTree as ET


_SUPPORTED_EXTS: Final[set[str]] = {".txt", ".md", ".pdf", ".docx"}

# OCR fallback is optional and only used for PDFs that have no extractable text.
_OCR_MAX_PAGES: Final[int] = 10
_OCR_DPI: Final[int] = 200


def supported_extensions() -> set[str]:
    return set(_SUPPORTED_EXTS)


def extract_text_from_path(path: str | Path) -> str:
    p = Path(path)
    data = p.read_bytes()
    return extract_text_from_bytes(p.name, data)


def extract_text_from_bytes(filename: str, data: bytes) -> str:
    name = (filename or "").strip() or "<upload>"
    ext = Path(name).suffix.lower()

    kind = _sniff_kind(ext, data)

    if kind == "text":
        return _decode_text_bytes(data)

    if kind == "pdf":
        return _extract_pdf(data)

    if kind == "docx":
        return _extract_docx(data)

    if ext == ".doc":
        raise ValueError(
            "Unsupported Word .doc file. Please upload a .docx instead (or export to PDF)."
        )

    raise ValueError(
        f"Unsupported file type: {ext}. Supported: {', '.join(sorted(_SUPPORTED_EXTS))}"
    )


def _decode_text_bytes(data: bytes) -> str:
    if not data:
        return ""
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _extract_pdf(data: bytes) -> str:
    if not data:
        return ""

    # pypdf is lightweight and works well in server environments.
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as e:
        # If pypdf isn't installed, still allow OCR-only extraction when optional
        # OCR dependencies + the Tesseract binary are present.
        ocr = _extract_pdf_ocr(data)
        if ocr:
            return ocr

        raise ValueError(
            "Missing dependency 'pypdf' required to read PDFs. "
            "Install with: pip install -e .  (or: pip install pypdf). "
            "For scanned/image-only PDFs you can enable OCR: pip install -e '.[ocr]' "
            "and install the Tesseract OCR binary."
        ) from e

    reader = PdfReader(io.BytesIO(data))
    if getattr(reader, "is_encrypted", False):
        # Best-effort: try decrypt with empty password.
        try:
            reader.decrypt("")
        except Exception:
            raise ValueError("PDF is encrypted/password-protected and cannot be read.")
    parts: list[str] = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        txt = txt.strip()
        if txt:
            parts.append(txt)

    joined = "\n\n".join(parts).strip()
    if joined:
        return joined

    # Fallback: OCR for scanned/image-only PDFs (best-effort, optional deps).
    ocr = _extract_pdf_ocr(data)
    return ocr or ""


def _extract_pdf_ocr(data: bytes) -> str:
    if not data:
        return ""

    # Optional dependencies: pytesseract + tesseract binary + PyMuPDF + Pillow.
    try:
        import pytesseract  # type: ignore

        # Will raise if the tesseract executable isn't available.
        pytesseract.get_tesseract_version()

        import fitz  # PyMuPDF  # type: ignore
        from PIL import Image  # type: ignore
    except Exception:
        return ""

    parts: list[str] = []
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            page_count = len(doc)
            limit = min(page_count, max(1, int(_OCR_MAX_PAGES)))

            mode_map = {1: "L", 3: "RGB", 4: "RGBA"}

            for i in range(limit):
                page = doc.load_page(i)
                pix = page.get_pixmap(dpi=int(_OCR_DPI))

                mode = mode_map.get(getattr(pix, "n", 3), "RGB")
                img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)

                txt = pytesseract.image_to_string(img) or ""
                txt = txt.strip()
                if txt:
                    parts.append(txt)
    except Exception:
        return ""

    joined = "\n\n".join(parts).strip()
    if joined:
        return joined

    if missing_python_docx:
        raise ValueError(
            "Missing dependency 'python-docx' required to reliably read DOCX files. "
            "Install with: pip install -e .  (or: pip install python-docx)"
        )

    return ""


def _extract_docx(data: bytes) -> str:
    if not data:
        return ""

    # Prefer python-docx when available (better fidelity for some docs),
    # but keep a stdlib fallback for environments where it isn't installed.
    missing_python_docx = False
    try:
        import docx  # type: ignore

        doc = docx.Document(io.BytesIO(data))
        parts: list[str] = []

        for p in getattr(doc, "paragraphs", []) or []:
            t = (getattr(p, "text", "") or "").strip()
            if t:
                parts.append(t)

        for table in getattr(doc, "tables", []) or []:
            for row in getattr(table, "rows", []) or []:
                cells = [(getattr(c, "text", "") or "").strip() for c in getattr(row, "cells", [])]
                line = "\t".join([c for c in cells if c])
                if line:
                    parts.append(line)

        joined = "\n".join(parts).strip()
        if joined:
            return joined
    except ModuleNotFoundError:
        missing_python_docx = True
    except Exception:
        # Any parsing errors fall through to the stdlib XML fallback.
        pass

    # DOCX is a ZIP file containing WordprocessingML XML.
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise ValueError("Invalid DOCX (not a zip archive).") from e

    # Main document text.
    xml_names = [
        "word/document.xml",
        "word/header1.xml",
        "word/header2.xml",
        "word/footer1.xml",
        "word/footer2.xml",
    ]

    parts: list[str] = []
    for xml_name in xml_names:
        if xml_name not in zf.namelist():
            continue
        try:
            xml_bytes = zf.read(xml_name)
        except Exception:
            continue

        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError:
            continue

        # WordprocessingML uses namespaces; match by localname.
        texts: list[str] = []
        for el in root.iter():
            if el.tag.endswith("}t") and el.text:
                texts.append(el.text)
            # Paragraph breaks.
            if el.tag.endswith("}p"):
                if texts and (texts[-1] != "\n"):
                    texts.append("\n")
        blob = "".join(texts)
        blob = "\n".join([ln.rstrip() for ln in blob.splitlines()])
        blob = blob.strip()
        if blob:
            parts.append(blob)

    return "\n\n".join(parts).strip()




def _sniff_kind(ext: str, data: bytes) -> str:
    """Infer file kind from extension + content.

    Returns: 'text' | 'pdf' | 'docx' | 'unknown'
    """

    ext = (ext or "").lower()
    head = data[:8] if data else b""

    if head.startswith(b"%PDF-"):
        return "pdf"

    # DOCX is a ZIP; attempt to verify it contains WordprocessingML.
    if head.startswith(b"PK"):
        try:
            zf = zipfile.ZipFile(io.BytesIO(data))
            names = set(zf.namelist())
            if "word/document.xml" in names:
                return "docx"
        except Exception:
            pass

    if ext in {".pdf"}:
        return "pdf"
    if ext in {".docx"}:
        return "docx"
    if ext in {".txt", ".md"} or not ext:
        return "text"

    # Some platforms mis-label docx as .doc; try content sniffing above.
    return "unknown"
