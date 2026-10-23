"""
Document Ingestion and Text Chunking.
Handles PDF, TXT, and Markdown files with page-level metadata retention.
"""

import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import pymupdf as fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    try:
        import fitz
        PYMUPDF_AVAILABLE = True
    except ImportError:
        PYMUPDF_AVAILABLE = False

try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False


def compute_file_hash(content_or_path) -> str:
    """Calculates SHA-256 hash of file content or path."""
    hasher = hashlib.sha256()
    if isinstance(content_or_path, (str, Path)) and os.path.exists(content_or_path):
        with open(content_or_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
    elif isinstance(content_or_path, bytes):
        hasher.update(content_or_path)
    elif isinstance(content_or_path, str):
        hasher.update(content_or_path.encode("utf-8"))
    return hasher.hexdigest()[:16]


def extract_text_from_pdf(file_bytes_or_path) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from PDF.
    Returns list of {'page': int, 'text': str}.
    """
    pages_data = []

    # Try PyMuPDF first
    if PYMUPDF_AVAILABLE:
        try:
            if isinstance(file_bytes_or_path, (bytes, bytearray)):
                doc = fitz.open(stream=file_bytes_or_path, filetype="pdf")
            else:
                doc = fitz.open(file_bytes_or_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                txt = page.get_text() or ""
                txt = txt.strip()
                if txt:
                    pages_data.append({"page": page_num + 1, "text": txt})
            doc.close()
            if pages_data:
                return pages_data
        except Exception:
            pass

    # Fallback to PyPDF
    if PYPDF_AVAILABLE:
        try:
            import io
            if isinstance(file_bytes_or_path, (bytes, bytearray)):
                reader = pypdf.PdfReader(io.BytesIO(file_bytes_or_path))
            else:
                reader = pypdf.PdfReader(file_bytes_or_path)
            for idx, page in enumerate(reader.pages):
                txt = page.extract_text() or ""
                txt = txt.strip()
                if txt:
                    pages_data.append({"page": idx + 1, "text": txt})
            if pages_data:
                return pages_data
        except Exception:
            pass

    return pages_data


def extract_text_from_txt(content_str: str) -> List[Dict[str, Any]]:
    """Extracts text from plaintext or markdown string, estimating page breaks every ~3000 chars."""
    clean = content_str.strip()
    if not clean:
        return []
    # If text contains form feeds or explicit page markers:
    if "\f" in clean:
        raw_pages = clean.split("\f")
        return [{"page": i + 1, "text": p.strip()} for i, p in enumerate(raw_pages) if p.strip()]
    elif "--- PAGE " in clean or "=== PAGE " in clean:
        parts = re.split(r"(?:---|\=\=\=)\s*PAGE\s*(\d+)\s*(?:---|\=\=\=)", clean)
        pages = []
        if len(parts) > 1:
            for i in range(1, len(parts), 2):
                p_num = int(parts[i])
                p_text = parts[i+1].strip()
                if p_text:
                    pages.append({"page": p_num, "text": p_text})
            return pages

    # Otherwise segment into logical pages of ~2500 characters
    chunk_size = 2500
    pages = []
    lines = clean.split("\n")
    current_page = []
    current_len = 0
    page_num = 1

    for line in lines:
        current_page.append(line)
        current_len += len(line) + 1
        if current_len >= chunk_size:
            pages.append({"page": page_num, "text": "\n".join(current_page).strip()})
            current_page = []
            current_len = 0
            page_num += 1

    if current_page:
        pages.append({"page": page_num, "text": "\n".join(current_page).strip()})

    return pages


def chunk_text(
    pages: List[Dict[str, Any]],
    doc_id: str,
    doc_name: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200
) -> List[Dict[str, Any]]:
    """
    Splits page text into coherent searchable passages with metadata.
    Preserves paragraphs and enumerations intact.
    """
    passages: List[Dict[str, Any]] = []
    chunk_counter = 0

    for page_entry in pages:
        page_num = page_entry["page"]
        text = page_entry["text"]
        if not text:
            continue

        cleaned = text.strip()
        # Split by paragraphs (double newlines or list headers)
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
        if not paragraphs:
            paragraphs = [cleaned]

        current_chunk_paras = []
        current_len = 0

        for para in paragraphs:
            p_len = len(para)
            if current_len + p_len > chunk_size and current_chunk_paras:
                chunk_str = "\n\n".join(current_chunk_paras).strip()
                if len(chunk_str) > 30:
                    chunk_counter += 1
                    passages.append({
                        "id": f"{doc_id}_p{page_num}_c{chunk_counter}",
                        "doc_id": doc_id,
                        "doc_name": doc_name,
                        "page": page_num,
                        "chunk_idx": chunk_counter,
                        "text": chunk_str,
                        "char_len": len(chunk_str)
                    })

                # Overlap: keep the last paragraph if small enough
                if len(current_chunk_paras[-1]) <= chunk_overlap:
                    current_chunk_paras = [current_chunk_paras[-1]]
                    current_len = len(current_chunk_paras[0])
                else:
                    current_chunk_paras = []
                    current_len = 0

            current_chunk_paras.append(para)
            current_len += p_len + 2

        if current_chunk_paras:
            chunk_str = "\n\n".join(current_chunk_paras).strip()
            if len(chunk_str) > 30:
                chunk_counter += 1
                passages.append({
                    "id": f"{doc_id}_p{page_num}_c{chunk_counter}",
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "page": page_num,
                    "chunk_idx": chunk_counter,
                    "text": chunk_str,
                    "char_len": len(chunk_str)
                })

    return passages

