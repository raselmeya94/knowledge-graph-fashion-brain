"""Paragraph-wise splitting for scraping construction."""

from __future__ import annotations

import hashlib
import re


def split_paragraphs(text: str) -> list[str]:
    """Split on blank-line boundaries; keep non-empty paragraphs in order."""

    cleaned = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        return []
    parts = re.split(r"\n\s*\n+", cleaned)
    paragraphs: list[str] = []
    for part in parts:
        block = re.sub(r"[ \t]+\n", "\n", part).strip()
        if block:
            paragraphs.append(block)
    if not paragraphs and cleaned:
        paragraphs = [cleaned]
    return paragraphs


def chunk_id_for(file_id: str, index: int, text: str) -> str:
    digest = hashlib.sha256(f"{file_id}:{index}:{text[:200]}".encode("utf-8")).hexdigest()[:16]
    return f"chk_{index:04d}_{digest}"


def build_chunks(file_id: str, text: str) -> list[dict]:
    chunks: list[dict] = []
    for index, paragraph in enumerate(split_paragraphs(text)):
        chunks.append(
            {
                "chunk_id": chunk_id_for(file_id, index, paragraph),
                "index": index,
                "text": paragraph,
                "char_count": len(paragraph),
            }
        )
    return chunks
