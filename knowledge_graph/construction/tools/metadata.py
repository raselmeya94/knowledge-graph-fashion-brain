"""Metadata collected before relationship identification."""

from __future__ import annotations

import re

_EMAIL = re.compile(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", re.IGNORECASE)
_URL = re.compile(r"https?://[^\s<>'\"]+")
_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
_DATE = re.compile(
    rf"\b\d{{4}}-\d{{2}}-\d{{2}}\b"
    rf"|\b\d{{1,2}}/\d{{1,2}}/\d{{2,4}}\b"
    rf"|\b{_MONTH}\s+\d{{1,2}},\s+\d{{4}}\b"
    rf"|\b{_MONTH}\s+\d{{1,2}}\b"
    rf"|\b\d{{1,2}}\s+{_MONTH}\s+\d{{4}}\b"
    rf"|\b{_MONTH}\s*[-–]\s*{_MONTH}\s+\d{{4}}\b",
    re.IGNORECASE,
)


class ExtractMetadataTool:
    name = "extract_metadata"
    description = "Collect title, source kind, size, addresses, links, and dates from extracted text."

    def run(self, payload: dict, ctx) -> dict:
        del ctx
        text = str(payload.get("text") or "")
        title = str(payload.get("title") or "").strip()
        if not title:
            for line in text.splitlines():
                if line.strip():
                    title = line.strip()[:180]
                    break
        if not title:
            title = str(payload.get("filename") or "Untitled document")
        links = payload.get("links") or []
        flat = re.sub(r"\s+", " ", text)
        urls = _URL.findall(flat)
        for link in links:
            href = str(link.get("href") or "")
            if href.startswith("http") and href not in urls:
                urls.append(href)
        return {
            "source_kind": payload.get("kind") or "text",
            "title": title,
            "filename": str(payload.get("filename") or ""),
            "document_type": payload.get("document_type") or payload.get("kind") or "text",
            "char_count": len(text),
            "emails": _unique(_EMAIL.findall(flat)),
            "urls": _unique(urls),
            "dates": _unique(_DATE.findall(flat)),
            "links": links,
        }


def _unique(values: list[str]) -> list[str]:
    seen: list[str] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen[:50]
