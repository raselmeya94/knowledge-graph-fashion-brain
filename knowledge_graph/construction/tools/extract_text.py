"""Plain-text, PDF, Word, and email extraction."""

from __future__ import annotations

import re
import zipfile
from io import BytesIO
from xml.etree import ElementTree

from knowledge_graph.construction.detect import detect_kind, payload_bytes, payload_text
from knowledge_graph.errors import ModelUnavailableError, ParserNotAvailable, ValidationFailed

_WORD_TEXT = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
_SUBJECT = re.compile(r"(?im)^subject:\s*(.+?)\s*$")


class ExtractTextTool:
    name = "extract_text"
    description = "Extract plain text from PDF, Word, email, or a text file. Optional Ollama cleanup."

    def run(self, payload: dict, ctx) -> dict:
        kind = detect_kind(payload)
        if kind in {"html", "json", "image"}:
            raise ValidationFailed(f"extract_text does not handle {kind}. Use the matching tool.")
        data = payload_bytes(payload)
        warnings: list[str] = []
        title = ""
        if kind == "pdf":
            text = _pdf_text(data)
        elif kind == "docx":
            text = _docx_text(data)
        elif kind == "doc":
            text, warnings = _doc_text(data)
        else:
            text = payload_text(payload, data)
            if kind == "email":
                match = _SUBJECT.search(text)
                title = match.group(1).strip() if match else ""
        text = text.replace("\x00", " ").strip()
        if not text:
            raise ParserNotAvailable("No text could be extracted from this source.")
        if payload.get("clean"):
            try:
                text = ctx.ollama.complete(
                    "Return only the cleaned plain text. Keep names, dates, and facts. Do not add commentary.\n\n" + text[:12000],
                    system="You clean extracted document text.",
                ).strip()
            except ModelUnavailableError as exc:
                warnings.append(exc.message)
        return {"kind": kind if kind != "text" else detect_kind(payload), "text": text, "title": title, "warnings": warnings}


def _pdf_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ParserNotAvailable("Install pypdf to extract PDF text.") from exc
    try:
        reader = PdfReader(BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        raise ParserNotAvailable(f"PDF text could not be extracted: {exc}") from exc
    return "\n".join(pages).strip()


def _docx_text(data: bytes) -> str:
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            xml = archive.read("word/document.xml")
    except (KeyError, zipfile.BadZipFile) as exc:
        raise ParserNotAvailable("DOCX file does not contain word/document.xml.") from exc
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as exc:
        raise ParserNotAvailable("DOCX document.xml is not valid XML.") from exc
    parts = [node.text for node in root.iter(_WORD_TEXT) if node.text]
    return "\n".join(parts).strip()


def _doc_text(data: bytes) -> tuple[str, list[str]]:
    """Recover readable runs from a legacy .doc file. Formatting is discarded."""

    decoded = data.decode("utf-16le", errors="ignore")
    pieces = re.findall(r"[\w][\w .,;:'\"&()/\-]{8,}", decoded)
    text = "\n".join(piece.strip() for piece in pieces if piece.strip())
    if len(text) < 12:
        ascii_runs = re.findall(rb"[\x20-\x7e]{12,}", data)
        text = "\n".join(run.decode("ascii") for run in ascii_runs)
    text = text.strip()
    if len(text) < 12:
        raise ParserNotAvailable("Legacy .doc text could not be recovered. Save the file as .docx.")
    return text, ["Legacy .doc text was recovered without formatting. Prefer .docx."]
