"""Source sniffing shared by the construction tools."""

from __future__ import annotations

import base64
import json
import re
from typing import Any

from knowledge_graph.errors import ValidationFailed

MAX_BYTES = 8 * 1024 * 1024

_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "doc",
    ".txt": "text",
    ".md": "text",
    ".eml": "email",
    ".html": "html",
    ".htm": "html",
    ".json": "json",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".gif": "image",
}

_MEDIA = {
    "application/pdf": "pdf",
    "application/json": "json",
    "text/html": "html",
    "message/rfc822": "email",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/msword": "doc",
}


def payload_bytes(payload: dict) -> bytes:
    data = payload.get("data")
    if isinstance(data, bytes):
        raw = data
    elif payload.get("data_base64"):
        try:
            raw = base64.b64decode(payload["data_base64"], validate=True)
        except Exception as exc:
            raise ValidationFailed("data_base64 is not valid base64.") from exc
    elif payload.get("text") is not None:
        raw = str(payload["text"]).encode("utf-8")
    elif payload.get("json_data") is not None:
        raw = json.dumps(payload["json_data"], ensure_ascii=False).encode("utf-8")
    else:
        raise ValidationFailed("Provide text, json_data, a file, or data_base64.")
    if len(raw) > MAX_BYTES:
        raise ValidationFailed("Source is larger than 8 MB.")
    return raw


def payload_text(payload: dict, data: bytes | None = None) -> str:
    if payload.get("text") is not None and not isinstance(payload.get("data"), bytes) and not payload.get("data_base64"):
        return str(payload["text"])
    raw = data if data is not None else payload_bytes(payload)
    return raw.decode("utf-8", errors="replace")


def detect_kind(payload: dict, data: bytes | None = None) -> str:
    explicit = str(payload.get("kind") or "").strip().lower()
    if explicit in {"pdf", "docx", "doc", "text", "email", "html", "json", "image"}:
        return explicit
    filename = str(payload.get("filename") or "").lower()
    for extension, kind in _EXTENSIONS.items():
        if filename.endswith(extension):
            return kind
    media = str(payload.get("media_type") or "").split(";", 1)[0].strip().lower()
    if media in _MEDIA:
        return _MEDIA[media]
    if media.startswith("image/"):
        return "image"
    if media.startswith("text/plain"):
        return "text"
    raw = data if data is not None else b""
    if not raw and (payload.get("text") is not None or payload.get("json_data") is not None or payload.get("data_base64")):
        try:
            raw = payload_bytes(payload)
        except ValidationFailed:
            raw = b""
    if raw.startswith(b"%PDF"):
        return "pdf"
    if raw.startswith(b"PK") and filename.endswith(".docx"):
        return "docx"
    if raw.startswith((b"\x89PNG", b"\xff\xd8\xff", b"GIF8")) or raw[:4] == b"RIFF":
        return "image"
    if payload.get("json_data") is not None:
        return "json"
    sample = raw[:4000].decode("utf-8", errors="replace").lstrip()
    if sample.startswith("{") or sample.startswith("["):
        return "json"
    if sample.lower().startswith("<!doctype html") or sample.lower().startswith("<html") or "<a " in sample.lower():
        return "html"
    head = "\n".join(sample.splitlines()[:30])
    if re.search(r"(?im)^from:\s+\S+", head) and re.search(r"(?im)^(to|subject):\s+\S+", head):
        return "email"
    return "text"


def decode_json(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        raise ValidationFailed("JSON source must be an object, a list, or a JSON string.")
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValidationFailed(f"JSON source could not be parsed: {exc.msg}.") from exc
