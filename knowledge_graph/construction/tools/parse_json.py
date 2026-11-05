"""Structured JSON parsing for relationship identification."""

from __future__ import annotations

import json

from knowledge_graph.construction.detect import decode_json, payload_text
from knowledge_graph.errors import ValidationFailed


class ParseJsonTool:
    name = "parse_json"
    description = "Parse a JSON object or list into structured records and a text view for relationship identification."

    def run(self, payload: dict, ctx) -> dict:
        del ctx
        raw = payload.get("json_data")
        if raw is None:
            raw = payload_text(payload)
        try:
            data = decode_json(raw)
        except ValidationFailed:
            raise
        except Exception as exc:
            raise ValidationFailed("JSON source could not be parsed.") from exc
        if isinstance(data, list):
            data = {"items": data}
        if not isinstance(data, dict):
            raise ValidationFailed("JSON source must be an object or a list.")
        return {
            "kind": "json",
            "title": str(data.get("title") or data.get("name") or data.get("subject") or ""),
            "text": _as_text(data),
            "structured": data,
            "warnings": [],
        }


def _as_text(data: dict) -> str:
    lines: list[str] = []

    def walk(prefix: str, value) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                walk(f"{prefix}.{key}" if prefix else str(key), item)
            return
        if isinstance(value, list):
            if value and all(not isinstance(item, (dict, list)) for item in value):
                lines.append(f"{prefix}: {', '.join(str(item) for item in value)}")
            else:
                for index, item in enumerate(value):
                    walk(f"{prefix}[{index}]", item)
            return
        if value is None or value == "":
            return
        lines.append(f"{prefix}: {value}")

    walk("", data)
    rendered = "\n".join(lines).strip()
    return rendered or json.dumps(data, ensure_ascii=False)
