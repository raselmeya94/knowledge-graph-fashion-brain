"""Answer formatting helpers kept beside answer_card.py.

Card pipeline lives in ``answer_card`` (build → write → validate → footer).
This module keeps shared labels, the writer prompt bundle, and light display helpers.
``polish_final_answer`` remains only as a thin migration fallback.
"""

from __future__ import annotations

import re

from fashion_brain.agents.answer_card import (
    INTENT_HINTS,
    SOURCE_LABELS,
    WRITER_PROMPT,
    build_answer_card,
    intent_hint,
    pluralize_garment,
    write_answer,
)

# Back-compat alias used by merger / analytics heading helpers.
SOURCE_DISPLAY_LABELS: dict[str, str] = {
    **SOURCE_LABELS,
    "e-commerce": SOURCE_LABELS["ecom_products"],
    "ecommerce": SOURCE_LABELS["ecom_products"],
    "plm": SOURCE_LABELS["plmex_products"],
    "plmex": SOURCE_LABELS["plmex_products"],
    "own products": SOURCE_LABELS["own_products"],
    "plm records": SOURCE_LABELS["plmex_products"],
    "plmex records": SOURCE_LABELS["plmex_products"],
}

SOURCE_HEADING_LABELS: dict[str, str] = {
    "ecom_products": "E-commerce",
    "plmex_products": "PLM",
    "own_products": "Own products",
}

# Writer prompt for RESPONSE_SYSTEM / write_answer (guide §1 + intent hints).
ANSWER_WRITING_RULES = WRITER_PROMPT
# Sample answers removed — they become templates. Card + validator own quality.
ANSWER_FEW_SHOTS = ""


def source_display_label(source_id: str | None) -> str:
    raw = str(source_id or "").strip()
    if not raw:
        return "our product data"
    key = raw.casefold().replace("_", " ").strip()
    compact = key.replace(" ", "")
    for candidate in (raw.casefold(), key, compact):
        if candidate in SOURCE_DISPLAY_LABELS:
            return SOURCE_DISPLAY_LABELS[candidate]
    cleaned = re.sub(r"(?i)\bknowledge\s*graphs?\b", "", raw)
    cleaned = re.sub(r"(?i)\b(?:neo4j|postgres|kg)\b", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" -–—|/\\`()")
    key2 = cleaned.casefold().replace("_", " ")
    if key2 in SOURCE_DISPLAY_LABELS:
        return SOURCE_DISPLAY_LABELS[key2]
    if "plmex" in key2 or key2 == "plm" or key2.endswith(" plm"):
        return SOURCE_LABELS["plmex_products"]
    if "ecom" in key2 or "commerce" in key2:
        return SOURCE_LABELS["ecom_products"]
    return cleaned or "our product data"


def source_heading_label(source_id: str | None) -> str:
    raw = str(source_id or "").strip()
    if raw in SOURCE_HEADING_LABELS:
        return SOURCE_HEADING_LABELS[raw]
    key = raw.casefold()
    for sid, label in SOURCE_HEADING_LABELS.items():
        if sid.casefold() == key:
            return label
    display = source_display_label(raw)
    if display == SOURCE_LABELS["ecom_products"]:
        return "E-commerce"
    if display == SOURCE_LABELS["plmex_products"]:
        return "PLM"
    if display == SOURCE_LABELS["own_products"]:
        return "Own products"
    return display[:1].upper() + display[1:] if display else "Catalog"


def title_label(value: str) -> str:
    """Title-case a facet/brand label for display."""

    raw = str(value or "").strip()
    if not raw:
        return raw
    if raw.isupper() or raw.islower() or re.fullmatch(r"[a-z0-9 _/-]+", raw):
        small = {"and", "or", "of", "the", "for", "in", "on", "a", "an"}
        words = []
        for i, word in enumerate(raw.replace("_", " ").split()):
            if i > 0 and word.casefold() in small:
                words.append(word.casefold())
            else:
                words.append(word[:1].upper() + word[1:] if word else word)
        return " ".join(words)
    return raw


# Public alias matching prior private helper name.
_title_label = title_label


def format_count(n: int | float | str) -> str:
    """Thousands-separated integer display."""

    try:
        return f"{int(float(str(n).replace(',', ''))):,}"
    except (TypeError, ValueError):
        return str(n)


def format_share(share: float | int | str | None) -> str:
    """Whole-number percentage when possible."""

    if share is None or share == "":
        return ""
    try:
        value = float(str(share).replace("%", "").strip())
    except ValueError:
        return str(share)
    if abs(value - round(value)) < 0.05:
        return f"{int(round(value))}%"
    return f"{value:.1f}%"


_CHART_OPEN_RE = re.compile(r"^```(?:chart|fbchart)\s*$", re.IGNORECASE)
_BARE_CHART_START_RE = re.compile(
    r'\{\s*"type"\s*:\s*"(?:bar|pie|line)"',
    re.IGNORECASE,
)


def repair_chart_blocks(text: str) -> str:
    """Close or drop truncated chart fences / bare chart JSON so the UI never hangs.

    Writers sometimes hit max-tokens mid-JSON (``{"type": "bar", "title``) or omit the
    closing ``` fence. Prefer dropping a broken chart over leaving raw JSON in the reply.
    """

    if not text:
        return text
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if _CHART_OPEN_RE.match(stripped):
            block = [raw]
            i += 1
            closed = False
            body_parts: list[str] = []
            while i < len(lines):
                ln = lines[i]
                block.append(ln)
                if ln.strip() == "```":
                    closed = True
                    i += 1
                    break
                body_parts.append(ln)
                i += 1
            body = "".join(body_parts).strip()
            if closed:
                try:
                    import json

                    payload = json.loads(body)
                    if isinstance(payload, dict) and payload.get("type"):
                        out.extend(block)
                        continue
                except Exception:
                    pass
                # Closed but invalid JSON — drop the broken viz.
                continue
            # Unclosed fence: try to salvage complete JSON, else drop.
            try:
                import json

                payload = json.loads(body)
                if isinstance(payload, dict) and payload.get("type"):
                    out.append("```chart\n")
                    out.append(json.dumps(payload, ensure_ascii=False) + "\n")
                    out.append("```\n")
            except Exception:
                pass
            continue
        # Bare truncated / complete chart JSON outside a fence.
        if _BARE_CHART_START_RE.search(stripped):
            blob = stripped
            j = i + 1
            # Pull continuation lines until blank, fence, heading, or Source footer.
            while j < len(lines):
                nxt = lines[j].strip()
                if not nxt or nxt.startswith("```") or nxt.startswith("#") or re.match(
                    r"(?i)^sources?\s*:", nxt
                ):
                    break
                if nxt.startswith("{") and _BARE_CHART_START_RE.search(nxt):
                    break
                blob += lines[j].rstrip("\n")
                j += 1
            try:
                import json

                # Isolate the outermost JSON object starting at first "{".
                start = blob.find("{")
                candidate = blob[start:] if start >= 0 else blob
                payload = json.loads(candidate)
                if isinstance(payload, dict) and str(payload.get("type", "")).casefold() in {
                    "bar",
                    "pie",
                    "line",
                }:
                    out.append("```chart\n")
                    out.append(json.dumps(payload, ensure_ascii=False) + "\n")
                    out.append("```\n")
                    i = j
                    continue
            except Exception:
                # Incomplete JSON (e.g. {"type": "bar", "title) — drop it.
                i = j
                continue
        out.append(raw)
        i += 1
    cleaned = "".join(out)
    # Safety: odd leftover chart openers without a following close.
    opens = len(re.findall(r"(?im)^```(?:chart|fbchart)\s*$", cleaned))
    closes = len(re.findall(r"(?im)^```\s*$", cleaned))
    # closes also counts chart openers that are ```chart — recount carefully:
    fence_lines = re.findall(r"(?im)^```[^\n]*$", cleaned)
    depth = 0
    for fl in fence_lines:
        if _CHART_OPEN_RE.match(fl.strip()) or fl.strip() == "```":
            if fl.strip() == "```" and depth > 0:
                depth -= 1
            elif _CHART_OPEN_RE.match(fl.strip()):
                depth += 1
    if depth > 0:
        cleaned = cleaned.rstrip() + "\n```\n"
    return cleaned


def chunk_preserving_charts(text: str, size: int = 64) -> list[str]:
    """Split prose for streaming, but keep each ```chart … ``` block atomic."""

    if not text:
        return []
    if size <= 0:
        return [text]
    parts: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        open_at = text.find("```chart", i)
        open_fb = text.find("```fbchart", i)
        candidates = [p for p in (open_at, open_fb) if p >= 0]
        next_open = min(candidates) if candidates else -1
        if next_open < 0:
            chunk = text[i:]
            for j in range(0, len(chunk), size):
                parts.append(chunk[j : j + size])
            break
        if next_open > i:
            chunk = text[i:next_open]
            for j in range(0, len(chunk), size):
                parts.append(chunk[j : j + size])
        # Find closing fence after the opener line.
        after = text.find("\n", next_open)
        close = text.find("\n```", after if after >= 0 else next_open)
        if close < 0:
            # Unclosed — emit remainder as one piece (caller should have repaired).
            parts.append(text[next_open:])
            break
        end = close + len("\n```")
        # Include trailing newline if present.
        if end < n and text[end] == "\n":
            end += 1
        parts.append(text[next_open:end])
        i = end
    return parts


def polish_final_answer(text: str, *, question: str = "", findings: str = "") -> str:
    """Thin migration fallback only — prefer build_answer_card + write_answer.

    Keeps a single card-style Source/Sources footer (does not infer sources from
    findings text). Does not reshape count/breakdown prose.
    """

    del findings  # never used for source inference — footers come from the card
    del question
    out = (text or "").strip()
    if not out:
        return out
    out = repair_chart_blocks(out)
    # Keep one trailing Source/Sources line; drop earlier duplicates only.
    footers = re.findall(r"(?im)^sources?\s*:\s*.+$", out)
    out = re.sub(r"(?im)^sources?\s*:\s*.+$", "", out)
    out = re.sub(
        r"(?im)^(?:page\s+\d+(?:\s+of\s+\d+)?|confidential|\[?\s*chart\s+placeholder\s*\]?)\s*$",
        "",
        out,
    )
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    if footers:
        out = f"{out}\n\n{footers[-1].strip()}"
    return out


__all__ = [
    "ANSWER_FEW_SHOTS",
    "ANSWER_WRITING_RULES",
    "INTENT_HINTS",
    "SOURCE_DISPLAY_LABELS",
    "SOURCE_HEADING_LABELS",
    "SOURCE_LABELS",
    "WRITER_PROMPT",
    "build_answer_card",
    "chunk_preserving_charts",
    "format_count",
    "format_share",
    "intent_hint",
    "pluralize_garment",
    "polish_final_answer",
    "repair_chart_blocks",
    "source_display_label",
    "source_heading_label",
    "title_label",
    "write_answer",
    "_title_label",
]
