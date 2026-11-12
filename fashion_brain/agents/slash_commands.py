"""Chat slash triggers — thin wrappers. User text stays free-form.

Design triggers (`/garment_design`, `/advanced_garment_design`) mean: draw sample
images from a product specification. Spec may come from prior chat or from
whatever the user typed/pasted after the trigger (any layout with enough fields).
Never re-route a design trigger through catalog filters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SLASH_RE = re.compile(
    r"^\s*/(?P<cmd>"
    r"trending|"
    r"specification|spec|"
    r"garment_design|garment_desing|"
    r"advanced_garment_design|advance_garment_design|"
    r"advanced_garment_desing|advance_garment_desing"
    r")\b[ \t]*(?P<body>.*)$",
    re.IGNORECASE | re.DOTALL,
)

_CMD_ALIASES = {
    "spec": "specification",
    "garment_desing": "garment_design",
    "advance_garment_design": "advanced_garment_design",
    "advanced_garment_desing": "advanced_garment_design",
    "advance_garment_desing": "advanced_garment_design",
}

# Catalog-only frames (design commands do not use these — they draw images).
_INTENT_FRAME = {
    "trending": (
        "Analyse what is trending for this request. "
        "Use only entities and constraints the user stated; "
        "resolve buyer, brand, season, garment, category, and similar from their words. "
        "Give an attribute-wise mix with counts and a short takeaway."
    ),
    "specification": (
        "Make a product specification for this request. "
        "Use only entities and constraints the user stated."
    ),
}

TRENDING_CLARIFY = (
    "What should I look at for trends?\n\n"
    "Add free text after `/trending` — whatever you know."
)

SPECIFICATION_CLARIFY = (
    "I can build a product specification — add free text after `/specification` "
    "(who it’s for, season, garment…)."
)

GARMENT_DESIGN_CLARIFY = (
    "I need a product specification to draw from.\n\n"
    "Use a sheet already in this chat, or paste field lines after the command, "
    "or run `/specification` first."
)


@dataclass(frozen=True)
class SlashCommand:
    name: str
    body: str = ""


def parse_slash_command(text: str) -> SlashCommand | None:
    match = _SLASH_RE.match(text or "")
    if not match:
        return None
    raw = match.group("cmd").casefold()
    name = _CMD_ALIASES.get(raw, raw)
    # Keep the body as the user wrote it (line breaks matter for pasted fields).
    body = (match.group("body") or "").strip()
    return SlashCommand(name=name, body=body)


def slash_needs_clarify(command: SlashCommand, *, has_specification: bool) -> str | None:
    if command.name == "trending" and not command.body:
        return TRENDING_CLARIFY
    if command.name == "specification" and not command.body:
        return SPECIFICATION_CLARIFY
    if command.name in {"garment_design", "advanced_garment_design"}:
        # Body may be a sheet OR a short note on top of chat history.
        if has_specification or command.body:
            return None
        return GARMENT_DESIGN_CLARIFY
    return None


def slash_rewrite_question(command: SlashCommand) -> str | None:
    """Catalog rewrite for /trending and /specification only — never for design."""

    if command.name in {"garment_design", "advanced_garment_design"}:
        return None
    frame = _INTENT_FRAME.get(command.name)
    if not frame or not command.body.strip():
        return None
    return f"{frame}\n\nUser request: {command.body.strip()}"
