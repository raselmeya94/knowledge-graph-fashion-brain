"""Compatibility shim — prefer ``answer_format`` + ``answer_card``.

Kept so existing imports of ``fashion_brain.agents.answer_formation`` continue to work
while call sites migrate to the card pipeline.
"""

from fashion_brain.agents.answer_format import (  # noqa: F401
    ANSWER_FEW_SHOTS,
    ANSWER_WRITING_RULES,
    SOURCE_DISPLAY_LABELS,
    SOURCE_HEADING_LABELS,
    SOURCE_LABELS,
    WRITER_PROMPT,
    build_answer_card,
    format_count,
    format_share,
    polish_final_answer,
    source_display_label,
    source_heading_label,
    title_label,
    write_answer,
    _title_label,
)

__all__ = [
    "ANSWER_FEW_SHOTS",
    "ANSWER_WRITING_RULES",
    "SOURCE_DISPLAY_LABELS",
    "SOURCE_HEADING_LABELS",
    "SOURCE_LABELS",
    "WRITER_PROMPT",
    "build_answer_card",
    "format_count",
    "format_share",
    "polish_final_answer",
    "source_display_label",
    "source_heading_label",
    "title_label",
    "write_answer",
    "_title_label",
]
