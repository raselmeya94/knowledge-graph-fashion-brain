"""Compat checks for answer_format shim (labels + thin polish)."""

from __future__ import annotations

from fashion_brain.agents.answer_format import (
    SOURCE_DISPLAY_LABELS,
    polish_final_answer,
    source_display_label,
    source_heading_label,
    title_label,
)


def test_source_label_mapping():
    assert SOURCE_DISPLAY_LABELS["plmex_products"] == "PLM records"
    assert SOURCE_DISPLAY_LABELS["ecom_products"] == "e-commerce listings"
    assert source_display_label("plmex_products") == "PLM records"
    assert source_heading_label("plmex_products") == "PLM"
    assert source_heading_label("ecom_products") == "E-commerce"


def test_title_label():
    assert title_label("primark") == "Primark"
    assert title_label("uk showroom sample") == "Uk Showroom Sample"


def test_polish_keeps_single_source_footer():
    raw = "Primark has 560 sweaters.\n\nSource: PLM records\n"
    out = polish_final_answer(raw, question="How many Primark sweater products?")
    assert "560" in out
    assert out.rstrip().endswith("Source: PLM records")


def test_polish_dedupes_duplicate_footers():
    raw = (
        "Primark has 560 sweaters.\n\n"
        "Source: E-commerce listings\n\n"
        "Source: PLM records\n"
    )
    out = polish_final_answer(raw)
    assert out.count("Source:") == 1
    assert "Source: PLM records" in out


def test_repair_drops_truncated_bare_chart_json():
    from fashion_brain.agents.answer_format import repair_chart_blocks

    raw = (
        "### Material & Construction\n"
        "Nylon is the most frequent material (45.2%).\n\n"
        '{"type": "bar", "title\n\n'
        "Source: E-commerce listings\n"
    )
    out = repair_chart_blocks(raw)
    assert '{"type"' not in out
    assert "Nylon is the most frequent" in out
    assert "Source: E-commerce listings" in out


def test_repair_wraps_complete_bare_chart_json():
    from fashion_brain.agents.answer_format import repair_chart_blocks

    raw = (
        "Brand mix:\n"
        '{"type": "bar", "title": "Brand", "labels": ["Zara"], "values": [10]}\n'
    )
    out = repair_chart_blocks(raw)
    assert "```chart" in out
    assert '"values": [10]' in out or '"values":[10]' in out.replace(" ", "")
    assert out.strip().endswith("```")


def test_repair_closes_or_drops_unclosed_chart_fence():
    from fashion_brain.agents.answer_format import repair_chart_blocks

    good_body = '{"type": "bar", "title": "Brand", "labels": ["A"], "values": [1]}'
    out = repair_chart_blocks(f"Intro\n```chart\n{good_body}\n")
    assert "```chart" in out
    assert out.rstrip().endswith("```")

    bad = repair_chart_blocks('Intro\n```chart\n{"type": "bar", "title\n')
    assert "```chart" not in bad
    assert '{"type"' not in bad


def test_chunk_preserving_charts_keeps_fence_atomic():
    from fashion_brain.agents.answer_format import chunk_preserving_charts

    chart = '```chart\n{"type": "bar", "title": "Brand", "labels": ["A"], "values": [1]}\n```\n'
    text = "Hello world. " + chart + "Done."
    parts = chunk_preserving_charts(text, size=8)
    joined = "".join(parts)
    assert joined == text
    assert any(p.startswith("```chart") and p.rstrip().endswith("```") for p in parts)
