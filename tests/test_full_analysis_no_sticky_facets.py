"""Full Primark analysis after final_confirm must not keep prior red-sweater filters."""

from __future__ import annotations

from fashion_brain.agents.contracts import Intention, ValidatedIntent
from fashion_brain.agents.query_understanding import apply_validated_intent


def test_apply_validated_full_analysis_replaces_sticky_facets():
    intention = Intention(
        intent="analysis",
        filters={"brand": "primark", "garment": "sweater", "color": "red"},
        group_by="category",
        analysis=True,
    )
    validated = ValidatedIntent(
        source="ecom_products",
        entity="primark",
        filters={"brand": "primark"},
        objective="full_analysis",
        analysis_depth="full",
        effective_question="full analysis of Primark",
    )
    out = apply_validated_intent(intention, validated)
    assert out.filters == {"brand": "primark"}
    assert out.analysis is True
    assert out.group_by is None


def test_apply_validated_focused_still_merges():
    intention = Intention(
        intent="breakdown",
        filters={"brand": "primark"},
    )
    validated = ValidatedIntent(
        filters={"color": "red"},
        analysis_depth="focused",
        group_by="color",
    )
    out = apply_validated_intent(intention, validated)
    assert out.filters.get("brand") == "primark"
    assert out.filters.get("color") == "red"
    assert out.group_by == "color"
