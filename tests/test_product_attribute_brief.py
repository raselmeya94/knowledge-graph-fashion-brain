"""C1-style product attribute briefs keep brand+garment and never group_by=material."""

from fashion_brain.agents._impl import _deterministic_intent, _apply_session_scope
from fashion_brain.agents.turn_planner import (
    _plan_product_attribute_brief,
    plan_turn,
)
from fashion_brain.ask import (
    _is_analysis,
    analysis_fields_for_product_brief,
    enrich_scope_from_question,
    is_product_attribute_brief_ask,
    named_product_brief_facets,
    resolve_catalog_plan,
)
from knowledge_graph.retrieval.ecom import is_analysis_facet_attr

C1 = (
    "I need a full Primark sweater brief — composition, construction, "
    "fit, care, colours, and price."
)


def test_c1_is_attribute_brief_not_material_group_by():
    assert is_product_attribute_brief_ask(C1)
    assert _is_analysis(C1)
    facets = named_product_brief_facets(C1)
    assert "material" in facets
    assert "construction" in facets
    assert "fit" in facets
    assert "care_instruction" in facets
    assert "color" in facets
    assert "price" in facets

    plan = resolve_catalog_plan(C1, None, carry=False)
    assert plan.get("analysis") is True
    assert plan.get("group_by") in (None, "")
    filters = plan.get("filters") or {}
    assert str(filters.get("brand") or "").casefold() == "primark"
    assert "sweater" in str(
        filters.get("garment") or filters.get("category") or ""
    ).casefold()

    scoped, group_by = enrich_scope_from_question(C1, dict(filters))
    assert group_by is None
    assert "sweater" in str(
        scoped.get("garment") or scoped.get("category") or ""
    ).casefold()


def test_session_scope_does_not_rewrite_c1_to_material_breakdown():
    summary = _deterministic_intent(C1, None)
    assert summary.group_by is None
    assert "material" not in (summary.actual_intent or "").casefold()
    assert "sweater" in str((summary.filters or {}).get("garment") or "").casefold()

    summary = _apply_session_scope(summary, {}, C1)
    assert summary.group_by is None
    assert "break down" not in (summary.actual_intent or "").casefold()
    assert "sweater" in str((summary.filters or {}).get("garment") or "").casefold()


def test_turn_planner_short_circuits_c1_attribute_brief():
    turn = _plan_product_attribute_brief(C1, {})
    assert turn.group_by is None
    assert turn.metric == "analysis"
    assert turn.analysis_depth == "full"
    assert str(turn.filters.get("brand") or "").casefold() == "primark"
    assert "sweater" in str(
        turn.filters.get("garment")
        or turn.filters.get("garment_type")
        or turn.filters.get("category")
        or ""
    ).casefold()

    # plan_turn must not call the LLM for this ask.
    planned = plan_turn(
        type("P", (), {"settings": type("S", (), {"request_timeout_seconds": 1})()})(),
        C1,
        [],
        {},
    )
    assert planned.llm_ok is True
    assert planned.group_by is None
    assert planned.metric == "analysis"
    assert "sweater" in str(
        planned.filters.get("garment")
        or planned.filters.get("category")
        or ""
    ).casefold()
    assert "break down" not in (planned.rewritten_question or "").casefold()
    assert "by material" not in (planned.rewritten_question or "").casefold()


def test_ops_brief_and_suggest_sheet_are_not_attribute_briefs():
    assert not is_product_attribute_brief_ask(
        "Matalan availability and price ops brief for in-stock vs mixed"
    )
    assert not is_product_attribute_brief_ask(
        "Generate a product specification for a typical Primark sweater"
    )
    assert not is_product_attribute_brief_ask("Break down Primark products by material")


def test_c1_brief_fields_are_only_named_facets():
    fields = analysis_fields_for_product_brief(C1)
    assert fields == [
        "material",
        "construction",
        "fit",
        "care_instruction",
        "color",
        "price_band",
    ]
    # No hierarchy / scrape / identity noise in the brief allowlist.
    for junk in (
        "target_group",
        "section",
        "family",
        "category",
        "subfamily",
        "search_text",
        "ref_norm",
        "availability",
    ):
        assert junk not in fields


def test_search_text_and_ref_norm_are_not_public_analysis_facets():
    assert not is_analysis_facet_attr("search_text")
    assert not is_analysis_facet_attr("ref_norm")
    assert is_analysis_facet_attr("material")
    assert is_analysis_facet_attr("care_instruction")
