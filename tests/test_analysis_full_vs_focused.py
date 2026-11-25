"""Full analysis = all facets; focused/breakdown = one facet; compare = side-by-side."""

from __future__ import annotations

from fashion_brain.ask import _is_analysis, resolve_catalog_plan
from fashion_brain.agents.contracts import Intention
from fashion_brain.planning.query_planner import plan_retrieve


def test_full_analysis_clears_group_by_and_sets_analysis():
    plan = resolve_catalog_plan(
        "Give me a full analysis of Primark",
        brand_names=["Primark", "Zara"],
    )
    assert plan["analysis"] is True
    assert plan.get("group_by") in (None, "")
    assert "primark" in str(plan.get("filters", {}).get("brand") or "").casefold()


def test_focused_breakdown_keeps_analysis_and_group_by():
    plan = resolve_catalog_plan(
        "Show Primark category breakdown",
        brand_names=["Primark", "Zara"],
    )
    assert plan["analysis"] is True
    assert str(plan.get("group_by") or "").casefold() == "category"
    assert "primark" in str(plan.get("filters", {}).get("brand") or "").casefold()


def test_material_wise_is_analysis_ask():
    assert _is_analysis("analysis all products with material wise")
    assert _is_analysis("Primark material breakdown")


def test_assortment_compare_is_full_side_by_side_analysis():
    plan = resolve_catalog_plan(
        "Compare Primark and Zara assortment profiles",
        brand_names=["Primark", "Zara"],
    )
    assert plan["analysis"] is True
    assert plan.get("group_by") in (None, "")
    brand = str(plan.get("filters", {}).get("brand") or "").casefold()
    assert "primark" in brand and "zara" in brand
    assert "|" in brand


def test_focused_analysis_intention_uses_analytical_tool():
    intention = Intention(
        intent="analysis",
        analysis=True,
        group_by="material",
        filters={"brand": "Primark"},
    )
    plan_retrieve(intention, "Primark material breakdown")
    assert intention.retrieve_mode == "analytical_tool"
