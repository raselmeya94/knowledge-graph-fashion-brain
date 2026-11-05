"""Material-wise asks must stay one dimension and not duplicate table+chart."""

from __future__ import annotations

from fashion_brain.ask import resolve_catalog_plan
from knowledge_graph.retrieval.ecom import _dimension_breakdown_lines


def test_analysis_material_wise_is_group_by_not_full_analysis():
    plan = resolve_catalog_plan("analysis all products with material wise")
    assert plan["group_by"] == "material"
    assert plan["analysis"] is False
    assert plan["filters"] == {}


def test_material_breakdown_is_chart_without_table():
    lines = _dimension_breakdown_lines(
        "material",
        [("polyester", 367), ("cotton", 169)],
        536,
        5,
    )
    text = "\n".join(lines)
    assert "```chart" in text
    assert "| Name | Count |" not in text
    assert "polyester" in text


def test_brand_breakdown_is_table_without_chart():
    lines = _dimension_breakdown_lines(
        "brand",
        [("zara", 462), ("walmart", 284)],
        746,
        5,
    )
    text = "\n".join(lines)
    assert "```chart" not in text
    assert "| Name | Count | Share % |" in text
