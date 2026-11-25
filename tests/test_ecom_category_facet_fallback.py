"""Ecom/plmex hierarchy role fallback when group_by=category is empty."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from knowledge_graph.retrieval.ecom import resolve_focused_analysis_field, resolve_group_by_with_fallback


def test_focused_analysis_falls_through_empty_category_to_family():
    platform = SimpleNamespace()
    filters = {"brand": "primark"}

    def _group_by(_platform, _gid, field, _filters):
        if field == "category":
            return [("(unspecified)", 4000), ("Clothes", 50)]
        if field == "family":
            return [("Clothing", 2800), ("Accessories", 900), ("Footwear", 300)]
        return [("(unspecified)", 100)]

    with (
        patch("knowledge_graph.retrieval.ecom.cypher_count_products", return_value=4050),
        patch("knowledge_graph.retrieval.ecom.cypher_group_by", side_effect=_group_by),
        patch(
            "knowledge_graph.retrieval.ecom._align_party_filters_for_graph",
            side_effect=lambda _p, _g, f: f,
        ),
    ):
        fields = resolve_focused_analysis_field(
            platform, "ecom_products", "category", filters
        )
    assert fields == ["family"]


def test_focused_analysis_keeps_category_when_populated():
    platform = SimpleNamespace()

    def _group_by(_platform, _gid, field, _filters):
        if field == "category":
            return [("Clothes", 1464), ("Jackets", 424), ("Accessories", 414)]
        return [("Clothing", 100)]

    with (
        patch("knowledge_graph.retrieval.ecom.cypher_count_products", return_value=2302),
        patch("knowledge_graph.retrieval.ecom.cypher_group_by", side_effect=_group_by),
        patch(
            "knowledge_graph.retrieval.ecom._align_party_filters_for_graph",
            side_effect=lambda _p, _g, f: f,
        ),
    ):
        fields = resolve_focused_analysis_field(
            platform, "ecom_products", "category", {"brand": "zara"}
        )
    assert fields == ["category"]


def test_count_group_by_fallback_helper():
    platform = SimpleNamespace()

    def _group_by(_platform, _gid, field, _filters):
        if field == "category":
            return [("(unspecified)", 500)]
        if field == "section":
            return [("Women", 300), ("Men", 150), ("Kids", 50)]
        return [("(unspecified)", 500)]

    with (
        patch("knowledge_graph.retrieval.ecom.cypher_count_products", return_value=500),
        patch("knowledge_graph.retrieval.ecom.cypher_group_by", side_effect=_group_by),
        patch(
            "knowledge_graph.retrieval.ecom._align_party_filters_for_graph",
            side_effect=lambda _p, _g, f: f,
        ),
    ):
        assert (
            resolve_group_by_with_fallback(
                platform, "ecom_products", "category", {"brand": "primark"}
            )
            == "section"
        )


def test_plmex_construction_falls_through_to_composition():
    """Construction empty on PLM → use composition blend strings (same meaning)."""

    platform = SimpleNamespace()

    def _group_by(_platform, _gid, field, _filters):
        if field == "construction":
            return [("(unspecified)", 560)]
        if field == "composition":
            # Need >=15% coverage of product total or focused analysis skips.
            return [
                ("100%Cotton", 120),
                ("50%Viscose 28%Polyester 22%Polyamide", 80),
                ("66%Polyester 32%Acrylic 2%Elastane", 70),
            ]
        if field == "material":
            return [("elastane", 139), ("cotton", 65)]
        return [("(unspecified)", 560)]

    with (
        patch("knowledge_graph.retrieval.ecom.cypher_count_products", return_value=560),
        patch("knowledge_graph.retrieval.ecom.cypher_group_by", side_effect=_group_by),
        patch(
            "knowledge_graph.retrieval.ecom._align_party_filters_for_graph",
            side_effect=lambda _p, _g, f: f,
        ),
    ):
        fields = resolve_focused_analysis_field(
            platform,
            "plmex_products",
            "construction",
            {"brand": "primark", "garment": "sweater"},
        )
    assert fields == ["composition"]
