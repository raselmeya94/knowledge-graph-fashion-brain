"""'all buyers in ecomm' must not treat 'comm' as a buyer or fuzzy-match Primark."""

from __future__ import annotations

from fashion_brain.agents import GraphRouterAgent, IntentionAgent
from fashion_brain.ask import resolve_catalog_plan
from knowledge_graph.models import GraphEdge, GraphNode


def test_analysis_all_buyers_in_ecomm_plans_brand_breakdown():
    for question in (
        "analysis all buyer in e comm data",
        "analysis all buyers in ecomm data",
        "analysis all buyer in e-comm data",
    ):
        plan = resolve_catalog_plan(question)
        assert plan["filters"] == {}
        assert plan["group_by"] == "brand"
        assert plan["analysis"] is True
        assert plan["preferred_graphs"] == ["ecom_products"]
        assert "comm" not in str(plan["filters"]).casefold()


def test_analysis_all_buyers_routes_to_ecom(tmp_path):
    from tests.conftest import make_platform, write_sample_catalog

    raw = write_sample_catalog(tmp_path / "raw")
    platform = make_platform(raw, tmp_path / "gen")
    brand = GraphNode(
        id="ecom:brand:primark",
        labels=["Entity", "Brand"],
        graph="ecom_products",
        properties={"name": "PRIMARK"},
    )
    product = GraphNode(
        id="ecom:primark:1",
        labels=["Entity", "Product"],
        graph="ecom_products",
        properties={"name": "Tee", "brand": "PRIMARK", "category": "Tops"},
    )
    platform.graph.upsert_node(brand)
    platform.graph.upsert_node(product)
    platform.graph.upsert_edge(
        GraphEdge(
            id="ecom:edge:brand:1",
            type="HAS_BRAND",
            start_id=product.id,
            end_id=brand.id,
            graph="ecom_products",
            properties={},
        )
    )

    question = "analysis all buyer in e comm data"
    intention = IntentionAgent().understand(platform, question, None, None)
    assert intention.filters == {}
    assert intention.group_by == "brand"
    assert intention.preferred_graphs == ["ecom_products"]
    assert intention.analysis is True

    choice = GraphRouterAgent().choose(intention, platform=platform, question=question)
    assert choice.graph == "ecom_products"
