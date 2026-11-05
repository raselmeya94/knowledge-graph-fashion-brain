"""Schema-aware question study: unique KG columns + live value matches."""

from __future__ import annotations

from fashion_brain.agents import GraphRouterAgent, IntentionAgent
from fashion_brain.ask import resolve_catalog_plan
from fashion_brain.planning.kg_catalog import build_field_catalog, ground_filters_with_catalog
from knowledge_graph.models import GraphEdge, GraphNode


def test_analysis_primark_uses_brand_on_ecom(tmp_path):
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

    question = "analysis primark data"
    catalog = build_field_catalog(platform, question)
    assert "ecom_products" in catalog.graphs
    assert "brand" in catalog.graphs["ecom_products"].attributes
    assert "buyer" in catalog.graphs["own_products"].attributes
    assert any(hit.attribute == "brand" and hit.graph_id == "ecom_products" for hit in catalog.hits)

    keyword = resolve_catalog_plan(question)
    grounded = ground_filters_with_catalog(keyword["filters"], catalog)
    assert grounded.get("brand", "").upper() == "PRIMARK"
    assert "garment" not in grounded
    assert "buyer" not in grounded
    assert catalog.preferred_graphs() == ["ecom_products"]

    intention = IntentionAgent().understand(platform, question, None, None)
    assert intention.analysis is True
    assert intention.filters.get("brand", "").upper() == "PRIMARK"
    assert intention.preferred_graphs == ["ecom_products"]

    choice = GraphRouterAgent().choose(intention, platform=platform, question=question)
    assert choice.graph == "ecom_products"
    assert choice.selected_graphs == ["ecom_products"]
