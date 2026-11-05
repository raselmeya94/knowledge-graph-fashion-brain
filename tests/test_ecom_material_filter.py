"""Cotton + brand-wise must filter materials and ignore grammar words like 'every'."""

from __future__ import annotations

from fashion_brain.ask import resolve_catalog_plan
from knowledge_graph.access import Principal
from knowledge_graph.models import GraphEdge, GraphNode
from knowledge_graph.retrieval.ecom import clean_product_filters, match_neo4j_products


def test_cotton_every_brand_wise_plan():
    plan = resolve_catalog_plan("how many products are are made by cotton every brand wise")
    assert plan["filters"] == {"material": "cotton"}
    assert plan["group_by"] == "brand"
    assert "garment" not in plan["filters"]


def test_analysis_material_wise_keeps_group_by():
    plan = resolve_catalog_plan("analysis all products with material wise")
    assert plan["group_by"] == "material"
    assert plan["analysis"] is False
    assert plan["filters"] == {}


def test_ecom_material_filter_matches_related_nodes(tmp_path):
    from tests.conftest import make_platform, write_sample_catalog

    platform = make_platform(write_sample_catalog(tmp_path / "raw"), tmp_path / "gen")
    cotton = GraphNode(
        id="ecom:mat:cotton",
        labels=["Entity", "Material"],
        graph="ecom_products",
        properties={"name": "Cotton 100"},
    )
    poly = GraphNode(
        id="ecom:mat:poly",
        labels=["Entity", "Material"],
        graph="ecom_products",
        properties={"name": "polyester"},
    )
    a = GraphNode(id="ecom:p:1", labels=["Entity", "Product"], graph="ecom_products", properties={"name": "A"})
    b = GraphNode(id="ecom:p:2", labels=["Entity", "Product"], graph="ecom_products", properties={"name": "B"})
    for node in (cotton, poly, a, b):
        platform.graph.upsert_node(node)
    platform.graph.upsert_edge(
        GraphEdge(id="e1", type="USES_MATERIAL", start_id=a.id, end_id=cotton.id, graph="ecom_products", properties={})
    )
    platform.graph.upsert_edge(
        GraphEdge(id="e2", type="USES_MATERIAL", start_id=b.id, end_id=poly.id, graph="ecom_products", properties={})
    )

    principal = Principal(subject="test", clearance="internal")
    cleaned = clean_product_filters({"material": "cotton", "garment": "every"})
    assert cleaned == {"material": "cotton"}
    matched = match_neo4j_products(platform, principal, "ecom_products", cleaned)
    assert [node.id for node in matched] == ["ecom:p:1"]
