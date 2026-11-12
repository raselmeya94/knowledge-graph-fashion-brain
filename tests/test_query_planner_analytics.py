"""Query planner + graph-agnostic analysis graph picking."""

from __future__ import annotations

from types import SimpleNamespace

from fashion_brain.agents.contracts import Intention
from fashion_brain.analytics.engine import pick_analysis_graphs
from fashion_brain.planning.query_planner import plan_retrieve


def test_plan_retrieve_marks_analysis_as_analytical_tool():
    intention = Intention(intent="product_count", analysis=True, filters={"buyer": "Papaya"})
    plan_retrieve(intention, "Analyse Papaya assortment")
    assert intention.retrieve_mode == "analytical_tool"
    assert intention.response_type == "analysis_report"
    assert "product_count" in intention.metrics
    assert "profile" in intention.operations


def test_plan_retrieve_shared_products_prefers_text2cypher():
    intention = Intention(intent="shared_products", group_by="material")
    plan_retrieve(intention, "Show materials shared across ecom and plmex")
    assert intention.retrieve_mode == "text2cypher"
    assert intention.response_type == "comparison"


def test_pick_analysis_graphs_prefers_buyer_capable_not_ecom_first():
    ecom = SimpleNamespace(
        id="ecom_products",
        kind="product",
        name="E-Commerce",
        description="retail scrape brands",
        capabilities=["brand_comparison", "price_analysis"],
        entities=["Brand", "Product"],
        status="active",
    )
    plmex = SimpleNamespace(
        id="plmex_products",
        kind="product",
        name="PLMEX",
        description="plm buyer factory sampling",
        capabilities=["buyer_analysis"],
        entities=["Buyer", "Product"],
        status="active",
    )
    platform = SimpleNamespace(kg_registry=SimpleNamespace(list_graphs=lambda: [ecom, plmex]))
    picked = pick_analysis_graphs(platform, filters={"buyer": "Papaya"})
    assert picked == ["plmex_products"]


def test_pick_analysis_graphs_honours_explicit_preferred():
    ecom = SimpleNamespace(
        id="ecom_products",
        kind="product",
        name="E-Commerce",
        description="retail",
        capabilities=["brand_comparison"],
        entities=["Brand"],
        status="active",
    )
    plmex = SimpleNamespace(
        id="plmex_products",
        kind="product",
        name="PLMEX",
        description="plm",
        capabilities=["buyer_analysis"],
        entities=["Buyer"],
        status="active",
    )
    platform = SimpleNamespace(kg_registry=SimpleNamespace(list_graphs=lambda: [ecom, plmex]))
    picked = pick_analysis_graphs(platform, preferred=["plmex_products"], filters={"brand": "Zara"})
    assert picked == ["plmex_products"]
