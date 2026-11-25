"""Price/availability history must prefer ecom (PriceChangeEvent source)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from fashion_brain.agents.contracts import Intention
from fashion_brain.orchestration.catalog_pipeline import _prefer_ecom_for_price_history


def test_price_drop_ask_forces_ecom_over_sticky_plmex():
    intention = Intention(
        intent="product_count",
        filters={"brand": "prince"},
        thinking="test",
        preferred_graphs=["plmex_products"],
        metrics=["price"],
    )
    with (
        patch(
            "fashion_brain.orchestration.catalog_pipeline._explicit_graphs_in_question",
            return_value=[],
        ),
        patch(
            "knowledge_graph.retrieval.ecom.registry_product_graphs",
            return_value=["plmex_products", "ecom_products"],
        ),
    ):
        out = _prefer_ecom_for_price_history(
            SimpleNamespace(),
            intention,
            question="Which products had the biggest price drops?",
        )
    assert out.preferred_graphs[0] == "ecom_products"


def test_assortment_ask_does_not_force_ecom():
    intention = Intention(
        intent="product_count",
        filters={"brand": "primark"},
        thinking="test",
        preferred_graphs=["plmex_products"],
    )
    out = _prefer_ecom_for_price_history(
        SimpleNamespace(),
        intention,
        question="Analyze Primark sweater assortment",
    )
    assert out.preferred_graphs[0] == "plmex_products"
