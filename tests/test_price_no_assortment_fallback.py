"""Price-history asks must not fall through to assortment analysis on PLM."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from knowledge_graph.retrieval.ecom import append_neo4j_price_history_answer


def test_plmex_price_ask_returns_unavailable_not_empty_false():
    lines: list[str] = []
    evidence: list = []
    caveats: list[str] = []
    with patch(
        "knowledge_graph.retrieval.ecom.cypher_count_nodes",
        return_value=0,
    ), patch(
        "knowledge_graph.retrieval.ecom.price_history_mode",
        return_value="movers_down",
    ):
        ok = append_neo4j_price_history_answer(
            SimpleNamespace(),
            SimpleNamespace(),
            "plmex_products",
            {},
            "Which products had the biggest price drops?",
            10,
            lines,
            evidence,
            caveats,
        )
    assert ok is True
    text = "\n".join(lines).casefold()
    assert "not available" in text
    assert "plm" in text
    assert "e-commerce" in text or "ecommerce" in text
    assert "product analysis" not in text
    assert any(getattr(e, "record_id", None) == "price_history_unavailable" for e in evidence)
