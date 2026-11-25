"""Simple brand counts should include grounded assortment highlights."""

from __future__ import annotations

from knowledge_graph.retrieval.ecom import _filter_phrase


def test_filter_phrase_has_leading_space():
    phrase = _filter_phrase({"brand": "primark"})
    assert phrase.startswith(" ")
    assert "productswith" not in f"There are 4,058 products{phrase}."
    assert f"There are 4,058 products{phrase}." == "There are 4,058 products with brand primark."
