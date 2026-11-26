"""End-user facet labels: merge duplicate surfaces, polish price bands."""

from __future__ import annotations

from knowledge_graph.retrieval.ecom import (
    _display_label,
    _merge_display_facet_rows,
    _price_band_display_label,
)


def test_price_band_display_is_human():
    assert _price_band_display_label("Eur:value") == "EUR · Value"
    assert _price_band_display_label("gbp:budget") == "GBP · Budget"
    assert _price_band_display_label("Usd_mid") == "USD · Mid"
    assert _display_label("Eur:value", attribute="price_band") == "EUR · Value"
    assert _display_label("gbp:premium") == "GBP · Premium"


def test_merge_duplicate_women_and_in_stock():
    rows = _merge_display_facet_rows(
        [
            ("Women", 10174),
            ("Women", 10160),
            ("Men", 3134),
            ("Men", 3115),
            ("Kids", 2529),
        ]
    )
    by_name = {name: count for name, count in rows}
    # Max, not sum — overlapping duplicate facet nodes must not inflate totals.
    assert by_name["Women"] == 10174
    assert by_name["Men"] == 3134
    assert by_name["Kids"] == 2529
    assert [name for name, _ in rows] == ["Women", "Men", "Kids"]

    avail = _merge_display_facet_rows(
        [
            ("In stock", 12973),
            ("In stock", 623),
            ("Mixed", 4335),
        ]
    )
    assert avail[0] == ("In stock", 12973)
    assert avail[1] == ("Mixed", 4335)


def test_availability_canon_and_mixed():
    assert _display_label("InStock") == "In stock"
    assert _display_label("instock") == "In stock"
    assert _display_label("mixed", attribute="availability") == "Mixed"
