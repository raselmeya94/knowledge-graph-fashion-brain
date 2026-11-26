"""Country-of-origin ISO ↔ English aliases for display + filter matching."""

from __future__ import annotations

import json
from pathlib import Path

from knowledge_graph.geo_country import (
    canonical_country_name,
    country_display_label,
    country_merge_key,
    known_country_alias_pairs_from_values,
)
from knowledge_graph.retrieval.ecom import _display_label, _merge_display_facet_rows


def test_cn_and_china_same_canonical():
    assert canonical_country_name("CN") == "China"
    assert canonical_country_name("China") == "China"
    assert canonical_country_name("china") == "China"
    assert country_merge_key("CN") == country_merge_key("China")
    assert country_display_label("CN") == "China"
    assert _display_label("CN", attribute="country_of_origin") == "China"
    assert _display_label("China", attribute="country_of_origin") == "China"


def test_common_apparel_origin_aliases():
    pairs = {
        "BD": "Bangladesh",
        "TR": "Turkey",
        "MA": "Morocco",
        "KH": "Cambodia",
        "PK": "Pakistan",
        "VN": "Vietnam",
        "IN": "India",
    }
    for code, name in pairs.items():
        assert country_display_label(code) == name
        assert country_display_label(name) == name
        assert country_merge_key(code) == country_merge_key(name)


def test_multi_origin_order_normalized():
    a = country_display_label("Cambodia Bangladesh")
    b = country_display_label("Bangladesh; Cambodia")
    c = country_display_label("KH BD")
    assert a == b == "Bangladesh; Cambodia"
    assert c == "Bangladesh; Cambodia"


def test_analysis_table_sums_cn_and_china():
    rows = _merge_display_facet_rows(
        [
            (_display_label("CN", attribute="country_of_origin"), 1530),
            (_display_label("China", attribute="country_of_origin"), 965),
            (_display_label("MA", attribute="country_of_origin"), 1091),
            (_display_label("Morocco", attribute="country_of_origin"), 4),
            (_display_label("BD", attribute="country_of_origin"), 498),
            (_display_label("Bangladesh", attribute="country_of_origin"), 660),
        ],
        combine="sum",
    )
    by_name = {name: count for name, count in rows}
    assert by_name["China"] == 1530 + 965
    assert by_name["Morocco"] == 1091 + 4
    assert by_name["Bangladesh"] == 498 + 660
    # Ranked by merged count: China 2495, Bangladesh 1158, Morocco 1095.
    assert [name for name, _ in rows][:3] == ["China", "Bangladesh", "Morocco"]


def test_live_ecom_country_stats_collisions():
    path = Path("data/generated/kg_stats/ecom_products/country_of_origin.json")
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = dict(payload.get("values") or {})
    collisions = known_country_alias_pairs_from_values(values)
    # At least the pairs we know from live meta.
    canons = {c["canonical"] for c in collisions}
    assert "China" in canons
    assert "Bangladesh" in canons
    assert "Turkey" in canons
    assert "Cambodia" in canons
    assert "Morocco" in canons
    # Printable audit for humans (kept in assert message if needed).
    assert collisions, "expected ISO↔name collisions in ecom country_of_origin stats"
