"""Party cues like 'at zara' / 'zara brand' must never become garment filters."""

from __future__ import annotations

from fashion_brain.ask import resolve_catalog_plan, sanitize_party_filters
from fashion_brain.planning.kg_catalog import FieldCatalog, GraphFields, ValueHit, ground_filters_with_catalog


def test_ecommerce_sources_count_prefers_ecom_not_garment():
    question = "how many products are stored inside e-commerces sources"
    plan = resolve_catalog_plan(question)
    assert plan["preferred_graphs"] == ["ecom_products"]
    assert "garment" not in plan["filters"]
    assert plan["filters"] == {} or set(plan["filters"]) <= {"brand", "buyer"}


def test_at_zara_brand_is_brand_not_garment():
    for question in (
        "how many products at zara brand",
        "how many products at zara",
        "how many products from zara",
        "count products for zara brand",
        "how many zara brand products",
    ):
        plan = resolve_catalog_plan(question)
        assert plan["filters"].get("brand") == "zara", question
        assert "garment" not in plan["filters"], question
        assert "garment_type" not in plan["filters"], question


def test_sanitize_strips_party_wrapped_as_garment():
    assert sanitize_party_filters({"brand": "zara", "garment": "at zara"}) == {"brand": "zara"}
    assert sanitize_party_filters({"brand": "zara", "garment": "zara brand"}) == {"brand": "zara"}
    assert sanitize_party_filters({"brand": "zara", "garment": "zara"}) == {"brand": "zara"}


def test_grounding_remaps_token_inside_garment_phrase():
    catalog = FieldCatalog(
        graphs={
            "ecom_products": GraphFields(
                graph_id="ecom_products",
                attributes=["brand", "category"],
            )
        },
        hits=[
            ValueHit(token="zara", attribute="brand", graph_id="ecom_products", matched_value="ZARA"),
        ],
    )
    grounded = ground_filters_with_catalog({"garment": "at zara"}, catalog, question="at zara")
    assert grounded.get("brand") == "ZARA"
    assert "garment" not in grounded
