"""Stock-status language must never become brand/buyer or a false InStock filter."""

from __future__ import annotations

from fashion_brain.ask import (
    _compare_party_names,
    _is_analysis,
    _is_availability_history_question,
    _is_grammar_party_name,
    _is_party_token,
    _reliable_filter,
    availability_history_mode,
    catalog_metric_family,
    enrich_ecom_question_filters,
    is_availability_price_ops_brief_ask,
    mentions_stock_status_comparison,
    resolve_catalog_plan,
    sanitize_party_filters,
)


A16 = (
    "For Matalan online, combine availability and pricing into one ops brief: "
    "stock-status mix, and whether price events are concentrated on in-stock vs "
    "mixed listings if the data supports that join."
)


def test_mixed_is_not_a_party_or_reliable_brand_value():
    assert _is_party_token("mixed") is False
    assert _is_grammar_party_name("mixed") is True
    assert _is_party_token("in-stock") is False
    assert _is_grammar_party_name("in stock") is True
    assert sanitize_party_filters({"brand": "mixed"}) == {}
    assert sanitize_party_filters({"brand": "in-stock | mixed"}) == {}


def test_in_stock_vs_mixed_is_not_a_brand_compare():
    assert _compare_party_names("in-stock vs mixed listings") == []
    assert _compare_party_names("Primark vs Zara") == ["Primark", "Zara"]


def test_a16_keeps_matalan_and_does_not_invent_stock_filters():
    assert mentions_stock_status_comparison(A16) is True
    assert _is_availability_history_question(A16) is False
    assert availability_history_mode(A16) is None
    assert is_availability_price_ops_brief_ask(A16) is True
    assert _is_analysis(A16) is False
    assert catalog_metric_family(A16) == "availability_price"

    plan = resolve_catalog_plan(A16, brand_names=["Matalan", "Primark", "Zara"])
    assert plan["filters"].get("brand") == "matalan"
    assert plan.get("analysis") is False

    enriched = enrich_ecom_question_filters(A16, plan["filters"])
    sanitized = sanitize_party_filters(enriched)
    assert sanitized.get("brand") == "matalan"
    assert "mixed" not in str(sanitized.get("brand") or "").casefold()
    assert "availability" not in sanitized


def test_a16_ops_brief_answer_is_not_whole_catalog_mix():
    """Answer builder must require a scoped brief, not full assortment tables."""

    from knowledge_graph.retrieval.ecom import answer_availability_price_ops_brief

    # Wrong ask type → no ops-brief answer (falls through elsewhere).
    assert answer_availability_price_ops_brief(None, "ecom_products", "full analysis of Primark") is None
    # Detector gates the builder even without a live Neo4j session.
    assert is_availability_price_ops_brief_ask("full analysis of Primark") is False


def test_plain_in_stock_filter_still_allowed():
    q = "How many Primark products are in stock?"
    assert mentions_stock_status_comparison(q) is False
    out = enrich_ecom_question_filters(q, {"brand": "Primark"})
    assert out.get("availability") == "InStock"
    assert out.get("brand") == "Primark"


def test_mixed_does_not_pass_as_invented_garment_noise():
    # Report English stays out of free-text garment filters; Mixed as availability
    # value is still allowed through field-specific paths.
    assert _reliable_filter("sweater") is True


def test_ground_all_fields_against_live_metadata(monkeypatch):
    """Every categorical field is meta-gated — unknown values are dropped, not assumed."""

    from fashion_brain.planning import metadata_resolver as mr

    def fake_index(platform, fields, per_field=200):
        return {
            "brand": [("Matalan", "matalan"), ("Primark", "primark")],
            "color": [("Blue", "blue")],
            "material": [("Cotton", "cotton")],
            "availability": [("InStock", "instock"), ("Mixed", "mixed")],
        }

    monkeypatch.setattr(mr, "build_value_index", fake_index)
    out = mr.ground_filters_against_live_metadata(
        object(),
        {
            "brand": "mixed",
            "color": "ops",
            "material": "Cotton",
            "availability": "InStock",
            "price_max": "25",
        },
        question="cotton products under 25",
    )
    assert out.get("material") == "Cotton"
    assert out.get("price_max") == "25"
    assert "brand" not in out  # mixed is not a brand in meta
    assert "color" not in out  # ops is not a color in meta
    assert out.get("availability") == "InStock"

    a16 = mr.ground_filters_against_live_metadata(
        object(),
        {"brand": "mixed", "availability": "InStock", "buyer": "Matalan"},
        question=A16,
    )
    assert "availability" not in a16
    assert "brand" not in a16 or str(a16.get("brand") or "").casefold() != "mixed"
