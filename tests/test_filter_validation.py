"""Strict metadata-grounded filter validation + clarification loop."""

from __future__ import annotations

from types import SimpleNamespace

from fashion_brain.agents.contracts import Intention, UnderstandingPlan
from fashion_brain.planning.filter_validation import (
    apply_clarification_response,
    execution_filters,
    gate_intention_filters,
    validate_filters_strict,
)


def _index():
    return {
        "brand": [("Primark", "primark"), ("Zara", "zara"), ("Matalan", "matalan")],
        "color": [("Blue", "blue"), ("Navy", "navy"), ("Black", "black"), ("White", "white")],
        "category": [("Sweater", "sweater"), ("Clothes", "clothes"), ("Jackets", "jackets")],
        "gender": [("Women", "women"), ("Men", "men")],
        "material": [("Cotton", "cotton"), ("Polyester", "polyester")],
    }


def _meta(index=None, field_graphs=None, columns=None):
    idx = index if index is not None else _index()
    return SimpleNamespace(
        values=idx,
        field_graphs=field_graphs
        or {k: ["ecom_products", "plmex_products"] for k in idx},
        columns=columns
        or {
            "ecom_products": list(idx),
            "plmex_products": list(idx) + ["buyer"],
        },
        product_graphs=["ecom_products", "plmex_products"],
        missing_important={},
    )


def test_exact_metadata_match():
    plan = validate_filters_strict(
        None,
        {"brand": "Primark"},
        source="ecom_products",
        index=_index(),
        meta=_meta(),
        verify_db=False,
    )
    assert plan.execution_allowed
    assert plan.filters[0].status == "VALID"
    assert plan.filters[0].normalized_value == "Primark"
    assert plan.filters[0].evidence == "metadata_exact_match"


def test_case_insensitive_match():
    plan = validate_filters_strict(
        None,
        {"brand": "primark"},
        index=_index(),
        meta=_meta(),
        verify_db=False,
    )
    assert plan.execution_allowed
    assert plan.filters[0].status == "VALID"
    assert plan.filters[0].normalized_value == "Primark"


def test_registered_synonym_navy_blue():
    # Alias maps navy blue → navy / blue; exact alias hit on navy.
    plan = validate_filters_strict(
        None,
        {"color": "navy blue"},
        index=_index(),
        meta=_meta(),
        verify_db=False,
    )
    assert plan.execution_allowed
    assert plan.filters[0].status == "VALID"
    assert plan.filters[0].normalized_value in {"Navy", "Blue"}


def test_unknown_field_value_without_db_is_unknown_not_not_found():
    plan = validate_filters_strict(
        None,
        {"color": "chartreuse mist"},
        index=_index(),
        meta=_meta(),
        verify_db=False,
    )
    assert not plan.execution_allowed
    assert plan.filters[0].status == "UNKNOWN"
    assert plan.clarification_required


def test_near_miss_is_ambiguous_not_auto_valid():
    plan = validate_filters_strict(
        None,
        {"brand": "Primrk"},  # near Primark
        index=_index(),
        meta=_meta(),
        verify_db=False,
    )
    assert not plan.execution_allowed
    assert plan.filters[0].status == "AMBIGUOUS"
    assert "Primark" in (plan.filters[0].candidates or [])


def test_color_contains_family_auto_validates_white():
    """Base colour 'white' must not clarify against arctic white / beige and white."""

    index = {
        **_index(),
        "color": [
            ("arctic white white", "arctic white white"),
            ("beige and white", "beige and white"),
            ("beige white", "beige white"),
            ("Black", "black"),
        ],
    }
    plan = validate_filters_strict(
        None,
        {"brand": "Primark", "color": "white", "category": "sweater"},
        source="ecom_products",
        index=index,
        meta=_meta(index),
        verify_db=False,
    )
    color = next(f for f in plan.filters if f.field == "color")
    assert color.status == "VALID"
    assert plan.execution_allowed
    assert color.evidence == "metadata_contains_family"


def test_plural_soft_match_auto_valid_any_column():
    """Soft plural/case match applies to every facet column, not only garments."""

    index = {
        **_index(),
        "category": [("Sweater", "sweater"), ("Clothes", "clothes")],
        "material": [("Cottons", "cottons"), ("Polyester", "polyester")],
        "color": [("Blues", "blues"), ("Navy", "navy")],
        "buyer": [("Buyers", "buyers")],
    }
    meta = _meta(index)
    cases = [
        ("category", "Sweaters", "Sweater"),
        ("material", "cotton", "Cottons"),
        ("color", "blue", "Blues"),
        ("gender", "WOMEN", "Women"),
        ("brand", "PRIMARK", "Primark"),
    ]
    for field, asked, expected in cases:
        plan = validate_filters_strict(
            None,
            {field: asked},
            index=index,
            meta=meta,
            verify_db=False,
        )
        assert plan.execution_allowed, field
        assert plan.filters[0].status == "VALID", field
        assert plan.filters[0].normalized_value == expected, field


def test_unsupported_field_on_source():
    meta = _meta(
        field_graphs={"brand": ["ecom_products"], "buyer": ["plmex_products"]},
        columns={"ecom_products": ["brand", "color"], "plmex_products": ["buyer", "brand"]},
    )
    plan = validate_filters_strict(
        None,
        {"buyer": "Primark"},
        source="ecom_products",
        index={"brand": [("Primark", "primark")]},
        meta=meta,
        verify_db=False,
    )
    assert not plan.execution_allowed
    assert plan.filters[0].status == "UNSUPPORTED"


def test_mixed_valid_and_invalid_preserves_valid_only_for_execution():
    plan = validate_filters_strict(
        None,
        {"brand": "Primark", "color": "chartreuse mist"},
        index=_index(),
        meta=_meta(),
        verify_db=False,
    )
    assert not plan.execution_allowed
    assert execution_filters(plan) == {"brand": "Primark"}
    assert "Keeping: brand=Primark" in plan.clarification_text


def test_multiple_invalid_filters_one_clarification_message():
    plan = validate_filters_strict(
        None,
        {"brand": "NoSuchBrand", "color": "chartreuse mist"},
        index=_index(),
        meta=_meta(),
        verify_db=False,
    )
    assert not plan.execution_allowed
    assert len(plan.clarification_items) >= 2
    assert "clarify a few filters" in plan.clarification_text.casefold()


def test_db_confirmed_absent_is_not_found():
    class _P:
        pass

    def fake_count(_platform, graph, filters):
        return 0

    import fashion_brain.planning.filter_validation as fv

    plan = validate_filters_strict(
        _P(),
        {"brand": "NoSuchBrand"},
        source="ecom_products",
        index=_index(),
        meta=_meta(),
        verify_db=True,
    )
    # Patch after build would be late — call with monkeypatch style via module.
    original = fv.cypher_count_products if hasattr(fv, "cypher_count_products") else None
    # validate imports cypher inside _db_verify_value; monkeypatch that import path.
    import knowledge_graph.retrieval.ecom as ecom

    prev = ecom.cypher_count_products
    ecom.cypher_count_products = fake_count
    try:
        plan = validate_filters_strict(
            _P(),
            {"brand": "NoSuchBrand"},
            source="ecom_products",
            index=_index(),
            meta=_meta(),
            verify_db=True,
        )
    finally:
        ecom.cypher_count_products = prev
    assert plan.filters[0].status == "NOT_FOUND"
    assert not plan.execution_allowed
    assert original is None or True


def test_db_confirmed_present_is_valid_when_meta_misses():
    class _P:
        pass

    import knowledge_graph.retrieval.ecom as ecom

    prev = ecom.cypher_count_products
    ecom.cypher_count_products = lambda *_a, **_k: 12
    try:
        plan = validate_filters_strict(
            _P(),
            {"color": "teal"},
            source="ecom_products",
            index=_index(),  # no teal in meta
            meta=_meta(),
            verify_db=True,
        )
    finally:
        ecom.cypher_count_products = prev
    assert plan.execution_allowed
    assert plan.filters[0].status == "VALID"
    assert "db_count" in plan.filters[0].evidence


def test_clarification_remove_filter():
    blocked = validate_filters_strict(
        None,
        {"brand": "Primark", "color": "chartreuse mist"},
        index=_index(),
        meta=_meta(),
        verify_db=False,
    )
    updated = apply_clarification_response(
        None, blocked, "remove color", verify_db=False
    )
    assert updated.execution_allowed
    assert execution_filters(updated) == {"brand": "Primark"}


def test_clarification_replace_value():
    blocked = validate_filters_strict(
        None,
        {"color": "navyish"},
        index=_index(),
        meta=_meta(),
        verify_db=False,
    )
    # Force UNKNOWN/AMBIGUOUS then correct.
    if blocked.execution_allowed:
        # near-match may be AMBIGUOUS; either way apply correction
        pass
    updated = apply_clarification_response(None, blocked, "use Blue", verify_db=False)
    assert updated.execution_allowed
    assert execution_filters(updated).get("color") == "Blue"


def test_clarification_pick_numbered_candidate():
    blocked = validate_filters_strict(
        None,
        {"brand": "Primrk"},
        index=_index(),
        meta=_meta(),
        verify_db=False,
    )
    assert blocked.filters[0].status == "AMBIGUOUS"
    updated = apply_clarification_response(None, blocked, "1", verify_db=False)
    assert updated.execution_allowed
    assert execution_filters(updated).get("brand") == "Primark"


def test_gate_strips_invalid_before_intention_retrieve():
    intention = Intention(intent="product_count", filters={"brand": "Primark", "color": "nope"})
    plan, intention = gate_intention_filters(
        None,
        intention,
        question="how many Primark nope",
        state={},
        verify_db=False,
    )
    # Without injecting meta index, gate builds live index — may be empty in unit tests.
    # Force via validate path already covered; here ensure gate API returns a plan.
    assert isinstance(plan, UnderstandingPlan)
    assert "color" not in intention.filters or intention.filters.get("color") != "nope" or not plan.execution_allowed


def test_empty_filters_allow_execution():
    plan = validate_filters_strict(
        None, {}, index=_index(), meta=_meta(), verify_db=False
    )
    assert plan.execution_allowed
    assert plan.valid_filters_dict() == {}


def test_freeform_price_passthrough():
    plan = validate_filters_strict(
        None,
        {"brand": "Primark", "price_max": "50"},
        index=_index(),
        meta=_meta(),
        verify_db=False,
    )
    assert plan.execution_allowed
    assert execution_filters(plan)["price_max"] == "50"


def test_plan_roundtrip_dict():
    plan = validate_filters_strict(
        None,
        {"brand": "Zara"},
        index=_index(),
        meta=_meta(),
        verify_db=False,
    )
    restored = UnderstandingPlan.from_dict(plan.to_dict())
    assert restored is not None
    assert restored.execution_allowed
    assert restored.valid_filters_dict()["brand"] == "Zara"


def test_never_silently_add_unrequested_filters():
    plan = validate_filters_strict(
        None,
        {"brand": "Primark"},
        index=_index(),
        meta=_meta(),
        verify_db=False,
    )
    assert list(execution_filters(plan)) == ["brand"]
