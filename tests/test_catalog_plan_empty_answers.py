"""Keyword plans must not invent empty answers via bad filters/routes."""

from fashion_brain.ask import (
    _is_price_history_question,
    _is_product_details_request,
    is_open_product_listing,
    plan_routes,
    resolve_catalog_plan,
    temporal_count_entity,
)
from knowledge_graph.retrieval.ecom import price_history_mode


def test_zara_colours_break_down_is_brand_color_breakdown():
    plan = resolve_catalog_plan("How do Zara colours break down?", None, carry=False)
    assert plan["filters"].get("brand", "").casefold() == "zara"
    assert plan["group_by"] == "color"
    assert "color" not in (plan["filters"] or {}) or plan["filters"].get("color") != "zara"


def test_sweaters_or_knitwear_are_garment_not_brands():
    plan = resolve_catalog_plan(
        "Which brand has the most sweaters or knitwear?", None, carry=False
    )
    brand = str(plan["filters"].get("brand") or "").casefold()
    assert "sweater" not in brand
    assert "knitwear" not in brand
    assert plan["group_by"] == "brand"
    assert "sweater" in str(plan["filters"].get("garment") or "").casefold()


def test_tshirts_listing_keeps_garment_filter():
    plan = resolve_catalog_plan("What t-shirts do we have?", None, carry=False)
    assert "t-shirt" in str(plan["filters"].get("garment") or "").casefold()


def test_style_id_routes_to_product_details():
    q = "What is the Zara style 08965901 made of?"
    assert _is_product_details_request(q)
    assert plan_routes(q) == ["product_details"]
    plan = resolve_catalog_plan(q, None, carry=False)
    assert plan["filters"].get("product_id") == "08965901"


def test_prices_moving_is_price_history_trend():
    q = "Overall, how are Zara prices moving?"
    assert _is_price_history_question(q)
    assert price_history_mode(q) == "trend"
    plan = resolve_catalog_plan(q, None, carry=False)
    assert plan["filters"].get("brand", "").casefold() == "zara"


def test_matalan_examples_capture_limit():
    plan = resolve_catalog_plan(
        "Can you show me about 10 Matalan products as examples?", None, carry=False
    )
    assert plan.get("limit") == 10


def test_primark_colours_keeps_brand():
    plan = resolve_catalog_plan("What colours does Primark sell most?", None, carry=False)
    assert plan["group_by"] == "color"
    assert plan["filters"].get("brand", "").casefold() == "primark"


def test_how_many_products_is_not_open_listing():
    q = "How many products does Primark have?"
    assert is_open_product_listing(q) is False
    assert plan_routes(q) == ["product_count"]


def test_which_products_cotton_is_open_listing():
    assert is_open_product_listing("Which products are mainly cotton?") is True


def test_materials_across_range_not_up_range_brands():
    plan = resolve_catalog_plan(
        "What materials show up most often across the range?", None, carry=False
    )
    brand = str(plan["filters"].get("brand") or "").casefold()
    assert "up" not in brand.split("|")
    assert "range" not in brand.split("|")
    assert plan["group_by"] == "material"


def test_availability_changes_maps_to_snapshot_count():
    q = "How many availability changes are stored?"
    assert temporal_count_entity(q) == "AvailabilitySnapshot"
    assert plan_routes(q) == ["product_count"]


def test_zara_tops_material_mixes_not_material_mixes_filter():
    from fashion_brain.ask import garment_type_from_question

    q = "In Zara tops, which material mixes are most common?"
    assert garment_type_from_question(q) == "tops"
    plan = resolve_catalog_plan(q, None, carry=False)
    assert plan["filters"].get("brand", "").casefold() == "zara"
    assert "top" in str(plan["filters"].get("garment") or "").casefold()
    assert "mix" not in str(plan["filters"].get("material") or "").casefold()
    assert plan["group_by"] == "material"


def test_full_ecom_material_report_not_scoped_to_garment_tops():
    """'top constructions / top materials' is ranking language, not garment=tops."""

    from fashion_brain.ask import garment_type_from_question

    q = (
        "For the full e-commerce assortment (every brand in the scrape), produce a "
        "composition / material mix report: which fibre blends dominate, top "
        "constructions if composition is thin, and a short note on how concentrated "
        "the top materials are."
    )
    assert garment_type_from_question(q) is None
    plan = resolve_catalog_plan(q, None, carry=False)
    assert not plan["filters"].get("garment")
    assert not plan["filters"].get("garment_type")
    assert not plan["filters"].get("subfamily")
    assert plan["group_by"] == "material"


def test_dresses_stem_and_brand_garment_pairs():
    from fashion_brain.ask import garment_type_from_question

    assert garment_type_from_question("Primark dresses") == "dress"
    plan = resolve_catalog_plan(
        "In Primark dresses, which materials are most common?", None, carry=False
    )
    assert plan["filters"].get("brand", "").casefold() == "primark"
    assert plan["filters"].get("garment") == "dress"
    assert plan["group_by"] == "material"

    jackets = resolve_catalog_plan("How many Zara jackets do we have?", None, carry=False)
    assert jackets["filters"].get("brand", "").casefold() == "zara"
    assert "jacket" in str(jackets["filters"].get("garment") or "")

    skirts = resolve_catalog_plan("Break down Zara skirts by material", None, carry=False)
    brand = str(skirts["filters"].get("brand") or "").casefold()
    assert "break" not in brand
    assert "down" not in brand
    assert "zara" in brand
    assert skirts["group_by"] == "material"


def test_compare_tshirt_keeps_garment():
    plan = resolve_catalog_plan(
        "Compare Primark and Zara t-shirt compositions", None, carry=False
    )
    assert "primark" in str(plan["filters"].get("brand") or "").casefold()
    assert "zara" in str(plan["filters"].get("brand") or "").casefold()
    assert "t-shirt" in str(plan["filters"].get("garment") or "").casefold()


def test_plmex_zara_composition_mixes_plan():
    plan = resolve_catalog_plan("PLMEX Zara composition mixes", None, carry=False)
    assert plan["filters"].get("brand", "").casefold() == "zara"
    assert plan["group_by"] == "material"


def test_swweaters_typo_grounds_to_sweater_not_brand_or_total():
    """Typo 'swweaters' must map via kg_stats to sweater — never a fake brand.

    Meta fast path used to drop the unknown token and answer the whole PLMEX
    total (~18000). Schema grounding should recover garment/category=sweater.
    """

    from fashion_brain.ask import is_catalog_dataset_total_ask

    q = "how many swweaters in plmex"
    assert is_catalog_dataset_total_ask(q) is False
    plan = resolve_catalog_plan(q, None, carry=False)
    brand = str(plan["filters"].get("brand") or "").casefold()
    assert "swweater" not in brand
    garment = str(
        plan["filters"].get("garment")
        or plan["filters"].get("category")
        or plan["filters"].get("garment_type")
        or ""
    ).casefold()
    assert "sweater" in garment


def test_a17_cross_catalog_sweater_not_fake_brands_or_group_by():
    """A17 report English must not become brand=shows|like|careful|… or group_by."""

    q = (
        "Compare how 'sweater' shows up in PLMEX versus how sweater-like product "
        "types appear in e-commerce. Be careful: category is a false friend across "
        "graphs — explain using garment_type / subfamily roles rather than merging "
        "category values blindly, and give separate counts per source."
    )
    plan = resolve_catalog_plan(q, None, carry=False)
    brand = str(plan["filters"].get("brand") or "").casefold()
    for junk in (
        "shows",
        "like",
        "careful",
        "false",
        "friend",
        "merging",
        "blindly",
        "separate",
        "counts",
    ):
        assert junk not in brand.split("|")
        assert junk not in brand
    assert "sweater" in str(plan["filters"].get("garment") or "").casefold()
    assert plan.get("group_by") in (None, "")


def test_sweater_token_in_cardigans_sweaters_label():
    from fashion_brain.planning.filter_validation import (
        _contains_family_match,
        _token_in_label,
    )

    assert _token_in_label("Cardigans Sweaters", "sweater") is True
    assert _contains_family_match("sweater", ["Cardigans Sweaters", "Sweaters"]) is True


def test_cross_catalog_sweater_compare_uses_kg_stats_not_cypher():
    """A17 must answer from per-source kg_stats — not hang in Text2Cypher."""

    from fashion_brain.agents.contracts import Intention
    from fashion_brain.ask import _is_grammar_party_name
    from fashion_brain.orchestration.catalog_pipeline import (
        _normalize_cross_catalog_type_intention,
        _try_cross_catalog_type_compare,
    )

    class _P:
        class settings:
            generated = None

    q = (
        "Compare how sweater shows up in PLMEX versus how sweater-like product "
        "types appear in e-commerce. Give separate counts per source."
    )
    # Agent pollution seen in prod: brand=thinking, subfamily=sweater, group_by=subfamily.
    assert _is_grammar_party_name("thinking") is True
    intention = Intention(
        intent="analysis",
        filters={
            "subfamily": "sweater",
            "brand": "thinking",
            "buyer": "thinking",
        },
        group_by="subfamily",
        metrics=[],
    )
    _normalize_cross_catalog_type_intention(
        intention, q, ["plmex_products", "ecom_products"]
    )
    assert intention.filters.get("garment") == "sweater"
    assert "subfamily" not in intention.filters
    assert "brand" not in intention.filters
    assert intention.group_by in (None, "")
    hit = _try_cross_catalog_type_compare(
        _P(),
        question=q,
        intention=intention,
        product_ids=["plmex_products", "ecom_products"],
    )
    assert hit is not None
    text = hit.text.casefold()
    assert "false friend" in text
    assert "10,996" in hit.text or "10996" in hit.text.replace(",", "")
    assert "plmex_products" in text and "ecom_products" in text
    assert "subfamily" in text or "garment_type" in text or "cardigan" in text
    # Ecom exact "sweaters" is sparse — must clarify compounds / synonyms.
    assert "almost unused" in text or "closest labels that include" in text
    assert "cardigan" in text or "jumper" in text
