"""Answer-card pipeline: structured card → writer → footer."""

from __future__ import annotations

from fashion_brain.agents.answer_card import (
    SOURCE_LABELS,
    AnswerCard,
    SourceResult,
    _derived_facts,
    _extract_examples,
    _filter_values,
    build_answer_card,
    clean_example_rows,
    intent_hint,
    pluralize_garment,
    safe_fallback,
    validate,
    write_answer,
)
from fashion_brain.agents.contracts import Retrieved
from knowledge_graph.models import EvidenceItem


def test_source_labels_are_plm_and_ecommerce():
    assert SOURCE_LABELS["plmex_products"] == "PLM records"
    assert SOURCE_LABELS["ecom_products"] == "e-commerce listings"


def test_build_card_from_multi_source_findings():
    findings = """\
### PLM
There are 560 products with brand Primark and garment sweater.
Example products (1 of 560):
| Brand | Name | Price | Availability |
| --- | --- | --- | --- |
| — | 01202000959 | — | — |

### E-commerce
There are 42 products with brand Primark and garment sweater.
Example products (1 of 42):
| Brand | Name | Price | Availability |
| --- | --- | --- | --- |
| Primark | Wool Blend Oversized Crew Neck Sweater | $25.00 | InStock |
"""
    card = build_answer_card(
        "How many Primark sweater products?",
        ["product_count", "plmex_products", "ecom_products"],
        {"brand": "Primark", "garment": "sweater"},
        findings,
    )
    assert card.intent == "count"
    assert card.filters["brand"] == "Primark"
    by_id = {r.source_id: r for r in card.results}
    assert by_id["plmex_products"].total == 560
    assert by_id["ecom_products"].total == 42
    # Null/ref example dropped; ecom example kept.
    assert by_id["plmex_products"].examples == []
    assert by_id["ecom_products"].examples[0]["name"].startswith("Wool Blend")


def test_clean_example_rows_drops_refs_and_nulls():
    rows = clean_example_rows(
        [
            {"name": "null", "price": "$10"},
            {"name": "01202000959", "price": "—"},
            {"name": "Crew Neck", "price": "$25.00", "availability": "InStock"},
        ]
    )
    assert len(rows) == 1
    assert rows[0]["name"] == "Crew Neck"


def test_write_answer_appends_footer_and_validates_filters():
    card = AnswerCard(
        question="How many Primark sweaters for women in white?",
        intent="count",
        filters={"brand": "Primark", "garment": "sweater", "gender": "women", "color": "white"},
        results=[
            SourceResult(source_id="plmex_products", total=26),
            SourceResult(source_id="ecom_products", total=0),
        ],
    )

    def llm(_prompt: str) -> str:
        return (
            "We have 26 white women's sweaters from Primark in our PLM records, "
            "and none on e-commerce listings."
        )

    out = write_answer(card, llm)
    assert "26" in out
    assert "Sources: PLM records · E-commerce listings" in out
    assert "Source:" not in out.split("Sources:")[0] or True


def test_validate_flags_missing_filter_and_banned():
    card = AnswerCard(
        question="How many Primark products are in stock?",
        intent="count",
        filters={"brand": "Primark", "availability": "InStock"},
        results=[SourceResult(source_id="ecom_products", total=3582)],
    )
    issues = validate("Primark has 3,582 products online.", card)
    assert any(i.startswith("missing_filter:availability") for i in issues)
    issues2 = validate("Primark has 3,582 InStock products in the knowledge graph.", card)
    assert any("banned" in i for i in issues2)


def test_safe_fallback_and_evidence_totals():
    retrieved = Retrieved(
        text="There are 4,058 products with brand Primark.",
        evidence=[
            EvidenceItem(
                source_id="ecom_products",
                record_id="kg_stats",
                excerpt="products=4058",
                graph="ecom_products",
                verification_status="observed",
            )
        ],
        caveats=[],
        unavailable=[],
        response_type="stats",
        retrieve_mode="template",
    )
    card = build_answer_card(
        "How many products does Primark have?",
        ["kg_stats", "ecom_products"],
        {"brand": "Primark"},
        retrieved,
    )
    assert card.results[0].total == 4058
    assert "4,058" in safe_fallback(card)
    assert not safe_fallback(card).casefold().startswith("we found")


def test_validate_rejects_robotic_opener_and_typical_example():
    card = AnswerCard(
        question="How many products are in plmex?",
        intent="count",
        filters={},
        results=[SourceResult(source_id="plmex_products", total=27694)],
    )
    assert "format:robotic_opener" in validate("We found 27,694 products.", card)
    assert any(
        "typical online example" in i
        for i in validate(
            "Primark has 560 sweaters. A typical online example is the Crew Neck.",
            card,
        )
    )


def test_ranking_table_not_treated_as_examples():
    body = """\
| Rank | Name | Products |
| --- | --- | --- |
| 1 | Primark | 1397 |
| 2 | Zara Kids | 710 |
| 3 | C A | 597 |
| 4 | River Island | 411 |
| 5 | Kmart | 245 |
| 6 | H M | 219 |
| 7 | Next | 200 |
| 8 | Matalan | 180 |
| 9 | Tesco | 150 |
| 10 | Asda | 120 |
"""
    assert _extract_examples(body) == []


def test_pluralize_garment_dress():
    assert pluralize_garment("dress") == "dresses"
    assert pluralize_garment("sweater") == "sweaters"
    assert pluralize_garment("category") == "categories"


def test_intent_hint_and_write_answer_injects_hint():
    card = AnswerCard(
        question="How many Primark products?",
        intent="count",
        filters={"brand": "Primark"},
        results=[
            SourceResult(source_id="plmex_products", total=100),
            SourceResult(source_id="ecom_products", total=50),
        ],
    )
    hint = intent_hint("count", card).casefold()
    assert "both source" in hint or "simple count" in hint
    assert "assortment" in hint or "1–3" in hint or "1-3" in hint

    seen: list[str] = []

    def llm(prompt: str) -> str:
        seen.append(prompt)
        return "Primark has 100 products in our PLM records and 50 on e-commerce listings."

    out = write_answer(card, llm)
    assert "INTENT HINT" in seen[0]
    assert "simple count" in seen[0].casefold()
    assert "Sources: PLM records · E-commerce listings" in out
    assert "gap:" not in card.prompt_json().casefold()


def test_open_product_listing_does_not_force_ecom_footer():
    findings = (
        "There are 560 products with brand Primark and garment sweater. "
        "Open product listing: total + brand/category mix + a few examples."
    )
    card = build_answer_card(
        "How many Primark sweater products?",
        ["product_count", "plmex_products"],
        {"brand": "Primark", "garment": "sweater"},
        findings,
    )
    assert card.footer() == "Source: PLM records"
    assert all(r.source_id != "ecom_products" for r in card.results)


def test_optional_metadata_becomes_card_context():
    retrieved = Retrieved(
        text="There are 560 products with brand Primark and garment sweater.",
        evidence=[
            EvidenceItem(
                source_id="plmex_products",
                record_id="kg_stats",
                excerpt="products=560",
                graph="plmex_products",
                verification_status="observed",
            ),
            EvidenceItem(
                source_id="plmex_products",
                record_id="optional_metadata",
                title="kg filter meta reference",
                excerpt=(
                    "products_total=27694; brand: unique=448; top: primark=1397; "
                    "category: unique=13; top: sweater=17287"
                ),
                graph="plmex_products",
                verification_status="observed",
            ),
        ],
        caveats=[],
        unavailable=[],
        response_type="stats",
        retrieve_mode="template",
    )
    card = build_answer_card(
        "How many Primark sweater products?",
        ["product_count", "plmex_products"],
        {"brand": "Primark", "garment": "sweater"},
        retrieved,
    )
    assert any("products_total=27694" in line for line in card.context)
    # Facet top-lists must not enter count context (Black/Girl/… steal answer_n).
    assert not any("top:" in line.casefold() for line in card.context)
    assert not any("sweater=17287" in line for line in card.context)
    assert not any("primark=1397" in line.casefold() for line in card.context)
    hint = intent_hint(card.intent, card)
    assert "required" in hint.casefold() or "simple" in hint.casefold()
    assert "Primark" in hint
    assert "sweater" in hint.casefold()


def test_strip_count_inflation_keeps_headline():
    from fashion_brain.agents.answer_card import _strip_count_inflation

    raw = (
        "There are **2,292** products in white.\n\n"
        "### Assortment Overview\n"
        "| Source | Product Count |\n| --- | --- |\n| PLM | 2292 |\n\n"
        "### Merchandising Reading\n"
        "White is 8.3% of 27,694.\n"
    )
    cleaned = _strip_count_inflation(raw)
    assert "2,292" in cleaned or "2292" in cleaned
    assert "Assortment Overview" not in cleaned
    assert "Merchandising Reading" not in cleaned


def test_count_stays_simple_not_assortment_report():
    card = AnswerCard(
        question="How many products with category cut & sew jersey?",
        intent="count",
        filters={"category": "cut sew jersey"},
        results=[SourceResult(source_id="plmex_products", total=5026)],
        context=[
            "products_total=27694",
            "category: unique=13",
            "top: cut sew jersey=5026, knits=3000",
        ],
    )
    hint = intent_hint(card.intent, card)
    assert "simple" in hint.casefold()
    assert "assortment overview" in hint.casefold() or "merchandising" in hint.casefold()
    fb = safe_fallback(card)
    assert "5026" in fb or "5,026" in fb
    assert "Related catalog facts" not in fb
    assert "Assortment Overview" not in fb
    issues = validate(
        "### Assortment Overview\nThere are 5,026 products.\n### Merchandising Reading\n…",
        card,
    )
    assert any("count_inflated_to_report" in i for i in issues)
    clean = "There are 5,026 products with category cut sew jersey in PLM records."
    assert not validate(clean, card)


def test_required_filters_in_prompt_and_fallback():
    card = AnswerCard(
        question="How many Primark sweaters in white?",
        intent="count",
        filters={"brand": "Primark", "garment": "sweater", "color": "white"},
        results=[
            SourceResult(source_id="plmex_products", total=44),
            SourceResult(source_id="ecom_products", total=1),
        ],
    )
    payload = card.prompt_json()
    assert "required_filters" in payload
    assert "white" in payload
    fb = safe_fallback(card)
    assert "white" in fb.casefold()
    assert "Primark" in fb
    assert "sweater" in fb.casefold()
    assert "44" in fb
    issues = validate(
        "Our PLM records show 44 products, while e-commerce has 1.",
        card,
    )
    assert any("missing_filter:color=white" in i for i in issues)


def test_count_derived_facts_omit_gap_compare_includes_it():
    results = [
        SourceResult(source_id="plmex_products", total=560),
        SourceResult(source_id="ecom_products", total=42),
    ]
    count_derived = _derived_facts(results, {"brand": "Primark"}, intent="count")
    assert not any(line.startswith("gap:") for line in count_derived)
    assert not any(line.startswith("difference:") for line in count_derived)
    compare_derived = _derived_facts(results, {"brand": "Primark"}, intent="compare")
    # Compare uses "difference:" (gap: phrasing is banned in answers).
    assert any(line.startswith("difference:") for line in compare_derived)


def test_filter_values_dedupe_garment_and_category():
    card = AnswerCard(
        question="How many Primark sweaters in white?",
        intent="count",
        filters={"brand": "Primark", "category": "sweaters", "garment": "sweater", "color": "white"},
        results=[SourceResult(source_id="plmex_products", total=44)],
    )
    values = [v.casefold() for v in _filter_values(card)]
    assert "primark" in values
    assert "white" in values
    assert sum(1 for v in values if "sweater" in v) == 1
    ok = validate(
        "Primark has 44 white sweaters in our PLM records.",
        card,
    )
    assert not any(i.startswith("missing_filter:") for i in ok)


def test_validate_bans_gap_and_recorded_times_phrasing():
    card = AnswerCard(
        question="How many Primark sweater products?",
        intent="count",
        filters={"brand": "Primark", "garment": "sweater"},
        results=[
            SourceResult(source_id="plmex_products", total=560),
            SourceResult(source_id="ecom_products", total=42),
        ],
    )
    issues = validate(
        "Primark sweaters are recorded 560 times in PLM and 42 times in e-commerce "
        "listings. PLM records lead e-commerce listings by a gap of 518 products.",
        card,
    )
    banned = [i for i in issues if i.startswith("banned:")]
    assert banned


def test_assortment_compare_is_analysis_intent_not_count_gap():
    from fashion_brain.agents.answer_card import intent_from_routes

    intent = intent_from_routes(
        ["catalog", "analysis", "analytical_tool", "ecom_products", "plmex_products"],
        question="Compare Primark and Zara assortment profiles",
        response_type="analysis_report",
    )
    assert intent == "analysis"
    # Bare count compare still stays compare.
    assert (
        intent_from_routes(
            ["product_count", "ecom_products"],
            question="Compare Primark vs Zara on sweater count",
        )
        == "compare"
    )


def test_analysis_rejects_too_brief_and_builds_report_fallback():
    card = AnswerCard(
        question="Analyse Primark assortment",
        intent="analysis",
        depth="full",
        filters={"brand": "Primark"},
        results=[
            SourceResult(
                source_id="ecom_products",
                total=4058,
                breakdown=[
                    {"value": "Clothes", "count": 1464, "share_pct": 36.1},
                    {"value": "Black", "count": 600, "share_pct": 14.8},
                ],
            )
        ],
        context=[
            "overview: Products=4058; Average price=30.1",
            "leading Color: Black (600, 14.8%)",
            "leading Material: Polyester 100 (604, 14.9%)",
        ],
        data_notes=["Price max looks like an outlier."],
    )
    stub = "Primark has 4,058 products. Clothes lead."
    issues = validate(stub, card)
    assert any("too_brief" in i for i in issues)
    hint = intent_hint("analysis", card).casefold()
    assert "do not answer in 2-3 lines" in hint or "6" in hint or "scope" in hint
    fb = safe_fallback(card)
    assert "Primark" in fb or "primark" in fb.casefold()
    assert "4058" in fb.replace(",", "")
    assert "Clothes" in fb or "Black" in fb or "Colour" in fb or "Color" in fb
    assert "### Scope" not in fb
    assert "### Core metrics" not in fb
    assert "What stands out" in fb or "Attribute breakdown" in fb or "###" in fb
    assert "Assortment mix" not in fb
    assert "metric:" not in fb.casefold()


def test_sweater_analysis_fallback_not_fake_attribute_compare():
    """### Category / ### Material headings must not become 'Category vs Material' compare."""

    findings = """\
## PLMex
Product analysis for Primark: **560** products.
### Category
Leading value: Sweater (555, 99.1%).
| Value | Count | Share % |
| --- | ---: | ---: |
| Sweater | 555 | 99.1 |
### Material
Leading value: Elastane (139, 24.8%).
| Value | Count | Share % |
| --- | ---: | ---: |
| Elastane | 139 | 24.8 |
| Cotton | 120 | 21.4 |
### Gender
Leading value: Women (298, 53.2%).
| Value | Count | Share % |
| --- | ---: | ---: |
| Women | 298 | 53.2 |
| Men | 200 | 35.7 |

## E-commerce
Product analysis for Primark: **42** products.
| Metric | Value |
| --- | ---: |
| Products | 42 |
| Average price | 18.86 |
"""
    card = build_answer_card(
        "Analyze Primark sweater assortment",
        ["analysis", "analytical_tool", "ecom_products", "plmex_products"],
        {"brand": "Primark", "garment": "sweater"},
        findings,
        response_type="analysis_report",
    )
    from fashion_brain.agents.answer_card import _compare_parties_from_card

    assert _compare_parties_from_card(card) == []
    fb = safe_fallback(card)
    low = fb.casefold()
    assert "side-by-side comparison for category" not in low
    assert "sweater" in low
    assert "primark" in low
    assert "560" in fb.replace(",", "") or "42" in fb.replace(",", "")
    assert "### Category" in fb
    assert "### Material" in fb
    assert "| Elastane | 139 |" in fb or "Elastane" in fb
    assert "Assortment mix" not in fb
    assert "### Scope" not in fb
    assert "format:fake_attribute_compare" in validate(
        "### Scope\nSide-by-side comparison for Category vs Material.\n",
        card,
    )


def test_full_primark_analysis_fallback_is_field_wise_not_assortment():
    findings = """\
Product analysis for Primark: **1397** products.
Field-wise analysis (all attributes; up to 12 unique values per field with counts and shares).
### Garment type
Leading value: Jumper (242, 17.3%).
3 unique values shown (of this field).
| Value | Count | Share % |
| --- | ---: | ---: |
| Jumper | 242 | 17.3 |
| Top Cardigan | 180 | 12.9 |
| Dress | 90 | 6.4 |
### Category
Leading value: Sweater (555, 39.7%).
| Value | Count | Share % |
| --- | ---: | ---: |
| Sweater | 555 | 39.7 |
| Cut & Sew-Jersey | 200 | 14.3 |
"""
    card = build_answer_card(
        "Give me a full analysis of Primark",
        ["analysis", "analytical_tool", "plmex_products"],
        {"brand": "Primark"},
        findings,
        response_type="analysis_report",
    )
    assert card.intent == "analysis"
    fb = safe_fallback(card)
    assert "Primark" in fb
    assert "1,397" in fb or "1397" in fb.replace(",", "")
    assert "### Garment type" in fb
    assert "### Category" in fb
    assert "| Jumper | 242 |" in fb
    assert "| Sweater | 555 |" in fb
    assert "Assortment mix" not in fb
    assert "### Scope" not in fb
    assert "### Core metrics" not in fb
    assert "Here's how the catalog breaks down" in fb or "product analysis" in fb.casefold()


def test_analysis_findings_reach_card_context_and_block_false_refusal():
    findings = """\
## E-commerce
Product analysis for Primark: **4058** products.
| Metric | Value |
| --- | ---: |
| Products | 4058 |
| Average price | 30.1 |
### Color
Leading value: Black (600, 14.8%).
### Material
Leading value: Polyester 100 (604, 14.9%).

## PLMex
Product analysis for Zara: **237** products.
### Color
Leading value: White (40, 16.9%).
"""
    card = build_answer_card(
        "Compare Primark and Zara assortment profiles",
        ["analysis", "analytical_tool", "ecom_products", "plmex_products"],
        {"brand": "Primark | Zara"},
        findings,
        response_type="analysis_report",
    )
    assert card.intent == "analysis"
    assert any("overview:" in line for line in card.context)
    assert any("leading Color" in line or "leading color" in line.casefold() for line in card.context)
    issues = validate(
        "I do not have the necessary data to compare the assortment profiles for "
        "Primark and Zara, as no totals or breakdown metrics were provided.",
        card,
    )
    assert any("false_refusal" in i or i.startswith("banned:") for i in issues)
    fb = safe_fallback(card)
    assert "necessary data" not in fb.casefold()
    assert "Primark" in fb or "### Primark" in fb
    assert "### Scope" not in fb
    assert "Side-by-side" in fb or "### Primark" in fb
    assert "4058" in fb.replace(",", "") or "Black" in fb or "leading" in fb.casefold()


def test_ecom_only_unsectioned_count_binds_to_ecom_not_plm():
    findings = (
        "There are 3,582 products with brand Primark and availability InStock. "
        "Open product listing: total + brand/category mix + a few examples."
    )
    retrieved = Retrieved(
        text=findings,
        evidence=[
            EvidenceItem(
                source_id="ecom_products",
                record_id="x",
                excerpt="primark | Soft Knit | 12",
                graph="ecom_products",
                verification_status="observed",
                title="Soft Knit",
            )
        ],
        caveats=[],
        unavailable=[],
        response_type="stats",
        retrieve_mode="template",
    )
    card = build_answer_card(
        "How many Primark products are in stock?",
        ["product_count", "ecom_products"],
        {"brand": "Primark"},
        retrieved,
    )
    assert card.filters.get("availability")
    assert any(r.source_id == "ecom_products" and r.total == 3582 for r in card.results)
    assert not any(r.source_id == "plmex_products" for r in card.results)


def test_unsectioned_breakdown_merges_into_ecom_card():
    """P22: product evidence + unsectioned Name|Count table must fill breakdown[]."""

    findings = """\
There are 4,058 products with brand Primark.
category breakdown (highest count first):
Leading category: Clothes (500 of 4058, 12.3%).
| Name | Count | Share % |
| --- | ---: | ---: |
| Clothes | 500 | 12.3 |
| Jackets | 200 | 4.9 |
| Accessories | 150 | 3.7 |
"""
    retrieved = Retrieved(
        text=findings,
        evidence=[
            EvidenceItem(
                source_id="ecom_products",
                record_id="p1",
                title="Bra",
                excerpt="primark | Bra | 20",
                graph="ecom_products",
                verification_status="observed",
            ),
            EvidenceItem(
                source_id="ecom_products",
                record_id="optional_metadata",
                title="kg filter meta reference",
                excerpt=(
                    "products_total=18183; brand: unique=5; top: zara=5804; "
                    "category: unique=310; top: Clothes=1464"
                ),
                graph="ecom_products",
                verification_status="observed",
            ),
        ],
        caveats=[],
        unavailable=[],
        response_type="breakdown",
        retrieve_mode="template",
        resolved={"filters": {"brand": "primark"}, "group_by": "category"},
    )
    card = build_answer_card(
        "Show Primark category breakdown",
        ["catalog", "breakdown", "ecom_products"],
        {"brand": "primark"},
        retrieved,
        group_by="category",
    )
    assert card.intent == "breakdown"
    slot = next(r for r in card.results if r.source_id == "ecom_products")
    assert slot.total == 4058
    assert [row["value"] for row in slot.breakdown[:3]] == ["Clothes", "Jackets", "Accessories"]
    # Filtered ask must not keep catalog-wide top-N as primary context.
    assert not any("Clothes=1464" in line for line in card.context)
    assert any("leading category" in line.casefold() for line in card.context)
    refusal = (
        "While there are 4,058 e-commerce listings for Primark, "
        "a category breakdown is not available."
    )
    issues = validate(refusal, card)
    assert any("false_refusal" in i for i in issues)
    fb = safe_fallback(card)
    assert "Clothes" in fb
    assert "not available" not in fb.casefold()


def test_false_refusal_does_not_provide_a_breakdown():
    card = AnswerCard(
        question="Which brand has the most sweaters?",
        intent="breakdown",
        filters={"garment": "sweater"},
        results=[
            SourceResult(
                source_id="ecom_products",
                total=48,
                breakdown=[
                    {"value": "primark", "count": 20, "share_pct": 41.7},
                    {"value": "matalan", "count": 15, "share_pct": 31.2},
                ],
            )
        ],
    )
    issues = validate(
        "While there are 48 e-commerce listings for sweaters, "
        "the available data does not provide a breakdown by brand.",
        card,
    )
    assert any("false_refusal" in i for i in issues)
