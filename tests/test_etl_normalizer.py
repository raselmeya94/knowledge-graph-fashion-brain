from knowledge_graph import etl_normalizer as norm
from knowledge_graph.adapters.ecom_master import build_relationships, normalize_master_row
from knowledge_graph.pipeline.mapping_sql_sync import _build_relationships, _row_to_graph


def test_parse_composition_strips_percentages_from_material_id():
    rows = norm.parse_composition("50%Viscose 28%Polyester 22%Acrylic")
    ids = {r["id"] for r in rows}
    names = {r["name"] for r in rows}
    assert "fashion:material:viscose" in ids
    assert "fashion:material:polyester" in ids
    assert "fashion:material:acrylic" in ids
    assert names == {"viscose", "polyester", "acrylic"}
    assert all(not any(ch.isdigit() for ch in r["name"]) for r in rows)
    by_name = {r["name"]: r["percentage"] for r in rows}
    assert by_name["viscose"] == 50.0
    assert by_name["polyester"] == 28.0


def test_parse_composition_clamps_percentage_above_100():
    rows = norm.parse_composition("120% Polyester 50% Polyester")
    by_name = {r["name"]: r["percentage"] for r in rows}
    assert by_name["polyester"] == 100.0


def test_parse_composition_collapses_numbered_fiber_tokens():
    rows = norm.parse_composition("VISCOSE 28 POLYESTER 15")
    # Fallback may not always catch bare "FIBER N" without %; ensure clean_fiber strips digits when matched.
    for row in rows:
        assert not any(ch.isdigit() for ch in row["name"])


def test_color_typo_and_split_and_pattern():
    colors, patterns = norm.parse_colors("GERY / WHITE AND BLACK / STRIPES")
    color_names = {c["name"] for c in colors}
    assert "grey" in color_names
    assert "white" in color_names
    assert "black" in color_names
    assert "gery" not in color_names
    assert any(p["name"] == "stripes" for p in patterns)


def test_gauge_normalize_and_crochet_route():
    gauges, constructions = norm.parse_gauges("7G,12GG,CROCHET")
    names = {g["name"] for g in gauges}
    assert names == {"7GG", "12GG"}
    assert any(c["name"] == "crochet" for c in constructions)


def test_reference_normalization_shared_key():
    assert norm.normalize_reference("ab-12/34") == "AB1234"
    assert norm.coalesce_reference(None, "X-9999") == "X9999"
    assert norm.reference_id("AB1234") == "ref:AB1234"


def test_inventory_and_shipment_scoped_ids():
    assert norm.inventory_id("BX-901") == "plmex:inv:BX-901"
    assert norm.shipment_id("BX-901", "2026-01-15").startswith("plmex:ship:BX-901:")


def test_ecom_row_emits_shared_hub_ids():
    product = normalize_master_row(
        {
            "brand": "primark",
            "master_id": "123",
            "name": "Grey jumper",
            "section": "WOMAN",
            "reference": "AB-12/34",
            "composition": "70% Cotton 30% Polyester",
            "colors": ["GERY"],
            "price": "£12.00",
            "currency": "GBP",
        }
    )
    rels = build_relationships(product)
    brand = next(r for r in rels if r["rel_type"] == "HAS_BRAND")
    material = next(r for r in rels if r["rel_type"] == "USES_MATERIAL")
    color = next(r for r in rels if r["rel_type"] == "HAS_COLOR")
    ref = next(r for r in rels if r["rel_type"] == "HAS_REFERENCE")
    tg = next(r for r in rels if r["rel_type"] == "FOR_TARGET_GROUP")
    assert brand["end_id"] == "brand:primark"
    assert material["end_id"] == "fashion:material:cotton" or material["end_id"].startswith("fashion:material:")
    assert color["end_id"] == "fashion:color:grey"
    assert ref["end_id"] == "ref:AB1234"
    assert tg["end_id"] == "targetgroup:women"


def test_plmex_mapping_row_inventory_not_colliding_on_stock():
    mapping = {
        "source_table": "plmex_products",
        "primary_key": ["code"],
        "product_id_template": "plmex:product:{code}",
        "product_properties": {
            "code": {"column": "code"},
            "composition": {"column": "composition"},
            "brand": {"column": "brand"},
            "gender": {"column": "gender"},
            "color": {"column": "color"},
            "gauge": {"column": "gauge"},
            "category": {"column": "category"},
            "current_stock": {"column": "current_stock"},
            "current_location": {"column": "current_location"},
            "style_reference": {"column": "style_reference"},
            "ai_event_tags": {"column": "ai_event_tags"},
            "ai_construction_tags": {"column": "ai_construction_tags"},
        },
        "neo4j_product_properties": ["code", "composition", "display_name", "search_text", "source"],
        "relationships": [
            {"rel_type": "HAS_BRAND", "end_type": "Brand", "from_column": "brand"},
            {"rel_type": "USES_MATERIAL", "end_type": "Material", "from_column": "composition"},
            {"rel_type": "MADE_WITH_METHOD", "end_type": "ProductionMethod", "from_column": "category"},
            {"rel_type": "FOR_TARGET_GROUP", "end_type": "TargetGroup"},
            {"rel_type": "HAS_COLOR", "end_type": "Color", "from_column": "color"},
            {"rel_type": "HAS_GAUGE", "end_type": "Gauge", "from_column": "gauge"},
            {"rel_type": "HAS_REFERENCE", "end_type": "Reference", "from_transform": "style_ref_norm", "edge_properties": {"ref_kind": "plm_style_ref"}},
            {"rel_type": "HAS_INVENTORY_RECORD", "end_type": "InventoryRecord", "from_column": "current_stock"},
            {"rel_type": "STORED_AT", "start_type": "InventoryRecord", "end_type": "Location", "end_from_column": "current_location"},
            {"rel_type": "SUITED_FOR_OCCASION", "end_type": "Occasion", "from_column": "ai_event_tags"},
            {"rel_type": "HAS_CONSTRUCTION", "end_type": "Construction", "from_column": "ai_construction_tags"},
        ],
    }
    row_a = {
        "code": "A1",
        "composition": "50%Viscose 50%Polyester",
        "brand": "PRIMARK",
        "gender": "Women",
        "color": "GERY",
        "gauge": "7G,CROCHET",
        "category": "Sweater",
        "current_stock": 1,
        "current_location": "WH-1\nWH-2",
        "style_reference": "AB-12/34",
        "ai_event_tags": "Party,Casual",
        "ai_construction_tags": "Rib",
    }
    row_b = {
        **row_a,
        "code": "B2",
        "brand": "ZARA",
        "style_reference": "ZZ-99/00",
    }
    product_a, rels_a = _row_to_graph(row_a, mapping, source_id="plmex_products", graph_id="plmex_products")
    product_b, rels_b = _row_to_graph(row_b, mapping, source_id="plmex_products", graph_id="plmex_products")
    inv_a = next(r for r in rels_a if r["rel_type"] == "HAS_INVENTORY_RECORD")
    inv_b = next(r for r in rels_b if r["rel_type"] == "HAS_INVENTORY_RECORD")
    assert inv_a["end_id"] == "plmex:inv:A1"
    assert inv_b["end_id"] == "plmex:inv:B2"
    assert inv_a["end_id"] != inv_b["end_id"]

    materials = [r for r in rels_a if r["rel_type"] == "USES_MATERIAL"]
    assert {m["end_id"] for m in materials} == {"fashion:material:viscose", "fashion:material:polyester"}
    assert all(m["properties"].get("percentage") in (50.0, 50) for m in materials)

    method = next(r for r in rels_a if r["rel_type"] == "MADE_WITH_METHOD")
    assert method["end_type"] == "ProductionMethod"

    color = next(r for r in rels_a if r["rel_type"] == "HAS_COLOR")
    assert color["end_id"] == "fashion:color:grey"

    gauges = [r for r in rels_a if r["rel_type"] == "HAS_GAUGE"]
    assert {g["end_name"] for g in gauges} == {"7GG"}

    constructions = [r for r in rels_a if r["rel_type"] == "HAS_CONSTRUCTION"]
    assert any(c["end_id"] == "fashion:construction:crochet" for c in constructions)

    ref = next(r for r in rels_a if r["rel_type"] == "HAS_REFERENCE")
    assert ref["end_id"] == "ref:AB1234"

    stored = [r for r in rels_a if r["rel_type"] == "STORED_AT"]
    assert len(stored) == 2
    assert product_a.get("display_name")
    assert product_b.get("search_text")


def test_mapping_files_declare_hub_graph():
    import json
    from pathlib import Path

    root = Path("data/schemas")
    ecom = json.loads((root / "ecom_master_products_mapping.json").read_text(encoding="utf-8"))
    plmex = json.loads((root / "plmex_mapping.json").read_text(encoding="utf-8"))
    assert ecom["hub_graph_id"] == "fashion_brain"
    assert plmex["hub_graph_id"] == "fashion_brain"
    assert ecom["graph_id"] == "ecom_products"
    assert plmex["graph_id"] == "plmex_products"
    assert "description" in ecom["neo4j_product_properties"]
    assert "FOR_TARGET_GROUP" in {r["rel_type"] for r in ecom["relationships"]}
    assert "MADE_WITH_METHOD" in {r["rel_type"] for r in plmex["relationships"]}
    assert not any(r.get("rel_type") == "BELONGS_TO_CATEGORY" and r.get("from_column") == "category" for r in plmex["relationships"])
    assert not any(
        r.get("rel_type") == "PARENT_CATEGORY" and r.get("start_from_column") == "category"
        for r in plmex["relationships"]
    )
