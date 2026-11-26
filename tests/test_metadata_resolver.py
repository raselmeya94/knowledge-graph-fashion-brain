"""Metadata resolver: repair wrong-field picks; clarify weak understanding."""

from __future__ import annotations

from types import SimpleNamespace

from fashion_brain.planning.metadata_resolver import (
    CORE_METADATA_FIELDS,
    Understanding,
    check_understanding,
    cross_check_important_columns,
    demote_color_words_from_category,
    match_spans,
    repair_filters,
)
from fashion_brain.agents.turn_planner import _pick_clarification_option
from fashion_brain.ask import resolve_catalog_plan
from knowledge_graph.ontology import normalize_name


def _index():
    return {
        "brand": [("Zara", "zara"), ("Primark", "primark")],
        "garment": [("Sweater", "sweater"), ("Jumper", "jumper")],
        "category": [("Clothes", "clothes"), ("Black", "black")],
        "color": [("black", "black"), ("white", "white")],
        "section": [("Women", "women")],
    }


def test_repair_moves_brand_from_wrong_field():
    repaired, unresolved = repair_filters({"category": "Zara", "garment": "jumper"}, _index())
    assert repaired.get("brand") == "Zara"
    assert repaired.get("garment") == "Jumper"
    assert "category" not in repaired or repaired.get("category") != "Zara"
    assert unresolved == []


def test_repair_prefers_color_over_category_for_black():
    repaired, unresolved = repair_filters({"category": "Black"}, _index())
    assert repaired.get("color")
    assert normalize_name(repaired["color"]) == "black"
    assert "category" not in repaired
    assert unresolved == []


def test_repair_keeps_color_black_not_category():
    repaired, unresolved = repair_filters(
        {"color": "black", "brand": "zara", "gender": "women"}, _index()
    )
    assert normalize_name(repaired.get("color") or "") == "black"
    assert "category" not in repaired
    assert unresolved == []


def test_demote_color_words_from_category():
    out = demote_color_words_from_category({"category": "Black", "brand": "zara"}, _index())
    assert normalize_name(out.get("color") or "") == "black"
    assert "category" not in out
    assert out.get("brand") == "zara"


def test_black_zara_women_keyword_plan():
    plan = resolve_catalog_plan(
        "How many black Zara products are there for women?",
        brand_names=["zara", "primark", "matalan", "c-and-a"],
    )
    filters = plan.get("filters") or {}
    assert normalize_name(str(filters.get("color") or "")) == "black"
    assert normalize_name(str(filters.get("brand") or "")) == "zara"
    assert "category" not in filters or normalize_name(str(filters.get("category"))) != "black"
    genderish = normalize_name(str(filters.get("gender") or filters.get("section") or ""))
    assert genderish in {"women", "woman", "female"}


def test_repair_near_miss_spelling():
    repaired, unresolved = repair_filters({"brand": "zarra"}, _index())
    assert repaired.get("brand") == "Zara"
    assert unresolved == []


def test_repair_unknown_value_unresolved():
    repaired, unresolved = repair_filters({"brand": "possible"}, _index())
    assert "brand" not in repaired
    assert any("possible" in u for u in unresolved)


def test_strict_field_never_trusts_missing_index():
    # brand is strict — empty index for brand must not keep invented value.
    repaired, unresolved = repair_filters(
        {"brand": "impossible", "price_max": "50"},
        {"garment": [("Sweater", "sweater")]},
    )
    assert "brand" not in repaired
    assert repaired.get("price_max") == "50"
    assert any("impossible" in u for u in unresolved)


def test_match_spans_finds_zara_jumper():
    spans = match_spans("How many Zara jumpers", _index())
    fields = {s.field for s in spans}
    assert "brand" in fields
    assert "garment" in fields


def test_pick_clarification_option_by_number():
    clar = {
        "options": ["Zara sweaters count", "Primark sweaters count"],
        "interpretations": [
            {"restated_question": "Zara sweaters count", "filters": {"brand": "Zara", "garment": "Sweater"}},
            {"restated_question": "Primark sweaters count", "filters": {"brand": "Primark"}},
        ],
    }
    hit = _pick_clarification_option("1", clar)
    assert hit["filters"]["brand"] == "Zara"
    hit2 = _pick_clarification_option("the second one", clar)
    assert hit2["filters"]["brand"] == "Primark"


class _FakeGraph:
    def find_nodes(self, **_kwargs):
        return []


class _FakeRegistry:
    def __init__(self, graphs):
        self._graphs = graphs

    def list_graphs(self):
        return self._graphs

    def get_graph(self, graph_id):
        for g in self._graphs:
            if g.id == graph_id:
                return g
        raise KeyError(graph_id)


def test_core_fields_include_plm_hierarchy():
    keys = set(CORE_METADATA_FIELDS)
    assert "plm_main_category" in keys
    assert "subcategory" in keys
    assert "plm_sub_sub_category" in keys
    assert "family" in keys
    assert "subfamily" in keys


def test_cross_check_important_columns_across_ecom_and_plmex(monkeypatch):
    """Resolver must inspect every product KG — not a single graph's columns."""
    ecom = SimpleNamespace(id="ecom_products", kind="product")
    plmex = SimpleNamespace(id="plmex_products", kind="product")
    platform = SimpleNamespace(
        kg_registry=_FakeRegistry([ecom, plmex]),
        graph=_FakeGraph(),
        settings=SimpleNamespace(),
    )

    def fake_overview(_platform, graph_id=None):
        if graph_id == "ecom_products":
            return {
                "schema": {
                    "Product": {
                        "attributes": {
                            "brand": "string",
                            "category": "string",
                            "color": "string",
                            "material": "string",
                            "price": "number",
                        }
                    }
                },
                "entities": ["Brand", "GarmentCategory", "Color", "Material"],
            }
        if graph_id == "plmex_products":
            return {
                "schema": {
                    "Product": {
                        "attributes": {
                            "buyer": "string",
                            "garment_type": "string",
                            "family": "string",
                            "season": "string",
                            "material": "string",
                        }
                    }
                },
                "entities": ["Company", "Season", "Material"],
            }
        return {"schema": {}, "entities": []}

    monkeypatch.setattr(
        "knowledge_graph.retrieval.metadata.metadata_overview",
        fake_overview,
    )
    monkeypatch.setattr(
        "fashion_brain.planning.metadata_resolver._product_graph_ids",
        lambda _p: ["ecom_products", "plmex_products"],
    )

    cov = cross_check_important_columns(platform)
    assert set(cov["product_graphs"]) == {"ecom_products", "plmex_products"}
    assert "ecom_products" in cov["field_graphs"].get("brand", [])
    assert "plmex_products" in cov["field_graphs"].get("buyer", [])
    assert "material" in cov["present_on_any"]
    # brand is ecom-side; buyer is plmex-side — each graph reports the other as missing
    assert "buyer" in (cov["missing_important"].get("ecom_products") or [])
    assert "brand" in (cov["missing_important"].get("plmex_products") or [])
