"""Specification sheet: requirements → analyze ALL fields → decide → buyer brief."""

from __future__ import annotations

from types import SimpleNamespace

from knowledge_graph.access import Principal
from knowledge_graph.retrieval import ecom as ecom_mod


INTERNAL = Principal(subject="test", clearance="internal")


def test_specification_analyzes_required_attributes_then_picks_best(monkeypatch):
    lines: list[str] = []
    evidence: list = []

    monkeypatch.setattr(ecom_mod, "clean_product_filters", lambda filters: dict(filters or {}))
    monkeypatch.setattr(ecom_mod, "cypher_count_products", lambda *_a, **_k: 10)
    monkeypatch.setattr(
        ecom_mod,
        "_specification_fields_for_graph",
        lambda *_a, **_k: ["category", "material", "color", "gender"],
    )
    monkeypatch.setattr(ecom_mod, "_graph_label", lambda *_a, **_k: "PLMEX")

    groups = {
        "category": [("Sweater", 6), ("Cardigan", 4)],
        "material": [("Wool", 7), ("Cotton", 3)],
        "color": [("Brown", 5), ("Black", 5)],
        "gender": [("Women", 8), ("Men", 2)],
    }

    def fake_group_by(_platform, _graph_id, field, _filters):
        return list(groups.get(field) or [])

    monkeypatch.setattr(ecom_mod, "cypher_group_by", fake_group_by)

    ecom_mod._append_neo4j_specification_for_graph(
        SimpleNamespace(),
        INTERNAL,
        "plmex_products",
        {"buyer": "zara", "season_name": "winter"},
        lines,
        evidence,
    )
    text = "\n".join(lines)
    assert "## 1. User Requirements" in text
    assert "Buyer: zara" in text or "- **Buyer:** zara" in text
    assert "Season: winter" in text or "- **Season:** winter" in text
    assert "### Attribute Analysis" in text
    assert "| Attribute | Leading Value | Count | Alternative Values |" in text
    assert "| Category | Sweater |" in text
    assert "| Material | Wool |" in text
    assert "| Gender | Women |" in text
    assert "## 4. Final Product Specification" in text
    assert "Proposed new product specification" in text
    assert "Pending PLMEX creation" in text
    assert "```json" in text
    assert '"category": "Sweater"' in text
    assert '"material": "Wool"' in text
    assert '"gender": "Women"' in text
    assert "Product Specification Decision Engine:" not in text
    assert evidence and evidence[0].record_id == "specification"
    assert "proposed_new=true" in evidence[0].excerpt
    assert "decision_engine=true" in evidence[0].excerpt


def test_specification_sheet_lists_every_required_field_even_when_missing(monkeypatch):
    lines: list[str] = []
    evidence: list = []

    monkeypatch.setattr(ecom_mod, "clean_product_filters", lambda filters: dict(filters or {}))
    monkeypatch.setattr(ecom_mod, "cypher_count_products", lambda *_a, **_k: 5)
    monkeypatch.setattr(
        ecom_mod,
        "_specification_fields_for_graph",
        lambda *_a, **_k: ["category", "gauge", "gsm"],
    )
    monkeypatch.setattr(ecom_mod, "_graph_label", lambda *_a, **_k: "PLMEX")
    monkeypatch.setattr(
        ecom_mod,
        "cypher_group_by",
        lambda _p, _g, field, _f: [("Sweater", 5)] if field == "category" else [],
    )

    ecom_mod._append_neo4j_specification_for_graph(
        SimpleNamespace(),
        INTERNAL,
        "plmex_products",
        {"buyer": "primark"},
        lines,
        evidence,
    )
    text = "\n".join(lines)
    assert "```json" in text
    assert '"category": "Sweater"' in text
    assert '"gauge": "Pending buyer confirmation"' in text
    assert '"gsm": "Pending buyer confirmation"' in text
    assert "Buyer confirmation" in text or "buyer confirmation" in text.casefold()
    assert "proposed_new=true" in evidence[0].excerpt
    assert "decision_engine=true" in evidence[0].excerpt


def test_specification_anchors_garment_hierarchy_before_other_fields(monkeypatch):
    """Independent modes must not mix a rare jacket type with a sweater category."""

    lines: list[str] = []
    evidence: list = []
    counts = {"base": 20, "after_ss": 12, "after_cat": 11, "after_gt": 10}

    def fake_count(_platform, _graph_id, filters):
        if filters.get("garment_type"):
            return counts["after_gt"]
        if filters.get("category"):
            return counts["after_cat"]
        if filters.get("plm_sub_sub_category"):
            return counts["after_ss"]
        return counts["base"]

    monkeypatch.setattr(ecom_mod, "clean_product_filters", lambda filters: dict(filters or {}))
    monkeypatch.setattr(ecom_mod, "cypher_count_products", fake_count)
    monkeypatch.setattr(
        ecom_mod,
        "_specification_fields_for_graph",
        lambda *_a, **_k: [
            "plm_sub_sub_category",
            "category",
            "garment_type",
            "material",
            "color",
        ],
    )
    monkeypatch.setattr(ecom_mod, "_graph_label", lambda *_a, **_k: "PLMEX")

    def fake_group_by(_platform, _graph_id, field, filters):
        if field == "plm_sub_sub_category":
            return [("Pullover Sweater", 12), ("Cardigan Sweater", 8)]
        if field == "category":
            return [("Sweater", 11), ("Jacket", 1)]
        if field == "garment_type":
            if filters.get("category") or filters.get("plm_sub_sub_category"):
                return [("Jumper", 9), ("Jacket Light Weight Bomber", 1)]
            return [("Jacket Light Weight Bomber", 10), ("Jumper", 9)]
        if field == "material":
            return [("Wool", 7), ("Cotton", 3)]
        if field == "color":
            return [("Brown", 6), ("Black", 4)]
        return []

    monkeypatch.setattr(ecom_mod, "cypher_group_by", fake_group_by)

    ecom_mod._append_neo4j_specification_for_graph(
        SimpleNamespace(),
        INTERNAL,
        "plmex_products",
        {"buyer": "zara", "season_name": "winter"},
        lines,
        evidence,
    )
    text = "\n".join(lines)
    assert '"sub_sub_category": "Pullover Sweater"' in text
    assert '"category": "Sweater"' in text
    assert '"garment_type": "Jumper"' in text
    # Alternatives may mention jacket historically; the proposed sheet must not pick it.
    sheet = text.split("```json", 1)[1].split("```", 1)[0]
    assert "Jacket Light Weight Bomber" not in sheet


def test_filter_where_accepts_plm_anchor_facets():
    """Anchoring must not invalidate the Cypher WHERE (was returning None)."""

    params: dict = {}
    where = ecom_mod._filter_where(
        {
            "buyer": "primark",
            "season_name": "winter",
            "plm_sub_sub_category": "pullover sweater",
            "category": "sweater",
            "garment_type": "jumper",
        },
        params,
    )
    assert where is not None
    assert any("HAS_PLM_SUB_SUB_CATEGORY" in clause or "plm_sub_sub_category" in clause for clause in where)
    assert "f_plm_sub_sub_category" in params or "f_garment_type" in params
    assert params.get("buyer") == "primark" or "buyer" in params


def test_failed_anchor_narrow_does_not_block_other_required_fields(monkeypatch):
    """If narrowing on plm_sub_sub_category breaks the filter, keep analyzing the rest."""

    lines: list[str] = []
    evidence: list = []

    def fake_count(_platform, _graph_id, filters):
        if filters.get("plm_sub_sub_category"):
            return None
        return 20

    def fake_group_by(_platform, _graph_id, field, filters):
        if filters.get("plm_sub_sub_category"):
            return []
        if field == "plm_sub_sub_category":
            return [("Pullover Sweater", 12)]
        if field == "gender":
            return [("Women", 15)]
        if field == "color":
            return [("Brown", 8)]
        if field == "composition":
            return [("100% Cotton", 7)]
        return []

    monkeypatch.setattr(ecom_mod, "clean_product_filters", lambda filters: dict(filters or {}))
    monkeypatch.setattr(ecom_mod, "cypher_count_products", fake_count)
    monkeypatch.setattr(ecom_mod, "cypher_group_by", fake_group_by)
    monkeypatch.setattr(
        ecom_mod,
        "_specification_fields_for_graph",
        lambda *_a, **_k: ["plm_sub_sub_category", "gender", "color", "composition"],
    )
    monkeypatch.setattr(ecom_mod, "_graph_label", lambda *_a, **_k: "PLMEX")

    ecom_mod._append_neo4j_specification_for_graph(
        SimpleNamespace(),
        INTERNAL,
        "plmex_products",
        {"buyer": "primark", "season_name": "winter"},
        lines,
        evidence,
    )
    text = "\n".join(lines)
    assert "| Gender | Women |" in text
    assert "| Color | Brown |" in text
    assert "| Composition | 100% Cotton |" in text
    assert '"gender": "Women"' in text
    assert '"colour": "Brown"' in text
    assert "proposed_new=true" in evidence[0].excerpt
    assert "decision_engine=true" in evidence[0].excerpt


def test_plmex_uses_full_predefined_spec_field_order():
    fields = ecom_mod._specification_fields_for_graph(SimpleNamespace(), "plmex_products")
    assert "plm_sub_sub_category" in fields
    assert "composition" in fields
    assert "gender" in fields
    assert "ai_construction_tags" in fields
    assert "gsm" in fields
    assert fields.index("plm_sub_sub_category") < fields.index("composition")
