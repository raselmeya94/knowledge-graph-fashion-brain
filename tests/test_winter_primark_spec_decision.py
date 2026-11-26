"""Acceptance: Winter Primark PLMEX product-spec asks must propose a NEW sheet."""

from __future__ import annotations

from types import SimpleNamespace

from fashion_brain.ask import (
    _is_product_details_request,
    _is_suggestion,
    plan_routes,
    resolve_catalog_plan,
)
from knowledge_graph.access import Principal
from knowledge_graph.retrieval import ecom as ecom_mod

INTERNAL = Principal(subject="test", clearance="internal")

WINTER_Q = (
    "Create a detailed Winter season product specification for Primark buyers using PLMEX data"
)


def test_winter_primark_create_spec_routes_to_suggestion_not_product_details():
    assert _is_suggestion(WINTER_Q)
    assert not _is_product_details_request(WINTER_Q)
    assert plan_routes(WINTER_Q) == ["specification"]
    plan = resolve_catalog_plan(WINTER_Q, None, carry=False)
    assert plan.get("suggest") is True
    assert str(plan.get("season_name") or "").casefold() == "winter"
    party = str((plan.get("filters") or {}).get("buyer") or (plan.get("filters") or {}).get("brand") or "")
    assert party.casefold() == "primark"
    assert plan.get("group_by") in (None, "")
    # Still retrieve-existing when there is no create/propose intent.
    assert plan_routes("Give me the full specification for a Primark sweater.") == ["product_details"]


def test_decision_engine_emits_buyer_brief_with_json_sheet(monkeypatch):
    lines: list[str] = []
    evidence: list = []

    monkeypatch.setattr(ecom_mod, "clean_product_filters", lambda filters: dict(filters or {}))
    monkeypatch.setattr(ecom_mod, "cypher_count_products", lambda *_a, **_k: 20)
    monkeypatch.setattr(
        ecom_mod,
        "_specification_fields_for_graph",
        lambda *_a, **_k: ["category", "material", "color", "gender", "ai_event_tags"],
    )
    monkeypatch.setattr(ecom_mod, "_graph_label", lambda *_a, **_k: "PLMEX")

    groups = {
        "category": [("Sweater", 14), ("Cardigan", 6)],
        "material": [("Wool", 12), ("Cotton", 8)],
        "color": [("Brown", 11), ("Black", 9)],
        "gender": [("Women", 16), ("Men", 4)],
        "ai_event_tags": [("Fall", 15), ("Winter", 10)],
    }

    monkeypatch.setattr(
        ecom_mod,
        "cypher_group_by",
        lambda _p, _g, field, _f: list(groups.get(field) or []),
    )

    ecom_mod._append_neo4j_specification_for_graph(
        SimpleNamespace(),
        INTERNAL,
        "plmex_products",
        {"buyer": "primark"},
        lines,
        evidence,
        season_name="winter",
    )
    text = "\n".join(lines)
    assert "## 1. User Requirements" in text
    assert "## 2. Historical Data Analysis" in text
    assert "### Attribute Analysis" in text
    assert "| Attribute | Leading Value | Count | Alternative Values |" in text
    assert "## 3. Analysis Summary" in text
    assert "## 4. Final Product Specification" in text
    assert "Proposed new product specification" in text
    assert "Pending PLMEX creation" in text
    assert "```json" in text
    assert '"approval_status": "Draft - Pending Buyer Review"' in text
    assert "Product Specification Decision Engine:" not in text
    assert "Historical attribute analysis" not in text
    # Winter preferred over Fall even when Fall has a higher raw count.
    assert '"ai_event_tags": "Winter"' in text or "| Ai Event Tags | Winter |" in text
    assert "| Material | Nm |" not in text
    assert "decision_engine=true" in evidence[0].excerpt
    assert "season=winter" in evidence[0].excerpt
    import re

    for m in re.finditer(r"\((\d+(?:\.\d+)?)%\)", text):
        assert float(m.group(1)) <= 100.0


def test_material_nm_is_not_a_usable_spec_value():
    from knowledge_graph.retrieval.ecom import _usable_value

    assert not _usable_value("Nm", "material")
    assert not _usable_value("nm", "material")
    assert _usable_value("Polyester", "material")


def test_winter_ask_rejects_fall_only_event_tags(monkeypatch):
    lines: list[str] = []
    evidence: list = []

    monkeypatch.setattr(ecom_mod, "clean_product_filters", lambda filters: dict(filters or {}))
    monkeypatch.setattr(ecom_mod, "cypher_count_products", lambda *_a, **_k: 10)
    monkeypatch.setattr(
        ecom_mod,
        "_specification_fields_for_graph",
        lambda *_a, **_k: ["ai_event_tags"],
    )
    monkeypatch.setattr(ecom_mod, "_graph_label", lambda *_a, **_k: "PLMEX")
    monkeypatch.setattr(
        ecom_mod,
        "cypher_group_by",
        lambda *_a, **_k: [("Fall", 8), ("Autumn", 2)],
    )

    ecom_mod._append_neo4j_specification_for_graph(
        SimpleNamespace(),
        INTERNAL,
        "plmex_products",
        {"buyer": "primark", "season_name": "winter"},
        lines,
        evidence,
    )
    text = "\n".join(lines)
    assert '"ai_event_tags": "Fall"' not in text
    assert "Pending buyer confirmation" in text
