"""System / registry metadata via kg_system catalog grounding — not question hardcodes."""

from __future__ import annotations

from fashion_brain.agents._impl import IntentionAgent, ThinkingBrain
from fashion_brain.ask import resolve_catalog_plan
from fashion_brain.planning.kg_catalog import SYSTEM_GRAPH_ID, build_field_catalog, catalog_selects_system_meta
from knowledge_graph.access import Principal
from knowledge_graph.retrieval.brain import BrainRetrieveRequest, retrieve
from knowledge_graph.retrieval.system_meta import build_system_meta
from tests.conftest import make_platform


def test_system_ontology_grounding_selects_kg_system(copied_raw, tmp_path):
    platform = make_platform(copied_raw, tmp_path / "generated", kg_url="")
    platform.rebuild()
    catalog = build_field_catalog(platform, "how many sources are present")
    assert SYSTEM_GRAPH_ID in catalog.graphs
    assert catalog_selects_system_meta(catalog)
    # Product ask still prefers product graphs, not system.
    product_catalog = build_field_catalog(platform, "how many cotton products for Primark")
    assert not catalog_selects_system_meta(product_catalog)


def test_present_is_not_a_garment_filter():
    plan = resolve_catalog_plan("how many sources are present", carry=False)
    filters = plan.get("filters") or {}
    assert "present" not in str(filters).casefold()


def test_intention_uses_catalog_not_prior_product_chat(copied_raw, tmp_path):
    platform = make_platform(copied_raw, tmp_path / "generated", kg_url="")
    platform.settings.answer_mode = "keywords"
    history = [
        {"role": "user", "content": "material wise cotton brand analysis in ecom"},
        {"role": "assistant", "content": "175 cotton products"},
    ]
    thinking = ThinkingBrain().read(platform, "how many sources are present", history=history)
    assert thinking.filters == {}
    assert thinking.references_prior_chat is False
    intention = IntentionAgent().understand(
        platform,
        "how many sources are present",
        history=history,
        summary=thinking,
    )
    assert intention.intent == "system_meta"
    assert intention.filters == {}
    assert SYSTEM_GRAPH_ID in (intention.preferred_graphs or [])


def test_brain_system_meta_lists_graphs(copied_raw, tmp_path):
    platform = make_platform(copied_raw, tmp_path / "generated", kg_url="")
    platform.rebuild()
    meta = build_system_meta(platform, persist=True)
    assert meta["graph_count"] >= 1
    assert "value_index" in meta
    assert meta["value_index"].get("topic")
    result = retrieve(
        platform,
        Principal(subject="test", clearance="internal"),
        BrainRetrieveRequest(intent="system_meta", question="how many sources are present"),
    )
    text = result.text.casefold()
    assert "registered knowledge graph" in text or "knowledge graph" in text
    assert "shared" in text
    assert result.evidence


def test_attribute_overview_question_grounds_system(copied_raw, tmp_path):
    platform = make_platform(copied_raw, tmp_path / "generated", kg_url="")
    platform.rebuild()
    catalog = build_field_catalog(platform, "what unique attributes exist per source")
    assert catalog_selects_system_meta(catalog)
