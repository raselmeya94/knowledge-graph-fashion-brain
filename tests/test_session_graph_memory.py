"""Same conversation id should keep catalog memory (ecom vs plmex)."""

from __future__ import annotations

from types import SimpleNamespace

from fashion_brain.agents.contracts import Intention
from fashion_brain.agents.conversation_context import ConversationContextAgent
from fashion_brain.orchestration.catalog_pipeline import (
    _apply_explicit_graph_preferences,
    _apply_request_graph_pin,
    _apply_session_graph_memory,
    _remember_graph,
    _resolve_meta_count_graph,
)


class _Graph:
    def __init__(self, gid: str, kind: str = "product"):
        self.id = gid
        self.kind = kind
        self.name = gid
        self.description = gid
        self.capabilities = ["brand", "buyer"] if "plmex" in gid else ["brand"]
        self.entities = []
        self.status = "active"


def test_remember_graph_skips_shared():
    state: dict = {}
    _remember_graph(state, ["shared", "ecom_products"])
    assert state["last_graph_id"] == "ecom_products"


def test_explicit_source_overrides_stale_planner_preference(monkeypatch):
    monkeypatch.setattr(
        "fashion_brain.ask.preferred_graphs_from_question",
        lambda _q, _p=None: ["plmex_products"],
    )
    intention = Intention(
        intent="product_count",
        preferred_graphs=["ecom_products"],
    )

    named = _apply_explicit_graph_preferences(
        intention,
        "how many sweater in plmex",
        object(),
    )

    assert named == ["plmex_products"]
    assert intention.preferred_graphs == ["plmex_products"]


def test_explicit_source_overrides_stale_validated_and_sticky_graphs(monkeypatch):
    monkeypatch.setattr(
        "fashion_brain.ask.preferred_graphs_from_question",
        lambda _q, _p=None: ["plmex_products"],
    )

    resolved = _resolve_meta_count_graph(
        object(),
        question="how many sweater in plmex",
        validated=SimpleNamespace(source="ecom_products"),
        graph_id="ecom_products",
        state={"last_graph_id": "ecom_products"},
    )

    assert resolved == "plmex_products"


def test_explicit_source_overrides_request_graph_pin(monkeypatch):
    monkeypatch.setattr(
        "fashion_brain.ask.preferred_graphs_from_question",
        lambda _q, _p=None: ["plmex_products"],
    )

    selected = _apply_request_graph_pin(
        ["plmex_products"],
        graph_id="ecom_products",
        intent="product_count",
        question="how many sweater in plmex",
        platform=object(),
    )

    assert selected == ["plmex_products"]


def test_request_graph_pin_remains_for_source_ambiguous_questions(monkeypatch):
    monkeypatch.setattr(
        "fashion_brain.ask.preferred_graphs_from_question",
        lambda _q, _p=None: [],
    )

    selected = _apply_request_graph_pin(
        ["plmex_products"],
        graph_id="ecom_products",
        intent="product_count",
        question="how many sweaters?",
        platform=object(),
    )

    assert selected == ["ecom_products"]


def test_session_graph_memory_prefers_last_ecom_when_brand_ambiguous(monkeypatch):
    platform = SimpleNamespace(
        kg_registry=SimpleNamespace(
            list_graphs=lambda: [_Graph("ecom_products"), _Graph("plmex_products")]
        )
    )
    monkeypatch.setattr(
        "knowledge_graph.retrieval.ecom.registry_product_graphs",
        lambda _p: ["ecom_products", "plmex_products"],
    )
    monkeypatch.setattr(
        "fashion_brain.ask.preferred_graphs_from_question",
        lambda _q, _p=None: [],
    )
    intention = Intention(
        intent="product_count",
        filters={"brand": "primark"},
        preferred_graphs=["plmex_products"],
    )
    context = SimpleNamespace(is_contextual=True)
    out = _apply_session_graph_memory(
        platform,
        intention,
        question="which product types does Primark make most often?",
        history=[{"role": "user", "content": "show primark products"}],
        context=context,
        state={"last_graph_id": "ecom_products"},
        graph_id=None,
    )
    # Sticky must collapse — not merely reorder — so value hits on plmex do not fan out.
    assert out.preferred_graphs == ["ecom_products"]


def test_session_graph_memory_pins_sticky_when_color_hits_both_catalogs(monkeypatch):
    """T2 ecom → T3 'of them are red' must not re-open plmex just because red exists there."""

    platform = SimpleNamespace(
        kg_registry=SimpleNamespace(
            list_graphs=lambda: [_Graph("ecom_products"), _Graph("plmex_products")]
        )
    )
    monkeypatch.setattr(
        "knowledge_graph.retrieval.ecom.registry_product_graphs",
        lambda _p: ["ecom_products", "plmex_products"],
    )
    monkeypatch.setattr(
        "fashion_brain.ask.preferred_graphs_from_question",
        lambda _q, _p=None: [],
    )
    intention = Intention(
        intent="breakdown",
        filters={"brand": "primark", "color": "red", "garment": "sweater"},
        group_by="color",
        preferred_graphs=["ecom_products", "plmex_products"],
    )
    out = _apply_session_graph_memory(
        platform,
        intention,
        question="now tell me how many sweater of them are red color?",
        history=[
            {"role": "user", "content": "How many products does Primark have?"},
            {"role": "user", "content": "And how many of them are sweaters?"},
        ],
        context=SimpleNamespace(is_contextual=True),
        state={"last_graph_id": "ecom_products", "last_result_id": "t2"},
        graph_id=None,
    )
    assert out.preferred_graphs == ["ecom_products"]


def test_context_agent_keeps_session_sticky_when_llm_unavailable():
    agent = ConversationContextAgent()
    decision = agent.analyze(
        "use the earlier Primark analysis and show top categories",
        history=[{"role": "user", "content": "analyze Primark assortment"}],
        conversation_state={"last_graph_id": "ecom_products", "last_result_id": "result_abc"},
        platform=SimpleNamespace(settings=SimpleNamespace(answer_mode="template")),
    )
    assert decision.is_contextual is True
    assert decision.decision == "RETRIEVE"
    # Party may be reused; restrictive facets must not be invented from empty prior filters.
    assert "color" not in (decision.active_filters or {})
    assert "garment" not in (decision.active_filters or {})
