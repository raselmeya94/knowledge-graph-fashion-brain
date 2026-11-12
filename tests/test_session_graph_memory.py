"""Same conversation id should keep catalog memory (ecom vs plmex)."""

from __future__ import annotations

from types import SimpleNamespace

from fashion_brain.agents.contracts import Intention
from fashion_brain.agents.conversation_context import ConversationContextAgent
from fashion_brain.orchestration.catalog_pipeline import _apply_session_graph_memory, _remember_graph


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
    assert out.preferred_graphs[0] == "ecom_products"


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
    assert decision.active_filters == {}
