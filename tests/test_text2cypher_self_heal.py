"""Unit tests for Text2Cypher 3-attempt self-heal via repair_feedback."""

from __future__ import annotations

import json
from types import SimpleNamespace

from fashion_brain.agents.contracts import Intention
from fashion_brain.agents.text2cypher import CypherPlan, Text2CypherAgent, TEXT2CYPHER_SYSTEM


def test_prompt_includes_optional_match_and_resolves_to_rules():
    assert "OPTIONAL MATCH" in TEXT2CYPHER_SYSTEM
    assert "RESOLVES_TO" in TEXT2CYPHER_SYSTEM
    assert "resolved_entities" in TEXT2CYPHER_SYSTEM or "<resolved_entities>" in TEXT2CYPHER_SYSTEM


def test_self_heal_retries_on_execute_error_then_succeeds(monkeypatch):
    agent = Text2CypherAgent()
    platform = SimpleNamespace(settings=SimpleNamespace(text2cypher_enabled=True))
    intention = Intention(intent="product_count", filters={"brand": "zara"})

    plan_bad = CypherPlan(
        cypher="MATCH (p:Product) RETURN p",
        params={"g": "ecom_products"},
        result_kind="count",
        explanation="bad",
    )
    plan_good = CypherPlan(
        cypher="MATCH (p:Product) WHERE p.graph=$g RETURN count(DISTINCT p.id) AS count",
        params={"g": "ecom_products"},
        result_kind="count",
        explanation="ok",
    )
    feedbacks: list[str | None] = []
    calls = {"n": 0}

    def fake_generate(*_a, **kwargs):
        feedbacks.append(kwargs.get("repair_feedback"))
        calls["n"] += 1
        if calls["n"] < 3:
            return [plan_bad]
        return [plan_good]

    def fake_execute(_platform, _principal, plan):
        if "count(DISTINCT" in plan.cypher:
            return [{"count": 42}]
        raise RuntimeError("SyntaxError: Invalid input 'p'")

    monkeypatch.setattr(agent, "generate", fake_generate)
    monkeypatch.setattr("fashion_brain.agents.text2cypher._execute", fake_execute)
    monkeypatch.setattr(
        "fashion_brain.agents.text2cypher._auto_resolve_entities",
        lambda *_a, **_k: {"brand": {"ids": ["b1"], "names": ["Zara"]}},
    )

    retrieved = agent.retrieve(
        platform,
        None,
        "How many Zara products?",
        intention,
        "ecom_products",
    )
    assert retrieved is not None
    assert "42" in retrieved.text
    assert calls["n"] == 3
    assert feedbacks[0] is None
    assert feedbacks[1] is not None and "Neo4j" in feedbacks[1]
    assert feedbacks[2] is not None


def test_self_heal_exhausted_returns_none(monkeypatch):
    agent = Text2CypherAgent()
    platform = SimpleNamespace(settings=SimpleNamespace(text2cypher_enabled=True))
    intention = Intention(intent="list", filters={"brand": "missing"})
    plan = CypherPlan(
        cypher="MATCH (p:Product) RETURN p.name AS name LIMIT 5",
        params={"g": "ecom_products"},
        result_kind="rows",
    )
    monkeypatch.setattr(agent, "generate", lambda *_a, **_k: [plan])
    monkeypatch.setattr(
        "fashion_brain.agents.text2cypher._execute",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        "fashion_brain.agents.text2cypher._auto_resolve_entities",
        lambda *_a, **_k: {},
    )
    assert (
        agent.retrieve(platform, None, "List missing brand", intention, "ecom_products")
        is None
    )


def test_generate_passes_temperature_zero_and_repair_block(monkeypatch):
    class Provider:
        def __init__(self):
            self.temperature = None
            self.prompt = ""

        def complete(self, prompt, *, timeout, system=None, temperature=None, **_k):
            self.temperature = temperature
            self.prompt = prompt
            return json.dumps(
                {
                    "cypher": "MATCH (p:Product) WHERE p.graph=$g RETURN count(p) AS c",
                    "params": {"g": "ecom_products"},
                    "result_kind": "count",
                }
            )

    provider = Provider()
    platform = SimpleNamespace(
        settings=SimpleNamespace(
            answer_mode="llm",
            text2cypher_enabled=True,
            request_timeout_seconds=10,
        )
    )
    monkeypatch.setattr(
        "fashion_brain.agents.text2cypher.build_model_provider", lambda _s: provider
    )
    monkeypatch.setattr(
        "fashion_brain.agents.text2cypher._labels_match_schema",
        lambda *_a, **_k: True,
    )
    plans = Text2CypherAgent().generate(
        platform,
        "Count products",
        Intention(intent="product_count"),
        "ecom_products",
        brands=[],
        schema_card="Product label: Product",
        repair_feedback="Previous Cypher failed: syntax error near RETURN",
    )
    assert plans
    assert provider.temperature == 0.0
    assert "<repair_feedback>" in provider.prompt
    assert "syntax error" in provider.prompt


def test_auto_resolve_merges_filter_terms(monkeypatch):
    from fashion_brain.agents.text2cypher import _auto_resolve_entities

    calls: list[str] = []

    class FakeResult:
        def __init__(self, term):
            self.status = "matched"
            self.hits = [object()]
            self._term = term

        def to_dict(self):
            return {"ids": [f"id-{self._term}"], "names": [self._term.title()], "status": "matched"}

    def fake_resolve(_platform, term, **_k):
        calls.append(term)
        return FakeResult(term)

    monkeypatch.setattr(
        "knowledge_graph.linking.canonical_resolve.resolve_canonical_entity",
        fake_resolve,
    )
    intention = Intention(intent="product_count", filters={"brand": "zara", "material": "cotton"})
    out = _auto_resolve_entities(
        SimpleNamespace(),
        "How many Zara cotton SKUs?",
        intention,
        extracted={"color": "red"},
        graph_id="ecom_products",
    )
    assert "brand" in out and out["brand"]["ids"] == ["id-zara"]
    assert "material" in out
    assert "color" in out
    assert "zara" in calls and "cotton" in calls
