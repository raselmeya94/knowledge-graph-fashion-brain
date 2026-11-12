import json
from types import SimpleNamespace

from fashion_brain.agents.contracts import Intention
from fashion_brain.agents.text2cypher import TEXT2CYPHER_SYSTEM, Text2CypherAgent, _parse_plans


class _Provider:
    def __init__(self):
        self.prompt = ""

    def complete(self, prompt, *, timeout, system=None):
        self.prompt = prompt
        return json.dumps(
            {
                "cypher": (
                    "MATCH (p:Product) WHERE p.graph = $g "
                    "RETURN count(DISTINCT p.id) AS count"
                ),
                "params": {"g": "wrong_graph", "brand_ids": ["unresolved-id"]},
                "result_kind": "count",
                "explanation": "Product count.",
            }
        )


def test_text2cypher_prompt_always_receives_complete_structured_context(monkeypatch):
    provider = _Provider()
    platform = SimpleNamespace(
        settings=SimpleNamespace(
            answer_mode="llm",
            text2cypher_enabled=True,
            graph_rag_mode="text2cypher",
            request_timeout_seconds=10,
        )
    )
    monkeypatch.setattr("fashion_brain.agents.text2cypher.build_model_provider", lambda _settings: provider)
    monkeypatch.setattr(
        "fashion_brain.agents.text2cypher._labels_match_schema",
        lambda *_args, **_kwargs: True,
    )

    intention = Intention(
        intent="product_count",
        filters={"brand": "zara", "material": "cotton", "color": "red"},
        analysis=True,
        limit=5,
    )
    plans = Text2CypherAgent().generate(
        platform,
        "Analyze Zara's red cotton products",
        intention,
        "products",
        resolved={"brand": {"ids": ["brand-1"], "names": ["Zara"]}},
        extracted={"brand": "Zara", "material": "cotton", "color": "red"},
        brands=[],
        candidates=[{"field": "material", "stored_value": "Cotton"}],
        schema_card="Product label: Product\nProperties: graph, id, material, color",
    )

    assert len(plans) == 1
    assert plans[0].params["g"] == "products"
    assert plans[0].params["brand_ids"] == ["brand-1"]
    assert "<intention>" in provider.prompt
    assert "<active_filters>" in provider.prompt
    assert '"material": "cotton"' in provider.prompt
    assert "<resolved_entities>" in provider.prompt
    assert '"brand-1"' in provider.prompt
    assert "<extracted_entities>" in provider.prompt
    assert "<candidates>" in provider.prompt
    assert "Analyze Zara's red cotton products" in provider.prompt


def test_text2cypher_prompt_covers_analysis_filters_and_temporal_grounding():
    assert "simultaneous filters are AND constraints" in TEXT2CYPHER_SYSTEM
    assert "up to five distinct" in TEXT2CYPHER_SYSTEM
    assert "real temporal property/event or snapshot relationship" in TEXT2CYPHER_SYSTEM
    assert "all applicable filters" in TEXT2CYPHER_SYSTEM


def test_text2cypher_applies_to_shared_products_not_bare_shared():
    agent = Text2CypherAgent()
    assert agent.applicable(Intention(intent="shared_products")) is True
    assert agent.applicable(Intention(intent="shared")) is False
    assert agent.applicable(Intention(intent="system_meta")) is False


def test_text2cypher_multi_graph_seeds_graph_ids(monkeypatch):
    provider = _Provider()
    platform = SimpleNamespace(
        settings=SimpleNamespace(
            answer_mode="llm",
            text2cypher_enabled=True,
            graph_rag_mode="off",
            request_timeout_seconds=10,
        )
    )
    monkeypatch.setattr("fashion_brain.agents.text2cypher.build_model_provider", lambda _settings: provider)
    monkeypatch.setattr(
        "fashion_brain.agents.text2cypher._labels_match_schema",
        lambda *_args, **_kwargs: True,
    )

    plans = Text2CypherAgent().generate(
        platform,
        "Show materials shared across ecom and plmex.",
        Intention(intent="shared_products", group_by="material"),
        "ecom_products",
        graph_ids=["ecom_products", "plmex_products"],
        schema_card="Product label: Product",
        brands=[],
    )

    assert len(plans) == 1
    assert plans[0].params["g"] == "ecom_products"
    assert plans[0].params["graph_ids"] == ["ecom_products", "plmex_products"]
    assert "MULTI-GRAPH" in provider.prompt or "p.graph IN $graph_ids" in provider.prompt
    assert "ecom_products" in provider.prompt and "plmex_products" in provider.prompt


def test_text2cypher_requires_json_instead_of_extracting_raw_query_text():
    raw_query = "MATCH (p:Product) RETURN count(p) AS count"
    assert _parse_plans(raw_query, default_graph="products") == []


def test_empty_filtered_product_queries_fall_back_to_templates(monkeypatch):
    agent = Text2CypherAgent()
    platform = SimpleNamespace(settings=SimpleNamespace(text2cypher_enabled=True))
    intention = Intention(
        intent="product_count",
        filters={"brand": "primark"},
        group_by="category",
    )
    plan = _parse_plans(
        json.dumps(
            {
                "cypher": "MATCH (p:Product) RETURN count(p) AS count",
                "params": {"g": "ecom_products"},
                "result_kind": "breakdown",
            }
        ),
        default_graph="ecom_products",
    )[0]
    monkeypatch.setattr(agent, "generate", lambda *_args, **_kwargs: [plan])

    def execute_rows(rows):
        return lambda *_args, **_kwargs: rows

    for rows in ([], [{"count": 0}]):
        monkeypatch.setattr(
            "fashion_brain.agents.text2cypher._execute",
            execute_rows(rows),
        )
        assert agent.retrieve(
            platform,
            None,
            "Show product counts by category for Primark",
            intention,
            "ecom_products",
        ) is None

    monkeypatch.setattr(
        "fashion_brain.agents.text2cypher._execute",
        execute_rows([{"category": "Clothes", "count": 1464}]),
    )
    retrieved = agent.retrieve(
        platform,
        None,
        "Show product counts by category for Primark",
        intention,
        "ecom_products",
    )
    assert retrieved is not None
    assert "1464" in retrieved.text
