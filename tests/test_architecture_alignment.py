"""Smoke tests for Fashion Brain architecture alignment (Parts 1–5)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from fashion_brain.agents import ConversationContextAgent, GraphRouterAgent, Intention, ProductSpecificationAgent
from fashion_brain.execution.merger import merge_retrieved
from fashion_brain.agents.contracts import Retrieved
from knowledge_graph.api import create_app
from knowledge_graph.models import EvidenceItem
from tests.conftest import make_platform


HEADERS = {"Authorization": "Bearer test-token", "X-Clearance": "internal"}


def test_plan_api_aliases_and_registry(copied_raw, tmp_path):
    platform = make_platform(copied_raw, tmp_path / "generated")
    platform.rebuild()
    client = TestClient(create_app(platform))

    sources = client.get("/api/sources", headers=HEADERS)
    assert sources.status_code == 200
    ids = {item["source_id"] for item in sources.json()["sources"]}
    assert "own_products" in ids
    assert "ecom_products" in ids

    registry = client.get("/api/graphs/registry", headers=HEADERS)
    assert registry.status_code == 200
    rows = registry.json()["registry"]
    assert any(item["graph_id"] == "own_products" for item in rows)
    assert any(item.get("capabilities") for item in rows)

    meta = client.get("/api/metadata", headers=HEADERS, params={"graph_id": "own_products"})
    assert meta.status_code == 200
    assert meta.json()["graph_id"] == "own_products"

    values = client.get(
        "/api/metadata/values",
        headers=HEADERS,
        params={"attribute": "material", "graph_id": "own_products", "limit": 10},
    )
    assert values.status_code == 200
    assert "values" in values.json()

    global_info = client.get("/api/graphs/global", headers=HEADERS)
    assert global_info.status_code == 200
    assert global_info.json()["graph_id"] == "global"

    schema = client.get("/api/sources/own_products/schema", headers=HEADERS)
    assert schema.status_code == 200
    assert schema.json()["source_id"] == "own_products"


def test_graph_router_multi_graph_and_context_agent(copied_raw, tmp_path):
    """Router must select from live registry rows — never a hard-coded source list."""

    platform = make_platform(copied_raw, tmp_path / "generated")
    platform.rebuild()
    intention = Intention(intent="product_count", filters={"brand": "acme"})
    choice = GraphRouterAgent().choose(
        intention,
        platform=platform,
        question="Compare products across sources for sweaters",
    )
    assert choice.insufficient_coverage is False
    assert choice.selected_graphs
    # Every selected id must exist in the registry API/store (no invented source names).
    known = {item.id for item in platform.kg_registry.list_graphs()}
    assert set(choice.selected_graphs) <= known
    if "compare" in "Compare products across sources for sweaters".casefold():
        # With 2+ product graphs registered, cross language enables Global KG.
        product_graphs = [g for g in platform.kg_registry.list_graphs() if g.kind == "product"]
        if len(product_graphs) >= 2:
            assert choice.use_global_kg is True

    context = ConversationContextAgent().analyze(
        "Now show only men's sweaters",
        history=[{"role": "user", "content": "Show brand sweaters"}],
        conversation_state={
            "last_result_id": "result_001",
            "available_attributes": ["brand", "gender", "category"],
            "references_prior_chat": True,
        },
    )
    assert context.decision in {"REFINE", "REUSE", "RETRIEVE"}
    assert ProductSpecificationAgent().required(Intention(intent="specification", suggest=True)) is True


def test_result_merger_keeps_provenance():
    left = Retrieved(
        text="PLM product",
        evidence=[
            EvidenceItem(
                source_id="own_products",
                record_id="PLM_001",
                title="Sweater",
                excerpt="cotton",
                verification_status="observed",
                graph="own_products",
            )
        ],
        caveats=[],
        unavailable=[],
    )
    right = Retrieved(
        text="Ecom product",
        evidence=[
            EvidenceItem(
                source_id="ecom_products",
                record_id="ECOM_501",
                title="Sweater",
                excerpt="cotton",
                verification_status="observed",
                graph="ecom_products",
            )
        ],
        caveats=[],
        unavailable=[],
    )
    merged = merge_retrieved([left, right], result_id="result_x")
    assert merged.result_id == "result_x"
    assert len(merged.evidence) == 2
    assert "not merged" in merged.caveats[-1].lower()
