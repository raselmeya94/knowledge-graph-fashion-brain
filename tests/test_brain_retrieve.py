"""Knowledge-graph /brain/retrieve and Fashion Brain HTTP client."""

from __future__ import annotations

from fastapi.testclient import TestClient

from fashion_brain.kg_client import retrieve_remote, uses_kg_api
from knowledge_graph.access import Principal
from knowledge_graph.api import create_app
from knowledge_graph.config import Settings
from knowledge_graph.retrieval.brain import BrainRetrieveRequest, retrieve
from tests.conftest import make_platform


def test_brain_retrieve_counts_cotton_products(copied_raw, tmp_path):
    platform = make_platform(copied_raw, tmp_path / "generated", kg_url="")
    platform.rebuild()
    result = retrieve(
        platform,
        Principal(subject="test", clearance="internal"),
        BrainRetrieveRequest(intent="product_count", filters={"material": "cotton"}, question="Which products use cotton?"),
    )
    assert "products" in result.text.lower()
    assert "cotton" in result.text.lower() or result.evidence
    assert not any("plmex" in caveat.lower() or "imported product" in caveat.lower() for caveat in result.caveats)


def test_brain_retrieve_http_endpoint(copied_raw, tmp_path):
    platform = make_platform(copied_raw, tmp_path / "generated", kg_url="")
    platform.rebuild()
    client = TestClient(create_app(platform))
    response = client.post(
        "/brain/retrieve",
        headers={"X-Clearance": "internal", "Authorization": "Bearer test-token"},
        json={"intent": "product_count", "filters": {"material": "cotton"}, "question": "Which products use cotton?"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "products" in payload["text"].lower()
    assert payload["evidence"]


def test_kg_client_uses_running_api(copied_raw, tmp_path, monkeypatch):
    platform = make_platform(copied_raw, tmp_path / "generated", kg_url="")
    platform.rebuild()
    app = create_app(platform)
    client = TestClient(app)

    class _Response:
        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200
            self.text = str(payload)
            self.content = b"{}"

        def json(self):
            return self._payload

    def fake_request(method, url, headers=None, json=None, data=None, files=None, params=None, timeout=None):
        assert url.endswith("/brain/retrieve")
        response = client.post("/brain/retrieve", headers=headers, json=json)
        return _Response(response.json())

    monkeypatch.setattr("fashion_brain.kg_client.httpx.request", fake_request)
    settings = Settings(kg_url="http://kg.test", api_token="test-token", request_timeout_seconds=30)
    assert uses_kg_api(settings)
    remote = retrieve_remote(
        settings,
        "internal",
        BrainRetrieveRequest(intent="product_count", filters={"material": "cotton"}),
    )
    assert "products" in remote.text.lower()


def test_brain_retrieve_accepts_graph_scope(copied_raw, tmp_path):
    platform = make_platform(copied_raw, tmp_path / "generated", kg_url="")
    platform.rebuild()
    result = retrieve(
        platform,
        Principal(subject="test", clearance="internal"),
        BrainRetrieveRequest(
            intent="product_count",
            filters={"material": "cotton"},
            question="Which products use cotton?",
            graph_id="own_products",
        ),
    )
    assert "own_products" in result.routes_hint or "products" in result.text.lower()


def test_graphs_reload_endpoint(copied_raw, tmp_path):
    platform = make_platform(copied_raw, tmp_path / "generated", kg_url="")
    platform.rebuild()
    client = TestClient(create_app(platform))
    response = client.post(
        "/graphs/reload",
        headers={"X-Clearance": "internal", "Authorization": "Bearer test-token"},
    )
    assert response.status_code == 200
    payload = response.json()
    # Reload runs as a background job; accept either sync stats or job envelope.
    if "job" in payload:
        assert payload.get("status") in {"started", "already_running", "completed"}
    else:
        assert payload.get("catalog_rows", 0) >= 0
        assert "nodes" in payload
