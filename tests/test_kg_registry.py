"""Multi-ontology / multi-graph registry APIs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from knowledge_graph.api import create_app
from knowledge_graph.config import DATA_ROOT
from tests.conftest import make_platform

HEADERS = {"Authorization": "Bearer test-token", "X-Clearance": "internal"}


def _platform(copied_raw, tmp_path, **overrides):
    schemas = tmp_path / "schemas"
    schemas.mkdir(exist_ok=True)
    shutil.copy(DATA_ROOT / "schemas" / "news-ontology.json", schemas / "news-ontology.json")
    shutil.copy(
        DATA_ROOT / "schemas" / "product-knowledge-ontology.json",
        schemas / "product-knowledge-ontology.json",
    )
    platform = make_platform(
        copied_raw,
        tmp_path / "generated",
        ollama_base_url="http://127.0.0.1:9",
        ontology_path=schemas / "product-knowledge-ontology.json",
        news_ontology_path=schemas / "news-ontology.json",
        **overrides,
    )
    platform.rebuild()
    return platform


def _client(platform):
    return TestClient(create_app(platform))


def test_create_ontology_graph_pipeline_commit_and_delete_edge(copied_raw, tmp_path):
    platform = _platform(copied_raw, tmp_path)
    client = _client(platform)

    ontology = client.post(
        "/ontologies",
        headers=HEADERS,
        json={
            "id": "buyer-scraping",
            "name": "Buyer scraping",
            "clone_from": "news",
            "description": "Custom buyer PDF ontology",
        },
    )
    assert ontology.status_code == 200, ontology.text
    body = ontology.json()["ontology"]
    assert body["id"] == "buyer-scraping"
    # Confirmed custom ontologies belong under schemas/ontologies/, not generated/.
    assert "schemas/ontologies/buyer-scraping.json" in body["path"].replace("\\", "/")
    assert "generated/ontologies" not in body["path"].replace("\\", "/")
    assert platform.kg_registry.ontology_file("buyer-scraping").exists()

    graph = client.post(
        "/graphs",
        headers=HEADERS,
        json={
            "id": "buyer-hm",
            "name": "H&M buyer graph",
            "ontology_id": "buyer-scraping",
            "kind": "custom",
        },
    )
    assert graph.status_code == 200, graph.text
    assert graph.json()["graph"]["id"] == "buyer-hm"

    listed = client.get("/graphs", headers=HEADERS)
    assert listed.status_code == 200
    ids = {item["id"] for item in listed.json()["graphs"]}
    assert "buyer-hm" in ids
    assert "own_products" in ids

    text = "H&M sources denim from Bangladesh for spring orders."
    proposed = client.post(
        "/relationships/propose",
        headers=HEADERS,
        json={
            "title": "Buyer note",
            "text": text,
            "kind": "text",
            "graph_id": "buyer-hm",
            "use_model": False,
            "structured": {
                "relationships": [
                    {
                        "type": "EXPORTS_TO",
                        "start_name": "H&M",
                        "start_type": "Company",
                        "end_name": "Bangladesh",
                        "end_type": "Country",
                        "evidence": "H&M sources denim from Bangladesh for spring orders.",
                    }
                ]
            },
        },
    )
    assert proposed.status_code == 200, proposed.text
    body = proposed.json()
    assert body["graph_id"] == "buyer-hm"
    edge = next(item for item in body["relationships"] if item["start_name"] == "H&M")
    assert edge["needs_ontology_approval"] is True

    pipeline = client.post(
        "/kg/pipeline",
        headers=HEADERS,
        json={
            "text": text,
            "kind": "text",
            "filename": "buyer.txt",
            "use_model": False,
            "graph_id": "buyer-hm",
            "structured": None,
        },
    )
    # Pipeline without structured still works; seed via tools queue for the ontology-gap edge.
    assert pipeline.status_code == 200, pipeline.text
    assert pipeline.json()["graph_id"] == "buyer-hm"

    queued = client.post(
        "/kg/tools/queue_review",
        headers=HEADERS,
        json={
            "source_kind": "text",
            "filename": "buyer.txt",
            "title": "Buyer note",
            "text": text,
            "graph_id": "buyer-hm",
            "ontology_id": "buyer-scraping",
            "relationships": body["relationships"],
            "warnings": body["warnings"],
        },
    )
    assert queued.status_code == 200, queued.text
    review_id = queued.json()["review"]["id"]
    assert queued.json()["review"]["graph_id"] == "buyer-hm"

    approved = client.post(
        f"/kg/reviews/{review_id}",
        headers=HEADERS,
        json={"decision": "approved"},
    )
    assert approved.status_code == 200
    committed = client.post(f"/kg/reviews/{review_id}/commit", headers=HEADERS)
    assert committed.status_code == 200, committed.text
    assert committed.json()["graph_id"] == "buyer-hm"

    ontology_path = Path(platform.kg_registry.ontology_file("buyer-scraping"))
    data = json.loads(ontology_path.read_text(encoding="utf-8"))
    assert ["Company", "Country"] in data["relationship_allowlist"]["EXPORTS_TO"]

    edges = platform.graph.find_edges(graph="buyer-hm", include_deleted=False)
    assert edges
    edge_id = edges[0].id
    deleted = client.delete(f"/graphs/buyer-hm/edges/{edge_id}", headers=HEADERS)
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["edge"]["properties"]["lifecycle"] == "deleted"
    assert not platform.graph.find_edges(graph="buyer-hm", include_deleted=False)

    extracted = client.post(
        "/extract",
        headers=HEADERS,
        json={"text": "Published by Northwind Mills.", "kind": "text"},
    )
    assert extracted.status_code == 200
    assert "Northwind Mills" in extracted.json()["text"]

    removed = client.delete(
        f"/kg/reviews/{review_id}/relationships/{edge['id']}",
        headers=HEADERS,
    )
    # Already committed — should fail validation
    assert removed.status_code == 422
