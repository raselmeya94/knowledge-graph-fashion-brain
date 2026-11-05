"""Scraping extract → chunk → propose → queue → commit (new ontology/graph)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from knowledge_graph.api import create_app
from knowledge_graph.config import DATA_ROOT
from knowledge_graph.graph.snapshots import master_snapshot_path
from tests.conftest import make_platform

HEADERS = {"Authorization": "Bearer test-token", "X-Clearance": "internal"}


def _platform(copied_raw, tmp_path):
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
    )
    platform.rebuild()
    return platform


def test_scraping_paragraph_flow_creates_graph_and_master_source(copied_raw, tmp_path):
    platform = _platform(copied_raw, tmp_path)
    client = TestClient(create_app(platform))

    text = (
        "Published by Northwind Mills.\n\n"
        "Bangladesh remained the European Union's largest denim supplier in January-August 2026, "
        "earning €869.13 million and capturing 29.40% of the bloc's total denim imports, "
        "according to Eurostat data."
    )
    extracted = client.post(
        "/scraping/extract",
        headers=HEADERS,
        json={"kind": "text", "filename": "buyer.txt", "text": text},
    )
    assert extracted.status_code == 200, extracted.text
    file_id = extracted.json()["file_id"]
    assert extracted.json()["char_count"] > 0

    chunked = client.post("/scraping/chunk", headers=HEADERS, json={"file_id": file_id})
    assert chunked.status_code == 200, chunked.text
    chunks = chunked.json()["chunks"]
    assert len(chunks) >= 2
    assert all(item["chunk_id"] for item in chunks)
    assert chunks[0]["file_id"] if "file_id" in chunks[0] else True

    proposed = client.post(
        "/scraping/propose",
        headers=HEADERS,
        json={"file_id": file_id, "ontology_id": "news", "graph_id": "news", "use_model": False},
    )
    assert proposed.status_code == 200, proposed.text
    relationships = proposed.json()["relationships"]
    assert relationships
    assert all("novelty" in item for item in relationships)
    assert all(item.get("chunk_id") or (item.get("properties") or {}).get("chunk_id") for item in relationships)

    queued = client.post(
        "/scraping/queue",
        headers=HEADERS,
        json={"file_id": file_id, "ontology_id": "news", "graph_id": "news"},
    )
    assert queued.status_code == 200, queued.text
    review_id = queued.json()["review"]["id"]
    assert queued.json()["review"]["metadata"]["file_id"] == file_id

    approved = client.post(
        f"/kg/reviews/{review_id}",
        headers=HEADERS,
        json={"decision": "approved"},
    )
    assert approved.status_code == 200

    committed = client.post(
        "/scraping/commit",
        headers=HEADERS,
        json={
            "review_id": review_id,
            "ontology": {
                "mode": "new",
                "id": "buyer-scraping",
                "name": "Buyer scraping",
                "clone_from": "news",
            },
            "graph": {"id": "buyer-hm", "name": "Buyer H&M graph"},
        },
    )
    assert committed.status_code == 200, committed.text
    body = committed.json()
    assert body["graph_id"] == "buyer-hm"
    assert body["ontology_id"] == "buyer-scraping"
    assert body["committed"]

    edges = platform.graph.find_edges(graph="buyer-hm", include_deleted=False)
    assert edges

    master = master_snapshot_path(platform.settings.generated)
    assert master.exists()
    master_data = json.loads(master.read_text(encoding="utf-8"))
    assert "buyer-hm" in master_data.get("sources", [])

    ont_path = Path(platform.kg_registry.ontology_file("buyer-scraping"))
    assert ont_path.exists()
