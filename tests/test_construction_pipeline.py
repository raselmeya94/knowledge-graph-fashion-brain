"""Knowledge-graph construction: extract a source, review relationships, then commit them."""

from __future__ import annotations

import io
import json
import shutil
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from knowledge_graph.api import create_app
from knowledge_graph.config import DATA_ROOT
from knowledge_graph.construction.pipeline import pipeline_for
from tests.conftest import make_platform

EMAIL = """From: Ada Lovelace <ada@example.com>
To: Grace Hopper <grace@example.com>
Subject: Linen shipment

Published by Northwind Mills.
"""

HTML = """<!DOCTYPE html>
<html><head><title>Spring newsletter</title></head>
<body>
<p>Unsubscribe from this newsletter.</p>
<p>Published by Northwind Mills.</p>
<a href="https://example.com/linen">Linen story</a>
</body></html>
"""

HEADERS = {"Authorization": "Bearer test-token", "X-Clearance": "internal"}


def _platform(copied_raw, tmp_path, **overrides):
    platform = make_platform(copied_raw, tmp_path / "generated", **overrides)
    platform.rebuild()
    return platform


def _client(platform):
    return TestClient(create_app(platform))


def _docx(text: str) -> bytes:
    xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body>
</w:document>"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", xml)
    return buffer.getvalue()


def test_tools_are_listed_and_ollama_status_is_reported(copied_raw, tmp_path):
    client = _client(_platform(copied_raw, tmp_path, ollama_base_url="http://127.0.0.1:9", ollama_timeout_seconds=1))
    tools = client.get("/kg/tools", headers=HEADERS)
    assert tools.status_code == 200
    names = {item["name"] for item in tools.json()["tools"]}
    assert {
        "extract_text",
        "extract_html",
        "parse_json",
        "understand_image",
        "extract_metadata",
        "list_relationships",
        "identify_relationships",
        "ollama",
        "queue_review",
        "ingest_graph",
    } <= names
    status = client.get("/kg/ollama", headers=HEADERS)
    assert status.status_code == 200
    assert status.json()["status"] == "unavailable"
    missing = client.post("/kg/tools/not_a_tool", headers=HEADERS, json={"text": "hello"})
    assert missing.status_code == 404


def test_email_review_edit_and_commit(copied_raw, tmp_path):
    platform = _platform(copied_raw, tmp_path, ollama_base_url="http://127.0.0.1:9")
    client = _client(platform)
    created = client.post(
        "/kg/pipeline",
        headers=HEADERS,
        json={"text": EMAIL, "kind": "email", "filename": "linen.eml", "use_model": False},
    )
    assert created.status_code == 200, created.text
    body = created.json()
    assert body["kind"] == "email"
    assert body["metadata"]["title"] == "Linen shipment"
    by_type = {item["type"]: item for item in body["relationships"]}
    assert by_type["SENT_BY"]["end_name"] == "Ada Lovelace"
    assert by_type["SENT_TO"]["end_name"] == "Grace Hopper"
    assert by_type["PUBLISHED_BY"]["end_name"] == "Northwind Mills"
    assert all(item["status"] == "new" for item in body["relationships"])

    review_id = body["review"]["id"]
    sent_to = by_type["SENT_TO"]["id"]
    edited = client.post(
        f"/kg/reviews/{review_id}",
        headers=HEADERS,
        json={
            "decision": "approved",
            "edits": [{"id": sent_to, "end_name": "Katherine Johnson"}],
        },
    )
    assert edited.status_code == 200
    committed = client.post(f"/kg/reviews/{review_id}/commit", headers=HEADERS)
    assert committed.status_code == 200, committed.text
    edges = platform.graph.find_edges(graph="news", include_deleted=False)
    pairs = {(edge.type, edge.end_id.split(":")[-1]) for edge in edges}
    assert ("SENT_BY", "ada-lovelace") in pairs
    assert ("SENT_TO", "katherine-johnson") in pairs
    assert ("PUBLISHED_BY", "northwind-mills") in pairs
    assert not any(edge.end_id.endswith("grace-hopper") for edge in edges)

    again = client.post(
        "/kg/pipeline",
        headers=HEADERS,
        json={"text": EMAIL, "kind": "email", "use_model": False},
    )
    second = {item["type"]: item["status"] for item in again.json()["relationships"]}
    assert second["SENT_BY"] == "existing"
    assert second["PUBLISHED_BY"] == "existing"
    assert second["SENT_TO"] == "new"


def test_html_json_docx_and_rejected_review(copied_raw, tmp_path):
    platform = _platform(copied_raw, tmp_path, ollama_base_url="http://127.0.0.1:9")
    client = _client(platform)

    html = client.post("/kg/pipeline", headers=HEADERS, json={"text": HTML, "filename": "news.html", "use_model": False})
    assert html.status_code == 200, html.text
    assert html.json()["metadata"]["document_type"] == "newsletter"
    assert html.json()["metadata"]["title"] == "Spring newsletter"
    assert html.json()["metadata"]["urls"] == ["https://example.com/linen"]
    assert html.json()["relationships"][0]["type"] == "PUBLISHED_BY"

    parsed = client.post(
        "/kg/tools/parse_json",
        headers=HEADERS,
        json={"json_data": {"title": "Lookbook", "author": "Ada Lovelace", "topic": "Linen"}},
    )
    assert parsed.status_code == 200
    identified = client.post(
        "/kg/tools/identify_relationships",
        headers=HEADERS,
        json={
            "title": "Lookbook",
            "text": parsed.json()["text"],
            "kind": "json",
            "structured": {"title": "Lookbook", "author": "Ada Lovelace", "topic": "Linen"},
            "use_model": False,
        },
    )
    kinds = {item["type"] for item in identified.json()["relationships"]}
    assert kinds == {"AUTHORED_BY", "ABOUT"}

    uploaded = client.post(
        "/kg/pipeline/upload",
        headers=HEADERS,
        files={"file": ("note.docx", _docx("Published by Northwind Mills."), "application/octet-stream")},
        data={"use_model": "false", "kind": "docx"},
    )
    assert uploaded.status_code == 200, uploaded.text
    assert uploaded.json()["relationships"][0]["end_name"] == "Northwind Mills"

    review_id = html.json()["review"]["id"]
    rejected = client.post(f"/kg/reviews/{review_id}", headers=HEADERS, json={"decision": "rejected"})
    assert rejected.status_code == 200
    blocked = client.post(f"/kg/reviews/{review_id}/commit", headers=HEADERS)
    assert blocked.status_code == 422
    edited = client.post(
        f"/kg/reviews/{review_id}",
        headers=HEADERS,
        json={"edits": [{"id": html.json()["relationships"][0]["id"], "type": "USES_MATERIAL", "end_type": "Material"}]},
    )
    assert edited.status_code == 200
    edited_rel = next(
        item for item in edited.json()["review"]["relationships"] if item["id"] == html.json()["relationships"][0]["id"]
    )
    assert edited_rel["needs_ontology_approval"] is True
    assert edited_rel["type"] == "USES_MATERIAL"


_PDF = b"\n".join(
    [
        b"%PDF-1.4",
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj",
        b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj",
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 144]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj",
        b"4 0 obj<</Length 73>>stream",
        b"BT /F1 12 Tf 36 100 Td (Published by Northwind Mills.) Tj ET",
        b"endstream",
        b"endobj",
        b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj",
        b"xref",
        b"0 6",
        b"0000000000 65535 f ",
        b"0000000009 00000 n ",
        b"0000000052 00000 n ",
        b"0000000101 00000 n ",
        b"0000000226 00000 n ",
        b"0000000350 00000 n ",
        b"trailer<</Size 6/Root 1 0 R>>",
        b"startxref",
        b"420",
        b"%%EOF",
    ]
)


def test_pdf_text_and_image_without_ollama(copied_raw, tmp_path):
    platform = _platform(copied_raw, tmp_path, ollama_base_url="http://127.0.0.1:9", ollama_timeout_seconds=1)
    pipeline = pipeline_for(platform)
    extracted = pipeline.run_tool("extract_text", {"filename": "note.pdf", "data": _PDF})
    assert "Northwind Mills" in extracted["text"]
    image = pipeline.run_tool(
        "understand_image",
        {"filename": "swatch.png", "data": b"\x89PNG\r\n\x1a\n" + b"0" * 32},
    )
    assert image["text"] == ""
    assert image["warnings"]


def test_legacy_doc_recovers_text(copied_raw, tmp_path):
    platform = _platform(copied_raw, tmp_path, ollama_base_url="http://127.0.0.1:9")
    data = "Published by Northwind Mills.".encode("utf-16le")
    result = pipeline_for(platform).run({"filename": "memo.doc", "data": data, "use_model": False})
    assert result["kind"] == "doc"
    assert result["relationships"][0]["type"] == "PUBLISHED_BY"
    assert any("Legacy .doc" in warning for warning in result["warnings"])


def test_unlisted_news_relationship_queued_then_extends_ontology(copied_raw, tmp_path):
    schemas = tmp_path / "schemas"
    schemas.mkdir()
    news_ontology = schemas / "news-ontology.json"
    shutil.copy(DATA_ROOT / "schemas" / "news-ontology.json", news_ontology)
    shutil.copy(DATA_ROOT / "schemas" / "product-knowledge-ontology.json", schemas / "product-knowledge-ontology.json")
    platform = _platform(
        copied_raw,
        tmp_path,
        ollama_base_url="http://127.0.0.1:9",
        ontology_path=schemas / "product-knowledge-ontology.json",
        news_ontology_path=news_ontology,
    )
    client = _client(platform)
    text = (
        "H&M sources denim from Bangladesh. "
        "Evidence for review: H&M and Bangladesh appear in this sentence about sourcing."
    )
    identified = client.post(
        "/kg/tools/identify_relationships",
        headers=HEADERS,
        json={
            "title": "Buyer sourcing note",
            "text": text,
            "kind": "text",
            "use_model": False,
            "structured": {
                "relationships": [
                    {
                        "type": "EXPORTS_TO",
                        "start_name": "H&M",
                        "start_type": "Company",
                        "end_name": "Bangladesh",
                        "end_type": "Country",
                        "evidence": "H&M sources denim from Bangladesh.",
                    }
                ]
            },
        },
    )
    assert identified.status_code == 200, identified.text
    body = identified.json()
    assert any(item["type"] == "EXPORTS_TO" and item["start_name"] == "H&M" for item in body["relationships"])
    edge = next(item for item in body["relationships"] if item["start_name"] == "H&M")
    assert edge["needs_ontology_approval"] is True
    assert any("not allowlisted yet" in warning for warning in body["warnings"])

    queued = client.post(
        "/kg/tools/queue_review",
        headers=HEADERS,
        json={
            "source_kind": "text",
            "filename": "sourcing.txt",
            "title": "Buyer sourcing note",
            "text": text,
            "relationships": body["relationships"],
            "warnings": body["warnings"],
        },
    )
    assert queued.status_code == 200
    review_id = queued.json()["review"]["id"]
    approved = client.post(f"/kg/reviews/{review_id}", headers=HEADERS, json={"decision": "approved"})
    assert approved.status_code == 200
    committed = client.post(f"/kg/reviews/{review_id}/commit", headers=HEADERS)
    assert committed.status_code == 200, committed.text
    data = json.loads(Path(news_ontology).read_text(encoding="utf-8"))
    assert ["Company", "Country"] in data["relationship_allowlist"]["EXPORTS_TO"]
