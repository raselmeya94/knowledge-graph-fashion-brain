"""Unit tests for OntologyRegistry (no live Neo4j)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from knowledge_graph.ontology_registry import (
    OntologyRegistry,
    build_ontology_registry,
    get_ontology_registry,
    invalidate_ontology_registry_cache,
)


def _write_ontology(path: Path, *, name: str, description: str, entities: list[dict], allowlist: dict) -> None:
    payload = {
        "name": name,
        "version": "1.0.0",
        "description": description,
        "entity_types": sorted({e["type"] for e in entities} | {"Product", "Brand"}),
        "relationship_allowlist": allowlist,
        "canonical_entities": entities,
    }
    for pairs in allowlist.values():
        for a, b in pairs:
            if a not in payload["entity_types"]:
                payload["entity_types"].append(a)
            if b not in payload["entity_types"]:
                payload["entity_types"].append(b)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def ontology_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "schemas" / "ontologies"
    root.mkdir(parents=True)
    _write_ontology(
        root / "shop.json",
        name="shop-ontology",
        description="E-commerce retail products brands prices availability stock",
        entities=[
            {
                "id": "brand:acme",
                "type": "Brand",
                "name": "Acme Fashion",
                "aliases": ["ACME", "Acme"],
            },
            {
                "id": "mat:cotton",
                "type": "Material",
                "name": "Cotton",
                "aliases": ["100% cotton"],
            },
        ],
        allowlist={
            "HAS_BRAND": [["Product", "Brand"]],
            "HAS_MATERIAL": [["Product", "Material"]],
        },
    )
    _write_ontology(
        root / "plm.json",
        name="plm-ontology",
        description="PLM sampling knitwear gauge machine production specifications",
        entities=[
            {
                "id": "buyer:nordic",
                "type": "Buyer",
                "name": "Nordic Retail",
                "aliases": ["Nordic"],
            }
        ],
        allowlist={"HAS_BUYER": [["Product", "Buyer"]]},
    )
    monkeypatch.setenv("FASHIONGPT_ONTOLOGY_GLOB", str(root / "*.json"))
    invalidate_ontology_registry_cache()
    return root


def test_build_registry_scans_glob(ontology_dir: Path):
    reg = build_ontology_registry(None)
    assert isinstance(reg, OntologyRegistry)
    assert {d.ontology_id for d in reg.list_domains()} == {"shop", "plm"}
    assert "acme" in reg.canonical_index
    assert "cotton" in reg.canonical_index


def test_domain_for_question_scores_description(ontology_dir: Path):
    reg = get_ontology_registry(None, force=True)
    shopish = reg.domain_for_question("What is the retail price availability stock for brands?")
    assert shopish and shopish[0].ontology_id == "shop"
    plmish = reg.domain_for_question("Show knitwear gauge and sampling machine specs")
    assert plmish and plmish[0].ontology_id == "plm"


def test_allowlist_for_domains(ontology_dir: Path):
    reg = get_ontology_registry(None, force=True)
    allow = reg.allowlist_for_domains(["shop"])
    assert "HAS_BRAND" in allow
    assert ["Product", "Brand"] in allow["HAS_BRAND"]
    assert "HAS_BUYER" not in allow
