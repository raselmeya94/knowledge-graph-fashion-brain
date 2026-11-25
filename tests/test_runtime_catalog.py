"""Unit tests for runtime catalog persist / reverse index / term grounding (no Neo4j)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import knowledge_graph.retrieval.runtime_catalog as rc
from knowledge_graph.retrieval.runtime_catalog import (
    RuntimeCatalog,
    ValueSchemaHit,
    _build_reverse_index,
    _persist,
    ground_query_terms,
    invalidate_runtime_catalog,
)


class _FakeSettings:
    runtime_catalog_dir = Path("generated/runtime_catalog")
    runtime_catalog_ttl_seconds = 600
    runtime_catalog_max_values_per_key = 100
    runtime_catalog_force_refresh = False
    schema_ttl_seconds = 600
    generated = Path("generated")

    def runtime_catalog_ttl(self) -> int:
        return 600

    @property
    def runtime_catalog(self) -> Path:
        # Overridden per-test via monkeypatch of catalog_root.
        return Path("generated/runtime_catalog")


class _FakePlatform:
    settings = _FakeSettings()
    graph = None


def _sample_catalog() -> RuntimeCatalog:
    return RuntimeCatalog(
        manifest={"ttl_seconds": 600, "graph_ids": ["ecom_products"]},
        live_schema={
            "labels": ["Product", "Brand", "Material"],
            "relationship_types": ["HAS_BRAND", "USES_MATERIAL"],
            "relationship_directions": [
                {"start": "Product", "type": "HAS_BRAND", "end": "Brand", "source": "ontology"}
            ],
            "graph_values": ["ecom_products"],
        },
        values_by_graph={
            "ecom_products": {
                "Brand.name": [{"value": "Zara", "count": 100}, {"value": "Primark", "count": 80}],
                "Product.category_path": [
                    {"value": "Women > Tops", "count": 40},
                    {"value": "Women > Dresses", "count": 20},
                ],
                "Material.name": [{"value": "cotton", "count": 50}, {"value": "polyester", "count": 60}],
            }
        },
        built_at=time.time(),
        source="test",
    )


def test_persist_and_load_roundtrip(tmp_path, monkeypatch):
    invalidate_runtime_catalog()
    monkeypatch.setattr(rc, "catalog_root", lambda platform=None, settings=None: tmp_path)

    catalog = _sample_catalog()
    catalog.reverse_index = _build_reverse_index(catalog, platform=None)
    _persist(catalog, tmp_path)

    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "live_schema.json").is_file()
    assert (tmp_path / "values" / "ecom_products.json").is_file()

    loaded = rc._load_from_disk(tmp_path, ttl=10_000)
    assert loaded is not None
    assert loaded.source == "disk"
    assert "ecom_products" in loaded.values_by_graph
    assert loaded.index_size() >= 1
    # Exact brand key present after reverse index rebuild.
    assert "zara" in loaded.reverse_index


def test_reverse_index_merges_db_values():
    from knowledge_graph.ontology import normalize_name

    catalog = _sample_catalog()
    index = _build_reverse_index(catalog, platform=None)
    hits = index.get("zara") or []
    assert hits
    assert any(h.label == "Brand" and h.property == "name" for h in hits)
    # normalize_name collapses punctuation → "women tops"
    tops = index.get(normalize_name("Women > Tops")) or []
    assert tops
    assert any("Tops" in h.exact_value or "tops" in h.exact_value.casefold() for h in tops)


def test_ground_query_terms_exact_and_fuzzy():
    catalog = _sample_catalog()
    catalog.reverse_index = _build_reverse_index(catalog, platform=None)
    # Inject a near-miss key for fuzzy
    catalog.reverse_index.setdefault("polyestar", catalog.reverse_index.get("polyester") or [])

    grounded = ground_query_terms(
        "Which Zara tops use cotton?",
        catalog=catalog,
        graph_ids=["ecom_products"],
        limit=20,
        threshold=80,
    )
    prompt = grounded.to_prompt_dict()
    assert "zara" in prompt or any("zara" in k for k in prompt)
    assert any("cotton" in k for k in prompt)
    # Compact: only matched terms, not full index.
    assert len(prompt) < 20
    assert len(json.dumps(prompt)) < 4000


def test_stale_disk_triggers_none(tmp_path, monkeypatch):
    invalidate_runtime_catalog()
    monkeypatch.setattr(rc, "catalog_root", lambda platform=None, settings=None: tmp_path)
    catalog = _sample_catalog()
    catalog.built_at = time.time() - 10_000
    catalog.reverse_index = _build_reverse_index(catalog, platform=None)
    _persist(catalog, tmp_path)
    # Overwrite manifest age
    man = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    man["built_epoch"] = time.time() - 10_000
    (tmp_path / "manifest.json").write_text(json.dumps(man), encoding="utf-8")

    loaded = rc._load_from_disk(tmp_path, ttl=60)
    assert loaded is None


def test_term_grounding_map_to_dict():
    hit = ValueSchemaHit(
        label="Brand",
        property="name",
        exact_value="Zara",
        graph_id="ecom_products",
        source="db",
        count=10,
    )
    from knowledge_graph.retrieval.runtime_catalog import TermGroundingMap

    m = TermGroundingMap(terms={"zara": [hit]}, unmatched=["foo"])
    d = m.to_dict()
    assert d["terms"]["zara"][0]["exact_value"] == "Zara"
    assert d["unmatched"] == ["foo"]
