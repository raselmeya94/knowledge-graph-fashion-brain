import pytest

from knowledge_graph.config import DATA_ROOT
from knowledge_graph.errors import ValidationFailed
from knowledge_graph.hashing import sha256_bytes
from knowledge_graph.models import SourceRegistry
from knowledge_graph.ontology import load_ontology, validate_ontology
from knowledge_graph.registry import load_manifest, load_registry, validate_registry
from tests.conftest import RAW, ROOT


def test_registry_and_ontology_load():
    registry = load_registry(RAW / "source_registry.json")
    ids = [source.source_id for source in registry.sources]
    assert "own_products" in ids
    assert "ecom_products" in ids
    assert registry.by_id("own_products").future_connectors == ["authorized_sql", "authorized_api"]
    ontology = load_ontology(DATA_ROOT / "schemas" / "product-knowledge-ontology.json")
    assert "Product" in ontology.entity_types
    assert "EventTag" in ontology.entity_types
    assert "NewsArticle" not in ontology.entity_types
    assert "USES_MATERIAL" in ontology.relationship_allowlist
    assert "HAS_EVENT_TAG" in ontology.relationship_allowlist
    validate_ontology(ontology)
    ecom = load_ontology(DATA_ROOT / "schemas" / "ecom-products-ontology.json")
    assert "Brand" in ecom.entity_types
    validate_ontology(ecom)


def test_registry_rejects_duplicates():
    registry = load_registry(RAW / "source_registry.json")
    registry.sources.append(registry.sources[0])
    with pytest.raises(ValidationFailed):
        validate_registry(registry)


def test_registry_schema_contains_connector_fields():
    schema = SourceRegistry.model_json_schema()
    assert "sources" in schema["properties"]


def test_manifest_hashes_match_files():
    for manifest_path in RAW.glob("*/manifest.json"):
        manifest = load_manifest(manifest_path)
        assert manifest.records
        for entry in manifest.records:
            path = manifest_path.parent / entry.path
            assert path.exists()
            assert entry.ingestion_status == "registered"
            assert len(entry.content_hash) == 64
            if entry.hash_method == "file":
                assert sha256_bytes(path.read_bytes()) == entry.content_hash
            assert entry.record_id == "*"
            assert entry.path == "products_data.csv"
            assert entry.synthetic is False


def test_catalog_headers_are_the_raw_columns():
    from knowledge_graph.adapters.products import RAW_COLUMNS

    path = RAW / "own_products" / "products_data.csv"
    if not path.is_file():
        pytest.skip("products_data.csv is not present under data/raw/own_products/")
    # DictReader-safe: every adapter column must appear in the CSV header.
    header = path.read_text(encoding="utf-8-sig").splitlines()[0].split(",")
    missing = [column for column in RAW_COLUMNS if column not in header]
    assert not missing, f"CSV missing columns required by RAW_COLUMNS: {missing}"


def test_ai_tag_columns_are_registered():
    from knowledge_graph.adapters.products import RAW_COLUMNS, normalize_product

    for column in (
        "AI Event Tags",
        "AI Construction Tags",
        "AI Material Tags",
        "AI Finish Tags",
        "AI Special Tags",
    ):
        assert column in RAW_COLUMNS
    product = normalize_product(
        {
            "Code": "T-1",
            "AI Event Tags": "spring, summer, smart casual",
            "AI Construction Tags": "polo shirt, menswear",
            "AI Material Tags": "knit fabric | jacquard knit fabric",
            "AI Finish Tags": "geometric print, allover print",
            "AI Special Tags": "",
        }
    )
    assert product["ai_event_tags"] == ["spring", "summer", "smart casual"]
    assert product["ai_construction_tags"] == ["polo shirt", "menswear"]
    assert product["ai_material_tags"] == ["knit fabric", "jacquard knit fabric"]
    assert product["ai_finish_tags"] == ["geometric print", "allover print"]
    assert product["ai_special_tags"] == []
