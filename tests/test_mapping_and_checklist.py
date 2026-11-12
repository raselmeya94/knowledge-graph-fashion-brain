from knowledge_graph.config import DATA_ROOT
from knowledge_graph.sources.mapping_store import get_mapping, mapping_exists, put_mapping


def test_put_and_get_mapping_roundtrip():
    source_id = "ui_test_source"
    path = DATA_ROOT / "schemas" / f"{source_id}_mapping.json"
    if path.exists():
        path.unlink()
    payload = {"source_table": "demo", "columns": {"a": "b"}}
    created = put_mapping(source_id, payload, replace=False)
    assert created["status"] == "created"
    assert mapping_exists(source_id)
    loaded = get_mapping(source_id)
    assert loaded["mapping"]["source_table"] == "demo"
    assert loaded["mapping"]["source_id"] == source_id
    replaced = put_mapping(source_id, {**payload, "version": 2}, replace=True)
    assert replaced["status"] == "replaced"
    assert get_mapping(source_id)["mapping"]["version"] == 2
    path.unlink(missing_ok=True)


def test_ecom_mapping_put_uses_legacy_filename():
    import json
    from pathlib import Path

    src = Path("postman/bodies/ecom_master_products_mapping.json")
    payload = json.loads(src.read_text(encoding="utf-8"))
    result = put_mapping("ecom_products", payload, replace=True)
    path = DATA_ROOT / "schemas" / "ecom_master_products_mapping.json"
    assert path.is_file()
    assert result["path"].endswith("ecom_master_products_mapping.json")
    assert mapping_exists("ecom_products") is True
    body = get_mapping("ecom_products")
    assert body["mapping"].get("source_table") == "master_products"
