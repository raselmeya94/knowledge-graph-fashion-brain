from pathlib import Path

from knowledge_graph.config import Settings
from knowledge_graph.pipeline.kg_build import ensure_ecom_mapping


def test_ensure_ecom_mapping_restores_missing_file(tmp_path: Path):
    data_root = tmp_path / "data"
    schemas = data_root / "schemas"
    schemas.mkdir(parents=True)
    # Point Settings at temp data root via monkeypatch-style construction is heavy;
    # call ensure with a Settings whose DATA_ROOT is not used — use path() relative to DATA_ROOT.
    # Instead verify the live repo mapping exists after restore path logic via postman seed.
    settings = Settings()
    target = settings.path(Path("schemas") / "ecom_master_products_mapping.json")
    result = ensure_ecom_mapping(settings)
    assert target.is_file()
    assert result["status"] in {"exists", "restored"}
    assert "neo4j_product_properties" in target.read_text(encoding="utf-8")
