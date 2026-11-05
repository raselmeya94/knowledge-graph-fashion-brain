import json

from fashion_brain.ask import _promote_type_filters, resolve_catalog_plan
from knowledge_graph.access import Principal
from knowledge_graph.retrieval.catalog_index import ensure_catalog_index, load_catalog_rows
from knowledge_graph.retrieval.products import FileProductRepository

INTERNAL = Principal(subject="analyst", clearance="internal")


def test_catalog_index_answers_a_material_filter_without_the_graph(tmp_path):
    folder = tmp_path / "envelopes" / "own_products"
    folder.mkdir(parents=True)
    _write(folder / "1.json", "1001", "Cotton Tee", "100% Cotton")
    _write(folder / "2.json", "1002", "Silk Shirt", "100% Silk")
    ensure_catalog_index(tmp_path)
    rows = load_catalog_rows(tmp_path)
    repository = FileProductRepository(rows)
    matches, total = repository.search(INTERNAL, material="cotton", limit=5)
    assert total == 1
    assert matches[0].record_id == "1001"
    assert matches[0].metadata["product"]["composition"] == "100% Cotton"


def test_sweater_matches_when_filter_uses_garment_type(tmp_path):
    """PLM stores Sweater on category and Jumper on garment_type; red may be 红色."""

    folder = tmp_path / "envelopes" / "own_products"
    folder.mkdir(parents=True)
    _write(
        folder / "1.json",
        "2001",
        "Red Sweater",
        "100% Cotton",
        category="Sweater",
        garment_type="Jumper",
        color="红色",
    )
    _write(
        folder / "2.json",
        "2002",
        "Blue Sweater",
        "100% Cotton",
        category="Sweater",
        garment_type="Jumper",
        color="蓝色",
    )
    ensure_catalog_index(tmp_path)
    repository = FileProductRepository(load_catalog_rows(tmp_path))
    assert repository.count(INTERNAL, garment="sweater", color="red") == 1
    assert repository.count(INTERNAL, garment_type="sweater", color="red") == 1
    assert repository.count(INTERNAL, category="sweater", color="red") == 1


def test_keyword_plan_and_type_promotion_for_sweater_red():
    plan = resolve_catalog_plan("show a sweater product with red color")
    assert plan["filters"]["garment"] == "sweater"
    assert plan["filters"]["color"] == "red"
    promoted = {"garment_type": "sweater", "color": "red"}
    _promote_type_filters(promoted)
    assert promoted == {"garment": "sweater", "color": "red"}


def _write(path, record_id, title, composition, **product):
    payload = {
        "product_id": record_id,
        "product_name": title,
        "material": composition,
        "composition": composition,
        "source_columns": {"noise": "x" * 500},
    }
    payload.update(product)
    path.write_text(
        json.dumps(
            {
                "record_id": record_id,
                "title": title,
                "classification": "internal",
                "source_path": "products_data.csv",
                "content_hash": "a" * 64,
                "synthetic": False,
                "source_type": "own_products",
                "lifecycle": "active",
                "metadata": {"product": payload},
            }
        ),
        encoding="utf-8",
    )
