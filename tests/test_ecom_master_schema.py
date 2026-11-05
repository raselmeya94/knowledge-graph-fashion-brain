from pathlib import Path

from knowledge_graph.adapters.ecom_master import build_relationships, graph_sketch, normalize_master_row, product_id
from knowledge_graph.ontology import load_ontology


def test_ecom_ontology_loads():
    path = Path("data/schemas/ecom-products-ontology.json")
    ontology = load_ontology(path)
    assert "Product" in ontology.entity_types
    assert "HAS_BRAND" in ontology.relationship_allowlist
    assert "USES_MATERIAL" in ontology.relationship_allowlist


def test_master_row_builds_relationships():
    row = {
        "brand": "primark",
        "master_id": "abc123",
        "name": "Crew Neck Jumper",
        "category": "Knitwear",
        "family": "Sweaters",
        "subfamily": "Jumpers",
        "composition": "60% Acrylic 40% Polyester",
        "composition_detail": [{"name": "Acrylic", "percentage": 60}, {"name": "Polyester", "percentage": 40}],
        "colors": ["Red", "Navy"],
        "availability": "in_stock",
        "currency": "GBP",
        "price": "8.00",
        "section": "Women",
        "domain": "primark.com",
        "catalog_id": "sweaters-2026",
        "construction": "knit",
    }
    product = normalize_master_row(row)
    assert product["product_id"] == product_id("primark", "abc123")
    assert product["color"] == "Red"

    rels = build_relationships(product)
    types = {item["rel_type"] for item in rels}
    assert "HAS_BRAND" in types
    assert "BELONGS_TO_CATEGORY" in types
    assert "HAS_FAMILY" in types
    assert "HAS_SUB_CATEGORY" in types
    assert "USES_MATERIAL" in types
    assert "HAS_COLOR" in types
    assert "PARENT_CATEGORY" in types
    assert "HAS_AVAILABILITY" in types
    assert "PRICED_IN" in types

    materials = [item for item in rels if item["rel_type"] == "USES_MATERIAL"]
    assert {item["end_name"] for item in materials} == {"Acrylic", "Polyester"}
    colors = [item for item in rels if item["rel_type"] == "HAS_COLOR"]
    assert {item["end_name"] for item in colors} == {"Red", "Navy"}

    sketch = graph_sketch(product)
    assert sketch["product_id"].startswith("ecom:primark:")
    assert sketch["properties"]["price"] == "8.00"
