"""Schema-hint prose must not become filters or force Text2Cypher."""

from __future__ import annotations

from fashion_brain.ask import _is_schema_prose_filter, _reliable_filter
from fashion_brain.planning.active_constraints import clean_filters, relationship_hints_for_filters
from knowledge_graph.retrieval.ecom import clean_product_filters


def test_schema_prose_rejected():
    junk = "toward garmentcategory filter"
    assert _is_schema_prose_filter(junk) is True
    assert _reliable_filter(junk) is False
    assert _reliable_filter("sweater") is True


def test_clean_product_filters_drops_schema_prose():
    cleaned = clean_product_filters(
        {
            "brand": "primark",
            "garment": "sweater",
            "category": "toward garmentcategory filter",
            "family": "toward garmentcategory filter",
        }
    )
    assert cleaned.get("brand") == "primark"
    assert cleaned.get("garment") == "sweater"
    assert "category" not in cleaned
    assert "family" not in cleaned


def test_active_constraint_clean_filters_drops_prose():
    out = clean_filters(
        {
            "brand": "primark",
            "garment": "sweater",
            "category": "toward :GarmentCategory",
        }
    )
    assert out == {"brand": "primark", "garment": "sweater"}


def test_relationship_hints_avoid_toward_phrase():
    schema = type(
        "S",
        (),
        {
            "facet_types": ["GarmentCategory"],
            "relationship_allowlist": {"BELONGS_TO_CATEGORY": [["Product", "GarmentCategory"]]},
        },
    )()
    hints = relationship_hints_for_filters(schema, {"category": "sweater"})
    assert "toward" not in hints.casefold()
    assert "GarmentCategory" in hints
