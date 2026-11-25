"""Dynamic product example table columns (filters + question hints)."""

from __future__ import annotations

from types import SimpleNamespace

from knowledge_graph.retrieval.ecom import _product_table_lines


def _node(**props):
    return SimpleNamespace(properties=dict(props), id=props.get("name") or "n")


def test_brand_filter_drops_brand_column():
    sample = [
        _node(name="Crew Tee", brand="Zara", price="$20", availability="InStock", color="White"),
        _node(name="Slim Tee", brand="Zara", price="$22", availability="Mixed", color="Black"),
    ]
    table = _product_table_lines(
        sample, {"brand": "Zara"}, "Show Zara products", limit=5
    )
    header = table[0]
    assert "Brand" not in header
    assert "Name" in header
    assert "Price" in header
    assert "Availability" in header
    assert all("Zara" not in row or row.startswith("| ---") for row in table[2:])


def test_question_adds_color_when_not_pinned():
    sample = [
        _node(
            name="Cotton Shirt",
            brand="Primark",
            price="$15",
            availability="InStock",
            color="Navy",
            material="Cotton",
        ),
    ]
    table = _product_table_lines(
        sample,
        {"material": "cotton"},
        "cotton shirts with colors",
        limit=5,
    )
    header = table[0]
    assert "Brand" in header
    assert "Color" in header
    assert "Material" not in header  # pinned by filter
    assert "Navy" in table[2]


def test_multi_party_keeps_brand_column():
    sample = [
        _node(name="A", brand="Primark", price="$10", availability="InStock"),
        _node(name="B", brand="Zara", price="$12", availability="InStock"),
    ]
    table = _product_table_lines(
        sample, {"brand": "Primark | Zara"}, "compare Primark and Zara", limit=5
    )
    assert "Brand" in table[0]


def test_force_price_keeps_empty_price_when_ranking():
    sample = [_node(name="Sample", brand="Zara", availability="InStock")]
    table = _product_table_lines(
        sample, {"brand": "Zara"}, "highest price Zara", limit=5, force_price=True
    )
    assert "Price" in table[0]
