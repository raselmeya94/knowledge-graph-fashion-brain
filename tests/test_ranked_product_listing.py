"""Top-N / highest-price product listings must not collapse to kg_stats counts."""

from __future__ import annotations

from fashion_brain.ask import (
    _is_price_history_question,
    is_ranked_product_listing_ask,
    product_price_sort_direction,
    ranked_product_listing_limit,
)
from knowledge_graph.retrieval.ecom import price_history_mode


def test_top_10_highest_prices_is_ranked_listing_not_history():
    q = "Show me the top 10 products with the highest prices among them."
    assert is_ranked_product_listing_ask(q) is True
    assert ranked_product_listing_limit(q) == 10
    assert product_price_sort_direction(q) == "desc"
    assert _is_price_history_question(q) is False
    assert price_history_mode(q) is None


def test_how_many_still_count_not_ranked():
    q = "How many products does Primark have?"
    assert is_ranked_product_listing_ask(q) is False
    assert _is_price_history_question(q) is False


def test_top_10_brand_names_is_facet_rank_not_product_listing():
    from fashion_brain.ask import is_facet_value_ranking_ask, question_has_catalog_deixis

    q = "list top 10 brand name of them"
    assert is_ranked_product_listing_ask(q) is False
    parsed = is_facet_value_ranking_ask(q, "plmex_products")
    assert parsed is not None
    assert parsed[0] == "brand"
    assert parsed[1] == 10
    assert question_has_catalog_deixis(q) is True


def test_brand_token_does_not_explicitly_name_catalog():
    from fashion_brain.ask import question_names_catalog_explicitly

    assert question_names_catalog_explicitly("list top 10 brand name of them") is False
    assert question_names_catalog_explicitly("how many brands in plmex") is True
    # normalize_name splits "e-commerce" → "e commerce"; must still count as explicit.
    assert (
        question_names_catalog_explicitly(
            "Which products had the biggest price drops in e-commerce?"
        )
        is True
    )


def test_kg_stats_gate_skips_ranked_listing():
    from fashion_brain.orchestration.catalog_pipeline import _try_kg_stats_meta_answer
    from fashion_brain.agents.contracts import Intention, ValidatedIntent

    hit = _try_kg_stats_meta_answer(
        None,
        question="Show me the top 10 products with the highest prices for Primark.",
        validated=ValidatedIntent(
            filters={"brand": "Primark"},
            objective="execute",
            metrics=["count"],
            effective_question="Show me the top 10 products with the highest prices for Primark.",
        ),
        graph_id="ecom_products",
        state={},
        intention=Intention(
            intent="product_count",
            filters={"brand": "Primark"},
            limit=10,
            metrics=["count"],
        ),
    )
    assert hit is None


def test_metadata_brand_ranking_restores_canonical_display_labels(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from fashion_brain.orchestration.catalog_pipeline import _try_kg_stats_facet_ranking

    graph_dir = tmp_path / "kg_stats" / "plmex_products"
    graph_dir.mkdir(parents=True)
    (graph_dir / "_meta.json").write_text(
        json.dumps({"graph_id": "plmex_products", "totals": {"products": 3000}}),
        encoding="utf-8",
    )
    (graph_dir / "brand.json").write_text(
        json.dumps(
            {
                "graph_id": "plmex_products",
                "filter": "brand",
                "unique_count": 3,
                "values": {"primark": 1397, "c a": 597, "h m": 219},
                "display_values": {
                    "primark": "PRIMARK",
                    "c a": "C&A",
                    "h m": "H&M",
                },
            }
        ),
        encoding="utf-8",
    )
    ontology = SimpleNamespace(
        canonical_entities=[
            SimpleNamespace(type="Brand", name="PRIMARK", aliases=["PRIMARK"]),
            SimpleNamespace(type="Brand", name="C&A", aliases=["C&A"]),
            SimpleNamespace(type="Brand", name="H&M", aliases=["H&M"]),
        ]
    )

    class _Registry:
        def get_graph(self, _graph_id):
            return SimpleNamespace(ontology_id="plmex")

        def load_ontology_object(self, _ontology_id):
            return ontology

    platform = SimpleNamespace(
        settings=SimpleNamespace(generated=tmp_path),
        kg_registry=_Registry(),
    )
    monkeypatch.setattr("fashion_brain.ask._facet_keys_from_kg_stats", lambda _gid=None: ["brand"])

    hit = _try_kg_stats_facet_ranking(
        platform,
        question="list top 10 brand name in plmex",
        graph_id="plmex_products",
        state={},
    )

    assert hit is not None
    retrieved, _intention, _graph_id = hit
    # Ontology / display_values restore readable brand surfaces (not scrubbed "c a").
    assert "| 1 | Primark | 1397 |" in retrieved.text or "| 1 | PRIMARK | 1397 |" in retrieved.text
    assert "| 2 | C&A | 597 |" in retrieved.text
    assert "| 3 | H&M | 219 |" in retrieved.text
    assert "C&A=597" in retrieved.evidence[0].excerpt
    assert "H&M=219" in retrieved.evidence[0].excerpt
    assert "c a=597" not in retrieved.evidence[0].excerpt
    assert "h m=219" not in retrieved.evidence[0].excerpt


def test_metadata_garment_type_ranking_polishes_scrubbed_names(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from fashion_brain.orchestration.catalog_pipeline import _try_kg_stats_facet_ranking

    graph_dir = tmp_path / "kg_stats" / "plmex_products"
    graph_dir.mkdir(parents=True)
    (graph_dir / "_meta.json").write_text(
        json.dumps({"graph_id": "plmex_products", "totals": {"products": 20000}}),
        encoding="utf-8",
    )
    (graph_dir / "garment_type.json").write_text(
        json.dumps(
            {
                "graph_id": "plmex_products",
                "filter": "garment_type",
                "unique_count": 4,
                "values": {
                    "jumper": 8494,
                    "top cardigan": 3961,
                    "jacket light weight bomber": 1908,
                    "hoodies sweatshirts": 327,
                },
                "display_values": {
                    "jumper": "Jumper",
                    "top cardigan": "Top Cardigan",
                    "jacket light weight bomber": "Jacket/Light Weight/Bomber",
                    "hoodies sweatshirts": "Hoodies & Sweatshirts",
                },
            }
        ),
        encoding="utf-8",
    )
    platform = SimpleNamespace(
        settings=SimpleNamespace(generated=tmp_path),
        kg_registry=None,
    )
    monkeypatch.setattr(
        "fashion_brain.ask._facet_keys_from_kg_stats",
        lambda _gid=None: ["garment_type"],
    )

    hit = _try_kg_stats_facet_ranking(
        platform,
        question="list top 10 garment types in plmex",
        graph_id="plmex_products",
        state={},
    )

    assert hit is not None
    retrieved, _intention, _graph_id = hit
    assert "| 1 | Jumper | 8494 |" in retrieved.text
    assert "| 2 | Top Cardigan | 3961 |" in retrieved.text
    assert "| 3 | Jacket/Light Weight/Bomber | 1908 |" in retrieved.text
    assert "Hoodies & Sweatshirts" in retrieved.text
    excerpt = retrieved.evidence[0].excerpt or ""
    assert "Jumper=8494" in excerpt
    assert "Jacket/Light Weight/Bomber=1908" in excerpt
    assert "jumper=8494" not in excerpt
    assert "jacket light weight bomber=1908" not in excerpt
