"""Generic filters contract: kg_stats global / single-dim / multi-dim cache-miss."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from knowledge_graph.access import Principal
from knowledge_graph.models import EvidenceItem
from knowledge_graph.retrieval import kg_stats
from knowledge_graph.retrieval.ecom import append_neo4j_count_answer, clean_product_filters
from fashion_brain.agents.contracts import Intention, IntentSummary, ValidatedIntent
from fashion_brain.orchestration.catalog_pipeline import (
    _meta_fast_path_eligible,
    _merge_query_filters,
    _reject_unfiltered_global_hit,
    _try_kg_stats_meta_answer,
)


INTERNAL = Principal(subject="t", clearance="internal")


def _write_stats(tmp_path: Path, graph_id: str = "ecom_products") -> Path:
    payload = {
        "version": 3,
        "graph_id": graph_id,
        "graph_name": "E-Commerce",
        "description": "Retail catalog",
        "source": "test",
        "updated_at": "2026-11-18T00:00:00Z",
        "totals": {"products": 18183, "brands": 3, "columns_indexed": 3},
        "unique_counts": {"brand": 3, "category": 2, "season": 2},
        "value_counts": {
            "brand": {"primark": 4200, "zara": 3100, "matalan": 900},
            "category": {"Sweater": 800, "Polo": 400},
            "season": {"SS25": 1200, "AW25": 900},
        },
    }
    payload = kg_stats.attach_products_by_breakdowns(payload)
    root = tmp_path / "kg_stats"
    root.mkdir(parents=True)
    (root / f"{graph_id}.json").write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


def test_attach_products_by_breakdowns_mirrors_dimensions(tmp_path):
    generated = _write_stats(tmp_path)
    stats = kg_stats.load_graph_stats("ecom_products", generated)
    assert stats is not None
    assert stats["products_by_brand"]["primark"] == 4200
    assert stats["products_by_category"]["Sweater"] == 800
    assert stats["products_by_season"]["SS25"] == 1200


def test_unfiltered_count_hits_global_kg_stats(tmp_path):
    generated = _write_stats(tmp_path)
    result = kg_stats.answer_simple_count("ecom_products", filters={}, generated=generated)
    assert result is not None
    assert result["kind"] == "product_total"
    assert result["n"] == 18183
    assert result["applied_filters"] == {}
    assert result["requested_filters"] == {}


def test_single_filter_brand_and_category_and_season(tmp_path):
    generated = _write_stats(tmp_path)
    brand = kg_stats.answer_simple_count(
        "ecom_products", filters={"brand": "Primark"}, generated=generated
    )
    assert brand is not None
    assert brand["n"] == 4200
    assert brand["filter"].casefold() == "filter: brand=primark"
    assert str(brand["applied_filters"].get("brand") or "").casefold() == "primark"

    category = kg_stats.answer_simple_count(
        "ecom_products", filters={"category": "sweater"}, generated=generated
    )
    assert category is not None
    assert category["n"] == 800
    assert str(category["applied_filters"]["category"]).casefold() == "sweater"

    season = kg_stats.answer_simple_count(
        "ecom_products", filters={"season": "SS25"}, generated=generated
    )
    assert season is not None
    assert season["n"] == 1200


def test_multi_filter_is_cache_miss(tmp_path):
    generated = _write_stats(tmp_path)
    result = kg_stats.answer_simple_count(
        "ecom_products",
        filters={"brand": "primark", "category": "Sweater"},
        generated=generated,
    )
    assert result is None


def test_resolve_filters_keeps_multi_keys_and_explicit_color(tmp_path):
    """Must not collapse brand+category → category alone, or color→category."""
    generated = _write_stats(tmp_path)
    # Add a color shard so color stays color (not rebound).
    stats = kg_stats.load_graph_stats("ecom_products", generated)
    assert stats is not None
    stats["value_counts"]["color"] = {"white": 50, "White Shirt": 3}
    stats = kg_stats.attach_products_by_breakdowns(stats)
    (generated / "kg_stats" / "ecom_products.json").write_text(
        __import__("json").dumps(stats), encoding="utf-8"
    )
    kg_stats.load_graph_stats.cache_clear() if hasattr(kg_stats.load_graph_stats, "cache_clear") else None

    multi = kg_stats.resolve_filters_against_stats(
        "ecom_products",
        {"brand": "primark", "garment": "sweater"},
        generated=generated,
    )
    assert "brand" in multi
    assert len(multi) >= 2

    color = kg_stats.resolve_filters_against_stats(
        "ecom_products",
        {"color": "white"},
        generated=generated,
    )
    assert color.get("color")
    assert "category" not in color


def test_same_party_brand_and_buyer_filters_are_deduplicated():
    assert clean_product_filters({"brand": "Primark", "buyer": "primark"}) == {
        "brand": "Primark"
    }
    assert clean_product_filters({"brand": "Primark", "buyer": "Zara"}) == {
        "brand": "Primark",
        "buyer": "Zara",
    }


def test_uncached_dimension_is_cache_miss(tmp_path):
    generated = _write_stats(tmp_path)
    result = kg_stats.answer_simple_count(
        "ecom_products",
        filters={"factory": "Dhaka Mill"},
        generated=generated,
    )
    assert result is None


def test_resolve_entity_to_filter_generic(tmp_path):
    generated = _write_stats(tmp_path)
    hit = kg_stats.resolve_entity_to_filter("ecom_products", "primark", generated)
    assert hit == {"brand": "primark"}
    miss = kg_stats.resolve_entity_to_filter("ecom_products", "unknown-party", generated)
    assert miss is None


def test_append_single_filter_shows_filter_provenance(tmp_path):
    generated = _write_stats(tmp_path)
    platform = SimpleNamespace(settings=SimpleNamespace(generated=generated), kg_registry=None)
    lines: list[str] = []
    evidence: list[EvidenceItem] = []
    with patch("knowledge_graph.retrieval.ecom.cypher_count_products") as cypher:
        append_neo4j_count_answer(
            platform,
            INTERNAL,
            "ecom_products",
            {"brand": "primark", "buyer": "primark"},
            None,
            None,
            lines,
            evidence,
            [],
            question="How many products does Primark have?",
        )
    assert "4,200" in lines[0]
    assert "buyer" not in lines[0].casefold()
    assert evidence[0].record_id == "kg_stats"
    assert "filter: brand=primark" in evidence[0].excerpt
    assert evidence[0].applied_filters.get("brand") == "primark"
    assert "buyer" not in evidence[0].applied_filters
    assert evidence[0].requested_filters.get("brand") == "primark"
    cypher.assert_not_called()


def test_append_uncached_dimension_does_not_return_global(tmp_path):
    generated = _write_stats(tmp_path)
    platform = SimpleNamespace(settings=SimpleNamespace(generated=generated), kg_registry=None)
    lines: list[str] = []
    evidence: list[EvidenceItem] = []

    with (
        patch("knowledge_graph.retrieval.ecom.cypher_count_products", return_value=42),
        patch("knowledge_graph.retrieval.ecom.cypher_group_by", return_value=None),
        patch(
            "knowledge_graph.retrieval.ecom.search_neo4j_products",
            return_value=([], 42),
        ),
    ):
        append_neo4j_count_answer(
            platform,
            INTERNAL,
            "ecom_products",
            {"factory": "Dhaka Mill"},
            None,
            None,
            lines,
            evidence,
            [],
            question="How many products from Dhaka Mill?",
        )
    assert lines
    assert "42" in lines[0]
    # Must not claim the global 18,183 total.
    assert "18,183" not in lines[0]
    assert not any(
        getattr(e, "record_id", None) == "kg_stats" and "products=18183" in str(e.excerpt)
        for e in evidence
    )


def test_merge_query_filters_preserves_thinking_scope():
    validated = ValidatedIntent(filters={}, metrics=["count"], source="ecom_products")
    summary = IntentSummary(
        actual_intent="count Primark",
        text="Scope: brand = primark",
        filters={"brand": "primark", "season": "SS25"},
    )
    merged = _merge_query_filters(validated=validated, summary=summary)
    assert merged["brand"] == "primark"
    assert merged["season"] == "SS25"


def test_reject_unfiltered_global_hit():
    evidence = [
        EvidenceItem(
            source_id="ecom_products",
            record_id="kg_stats",
            excerpt="products=18183",
            verification_status="observed",
            requested_filters={"brand": "primark"},
            applied_filters={},
        )
    ]
    assert _reject_unfiltered_global_hit({"brand": "primark"}, evidence) is True
    evidence[0].applied_filters = {"brand": "primark"}
    evidence[0].excerpt = "filter: brand=primark → 4200 (from generated/kg_stats)"
    assert _reject_unfiltered_global_hit({"brand": "primark"}, evidence) is False


def test_meta_answer_uses_thinking_filters_not_global(tmp_path):
    generated = _write_stats(tmp_path)
    platform = SimpleNamespace(settings=SimpleNamespace(generated=generated), kg_registry=None)
    validated = ValidatedIntent(
        source="ecom_products",
        filters={},  # QU missed the brand — systemic bug case
        metrics=["count"],
        analysis_depth="unset",
        effective_question="How many products does Primark have?",
    )
    summary = IntentSummary(
        actual_intent="Count Primark products",
        text="Scope: brand = primark",
        filters={"brand": "primark"},
    )
    hit = _try_kg_stats_meta_answer(
        platform,
        question="How many products does Primark have?",
        validated=validated,
        graph_id=None,
        state={},
        summary=summary,
    )
    assert hit is not None
    retrieved, intention, gid = hit
    assert gid == "ecom_products"
    assert intention.filters.get("brand") == "primark"
    assert "4,200" in (retrieved.text or "")
    assert "18,183" not in (retrieved.text or "")
    assert any(
        getattr(e, "applied_filters", {}).get("brand") == "primark" for e in retrieved.evidence
    )


def test_meta_answer_multi_filter_returns_none(tmp_path):
    generated = _write_stats(tmp_path)
    platform = SimpleNamespace(settings=SimpleNamespace(generated=generated), kg_registry=None)
    validated = ValidatedIntent(
        source="ecom_products",
        filters={"brand": "primark", "category": "Sweater"},
        metrics=["count"],
    )
    hit = _try_kg_stats_meta_answer(
        platform,
        question="How many Primark sweaters?",
        validated=validated,
        graph_id=None,
        state={},
    )
    assert hit is None


def test_meta_answer_entity_resolves_without_hardcoding_brand(tmp_path):
    generated = _write_stats(tmp_path)
    platform = SimpleNamespace(settings=SimpleNamespace(generated=generated), kg_registry=None)
    validated = ValidatedIntent(
        source="ecom_products",
        entity="primark",
        filters={},
        metrics=["count"],
    )
    hit = _try_kg_stats_meta_answer(
        platform,
        question="How many products does Primark have?",
        validated=validated,
        graph_id=None,
        state={},
    )
    assert hit is not None
    _retrieved, intention, _gid = hit
    assert intention.filters.get("brand") == "primark"


def test_meta_answer_reuses_validated_stats_result(tmp_path):
    from fashion_brain.agents.contracts import ValidatedIntent

    generated = _write_stats(tmp_path)
    platform = SimpleNamespace(settings=SimpleNamespace(generated=generated), kg_registry=None)
    validated = ValidatedIntent(
        source="ecom_products",
        metrics=["count"],
        effective_question="How many products are in the catalog?",
    )
    answer_simple_count = kg_stats.answer_simple_count
    with (
        patch(
            "knowledge_graph.retrieval.kg_stats.answer_simple_count",
            wraps=answer_simple_count,
        ) as count_lookup,
        patch("fashion_brain.ask.live_brand_names", return_value=[]),
        patch(
            "fashion_brain.ask.enrich_scope_from_question",
            side_effect=lambda _question, filters, **_kwargs: (filters, None),
        ),
    ):
        hit = _try_kg_stats_meta_answer(
            platform,
            question="How many products are in the catalog?",
            validated=validated,
            graph_id=None,
            state={},
        )

    assert hit is not None
    retrieved, _intention, _gid = hit
    assert "18,183" in retrieved.text
    assert count_lookup.call_count == 1
    # Meta fast path now attaches selected filter shards so answers can show related key=value.
    assert any(item.record_id == "optional_metadata" for item in retrieved.evidence)


def test_meta_fast_path_does_not_skip_context_resolution():
    question = "How many products are in the catalog?"
    assert _meta_fast_path_eligible(question, {})
    assert not _meta_fast_path_eligible(
        "How many are red?",
        {"active_filters": {"garment": "sweater"}, "last_result_id": "previous"},
        [{"role": "user", "content": "How many sweater products?"}],
    )


def test_meta_answer_category_includes_related_key_value(tmp_path):
    """Filtered kg_stats counts should carry optional_metadata (unique/top/products_total)."""

    generated = _write_stats(tmp_path)
    platform = SimpleNamespace(settings=SimpleNamespace(generated=generated), kg_registry=None)
    validated = ValidatedIntent(
        source="ecom_products",
        filters={"category": "Sweater"},
        metrics=["count"],
        analysis_depth="unset",
        effective_question="How many products with category Sweater?",
    )
    with (
        patch("fashion_brain.ask.live_brand_names", return_value=[]),
        patch(
            "fashion_brain.ask.enrich_scope_from_question",
            side_effect=lambda _question, filters, **_kwargs: (filters, None),
        ),
    ):
        hit = _try_kg_stats_meta_answer(
            platform,
            question="How many products with category Sweater?",
            validated=validated,
            graph_id="ecom_products",
            state={},
        )
    assert hit is not None
    retrieved, intention, gid = hit
    assert gid == "ecom_products"
    assert intention.filters.get("category")
    assert "800" in (retrieved.text or "")
    meta_items = [e for e in retrieved.evidence if e.record_id == "optional_metadata"]
    assert meta_items
    excerpt = " ".join(str(e.excerpt or "") for e in meta_items).casefold()
    assert "products_total=" in excerpt or "category:" in excerpt
    assert "unique=" in excerpt or "top:" in excerpt


def test_meta_answer_distinct_facet_includes_related_key_value(tmp_path):
    generated = _write_stats(tmp_path)
    platform = SimpleNamespace(settings=SimpleNamespace(generated=generated), kg_registry=None)
    validated = ValidatedIntent(
        source="ecom_products",
        entity="category",
        filters={},
        metrics=["count"],
        analysis_depth="unset",
        effective_question="How many categories are in the catalog?",
    )
    with (
        patch("fashion_brain.ask.live_brand_names", return_value=[]),
        patch(
            "fashion_brain.ask.enrich_scope_from_question",
            side_effect=lambda _question, filters, **_kwargs: (filters, None),
        ),
        patch("fashion_brain.ask._facet_entity_count", return_value="category"),
    ):
        hit = _try_kg_stats_meta_answer(
            platform,
            question="How many categories are in the catalog?",
            validated=validated,
            graph_id="ecom_products",
            state={},
        )
    assert hit is not None
    retrieved, intention, _gid = hit
    assert intention.count_entity == "category" or not intention.filters
    meta_items = [e for e in retrieved.evidence if e.record_id == "optional_metadata"]
    assert meta_items
    excerpt = " ".join(str(e.excerpt or "") for e in meta_items).casefold()
    assert "category" in excerpt or "products_total=" in excerpt
