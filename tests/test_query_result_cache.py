"""Unit tests for QueryResultCache (no live Neo4j)."""

from __future__ import annotations

import time
from types import SimpleNamespace

from fashion_brain.retrieval.query_result_cache import (
    cache_key,
    clear_query_result_cache,
    get_cached_result,
    put_cached_result,
)


def setup_function():
    clear_query_result_cache()


def test_cache_key_stable_and_order_independent():
    a = cache_key(
        "How many Primark products?",
        ontology_domains=["ecom", "plmex"],
        graph_ids=["ecom_products"],
        metric_family="catalog",
        intent="product_count",
    )
    b = cache_key(
        "how many  primark products?",
        ontology_domains=["plmex", "ecom"],
        graph_ids=["ecom_products"],
        metric_family="catalog",
        intent="product_count",
    )
    assert a == b


def test_put_get_and_ttl_expiry():
    platform = SimpleNamespace(settings=SimpleNamespace(query_cache_ttl_seconds=0.15))
    key = put_cached_result(
        "How many Zara products?",
        text="Zara has 5804 products.",
        platform=platform,
        ontology_domains=["ecom"],
        graph_ids=["ecom_products"],
        metric_family="catalog",
        intent="product_count",
    )
    assert key
    hit = get_cached_result(
        "How many Zara products?",
        platform=platform,
        ontology_domains=["ecom"],
        graph_ids=["ecom_products"],
        metric_family="catalog",
        intent="product_count",
    )
    assert hit and "5804" in hit["text"]
    time.sleep(0.2)
    miss = get_cached_result(
        "How many Zara products?",
        platform=platform,
        ontology_domains=["ecom"],
        graph_ids=["ecom_products"],
        metric_family="catalog",
        intent="product_count",
    )
    assert miss is None


def test_metric_family_mismatch_does_not_serve():
    platform = SimpleNamespace(settings=SimpleNamespace(query_cache_ttl_seconds=60))
    put_cached_result(
        "Primark prices",
        text="price findings",
        platform=platform,
        ontology_domains=["ecom"],
        graph_ids=["ecom_products"],
        metric_family="price",
        intent="product_count",
    )
    assert (
        get_cached_result(
            "Primark prices",
            platform=platform,
            ontology_domains=["ecom"],
            graph_ids=["ecom_products"],
            metric_family="availability",
            intent="product_count",
        )
        is None
    )


def test_humanize_findings_labels():
    from knowledge_graph.retrieval.cypher_safe import humanize_findings_labels

    assert humanize_findings_labels("brand:label:zara") == "Zara"
    assert humanize_findings_labels("womens-shoes") == "Womens Shoes"
    assert humanize_findings_labels(
        "x:y:ignored",
        row={"id": "x:y:ignored", "name": "Pretty Name"},
        key="id",
    ) == "Pretty Name"
    assert humanize_findings_labels(42) == "42"
