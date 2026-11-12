from knowledge_graph.adapters.ecom_events import build_event_graph
from knowledge_graph.access import Principal
from knowledge_graph.retrieval import ecom


def test_build_event_graph_same_currency_price_drop():
    sketch = build_event_graph(
        {
            "id": 22644,
            "brand": "c-and-a",
            "master_id": "c-and-a:2256738",
            "change_type": "price",
            "fields_changed": ["price"],
            "detected_at": "2026-10-01T12:00:00+00:00",
            "scraped_at": "2026-10-01T12:00:00+00:00",
            "before_json": {"price": 25.99, "currency": "EUR"},
            "after_json": {"price": 19.99, "currency": "EUR"},
            "scrape_job_id": "job-1",
        }
    )
    assert sketch is not None
    assert sketch["product_id"] == "ecom:c-and-a:c-and-a:2256738"
    labels = {n["id"]: n["labels"] for n in sketch["nodes"]}
    assert any("PriceChangeEvent" in labs for labs in labels.values())
    assert any("PriceSnapshot" in labs for labs in labels.values())
    change = next(n for n in sketch["nodes"] if "PriceChangeEvent" in n["labels"])
    assert change["properties"]["direction"] == "down"
    assert change["properties"]["currency_changed"] is False
    assert change["properties"]["delta_pct"] < 0
    assert sketch["product_patch"]["last_price_direction"] == "down"
    rels = {e["rel_type"] for e in sketch["edges"]}
    assert "HAS_PRICE_CHANGE" in rels
    assert "HAS_PRICE_SNAPSHOT" in rels


def test_build_event_graph_currency_switch():
    sketch = build_event_graph(
        {
            "id": 23768,
            "brand": "primark",
            "master_id": "primark:991180097720",
            "change_type": "price",
            "fields_changed": ["currency", "price"],
            "detected_at": "2026-11-08T10:00:00+06:00",
            "before_json": {"price": 14.0, "currency": "USD"},
            "after_json": {"price": 8.0, "currency": "GBP"},
        }
    )
    change = next(n for n in sketch["nodes"] if "PriceChangeEvent" in n["labels"])
    assert change["properties"]["direction"] == "currency_change"
    assert change["properties"]["currency_changed"] is True
    assert change["properties"]["delta_pct"] is None


def test_build_event_graph_availability():
    sketch = build_event_graph(
        {
            "id": 23774,
            "brand": "primark",
            "master_id": "primark:991195273720",
            "change_type": "availability",
            "fields_changed": ["availability", "in_stock_count", "out_stock_count"],
            "detected_at": "2026-11-08T10:47:08+06:00",
            "before_json": {"availability": "InStock", "in_stock_count": 6, "out_stock_count": 0},
            "after_json": {"availability": "Mixed", "in_stock_count": 5, "out_stock_count": 1},
        }
    )
    assert any("AvailabilitySnapshot" in n["labels"] for n in sketch["nodes"])
    rels = {e["rel_type"] for e in sketch["edges"]}
    assert "HAS_AVAILABILITY_SNAPSHOT" in rels
    assert "SNAPSHOT_STATUS" in rels


def test_build_event_graph_created_empty_skipped():
    assert (
        build_event_graph(
            {
                "id": 1,
                "brand": "primark",
                "master_id": "primark:1",
                "change_type": "created",
                "fields_changed": [],
                "before_json": {},
                "after_json": {},
            }
        )
        is None
    )


def test_price_trend_reports_currency_switches_without_faking_a_percent_trend(monkeypatch):
    monkeypatch.setattr(
        ecom,
        "cypher_price_trend_summary",
        lambda *_args, **_kwargs: {"change_events": 0},
    )
    monkeypatch.setattr(
        ecom,
        "cypher_price_currency_switch_summary",
        lambda *_args, **_kwargs: {
            "events": 165,
            "products": 161,
            "pairs": [
                {"pair": "USD→GBP", "events": 139, "products": 139},
                {"pair": "GBP→USD", "events": 26, "products": 26},
            ],
        },
    )
    lines, evidence, caveats = [], [], []

    handled = ecom.append_neo4j_price_history_answer(
        platform=None,
        principal=Principal(subject="analyst", clearance="internal"),
        graph_id="ecom_products",
        filters={"brand": "primark"},
        question="What are the average price change trends for Primark?",
        limit=10,
        lines=lines,
        evidence=evidence,
        caveats=caveats,
    )

    assert handled
    assert lines[0] == (
        "No comparable percentage price trend is available for Primark: the 165 recorded events "
        "are currency-label changes across 161 products, not same-currency price moves."
    )
    assert "| Currency change | Events | Products |" in lines
    assert "| USD→GBP | 139 | 139 |" in lines
    assert "| GBP→USD | 26 | 26 |" in lines
    assert not any("average" in line.casefold() and "%" in line for line in lines)
    assert evidence[0].record_id == "price_currency_switch_summary"
    assert "165" in evidence[0].excerpt
    assert caveats


def test_category_price_trend_explains_currency_switch_only_coverage(monkeypatch):
    monkeypatch.setattr(ecom, "cypher_price_trend_by_category", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        ecom,
        "cypher_price_currency_switch_summary",
        lambda *_args, **_kwargs: {
            "events": 8,
            "products": 7,
            "pairs": [{"pair": "EUR→GBP", "events": 8, "products": 7}],
        },
    )
    lines, evidence, caveats = [], [], []

    handled = ecom.append_neo4j_price_history_answer(
        platform=None,
        principal=Principal(subject="analyst", clearance="internal"),
        graph_id="ecom_products",
        filters={},
        question="Which categories are seeing decreasing price trends?",
        limit=10,
        lines=lines,
        evidence=evidence,
        caveats=caveats,
    )

    assert handled
    assert any("no category breakdown available" in line for line in lines)
    assert "| EUR→GBP | 8 | 7 |" in lines
    assert evidence[0].record_id == "price_currency_switch_summary"


def test_price_mover_request_explains_currency_switch_only_results(monkeypatch):
    monkeypatch.setattr(ecom, "cypher_price_movers", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        ecom,
        "cypher_price_currency_switch_summary",
        lambda *_args, **_kwargs: {
            "events": 2,
            "products": 2,
            "pairs": [{"pair": "USD→GBP", "events": 2, "products": 2}],
        },
    )
    lines, evidence, caveats = [], [], []

    handled = ecom.append_neo4j_price_history_answer(
        platform=None,
        principal=Principal(subject="analyst", clearance="internal"),
        graph_id="ecom_products",
        filters={"brand": "primark"},
        question="Which products had the largest price decreases for Primark?",
        limit=10,
        lines=lines,
        evidence=evidence,
        caveats=caveats,
    )

    assert handled
    assert any("not same-currency price moves" in line for line in lines)
    assert "| USD→GBP | 2 | 2 |" in lines


def test_brand_trend_comparison_keeps_rows_together_before_switch_details(monkeypatch):
    monkeypatch.setattr(
        "fashion_brain.ask.split_multi_party_filter",
        lambda _value: ["zara", "primark"],
    )
    monkeypatch.setattr(
        ecom,
        "cypher_price_trend_summary",
        lambda *_args, **_kwargs: {"change_events": 0},
    )

    def switches(_platform, _graph, filters):
        if filters.get("brand") == "primark":
            return {
                "events": 3,
                "products": 2,
                "pairs": [{"pair": "USD→GBP", "events": 3, "products": 2}],
            }
        return {"events": 0, "products": 0, "pairs": []}

    monkeypatch.setattr(ecom, "cypher_price_currency_switch_summary", switches)
    monkeypatch.setattr(ecom, "cypher_price_movers", lambda *_args, **_kwargs: [])
    lines, evidence, caveats = [], [], []

    handled = ecom.append_neo4j_price_history_answer(
        platform=None,
        principal=Principal(subject="analyst", clearance="internal"),
        graph_id="ecom_products",
        filters={"brand": "zara|primark"},
        question="Compare average price change trends for Zara and Primark",
        limit=10,
        lines=lines,
        evidence=evidence,
        caveats=caveats,
    )

    assert handled
    assert lines[:5] == [
        "Comparing historical price moves by brand:",
        "| Brand | Avg change | Events | Products | Up | Down | Unchanged |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| Zara | — | 0 | 0 | 0 | 0 | 0 |",
        "| Primark | — | 0 | 0 | 0 | 0 | 0 |",
    ]
    assert "| USD→GBP | 3 | 2 |" in lines
