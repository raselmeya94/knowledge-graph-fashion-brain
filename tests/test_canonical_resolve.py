"""Unit tests for resolve_canonical_entity (mocked Neo4j session)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from knowledge_graph.linking.canonical_resolve import (
    CanonicalHit,
    CanonicalResolveResult,
    resolve_canonical_entity,
)


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


def _platform_with_session(run_fn):
    session = MagicMock()
    session.run.side_effect = run_fn
    session.__enter__ = lambda self: session
    session.__exit__ = lambda *a: False
    graph = SimpleNamespace(_session=lambda: session)
    return SimpleNamespace(graph=graph), session


def test_resolve_skips_short_terms():
    platform, _ = _platform_with_session(lambda *_a, **_k: _FakeResult([]))
    result = resolve_canonical_entity(platform, "a")
    assert result.status == "skipped"
    assert result.hits == []


def test_resolve_no_session_returns_no_match():
    platform = SimpleNamespace(graph=None)
    result = resolve_canonical_entity(platform, "Zara")
    assert result.status == "no_match"


def test_resolve_fulltext_and_resolves_to():
    calls: list[str] = []

    def run(cypher, params=None):
        calls.append(cypher)
        if "db.index.fulltext.queryNodes" in cypher:
            if params and params.get("label") == "SourceEntity":
                return _FakeResult(
                    [
                        {
                            "id": "src-1",
                            "name": "Zara Home",
                            "nn": "zara home",
                            "score": 4.2,
                            "graph": "ecom_products",
                        }
                    ]
                )
            return _FakeResult([])
        if "RESOLVES_TO" in cypher:
            return _FakeResult(
                [{"sid": "src-1", "cid": "can-zara", "cname": "Zara"}]
            )
        # contains fallback
        return _FakeResult([])

    platform, _ = _platform_with_session(run)
    result = resolve_canonical_entity(
        platform,
        "Zara",
        graph_id="ecom_products",
        labels=["SourceEntity"],
        limit=5,
    )
    assert result.status == "matched"
    assert result.hits
    assert result.hits[0].id == "src-1"
    assert result.hits[0].canonical_id == "can-zara"
    assert result.hits[0].canonical_name == "Zara"
    assert result.hits[0].source == "resolves_to"
    payload = result.to_dict()
    assert payload["canonical_ids"] == ["can-zara"]


def test_resolve_contains_fallback_when_fulltext_empty():
    def run(cypher, params=None):
        if "db.index.fulltext.queryNodes" in cypher:
            raise RuntimeError("index missing")
        if "CONTAINS" in cypher:
            return _FakeResult(
                [
                    {
                        "id": "brand-1",
                        "name": "Primark",
                        "nn": "primark",
                        "graph": "ecom_products",
                    }
                ]
            )
        return _FakeResult([])

    platform, _ = _platform_with_session(run)
    result = resolve_canonical_entity(
        platform, "Primark", labels=["Brand"], graph_id="ecom_products"
    )
    assert result.status == "matched"
    assert result.hits[0].id == "brand-1"
    assert result.hits[0].source in {"normalized", "alias"}
    assert result.to_dict()["names"] == ["Primark"]


def test_kg_tool_resolve_wrapper(monkeypatch):
    from fashion_brain.execution import kg_tools

    monkeypatch.setattr(
        "knowledge_graph.linking.canonical_resolve.resolve_canonical_entity",
        lambda *_a, **_k: CanonicalResolveResult(
            term="cotton",
            status="matched",
            hits=[
                CanonicalHit(
                    label="Material",
                    id="mat-1",
                    name="Cotton",
                    normalized_name="cotton",
                    score=3.0,
                    source="alias",
                )
            ],
        ),
    )
    out = kg_tools.dispatch_kg_tool(
        SimpleNamespace(),
        "resolve_canonical_entity",
        {"term": "cotton", "graph_id": "ecom_products"},
    )
    assert out["status"] == "matched"
    assert out["ids"] == ["mat-1"]
