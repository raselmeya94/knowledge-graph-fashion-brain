"""Live Neo4j schema snapshot (labels, rels, properties, graph/Domain values).

Built from Neo4j introspection procedures — not hardcoded ontology lists.
Cached in-process with TTL; warmed on KG API startup and invalidated on schema change.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

LOGGER = logging.getLogger("knowledge_graph.live_schema")

_LOCK = threading.Lock()
_CACHE: LiveSchemaSnapshot | None = None
_CACHE_TS: float = 0.0
_CACHE_TTL_SEC = 600.0


@dataclass
class LiveSchemaSnapshot:
    """Compact live Neo4j schema for Text2Cypher prompt pruning."""

    labels: list[str] = field(default_factory=list)
    relationship_types: list[str] = field(default_factory=list)
    node_properties: dict[str, list[str]] = field(default_factory=dict)
    rel_properties: dict[str, list[str]] = field(default_factory=dict)
    graph_values: list[str] = field(default_factory=list)
    domain_names: list[str] = field(default_factory=list)
    built_at: float = 0.0
    source: str = "neo4j"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def labels_for_graphs(self, graph_ids: list[str] | None = None) -> list[str]:
        """Return all known labels (graph scoping is applied at query time via p.graph)."""

        return list(self.labels)

    def compact_card(
        self,
        *,
        keep_labels: set[str] | None = None,
        keep_rels: set[str] | None = None,
        max_props_per_label: int = 24,
    ) -> str:
        """Render a pruned schema card for LLM Cypher prompts."""

        labels = [lab for lab in self.labels if not keep_labels or lab in keep_labels]
        rels = [rel for rel in self.relationship_types if not keep_rels or rel in keep_rels]
        lines: list[str] = [
            "Live Neo4j schema (introspected; use only these labels/rels/properties):",
            f"Graphs present: {', '.join(self.graph_values) or '(none)'}",
        ]
        if self.domain_names:
            lines.append(f"Domain nodes: {', '.join(self.domain_names[:40])}")
        lines.append("Node labels: " + ", ".join(labels[:80]))
        lines.append("Relationship types: " + ", ".join(rels[:80]))
        lines.append("Node properties (selected):")
        for lab in labels[:40]:
            props = list(self.node_properties.get(lab) or [])[:max_props_per_label]
            if props:
                lines.append(f"- {lab}: {', '.join(props)}")
        if any(self.rel_properties.get(r) for r in rels[:20]):
            lines.append("Relationship properties (selected):")
            for rel in rels[:20]:
                props = list(self.rel_properties.get(rel) or [])[:12]
                if props:
                    lines.append(f"- {rel}: {', '.join(props)}")
        return "\n".join(lines)


def invalidate_live_schema_cache() -> None:
    global _CACHE, _CACHE_TS
    with _LOCK:
        _CACHE = None
        _CACHE_TS = 0.0


def get_live_schema(platform, *, force: bool = False) -> LiveSchemaSnapshot | None:
    """Return cached live schema, refreshing when stale."""

    global _CACHE, _CACHE_TS
    now = time.monotonic()
    with _LOCK:
        if not force and _CACHE is not None and (now - _CACHE_TS) < _CACHE_TTL_SEC:
            return _CACHE
    snap = build_live_schema(platform)
    if snap is None:
        return _CACHE
    with _LOCK:
        _CACHE = snap
        _CACHE_TS = time.monotonic()
    return snap


def warm_live_schema(platform) -> dict[str, Any]:
    """Build and cache live schema (KG startup / schema hooks)."""

    snap = get_live_schema(platform, force=True)
    if snap is None:
        return {"status": "unavailable"}
    return {
        "status": "ok",
        "labels": len(snap.labels),
        "relationship_types": len(snap.relationship_types),
        "graphs": list(snap.graph_values),
        "domains": len(snap.domain_names),
    }


def build_live_schema(platform) -> LiveSchemaSnapshot | None:
    """Introspect Neo4j for labels, rels, properties, graph/Domain values."""

    session_factory = getattr(getattr(platform, "graph", None), "_session", None)
    if not callable(session_factory):
        LOGGER.warning("live schema: Neo4j session unavailable")
        return None

    labels: list[str] = []
    rel_types: list[str] = []
    node_props: dict[str, list[str]] = {}
    rel_props: dict[str, list[str]] = {}
    graphs: list[str] = []
    domains: list[str] = []

    try:
        with session_factory() as session:
            try:
                labels = sorted(
                    {
                        str(r["label"])
                        for r in session.run("CALL db.labels() YIELD label RETURN label")
                        if r.get("label")
                    }
                )
            except Exception as exc:
                LOGGER.warning("db.labels failed: %s", exc)

            try:
                rel_types = sorted(
                    {
                        str(r["relationshipType"])
                        for r in session.run(
                            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
                        )
                        if r.get("relationshipType")
                    }
                )
            except Exception as exc:
                LOGGER.warning("db.relationshipTypes failed: %s", exc)

            try:
                for row in session.run(
                    "CALL db.schema.nodeTypeProperties() "
                    "YIELD nodeLabels, propertyName "
                    "RETURN nodeLabels, propertyName"
                ):
                    labs = row.get("nodeLabels") or []
                    prop = str(row.get("propertyName") or "").strip()
                    if not prop:
                        continue
                    for lab in labs:
                        key = str(lab)
                        node_props.setdefault(key, [])
                        if prop not in node_props[key]:
                            node_props[key].append(prop)
            except Exception as exc:
                LOGGER.warning("db.schema.nodeTypeProperties failed: %s", exc)

            try:
                for row in session.run(
                    "CALL db.schema.relTypeProperties() "
                    "YIELD relType, propertyName "
                    "RETURN relType, propertyName"
                ):
                    rel = str(row.get("relType") or "").strip()
                    prop = str(row.get("propertyName") or "").strip()
                    if not rel or not prop:
                        continue
                    rel_props.setdefault(rel, [])
                    if prop not in rel_props[rel]:
                        rel_props[rel].append(prop)
            except Exception as exc:
                LOGGER.warning("db.schema.relTypeProperties failed: %s", exc)

            try:
                graphs = sorted(
                    {
                        str(r["g"])
                        for r in session.run(
                            "MATCH (n) WHERE n.graph IS NOT NULL "
                            "RETURN DISTINCT n.graph AS g LIMIT 50"
                        )
                        if r.get("g") not in (None, "")
                    }
                )
            except Exception as exc:
                LOGGER.warning("distinct graph values failed: %s", exc)

            if "Domain" in labels:
                try:
                    domains = sorted(
                        {
                            str(r["name"])
                            for r in session.run(
                                "MATCH (d:Domain) "
                                "RETURN DISTINCT coalesce(d.name, d.id) AS name LIMIT 80"
                            )
                            if r.get("name")
                        }
                    )
                except Exception as exc:
                    LOGGER.warning("Domain names failed: %s", exc)
    except Exception as exc:
        LOGGER.warning("live schema build failed: %s", exc)
        return None

    return LiveSchemaSnapshot(
        labels=labels,
        relationship_types=rel_types,
        node_properties={k: sorted(v) for k, v in node_props.items()},
        rel_properties={k: sorted(v) for k, v in rel_props.items()},
        graph_values=graphs,
        domain_names=domains,
        built_at=time.time(),
        source="neo4j",
    )
