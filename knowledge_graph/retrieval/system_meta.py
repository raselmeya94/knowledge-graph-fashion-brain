"""System-level knowledge-graph / source metadata for Fashion Brain.

Persists a live snapshot under ``generated/kg_system_meta.json`` and answers
registry/system questions from that catalog — not from product-row filters.

Routing should prefer the registered ``kg_system`` graph via schema/value
grounding (ontology aliases + live graph_id/capability/attribute values),
not via hard-coded English question phrases in the ask planner.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from knowledge_graph.config import DATA_ROOT
from knowledge_graph.models import EvidenceItem
from knowledge_graph.ontology import normalize_name
from knowledge_graph.retrieval.metadata import discover_values

SYSTEM_GRAPH_ID = "kg_system"
_META_PATH = Path(DATA_ROOT) / "generated" / "kg_system_meta.json"
_VALUE_CAP = 25
_ATTR_PROBE = (
    "brand",
    "buyer",
    "category",
    "material",
    "color",
    "season",
    "gender",
    "availability",
    "family",
    "department",
)

# Indexed attributes for grounding / discover_values on kg_system.
_SYSTEM_INDEX_ATTRS = (
    "graph_id",
    "graph_name",
    "kind",
    "capability",
    "entity",
    "attribute",
    "source_id",
    "topic",
)


def system_meta_path() -> Path:
    return _META_PATH


def build_system_meta(platform, *, persist: bool = True) -> dict[str, Any]:
    """Collect per-graph basics: description, counts, attributes, sample unique values."""

    # Prefer registry rows directly so we do not recurse through metadata_overview → system build.
    rows = _registry_graph_rows(platform)
    graphs_out: list[dict[str, Any]] = []
    index: dict[str, list[str]] = {key: [] for key in _SYSTEM_INDEX_ATTRS}

    def _index(attr: str, *values: Any) -> None:
        bucket = index.setdefault(attr, [])
        seen = {normalize_name(v) for v in bucket}
        for value in values:
            text = str(value or "").strip()
            if not text:
                continue
            key = normalize_name(text)
            if not key or key in seen:
                continue
            seen.add(key)
            bucket.append(text)

    for row in rows:
        graph_id = str(row.get("graph_id") or row.get("id") or "")
        if not graph_id or graph_id == SYSTEM_GRAPH_ID:
            continue
        kind = str(row.get("kind") or "")
        name = _graph_name(platform, graph_id) or graph_id
        product_count = _count_label(platform, graph_id, "Product") if kind == "product" else None
        node_totals = _node_totals(platform, graph_id)
        attrs = _attributes_from_row(row)
        unique: dict[str, Any] = {}
        if kind == "product":
            for attr in attrs:
                if attr not in _ATTR_PROBE and len(unique) >= 8:
                    continue
                if attr not in _ATTR_PROBE:
                    continue
                try:
                    discovered = discover_values(
                        platform, graph_id=graph_id, attribute=attr, limit=_VALUE_CAP, offset=0
                    )
                except Exception:
                    continue
                values = list(discovered.get("values") or [])
                if not values:
                    continue
                unique[attr] = {
                    "sample": values,
                    "total": int(discovered.get("total") or len(values)),
                    "capped": int(discovered.get("total") or 0) > len(values),
                }

        description = str(row.get("description") or "")
        graphs_out.append(
            {
                "graph_id": graph_id,
                "name": name,
                "kind": kind,
                "description": description,
                "status": str(row.get("status") or "active"),
                "source_ids": list(row.get("source_ids") or []),
                "capabilities": list(row.get("capabilities") or []),
                "entities": list(row.get("entities") or []),
                "relationships": list(row.get("relationships") or []),
                "attributes": attrs,
                "product_count": product_count,
                "node_totals": node_totals,
                "unique_values": unique,
            }
        )
        _index("graph_id", graph_id)
        _index("graph_name", name)
        _index("kind", kind)
        _index("source_id", *(row.get("source_ids") or []))
        _index("capability", *(row.get("capabilities") or []))
        _index("entity", *(row.get("entities") or []))
        _index("attribute", *attrs)
        _index("topic", *description.replace(",", " ").split())

    # Ontology aliases become live topic values so grounding stays schema-driven.
    for alias in _ontology_aliases(platform):
        _index("topic", alias)

    shared = _shared_meta(platform)
    _index("graph_id", "shared", SYSTEM_GRAPH_ID)
    _index("graph_name", "Shared Interconnect", "System Metadata Knowledge Graph")
    _index("kind", "interconnect", "system")
    _index("capability", *(shared.get("capabilities") or []))
    _index("topic", "shared", "interconnect", "registry", "system")

    payload = {
        "version": 1,
        "graph_id": SYSTEM_GRAPH_ID,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_count": len([g for g in graphs_out if g.get("kind") != "interconnect"]),
        "graph_count": len(graphs_out),
        "graphs": graphs_out,
        "shared": shared,
        "value_index": {key: sorted(vals, key=str.casefold) for key, vals in index.items() if vals},
        "attributes": list(_SYSTEM_INDEX_ATTRS),
    }
    if persist:
        try:
            _META_PATH.parent.mkdir(parents=True, exist_ok=True)
            _META_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        except Exception:
            pass
    return payload


def load_or_build_system_meta(platform, *, refresh: bool = True) -> dict[str, Any]:
    if refresh:
        return build_system_meta(platform, persist=True)
    if _META_PATH.exists():
        try:
            return json.loads(_META_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return build_system_meta(platform, persist=True)


def system_value_index(platform=None, *, refresh: bool = False) -> dict[str, list[str]]:
    """Attribute → distinct values for kg_system grounding / discover_values."""

    if platform is not None and (refresh or not _META_PATH.exists()):
        meta = build_system_meta(platform, persist=True)
    elif _META_PATH.exists():
        try:
            meta = json.loads(_META_PATH.read_text(encoding="utf-8"))
        except Exception:
            meta = build_system_meta(platform, persist=True) if platform is not None else {}
    else:
        meta = build_system_meta(platform, persist=True) if platform is not None else {}
    return {str(k): list(v) for k, v in (meta.get("value_index") or {}).items()}


def lookup_system_value(attribute: str, token: str, platform=None) -> str | None:
    """Match a free token against the system meta value index (same idea as product brands)."""

    want = normalize_name(token)
    if not want:
        return None
    index = system_value_index(platform, refresh=False)
    values = index.get(attribute) or []
    for value in values:
        name = normalize_name(value)
        if name == want or (len(want) >= 4 and (want in name or name in want)):
            return value
    return None


def append_system_meta_answer(
    platform,
    lines: list[str],
    evidence: list[EvidenceItem],
    caveats: list[str],
    *,
    question: str = "",
) -> None:
    """Write grounded system/registry findings for one ask turn."""

    meta = load_or_build_system_meta(platform, refresh=True)
    graphs = list(meta.get("graphs") or [])
    shared = dict(meta.get("shared") or {})
    active = [g for g in graphs if str(g.get("status") or "active") == "active"]
    product_graphs = [g for g in active if g.get("kind") == "product"]
    document_graphs = [g for g in active if g.get("kind") in {"documents", "custom"}]

    lines.append(
        f"There are {len(active)} registered knowledge graphs / sources "
        f"({len(product_graphs)} product, {len(document_graphs)} documents/custom), "
        f"plus the shared interconnect and the system metadata catalog (`{SYSTEM_GRAPH_ID}`)."
    )
    if product_graphs:
        lines.append("Product source graphs:")
        for graph in product_graphs:
            count = graph.get("product_count")
            count_bit = f"{count} products" if count is not None else "product count unavailable"
            lines.append(
                f"- {graph.get('name')} (`{graph.get('graph_id')}`): {count_bit}. "
                f"{graph.get('description') or 'No description in the registry.'}"
            )
            caps = ", ".join(graph.get("capabilities") or []) or "none listed"
            ents = ", ".join(graph.get("entities") or []) or "none listed"
            lines.append(f"  Capabilities: {caps}.")
            lines.append(f"  Entities: {ents}.")
            attrs = list(graph.get("attributes") or [])
            if attrs:
                shown = ", ".join(attrs[:24])
                lines.append(
                    f"  Product attributes ({len(attrs)}): {shown}"
                    + ("…" if len(attrs) > 24 else "")
                    + "."
                )
            unique = graph.get("unique_values") or {}
            for attr, payload in list(unique.items())[:6]:
                sample = ", ".join(str(v) for v in (payload.get("sample") or [])[:12])
                total = payload.get("total") or 0
                more = f" ({total} distinct)" if total else ""
                lines.append(
                    f"  Unique {attr}{more}: {sample}"
                    + ("…" if payload.get("capped") else "")
                    + "."
                )

    if document_graphs:
        lines.append("Document / news graphs:")
        for graph in document_graphs:
            lines.append(
                f"- {graph.get('name')} (`{graph.get('graph_id')}`): "
                f"{graph.get('description') or 'No description in the registry.'}"
            )

    lines.append("Shared / Global interconnect:")
    lines.append(
        shared.get("description")
        or "The shared layer links equivalent entities across source graphs (canonical materials and cross-source mappings). It is not a product catalog."
    )
    materials = shared.get("canonical_materials") or []
    if materials:
        lines.append(
            f"Canonical materials currently on shared ({len(materials)}): "
            + ", ".join(str(m) for m in materials[:20])
            + ("…" if len(materials) > 20 else "")
            + "."
        )
    else:
        lines.append("No canonical Material nodes were found on the shared interconnect yet.")

    text_blob = question.casefold()
    if "attribute" in text_blob or "unique" in text_blob:
        caveats.append(
            "Unique values are sampled (capped) so prompts stay small; totals are included when available."
        )

    evidence.append(
        EvidenceItem(
            source_id=SYSTEM_GRAPH_ID,
            record_id="system_meta",
            title="Knowledge graph registry / system metadata",
            excerpt=f"{len(active)} active graphs; shared interconnect documented.",
            source_path=str(_META_PATH.as_posix()) if _META_PATH.exists() else SYSTEM_GRAPH_ID,
            verification_status="observed",
            graph=SYSTEM_GRAPH_ID,
            synthetic=False,
        )
    )
    for graph in active[:6]:
        evidence.append(
            EvidenceItem(
                source_id=str(graph.get("graph_id")),
                record_id=str(graph.get("graph_id")),
                title=str(graph.get("name") or graph.get("graph_id")),
                excerpt=str(graph.get("description") or "")[:240],
                source_path=str(graph.get("graph_id")),
                verification_status="observed",
                graph=str(graph.get("graph_id")),
                synthetic=False,
            )
        )


def _registry_graph_rows(platform) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        from knowledge_graph.kg_registry.store import enrich_graph_record

        for graph in platform.kg_registry.list_graphs():
            try:
                ontology = platform.kg_registry.load_ontology_object(graph.ontology_id)
            except Exception:
                ontology = None
            enriched = enrich_graph_record(graph, ontology)
            schema: dict[str, Any] = {}
            if ontology is not None and "Product" in set(ontology.entity_types):
                from knowledge_graph.retrieval.metadata import _product_attributes_for_graph

                schema["Product"] = {
                    "attributes": _product_attributes_for_graph(platform, enriched.id, enriched)
                }
            rows.append(
                {
                    "graph_id": enriched.id,
                    "id": enriched.id,
                    "kind": enriched.kind,
                    "source_ids": list(enriched.source_ids or []),
                    "description": enriched.description,
                    "entities": list(enriched.entities or []),
                    "relationships": list(enriched.relationships or []),
                    "capabilities": list(enriched.capabilities or []),
                    "status": enriched.status,
                    "schema": schema,
                }
            )
    except Exception:
        return rows
    return rows


def _ontology_aliases(platform) -> list[str]:
    aliases: list[str] = []
    try:
        ontology = platform.kg_registry.load_ontology_object("system")
    except Exception:
        return aliases
    for entity in ontology.canonical_entities:
        aliases.append(entity.name)
        aliases.extend(entity.aliases or [])
    return aliases


def _attributes_from_row(row: dict[str, Any]) -> list[str]:
    schema = row.get("schema") or {}
    product = schema.get("Product") if isinstance(schema, dict) else None
    attrs = list((product or {}).get("attributes") or {})
    return sorted({str(a) for a in attrs if a})


def _count_label(platform, graph_id: str, label: str) -> int:
    try:
        return len(platform.graph.find_nodes(graph=graph_id, label=label, include_deleted=False))
    except Exception:
        return 0


def _node_totals(platform, graph_id: str) -> dict[str, int]:
    totals: dict[str, int] = {}
    try:
        for label in ("Product", "Brand", "Material", "GarmentCategory", "Document", "Color", "Season"):
            n = _count_label(platform, graph_id, label)
            if n:
                totals[label] = n
    except Exception:
        pass
    return totals


def _graph_name(platform, graph_id: str) -> str | None:
    try:
        for item in platform.kg_registry.list_graphs():
            if item.id == graph_id:
                return str(item.name or graph_id)
    except Exception:
        pass
    return None


def _shared_meta(platform) -> dict[str, Any]:
    materials: list[str] = []
    try:
        nodes = platform.graph.find_nodes(graph="shared", label="Material", include_deleted=False)
        for node in nodes:
            name = str(node.properties.get("name") or "").strip()
            if name:
                materials.append(name)
    except Exception:
        pass
    materials = sorted(set(materials), key=str.casefold)
    return {
        "graph_id": "shared",
        "kind": "interconnect",
        "description": (
            "Global / shared interconnect joins equivalent entities across source-specific knowledge graphs "
            "(for example canonical materials via RESOLVES_TO). Source product facts stay in each source KG."
        ),
        "canonical_materials": materials,
        "material_count": len(materials),
        "capabilities": ["canonical_materials", "products_sharing_material", "cross_source_mapping"],
    }
