"""Generic Postgres → Neo4j sync driven by source mapping JSON.

Any authorized_sql product KG can build when mapping + connection_id (+ ontology) exist.
No hard-coded source ids beyond optional DSN fallbacks.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from knowledge_graph.errors import ValidationFailed
from knowledge_graph.ids import edge_id, register_source_identity, slug
from knowledge_graph.logging_config import get_logger
from knowledge_graph.models import GraphEdge, GraphNode
from knowledge_graph.ontology import combine_ontologies, normalize_name
from knowledge_graph.sources.mapping_store import get_mapping, mapping_exists

LOGGER = get_logger("knowledge_graph.mapping_sql_sync")

_COMPOSITION = re.compile(
    r"(?P<pct>\d+(?:\.\d+)?)\s*%\s*(?P<name>[A-Za-z][A-Za-z0-9 \-/]+)|(?P<only>[A-Za-z][A-Za-z0-9 \-/]{1,40})"
)
_SPLIT_NOTES = re.compile(r"split\s+on\s+comma", re.I)


def sync_state_path(settings, source_id: str) -> Path:
    return settings.generated / f"{source_id}_sync_state.json"


def load_sync_state(settings, source_id: str) -> dict:
    path = sync_state_path(settings, source_id)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_sync_state(settings, source_id: str, state: dict) -> None:
    path = sync_state_path(settings, source_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def sync_mapped_sql_to_neo4j(
    platform,
    source_id: str,
    *,
    full: bool = False,
    limit: int | None = None,
    refresh_generated: bool = True,
) -> dict[str, Any]:
    """Read mapped Postgres table and upsert Product + relationship nodes into Neo4j."""

    if not mapping_exists(source_id):
        raise ValidationFailed(
            f"Mapping for '{source_id}' not found. PUT /api/sources/{source_id}/mapping first."
        )
    mapping = get_mapping(source_id)["mapping"]
    graph_id = str(mapping.get("graph_id") or source_id)
    table = str(mapping.get("source_table") or "").strip()
    if not table:
        raise ValidationFailed(f"Mapping for '{source_id}' is missing source_table.")

    ontology_id = str(mapping.get("ontology_id") or "").strip()
    _apply_ontology(platform, ontology_id, graph_id)
    _register_identity(source_id, mapping)

    dsn = _resolve_dsn(platform.settings, mapping)
    columns = _select_columns(mapping)
    watermark_cols = [c for c in (mapping.get("watermark_columns") or []) if c in columns]
    watermark_col = watermark_cols[0] if watermark_cols else None

    state = load_sync_state(platform.settings, source_id)
    since = None if full else state.get("last_watermark")
    products = 0
    node_writes = 0
    edge_writes = 0
    errors: list[str] = []
    latest_wm = since
    facets: dict[str, int] = {}

    save_sync_state(
        platform.settings,
        source_id,
        {
            "status": "running",
            "last_run_at": _utc_now(),
            "graph": graph_id,
            "source_table": table,
            "full": full,
            "last_products": 0,
        },
    )
    LOGGER.info(
        "mapping sql sync start source=%s table=%s full=%s limit=%s since=%s",
        source_id,
        table,
        full,
        limit,
        since,
    )

    for row in _iter_rows(dsn, table, columns, watermark_col=watermark_col, since=since, limit=limit):
        try:
            product, rels = _row_to_graph(row, mapping, source_id=source_id, graph_id=graph_id)
            tallies = _upsert(platform.graph, product, rels, graph_id=graph_id, source_id=source_id)
            products += 1
            node_writes += tallies["nodes"]
            edge_writes += tallies["edges"]
            facet = str(product.get("brand") or product.get("buyer") or "")
            if facet:
                facets[facet] = facets.get(facet, 0) + 1
            if watermark_col and row.get(watermark_col) is not None:
                text = _cell(row.get(watermark_col))
                if latest_wm is None or text > str(latest_wm):
                    latest_wm = text
            if products % 500 == 0:
                extras: dict[str, Any] = {}
                # Mid-sync kg_stats (same idea as ecom) so counts work before sync finishes.
                if products % 1000 == 0:
                    try:
                        extras["generated"] = _refresh_mapped_kg_stats(platform, graph_id, mapping)
                    except Exception as exc:
                        extras["generated_error"] = str(exc)
                        LOGGER.warning(
                            "mapping sql sync mid-run kg_stats failed source=%s: %s",
                            source_id,
                            exc,
                        )
                save_sync_state(
                    platform.settings,
                    source_id,
                    {
                        "status": "running",
                        "last_run_at": _utc_now(),
                        "graph": graph_id,
                        "source_table": table,
                        "last_products": products,
                        "last_watermark": latest_wm,
                        "brands": facets,
                        "node_writes": node_writes,
                        "edge_writes": edge_writes,
                        "errors": len(errors),
                        "full": full,
                        **extras,
                    },
                )
                LOGGER.info("mapping sql sync progress source=%s products=%s", source_id, products)
        except Exception as exc:
            key = _row_key(row, mapping)
            errors.append(f"{key}: {exc}")
            LOGGER.warning("mapping sql sync row failed %s: %s", key, exc, exc_info=True)
            if len(errors) >= 50:
                errors.append("… further errors truncated")
                break

    shared_counts = None
    shared_error = None
    if refresh_generated:
        try:
            shared_counts = platform.build_shared()
        except Exception as exc:
            shared_error = str(exc)
            errors.append(f"shared graph refresh failed: {exc}")
            LOGGER.exception("mapping sql sync shared graph refresh failed source=%s", source_id)

    finished = _utc_now()
    report = {
        "ok": not errors,
        "graph": graph_id,
        "source_id": source_id,
        "source_table": table,
        "products": products,
        "node_writes": node_writes,
        "edge_writes": edge_writes,
        "brands": facets,
        "errors": errors,
        "last_watermark": latest_wm,
        "finished_at": finished,
    }
    if shared_counts is not None:
        report["shared"] = shared_counts
    if shared_error is not None:
        report["shared_error"] = shared_error
    save_sync_state(
        platform.settings,
        source_id,
        {
            "status": "failed" if errors else "ok",
            "last_run_at": finished,
            "finished_at": finished,
            "graph": graph_id,
            "source_table": table,
            "last_products": products,
            "last_watermark": latest_wm,
            "brands": facets,
            "node_writes": node_writes,
            "edge_writes": edge_writes,
            "errors": len(errors),
            "full": full,
            "shared": shared_counts,
            "shared_error": shared_error,
        },
    )
    if refresh_generated:
        try:
            report["generated"] = _refresh_mapped_kg_stats(platform, graph_id, mapping)
        except Exception as exc:
            LOGGER.warning("mapping sql sync kg_stats refresh failed: %s", exc)
            report["stats_error"] = str(exc)
        try:
            from knowledge_graph.graph.snapshots import save_master_snapshot, save_source_snapshots

            save_source_snapshots(platform.graph, platform.settings.generated, [graph_id])
            if shared_error is None:
                save_master_snapshot(platform.graph, platform.settings.generated)
        except Exception as exc:
            LOGGER.warning("mapping sql sync snapshot refresh failed: %s", exc)
        try:
            from knowledge_graph.retrieval.system_meta import build_system_meta

            build_system_meta(platform, persist=True)
        except Exception as exc:
            LOGGER.warning("mapping sql sync system meta failed: %s", exc)
    LOGGER.info(
        "mapping sql sync done source=%s products=%s nodes=%s edges=%s errors=%s",
        source_id,
        products,
        node_writes,
        edge_writes,
        len(errors),
    )
    return report


def _stats_attributes_from_mapping(mapping: dict[str, Any]) -> list[str]:
    """Every categorical facet this mapping defines — properties + relationship columns.

    No fixed fashion field list: each KG's mapping is the source of truth.
    """

    from knowledge_graph.retrieval.ecom import is_analysis_facet_attr
    from knowledge_graph.retrieval.runtime_schema import _facet_filter_key

    keys: set[str] = set()
    for name in (mapping.get("product_properties") or {}):
        keys.add(str(name).strip())
    for rel in mapping.get("relationships") or []:
        if not isinstance(rel, dict):
            continue
        col = str(rel.get("from_column") or "").strip()
        if col:
            keys.add(col)
        end = str(rel.get("end_type") or "").strip()
        if end:
            keys.add(_facet_filter_key(end))
    return sorted(name for name in keys if name and is_analysis_facet_attr(name))


def _refresh_mapped_kg_stats(platform, graph_id: str, mapping: dict[str, Any]) -> dict[str, Any]:
    """Write ``generated/kg_stats/{graph_id}.json`` from Neo4j (mid-sync or finish)."""

    from knowledge_graph.retrieval.kg_stats import refresh_product_graph_stats

    stats = refresh_product_graph_stats(
        platform,
        graph_id,
        attributes=_stats_attributes_from_mapping(mapping),
    )
    totals = stats.get("totals") or {}
    return {
        "graph": graph_id,
        "stats_products": totals.get("products"),
        "stats_brands": totals.get("brands"),
        "stats_buyers": totals.get("buyers"),
        "stats_path": str(platform.settings.generated / "kg_stats" / f"{graph_id}.json"),
    }


def _resolve_dsn(settings, mapping: dict[str, Any]) -> str:
    connection_id = str(mapping.get("connection_id") or "").strip()
    if connection_id:
        from knowledge_graph.sources.db_connections import resolve_connection_dsn

        return resolve_connection_dsn(settings, connection_id)
    # Optional env fallbacks for known names — still dynamic via mapping first.
    for key in ("FASHIONGPT_PLMEX_DSN", "FASHIONGPT_SQL_DSN"):
        value = (os.environ.get(key) or "").strip()
        if value:
            return value
    raise ValidationFailed(
        "No database connection for this mapping. Set mapping.connection_id "
        "(POST /api/db/connections) or FASHIONGPT_PLMEX_DSN / FASHIONGPT_SQL_DSN."
    )


def _select_columns(mapping: dict[str, Any]) -> list[str]:
    cols: list[str] = []
    seen: set[str] = set()
    for spec in (mapping.get("product_properties") or {}).values():
        if not isinstance(spec, dict):
            continue
        col = str(spec.get("column") or "").strip()
        if col and col not in seen:
            seen.add(col)
            cols.append(col)
    for rel in mapping.get("relationships") or []:
        for key in ("from_column", "fallback_column", "start_from_column", "end_from_column"):
            col = str(rel.get(key) or "").strip()
            if col and col not in seen:
                seen.add(col)
                cols.append(col)
    for col in mapping.get("primary_key") or []:
        c = str(col).strip()
        if c and c not in seen:
            seen.add(c)
            cols.append(c)
    if not cols:
        raise ValidationFailed("Mapping has no product_properties columns to SELECT.")
    return cols


def _iter_rows(
    dsn: str,
    table: str,
    columns: list[str],
    *,
    watermark_col: str | None,
    since: str | None,
    limit: int | None,
) -> Iterator[dict[str, Any]]:
    try:
        import psycopg
        from psycopg import sql
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise ValidationFailed("psycopg is not installed. Run: pip install 'psycopg[binary]'") from exc

    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table):
        raise ValidationFailed(f"Unsafe source_table {table!r}")
    for col in columns:
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", col):
            raise ValidationFailed(f"Unsafe column {col!r}")

    select = sql.SQL("SELECT {fields} FROM {table}").format(
        fields=sql.SQL(", ").join(sql.Identifier(c) for c in columns),
        table=sql.Identifier(table),
    )
    params: dict[str, Any] = {}
    if watermark_col and since:
        select = sql.SQL("{base} WHERE {wm} > %(since)s").format(
            base=select,
            wm=sql.Identifier(watermark_col),
        )
        params["since"] = since
    if watermark_col:
        select = sql.SQL("{base} ORDER BY {wm} ASC NULLS LAST").format(
            base=select,
            wm=sql.Identifier(watermark_col),
        )
    if limit is not None:
        select = sql.SQL("{base} LIMIT %(limit)s").format(base=select)
        params["limit"] = int(limit)

    produced = 0
    with psycopg.connect(dsn) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(select, params)
            while True:
                batch = cur.fetchmany(200)
                if not batch:
                    break
                for row in batch:
                    yield dict(row)
                    produced += 1
                    if limit is not None and produced >= limit:
                        return


def _row_to_graph(
    row: dict[str, Any],
    mapping: dict[str, Any],
    *,
    source_id: str,
    graph_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    props_map = mapping.get("product_properties") or {}
    product: dict[str, Any] = {"source": source_id, "source_table": mapping.get("source_table")}
    for key, spec in props_map.items():
        if not isinstance(spec, dict):
            continue
        col = str(spec.get("column") or key)
        product[key] = row.get(col)

    # Flatten primary keys onto product for template formatting.
    for pk in mapping.get("primary_key") or []:
        product.setdefault(str(pk), row.get(str(pk)))

    template = str(mapping.get("product_id_template") or "{code}")
    try:
        product_id = template.format(**{k: _cell(v) for k, v in product.items()})
    except Exception as exc:
        raise ValueError(f"product_id_template failed: {exc}") from exc
    if not product_id or "{" in product_id:
        raise ValueError(f"Could not build product_id from template {template!r}")
    product["product_id"] = product_id
    product["name"] = _cell(product.get("name") or product.get("code") or product_id)

    neo_keys = mapping.get("neo4j_product_properties") or []
    node_props = {
        key: product.get(key)
        for key in neo_keys
        if product.get(key) not in (None, "", [])
    }
    node_props.setdefault("code", product.get("code"))
    node_props["source"] = source_id
    product["_node_props"] = node_props

    rels = _build_relationships(product, mapping.get("relationships") or [])
    return product, rels


def _build_relationships(product: dict[str, Any], specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pid = product["product_id"]
    rels: list[dict[str, Any]] = []

    def add(rel_type: str, end_type: str, name: str, *, field_ref: str, properties: dict | None = None) -> None:
        value = _present(name)
        if not value:
            return
        rels.append(
            {
                "rel_type": rel_type,
                "start_id": pid,
                "start_type": "Product",
                "end_type": end_type,
                "end_name": value,
                "end_id": f"{end_type.lower()}:{normalize_name(value).replace(' ', '-')}",
                "field_ref": field_ref,
                "properties": properties or {},
            }
        )

    for spec in specs:
        rel_type = str(spec.get("rel_type") or "").upper()
        end_type = str(spec.get("end_type") or "")
        if not rel_type or not end_type:
            continue

        if rel_type == "PARENT_CATEGORY":
            child = _present(product.get(str(spec.get("start_from_column") or "")))
            parent = _present(product.get(str(spec.get("end_from_column") or "")))
            start_type = str(spec.get("start_type") or end_type)
            if child and parent:
                rels.append(
                    {
                        "rel_type": "PARENT_CATEGORY",
                        "start_type": start_type,
                        "start_name": child,
                        "start_id": f"{start_type.lower()}:{normalize_name(child).replace(' ', '-')}",
                        "end_type": end_type,
                        "end_name": parent,
                        "end_id": f"{end_type.lower()}:{normalize_name(parent).replace(' ', '-')}",
                        "field_ref": f"{spec.get('start_from_column')}→{spec.get('end_from_column')}",
                        "properties": {},
                    }
                )
            continue

        if rel_type in {"USES_MATERIAL", "PRIMARY_MATERIAL"}:
            materials = _materials(product)
            if rel_type == "USES_MATERIAL":
                for material, percentage in materials:
                    props = {"percentage": percentage} if percentage is not None else {}
                    add(rel_type, end_type, material, field_ref="composition", properties=props)
            elif materials:
                primary = max(
                    materials,
                    key=lambda item: (item[1] is not None, item[1] if item[1] is not None else -1.0),
                )
                props = {"percentage": primary[1]} if primary[1] is not None else {}
                add(rel_type, end_type, primary[0], field_ref="composition", properties=props)
            continue

        col = str(spec.get("from_column") or "")
        raw = product.get(col)
        if raw in (None, "") and spec.get("fallback_column"):
            raw = product.get(str(spec.get("fallback_column")))
        notes = str(spec.get("notes") or "")
        split = bool(_SPLIT_NOTES.search(notes)) or col.startswith("ai_")

        if rel_type == "HAS_INVENTORY_RECORD":
            stock = _present(product.get("current_stock"))
            loc = _present(product.get("current_location"))
            if stock or loc:
                label = f"stock:{stock or '0'}" + (f"@{loc[:80]}" if loc else "")
                add(rel_type, end_type, label, field_ref=col or "current_stock")
            continue
        if rel_type == "HAS_SHIPMENT":
            date = _present(product.get("shipment_date"))
            qty = _present(product.get("shipment_quantity"))
            if date or qty:
                label = f"ship:{date or 'unknown'}" + (f"x{qty}" if qty else "")
                add(rel_type, end_type, label, field_ref=col or "shipment_date")
            continue

        values = _split_values(raw) if split else ([_present(raw)] if _present(raw) else [])
        for index, value in enumerate(values):
            props = {"position": index, "is_primary": index == 0} if rel_type in {"HAS_COLOR", "PRIMARY_COLOR"} else {}
            if rel_type == "PRIMARY_COLOR" and index > 0:
                continue
            if rel_type == "PRIMARY_COLOR":
                add(rel_type, end_type, value, field_ref=col, properties={"is_primary": True})
            else:
                add(rel_type, end_type, value, field_ref=col, properties=props)
    return rels


def _upsert(graph, product: dict[str, Any], relationships: list[dict[str, Any]], *, graph_id: str, source_id: str) -> dict[str, int]:
    pid = product["product_id"]
    nodes = 0
    edges = 0
    display = _present(product.get("name")) or pid
    graph.upsert_node(
        GraphNode(
            id=pid,
            labels=["Entity", "Record", "Product"],
            graph=graph_id,
            properties={
                **(product.get("_node_props") or {}),
                "id": pid,
                "name": display,
                "normalized_name": normalize_name(display),
                "classification": "internal",
                "prov_source_id": graph_id,
                "prov_record_id": _cell(product.get("code") or pid),
                "prov_source_type": source_id,
                "lifecycle": "active",
            },
        )
    )
    nodes += 1

    for rel in relationships:
        end_id = rel.get("end_id") or f"{rel['end_type'].lower()}:{slug(rel['end_name'])}"
        end_type = rel["end_type"]
        if rel["rel_type"] == "PARENT_CATEGORY":
            start_id = rel.get("start_id") or f"{rel['start_type'].lower()}:{slug(rel['start_name'])}"
            for node_id, node_type, node_name in (
                (start_id, rel["start_type"], rel["start_name"]),
                (end_id, end_type, rel["end_name"]),
            ):
                nn = normalize_name(node_name)
                graph.upsert_node(
                    GraphNode(
                        id=node_id,
                        labels=["Entity", "SourceEntity", node_type],
                        graph=graph_id,
                        properties={
                            "id": node_id,
                            "name": nn or str(node_name or "").strip(),
                            "normalized_name": nn,
                            "entity_type": node_type,
                            "lifecycle": "active",
                            "prov_source_id": graph_id,
                        },
                    )
                )
                nodes += 1
            graph.upsert_edge(
                GraphEdge(
                    id=edge_id(rel["rel_type"], start_id, end_id, pid, rel.get("field_ref")),
                    type=rel["rel_type"],
                    start_id=start_id,
                    end_id=end_id,
                    graph=graph_id,
                    properties={
                        "lifecycle": "active",
                        "prov_field_ref": rel.get("field_ref"),
                        "prov_source_id": graph_id,
                        **(rel.get("properties") or {}),
                    },
                )
            )
            edges += 1
            continue

        nn = normalize_name(rel["end_name"])
        graph.upsert_node(
            GraphNode(
                id=end_id,
                labels=["Entity", "SourceEntity", end_type],
                graph=graph_id,
                properties={
                    "id": end_id,
                    "name": nn or str(rel.get("end_name") or "").strip(),
                    "normalized_name": nn,
                    "entity_type": end_type,
                    "lifecycle": "active",
                    "prov_source_id": graph_id,
                    "prov_field_ref": rel.get("field_ref"),
                },
            )
        )
        nodes += 1
        graph.upsert_edge(
            GraphEdge(
                id=edge_id(rel["rel_type"], pid, end_id, pid, rel.get("field_ref")),
                type=rel["rel_type"],
                start_id=pid,
                end_id=end_id,
                graph=graph_id,
                properties={
                    "lifecycle": "active",
                    "prov_field_ref": rel.get("field_ref"),
                    "prov_source_id": graph_id,
                    **(rel.get("properties") or {}),
                },
            )
        )
        edges += 1
    return {"nodes": nodes, "edges": edges}


def _apply_ontology(platform, ontology_id: str, graph_id: str) -> None:
    if not ontology_id or not hasattr(platform.graph, "ontology"):
        return
    try:
        ontology = platform.kg_registry.load_ontology_object(ontology_id)
    except Exception:
        try:
            ontology = platform.ontology_for(graph_id)
        except Exception:
            return
    platform.graph.ontology = combine_ontologies(platform.graph.ontology, ontology)
    if hasattr(platform.graph, "_allowed_rel_types"):
        platform.graph._allowed_rel_types = set(platform.graph.ontology.relationship_allowlist) | {"RESOLVES_TO"}
    if hasattr(platform.graph, "_allowed_labels"):
        platform.graph._allowed_labels = set(platform.graph.ontology.entity_types) | {
            "Entity",
            "SourceEntity",
            "Canonical",
            "Record",
        }


def _register_identity(source_id: str, mapping: dict[str, Any]) -> None:
    template = str(mapping.get("product_id_template") or "x:product:{id}")
    prefix = template.split(":", 1)[0].strip() or slug(source_id)[:16] or "src"
    register_source_identity(source_id, prefix, "product", "Product")


def _materials(product: dict[str, Any]) -> list[tuple[str, float | None]]:
    text = _present(product.get("composition"))
    if not text:
        return []
    found: list[tuple[str, float | None]] = []
    for match in _COMPOSITION.finditer(text):
        if match.group("name"):
            pct = match.group("pct")
            found.append((match.group("name").strip(), float(pct) if pct else None))
        elif match.group("only"):
            token = match.group("only").strip()
            if token.lower() not in {"and", "with", "of"}:
                found.append((token, None))
    return found


def _split_values(raw: Any) -> list[str]:
    text = _present(raw)
    if not text:
        return []
    return [part.strip() for part in text.split(",") if _present(part)]


def _row_key(row: dict[str, Any], mapping: dict[str, Any]) -> str:
    parts = [_cell(row.get(str(pk))) for pk in (mapping.get("primary_key") or ["code"])]
    return ":".join(p for p in parts if p) or "?"


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value).strip()


def _present(value: Any) -> str:
    text = _cell(value)
    if not text or text.lower() in {"null", "none", "n/a", "na", "-", "#n/a"}:
        return ""
    return text


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
