"""Sync e-comm Postgres master_products (+ change events) → Neo4j.

Pipeline: Postgres → normalize/schema → relationships → Neo4j upsert.
Requires FASHIONGPT_ECOM_DSN in .env.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from knowledge_graph.adapters.ecom_events import build_event_graph
from knowledge_graph.adapters.ecom_master import build_relationships, graph_sketch, normalize_master_row
from knowledge_graph.config import Settings
from knowledge_graph.errors import ValidationFailed
from knowledge_graph.ids import edge_id, register_source_identity, slug
from knowledge_graph.logging_config import get_logger
from knowledge_graph.models import GraphEdge, GraphNode
from knowledge_graph.ontology import load_ontology, normalize_name

LOGGER = get_logger("knowledge_graph.ecom_sync")

GRAPH_ID = "ecom_products"
SOURCE_TYPE = "ecom_products"
_SELECT = """
SELECT
  master_id, brand, name, reference, display_reference, description,
  composition, composition_detail, construction, care, fit, country_of_origin,
  section, category, family, subfamily, category_path,
  availability, in_stock_count, out_stock_count,
  currency, price, original_price, price_min, price_max,
  image, images, colors, color_count,
  product_url, product_url_with_variant, source_url, domain, catalog_id,
  scraped_at, content_hash, pdp_enriched, updated_at
FROM master_products
WHERE (%(brand)s::text IS NULL OR brand = %(brand)s)
  AND (%(since)s::timestamptz IS NULL OR updated_at > %(since)s)
ORDER BY updated_at ASC NULLS LAST, brand ASC, master_id ASC
"""
_EVENTS_SELECT = """
SELECT
  id, brand, master_id, product_url, scrape_job_id, catalog_id,
  detected_at, scraped_at, change_type, fields_changed,
  before_json, after_json, content_hash_before, content_hash_after
FROM product_change_events
WHERE id > %(after_id)s
  AND (%(brand)s::text IS NULL OR brand = %(brand)s)
ORDER BY id ASC
"""


def register_ecom_identity() -> None:
    register_source_identity(SOURCE_TYPE, "ecom", "product", "Product")


def sync_state_path(settings: Settings) -> Path:
    return settings.generated / "ecom_sync_state.json"


def events_sync_state_path(settings: Settings) -> Path:
    return settings.generated / "ecom_events_sync_state.json"


def load_sync_state(settings: Settings) -> dict:
    path = sync_state_path(settings)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_sync_state(settings: Settings, state: dict) -> None:
    path = sync_state_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def load_events_sync_state(settings: Settings) -> dict:
    path = events_sync_state_path(settings)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_events_sync_state(settings: Settings, state: dict) -> None:
    path = events_sync_state_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def require_ecom_dsn(settings: Settings) -> str:
    dsn = settings.resolve_ecom_dsn()
    if not dsn:
        raise ValidationFailed(
            "Postgres DSN is empty. Set FASHIONGPT_ECOM_DSN or POSTGRES_HOST/USER/PASSWORD/DB in .env."
        )
    return dsn


def _load_ecom_ontology(platform):
    """Prefer API-registered ecom ontology; fall back to classic schema filenames."""

    try:
        return platform.kg_registry.load_ontology_object("ecom")
    except Exception:
        pass
    settings: Settings = platform.settings
    candidates = [
        settings.path(Path("schemas/ontologies/ecom.json")),
        settings.path(Path("schemas/ecom-products-ontology.json")),
    ]
    for path in candidates:
        if path.is_file():
            return load_ontology(path)
    raise ValidationFailed(
        "E-com ontology not found. POST /ontologies with id=ecom first "
        "(writes data/schemas/ontologies/ecom.json), or place ecom-products-ontology.json under data/schemas/."
    )


def iter_master_products(
    dsn: str,
    *,
    brand: str | None = None,
    since: str | None = None,
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise ValidationFailed("psycopg is not installed. Run: pip install 'psycopg[binary]'") from exc

    produced = 0
    with psycopg.connect(dsn) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(_SELECT, {"brand": brand, "since": since})
            while True:
                batch = cur.fetchmany(200)
                if not batch:
                    break
                for row in batch:
                    yield dict(row)
                    produced += 1
                    if limit is not None and produced >= limit:
                        return


def iter_change_events(
    dsn: str,
    *,
    after_id: int = 0,
    brand: str | None = None,
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise ValidationFailed("psycopg is not installed. Run: pip install 'psycopg[binary]'") from exc

    produced = 0
    with psycopg.connect(dsn) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(_EVENTS_SELECT, {"after_id": after_id, "brand": brand})
            while True:
                batch = cur.fetchmany(200)
                if not batch:
                    break
                for row in batch:
                    yield dict(row)
                    produced += 1
                    if limit is not None and produced >= limit:
                        return


def upsert_event_graph(graph, sketch: dict[str, Any]) -> dict[str, int]:
    """Write PriceChangeEvent / PriceSnapshot / AvailabilitySnapshot nodes for one PG event."""

    pid = sketch["product_id"]
    product = graph.get_node(pid, include_deleted=False)
    stub = 0
    if product is None:
        # Keep temporal facts even when master sync has not created the Product yet.
        graph.upsert_node(
            GraphNode(
                id=pid,
                labels=["Entity", "Record", "Product"],
                graph=GRAPH_ID,
                properties={
                    "id": pid,
                    "name": sketch.get("master_id") or pid,
                    "normalized_name": normalize_name(str(sketch.get("master_id") or pid)),
                    "brand": sketch.get("brand"),
                    "master_id": sketch.get("master_id"),
                    "lifecycle": "active",
                    "stub": True,
                    "classification": "internal",
                    "prov_source_id": GRAPH_ID,
                    "prov_source_type": SOURCE_TYPE,
                    "prov_record_id": f"{sketch.get('brand')}:{sketch.get('master_id')}",
                },
            )
        )
        product = graph.get_node(pid, include_deleted=False)
        stub = 1

    nodes = 0
    edges = 0
    for item in sketch.get("nodes") or []:
        props = {
            **(item.get("properties") or {}),
            "id": item["id"],
            "graph": GRAPH_ID,
            "lifecycle": "active",
            "prov_source_id": GRAPH_ID,
            "prov_source_type": SOURCE_TYPE,
            "classification": "internal",
        }
        graph.upsert_node(
            GraphNode(
                id=item["id"],
                labels=list(item.get("labels") or ["Entity"]),
                graph=GRAPH_ID,
                properties=props,
            )
        )
        nodes += 1

    for rel in sketch.get("edges") or []:
        start_id = rel["start_id"]
        end_id = rel["end_id"]
        graph.upsert_edge(
            GraphEdge(
                id=edge_id(rel["rel_type"], start_id, end_id, str(sketch.get("pg_event_id")), rel.get("field_ref")),
                type=rel["rel_type"],
                start_id=start_id,
                end_id=end_id,
                graph=GRAPH_ID,
                properties={
                    "lifecycle": "active",
                    "prov_field_ref": rel.get("field_ref"),
                    "prov_source_id": GRAPH_ID,
                    "pg_event_id": sketch.get("pg_event_id"),
                },
            )
        )
        edges += 1

    patch = sketch.get("product_patch") or {}
    if patch:
        merged = dict(product.properties or {})
        merged.update(patch)
        merged["id"] = pid
        graph.upsert_node(
            GraphNode(
                id=pid,
                labels=list(product.labels or ["Entity", "Record", "Product"]),
                graph=product.graph or GRAPH_ID,
                properties=merged,
            )
        )

    return {"nodes": nodes, "edges": edges, "orphan": 0, "stub": stub}


def upsert_product_graph(graph, product: dict[str, Any], relationships: list[dict[str, Any]]) -> dict[str, int]:
    """Write one product + its relationship endpoints into Neo4j (or memory graph)."""

    sketch = graph_sketch(product)
    pid = product["product_id"]
    nodes = 0
    edges = 0

    graph.upsert_node(
        GraphNode(
            id=pid,
            labels=["Entity", "Record", "Product"],
            graph=GRAPH_ID,
            properties={
                **sketch["properties"],
                "id": pid,
                "name": product.get("product_name") or pid,
                "normalized_name": normalize_name(product.get("product_name") or pid),
                "brand": product.get("brand"),
                "master_id": product.get("master_id"),
                "classification": "internal",
                "prov_source_id": GRAPH_ID,
                "prov_record_id": f"{product.get('brand')}:{product.get('master_id')}",
                "prov_source_type": SOURCE_TYPE,
                "lifecycle": "active",
            },
        )
    )
    nodes += 1

    local_ids = {"record": pid}
    for rel in relationships:
        end_id = rel.get("end_id") or f"{rel['end_type'].lower()}:{slug(rel['end_name'])}"
        end_type = rel["end_type"]
        if rel["rel_type"] == "PARENT_CATEGORY":
            start_id = rel.get("start_id") or f"garmentcategory:{slug(rel['start_name'])}"
            for node_id, node_type, node_name in (
                (start_id, rel["start_type"], rel["start_name"]),
                (end_id, end_type, rel["end_name"]),
            ):
                nn = normalize_name(node_name)
                graph.upsert_node(
                    GraphNode(
                        id=node_id,
                        labels=["Entity", "SourceEntity", node_type],
                        graph=GRAPH_ID,
                        properties={
                            "id": node_id,
                            # Facet display name = normalized (lowercase); matching uses normalized_name.
                            "name": nn or str(node_name or "").strip(),
                            "normalized_name": nn,
                            "entity_type": node_type,
                            "lifecycle": "active",
                            "prov_source_id": GRAPH_ID,
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
                    graph=GRAPH_ID,
                    properties={
                        "lifecycle": "active",
                        "prov_field_ref": rel.get("field_ref"),
                        "prov_source_id": GRAPH_ID,
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
                graph=GRAPH_ID,
                properties={
                    "id": end_id,
                    "name": nn or str(rel.get("end_name") or "").strip(),
                    "normalized_name": nn,
                    "entity_type": end_type,
                    "lifecycle": "active",
                    "prov_source_id": GRAPH_ID,
                    "prov_field_ref": rel.get("field_ref"),
                },
            )
        )
        nodes += 1
        local_ids[end_id] = end_id
        graph.upsert_edge(
            GraphEdge(
                id=edge_id(rel["rel_type"], pid, end_id, pid, rel.get("field_ref")),
                type=rel["rel_type"],
                start_id=pid,
                end_id=end_id,
                graph=GRAPH_ID,
                properties={
                    "lifecycle": "active",
                    "prov_field_ref": rel.get("field_ref"),
                    "prov_source_id": GRAPH_ID,
                    **(rel.get("properties") or {}),
                },
            )
        )
        edges += 1

    return {"nodes": nodes, "edges": edges}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_sync_state(
    settings: Settings,
    *,
    status: str,
    products: int,
    brands: dict[str, int],
    errors: int,
    last_updated_at: str | None,
    since: str | None = None,
    full: bool = False,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist ecom sync progress so generated/ stays current during long runs."""

    state = {
        "status": status,
        "last_run_at": _utc_now(),
        "last_updated_at": last_updated_at,
        "last_products": products,
        "brands": brands,
        "errors": errors,
        "full": full,
        "since": since,
        "graph": GRAPH_ID,
        "source_table": "master_products",
    }
    if extras:
        state.update(extras)
    save_sync_state(settings, state)
    return state


def refresh_ecom_stats_only(platform) -> dict[str, Any]:
    """Lightweight mid-sync refresh: column value counts for fast basic asks."""

    from knowledge_graph.retrieval.kg_stats import refresh_ecom_stats

    stats = refresh_ecom_stats(platform)
    totals = stats.get("totals") or {}
    return {
        "graph": GRAPH_ID,
        "stats_products": totals.get("products"),
        "stats_brands": totals.get("brands"),
        "stats_path": str(platform.settings.generated / "kg_stats" / f"{GRAPH_ID}.json"),
    }


def refresh_generated_after_ecom_sync(platform, *, heavy: bool = True) -> dict[str, Any]:
    """Refresh stats (+ optional snapshots/system meta) so generated/ matches the live graph.

    ``heavy=False`` only rewrites kg_stats (safe during long syncs). Full snapshots /
    system meta dump Neo4j and should run at sync end only.
    """

    settings: Settings = platform.settings
    refreshed: dict[str, Any] = {"graph": GRAPH_ID}
    try:
        refreshed["stats"] = refresh_ecom_stats_only(platform)
    except Exception as exc:
        LOGGER.warning("ecom sync: kg stats refresh failed: %s", exc)
        refreshed["stats_error"] = str(exc)
    if not heavy:
        return refreshed

    from knowledge_graph.graph.snapshots import save_master_snapshot, save_source_snapshots
    from knowledge_graph.retrieval.system_meta import build_system_meta

    try:
        refreshed["shared"] = platform.build_shared()
    except Exception as exc:
        LOGGER.exception("ecom sync: shared graph refresh failed")
        refreshed["shared_error"] = str(exc)
        return refreshed
    try:
        paths = save_source_snapshots(platform.graph, settings.generated, [GRAPH_ID])
        refreshed["source_snapshot"] = [str(path) for path in paths]
    except Exception as exc:
        LOGGER.warning("ecom sync: source snapshot refresh failed: %s", exc)
        refreshed["source_snapshot_error"] = str(exc)
    try:
        master = save_master_snapshot(platform.graph, settings.generated)
        refreshed["master_snapshot"] = str(master) if master else None
    except Exception as exc:
        LOGGER.warning("ecom sync: master snapshot refresh failed: %s", exc)
        refreshed["master_snapshot_error"] = str(exc)
    try:
        meta = build_system_meta(platform, persist=True)
        refreshed["system_meta"] = {
            "graphs": len(meta.get("graphs") or []),
            "path": str(settings.generated / "kg_system_meta.json"),
        }
    except Exception as exc:
        LOGGER.warning("ecom sync: system meta refresh failed: %s", exc)
        refreshed["system_meta_error"] = str(exc)
    try:
        from knowledge_graph.retrieval.schema_hooks import on_schema_changed

        refreshed["schema_hooks"] = on_schema_changed(platform, graph_id=GRAPH_ID)
    except Exception as exc:
        LOGGER.warning("ecom sync: schema hooks failed: %s", exc)
        refreshed["schema_hooks_error"] = str(exc)
    return refreshed


def _apply_ecom_ontology(platform) -> None:
    """Allowlist ecom labels/rels on the live graph backend."""

    ecom_ontology = _load_ecom_ontology(platform)
    if not hasattr(platform.graph, "ontology"):
        return
    from knowledge_graph.ontology import combine_ontologies

    platform.graph.ontology = combine_ontologies(platform.graph.ontology, ecom_ontology)
    if hasattr(platform.graph, "_allowed_rel_types"):
        platform.graph._allowed_rel_types = set(platform.graph.ontology.relationship_allowlist) | {"RESOLVES_TO"}
    if hasattr(platform.graph, "_allowed_labels"):
        platform.graph._allowed_labels = set(platform.graph.ontology.entity_types) | {
            "Entity",
            "SourceEntity",
            "Canonical",
            "Record",
        }


def sync_ecom_events_to_neo4j(
    platform,
    *,
    brand: str | None = None,
    full: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Incremental sync of product_change_events → price/availability temporal nodes."""

    register_ecom_identity()
    settings: Settings = platform.settings
    dsn = require_ecom_dsn(settings)
    _apply_ecom_ontology(platform)

    state = load_events_sync_state(settings)
    after_id = 0 if full else int(state.get("last_event_id") or 0)
    rows = 0
    applied = 0
    skipped = 0
    stubs = 0
    node_writes = 0
    edge_writes = 0
    errors: list[str] = []
    last_event_id = after_id
    by_type: dict[str, int] = {}

    save_events_sync_state(
        settings,
        {
            "status": "running",
            "last_run_at": _utc_now(),
            "last_event_id": after_id,
            "brand_filter": brand,
            "full": full,
            "limit": limit,
        },
    )
    LOGGER.info("ecom events sync start after_id=%s brand=%s limit=%s", after_id, brand, limit)

    for row in iter_change_events(dsn, after_id=after_id, brand=brand, limit=limit):
        rows += 1
        event_pk = row.get("id")
        try:
            if event_pk is not None:
                last_event_id = max(last_event_id, int(event_pk))
            change_type = str(row.get("change_type") or "")
            by_type[change_type] = by_type.get(change_type, 0) + 1
            sketch = build_event_graph(row)
            if sketch is None:
                skipped += 1
                continue
            tallies = upsert_event_graph(platform.graph, sketch)
            if tallies.get("orphan"):
                skipped += 1
                continue
            applied += 1
            stubs += int(tallies.get("stub") or 0)
            node_writes += tallies["nodes"]
            edge_writes += tallies["edges"]
            if applied % 500 == 0:
                save_events_sync_state(
                    settings,
                    {
                        "status": "running",
                        "last_run_at": _utc_now(),
                        "last_event_id": last_event_id,
                        "rows": rows,
                        "applied": applied,
                        "skipped": skipped,
                        "stubs": stubs,
                        "by_type": by_type,
                        "node_writes": node_writes,
                        "edge_writes": edge_writes,
                        "errors": len(errors),
                        "brand_filter": brand,
                        "full": full,
                    },
                )
                LOGGER.info("ecom events sync progress applied=%s last_id=%s", applied, last_event_id)
        except Exception as exc:
            key = f"{row.get('brand')}:{row.get('master_id')}:{event_pk}"
            errors.append(f"{key}: {exc}")
            LOGGER.warning("ecom events sync row failed %s: %s", key, exc, exc_info=True)
            if len(errors) >= 50:
                errors.append("… further errors truncated")
                break

    finished = _utc_now()
    report = {
        "ok": not errors,
        "graph": GRAPH_ID,
        "source_table": "product_change_events",
        "after_id": after_id,
        "last_event_id": last_event_id,
        "rows": rows,
        "applied": applied,
        "skipped": skipped,
        "stubs": stubs,
        "by_type": by_type,
        "node_writes": node_writes,
        "edge_writes": edge_writes,
        "errors": errors,
        "finished_at": finished,
    }
    save_events_sync_state(
        settings,
        {
            "status": "failed" if errors else "ok",
            "last_run_at": finished,
            "last_event_id": last_event_id,
            "rows": rows,
            "applied": applied,
            "skipped": skipped,
            "stubs": stubs,
            "by_type": by_type,
            "node_writes": node_writes,
            "edge_writes": edge_writes,
            "errors": len(errors),
            "brand_filter": brand,
            "full": full,
            "finished_at": finished,
        },
    )
    LOGGER.info(
        "ecom events sync done rows=%s applied=%s stubs=%s errors=%s last_id=%s",
        rows,
        applied,
        stubs,
        len(errors),
        last_event_id,
    )
    return report


def sync_ecom_to_neo4j(
    platform,
    *,
    brand: str | None = None,
    full: bool = False,
    limit: int | None = None,
    refresh_generated: bool = True,
    sync_events: bool = True,
) -> dict[str, Any]:
    """Read master_products and upsert into the live graph backend."""

    register_ecom_identity()
    settings: Settings = platform.settings
    dsn = require_ecom_dsn(settings)
    _apply_ecom_ontology(platform)

    state = load_sync_state(settings)
    since = None if full else state.get("last_updated_at")
    cap = limit if limit is not None else settings.ecom_sync_limit

    products = 0
    node_writes = 0
    edge_writes = 0
    errors: list[str] = []
    latest_updated = since
    brands: dict[str, int] = {}
    progress_every = max(100, int(settings.ecom_sync_batch_size or 500))

    _write_sync_state(
        settings,
        status="running",
        products=0,
        brands={},
        errors=0,
        last_updated_at=latest_updated,
        since=since,
        full=full,
        extras={"brand_filter": brand, "limit": cap},
    )
    LOGGER.info("ecom sync start brand=%s since=%s limit=%s", brand, since, cap)
    for row in iter_master_products(dsn, brand=brand, since=since, limit=cap):
        try:
            product = normalize_master_row(row)
            rels = build_relationships(product)
            tallies = upsert_product_graph(platform.graph, product, rels)
            products += 1
            node_writes += tallies["nodes"]
            edge_writes += tallies["edges"]
            brands[product["brand"]] = brands.get(product["brand"], 0) + 1
            updated = row.get("updated_at")
            if updated is not None:
                text = updated.isoformat() if hasattr(updated, "isoformat") else str(updated)
                if latest_updated is None or text > str(latest_updated):
                    latest_updated = text
            if products % progress_every == 0:
                extras: dict[str, Any] = {
                    "node_writes": node_writes,
                    "edge_writes": edge_writes,
                }
                # Mid-sync: refresh lightweight column stats only (not Neo4j snapshots).
                if products % (progress_every * 2) == 0:
                    try:
                        extras["generated"] = refresh_generated_after_ecom_sync(platform, heavy=False)
                    except Exception as exc:
                        extras["generated_error"] = str(exc)
                _write_sync_state(
                    settings,
                    status="running",
                    products=products,
                    brands=brands,
                    errors=len(errors),
                    last_updated_at=latest_updated,
                    since=since,
                    full=full,
                    extras=extras,
                )
                LOGGER.info("ecom sync progress products=%s", products)
        except Exception as exc:
            key = f"{row.get('brand')}:{row.get('master_id')}"
            errors.append(f"{key}: {exc}")
            # Include the exception text — JSON logs only keep error_type from exc_info.
            LOGGER.warning("ecom sync row failed %s: %s", key, exc, exc_info=True)
            if len(errors) >= 50:
                errors.append("… further errors truncated")
                break

    finished = _utc_now()
    new_state = _write_sync_state(
        settings,
        status="failed" if errors else "ok",
        products=products,
        brands=brands,
        errors=len(errors),
        last_updated_at=latest_updated,
        since=since,
        full=full,
        extras={"node_writes": node_writes, "edge_writes": edge_writes, "finished_at": finished},
    )
    report = {
        "ok": not errors,
        "graph": GRAPH_ID,
        "products": products,
        "node_writes": node_writes,
        "edge_writes": edge_writes,
        "brands": brands,
        "since": since,
        "last_updated_at": latest_updated,
        "errors": errors,
        "finished_at": finished,
        "sync_state": new_state,
    }
    if sync_events:
        # Events attach to existing Product nodes; run after product upsert.
        # Never inherit product `limit` — watermark is independent and must not skip ids.
        try:
            report["events"] = sync_ecom_events_to_neo4j(
                platform,
                brand=brand,
                full=full,
                limit=None,
            )
            if not report["events"].get("ok"):
                report["ok"] = False
        except Exception as exc:
            LOGGER.warning("ecom events sync failed: %s", exc, exc_info=True)
            report["events_error"] = str(exc)
            report["ok"] = False
    if refresh_generated and report.get("ok"):
        report["generated"] = refresh_generated_after_ecom_sync(platform)
        if report["generated"].get("shared_error"):
            report["shared_error"] = report["generated"]["shared_error"]
            report["ok"] = False
        # Re-stamp state after generated refresh so mtime/content stay latest.
        new_state = _write_sync_state(
            settings,
            status="ok" if report["ok"] else "failed",
            products=products,
            brands=brands,
            errors=0 if report["ok"] else 1,
            last_updated_at=latest_updated,
            since=since,
            full=full,
            extras={
                "node_writes": node_writes,
                "edge_writes": edge_writes,
                "finished_at": finished,
                "generated": report.get("generated"),
                "events": report.get("events"),
            },
        )
        report["sync_state"] = new_state
    LOGGER.info("ecom sync done products=%s edges=%s errors=%s", products, edge_writes, len(errors))
    return report


def ping_ecom_db(settings: Settings) -> dict[str, Any]:
    """Test the configured DSN and return master_products counts by brand."""

    dsn = require_ecom_dsn(settings)
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise ValidationFailed("psycopg is not installed. Run: pip install 'psycopg[binary]'") from exc

    with psycopg.connect(dsn) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT brand, count(*)::int AS n FROM master_products GROUP BY brand ORDER BY brand"
            )
            rows = list(cur.fetchall())
            cur.execute("SELECT count(*)::int AS n FROM master_products")
            total = cur.fetchone()["n"]
    return {
        "ok": True,
        "total": total,
        "by_brand": {row["brand"]: row["n"] for row in rows},
    }
