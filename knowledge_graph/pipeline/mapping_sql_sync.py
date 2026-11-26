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

from knowledge_graph import etl_normalizer as norm
from knowledge_graph.errors import ValidationFailed
from knowledge_graph.ids import edge_id, register_source_identity, slug
from knowledge_graph.logging_config import get_logger
from knowledge_graph.models import GraphEdge, GraphNode
from knowledge_graph.ontology import combine_ontologies, facet_name_fields, normalize_name
from knowledge_graph.sources.mapping_store import get_mapping, mapping_exists

LOGGER = get_logger("knowledge_graph.mapping_sql_sync")

_SPLIT_NOTES = re.compile(r"split\s+on\s+comma", re.I)
_SHARED_HUB_TYPES = frozenset(
    {
        "Brand",
        "Material",
        "Color",
        "ColorFamily",
        "MaterialGroup",
        "TargetGroup",
        "Reference",
        "Gauge",
        "Construction",
        "Occasion",
        "Pattern",
        "Fit",
        "CareInstruction",
        "RetailGroup",
        "PriceBand",
        "SeasonYear",
    }
)


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
    buyer_prefixes: list[str] | None = None,
    newest_first: bool = False,
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
    # Limited product slices should prefer newest rows for better ecom overlap.
    order_desc = bool(newest_first or (limit is not None and full))

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
            "limit": limit,
            "buyer_prefixes": buyer_prefixes or [],
        },
    )
    LOGGER.info(
        "mapping sql sync start source=%s table=%s full=%s limit=%s since=%s buyers=%s newest_first=%s",
        source_id,
        table,
        full,
        limit,
        since,
        buyer_prefixes,
        order_desc,
    )

    for row in _iter_rows(
        dsn,
        table,
        columns,
        watermark_col=watermark_col,
        since=since,
        limit=limit,
        order_desc=order_desc,
        buyer_prefixes=buyer_prefixes,
    ):
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
            from knowledge_graph.pipeline.fashion_brain_enrichment import run_enrichment

            report["fashion_brain_enrichment"] = run_enrichment(platform)
        except Exception as exc:
            LOGGER.warning("mapping sql sync fashion_brain enrichment failed: %s", exc)
            report["enrichment_error"] = str(exc)
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
    order_desc: bool = False,
    buyer_prefixes: list[str] | None = None,
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
    where_parts: list[Any] = []
    if watermark_col and since:
        where_parts.append(sql.SQL("{wm} > %(since)s").format(wm=sql.Identifier(watermark_col)))
        params["since"] = since

    clean_buyers = [str(b).strip() for b in (buyer_prefixes or []) if str(b).strip()]
    if clean_buyers:
        if "buyer" not in columns:
            raise ValidationFailed("buyer_prefixes requires 'buyer' in mapping columns")
        buyer_clauses = []
        for i, prefix in enumerate(clean_buyers):
            key = f"buyer_p{i}"
            buyer_clauses.append(
                sql.SQL("UPPER(COALESCE({col}, '')) LIKE {ph}").format(
                    col=sql.Identifier("buyer"),
                    ph=sql.Placeholder(key),
                )
            )
            params[key] = f"%{prefix.upper()}%"
        where_parts.append(sql.SQL("({})").format(sql.SQL(" OR ").join(buyer_clauses)))

    if where_parts:
        select = sql.SQL("{base} WHERE {pred}").format(
            base=select,
            pred=sql.SQL(" AND ").join(where_parts),
        )

    if watermark_col:
        direction = sql.SQL("DESC NULLS LAST") if order_desc else sql.SQL("ASC NULLS LAST")
        select = sql.SQL("{base} ORDER BY {wm} {direction}").format(
            base=select,
            wm=sql.Identifier(watermark_col),
            direction=direction,
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

    _apply_row_transforms(product, source_id)

    template = str(mapping.get("product_id_template") or "{code}")
    try:
        product_id = template.format(**{k: _cell(v) for k, v in product.items()})
    except Exception as exc:
        raise ValueError(f"product_id_template failed: {exc}") from exc
    if not product_id or "{" in product_id:
        raise ValueError(f"Could not build product_id from template {template!r}")
    product["product_id"] = product_id
    product["name"] = _cell(
        product.get("name")
        or product.get("display_name")
        or product.get("code")
        or product_id
    )

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


def _apply_row_transforms(product: dict[str, Any], source_id: str) -> None:
    """Normalize shared fields before relationship / property emission."""

    brand, tg_from_brand = norm.clean_brand(product.get("brand"))
    if brand is not None:
        product["brand"] = brand
    elif "brand" in product:
        product["brand"] = None
    if tg_from_brand:
        product["_target_group"] = tg_from_brand

    if product.get("gender"):
        product["_target_group"] = product.get("_target_group") or norm.target_group_from_gender(
            product.get("gender")
        )
    if product.get("section") or product.get("category_path"):
        product["_target_group"] = product.get("_target_group") or norm.target_group_from_section(
            product.get("section"), product.get("category_path")
        )

    product["season_year"] = norm.clean_season_year(product.get("season_year")) or product.get(
        "season_year"
    )

    price_num = norm.parse_decimal(product.get("price"))
    original_price_num = norm.parse_decimal(product.get("original_price"))
    if price_num is not None:
        product["price_num"] = price_num
    if original_price_num is not None:
        product["original_price_num"] = original_price_num
    discount = norm.pct_discount(original_price_num, price_num)
    if discount is not None:
        product["discount_pct"] = discount
    band = norm.price_band(price_num, product.get("currency"))
    if band:
        product["_price_band"] = band

    style_ref = norm.normalize_reference(product.get("style_reference"))
    cust_ref = norm.normalize_reference(product.get("customer_style_reference"))
    ecom_ref = norm.coalesce_reference(product.get("reference"), product.get("display_reference"))
    if style_ref:
        product["_style_ref_norm"] = style_ref
    if cust_ref:
        product["_cust_ref_norm"] = cust_ref
    if ecom_ref:
        product["_ref_norm"] = ecom_ref
        product["ref_norm"] = ecom_ref

    if source_id.startswith("plmex") or product.get("code"):
        product["display_name"] = norm.synthesize_plmex_display_name(product)
        product["search_text"] = norm.synthesize_plmex_search_text(product)
        product.setdefault("name", product["display_name"])
    else:
        product["search_text"] = norm.synthesize_ecom_search_text(product)


def _entity_id(end_type: str, name: str, *, id_template: str | None = None, product: dict | None = None) -> str:
    value = _present(name)
    if not value:
        raise ValueError("empty entity name")
    if id_template and product is not None:
        ctx = {**product, "token": value, "fiber": value, "value": value, "name": value}
        try:
            rendered = id_template.format(**{k: _cell(v) for k, v in ctx.items()})
            if rendered and "{" not in rendered:
                return rendered
        except Exception:
            pass
    kind_map = {
        "Brand": "brand",
        "Material": "material",
        "Color": "color",
        "Gauge": "gauge",
        "Construction": "construction",
        "Occasion": "occasion",
        "Pattern": "pattern",
        "Fit": "fit",
        "CareInstruction": "care",
        "TargetGroup": "targetgroup",
        "Reference": "reference",
        "RetailGroup": "retailgroup",
        "MaterialGroup": "materialgroup",
        "ColorFamily": "colorfamily",
        "PriceBand": "priceband",
        "SeasonYear": "seasonyear",
        "ProductionMethod": "method",
    }
    kind = kind_map.get(end_type)
    if kind:
        if end_type == "Reference":
            ref = norm.normalize_reference(value) or value
            return norm.reference_id(ref)
        if end_type == "PriceBand":
            return norm.hub_id("priceband", value)
        if end_type == "ProductionMethod":
            return f"plmex:method:{slug(normalize_name(value) or value)}"
        return norm.hub_id(kind, value)
    prefix = "plmex" if end_type.startswith("Plm") or end_type in {
        "Buyer",
        "Gender",
        "SampleType",
        "DesignPurpose",
        "MachineType",
        "Department",
        "ShoppingDepartment",
        "GarmentType",
        "Unit",
        "Location",
        "Person",
        "InventoryRecord",
        "Shipment",
        "EventTag",
        "ConstructionTag",
        "AiMaterialTag",
        "FinishTag",
        "SpecialTag",
        "DesignFeature",
        "StyleFeature",
    } else end_type.lower()
    if end_type in {"InventoryRecord", "Shipment", "Location", "Person", "Buyer", "Unit", "Gender",
                    "SampleType", "DesignPurpose", "MachineType", "Department", "ShoppingDepartment",
                    "GarmentType", "EventTag", "ConstructionTag", "AiMaterialTag", "FinishTag",
                    "SpecialTag", "DesignFeature", "StyleFeature", "ProductionMethod"}:
        return f"plmex:{end_type.lower()}:{slug(normalize_name(value) or value)}"
    return f"{end_type.lower()}:{normalize_name(value).replace(' ', '-')}"


def _build_relationships(product: dict[str, Any], specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pid = product["product_id"]
    rels: list[dict[str, Any]] = []

    def add(
        rel_type: str,
        end_type: str,
        name: str,
        *,
        field_ref: str,
        properties: dict | None = None,
        end_id: str | None = None,
        start_id: str | None = None,
        start_type: str = "Product",
        start_name: str | None = None,
        node_properties: dict | None = None,
    ) -> None:
        value = _present(name)
        if not value and not end_id:
            return
        resolved_end = end_id or _entity_id(end_type, value or "unknown", product=product)
        item = {
            "rel_type": rel_type,
            "start_id": start_id or pid,
            "start_type": start_type,
            "end_type": end_type,
            "end_name": value or resolved_end,
            "end_id": resolved_end,
            "field_ref": field_ref,
            "properties": properties or {},
        }
        if start_name:
            item["start_name"] = start_name
        if node_properties:
            item["node_properties"] = node_properties
        rels.append(item)

    materials = norm.parse_composition_detail(
        product.get("composition_detail"), product.get("composition")
    )
    colors, patterns_from_color = norm.parse_colors(
        product.get("colors") if product.get("colors") not in (None, "", []) else product.get("color")
    )
    gauges, constructions_from_gauge = norm.parse_gauges(product.get("gauge"))

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
                        "start_id": _entity_id(start_type, child, product=product),
                        "end_type": end_type,
                        "end_name": parent,
                        "end_id": _entity_id(end_type, parent, product=product),
                        "field_ref": f"{spec.get('start_from_column')}→{spec.get('end_from_column')}",
                        "properties": {},
                    }
                )
            continue

        if rel_type == "IN_DEPARTMENT":
            person = _present(product.get(str(spec.get("start_from_column") or "merchandiser")))
            dept = _present(product.get(str(spec.get("end_from_column") or "department")))
            if person and dept:
                add(
                    rel_type,
                    end_type,
                    dept,
                    field_ref="merchandiser→department",
                    start_id=_entity_id("Person", person, product=product),
                    start_type="Person",
                    start_name=person,
                )
            continue

        if rel_type == "STORED_AT" and str(spec.get("start_type") or "") == "InventoryRecord":
            code = _present(product.get("code"))
            loc = _present(product.get("current_location"))
            if code and loc:
                for line in re.split(r"[\r\n]+", loc):
                    token = _present(line)
                    if token:
                        add(
                            "STORED_AT",
                            "Location",
                            token,
                            field_ref="current_location",
                            start_id=norm.inventory_id(code),
                            start_type="InventoryRecord",
                            start_name=f"inv:{code}",
                        )
            continue

        if rel_type in {"USES_MATERIAL", "PRIMARY_MATERIAL"}:
            if rel_type == "USES_MATERIAL":
                for material in materials:
                    props = {"percentage": material["percentage"]} if material["percentage"] is not None else {}
                    props["provenance"] = "composition"
                    add(
                        rel_type,
                        end_type,
                        material["name"],
                        field_ref="composition",
                        properties=props,
                        end_id=material["id"],
                    )
            elif materials:
                primary = max(
                    materials,
                    key=lambda item: (
                        item["percentage"] is not None,
                        item["percentage"] if item["percentage"] is not None else -1.0,
                    ),
                )
                props = {"percentage": primary["percentage"]} if primary["percentage"] is not None else {}
                add(
                    rel_type,
                    end_type,
                    primary["name"],
                    field_ref="composition",
                    properties=props,
                    end_id=primary["id"],
                )
            continue

        if rel_type in {"HAS_COLOR", "PRIMARY_COLOR"}:
            if not colors:
                continue
            if rel_type == "PRIMARY_COLOR":
                add(rel_type, end_type, colors[0]["name"], field_ref="color", properties={"is_primary": True}, end_id=colors[0]["id"])
            else:
                for index, color in enumerate(colors):
                    add(
                        rel_type,
                        end_type,
                        color["name"],
                        field_ref="color",
                        properties={"position": index, "is_primary": index == 0},
                        end_id=color["id"],
                    )
            continue

        if rel_type == "HAS_PATTERN":
            for pattern in patterns_from_color:
                add(rel_type, end_type, pattern["name"], field_ref="color", end_id=pattern["id"])
            continue

        if rel_type == "HAS_GAUGE":
            for gauge in gauges:
                add(rel_type, end_type, gauge["name"], field_ref="gauge", end_id=gauge["id"])
            continue

        if rel_type == "HAS_CONSTRUCTION" and constructions_from_gauge and not str(spec.get("from_column") or "").startswith("ai_"):
            # Only emit gauge-derived constructions when mapping has no explicit column (or after tags).
            pass

        if rel_type == "FOR_TARGET_GROUP":
            tg = _present(product.get("_target_group"))
            if tg:
                add(
                    rel_type,
                    end_type,
                    tg,
                    field_ref="target_group",
                    properties={"provenance": "rule"},
                    end_id=norm.hub_id("targetgroup", tg),
                )
            continue

        if rel_type == "IN_PRICE_BAND":
            band = _present(product.get("_price_band"))
            if band:
                add(rel_type, end_type, band, field_ref="price", end_id=norm.hub_id("priceband", band))
            continue

        if rel_type == "HAS_REFERENCE":
            kind = str((spec.get("edge_properties") or {}).get("ref_kind") or "")
            if kind == "plm_style_ref" or str(spec.get("from_transform") or "") == "style_ref_norm":
                ref = product.get("_style_ref_norm")
                raw = product.get("style_reference")
            elif kind == "customer_style_ref" or str(spec.get("from_transform") or "") == "cust_ref_norm":
                ref = product.get("_cust_ref_norm")
                raw = product.get("customer_style_reference")
            else:
                ref = product.get("_ref_norm")
                raw = product.get("reference") or product.get("display_reference")
            if ref:
                props = {
                    "raw_reference": _present(raw),
                    "ref_kind": kind or "ecom_ref",
                }
                add(rel_type, "Reference", ref, field_ref="reference", properties=props, end_id=norm.reference_id(ref))
            continue

        if rel_type == "HAS_BRAND":
            brand = _present(product.get("brand"))
            if brand:
                add(rel_type, end_type, brand, field_ref="brand", end_id=norm.hub_id("brand", brand))
            continue

        if rel_type == "HAS_INVENTORY_RECORD":
            code = _present(product.get("code"))
            stock = product.get("current_stock")
            loc = _present(product.get("current_location"))
            if code and (stock is not None or loc):
                inv_id = norm.inventory_id(code)
                add(
                    rel_type,
                    end_type,
                    f"inv:{code}",
                    field_ref="current_stock",
                    end_id=inv_id,
                    node_properties={
                        "current_stock": stock,
                        "current_location": loc or None,
                        "updated_at": _cell(product.get("updated_at")) or None,
                    },
                )
            continue

        if rel_type == "HAS_SHIPMENT":
            code = _present(product.get("code"))
            date = _present(product.get("shipment_date"))
            qty = product.get("shipment_quantity")
            if code and (date or qty is not None):
                ship_id = norm.shipment_id(code, date)
                add(
                    rel_type,
                    end_type,
                    f"ship:{code}:{date or 'unknown'}",
                    field_ref="shipment_date",
                    end_id=ship_id,
                    node_properties={
                        "shipment_date": date or None,
                        "shipment_quantity": qty,
                    },
                )
            continue

        if rel_type == "STORED_AT" and not spec.get("start_type"):
            # Legacy Product→Location; prefer InventoryRecord path when present.
            continue

        col = str(spec.get("from_column") or "")
        raw = product.get(col) if col else None
        if raw in (None, "") and spec.get("fallback_column"):
            raw = product.get(str(spec.get("fallback_column")))
        notes = str(spec.get("notes") or "")
        split_regex = str(spec.get("split_regex") or "").strip()
        split = bool(_SPLIT_NOTES.search(notes)) or col.startswith("ai_") or bool(spec.get("split"))

        if rel_type == "SUITED_FOR_OCCASION":
            end_type = "Occasion"
        elif rel_type == "HAS_CONSTRUCTION" and col.startswith("ai_"):
            end_type = "Construction"
        elif rel_type == "MADE_WITH_METHOD":
            end_type = "ProductionMethod"

        if split_regex:
            values = [p.strip() for p in re.split(split_regex, _present(raw)) if _present(p)]
        elif split:
            values = _split_values(raw)
        else:
            values = [_present(raw)] if _present(raw) else []

        # Append gauge-derived crochet constructions alongside ai construction tags.
        if rel_type == "HAS_CONSTRUCTION":
            for construction in constructions_from_gauge:
                add(
                    rel_type,
                    "Construction",
                    construction["name"],
                    field_ref="gauge",
                    properties={"provenance": "gauge"},
                    end_id=construction["id"],
                )

        for index, value in enumerate(values):
            props: dict[str, Any] = {}
            if rel_type in {"HAS_COLOR", "PRIMARY_COLOR"}:
                props = {"position": index, "is_primary": index == 0}
            if rel_type in {"SUITED_FOR_OCCASION", "HAS_CONSTRUCTION"} and col.startswith("ai_"):
                props = {"provenance": "ai_tag"}
            if rel_type == "PRIMARY_COLOR" and index > 0:
                continue
            edge_props = spec.get("edge_properties")
            if isinstance(edge_props, dict):
                props = {
                    **props,
                    **{k: v for k, v in edge_props.items() if not str(v).startswith("{")},
                }
            add(rel_type, end_type, value, field_ref=col or rel_type.lower(), properties=props)
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
        start_id = rel.get("start_id") or pid
        start_type = str(rel.get("start_type") or "Product")
        node_graph = "fashion_brain" if end_type in _SHARED_HUB_TYPES else graph_id

        if rel["rel_type"] == "PARENT_CATEGORY" or start_type != "Product":
            endpoints = [
                (end_id, end_type, rel.get("end_name"), node_graph),
            ]
            if start_type != "Product":
                start_graph = "fashion_brain" if start_type in _SHARED_HUB_TYPES else graph_id
                endpoints.insert(
                    0,
                    (start_id, start_type, rel.get("start_name") or start_id, start_graph),
                )
            if rel["rel_type"] == "PARENT_CATEGORY":
                endpoints = [
                    (start_id, start_type, rel["start_name"], graph_id),
                    (end_id, end_type, rel["end_name"], graph_id),
                ]
            for node_id, node_type, node_name, n_graph in endpoints:
                props = {
                    "id": node_id,
                    **facet_name_fields(node_name),
                    "entity_type": node_type,
                    "lifecycle": "active",
                    "prov_source_id": n_graph,
                }
                if rel.get("node_properties") and node_id == end_id:
                    props.update({k: v for k, v in rel["node_properties"].items() if v is not None})
                graph.upsert_node(
                    GraphNode(
                        id=node_id,
                        labels=["Entity", "SourceEntity", node_type],
                        graph=n_graph,
                        properties=props,
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

        end_props = {
            "id": end_id,
            **facet_name_fields(rel.get("end_name")),
            "entity_type": end_type,
            "lifecycle": "active",
            "prov_source_id": node_graph,
            "prov_field_ref": rel.get("field_ref"),
        }
        if rel.get("node_properties"):
            end_props.update({k: v for k, v in rel["node_properties"].items() if v is not None})
        graph.upsert_node(
            GraphNode(
                id=end_id,
                labels=["Entity", "SourceEntity", end_type],
                graph=node_graph,
                properties=end_props,
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
    ontology = None
    try:
        ontology = platform.kg_registry.load_ontology_object(ontology_id)
    except Exception as exc:
        LOGGER.warning("mapping sql sync ontology_id=%s load failed: %s", ontology_id, exc)
        try:
            ontology = platform.ontology_for(graph_id)
        except Exception as exc2:
            LOGGER.warning("mapping sql sync ontology_for(%s) failed: %s", graph_id, exc2)
            return
    if ontology is None:
        return
    platform.graph.ontology = combine_ontologies(platform.graph.ontology, ontology)
    if hasattr(platform.graph, "_allowed_rel_types"):
        platform.graph._allowed_rel_types = set(platform.graph.ontology.relationship_allowlist) | {
            "RESOLVES_TO",
            "SIMILAR_TO",
            "IN_GROUP",
            "IN_MATERIAL_GROUP",
            "IN_COLOR_FAMILY",
            "MAPS_TO_CATEGORY",
            "REFERS_TO_MATERIAL",
            "SAME_ORG_UNIT_AS",
            "EQUIVALENT_CATEGORY",
            "FOR_TARGET_GROUP",
            "MADE_WITH_METHOD",
            "SUITED_FOR_OCCASION",
            "HAS_PATTERN",
            "IN_PRICE_BAND",
            "IN_DEPARTMENT",
            "HAS_REFERENCE",
        }
    if hasattr(platform.graph, "_allowed_labels"):
        platform.graph._allowed_labels = set(platform.graph.ontology.entity_types) | {
            "Entity",
            "SourceEntity",
            "Canonical",
            "Record",
            "TargetGroup",
            "ProductionMethod",
            "Occasion",
            "Construction",
            "Pattern",
            "Reference",
            "RetailGroup",
            "MaterialGroup",
            "ColorFamily",
            "PriceBand",
            "Buyer",
            "CanonicalCategory",
            "InventoryRecord",
            "Shipment",
            "Location",
            "Person",
            "Gauge",
            "Department",
            "ShoppingDepartment",
            "SeasonYear",
            "Gender",
            "GarmentType",
            "SampleType",
            "DesignPurpose",
            "MachineType",
            "DesignFeature",
            "StyleFeature",
            "Unit",
            "EventTag",
            "ConstructionTag",
            "AiMaterialTag",
            "FinishTag",
            "SpecialTag",
            "PlmMainCategory",
            "PlmSubCategory",
            "PlmSubSubCategory",
        }
    LOGGER.info(
        "mapping sql sync ontology applied id=%s labels=%s",
        ontology_id,
        len(getattr(platform.graph, "_allowed_labels", []) or []),
    )


def _register_identity(source_id: str, mapping: dict[str, Any]) -> None:
    template = str(mapping.get("product_id_template") or "x:product:{id}")
    prefix = template.split(":", 1)[0].strip() or slug(source_id)[:16] or "src"
    register_source_identity(source_id, prefix, "product", "Product")


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
