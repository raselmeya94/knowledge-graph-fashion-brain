"""Per-KG rollup stats written at sync/rebuild time for fast basic queries.

Stores totals + every unique column value count under ``data/generated/kg_stats/``.
Ask/retrieve paths should prefer these files for unfiltered counts instead of
loading every Neo4j Product node or probing metadata live.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from knowledge_graph.config import DATA_ROOT, Settings
from knowledge_graph.logging_config import get_logger

LOGGER = get_logger("knowledge_graph.kg_stats")

# Categorical / faceted columns from master_products (and matching Neo4j props).
_ECOM_COUNT_COLUMNS = (
    "brand",
    "category",
    "family",
    "subfamily",
    "section",
    "availability",
    "currency",
    "domain",
    "construction",
    "fit",
    "country_of_origin",
    "catalog_id",
    "enriched_from",
)


def stats_root(generated: Path | None = None) -> Path:
    root = Path(generated) if generated is not None else Path(DATA_ROOT) / "generated"
    return root / "kg_stats"


def stats_path(graph_id: str, generated: Path | None = None) -> Path:
    return stats_root(generated) / f"{graph_id}.json"


def index_path(generated: Path | None = None) -> Path:
    return stats_root(generated) / "index.json"


def load_graph_stats(graph_id: str, generated: Path | None = None) -> dict[str, Any] | None:
    path = stats_path(graph_id, generated)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def product_total(graph_id: str, generated: Path | None = None) -> int | None:
    stats = load_graph_stats(graph_id, generated)
    if not stats:
        return None
    totals = stats.get("totals") or {}
    value = totals.get("products")
    return int(value) if value is not None else None


def value_count(graph_id: str, attribute: str, value: str, generated: Path | None = None) -> int | None:
    stats = load_graph_stats(graph_id, generated)
    if not stats:
        return None
    bucket = (stats.get("value_counts") or {}).get(attribute) or {}
    from knowledge_graph.ontology import normalize_name

    want = normalize_name(value)
    for key, count in bucket.items():
        if normalize_name(str(key)) == want:
            return int(count)
    return None


def save_graph_stats(graph_id: str, payload: dict[str, Any], generated: Path | None = None) -> Path:
    root = stats_root(generated)
    root.mkdir(parents=True, exist_ok=True)
    path = stats_path(graph_id, generated)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _update_index(graph_id, payload, generated)
    return path


def _update_index(graph_id: str, payload: dict[str, Any], generated: Path | None) -> None:
    path = index_path(generated)
    index: dict[str, Any] = {"version": 1, "graphs": {}}
    if path.is_file():
        try:
            index = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    graphs = dict(index.get("graphs") or {})
    totals = payload.get("totals") or {}
    graphs[graph_id] = {
        "products": totals.get("products"),
        "brands": totals.get("brands"),
        "buyers": totals.get("buyers"),
        "updated_at": payload.get("updated_at"),
        "path": stats_path(graph_id, generated).name,
    }
    index["graphs"] = graphs
    index["updated_at"] = payload.get("updated_at")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_ecom_stats_from_postgres(dsn: str, *, generated: Path | None = None) -> dict[str, Any]:
    """Aggregate master_products column value counts — source of truth for ecom_products."""

    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RuntimeError("psycopg is required to build ecom stats") from exc

    value_counts: dict[str, dict[str, int]] = {}
    unique_counts: dict[str, int] = {}
    with psycopg.connect(dsn) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT count(*)::int AS n FROM master_products")
            products = int(cur.fetchone()["n"])
            # Discover which count columns actually exist.
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'master_products'
                """
            )
            existing = {str(row["column_name"]) for row in cur.fetchall()}
            columns = [col for col in _ECOM_COUNT_COLUMNS if col in existing]
            for column in columns:
                cur.execute(
                    f"""
                    SELECT COALESCE(NULLIF(TRIM({column}::text), ''), '(empty)') AS value,
                           count(*)::int AS n
                    FROM master_products
                    GROUP BY 1
                    ORDER BY n DESC, value ASC
                    """
                )
                bucket = {str(row["value"]): int(row["n"]) for row in cur.fetchall()}
                value_counts[column] = bucket
                unique_counts[column] = len(bucket)

    brands = unique_counts.get("brand") or len(value_counts.get("brand") or {})
    payload = {
        "version": 1,
        "graph_id": "ecom_products",
        "source": "postgres.master_products",
        "updated_at": _utc_now(),
        "totals": {
            "products": products,
            "brands": brands,
            "buyers": 0,
            "columns_indexed": len(value_counts),
        },
        "unique_counts": unique_counts,
        "value_counts": value_counts,
    }
    save_graph_stats("ecom_products", payload, generated)
    LOGGER.info(
        "wrote ecom kg stats products=%s brands=%s columns=%s",
        products,
        brands,
        len(value_counts),
    )
    return payload


def build_product_graph_stats_from_neo4j(
    platform,
    graph_id: str,
    *,
    attributes: list[str] | None = None,
    generated: Path | None = None,
) -> dict[str, Any]:
    """Fallback Neo4j aggregation for non-SQL product graphs."""

    settings: Settings = platform.settings
    out_dir = generated or settings.generated
    attrs = list(attributes or ("brand", "buyer", "category", "material", "color", "season", "gender"))
    products = 0
    value_counts: dict[str, dict[str, int]] = {}
    unique_counts: dict[str, int] = {}

    session_factory = getattr(platform.graph, "_session", None)
    if callable(session_factory):
        with session_factory() as session:
            row = session.run(
                "MATCH (n:Entity:Product {graph: $g}) "
                "WHERE coalesce(n.lifecycle,'active') <> 'deleted' "
                "RETURN count(n) AS c",
                {"g": graph_id},
            ).single()
            products = int(row["c"]) if row else 0
            for attr in attrs:
                rows = list(
                    session.run(
                        f"MATCH (n:Entity:Product {{graph: $g}}) "
                        f"WHERE coalesce(n.lifecycle,'active') <> 'deleted' "
                        f"WITH coalesce(toString(n.{attr}), '(empty)') AS value "
                        f"RETURN value, count(*) AS n ORDER BY n DESC, value ASC",
                        {"g": graph_id},
                    )
                )
                # Property may not exist — skip empty-only noise when products>0 and all empty.
                bucket = {str(r["value"]): int(r["n"]) for r in rows}
                if not bucket or (len(bucket) == 1 and "(empty)" in bucket and products > 0):
                    # Try related Brand nodes for brand attr.
                    if attr == "brand":
                        brand_rows = list(
                            session.run(
                                "MATCH (p:Entity:Product {graph: $g})-[:HAS_BRAND]->(b:Entity) "
                                "WHERE coalesce(p.lifecycle,'active') <> 'deleted' "
                                "RETURN coalesce(b.name, '(empty)') AS value, count(*) AS n "
                                "ORDER BY n DESC, value ASC",
                                {"g": graph_id},
                            )
                        )
                        bucket = {str(r["value"]): int(r["n"]) for r in brand_rows}
                if bucket:
                    value_counts[attr] = bucket
                    unique_counts[attr] = len(bucket)
    else:
        nodes = platform.graph.find_nodes(graph=graph_id, label="Product", include_deleted=False)
        products = len(nodes)
        for attr in attrs:
            bucket: dict[str, int] = {}
            for node in nodes:
                raw = node.properties.get(attr)
                key = str(raw).strip() if raw not in (None, "") else "(empty)"
                bucket[key] = bucket.get(key, 0) + 1
            if bucket:
                value_counts[attr] = dict(sorted(bucket.items(), key=lambda item: (-item[1], item[0])))
                unique_counts[attr] = len(bucket)

    payload = {
        "version": 1,
        "graph_id": graph_id,
        "source": "neo4j",
        "updated_at": _utc_now(),
        "totals": {
            "products": products,
            "brands": unique_counts.get("brand", 0),
            "buyers": unique_counts.get("buyer", 0),
            "columns_indexed": len(value_counts),
        },
        "unique_counts": unique_counts,
        "value_counts": value_counts,
    }
    save_graph_stats(graph_id, payload, out_dir)
    LOGGER.info("wrote neo4j kg stats graph=%s products=%s", graph_id, products)
    return payload


def refresh_ecom_stats(platform) -> dict[str, Any]:
    """Build ecom_products stats from Postgres when DSN is set, else Neo4j."""

    settings: Settings = platform.settings
    dsn = settings.resolve_ecom_dsn()
    if dsn:
        try:
            return build_ecom_stats_from_postgres(dsn, generated=settings.generated)
        except Exception as exc:
            LOGGER.warning("postgres ecom stats failed, falling back to neo4j: %s", exc)
    return build_product_graph_stats_from_neo4j(platform, "ecom_products", generated=settings.generated)
