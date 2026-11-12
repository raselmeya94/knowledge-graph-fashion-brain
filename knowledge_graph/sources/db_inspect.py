"""Inspect the configured e-comm Postgres connection and table metadata.

Does not return passwords or full DSNs. Uses FASHIONGPT_ECOM_DSN / POSTGRES_*.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from knowledge_graph.config import Settings
from knowledge_graph.errors import ValidationFailed
from knowledge_graph.pipeline.ecom_sync import require_ecom_dsn


DEFAULT_TABLE = "master_products"
DEFAULT_SCHEMA = "public"


def inspect_ecom_connection(
    settings: Settings,
    *,
    table: str = DEFAULT_TABLE,
    schema: str = DEFAULT_SCHEMA,
) -> dict[str, Any]:
    """Ping Postgres and return database / table / column metadata."""

    dsn = require_ecom_dsn(settings)
    table_name = (table or DEFAULT_TABLE).strip()
    schema_name = (schema or DEFAULT_SCHEMA).strip()
    if not table_name or not schema_name:
        raise ValidationFailed("table and schema must be non-empty.")
    if not _safe_ident(table_name) or not _safe_ident(schema_name):
        raise ValidationFailed("table and schema may only contain letters, numbers, and underscores.")

    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise ValidationFailed("psycopg is not installed. Run: pip install 'psycopg[binary]'") from exc

    connection_meta = _connection_meta(settings, dsn)
    with psycopg.connect(dsn) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT current_database() AS database, current_user AS db_user, version() AS version")
            live = cur.fetchone() or {}
            connection_meta["database"] = live.get("database") or connection_meta.get("database")
            connection_meta["user"] = live.get("db_user") or connection_meta.get("user")
            connection_meta["server_version"] = _short_version(live.get("version"))

            cur.execute(
                """
                SELECT EXISTS (
                  SELECT 1
                  FROM information_schema.tables
                  WHERE table_schema = %(schema)s AND table_name = %(table)s
                ) AS present
                """,
                {"schema": schema_name, "table": table_name},
            )
            exists = bool((cur.fetchone() or {}).get("present"))

            columns: list[dict[str, Any]] = []
            row_count: int | None = None
            brands: dict[str, int] = {}
            if exists:
                cur.execute(
                    """
                    SELECT column_name, data_type, is_nullable, ordinal_position
                    FROM information_schema.columns
                    WHERE table_schema = %(schema)s AND table_name = %(table)s
                    ORDER BY ordinal_position ASC
                    """,
                    {"schema": schema_name, "table": table_name},
                )
                columns = [
                    {
                        "name": row["column_name"],
                        "data_type": row["data_type"],
                        "nullable": str(row["is_nullable"]).upper() == "YES",
                        "position": int(row["ordinal_position"]),
                    }
                    for row in cur.fetchall()
                ]
                # Identifiers already validated; quote as "schema"."table".
                cur.execute(f'SELECT count(*)::int AS n FROM "{schema_name}"."{table_name}"')
                row_count = int((cur.fetchone() or {}).get("n") or 0)
                column_names = {item["name"] for item in columns}
                if "brand" in column_names:
                    cur.execute(
                        f"""
                        SELECT brand, count(*)::int AS n
                        FROM "{schema_name}"."{table_name}"
                        GROUP BY brand
                        ORDER BY brand
                        """
                    )
                    brands = {str(row["brand"]): int(row["n"]) for row in cur.fetchall()}

            cur.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = %(schema)s AND table_type = 'BASE TABLE'
                ORDER BY table_name
                """,
                {"schema": schema_name},
            )
            schema_tables = [str(row["table_name"]) for row in cur.fetchall()]

    return {
        "ok": True,
        "connection": connection_meta,
        "table": {
            "schema": schema_name,
            "name": table_name,
            "exists": exists,
            "row_count": row_count,
            "columns": columns,
            "column_count": len(columns),
        },
        "brands": brands,
        "schema_tables": schema_tables,
        "note": "Uses configured FASHIONGPT_ECOM_DSN / POSTGRES_* from .env. Password is never returned.",
    }


def _connection_meta(settings: Settings, dsn: str) -> dict[str, Any]:
    parsed = urlparse(dsn)
    host = parsed.hostname or settings.postgres_host or "127.0.0.1"
    port = parsed.port or settings.postgres_port or 5432
    database = (parsed.path or "").lstrip("/") or settings.postgres_db or ""
    user = parsed.username or settings.postgres_user or ""
    return {
        "host": host,
        "port": int(port),
        "database": database,
        "user": user,
        "dsn_configured": True,
        "driver": "postgresql",
    }


def _safe_ident(value: str) -> bool:
    return bool(value) and value.replace("_", "").isalnum() and value[0].isalpha()


def _short_version(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.split("\n", 1)[0][:160]
