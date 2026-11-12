"""Named database connection configs for UI connection manager.

Passwords are never returned. Prefer ``password_env`` (reads from process env /
settings) over storing secrets in JSON.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from pydantic import BaseModel

from knowledge_graph.config import Settings
from knowledge_graph.errors import NotFoundError, ValidationFailed

_SAFE_ID = re.compile(r"^[a-z0-9]+(?:[_-][a-z0-9]+)*$")


class DbConnectionBody(BaseModel):
    id: str
    host: str = "127.0.0.1"
    port: int = 5432
    database: str
    user: str = "postgres"
    # Prefer env var name; optional one-shot password for local test only (not echoed back).
    password_env: str | None = "POSTGRES_PASSWORD"
    password: str | None = None
    table_default: str = "master_products"
    schema_default: str = "public"
    driver: str = "postgresql"
    description: str = ""


def connections_path(settings: Settings) -> Path:
    return settings.generated / "db_connections.json"


def list_connections(settings: Settings) -> list[dict[str, Any]]:
    return [_public(row) for row in _load(settings)]


def get_connection(settings: Settings, connection_id: str) -> dict[str, Any]:
    for row in _load(settings):
        if row.get("id") == connection_id:
            return _public(row)
    raise NotFoundError(f"Database connection '{connection_id}' was not found.")


def upsert_connection(settings: Settings, body: DbConnectionBody) -> dict[str, Any]:
    cid = _require_id(body.id)
    rows = _load(settings)
    record = {
        "id": cid,
        "host": body.host.strip() or "127.0.0.1",
        "port": int(body.port or 5432),
        "database": body.database.strip(),
        "user": body.user.strip() or "postgres",
        "password_env": (body.password_env or "").strip() or None,
        "table_default": body.table_default or "master_products",
        "schema_default": body.schema_default or "public",
        "driver": body.driver or "postgresql",
        "description": body.description or "",
    }
    if not record["database"]:
        raise ValidationFailed("database is required.")
    # Optional local-dev password stored only if explicitly sent (not returned by GET).
    if body.password:
        record["password"] = body.password
    rows = [row for row in rows if row.get("id") != cid]
    rows.append(record)
    _save(settings, rows)
    return _public(record)


def resolve_connection_dsn(settings: Settings, connection_id: str) -> str:
    """Build a Postgres DSN for a named connection (passwords never logged)."""

    return _dsn_for(settings, _raw(settings, connection_id))


def test_connection(
    settings: Settings,
    connection_id: str,
    *,
    table: str | None = None,
    schema: str | None = None,
) -> dict[str, Any]:
    record = _raw(settings, connection_id)
    dsn = _dsn_for(settings, record)
    table_name = table or record.get("table_default") or "master_products"
    schema_name = schema or record.get("schema_default") or "public"
    # Temporarily override resolve path by inspecting with an explicit DSN helper.
    return _inspect_dsn(dsn, table=table_name, schema=schema_name, connection_id=connection_id)


def _inspect_dsn(dsn: str, *, table: str, schema: str, connection_id: str) -> dict[str, Any]:
    from knowledge_graph.sources import db_inspect as mod

    # Reuse inspect logic with a tiny Settings shim via monkeypatch-free path:
    # call internal pieces by temporarily using require-style connect.
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise ValidationFailed("psycopg is not installed. Run: pip install 'psycopg[binary]'") from exc

    # Delegate column/table discovery by constructing a minimal fake settings object
    # is heavy — instead duplicate the inspect call pattern with the DSN directly.
    from knowledge_graph.sources.db_inspect import DEFAULT_SCHEMA, DEFAULT_TABLE, _safe_ident, _short_version

    table_name = (table or DEFAULT_TABLE).strip()
    schema_name = (schema or DEFAULT_SCHEMA).strip()
    if not _safe_ident(table_name) or not _safe_ident(schema_name):
        raise ValidationFailed("table and schema may only contain letters, numbers, and underscores.")

    with psycopg.connect(dsn) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT current_database() AS database, current_user AS db_user, version() AS version")
            live = cur.fetchone() or {}
            cur.execute(
                """
                SELECT EXISTS (
                  SELECT 1 FROM information_schema.tables
                  WHERE table_schema = %(schema)s AND table_name = %(table)s
                ) AS present
                """,
                {"schema": schema_name, "table": table_name},
            )
            exists = bool((cur.fetchone() or {}).get("present"))
            columns: list[dict[str, Any]] = []
            row_count = None
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
                cur.execute(f'SELECT count(*)::int AS n FROM "{schema_name}"."{table_name}"')
                row_count = int((cur.fetchone() or {}).get("n") or 0)
                if any(col["name"] == "brand" for col in columns):
                    cur.execute(
                        f"""
                        SELECT brand, count(*)::int AS n
                        FROM "{schema_name}"."{table_name}"
                        GROUP BY brand ORDER BY brand
                        """
                    )
                    brands = {str(row["brand"]): int(row["n"]) for row in cur.fetchall()}

    return {
        "ok": True,
        "connection_id": connection_id,
        "connection": {
            "database": live.get("database"),
            "user": live.get("db_user"),
            "server_version": _short_version(live.get("version")),
            "dsn_configured": True,
            "driver": "postgresql",
        },
        "table": {
            "schema": schema_name,
            "name": table_name,
            "exists": exists,
            "row_count": row_count,
            "columns": columns,
            "column_count": len(columns),
        },
        "brands": brands,
    }


def _dsn_for(settings: Settings, record: dict[str, Any]) -> str:
    password = ""
    if record.get("password"):
        password = str(record["password"])
    elif record.get("password_env"):
        env_name = str(record["password_env"])
        password = os.environ.get(env_name, "") or settings.postgres_password
        # Misconfigured records sometimes store the password itself in password_env.
        if not password and env_name and not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", env_name):
            password = env_name
        elif not password and re.match(r"^[A-Za-z0-9+/=._-]{8,}$", env_name) and "_" not in env_name:
            # Likely a literal password accidentally stored as password_env.
            password = env_name
    else:
        password = settings.postgres_password
    user = quote_plus(str(record.get("user") or "postgres"))
    host = record.get("host") or "127.0.0.1"
    port = int(record.get("port") or 5432)
    database = record.get("database") or ""
    if password:
        return f"postgresql://{user}:{quote_plus(password)}@{host}:{port}/{database}"
    return f"postgresql://{user}@{host}:{port}/{database}"


def _load(settings: Settings) -> list[dict[str, Any]]:
    path = connections_path(settings)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        rows = data.get("connections") or []
    elif isinstance(data, list):
        rows = data
    else:
        rows = []
    return [row for row in rows if isinstance(row, dict) and row.get("id")]


def _save(settings: Settings, rows: list[dict[str, Any]]) -> None:
    path = connections_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1, "connections": rows}, indent=2) + "\n", encoding="utf-8")


def _raw(settings: Settings, connection_id: str) -> dict[str, Any]:
    for row in _load(settings):
        if row.get("id") == connection_id:
            return row
    raise NotFoundError(f"Database connection '{connection_id}' was not found.")


def _public(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "host": row.get("host"),
        "port": row.get("port"),
        "database": row.get("database"),
        "user": row.get("user"),
        "password_env": row.get("password_env"),
        "password_set": bool(row.get("password") or row.get("password_env")),
        "table_default": row.get("table_default") or "master_products",
        "schema_default": row.get("schema_default") or "public",
        "driver": row.get("driver") or "postgresql",
        "description": row.get("description") or "",
    }


def _require_id(value: str) -> str:
    text = (value or "").strip().casefold()
    if not text or not _SAFE_ID.match(text):
        raise ValidationFailed(f"Invalid connection id {value!r}.")
    return text
