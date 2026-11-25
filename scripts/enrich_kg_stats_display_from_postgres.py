"""Enrich kg_stats filter shards with display_values from live Postgres (plmex).

Match keys stay normalize_name'd; ``display_values`` stores the real DB surfaces
(e.g. cut sew jersey → Cut & Sew-Jersey). Safe to re-run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
import psycopg
from psycopg.rows import dict_row

from knowledge_graph.retrieval.kg_stats import (
    collapse_value_bucket,
    filter_stats_path,
    graph_stats_dir,
    load_filter_stats,
)

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

GRAPH_ID = "plmex_products"
# Postgres column → kg_stats filter name
COLUMNS = {
    "garment_type": "garment_type",
    "category": "category",
    "plm_main_category": "plm_main_category",
    "brand": "brand",
    "buyer": "buyer",
    "family": "family",
    "department": "department",
    "gender": "gender",
    "color": "color",
}


def main() -> None:
    dsn = (os.environ.get("FASHIONGPT_PLMEX_DSN") or "").strip()
    if not dsn:
        raise SystemExit("FASHIONGPT_PLMEX_DSN missing")
    generated = ROOT / "data" / "generated"
    root = graph_stats_dir(GRAPH_ID, generated)
    if not root.is_dir():
        raise SystemExit(f"missing stats dir {root}")

    with psycopg.connect(dsn) as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'plmex_products'
            """
        )
        existing_cols = {str(r["column_name"]) for r in cur.fetchall()}
        for pg_col, filter_name in COLUMNS.items():
            if pg_col not in existing_cols:
                print(f"skip {pg_col}: column missing")
                continue
            cur.execute(
                f"""
                SELECT "{pg_col}" AS value, COUNT(*)::int AS n
                FROM plmex_products
                WHERE "{pg_col}" IS NOT NULL AND BTRIM("{pg_col}"::text) <> ''
                GROUP BY 1
                """
            )
            bucket = {str(r["value"]): int(r["n"]) for r in cur.fetchall()}
            values, display = collapse_value_bucket(bucket)
            shard = load_filter_stats(GRAPH_ID, filter_name, generated) or {
                "graph_id": GRAPH_ID,
                "filter": filter_name,
            }
            # Prefer existing counts when present (stats snapshot); always refresh display.
            existing = dict(shard.get("values") or {})
            if existing:
                # Keep counts from shard; only overlay display from Postgres.
                _, display_only = collapse_value_bucket(bucket)
                merged_display = dict(display_only)
                # Also derive from any rich keys already in the shard.
                _, from_existing = collapse_value_bucket(existing)
                for key, surface in from_existing.items():
                    prev = merged_display.get(key)
                    if prev is None or (
                        sum(ch in surface for ch in "&/-")
                        > sum(ch in str(prev) for ch in "&/-")
                    ):
                        merged_display[key] = surface
                shard["display_values"] = merged_display
                shard["values"] = existing
            else:
                shard["values"] = values
                shard["display_values"] = display
                shard["unique_count"] = len(
                    [k for k in values if k not in {"", "(empty)", "(unspecified)"}]
                )
            path = filter_stats_path(GRAPH_ID, filter_name, generated)
            path.write_text(json.dumps(shard, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            sample = list(shard.get("display_values", {}).items())[:3]
            print(f"wrote {path.name}: display={len(shard.get('display_values') or {})} sample={sample}")


if __name__ == "__main__":
    main()
