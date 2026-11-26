"""Wipe + full rebuild Fashion Brain v2 (ecom + plmex + enrichment + health).

Usage:
  python scripts/rebuild_fashion_brain_v2.py                 # full wipe + sync
  python scripts/rebuild_fashion_brain_v2.py --limit 200     # smoke rebuild
  python scripts/rebuild_fashion_brain_v2.py --skip-wipe
  python scripts/rebuild_fashion_brain_v2.py --only enrich
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def wipe_graphs(platform, graphs: list[str]) -> dict[str, int]:
    """Delete graph nodes in small auto-committed batches to avoid Neo4j tx memory OOM."""

    deleted: dict[str, int] = {}
    batch = 200
    for gid in graphs:
        total = 0
        print(f"wipe start {gid}", flush=True)
        while True:
            # Fresh session/tx each batch so memory can reclaim.
            with platform.graph._session() as session:
                row = session.run(
                    "MATCH (n:Entity {graph:$g}) WITH n LIMIT $limit "
                    "DETACH DELETE n RETURN count(*) AS c",
                    {"g": gid, "limit": batch},
                ).single()
                c = int(row["c"]) if row else 0
            total += c
            if total % 2000 == 0 or c == 0:
                print(f"wipe {gid}: batch={c} total={total}", flush=True)
            if c == 0:
                break
        deleted[gid] = total
    return deleted


def clear_sync_state(settings) -> list[str]:
    removed = []
    for name in ("ecom_sync_state.json", "plmex_products_sync_state.json"):
        path = Path(settings.generated) / name
        if path.is_file():
            path.unlink()
            removed.append(str(path))
    return removed


def apply_constraints(platform) -> list[str]:
    from knowledge_graph.graph.constraints import SCHEMA_STATEMENTS, apply_schema_statements

    with platform.graph._session() as session:
        return apply_schema_statements(session, SCHEMA_STATEMENTS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit rows per source (smoke / product slice)")
    parser.add_argument("--skip-wipe", action="store_true")
    parser.add_argument(
        "--only",
        choices=["all", "wipe", "ecom", "plmex", "enrich"],
        default="all",
    )
    parser.add_argument("--brand", default=None, help="Optional ecom brand filter")
    parser.add_argument(
        "--plmex-buyers",
        default=None,
        help="Comma-separated buyer prefixes for plmex slice (e.g. PRIMARK,MATALAN,INDITEX,C&A,WALMART)",
    )
    args = parser.parse_args()
    plmex_buyers = [b.strip() for b in str(args.plmex_buyers or "").split(",") if b.strip()]

    from knowledge_graph.config import load_settings
    from knowledge_graph.logging_config import configure_logging
    from knowledge_graph.pipeline.service import Platform

    settings = load_settings()
    configure_logging(settings.log_level, settings.log_json)
    platform = Platform(settings)
    report: dict = {"ok": True, "steps": {}}

    try:
        print("before_counts", platform.graph.counts(), flush=True)

        if args.only in {"all", "wipe"} and not args.skip_wipe:
            print("=== STEP 1: wipe ecom_products + plmex_products + fashion_brain ===", flush=True)
            wiped = wipe_graphs(platform, ["ecom_products", "plmex_products", "fashion_brain"])
            removed = clear_sync_state(settings)
            report["steps"]["wipe"] = {"deleted": wiped, "state_removed": removed}
            print("wipe_done", json.dumps(report["steps"]["wipe"]), flush=True)
            if args.only == "wipe":
                print("after_counts", platform.graph.counts(), flush=True)
                return 0

        if args.only in {"all", "ecom", "plmex"}:
            print("=== STEP 2: apply Neo4j constraints/indexes ===", flush=True)
            applied = apply_constraints(platform)
            report["steps"]["constraints"] = {"applied": len(applied)}
            print(f"constraints_applied={len(applied)}", flush=True)

        if args.only in {"all", "ecom"}:
            print("=== STEP 3: full ecom sync ===", flush=True)
            from knowledge_graph.pipeline.ecom_sync import sync_ecom_to_neo4j

            ecom = sync_ecom_to_neo4j(
                platform,
                brand=args.brand,
                full=True,
                limit=args.limit,
                refresh_generated=True,
            )
            report["steps"]["ecom"] = {
                "ok": ecom.get("ok"),
                "products": ecom.get("products"),
                "node_writes": ecom.get("node_writes") or ecom.get("nodes"),
                "edge_writes": ecom.get("edge_writes") or ecom.get("edges"),
                "errors": (ecom.get("errors") or [])[:10],
            }
            print("ecom_done", json.dumps(report["steps"]["ecom"]), flush=True)
            if not ecom.get("ok"):
                report["ok"] = False

        if args.only in {"all", "plmex"}:
            print("=== STEP 4: full plmex sync ===", flush=True)
            from knowledge_graph.pipeline.mapping_sql_sync import sync_mapped_sql_to_neo4j

            plmex = sync_mapped_sql_to_neo4j(
                platform,
                "plmex_products",
                full=True,
                limit=args.limit,
                refresh_generated=True,
                buyer_prefixes=plmex_buyers or None,
                newest_first=True,
            )
            report["steps"]["plmex"] = {
                "ok": plmex.get("ok"),
                "products": plmex.get("products"),
                "node_writes": plmex.get("node_writes"),
                "edge_writes": plmex.get("edge_writes"),
                "errors": (plmex.get("errors") or [])[:10],
                "enrichment": plmex.get("fashion_brain_enrichment"),
            }
            print("plmex_done", json.dumps(report["steps"]["plmex"], default=str), flush=True)
            if not plmex.get("ok"):
                report["ok"] = False

        if args.only in {"all", "enrich"}:
            print("=== STEP 5: Fashion Brain enrichment + health ===", flush=True)
            from knowledge_graph.pipeline.fashion_brain_enrichment import (
                export_cypher_file,
                run_enrichment,
                run_health_checks,
            )

            export_cypher_file()
            enrich = run_enrichment(platform)
            health = run_health_checks(platform)
            report["steps"]["enrichment"] = enrich
            report["steps"]["health"] = health
            print("enrich_done", json.dumps({"enrich": enrich, "health": health}, default=str), flush=True)
            if not enrich.get("ok") or not health.get("ok"):
                report["ok"] = False

        print("=== FINAL COUNTS ===", flush=True)
        print(platform.graph.counts(), flush=True)

        # Quick shared-hub sample
        with platform.graph._session() as session:
            samples = {
                "materials_with_digits": session.run(
                    "MATCH (m:Material) WHERE m.name =~ '.*\\\\d+.*' RETURN m.name AS n LIMIT 10"
                ).data(),
                "shared_brand_sample": session.run(
                    "MATCH (plm:Product)-[:HAS_BRAND]->(b:Brand)<-[:HAS_BRAND]-(ecom:Product) "
                    "WHERE (plm.source STARTS WITH 'plmex' OR plm.graph='plmex_products') "
                    "AND (ecom.source='ecom' OR ecom.graph='ecom_products') "
                    "RETURN b.id AS brand_id, b.name AS brand, count(DISTINCT plm) AS plm_n, "
                    "count(DISTINCT ecom) AS ecom_n LIMIT 10"
                ).data(),
                "shared_ref_sample": session.run(
                    "MATCH (plm:Product)-[:HAS_REFERENCE]->(r:Reference)<-[:HAS_REFERENCE]-(ecom:Product) "
                    "WHERE plm.id <> ecom.id "
                    "RETURN r.id AS ref, count(*) AS n LIMIT 10"
                ).data(),
                "hub_graph_counts": session.run(
                    "MATCH (n:Entity {graph:'fashion_brain'}) "
                    "RETURN labels(n) AS labels, count(*) AS n ORDER BY n DESC LIMIT 20"
                ).data(),
            }
            report["steps"]["samples"] = samples
            print("samples", json.dumps(samples, default=str)[:4000], flush=True)

        out = Path(settings.generated) / "fashion_brain_v2_rebuild_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        print("report_path", out, flush=True)
        return 0 if report.get("ok") else 1
    finally:
        closer = getattr(platform.graph, "close", None)
        if callable(closer):
            closer()


if __name__ == "__main__":
    raise SystemExit(main())
