"""Product-ready gate for tomorrow publish (scoped ecom + capped plmex).

Runs enrichment (optional), health checks, connectivity, and smoke Cypher.
Exit 0 only when the minimal publish bar passes.

Usage:
  python scripts/product_ready_gate.py
  python scripts/product_ready_gate.py --skip-enrich
  python scripts/product_ready_gate.py --min-shared-hubs 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Scoped product release capabilities (documented for ops / UI copy).
SUPPORTED_V0 = [
    "ecom catalog filter/count by brand, section, color, material, price",
    "plmex sample lookup by code / buyer / season (capped overlapping buyers)",
    "material % filters via USES_MATERIAL.percentage (e.g. viscose >= 30)",
    "shared hubs: Brand, Material, Color, TargetGroup, Reference",
    "RESOLVES_TO when shared Reference exists after enrichment",
]
UNSUPPORTED_V0 = [
    "full 125k+ plmex dump",
    "vector / CLIP visual similarity",
    "SIMILAR_TO attribute matching",
    "OrgUnit / white-space AI",
    "unguarded free Text2Cypher as primary path",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-enrich", action="store_true")
    parser.add_argument("--min-ecom", type=int, default=10000)
    parser.add_argument("--min-plmex", type=int, default=5000)
    parser.add_argument("--min-shared-hubs", type=int, default=5)
    parser.add_argument("--min-resolves", type=int, default=0)
    args = parser.parse_args()

    from knowledge_graph.config import load_settings
    from knowledge_graph.logging_config import configure_logging
    from knowledge_graph.pipeline.fashion_brain_enrichment import run_enrichment, run_health_checks
    from knowledge_graph.pipeline.service import Platform

    settings = load_settings()
    configure_logging(settings.log_level, settings.log_json)
    platform = Platform(settings)
    report: dict = {
        "ok": True,
        "supported_v0": SUPPORTED_V0,
        "unsupported_v0": UNSUPPORTED_V0,
        "steps": {},
    }

    if not args.skip_enrich:
        print("=== enrich ===", flush=True)
        enrich = run_enrichment(platform)
        report["steps"]["enrichment"] = enrich
        print(json.dumps(enrich, default=str)[:800], flush=True)
        if not enrich.get("ok"):
            report["ok"] = False

    print("=== health ===", flush=True)
    health = run_health_checks(platform)
    report["steps"]["health"] = health
    print(json.dumps(health, default=str), flush=True)
    if not health.get("ok"):
        report["ok"] = False

    print("=== counts + smoke ===", flush=True)
    smoke: dict = {}
    with platform.graph._session() as session:
        ecom_n = session.run(
            "MATCH (p:Product) WHERE p.graph='ecom_products' OR p.source='ecom' RETURN count(p) AS n"
        ).single()["n"]
        plmex_n = session.run(
            "MATCH (p:Product) WHERE p.graph='plmex_products' OR p.source STARTS WITH 'plmex' "
            "RETURN count(p) AS n"
        ).single()["n"]
        shared = session.run(
            "MATCH (plm:Product)-->(hub)<--(ecom:Product) "
            "WHERE (plm.graph='plmex_products' OR plm.source STARTS WITH 'plmex') "
            "AND (ecom.graph='ecom_products' OR ecom.source='ecom') "
            "RETURN count(DISTINCT hub) AS n"
        ).single()["n"]
        resolves = session.run("MATCH ()-[r:RESOLVES_TO]->() RETURN count(r) AS n").single()["n"]
        viscose = session.run(
            "MATCH (p:Product)-[r:USES_MATERIAL]->(m:Material) "
            "WHERE toLower(coalesce(m.name,'')) = 'viscose' AND r.percentage >= 30 "
            "RETURN p.graph AS g, count(DISTINCT p) AS n ORDER BY g"
        ).data()
        bad_pct = session.run(
            "MATCH ()-[r:USES_MATERIAL|PRIMARY_MATERIAL]->(:Material) "
            "WHERE r.percentage IS NOT NULL AND (r.percentage < 0 OR r.percentage > 100) "
            "RETURN count(r) AS n"
        ).single()["n"]
        retail = session.run("MATCH (g:RetailGroup) RETURN count(g) AS n").single()["n"]
        canonical = session.run("MATCH (c:CanonicalCategory) RETURN count(c) AS n").single()["n"]

        smoke = {
            "ecom_products": int(ecom_n),
            "plmex_products": int(plmex_n),
            "shared_hubs": int(shared),
            "resolves_to": int(resolves),
            "viscose_ge_30": viscose,
            "bad_material_percentages": int(bad_pct),
            "retail_groups": int(retail),
            "canonical_categories": int(canonical),
        }
        report["steps"]["smoke"] = smoke
        print(json.dumps(smoke, default=str), flush=True)

        gates = {
            "ecom_min": ecom_n >= args.min_ecom,
            "plmex_min": plmex_n >= args.min_plmex,
            "shared_hubs_min": shared >= args.min_shared_hubs,
            "resolves_min": resolves >= args.min_resolves,
            "no_bad_pct": bad_pct == 0,
            "health_ok": bool(health.get("ok")),
        }
        report["gates"] = gates
        if not all(gates.values()):
            report["ok"] = False

    out_path = Path(settings.generated) / "product_ready_gate.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("=== VERDICT ===", flush=True)
    print(json.dumps({"ok": report["ok"], "gates": report.get("gates"), "wrote": str(out_path)}, indent=2))
    print("SUPPORTED_V0:", *SUPPORTED_V0, sep="\n- ")
    print("UNSUPPORTED_V0:", *UNSUPPORTED_V0, sep="\n- ")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
