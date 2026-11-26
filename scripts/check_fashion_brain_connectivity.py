"""Run Fashion Brain Neo4j connectivity health checks (read-only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHECKS = [
    (
        "product_counts",
        """
        MATCH (p:Product)
        WHERE coalesce(p.lifecycle,'active') <> 'deleted'
        RETURN p.graph AS graph, coalesce(p.source, p.prov_source_type) AS source, count(p) AS n
        ORDER BY n DESC
        """,
    ),
    (
        "polluted_materials_should_be_0",
        """
        MATCH (m:Material)
        WHERE m.name =~ '.*\\d+.*'
        RETURN count(m) AS polluted_count, collect(m.name)[0..15] AS samples
        """,
    ),
    (
        "inventory_collisions_should_be_0",
        """
        MATCH (p:Product)-[:HAS_INVENTORY_RECORD]->(inv:InventoryRecord)
        WITH inv, count(p) AS prod_count
        WHERE prod_count > 1
        RETURN count(inv) AS collided_inventory_nodes
        """,
    ),
    (
        "shared_hubs_between_plmex_and_ecom",
        """
        MATCH (plm:Product)-->(hub)<--(ecom:Product)
        WHERE (plm.source STARTS WITH 'plmex' OR plm.graph = 'plmex_products')
          AND (ecom.source = 'ecom' OR ecom.graph = 'ecom_products')
          AND plm.id <> ecom.id
        RETURN labels(hub)[0] AS hub_label,
               coalesce([x IN labels(hub) WHERE NOT x IN ['Entity','SourceEntity','Record','Canonical']][0], labels(hub)[0]) AS shared_hub_type,
               count(DISTINCT hub) AS shared_nodes
        ORDER BY shared_nodes DESC
        """,
    ),
    (
        "shared_brand_detail",
        """
        MATCH (plm:Product)-[:HAS_BRAND]->(b:Brand)<-[:HAS_BRAND]-(ecom:Product)
        WHERE (plm.source STARTS WITH 'plmex' OR plm.graph = 'plmex_products')
          AND (ecom.source = 'ecom' OR ecom.graph = 'ecom_products')
        RETURN b.id AS brand_id, b.name AS brand,
               count(DISTINCT plm) AS plm_products,
               count(DISTINCT ecom) AS ecom_products
        ORDER BY ecom_products + plm_products DESC
        LIMIT 15
        """,
    ),
    (
        "shared_material_detail",
        """
        MATCH (plm:Product)-[:USES_MATERIAL|PRIMARY_MATERIAL]->(m:Material)<-[:USES_MATERIAL|PRIMARY_MATERIAL]-(ecom:Product)
        WHERE (plm.source STARTS WITH 'plmex' OR plm.graph = 'plmex_products')
          AND (ecom.source = 'ecom' OR ecom.graph = 'ecom_products')
        RETURN m.id AS material_id, m.name AS material,
               count(DISTINCT plm) AS plm_products,
               count(DISTINCT ecom) AS ecom_products
        ORDER BY ecom_products + plm_products DESC
        LIMIT 15
        """,
    ),
    (
        "shared_reference_and_resolves_to",
        """
        MATCH (plm:Product)-[:HAS_REFERENCE]->(r:Reference)<-[:HAS_REFERENCE]-(ecom:Product)
        WHERE (plm.source STARTS WITH 'plmex' OR plm.graph = 'plmex_products')
          AND (ecom.source = 'ecom' OR ecom.graph = 'ecom_products')
          AND plm.id <> ecom.id
        OPTIONAL MATCH (plm)-[rel:RESOLVES_TO]->(ecom)
        RETURN count(DISTINCT r) AS shared_refs,
               count(DISTINCT plm) AS plm_with_shared_ref,
               count(DISTINCT ecom) AS ecom_with_shared_ref,
               count(rel) AS resolves_to_edges
        """,
    ),
    (
        "targetgroup_bridge",
        """
        MATCH (plm:Product)-[:FOR_TARGET_GROUP]->(tg:TargetGroup)<-[:FOR_TARGET_GROUP]-(ecom:Product)
        WHERE (plm.source STARTS WITH 'plmex' OR plm.graph = 'plmex_products')
          AND (ecom.source = 'ecom' OR ecom.graph = 'ecom_products')
        RETURN tg.id AS targetgroup_id, tg.name AS name,
               count(DISTINCT plm) AS plm_n, count(DISTINCT ecom) AS ecom_n
        ORDER BY plm_n + ecom_n DESC
        LIMIT 10
        """,
    ),
    (
        "fashion_brain_hub_inventory",
        """
        MATCH (n:Entity {graph:'fashion_brain'})
        WHERE coalesce(n.lifecycle,'active') <> 'deleted'
        WITH [x IN labels(n) WHERE NOT x IN ['Entity','SourceEntity','Record','Canonical']][0] AS kind, count(*) AS n
        RETURN kind, n
        ORDER BY n DESC
        """,
    ),
    (
        "orphan_products_no_brand_no_material",
        """
        MATCH (p:Product)
        WHERE coalesce(p.lifecycle,'active') <> 'deleted'
          AND NOT (p)-[:HAS_BRAND]->()
          AND NOT (p)-[:USES_MATERIAL|PRIMARY_MATERIAL]->()
        RETURN coalesce(p.graph, p.source) AS graph, count(p) AS orphan_count
        ORDER BY orphan_count DESC
        """,
    ),
    (
        "plmex_method_vs_old_category_parent",
        """
        MATCH (p:Product {graph:'plmex_products'})
        OPTIONAL MATCH (p)-[:MADE_WITH_METHOD]->(m)
        OPTIONAL MATCH (p)-[:BELONGS_TO_CATEGORY]->(old)
        RETURN count(DISTINCT p) AS products,
               count(DISTINCT m) AS with_production_method,
               count(DISTINCT old) AS still_old_belongs_to_category
        """,
    ),
    (
        "retailgroup_links",
        """
        MATCH (g:RetailGroup)
        OPTIONAL MATCH (b)-[:IN_GROUP]->(g)
        RETURN g.name AS retail_group, count(DISTINCT b) AS members
        ORDER BY members DESC
        """,
    ),
]


def main() -> int:
    from knowledge_graph.config import load_settings
    from knowledge_graph.pipeline.service import Platform

    platform = Platform(load_settings())
    out: dict = {"counts": platform.graph.counts(), "checks": {}}
    try:
        with platform.graph._session() as session:
            for name, cypher in CHECKS:
                try:
                    rows = [dict(r) for r in session.run(cypher)]
                    out["checks"][name] = rows
                    print(f"\n=== {name} ===")
                    print(json.dumps(rows, indent=2, default=str)[:3000])
                except Exception as exc:
                    out["checks"][name] = {"error": str(exc)}
                    print(f"\n=== {name} ERROR ===\n{exc}")

        # Verdict
        polluted = (out["checks"].get("polluted_materials_should_be_0") or [{}])[0]
        collided = (out["checks"].get("inventory_collisions_should_be_0") or [{}])[0]
        hubs = out["checks"].get("shared_hubs_between_plmex_and_ecom") or []
        refs = (out["checks"].get("shared_reference_and_resolves_to") or [{}])[0]
        verdict = {
            "polluted_materials_ok": int(polluted.get("polluted_count") or 0) == 0,
            "inventory_ok": int(collided.get("collided_inventory_nodes") or 0) == 0,
            "has_shared_hubs": bool(hubs) and any(int(r.get("shared_nodes") or 0) > 0 for r in hubs if isinstance(r, dict)),
            "has_shared_refs": int(refs.get("shared_refs") or 0) > 0 if isinstance(refs, dict) else False,
            "fully_connected": False,
        }
        verdict["fully_connected"] = all(
            [
                verdict["polluted_materials_ok"],
                verdict["inventory_ok"],
                verdict["has_shared_hubs"],
            ]
        )
        out["verdict"] = verdict
        print("\n=== VERDICT ===")
        print(json.dumps(verdict, indent=2))
        path = Path(platform.settings.generated) / "fashion_brain_connectivity_check.json"
        path.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
        print("wrote", path)
        return 0 if verdict["fully_connected"] else 2
    finally:
        closer = getattr(platform.graph, "close", None)
        if callable(closer):
            closer()


if __name__ == "__main__":
    raise SystemExit(main())
