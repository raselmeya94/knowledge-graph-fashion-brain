"""Post-sync Fashion Brain enrichment: RetailGroup, MaterialGroup, ColorFamily, RESOLVES_TO.

Run after ecom + plmex sync batches so shared hubs become queryable bridges.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knowledge_graph.config import DATA_ROOT
from knowledge_graph.logging_config import get_logger

LOGGER = get_logger("knowledge_graph.fashion_brain_enrichment")

BRIDGES_PATH = DATA_ROOT / "schemas" / "semantic_bridges.json"

HEALTH_CHECKS = [
    (
        "polluted_materials",
        "MATCH (m:Material) WHERE m.name =~ '.*\\\\d+.*' RETURN count(m) AS n",
        0,
    ),
    (
        "collided_inventory",
        "MATCH (p:Product)-[:HAS_INVENTORY_RECORD]->(inv:InventoryRecord) "
        "WITH inv, count(p) AS prod_count WHERE prod_count > 1 RETURN count(inv) AS n",
        0,
    ),
    (
        "bad_material_percentages",
        "MATCH ()-[r:USES_MATERIAL|PRIMARY_MATERIAL]->(:Material) "
        "WHERE r.percentage IS NOT NULL AND (r.percentage < 0 OR r.percentage > 100) "
        "RETURN count(r) AS n",
        0,
    ),
    (
        "shared_hubs",
        # Prefer graph ids (source values differ across syncs). Soft gate: ok if either side empty.
        "OPTIONAL MATCH (e:Product) WHERE e.graph='ecom_products' OR e.source='ecom' "
        "WITH count(e) AS ecom_n "
        "OPTIONAL MATCH (p:Product) WHERE p.graph='plmex_products' OR p.source STARTS WITH 'plmex' "
        "WITH ecom_n, count(p) AS plm_n "
        "OPTIONAL MATCH (plm:Product)-->(hub)<--(ecom:Product) "
        "WHERE (plm.graph='plmex_products' OR plm.source STARTS WITH 'plmex') "
        "AND (ecom.graph='ecom_products' OR ecom.source='ecom') "
        "WITH ecom_n, plm_n, count(DISTINCT hub) AS shared "
        "RETURN CASE WHEN ecom_n=0 OR plm_n=0 THEN 1 ELSE shared END AS n",
        None,  # value reported; hard fail applied in run_health_checks when both sides loaded
    ),
]


def load_bridges() -> dict[str, Any]:
    if not BRIDGES_PATH.is_file():
        return {}
    return json.loads(BRIDGES_PATH.read_text(encoding="utf-8"))


def enrichment_statements(bridges: dict[str, Any] | None = None) -> list[tuple[str, dict[str, Any]]]:
    data = bridges or load_bridges()
    statements: list[tuple[str, dict[str, Any]]] = []

    retail_groups = data.get("retail_groups") or []
    if retail_groups:
        statements.append(
            (
                """
                UNWIND $rows AS row
                MERGE (g:Entity:RetailGroup {id: 'group:' + toLower(replace(replace(row.group, '&', 'and'), ' ', '-'))})
                SET g.name = row.group, g.normalized_name = toLower(row.group), g.graph = 'fashion_brain', g.lifecycle = 'active'
                WITH g, row
                UNWIND row.buyers AS bName
                OPTIONAL MATCH (by:Buyer)
                WHERE toUpper(coalesce(by.name, '')) = toUpper(bName)
                FOREACH (_ IN CASE WHEN by IS NOT NULL THEN [1] ELSE [] END |
                  MERGE (by)-[:IN_GROUP]->(g)
                )
                WITH g, row
                UNWIND row.brands AS brName
                OPTIONAL MATCH (br:Brand)
                WHERE toLower(coalesce(br.name, '')) = toLower(brName)
                   OR toLower(coalesce(br.normalized_name, '')) = toLower(brName)
                FOREACH (_ IN CASE WHEN br IS NOT NULL THEN [1] ELSE [] END |
                  MERGE (br)-[:IN_GROUP]->(g)
                )
                """,
                {"rows": retail_groups},
            )
        )

    material_groups = data.get("material_groups") or {}
    mat_rows = [
        {"mat": mat, "grp": grp}
        for grp, mats in material_groups.items()
        for mat in mats
    ]
    if mat_rows:
        statements.append(
            (
                """
                UNWIND $rows AS m
                MATCH (mat:Material)
                WHERE toLower(coalesce(mat.name, '')) = toLower(m.mat)
                   OR toLower(coalesce(mat.normalized_name, '')) = toLower(m.mat)
                MERGE (mg:Entity:MaterialGroup {id: 'materialgroup:' + m.grp})
                SET mg.name = m.grp, mg.normalized_name = m.grp, mg.graph = 'fashion_brain', mg.lifecycle = 'active'
                MERGE (mat)-[:IN_MATERIAL_GROUP]->(mg)
                """,
                {"rows": mat_rows},
            )
        )

    color_families = data.get("color_families") or {}
    color_rows = [
        {"colors": colors, "fam": fam}
        for fam, colors in color_families.items()
    ]
    if color_rows:
        statements.append(
            (
                """
                UNWIND $rows AS cf
                MERGE (f:Entity:ColorFamily {id: 'colorfamily:' + cf.fam})
                SET f.name = cf.fam, f.normalized_name = cf.fam, f.graph = 'fashion_brain', f.lifecycle = 'active'
                WITH f, cf
                UNWIND cf.colors AS cName
                MATCH (col:Color)
                WHERE toLower(coalesce(col.name, '')) = toLower(cName)
                   OR toLower(coalesce(col.normalized_name, '')) = toLower(cName)
                MERGE (col)-[:IN_COLOR_FAMILY]->(f)
                """,
                {"rows": color_rows},
            )
        )

    statements.append(
        (
            """
            MATCH (plm:Product)-[:HAS_REFERENCE]->(r:Reference)<-[:HAS_REFERENCE]-(ecom:Product)
            WHERE (plm.source STARTS WITH 'plmex' OR plm.graph = 'plmex_products')
              AND (ecom.source = 'ecom' OR ecom.graph = 'ecom_products')
              AND plm.id <> ecom.id
            MERGE (plm)-[rel:RESOLVES_TO]->(ecom)
            SET rel.via = 'reference', rel.ref = coalesce(r.name, r.id), rel.confidence = 0.95, rel.lifecycle = 'active'
            """,
            {},
        )
    )

    # Heal legacy bad percentages written before clamp (query-safe 0..100).
    statements.append(
        (
            """
            MATCH ()-[r:USES_MATERIAL|PRIMARY_MATERIAL]->(:Material)
            WHERE r.percentage IS NOT NULL AND r.percentage > 100
            SET r.percentage = 100.0, r.percentage_clamped = true
            """,
            {},
        )
    )
    statements.append(
        (
            """
            MATCH ()-[r:USES_MATERIAL|PRIMARY_MATERIAL]->(:Material)
            WHERE r.percentage IS NOT NULL AND r.percentage < 0
            REMOVE r.percentage
            SET r.percentage_dropped = true
            """,
            {},
        )
    )

    statements.append(
        (
            """
            MATCH (t:AiMaterialTag), (m:Material)
            WHERE toLower(coalesce(t.name, '')) CONTAINS toLower(coalesce(m.name, ''))
               OR toLower(coalesce(t.normalized_name, '')) CONTAINS toLower(coalesce(m.normalized_name, ''))
            MERGE (t)-[:REFERS_TO_MATERIAL]->(m)
            """,
            {},
        )
    )

    # Canonical category bridges from semantic_bridges.json
    categories = data.get("canonical_categories") or {}
    cat_rows: list[dict[str, Any]] = []
    for slug, spec in categories.items():
        cat_rows.append(
            {
                "id": f"cat:{slug.replace('_', '-')}",
                "name": slug.replace("_", " "),
                "parent": spec.get("parent"),
                "aliases": list(
                    dict.fromkeys(
                        (spec.get("ecom_aliases") or [])
                        + (spec.get("plmex_sub_sub_aliases") or [])
                        + (spec.get("plmex_garment_type_aliases") or [])
                    )
                ),
            }
        )
    if cat_rows:
        statements.append(
            (
                """
                UNWIND $rows AS row
                MERGE (c:Entity:CanonicalCategory {id: row.id})
                SET c.name = row.name, c.normalized_name = toLower(row.name),
                    c.parent = row.parent, c.graph = 'fashion_brain', c.lifecycle = 'active'
                WITH c, row
                UNWIND row.aliases AS alias
                OPTIONAL MATCH (n)
                WHERE (n:Subfamily OR n:GarmentType OR n:PlmSubSubCategory OR n:PlmSubCategory OR n:GarmentCategory OR n:Family)
                  AND (
                    toLower(coalesce(n.name, '')) = toLower(alias)
                    OR toLower(coalesce(n.normalized_name, '')) = toLower(alias)
                  )
                FOREACH (_ IN CASE WHEN n IS NOT NULL THEN [1] ELSE [] END |
                  MERGE (n)-[:MAPS_TO_CATEGORY]->(c)
                )
                """,
                {"rows": cat_rows},
            )
        )

    return statements


def run_enrichment(platform, *, bridges: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute enrichment Cypher against Neo4j when available."""

    statements = enrichment_statements(bridges)
    graph = platform.graph
    applied = 0
    errors: list[str] = []

    session_factory = getattr(graph, "session", None) or getattr(graph, "_session", None)
    driver = getattr(graph, "driver", None) or getattr(graph, "_driver", None)

    if driver is not None:
        with driver.session() as session:
            for cypher, params in statements:
                try:
                    session.run(cypher, params)
                    applied += 1
                except Exception as exc:
                    errors.append(str(exc))
                    LOGGER.warning("enrichment statement failed: %s", exc)
    elif callable(session_factory):
        with session_factory() as session:
            for cypher, params in statements:
                try:
                    session.run(cypher, params)
                    applied += 1
                except Exception as exc:
                    errors.append(str(exc))
                    LOGGER.warning("enrichment statement failed: %s", exc)
    else:
        return {
            "ok": False,
            "applied": 0,
            "errors": ["Neo4j session unavailable; enrichment requires a live driver"],
            "statements": len(statements),
        }

    health = run_health_checks(platform)
    return {
        "ok": not errors,
        "applied": applied,
        "statements": len(statements),
        "errors": errors,
        "health": health,
    }


def run_health_checks(platform) -> dict[str, Any]:
    driver = getattr(platform.graph, "driver", None) or getattr(platform.graph, "_driver", None)
    if driver is None:
        return {"ok": False, "error": "no neo4j driver"}
    out: dict[str, Any] = {}
    with driver.session() as session:
        for name, cypher, expected in HEALTH_CHECKS:
            try:
                row = session.run(cypher).single()
                value = int(row["n"]) if row and row.get("n") is not None else None
                if name == "shared_hubs":
                    # Soft when one source missing (returns 1); hard-require shared>0 when both loaded.
                    ok = value is not None and value > 0
                else:
                    ok = expected is None or value == expected
                out[name] = {
                    "value": value,
                    "ok": ok,
                    "expected": expected if name != "shared_hubs" else ">0 when both sources loaded",
                }
            except Exception as exc:
                out[name] = {"ok": False, "error": str(exc)}
    out["ok"] = all(item.get("ok") for item in out.values() if isinstance(item, dict))
    return out


def export_cypher_file(path: Path | None = None) -> Path:
    """Write a runnable .cypher file for manual Neo4j Browser use."""

    target = path or (DATA_ROOT / "schemas" / "fashion_brain_enrichment.cypher")
    bridges = load_bridges()
    chunks = [
        "// Fashion Brain post-sync enrichment",
        "// Generated from semantic_bridges.json — prefer run_enrichment(platform) for parameterized execution.",
        "",
    ]
    for cypher, params in enrichment_statements(bridges):
        chunks.append("// params: " + json.dumps(params)[:200])
        chunks.append(cypher.strip() + ";")
        chunks.append("")
    target.write_text("\n".join(chunks) + "\n", encoding="utf-8")
    return target
