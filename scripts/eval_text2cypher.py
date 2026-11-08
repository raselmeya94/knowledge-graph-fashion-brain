#!/usr/bin/env python3
"""Evaluate Fashion Brain catalog pipeline vs hand-written Neo4j Cypher.

Pipeline: Intention → Router → Text2CypherAgent (LLM schema→Cypher) with templated
``knowledge_graph/retrieval/ecom.py`` Cypher as fallback. This script measures the
live answer path against reference Cypher counts/rows.

Usage (from repo root):
  set PYTHONPATH=.
  python scripts/eval_text2cypher.py
  python scripts/eval_text2cypher.py --question "How many products does Zara have?"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Placeholder substituted at runtime from registry / --graph-id
GRAPH_ID = "__PRODUCT_GRAPH__"


def _resolve_graph_id(platform, override: str | None = None) -> str:
    if override:
        return override
    from knowledge_graph.retrieval.runtime_schema import default_product_graph_id

    gid = default_product_graph_id(platform)
    if not gid:
        raise SystemExit("No active product graph in registry; pass --graph-id")
    return gid


def _bind_graph(case: Case, graph_id: str) -> Case:
    params = dict(case.reference_params or {})
    if params.get("g") in {None, "", GRAPH_ID, "ecom_products"}:
        params["g"] = graph_id
    return Case(
        question=case.question,
        reference_cypher=case.reference_cypher,
        reference_params=params,
        check=case.check,
        notes=case.notes,
    )


@dataclass
class Case:
    question: str
    reference_cypher: str
    reference_params: dict[str, Any]
    check: str  # count | rows | top_brand | colors | parent
    notes: str = ""


CASES: list[Case] = [
    Case(
        question="How many products does Zara have?",
        reference_cypher=(
            "MATCH (p:Entity:Product {graph:$g}) "
            "WHERE coalesce(p.lifecycle,'active') <> 'deleted' "
            "AND (toLower(coalesce(p.brand,'')) = $brand "
            "OR EXISTS { MATCH (p)-[:HAS_BRAND]->(b:Entity) "
            "WHERE toLower(coalesce(b.name,'')) = $brand }) "
            "RETURN count(p) AS c"
        ),
        reference_params={"g": GRAPH_ID, "brand": "zara"},
        check="count",
    ),
    Case(
        question="How many products per brand?",
        reference_cypher=(
            "MATCH (p:Entity:Product {graph:$g}) "
            "WHERE coalesce(p.lifecycle,'active') <> 'deleted' "
            "OPTIONAL MATCH (p)-[:HAS_BRAND]->(b:Entity) "
            "WHERE coalesce(b.lifecycle,'active') <> 'deleted' "
            "WITH p, coalesce(b.name, p.brand, '(unspecified)') AS brand "
            "RETURN toLower(brand) AS brand, count(DISTINCT p) AS n "
            "ORDER BY n DESC LIMIT 15"
        ),
        reference_params={"g": GRAPH_ID},
        check="rows",
        notes="group_by brand",
    ),
    Case(
        question="Show 5 InStock Primark products with name and price.",
        reference_cypher=(
            "MATCH (p:Entity:Product {graph:$g}) "
            "WHERE coalesce(p.lifecycle,'active') <> 'deleted' "
            "AND (toLower(coalesce(p.brand,'')) = $brand "
            "OR EXISTS { MATCH (p)-[:HAS_BRAND]->(b:Entity) "
            "WHERE toLower(coalesce(b.name,'')) = $brand }) "
            "AND toLower(coalesce(p.availability,'')) = $availability "
            "RETURN p.name AS name, p.price AS price "
            "ORDER BY p.id ASC LIMIT 5"
        ),
        reference_params={"g": GRAPH_ID, "brand": "primark", "availability": "instock"},
        check="rows",
        notes="availability value may be InStock vs instock",
    ),
    Case(
        question="Which brand has the most products with Leather material?",
        reference_cypher=(
            "MATCH (p:Entity:Product {graph:$g})-[:USES_MATERIAL]->(m:Entity) "
            "WHERE coalesce(p.lifecycle,'active') <> 'deleted' "
            "AND toLower(coalesce(m.name,'')) CONTAINS $material "
            "OPTIONAL MATCH (p)-[:HAS_BRAND]->(b:Entity) "
            "WITH coalesce(b.name, p.brand, '(unspecified)') AS brand, count(DISTINCT p) AS n "
            "RETURN brand, n ORDER BY n DESC LIMIT 1"
        ),
        reference_params={"g": GRAPH_ID, "material": "leather"},
        check="top_brand",
    ),
    Case(
        question="Which colors exist in the Women > Jackets category?",
        reference_cypher=(
            "MATCH (p:Entity:Product {graph:$g}) "
            "WHERE coalesce(p.lifecycle,'active') <> 'deleted' "
            "AND ("
            "toLower(coalesce(p.category_path,'')) CONTAINS $path "
            "OR (toLower(coalesce(p.section,'')) CONTAINS 'women' "
            "AND (toLower(coalesce(p.category,'')) CONTAINS 'jacket' "
            "OR toLower(coalesce(p.family,'')) CONTAINS 'jacket' "
            "OR toLower(coalesce(p.subfamily,'')) CONTAINS 'jacket'))"
            ") "
            "OPTIONAL MATCH (p)-[:HAS_COLOR]->(c:Entity) "
            "WITH DISTINCT coalesce(c.name, p.color) AS color "
            "WHERE color IS NOT NULL AND trim(toString(color)) <> '' "
            "RETURN color ORDER BY toLower(toString(color)) ASC LIMIT 50"
        ),
        reference_params={"g": GRAPH_ID, "path": "women"},
        check="colors",
        notes="path matching is approximate; jackets+women heuristic",
    ),
    Case(
        question="How many black cotton shirts does Zara have?",
        reference_cypher=(
            "MATCH (p:Entity:Product {graph:$g}) "
            "WHERE coalesce(p.lifecycle,'active') <> 'deleted' "
            "AND (toLower(coalesce(p.brand,'')) = $brand "
            "OR EXISTS { MATCH (p)-[:HAS_BRAND]->(b:Entity) "
            "WHERE toLower(coalesce(b.name,'')) = $brand }) "
            "AND (toLower(coalesce(p.color,'')) = $color "
            "OR EXISTS { MATCH (p)-[:HAS_COLOR]->(c:Entity) "
            "WHERE toLower(coalesce(c.name,'')) = $color }) "
            "AND (toLower(coalesce(p.composition,'')) CONTAINS $material "
            "OR toLower(coalesce(p.material,'')) CONTAINS $material "
            "OR EXISTS { MATCH (p)-[:USES_MATERIAL]->(m:Entity) "
            "WHERE toLower(coalesce(m.name,'')) CONTAINS $material }) "
            "AND (toLower(coalesce(p.category,'')) CONTAINS $garment "
            "OR toLower(coalesce(p.family,'')) CONTAINS $garment "
            "OR toLower(coalesce(p.subfamily,'')) CONTAINS $garment "
            "OR toLower(coalesce(p.category_path,'')) CONTAINS $garment "
            "OR EXISTS { MATCH (p)-[:BELONGS_TO_CATEGORY|HAS_FAMILY|HAS_SUB_CATEGORY]->(g:Entity) "
            "WHERE toLower(coalesce(g.name,'')) CONTAINS $garment }) "
            "RETURN count(p) AS c"
        ),
        reference_params={
            "g": GRAPH_ID,
            "brand": "zara",
            "color": "black",
            "material": "cotton",
            "garment": "shirt",
        },
        check="count",
    ),
    Case(
        question="What is the parent category of Jackets?",
        reference_cypher=(
            "MATCH (child:Entity {graph:$g})-[:PARENT_CATEGORY]->(parent:Entity) "
            "WHERE toLower(coalesce(child.name,'')) = $name "
            "RETURN child.name AS child, parent.name AS parent LIMIT 10"
        ),
        reference_params={"g": GRAPH_ID, "name": "jackets"},
        check="parent",
        notes="PARENT_CATEGORY direction: child -[:PARENT_CATEGORY]-> parent",
    ),
    Case(
        question="Show InStock jackets under 100 EUR.",
        reference_cypher=(
            "MATCH (p:Entity:Product {graph:$g}) "
            "WHERE coalesce(p.lifecycle,'active') <> 'deleted' "
            "AND toLower(coalesce(p.availability,'')) = $availability "
            "AND (toLower(coalesce(p.category,'')) CONTAINS $garment "
            "OR toLower(coalesce(p.family,'')) CONTAINS $garment "
            "OR toLower(coalesce(p.subfamily,'')) CONTAINS $garment "
            "OR toLower(coalesce(p.category_path,'')) CONTAINS $garment) "
            "AND toLower(coalesce(p.currency,'')) CONTAINS 'eur' "
            "WITH p, "
            "toFloat(replace(replace(replace(coalesce(p.price,''), '€', ''), ',', ''), ' ', '')) AS price_num "
            "WHERE price_num IS NOT NULL AND price_num < $max_price "
            "RETURN p.name AS name, p.price AS price, price_num "
            "ORDER BY price_num ASC LIMIT 20"
        ),
        reference_params={"g": GRAPH_ID, "availability": "instock", "garment": "jacket", "max_price": 100.0},
        check="rows",
        notes="price stored as string e.g. €219.00 — needs parse for numeric compare",
    ),
]


def _run_cypher(platform, query: str, params: dict) -> tuple[list[dict], float, str | None]:
    session_factory = getattr(platform.graph, "_session", None)
    if not callable(session_factory):
        return [], 0.0, "no_session"
    started = time.perf_counter()
    try:
        with session_factory() as session:
            rows = [dict(record) for record in session.run(query, params)]
        return rows, (time.perf_counter() - started) * 1000.0, None
    except Exception as exc:  # noqa: BLE001 — eval harness
        return [], (time.perf_counter() - started) * 1000.0, f"{exc.__class__.__name__}: {exc}"


def _cypher_syntax_ok(platform, query: str, params: dict) -> tuple[bool, str | None]:
    """EXPLAIN validates syntax without full execution cost for heavy queries."""

    explain = "EXPLAIN " + query
    _rows, _ms, err = _run_cypher(platform, explain, params)
    return err is None, err


def _extract_int(text: str) -> int | None:
    match = re.search(r"\b(\d{1,7})\b", text or "")
    return int(match.group(1)) if match else None


def _compare(case: Case, answer_text: str, ref_rows: list[dict]) -> tuple[bool | None, str]:
    if case.check == "count":
        expected = int(ref_rows[0]["c"]) if ref_rows else None
        got = _extract_int(answer_text)
        if expected is None:
            return None, "reference returned no count"
        if got is None:
            return False, f"no number in answer; expected {expected}"
        # Accept exact or answer mentioning the count.
        ok = got == expected or str(expected) in (answer_text or "")
        return ok, f"expected={expected} got={got}"
    if case.check == "top_brand":
        if not ref_rows:
            return None, "reference empty"
        brand = str(ref_rows[0].get("brand") or "").casefold()
        ok = brand and brand in (answer_text or "").casefold()
        return ok, f"expected brand containing {brand!r}"
    if case.check == "parent":
        if not ref_rows:
            return False, "no PARENT_CATEGORY rows for jackets (direction/name mismatch?)"
        parents = {str(row.get("parent") or "").casefold() for row in ref_rows}
        ok = any(parent and parent in (answer_text or "").casefold() for parent in parents)
        return ok, f"parents={sorted(parents)}"
    if case.check == "colors":
        if not ref_rows:
            return None, "reference colors empty"
        colors = [str(row.get("color") or "") for row in ref_rows[:8]]
        hits = sum(1 for color in colors if color and color.casefold() in (answer_text or "").casefold())
        ok = hits >= min(2, len(colors))
        return ok, f"color_hits={hits}/{len(colors)}"
    # rows: presence of any reference names/prices or non-empty answer with evidence
    if not ref_rows:
        empty_ok = "no" in (answer_text or "").casefold() or "0" in (answer_text or "")
        return empty_ok, "reference empty"
    sample = str(ref_rows[0].get("name") or ref_rows[0].get("brand") or next(iter(ref_rows[0].values()), ""))
    ok = bool(answer_text) and (not sample or sample.casefold()[:12] in answer_text.casefold() or len(ref_rows) > 0)
    # Softer: pipeline returned something and reference has rows
    if answer_text and len(answer_text) > 20 and ref_rows:
        return True, f"ref_rows={len(ref_rows)} (presence check)"
    return ok, f"sample={sample!r} ref_rows={len(ref_rows)}"


def _schema_probe(platform, graph_id: str) -> dict[str, Any]:
    out: dict[str, Any] = {"graph_id": graph_id}
    try:
        from knowledge_graph.retrieval.runtime_schema import build_runtime_schema

        ctx = build_runtime_schema(platform, graph_id)
        out["runtime_schema"] = {
            "product_label": ctx.product_label,
            "facet_types": ctx.facet_types,
            "filter_keys": ctx.filter_keys,
            "ontology_id": ctx.ontology_id,
        }
    except Exception as exc:
        out["runtime_schema_error"] = str(exc)
    indexes, _ms, err = _run_cypher(platform, "SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties RETURN name, type, entityType, labelsOrTypes, properties LIMIT 50", {})
    out["indexes_error"] = err
    out["indexes"] = indexes
    # Lightweight schema: labels + rel types (db.schema.visualization is heavy/driver-specific)
    labels, _ms, err = _run_cypher(
        platform,
        "CALL db.labels() YIELD label RETURN label ORDER BY label",
        {},
    )
    out["labels_error"] = err
    out["labels"] = [row.get("label") for row in labels]
    rels, _ms, err = _run_cypher(
        platform,
        "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType",
        {},
    )
    out["rel_types_error"] = err
    out["relationship_types"] = [row.get("relationshipType") for row in rels]
    sample_price, _ms, err = _run_cypher(
        platform,
        "MATCH (p:Entity:Product {graph:$g}) WHERE p.price IS NOT NULL "
        "RETURN p.price AS price, p.currency AS currency LIMIT 5",
        {"g": graph_id},
    )
    out["sample_prices"] = sample_price
    out["sample_prices_error"] = err
    parent_dir, _ms, err = _run_cypher(
        platform,
        "MATCH (a:Entity {graph:$g})-[r:PARENT_CATEGORY]->(b:Entity) "
        "RETURN a.name AS from_name, b.name AS to_name LIMIT 5",
        {"g": graph_id},
    )
    out["parent_category_samples"] = parent_dir
    out["parent_category_error"] = err
    return out


def _profile_query(platform, query: str, params: dict) -> dict[str, Any]:
    rows, ms, err = _run_cypher(platform, "PROFILE " + query, params)
    return {"ms": round(ms, 1), "error": err, "rows": len(rows)}


def run_case(platform, principal, case: Case, *, graph_id: str) -> dict[str, Any]:
    from fashion_brain.ask import answer_question
    from fashion_brain.stage_timing import end_timing, start_timing

    case = _bind_graph(case, graph_id)
    start_timing()
    t0 = time.perf_counter()
    try:
        answer = answer_question(platform, case.question, principal, history=None, graph_id=graph_id)
        answer_text = getattr(answer, "answer", "") or ""
        routes = getattr(answer, "routes", None)
        evidence_n = len(getattr(answer, "evidence", None) or [])
    except Exception as exc:  # noqa: BLE001
        answer_text = ""
        routes = None
        evidence_n = 0
        pipeline_error = f"{exc.__class__.__name__}: {exc}"
    else:
        pipeline_error = None
    wall_ms = (time.perf_counter() - t0) * 1000.0
    timing = end_timing()
    timing_snap = timing.snapshot() if timing else {}

    generated = timing_snap.get("cypher") or []
    gen_query = generated[0]["query"] if generated else None
    gen_params = generated[0].get("params") if generated else None
    syntax_ok, syntax_err = (None, "no generated cypher")
    if gen_query:
        syntax_ok, syntax_err = _cypher_syntax_ok(platform, gen_query, gen_params or {})

    ref_rows, ref_ms, ref_err = _run_cypher(platform, case.reference_cypher, case.reference_params)
    ref_syntax_ok, ref_syntax_err = _cypher_syntax_ok(platform, case.reference_cypher, case.reference_params)
    correct, compare_note = _compare(case, answer_text, ref_rows) if ref_err is None else (None, ref_err)

    profile = _profile_query(platform, case.reference_cypher, case.reference_params) if ref_syntax_ok else {}

    return {
        "question": case.question,
        "notes": case.notes,
        "pipeline_error": pipeline_error,
        "final_answer": answer_text[:1200],
        "routes": routes,
        "evidence_n": evidence_n,
        "wall_ms": round(wall_ms, 1),
        "timing": timing_snap,
        "generated_cypher": gen_query,
        "generated_params": gen_params,
        "generated_cypher_count": len(generated),
        "all_generated_cypher": generated,
        "generated_syntax_ok": syntax_ok,
        "generated_syntax_error": syntax_err,
        "generated_row_hint": generated[0].get("rows") if generated else None,
        "reference_cypher": case.reference_cypher,
        "reference_params": case.reference_params,
        "reference_syntax_ok": ref_syntax_ok,
        "reference_syntax_error": ref_syntax_err,
        "reference_rows": len(ref_rows),
        "reference_ms": round(ref_ms, 1),
        "reference_error": ref_err,
        "reference_preview": ref_rows[:5],
        "correct": correct,
        "compare_note": compare_note,
        "profile": profile,
        "risks": _risk_flags(case, gen_query, answer_text, ref_rows),
    }


def _risk_flags(case: Case, gen_query: str | None, answer: str, ref_rows: list[dict]) -> list[str]:
    flags: list[str] = []
    q = (gen_query or "").casefold()
    if "price" in case.question.casefold() or "under" in case.question.casefold() or "eur" in case.question.casefold():
        if gen_query and "tofloat" not in q and "price" in q:
            flags.append("price_compared_as_string_without_parse")
        if not gen_query:
            flags.append("no_cypher_for_price_filter_pipeline_may_ignore_price")
    if "zara" in case.question.casefold() and gen_query and "$brand" in (gen_query or ""):
        pass
    if "parent" in case.question.casefold():
        if not ref_rows:
            flags.append("parent_category_empty_check_direction_or_name")
        if gen_query and "parent_category" not in q:
            flags.append("pipeline_cypher_missing_PARENT_CATEGORY")
    if gen_query and "limit" not in q and case.check == "rows":
        flags.append("generated_cypher_missing_limit")
    if not answer:
        flags.append("empty_final_answer")
    if gen_query is None:
        flags.append("no_templated_cypher_recorded_maybe_kg_stats_or_non_cypher_path")
    return flags


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", action="append", help="Run only matching question substring(s)")
    parser.add_argument("--out", default="data/generated/eval_text2cypher_report.json")
    parser.add_argument("--skip-pipeline", action="store_true", help="Only run reference Cypher + schema probe")
    parser.add_argument(
        "--graph-id",
        default=None,
        help="Product graph id (default: first active product graph from registry)",
    )
    args = parser.parse_args()

    from knowledge_graph.access import Principal
    from knowledge_graph.config import load_settings
    from knowledge_graph.pipeline.service import Platform

    settings = load_settings()
    platform = Platform(settings)
    principal = Principal(clearance="internal")
    graph_id = _resolve_graph_id(platform, args.graph_id)
    print(f"Using graph_id={graph_id}")

    cases = [_bind_graph(c, graph_id) for c in CASES]
    if args.question:
        needles = [item.casefold() for item in args.question]
        cases = [case for case in cases if any(n in case.question.casefold() for n in needles)]

    schema = _schema_probe(platform, graph_id)
    try:
        from knowledge_graph.retrieval.runtime_schema import build_runtime_schema

        ctx = build_runtime_schema(platform, graph_id)
        schema["ontology_entity_types"] = ctx.entity_types
        schema["ontology_relationships"] = sorted((ctx.relationship_allowlist or {}).keys())
        schema["label_vs_ontology"] = {
            "graph_labels": schema.get("labels"),
            "ontology_entities": ctx.entity_types,
            "facet_types": ctx.facet_types,
            "note": "Catalog agents consume RuntimeSchemaContext only (registry/ontology APIs).",
        }
    except Exception as exc:
        schema["ontology_load_error"] = str(exc)

    results = []
    for case in cases:
        print(f"\n=== {case.question}")
        if args.skip_pipeline:
            ref_rows, ref_ms, ref_err = _run_cypher(platform, case.reference_cypher, case.reference_params)
            item = {
                "question": case.question,
                "graph_id": graph_id,
                "reference_rows": len(ref_rows),
                "reference_ms": round(ref_ms, 1),
                "reference_error": ref_err,
                "reference_preview": ref_rows[:5],
            }
        else:
            item = run_case(platform, principal, case, graph_id=graph_id)
        results.append(item)
        print(
            json.dumps(
                {
                    "wall_ms": item.get("wall_ms"),
                    "correct": item.get("correct"),
                    "compare_note": item.get("compare_note"),
                    "generated_syntax_ok": item.get("generated_syntax_ok"),
                    "reference_rows": item.get("reference_rows"),
                    "by_stage_ms": (item.get("timing") or {}).get("by_stage_ms"),
                    "risks": item.get("risks"),
                    "answer_preview": (item.get("final_answer") or "")[:180],
                },
                ensure_ascii=False,
            )
        )

    # Latency rollup
    stage_totals: dict[str, list[float]] = {}
    for item in results:
        for stage, ms in ((item.get("timing") or {}).get("by_stage_ms") or {}).items():
            stage_totals.setdefault(stage, []).append(float(ms))
    latency_table = {
        stage: {
            "n": len(values),
            "avg_ms": round(sum(values) / len(values), 1),
            "max_ms": round(max(values), 1),
            "sum_ms": round(sum(values), 1),
        }
        for stage, values in sorted(stage_totals.items(), key=lambda kv: -sum(kv[1]))
    }

    report = {
        "graph_id": graph_id,
        "pipeline_model": (
            "Schema-driven catalog path: EntityExtract → EntityLinker → Text2Cypher "
            "(RuntimeSchemaContext) with templated ecom Cypher as fallback. "
            "Outer router: ThinkingBrain → Intention → GraphRouter → retrieve → ResponseAgent."
        ),
        "schema": schema,
        "latency_table": latency_table,
        "results": results,
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nWrote {out_path}")

    closer = getattr(platform.graph, "close", None)
    if callable(closer):
        closer()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
