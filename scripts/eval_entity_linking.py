#!/usr/bin/env python3
"""Evaluate extract → link → Text-to-Cypher entity grounding.

Logs extracted entities, linked ids, generated Cypher, row count, pass/fail.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GRAPH_ID: str | None = None  # resolved from registry at runtime


def _resolve_graph_id(platform, override: str | None = None) -> str:
    if override:
        return override
    from knowledge_graph.retrieval.runtime_schema import default_product_graph_id

    gid = default_product_graph_id(platform)
    if not gid:
        raise SystemExit("No active product graph in registry; pass --graph-id")
    return gid


@dataclass
class Case:
    question: str
    check: Callable[[dict[str, Any]], tuple[bool, str]]
    notes: str = ""


def _has_category_ids(payload: dict, *needles: str) -> bool:
    resolved = payload.get("resolved") or {}
    cat = resolved.get("category") or {}
    ids = " ".join(cat.get("ids") or []).casefold()
    names = " ".join(cat.get("normalized_names") or []).casefold()
    blob = ids + " " + names
    return any(n.casefold() in blob for n in needles)


def _status(payload: dict, field: str) -> str:
    resolved = payload.get("resolved") or {}
    return str((resolved.get(field) or {}).get("status") or "")


CASES: list[Case] = [
    Case(
        "jacket",
        lambda p: (_has_category_ids(p, "jackets") and _status(p, "category") != "no_match", "category→jackets"),
    ),
    Case(
        "Jackets",
        lambda p: (_has_category_ids(p, "jackets"), "category→jackets"),
    ),
    Case(
        "JACKETS",
        lambda p: (_has_category_ids(p, "jackets"), "category→jackets"),
    ),
    Case(
        "jackets",
        lambda p: (_has_category_ids(p, "jackets"), "category→jackets"),
    ),
    Case(
        "leather jackets from zara",
        lambda p: (
            _status(p, "brand") in {"matched", "ambiguous"}
            and _status(p, "material") in {"matched", "ambiguous"}
            and _status(p, "category") in {"matched", "ambiguous"},
            "brand+material+category linked",
        ),
    ),
    Case(
        "puffer coat",
        lambda p: (
            _status(p, "category") in {"matched", "ambiguous", "no_match"},
            "nearest category or clear no_match",
        ),
        notes="prefer puffer jackets via alias",
    ),
    Case(
        "jaket",
        lambda p: (_has_category_ids(p, "jackets") or _status(p, "category") == "matched", "misspelling→jackets"),
    ),
    Case(
        "xyzzy-not-a-real-garment-qqq",
        lambda p: (
            _status(p, "category") == "no_match"
            or p.get("clarified") is True
            or (
                bool((p.get("extracted") or {}).get("category"))
                and _status(p, "category") == "no_match"
            )
            or (
                "xyzzy" in str((p.get("extracted") or {}).get("category") or "").casefold()
                and _status(p, "category") in {"no_match", ""}
            ),
            "nonsense→no_match",
        ),
        notes="should clarify, not empty success",
    ),
    Case(
        "InStock black cotton shirts under 50 EUR",
        lambda p: (
            _status(p, "availability") in {"matched", "ambiguous"}
            and _status(p, "color") in {"matched", "ambiguous"}
            and _status(p, "material") in {"matched", "ambiguous"}
            and _status(p, "category") in {"matched", "ambiguous"},
            "availability+color+material+category",
        ),
    ),
]


def run_case(platform, case: Case, *, run_cypher: bool, graph_id: str) -> dict[str, Any]:
    from fashion_brain.agents.contracts import Intention
    from fashion_brain.agents.entity_extract import EntityExtractAgent
    from fashion_brain.agents.text2cypher import Text2CypherAgent
    from fashion_brain.planning.runtime_schema import get_runtime_schema
    from knowledge_graph.linking.entity_linker import (
        clarification_message,
        link_entities,
        list_live_brands,
        resolved_payload,
    )
    from knowledge_graph.retrieval.cypher_safe import run_read_cypher

    started = time.perf_counter()
    schema = get_runtime_schema(platform, graph_id)
    brands = list_live_brands(platform, graph_id)
    extracted = EntityExtractAgent().extract(platform, case.question, brands=brands, schema=schema)
    link_results = link_entities(platform, extracted.as_link_dict(), graph_id=graph_id)
    resolved = resolved_payload(link_results)
    clarify = clarification_message(link_results)
    payload: dict[str, Any] = {
        "question": case.question,
        "graph_id": graph_id,
        "extracted": extracted.to_dict(),
        "link_results": [r.to_dict() for r in link_results],
        "resolved": resolved,
        "clarified": bool(clarify),
        "clarification": clarify,
        "cypher": None,
        "params": None,
        "row_count": None,
        "notes": case.notes,
    }

    if clarify and any(r.status == "no_match" for r in link_results):
        # For nonsense terms, clarification without Cypher is success path.
        payload["row_count"] = 0
    elif run_cypher:
        intention = Intention(
            intent="product_count",
            filters={},
            group_by=extracted.group_by,
            limit=extracted.limit,
            analysis=extracted.intent == "breakdown",
            count_entity=extracted.count_entity,
        )
        plan = Text2CypherAgent().generate(
            platform,
            case.question,
            intention,
            graph_id,
            resolved=resolved,
            extracted=extracted.to_dict(),
            brands=brands,
        )
        if plan is not None:
            payload["cypher"] = plan.cypher
            payload["params"] = plan.params
            try:
                rows = run_read_cypher(platform, plan.cypher, plan.params)
                payload["row_count"] = len(rows)
            except Exception as exc:
                payload["cypher_error"] = str(exc)

    ok, note = case.check(payload)
    payload["pass"] = bool(ok)
    payload["check_note"] = note
    payload["wall_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/generated/eval_entity_linking_report.json")
    parser.add_argument("--no-cypher", action="store_true", help="Only test extract+link")
    parser.add_argument("--question", action="append", default=[])
    parser.add_argument(
        "--graph-id",
        default=None,
        help="Product graph id (default: first active product graph from registry)",
    )
    args = parser.parse_args()

    from knowledge_graph.config import load_settings
    from knowledge_graph.pipeline.service import Platform

    platform = Platform(load_settings())
    graph_id = _resolve_graph_id(platform, args.graph_id)
    print(f"Using graph_id={graph_id}")
    cases = CASES
    if args.question:
        needles = [q.casefold() for q in args.question]
        cases = [c for c in CASES if any(n in c.question.casefold() for n in needles)]

    results = []
    for case in cases:
        print(f"=== {case.question}")
        row = run_case(platform, case, run_cypher=not args.no_cypher, graph_id=graph_id)
        results.append(row)
        print(
            json.dumps(
                {
                    "pass": row["pass"],
                    "check_note": row["check_note"],
                    "resolved": row["resolved"],
                    "clarified": row["clarified"],
                    "row_count": row["row_count"],
                    "wall_ms": row["wall_ms"],
                    "cypher": (row.get("cypher") or "")[:180],
                },
                ensure_ascii=False,
            )
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "graph_id": graph_id,
        "passed": sum(1 for r in results if r["pass"]),
        "total": len(results),
        "results": results,
    }
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out} ({summary['passed']}/{summary['total']} passed)")
    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
