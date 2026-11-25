#!/usr/bin/env python3
"""Score kg_stats metadata questions against expected shard answers.

Default mode is **direct** (no LLM): ``answer_simple_count`` / filter shards only.
Use ``--via-brain`` to run Fashion Brain ``draft_catalog`` and score system answers.

Usage (repo root):
  python scripts/generate_kg_stats_meta_eval.py
  python scripts/eval_kg_stats_meta.py
  python scripts/eval_kg_stats_meta.py --via-brain
  python scripts/eval_kg_stats_meta.py --via-brain --limit 20
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data" / "generated" / "eval"
QUESTIONS = OUT_DIR / "kg_stats_meta_questions.json"
RESULTS = OUT_DIR / "kg_stats_meta_results.json"
RESULTS_BRAIN = OUT_DIR / "kg_stats_meta_results_brain.json"


def _load_questions(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("questions") or [])


def _direct_answer(row: dict[str, Any]) -> dict[str, Any]:
    from knowledge_graph.retrieval.kg_stats import (
        answer_simple_count,
        dimension_bucket,
        product_total,
        unique_count,
    )

    graph_id = str(row.get("graph_id") or "")
    kind = str((row.get("expected") or {}).get("kind") or row.get("type") or "")
    attr = row.get("filter")
    value = row.get("value")
    limit = int(row.get("limit") or (row.get("expected") or {}).get("limit") or 10)

    if kind in {"product_total"}:
        n = product_total(graph_id)
        return {"ok": n is not None, "n": n, "kind": "product_total", "source": "kg_stats"}

    if kind in {"distinct_facet"}:
        n = unique_count(graph_id, str(attr))
        return {"ok": n is not None, "n": n, "kind": "distinct_facet", "attribute": attr, "source": "kg_stats"}

    if kind in {"value_count", "value_count_zero"}:
        meta = answer_simple_count(graph_id, filters={str(attr): str(value)})
        if meta is None:
            return {"ok": False, "n": None, "kind": "value_count", "error": "cache_miss"}
        return {
            "ok": True,
            "n": int(meta["n"]),
            "kind": "value_count",
            "attribute": meta.get("attribute"),
            "value": meta.get("value"),
            "source": "kg_stats",
        }

    if kind in {"facet_top_n"}:
        bucket = dimension_bucket(graph_id, str(attr)) or {}
        empty = {"", "(empty)", "(unspecified)", "n/a", "na", "none", "null", "-", "—"}
        ranked = sorted(
            (
                (str(k), int(v))
                for k, v in bucket.items()
                if str(k).strip().casefold() not in empty and int(v) > 0
            ),
            key=lambda item: (-item[1], item[0].casefold()),
        )[:limit]
        return {
            "ok": True,
            "kind": "facet_top_n",
            "attribute": attr,
            "rows": [{"rank": i + 1, "name": n, "count": c} for i, (n, c) in enumerate(ranked)],
            "source": "kg_stats",
        }

    return {"ok": False, "error": f"unknown kind {kind}"}


def _extract_int(text: str, *, prefer_distinct: bool = False) -> int | None:
    """Prefer counts from kg_stats-style lines / prose."""
    raw = text or ""
    preferred = (
        (
            r"unique\s*=\s*(\d[\d,]*)",
            r"\b(\d[\d,]*)\s+(?:unique|distinct)\b",
            r"\bthere are\s+(\d[\d,]*)\s+(?:unique|distinct)\b",
            r"products\s*=\s*(\d[\d,]*)",
            r"\b(\d[\d,]*)\s+products?\b",
            r"\bthere are\s+(\d[\d,]*)\b",
            r"count\s*[:=]\s*(\d[\d,]*)",
        )
        if prefer_distinct
        else (
            r"products\s*=\s*(\d[\d,]*)",
            r"\b(\d[\d,]*)\s+products?\b",
            r"\bthere are\s+(\d[\d,]*)\s+products?\b",
            r"\bthere are\s+(\d[\d,]*)\b",
            r"count\s*[:=]\s*(\d[\d,]*)",
            r"unique\s*=\s*(\d[\d,]*)",
            r"\b(\d[\d,]*)\s+(?:unique|distinct)\b",
        )
    )
    for pat in preferred:
        m = re.search(pat, raw, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1).replace(",", ""))
            except ValueError:
                pass
    nums = re.findall(r"\d[\d,]*", raw)
    if not nums:
        return None
    try:
        return max(int(n.replace(",", "")) for n in nums)
    except ValueError:
        return None


def _parse_ranked_rows(text: str, limit: int) -> list[dict[str, Any]]:
    """Parse markdown tables or '1. Name — 123' facet rankings from response text."""
    rows: list[dict[str, Any]] = []
    for line in (text or "").splitlines():
        s = line.strip()
        # Markdown table: | 1 | Primark | 1397 |
        m = re.match(
            r"^\|\s*(\d+)\s*\|\s*(.+?)\s*\|\s*(\d[\d,]*)\s*\|?\s*$",
            s,
        )
        if not m:
            m = re.match(
                r"^\s*(?:#)?\s*(\d+)[\.\)\:\-]\s+(.+?)\s*(?:[—\-–:|]\s*|\s+)(\d[\d,]*)\s*$",
                s,
            )
        if not m:
            continue
        try:
            rank = int(m.group(1))
            name = m.group(2).strip(" \t-—–|:")
            count = int(m.group(3).replace(",", ""))
        except ValueError:
            continue
        if name.casefold() in {"name", "brand", "value", "---", ":---", "---:"}:
            continue
        rows.append({"rank": rank, "name": name, "count": count})
        if len(rows) >= limit:
            break
    return rows


def _brain_answer(row: dict[str, Any], platform, principal) -> dict[str, Any]:
    from fashion_brain.ask import draft_catalog

    q = str(row.get("question") or "")
    state: dict[str, Any] = {}
    t0 = time.perf_counter()
    try:
        retrieved, routes, intention = draft_catalog(
            platform,
            q,
            principal,
            history=[],
            conversation_state=state,
        )
        err = None
    except Exception as exc:  # noqa: BLE001 — eval must continue
        ms = (time.perf_counter() - t0) * 1000.0
        return {
            "ok": False,
            "text": "",
            "n": None,
            "rows": [],
            "routes": [],
            "filters": {},
            "ms": round(ms, 1),
            "error": f"{type(exc).__name__}: {exc}",
            "source": "draft_catalog",
        }

    ms = (time.perf_counter() - t0) * 1000.0
    text = str(getattr(retrieved, "text", "") or "")
    qtype = str(row.get("type") or "")
    limit = int(row.get("limit") or (row.get("expected") or {}).get("limit") or 10)
    rows = _parse_ranked_rows(text, limit) if qtype == "facet_top_n" else []
    # Prefer structured kg_stats evidence when present.
    n = _extract_int(text, prefer_distinct=(qtype == "distinct_facet"))
    evidence = list(getattr(retrieved, "evidence", None) or [])
    for item in evidence:
        if getattr(item, "record_id", None) != "kg_stats":
            continue
        excerpt = str(getattr(item, "excerpt", "") or "")
        m = re.search(r"products\s*=\s*(\d[\d,]*)", excerpt, flags=re.IGNORECASE)
        if m:
            try:
                n = int(m.group(1).replace(",", ""))
            except ValueError:
                pass
        m2 = re.search(r"unique\s*=\s*(\d[\d,]*)", excerpt, flags=re.IGNORECASE)
        if m2 and str(row.get("type")) == "distinct_facet":
            try:
                n = int(m2.group(1).replace(",", ""))
            except ValueError:
                pass
    return {
        "ok": bool(text.strip()),
        "text": text,
        "n": n,
        "rows": rows,
        "routes": list(routes or []),
        "filters": dict(getattr(intention, "filters", None) or {}),
        "ms": round(ms, 1),
        "source": "draft_catalog",
    }


def _score_numeric(want: Any, have: Any) -> tuple[bool, str]:
    passed = have is not None and want is not None and int(have) == int(want)
    return passed, f"want={want} got={have}"


def _norm_facet_name(name: Any) -> str:
    """Align generator casing/punctuation with response prose (C&A ↔ c a)."""
    s = str(name or "").casefold().strip()
    s = re.sub(r"[^\w\s]+", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def _score_facet_rows(want_rows: list[dict[str, Any]], got_rows: list[dict[str, Any]]) -> tuple[bool, str]:
    if not got_rows:
        return False, f"want_rows={len(want_rows)} got_rows=0"
    n = min(len(want_rows), len(got_rows))
    matched = 0
    for i in range(n):
        if _norm_facet_name(want_rows[i].get("name")) == _norm_facet_name(got_rows[i].get("name")) and int(
            want_rows[i].get("count")
        ) == int(got_rows[i].get("count")):
            matched += 1
        else:
            break
    ok = matched == n and n > 0
    # Require at least half of expected rows when brain truncated.
    if ok and n < len(want_rows) and n < max(3, len(want_rows) // 2):
        ok = False
    return ok, f"want_rows={len(want_rows)} got_rows={len(got_rows)} matched={matched}"


def _score(row: dict[str, Any], got: dict[str, Any]) -> dict[str, Any]:
    exp = dict(row.get("expected") or {})
    kind = str(exp.get("kind") or "")
    passed = False
    detail = ""
    if kind in {"product_total", "distinct_facet", "value_count", "value_count_zero"}:
        passed, detail = _score_numeric(exp.get("n"), got.get("n"))
    elif kind == "facet_top_n":
        passed, detail = _score_facet_rows(list(exp.get("rows") or []), list(got.get("rows") or []))
    else:
        detail = "unsupported"
    return {
        "id": row.get("id"),
        "type": row.get("type"),
        "graph_id": row.get("graph_id"),
        "filter": row.get("filter"),
        "value": row.get("value"),
        "question": row.get("question"),
        "passed": passed,
        "detail": detail,
        "expected": exp,
        "got": got,
    }


def _write_results(path: Path, summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--limit", type=int, default=0, help="0 = all")
    parser.add_argument("--types", default="", help="Comma filter on question type")
    parser.add_argument("--graphs", default="", help="Comma filter on graph_id")
    parser.add_argument("--via-brain", action="store_true", help="Score Fashion Brain draft_catalog answers")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25, help="Checkpoint write interval")
    args = parser.parse_args()

    if not args.questions.is_file():
        raise SystemExit(f"missing {args.questions} — run scripts/generate_kg_stats_meta_eval.py first")

    rows = _load_questions(args.questions)
    type_set = {t.strip() for t in args.types.split(",") if t.strip()}
    graph_set = {g.strip() for g in args.graphs.split(",") if g.strip()}
    if type_set:
        rows = [r for r in rows if str(r.get("type")) in type_set]
    if graph_set:
        rows = [r for r in rows if str(r.get("graph_id")) in graph_set]
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    platform = principal = None
    if args.via_brain:
        from knowledge_graph.access import Principal
        from knowledge_graph.config import load_settings
        from knowledge_graph.pipeline.service import Platform

        platform = Platform(load_settings())
        principal = Principal(subject="meta_eval", clearance="internal")

    out_path = RESULTS_BRAIN if args.via_brain else RESULTS
    results: list[dict[str, Any]] = []
    passed = 0
    brain_ms_total = 0.0
    t0 = time.perf_counter()
    for i, row in enumerate(rows, start=1):
        if args.via_brain and platform is not None and principal is not None:
            brain = _brain_answer(row, platform, principal)
            scored = _score(row, brain)
            scored["mode"] = "via_brain"
            scored["brain"] = brain
            brain_ms_total += float(brain.get("ms") or 0.0)
            # Keep direct oracle for debugging mismatches.
            try:
                scored["direct"] = _direct_answer(row)
            except Exception as exc:  # noqa: BLE001
                scored["direct"] = {"ok": False, "error": str(exc)}
        else:
            direct = _direct_answer(row)
            scored = _score(row, direct)
            scored["mode"] = "direct"

        if scored["passed"]:
            passed += 1
        results.append(scored)
        status = "PASS" if scored["passed"] else "FAIL"
        extra = ""
        if args.via_brain:
            b = scored.get("brain") or {}
            extra = f" [{b.get('ms')}ms routes={b.get('routes')}]"
        print(
            f"{status} {scored['id']} [{scored['type']}] {scored['detail']} :: "
            f"{scored['question'][:90]}{extra}",
            flush=True,
        )
        if args.progress_every > 0 and (i % args.progress_every == 0 or i == len(rows)):
            elapsed = (time.perf_counter() - t0) * 1000.0
            summary = {
                "total": len(results),
                "passed": passed,
                "failed": len(results) - passed,
                "pass_rate": round(100.0 * passed / len(results), 2) if results else 0.0,
                "elapsed_ms": round(elapsed, 1),
                "brain_ms_total": round(brain_ms_total, 1) if args.via_brain else None,
                "via_brain": bool(args.via_brain),
                "partial": i < len(rows),
            }
            _write_results(out_path, summary, results)
            print(f"checkpoint {i}/{len(rows)} → {out_path}", flush=True)
        if args.fail_fast and not scored["passed"]:
            break

    elapsed = (time.perf_counter() - t0) * 1000.0
    summary = {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": round(100.0 * passed / len(results), 2) if results else 0.0,
        "elapsed_ms": round(elapsed, 1),
        "brain_ms_total": round(brain_ms_total, 1) if args.via_brain else None,
        "via_brain": bool(args.via_brain),
        "partial": False,
    }
    _write_results(out_path, summary, results)
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
