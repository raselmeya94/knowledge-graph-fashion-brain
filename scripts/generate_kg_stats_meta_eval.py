#!/usr/bin/env python3
"""Build a kg_stats metadata question bank that covers every filter shard.

Reads ``data/generated/kg_stats/<graph>/`` and emits up to ``--limit`` questions
with expected answers (from the same shards). No Fashion Brain / Neo4j needed.

Usage (repo root):
  python scripts/generate_kg_stats_meta_eval.py
  python scripts/generate_kg_stats_meta_eval.py --limit 500 --graphs plmex_products,ecom_products
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "data" / "generated" / "kg_stats"
OUT_DIR = ROOT / "data" / "generated" / "eval"
OUT_JSON = OUT_DIR / "kg_stats_meta_questions.json"
OUT_TXT = OUT_DIR / "kg_stats_meta_questions.txt"

_EMPTY = {"", "(empty)", "(unspecified)", "n/a", "na", "none", "null", "-", "—"}

# Prefer human-facing coverage first when we hit the question budget.
_PRIORITY_FILTERS = (
    "brand",
    "buyer",
    "category",
    "garment_type",
    "gender",
    "availability",
    "section",
    "family",
    "subfamily",
    "material",
    "color",
    "department",
    "plm_main_category",
    "plm_sub_category",
    "plm_sub_sub_category",
    "season_year",
    "fit",
    "construction",
    "currency",
    "domain",
    "country_of_origin",
    "merchandiser",
    "designer",
)


def _human(attr: str) -> str:
    return str(attr or "").replace("_", " ").strip()


def _plural(attr: str) -> str:
    label = _human(attr)
    if label.endswith("y") and label not in {"availability"}:
        return label[:-1] + "ies"
    if label.endswith("s"):
        return label
    return label + "s"


def _graph_label(graph_id: str, meta: dict[str, Any]) -> str:
    name = str(meta.get("graph_name") or graph_id)
    if "plm" in graph_id.casefold():
        return "plmex"
    if "ecom" in graph_id.casefold():
        return "ecom"
    return name


def _usable_values(shard: dict[str, Any]) -> list[tuple[str, int]]:
    values = dict(shard.get("values") or {})
    ranked = [
        (str(k), int(v))
        for k, v in values.items()
        if str(k).strip().casefold() not in _EMPTY and int(v) > 0
    ]
    ranked.sort(key=lambda item: (-item[1], item[0].casefold()))
    return ranked


def _load_graph(graph_id: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = STATS / graph_id
    meta = json.loads((root / "_meta.json").read_text(encoding="utf-8"))
    shards: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*.json")):
        if path.name == "_meta.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            shards[path.stem] = payload
    return meta, shards


def _qid(prefix: str, n: int) -> str:
    return f"{prefix}{n:03d}"


def build_questions(
    graphs: list[str],
    *,
    limit: int = 500,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seq = 0

    def add(row: dict[str, Any]) -> bool:
        nonlocal seq
        if len(rows) >= limit:
            return False
        seq += 1
        row.setdefault("id", _qid("M", seq))
        rows.append(row)
        return True

    for graph_id in graphs:
        meta, shards = _load_graph(graph_id)
        label = _graph_label(graph_id, meta)
        products = int((meta.get("totals") or {}).get("products") or 0)
        filters = list(meta.get("filters") or list(shards.keys()))

        # 1) Product total
        if not add(
            {
                "type": "product_total",
                "graph_id": graph_id,
                "filter": None,
                "value": None,
                "question": f"How many products are in the {label} data?",
                "expected": {"kind": "product_total", "n": products},
                "paraphrases": [
                    f"What is the total product count in {label}?",
                    f"How many items are there in {label}?",
                ],
            }
        ):
            return rows

        # Priority ordering for coverage under the budget.
        ordered = [f for f in _PRIORITY_FILTERS if f in filters]
        ordered += [f for f in filters if f not in ordered]

        for attr in ordered:
            shard = shards.get(attr) or {}
            usable = _usable_values(shard)
            if not usable:
                continue
            # Prefer runtime unique_count() semantics when available; else usable length.
            try:
                from knowledge_graph.retrieval.kg_stats import unique_count as _uc

                runtime_unique = _uc(graph_id, attr)
            except Exception:
                runtime_unique = None
            unique = int(runtime_unique) if runtime_unique is not None else len(usable)
            if unique <= 0:
                continue
            # Skip pure id dumps (one value per product) for value questions —
            # still cover distinct count once.
            is_id_like = unique >= max(5000, int(products * 0.8)) if products else unique >= 5000

            # 2) Distinct facet count
            if not add(
                {
                    "type": "distinct_facet",
                    "graph_id": graph_id,
                    "filter": attr,
                    "value": None,
                    "question": f"How many {_plural(attr)} are in {label}?",
                    "expected": {
                        "kind": "distinct_facet",
                        "attribute": attr,
                        "n": unique,
                    },
                    "paraphrases": [
                        f"How many unique {_human(attr)} values in {label}?",
                        f"What is the number of distinct {_plural(attr)} in {label}?",
                    ],
                }
            ):
                return rows

            if is_id_like:
                continue

            # 3) Top-N ranking (only when enough values)
            if len(usable) >= 3:
                n = min(top_n, len(usable))
                expected_rows = [
                    {"rank": i + 1, "name": name, "count": count}
                    for i, (name, count) in enumerate(usable[:n])
                ]
                if not add(
                    {
                        "type": "facet_top_n",
                        "graph_id": graph_id,
                        "filter": attr,
                        "value": None,
                        "limit": n,
                        "question": f"List top {n} {_human(attr)} names in {label}",
                        "expected": {
                            "kind": "facet_top_n",
                            "attribute": attr,
                            "limit": n,
                            "rows": expected_rows,
                        },
                        "paraphrases": [
                            f"Show the top {n} {_plural(attr)} in {label}",
                            f"What are the top {n} {_plural(attr)} by product count in {label}?",
                        ],
                    }
                ):
                    return rows

            # 4) Value counts — top 1–2 (priority filters get 3)
            value_budget = 3 if attr in _PRIORITY_FILTERS else 2
            for name, count in usable[:value_budget]:
                # Natural phrasing; sweater/category style uses garment alias on priority types.
                q_main = f"How many products with {_human(attr)} {name} in {label}?"
                paraphrases = [
                    f"How many {label} products have {_human(attr)} {name}?",
                    f"Count products where {_human(attr)} is {name} in {label}",
                ]
                if attr in {"category", "garment_type"} and re.fullmatch(r"[A-Za-z0-9 \-_/]+", name):
                    paraphrases.append(f"How many {name} in {label}?")
                if not add(
                    {
                        "type": "value_count",
                        "graph_id": graph_id,
                        "filter": attr,
                        "value": name,
                        "question": q_main,
                        "expected": {
                            "kind": "value_count",
                            "attribute": attr,
                            "value": name,
                            "n": count,
                        },
                        "paraphrases": paraphrases,
                    }
                ):
                    return rows

            # 5) Missing-value probe (expected 0) for priority filters only
            if attr in _PRIORITY_FILTERS and usable:
                miss = f"__missing_{attr}__"
                if not add(
                    {
                        "type": "value_count_zero",
                        "graph_id": graph_id,
                        "filter": attr,
                        "value": miss,
                        "question": f"How many products with {_human(attr)} {miss} in {label}?",
                        "expected": {
                            "kind": "value_count",
                            "attribute": attr,
                            "value": miss,
                            "n": 0,
                        },
                        "paraphrases": [],
                    }
                ):
                    return rows

    return rows


def expand_with_paraphrases(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    """Optionally promote paraphrases into first-class questions until limit."""

    out = list(rows)
    if len(out) >= limit:
        return out[:limit]
    seq = len(out)
    for row in rows:
        for para in list(row.get("paraphrases") or []):
            if len(out) >= limit:
                return out
            seq += 1
            clone = dict(row)
            clone["id"] = _qid("P", seq)
            clone["question"] = para
            clone["parent_id"] = row.get("id")
            clone["is_paraphrase"] = True
            clone["paraphrases"] = []
            out.append(clone)
    return out


def write_outputs(rows: list[dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "count": len(rows),
        "source": "kg_stats filter shards",
        "questions": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "kg_stats metadata evaluation questions",
        f"count={len(rows)}",
        "expected answers are grounded in data/generated/kg_stats/<graph>/<filter>.json",
        "",
    ]
    by_type: dict[str, int] = {}
    by_graph: dict[str, int] = {}
    for row in rows:
        by_type[str(row.get("type"))] = by_type.get(str(row.get("type")), 0) + 1
        by_graph[str(row.get("graph_id"))] = by_graph.get(str(row.get("graph_id")), 0) + 1
        exp = row.get("expected") or {}
        n = exp.get("n")
        extra = ""
        if row.get("type") == "facet_top_n":
            extra = f" top{exp.get('limit')}"
        elif row.get("filter"):
            extra = f" {row.get('filter')}"
            if row.get("value") is not None:
                extra += f"={row.get('value')}"
        lines.append(
            f"{row['id']}\t[{row.get('type')}]{extra}\t{row['question']}\t"
            f"expect_n={n if n is not None else 'rows'}\tgraph={row.get('graph_id')}"
        )
    lines[2:2] = [
        f"by_type={json.dumps(by_type, sort_keys=True)}",
        f"by_graph={json.dumps(by_graph, sort_keys=True)}",
        "",
    ]
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--graphs",
        default="plmex_products,ecom_products",
        help="Comma-separated kg_stats graph folders",
    )
    parser.add_argument(
        "--include-paraphrases",
        action="store_true",
        help="Promote paraphrase variants into the bank until --limit",
    )
    args = parser.parse_args()
    graphs = [g.strip() for g in str(args.graphs).split(",") if g.strip()]
    for gid in graphs:
        if not (STATS / gid / "_meta.json").is_file():
            raise SystemExit(f"missing kg_stats meta for {gid}")

    rows = build_questions(graphs, limit=args.limit, top_n=args.top_n)
    if args.include_paraphrases:
        rows = expand_with_paraphrases(rows, limit=args.limit)
    write_outputs(rows)
    print(f"wrote {len(rows)} questions → {OUT_JSON}")
    print(f"index → {OUT_TXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
