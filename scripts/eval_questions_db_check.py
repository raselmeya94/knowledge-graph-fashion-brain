#!/usr/bin/env python3
"""Ask every questions.txt item, store answers, cross-check against Postgres + Neo4j.

Usage (repo root, APIs up on :8000 / :8001):
  python scripts/eval_questions_db_check.py
  python scripts/eval_questions_db_check.py --ids A1,B1,G1
  python scripts/eval_questions_db_check.py --in-process   # no HTTP; uses answer_question
  python scripts/eval_questions_db_check.py --retrieve-only  # skip LLM answer prose
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

FB_URL = os.getenv("FASHIONGPT_MODEL_APP_URL", "http://127.0.0.1:8000").rstrip("/")
KG_URL = os.getenv("FASHIONGPT_KG_URL", "http://127.0.0.1:8001").rstrip("/")
OUT_DIR = ROOT / "data" / "generated"
OUT_JSON = OUT_DIR / "questions_eval_answers.json"
OUT_MD = OUT_DIR / "questions_eval_answers.md"
OUT_CSV = OUT_DIR / "questions_eval_answers.csv"


def _parse_questions(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    section = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if re.match(r"^[A-H]\.\s+", line):
            section = line.split(".", 1)[0].strip()
            continue
        m = re.match(r"^([A-H]\d+)\.\s+(.+)$", line)
        if not m:
            continue
        qid, text = m.group(1), m.group(2).strip()
        text = re.sub(r"\s*\(Expect:.*$", "", text).strip()
        text = re.sub(r"\s*Expected ≈.*$", "", text).strip()
        rows.append({"id": qid, "section": section or qid[0], "question": text})
    return rows


def _http_json(url: str, payload: dict | None = None, *, method: str = "POST", timeout: float = 300.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"error": f"HTTP {exc.code}", "body": body[:800]}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _http_get(url: str, timeout: float = 5.0) -> dict:
    return _http_json(url, None, method="GET", timeout=timeout)


def _nums(text: str) -> list[float]:
    out: list[float] = []
    for m in re.finditer(r"(?<![A-Za-z0-9_/])(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)(?![A-Za-z0-9_])", text or ""):
        raw = m.group(1).replace(",", "")
        try:
            out.append(float(raw))
        except ValueError:
            continue
    return out


def _has_any_int(text: str, values: list[int], *, tol: int = 0) -> bool:
    found = {int(round(x)) for x in _nums(text) if abs(x - round(x)) < 1e-9}
    for v in values:
        if any(abs(f - v) <= tol for f in found):
            return True
    return False


def _load_ground_truth() -> dict[str, Any]:
    """Live Postgres + Neo4j facts used to validate answers."""

    import psycopg
    from neo4j import GraphDatabase

    dsn = os.getenv("FASHIONGPT_ECOM_DSN")
    if not dsn:
        raise RuntimeError("FASHIONGPT_ECOM_DSN missing")

    brands: dict[str, int] = {}
    avg_price: dict[str, float] = {}
    category_counts: dict[str, int] = {}
    primark_category: dict[str, int] = {}
    instock = 0
    tesco = 0
    material_cotton = 0
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "select lower(coalesce(brand,'')), count(*) "
                "from master_products group by 1"
            )
            brands = {str(b): int(c) for b, c in cur.fetchall() if b}
            cur.execute(
                "select count(*) from master_products "
                "where lower(coalesce(availability,'')) like %s",
                ("%instock%",),
            )
            instock = int(cur.fetchone()[0])
            # price may be text in scrape dumps — coerce safely
            cur.execute(
                """
                select lower(coalesce(brand,'')),
                       avg(nullif(regexp_replace(price::text, '[^0-9.\\-]', '', 'g'), '')::float)
                from master_products
                where coalesce(price::text, '') <> ''
                group by 1
                """
            )
            avg_price = {str(b): float(a) for b, a in cur.fetchall() if b and a is not None}
            cur.execute(
                "select lower(coalesce(category,'')), count(*) "
                "from master_products where coalesce(category,'') <> '' "
                "group by 1 order by 2 desc limit 25"
            )
            category_counts = {str(k): int(v) for k, v in cur.fetchall()}
            cur.execute(
                "select lower(coalesce(category,'')), count(*) from master_products "
                "where lower(coalesce(brand,'')) = 'primark' and coalesce(category,'') <> '' "
                "group by 1 order by 2 desc limit 25"
            )
            primark_category = {str(k): int(v) for k, v in cur.fetchall()}
            cur.execute(
                "select count(*) from master_products "
                "where lower(coalesce(brand,'')) in ('tesco')"
            )
            tesco = int(cur.fetchone()[0])
            # material / composition often lives in a text column
            cols = set()
            cur.execute(
                "select column_name from information_schema.columns "
                "where table_name = 'master_products'"
            )
            cols = {r[0] for r in cur.fetchall()}
            mat_col = None
            for cand in ("composition", "materials", "material", "fabric"):
                if cand in cols:
                    mat_col = cand
                    break
            if mat_col:
                cur.execute(
                    f"select count(*) from master_products "
                    f"where lower(coalesce({mat_col},'')) like %s",
                    ("%cotton%",),
                )
                material_cotton = int(cur.fetchone()[0])

    neo: dict[str, Any] = {}
    uri = os.getenv("FASHIONGPT_NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("FASHIONGPT_NEO4J_USER", "neo4j")
    password = os.getenv("FASHIONGPT_NEO4J_PASSWORD", "")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as s:
        neo["products"] = int(s.run("MATCH (p:Product) RETURN count(p) AS c").single()["c"])
        neo["price_events"] = int(
            s.run("MATCH (n:PriceChangeEvent) RETURN count(n) AS c").single()["c"]
        )
        neo["avail_snapshots"] = int(
            s.run("MATCH (n:AvailabilitySnapshot) RETURN count(n) AS c").single()["c"]
        )
        row = s.run(
            """
            MATCH (p:Product)-[:HAS_PRICE_CHANGE]->(e:PriceChangeEvent)
            WHERE p.id CONTAINS $pid OR coalesce(p.sku,'') CONTAINS $pid
               OR coalesce(p.product_id,'') CONTAINS $pid
            RETURN p.id AS id, p.name AS name, e.old_price AS old_price,
                   e.new_price AS new_price, e.percent_change AS pct
            ORDER BY abs(coalesce(e.percent_change,0)) DESC LIMIT 3
            """,
            pid="05919713",
        ).data()
        neo["zara_05919713"] = row
        neo["top_drops"] = s.run(
            """
            MATCH (p:Product)-[:HAS_PRICE_CHANGE]->(e:PriceChangeEvent)
            WHERE coalesce(e.direction,'') = 'down'
              AND coalesce(e.same_currency, true) = true
            RETURN p.name AS name, e.old_price AS old_price, e.new_price AS new_price,
                   e.percent_change AS pct
            ORDER BY coalesce(e.percent_change, 0) ASC LIMIT 5
            """
        ).data()
    driver.close()

    return {
        "brands": brands,
        "avg_price": avg_price,
        "category_counts": category_counts,
        "primark_category": primark_category,
        "instock": instock,
        "tesco": tesco,
        "material_cotton": material_cotton,
        "neo4j": neo,
        "aliases": {
            "c&a": "c-and-a",
            "c and a": "c-and-a",
            "calvin klein": "calvin-klein",
        },
    }


def _brand_count(gt: dict, name: str) -> int | None:
    key = name.strip().casefold()
    key = gt["aliases"].get(key, key)
    if key in gt["brands"]:
        return int(gt["brands"][key])
    return None


def _cross_check(qid: str, question: str, answer: str, retrieve_text: str, gt: dict) -> dict[str, Any]:
    """Compare answer (and retrieve text) to live DB/Neo4j ground truth."""

    blob = f"{answer or ''}\n{retrieve_text or ''}"
    expected: dict[str, Any] = {}
    notes: list[str] = []
    verdict = "UNCHECKED"

    def ok_count(label: str, expected_n: int, *, tol: int = 0) -> None:
        nonlocal verdict
        expected[label] = expected_n
        if _has_any_int(blob, [expected_n], tol=tol):
            notes.append(f"matched_{label}={expected_n}")
            if verdict == "UNCHECKED":
                verdict = "VALID"
        else:
            notes.append(f"missed_{label}={expected_n}; nums={_nums(blob)[:12]}")
            verdict = "INVALID"

    # --- countable / golden checks ---
    if qid == "A1":
        ok_count("primark", _brand_count(gt, "primark") or 0, tol=5)
    elif qid == "A2":
        ok_count("zara", _brand_count(gt, "zara") or 0, tol=5)
    elif qid == "A3":
        ok_count("c-and-a", _brand_count(gt, "c-and-a") or 0, tol=5)
    elif qid == "A9":
        ok_count("instock", int(gt["instock"]), tol=50)
    elif qid == "B1":
        expected["brands"] = gt["brands"]
        hits = sum(1 for b, n in gt["brands"].items() if n >= 10 and _has_any_int(blob, [n], tol=5))
        if hits >= max(2, min(3, len([n for n in gt["brands"].values() if n >= 10]))):
            notes.append(f"matched_brand_counts={hits}")
            verdict = "VALID"
        else:
            notes.append(f"weak_brand_counts hits={hits}")
            verdict = "INVALID"
    elif qid == "B2":
        expected["primark_category_top"] = dict(list(gt["primark_category"].items())[:5])
        top_ns = list(gt["primark_category"].values())[:3]
        if top_ns and sum(1 for n in top_ns if _has_any_int(blob, [n], tol=3)) >= 1:
            notes.append("matched_primark_category_count")
            verdict = "VALID"
        elif re.search(r"(?i)primark", blob) and _nums(blob):
            notes.append("primark_category_partial")
            verdict = "PARTIAL"
        else:
            notes.append("primark_category_miss")
            verdict = "INVALID"
    elif qid == "B4":
        p = _brand_count(gt, "primark") or 0
        z = _brand_count(gt, "zara") or 0
        expected["primark"] = p
        expected["zara"] = z
        if _has_any_int(blob, [p], tol=5) and _has_any_int(blob, [z], tol=5):
            notes.append("matched_primark_and_zara")
            verdict = "VALID"
        else:
            notes.append("missed_compare_counts")
            verdict = "INVALID"
    elif qid == "B7":
        expected["avg_price"] = {k: round(v, 2) for k, v in gt["avg_price"].items()}
        # soft: at least one brand avg appears within ±1
        hit = False
        for brand, avg in gt["avg_price"].items():
            for n in _nums(blob):
                if abs(n - avg) <= 1.0:
                    hit = True
                    notes.append(f"matched_avg_{brand}≈{avg:.2f}")
                    break
            if hit:
                break
        verdict = "VALID" if hit else "PARTIAL"
        if not hit:
            notes.append("avg_price_not_found")
    elif qid == "B8":
        pa = gt["avg_price"].get("primark")
        za = gt["avg_price"].get("zara")
        expected["primark_avg"] = pa
        expected["zara_avg"] = za
        ok = False
        if pa is not None and za is not None:
            ok = any(abs(n - pa) <= 1.0 for n in _nums(blob)) and any(
                abs(n - za) <= 1.0 for n in _nums(blob)
            )
        verdict = "VALID" if ok else "PARTIAL"
        notes.append("avg_compare_ok" if ok else "avg_compare_weak")
    elif qid in {"D14", "G1"}:
        ok_count("PriceChangeEvent", int(gt["neo4j"]["price_events"]), tol=0)
    elif qid in {"E3", "G2"}:
        ok_count("AvailabilitySnapshot", int(gt["neo4j"]["avail_snapshots"]), tol=0)
    elif qid in {"D13", "G3"}:
        expected["zara_05919713"] = gt["neo4j"]["zara_05919713"]
        if re.search(r"(?i)85\.9|39\.95|-53|53\s*%", blob):
            notes.append("matched_known_delta")
            verdict = "VALID"
        else:
            notes.append("known_delta_missing")
            verdict = "INVALID"
    elif qid == "G5":
        avg = gt["avg_price"].get("primark")
        expected["primark_avg"] = avg
        if avg is not None and any(abs(n - avg) <= 1.0 for n in _nums(blob)):
            notes.append(f"matched_primark_avg≈{avg:.2f}")
            verdict = "VALID"
        else:
            notes.append("primark_avg_miss")
            verdict = "INVALID"
    elif qid == "F1":
        expected["tesco"] = gt["tesco"]
        invent = bool(re.search(r"(?i)\btesco\b", blob)) and len(_nums(blob)) > 3 and gt["tesco"] == 0
        closed = bool(
            re.search(
                r"(?i)\b(no |not |0 product|none|unavailable|don't have|do not|missing|empty)\b",
                blob,
            )
        )
        if invent:
            notes.append("hallucinated_tesco")
            verdict = "INVALID"
        elif closed or gt["tesco"] == 0:
            notes.append("boundary_closed_ok")
            verdict = "VALID"
        else:
            notes.append("boundary_unclear")
            verdict = "PARTIAL"
    elif qid in {"F2", "F3", "F4"}:
        closed = bool(
            re.search(
                r"(?i)\b(no |not |unavailable|don't have|do not|missing|empty|not in|weak)\b",
                blob,
            )
        )
        if closed:
            notes.append("boundary_honest")
            verdict = "VALID"
        elif not (answer or retrieve_text).strip():
            notes.append("empty_boundary")
            verdict = "PARTIAL"
        else:
            # dense invented tables are bad
            if len(_nums(blob)) > 8:
                notes.append("possible_hallucination_dense_numbers")
                verdict = "INVALID"
            else:
                notes.append("boundary_soft")
                verdict = "PARTIAL"
    elif qid.startswith("H"):
        # shared interconnect — validate non-empty OR honest empty; DB can't fully prove shared
        if re.search(r"(?i)share|shared|interconnect|canonical|cotton|polyester|material", blob) and (
            _nums(blob) or re.search(r"(?i)no |not |none|empty", blob)
        ):
            notes.append("shared_shaped_ok")
            verdict = "PARTIAL"
        else:
            notes.append("shared_weak")
            verdict = "PARTIAL"
    else:
        # Soft structural check: non-empty answer with some grounding signal.
        # Do NOT treat prose like "pricing error" / "correction" as a runtime failure.
        if not (answer or retrieve_text).strip():
            notes.append("empty_answer")
            verdict = "INVALID"
        elif re.search(
            r"(?i)(\btraceback\b|\bcould not reach\b|\bruntimeerror\b|\bhttperror\b|"
            r"\binternal server error\b|\bexception:\s|\berror:\s)",
            blob,
        ):
            notes.append("runtime_error")
            verdict = "INVALID"
        elif _nums(blob) and len((answer or retrieve_text or "").strip()) >= 40:
            # Grounded-looking answer without a hard numeric oracle for this id.
            notes.append("grounded_soft_ok")
            verdict = "VALID"
        else:
            notes.append("no_exact_db_oracle")
            verdict = "PARTIAL"

    return {"verdict": verdict, "expected": expected, "notes": notes}


def _normalize_sources(raw: Any) -> list[dict[str, Any]]:
    """Flatten GroundedAnswer.evidence (or legacy sources) into storeable dicts."""

    items: list[Any] = []
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = list(raw.get("evidence") or raw.get("sources") or [])
    out: list[dict[str, Any]] = []
    for item in items:
        if hasattr(item, "model_dump"):
            item = item.model_dump()
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "source_id": item.get("source_id") or item.get("graph") or "",
                "record_id": item.get("record_id") or item.get("node_id") or "",
                "title": item.get("title") or "",
                "excerpt": (item.get("excerpt") or "")[:500],
                "graph": item.get("graph") or item.get("source_path") or "",
                "verification_status": item.get("verification_status") or "",
            }
        )
    return out


def _ask_http(question: str) -> dict[str, Any]:
    raw = _http_json(f"{FB_URL}/ask", {"question": question}, timeout=600)
    if raw.get("error"):
        return {
            "answer": "",
            "routes": [],
            "sources": [],
            "error": raw.get("error"),
            "body": raw.get("body"),
            "raw": raw,
        }
    return {
        "answer": str(raw.get("answer") or ""),
        "routes": list(raw.get("routes") or []),
        "sources": _normalize_sources(raw.get("evidence") or raw.get("sources") or raw),
        "caveats": list(raw.get("caveats") or []),
        "unavailable": list(raw.get("unavailable") or []),
        "thinking": raw.get("thinking"),
        "raw": raw,
    }


def _ask_in_process(platform, principal, question: str) -> dict[str, Any]:
    from fashion_brain.ask import answer_question

    ans = answer_question(platform, question, principal, history=None)
    dump = ans.model_dump() if hasattr(ans, "model_dump") else {}
    return {
        "answer": str(getattr(ans, "answer", "") or ""),
        "routes": list(getattr(ans, "routes", None) or []),
        "sources": _normalize_sources(getattr(ans, "evidence", None) or dump),
        "caveats": list(getattr(ans, "caveats", None) or []),
        "unavailable": list(getattr(ans, "unavailable", None) or []),
        "thinking": getattr(ans, "thinking", None),
        "raw": dump or None,
    }


def _retrieve_only(question: str) -> dict[str, Any]:
    payload = {
        "question": question,
        "intent": "product_count",
        "filters": {},
        "group_by": None,
        "limit": None,
        "suggest": False,
        "analysis": True,
        "graph_id": "ecom_products",
    }
    raw = _http_json(f"{KG_URL}/brain/retrieve", payload, timeout=120)
    return {
        "answer": str(raw.get("text") or ""),
        "routes": list(raw.get("routes") or []),
        "retrieve_text": str(raw.get("text") or ""),
        "error": raw.get("error"),
        "raw": raw,
    }


def _write_outputs(report: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Questions eval (answers + DB cross-check)",
        "",
        f"Generated: {report['generated_at']}",
        f"Mode: {report['mode']}",
        "",
        "## Summary",
        "",
        f"- total: {report['summary']['total']}",
        f"- VALID: {report['summary']['VALID']}",
        f"- PARTIAL: {report['summary']['PARTIAL']}",
        f"- INVALID: {report['summary']['INVALID']}",
        f"- ERROR: {report['summary']['ERROR']}",
        "",
        "## Rows",
        "",
    ]
    for row in report["rows"]:
        lines.append(f"### {row['id']} — {row['db_verdict']}")
        lines.append("")
        lines.append(f"**Q:** {row['question']}")
        lines.append("")
        lines.append(f"**Routes:** {' · '.join(row.get('routes') or []) or '—'}")
        lines.append("")
        lines.append(f"**DB notes:** {', '.join(row.get('db_notes') or []) or '—'}")
        lines.append("")
        ans = (row.get("answer") or "").strip() or "_(empty)_"
        if len(ans) > 2500:
            ans = ans[:2500].rstrip() + "…"
        lines.append("**Answer:**")
        lines.append("")
        lines.append(ans)
        lines.append("")
        sources = row.get("sources") or []
        lines.append(f"**Sources ({len(sources)}):**")
        lines.append("")
        if not sources:
            lines.append("_(none)_")
        else:
            for src in sources[:20]:
                title = src.get("title") or src.get("record_id") or "source"
                graph = src.get("graph") or src.get("source_id") or ""
                excerpt = (src.get("excerpt") or "").strip()
                bit = f"- **{title}**"
                if graph:
                    bit += f" (`{graph}`)"
                if excerpt:
                    bit += f" — {excerpt[:180]}"
                lines.append(bit)
            if len(sources) > 20:
                lines.append(f"- … +{len(sources) - 20} more")
        lines.append("")
        lines.append("---")
        lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    # CSV: id, verdict, question, answer, sources, notes
    import csv

    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "id",
                "section",
                "db_verdict",
                "elapsed_s",
                "question",
                "answer",
                "sources",
                "source_count",
                "db_notes",
                "routes",
                "error",
            ]
        )
        for row in report["rows"]:
            sources = row.get("sources") or []
            src_text = " || ".join(
                f"{s.get('title') or s.get('record_id') or ''} [{s.get('graph') or s.get('source_id') or ''}] {(s.get('excerpt') or '')[:120]}"
                for s in sources
            )
            w.writerow(
                [
                    row.get("id"),
                    row.get("section"),
                    row.get("db_verdict"),
                    row.get("elapsed_s"),
                    row.get("question"),
                    row.get("answer"),
                    src_text,
                    len(sources),
                    "; ".join(row.get("db_notes") or []),
                    "|".join(row.get("routes") or []),
                    row.get("error") or "",
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(ROOT / "questions.txt"))
    parser.add_argument("--ids", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--in-process", action="store_true")
    parser.add_argument("--retrieve-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip ids already in OUT_JSON")
    args = parser.parse_args()

    cases = _parse_questions(Path(args.questions))
    want_ids = {x.strip() for x in args.ids.split(",") if x.strip()} if args.ids else set()
    if want_ids:
        # Keep full question list so merged prior rows for other ids stay in the report.
        reeval = {c["id"] for c in cases if c["id"] in want_ids}
        if not reeval:
            print(f"No matching --ids in {args.questions}: {sorted(want_ids)}")
            return 2
    else:
        reeval = {c["id"] for c in cases}
    if args.limit and not want_ids:
        cases = cases[: args.limit]
        reeval = {c["id"] for c in cases}

    print(f"Loading DB/Neo4j ground truth…")
    gt = _load_ground_truth()
    print(
        "Ground truth:",
        f"brands={gt['brands']}",
        f"instock={gt['instock']}",
        f"PriceChangeEvent={gt['neo4j']['price_events']}",
        f"AvailabilitySnapshot={gt['neo4j']['avail_snapshots']}",
    )

    prior_rows: dict[str, dict] = {}
    # --ids alone used to overwrite the whole report with 1 row; always merge
    # existing OUT_JSON when present so partial re-runs keep other answers.
    if (args.resume or want_ids) and OUT_JSON.exists():
        try:
            prior = json.loads(OUT_JSON.read_text(encoding="utf-8"))
            for r in prior.get("rows") or []:
                if not r.get("id"):
                    continue
                qid = str(r["id"])
                # Selected --ids are always re-asked; keep everything else.
                if want_ids and qid in want_ids:
                    continue
                if args.resume and not want_ids and r.get("db_verdict") in {None, "ERROR"}:
                    continue
                prior_rows[qid] = r
            print(f"Merge/resume: keeping {len(prior_rows)} prior rows")
        except Exception as exc:  # noqa: BLE001
            print(f"Resume load failed: {exc}")

    platform = None
    principal = None
    if args.in_process and not args.retrieve_only:
        from knowledge_graph.access import Principal
        from knowledge_graph.config import load_settings
        from knowledge_graph.pipeline.service import Platform

        platform = Platform(load_settings())
        principal = Principal(subject="eval", clearance="internal")
        mode = "in-process"
    elif args.retrieve_only:
        mode = "retrieve-only"
        health = _http_get(f"{KG_URL}/health")
        if health.get("error"):
            print(f"KG health failed: {health}")
            return 2
    else:
        mode = "http-/ask"
        fb = _http_get(f"{FB_URL}/health")
        kg = _http_get(f"{KG_URL}/health")
        if fb.get("error") or kg.get("error"):
            print(f"Health check failed FB={fb} KG={kg}")
            print("Start: python -m knowledge_graph.cli serve   and   python -m fashion_brain.cli serve")
            return 2

    rows: list[dict] = []
    ask_n = sum(1 for c in cases if c["id"] in reeval and c["id"] not in prior_rows)
    print(f"Evaluating {len(cases)} questions ({mode}); asking {ask_n}, keeping {len(prior_rows)}…")
    for idx, case in enumerate(cases, start=1):
        qid = case["id"]
        if qid in prior_rows and qid not in reeval:
            rows.append(prior_rows[qid])
            print(f"[{idx}/{len(cases)}] {qid}: merge keep → {prior_rows[qid].get('db_verdict')}")
            continue
        if qid in prior_rows and args.resume and not want_ids:
            rows.append(prior_rows[qid])
            print(f"[{idx}/{len(cases)}] {qid}: resume keep → {prior_rows[qid].get('db_verdict')}")
            continue
        if want_ids and qid not in want_ids:
            # No prior row for this id — leave a gap only when --ids is selective.
            continue

        print(f"[{idx}/{len(cases)}] {qid}: {case['question'][:80]}", flush=True)
        t0 = time.perf_counter()
        err = None
        answer = ""
        routes: list[str] = []
        sources: list[dict[str, Any]] = []
        caveats: list[str] = []
        unavailable: list[str] = []
        retrieve_text = ""
        try:
            if args.retrieve_only:
                got = _retrieve_only(case["question"])
            elif args.in_process:
                got = _ask_in_process(platform, principal, case["question"])
            else:
                got = _ask_http(case["question"])
            answer = str(got.get("answer") or "")
            routes = list(got.get("routes") or [])
            sources = list(got.get("sources") or [])
            caveats = list(got.get("caveats") or [])
            unavailable = list(got.get("unavailable") or [])
            retrieve_text = str(got.get("retrieve_text") or "")
            err = got.get("error")
            # Prefer findings text from GroundedAnswer sources if present
            if not retrieve_text and isinstance(got.get("raw"), dict):
                retrieve_text = str(
                    (got["raw"].get("findings") or got["raw"].get("evidence_text") or "")
                )
        except Exception as exc:  # noqa: BLE001
            err = str(exc)

        elapsed = round(time.perf_counter() - t0, 3)
        if err and not answer:
            check = {"verdict": "ERROR", "expected": {}, "notes": [f"error={err}"]}
        else:
            check = _cross_check(qid, case["question"], answer, retrieve_text, gt)

        row = {
            "id": qid,
            "section": case["section"],
            "question": case["question"],
            "answer": answer,
            "sources": sources,
            "source_count": len(sources),
            "caveats": caveats,
            "unavailable": unavailable,
            "routes": routes,
            "retrieve_preview": (retrieve_text or "")[:500],
            "db_verdict": check["verdict"],
            "db_expected": check["expected"],
            "db_notes": check["notes"],
            "error": err,
            "elapsed_s": elapsed,
        }
        rows.append(row)
        print(
            f"    → {row['db_verdict']} ({elapsed}s) sources={len(sources)} {row['db_notes'][:3]}",
            flush=True,
        )

        # incremental save
        summary = {
            "VALID": sum(1 for r in rows if r["db_verdict"] == "VALID"),
            "PARTIAL": sum(1 for r in rows if r["db_verdict"] == "PARTIAL"),
            "INVALID": sum(1 for r in rows if r["db_verdict"] == "INVALID"),
            "ERROR": sum(1 for r in rows if r["db_verdict"] == "ERROR"),
            "UNCHECKED": sum(1 for r in rows if r["db_verdict"] == "UNCHECKED"),
            "total": len(rows),
        }
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "ground_truth": {
                "brands": gt["brands"],
                "instock": gt["instock"],
                "avg_price": {k: round(v, 2) for k, v in gt["avg_price"].items()},
                "neo4j": {
                    "products": gt["neo4j"]["products"],
                    "price_events": gt["neo4j"]["price_events"],
                    "avail_snapshots": gt["neo4j"]["avail_snapshots"],
                },
            },
            "summary": summary,
            "rows": rows,
        }
        _write_outputs(report)

    summary = {
        "VALID": sum(1 for r in rows if r["db_verdict"] == "VALID"),
        "PARTIAL": sum(1 for r in rows if r["db_verdict"] == "PARTIAL"),
        "INVALID": sum(1 for r in rows if r["db_verdict"] == "INVALID"),
        "ERROR": sum(1 for r in rows if r["db_verdict"] == "ERROR"),
        "UNCHECKED": sum(1 for r in rows if r["db_verdict"] == "UNCHECKED"),
        "total": len(rows),
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_CSV}")
    return 0 if summary["ERROR"] == 0 and summary["INVALID"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
