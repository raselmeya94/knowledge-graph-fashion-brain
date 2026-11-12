#!/usr/bin/env python3
"""Systemic eval of questions.txt (+ dynamic paraphrases).

Scores plan hygiene, routing, and grounded brain retrieve — not a hardcoded
Q→answer allowlist. Paraphrases prove the planner generalizes.

Usage (repo root):
  python scripts/eval_questions_systemic.py
  python scripts/eval_questions_systemic.py --full-ask   # also hit Fashion Brain /ask
  python scripts/eval_questions_systemic.py --ids D3,D10,G3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FB_URL = "http://127.0.0.1:8000"
KG_URL = "http://127.0.0.1:8001"
OUT_DIR = ROOT / "data" / "generated"
OUT_JSON = OUT_DIR / "eval_questions_results.json"


def _parse_questions(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    section = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if re.match(r"^[A-G]\.\s+", line):
            section = line.split(".", 1)[0].strip()
            continue
        m = re.match(r"^([A-G]\d+)\.\s+(.+)$", line)
        if not m:
            continue
        qid, text = m.group(1), m.group(2).strip()
        # Drop parenthetical expect notes from F/G lines when present on same line.
        text = re.sub(r"\s*\(Expect:.*$", "", text).strip()
        rows.append({"id": qid, "section": section or qid[0], "question": text})
    return rows


def _paraphrases(qid: str, question: str) -> list[str]:
    """Dynamic variants — same intent family, different surface form."""

    out: list[str] = []
    q = question
    swaps = [
        (r"(?i)^how many\b", "What is the count of"),
        (r"(?i)^which products had\b", "Show products with"),
        (r"(?i)^what are the average price change trends for\b", "Overall price trend for"),
        (r"(?i)^compare historical price moves for similar\b", "Side-by-side historical price trends for"),
        (r"(?i)^analyze product distribution by\b", "Break down products by"),
        (r"(?i)^show product counts by\b", "Product totals grouped by"),
        (r"(?i)^list products from\b", "Which products are from"),
        (r"(?i)^give me the full specification for\b", "Full product specs for"),
    ]
    for pat, repl in swaps:
        if re.search(pat, q):
            out.append(re.sub(pat, repl, q, count=1))
            break
    # Case / connector variants for brand compares.
    if re.search(r"(?i)\b(zara)\b.*\b(primark)\b|\b(primark)\b.*\b(zara)\b", q):
        out.append(re.sub(r"(?i)\bzara\b", "ZARA", q, count=1))
        if " and " in q.casefold():
            out.append(re.sub(r"(?i)\band\b", "vs", q, count=1))
        elif " between " in q.casefold() or " or " in q.casefold():
            out.append(re.sub(r"(?i)\bor\b", "and", q, count=1))
    # Generic price-history family probes (not tied to one ID).
    if qid.startswith("D") or qid.startswith("G"):
        if re.search(r"(?i)price", q) and re.search(r"(?i)decreas|drop|cheaper", q):
            out.append("Which catalog items got cheaper the most in the same currency?")
        if re.search(r"(?i)average|trend", q) and re.search(r"(?i)zara|primark|c&a|c-and-a", q):
            brand = "Zara"
            for b in ("Primark", "Zara", "C&A", "Matalan"):
                if b.casefold() in q.casefold() or (b == "C&A" and "c&a" in q.casefold()):
                    brand = b
                    break
            out.append(f"Summarize {brand}'s historical price movement.")
    # Dedup while preserving order.
    seen: set[str] = set()
    uniq: list[str] = []
    for item in out:
        key = item.casefold().strip()
        if key and key != q.casefold() and key not in seen:
            seen.add(key)
            uniq.append(item)
    return uniq[:3]


def _http_json(url: str, payload: dict, timeout: float = 120.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"error": f"HTTP {exc.code}", "body": body[:500]}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _junk_in_filters(filters: dict) -> list[str]:
    from fashion_brain.ask import (
        _METRIC_OPERATOR_WORDS,
        _TASK_WORDS,
        _reliable_filter,
        split_multi_party_filter,
    )

    bad: list[str] = []
    for key, value in (filters or {}).items():
        text = str(value or "")
        if key in {"brand", "buyer"}:
            parts = split_multi_party_filter(text) or [text]
            for part in parts:
                if not _reliable_filter(part):
                    bad.append(f"{key}={part}")
            continue
        if not text:
            continue
        tokens = set(re.findall(r"[a-z0-9&'-]+", text.casefold()))
        if tokens & _TASK_WORDS or tokens & _METRIC_OPERATOR_WORDS:
            bad.append(f"{key}={text}")
        elif key in {"garment", "garment_type", "category", "family", "subfamily"} and not _reliable_filter(text):
            bad.append(f"{key}={text}")
    return bad


def _score_row(
    *,
    qid: str,
    section: str,
    question: str,
    plan: dict,
    routes: list[str],
    mode: str | None,
    thinking: str,
    intention: dict,
    retrieve: dict,
    ask: dict | None,
) -> dict[str, Any]:
    filters = dict(plan.get("filters") or {})
    junk = _junk_in_filters(filters)
    intent_filters = dict(intention.get("filters") or {})
    junk += [f"intent:{x}" for x in _junk_in_filters(intent_filters) if f"{x}" not in junk]

    text = str((retrieve or {}).get("text") or "")
    err = (retrieve or {}).get("error")
    caveats = list((retrieve or {}).get("caveats") or [])
    evidence = list((retrieve or {}).get("evidence") or [])
    unavailable = list((retrieve or {}).get("unavailable") or [])

    notes: list[str] = []
    status = "PASS"

    if junk:
        status = "FAIL"
        notes.append(f"junk_filters={junk}")

    if err:
        status = "FAIL"
        notes.append(f"retrieve_error={err}")

    # Section-aware systemic expectations (capability classes, not exact strings).
    price_ask = bool(mode) or bool(
        re.search(r"(?i)\b(price|prices)\b", question)
        and re.search(r"(?i)change|trend|history|historical|decrease|increase|drop|rise|move|mover|fluctuat|cheaper", question)
    )
    if section in {"D", "G"} and price_ask and routes and routes[0] not in {"product_count", "products"}:
        status = "FAIL" if status == "PASS" else status
        notes.append(f"bad_route_for_price={routes}")

    if section == "F":
        # Boundary: must not invent dense product tables for missing entities.
        invented = bool(re.search(r"(?i)\btesco\b", text)) and "product" in text.casefold() and len(evidence) > 2
        if invented:
            status = "FAIL"
            notes.append("boundary_hallucination_tesco")
        elif not text.strip() and not unavailable and not caveats:
            status = "PARTIAL"
            notes.append("boundary_empty_without_caveat")
        elif re.search(r"(?i)no |not |unavailable|empty|missing|do not|don't|not in|not available|0 product", text) or caveats or unavailable:
            notes.append("boundary_closed_ok")
        else:
            status = "PARTIAL" if status == "PASS" else status
            notes.append("boundary_unclear")

    # Wrong-shaped answers: product-count tables for average-price / composition asks.
    if re.search(r"(?i)average prices?|avg price", question) and not re.search(
        r"(?i)change|trend|history|historical|delta", question
    ):
        if re.search(r"(?i)comparing product counts|products with brand", text) and not re.search(
            r"(?i)avg price|average price", text
        ):
            status = "FAIL"
            notes.append("avg_price_answered_as_count")
    if re.search(r"(?i)composition|care instructions", question) and re.search(
        r"(?i)comparing product counts|0 products with garment", text
    ):
        status = "PARTIAL" if status == "PASS" else status
        notes.append("spec_capability_thin")

    if section in {"D", "G"} and (price_ask or re.search(r"(?i)pricechangeevent|got cheaper", question)) and not err:
        if mode is None and re.search(r"(?i)price|cheaper|pricechangeevent", question) and not re.search(
            r"(?i)availabilitysnapshot", question
        ):
            # Average current price asks are not historical.
            if not re.search(r"(?i)average price of|right now|current-state", question):
                status = "FAIL"
                notes.append("price_history_mode_missed")
        if not text.strip():
            status = "FAIL"
            notes.append("empty_price_answer")
        elif re.search(r"(?i)don't have any data|no data on historical|insufficient", text):
            status = "FAIL"
            notes.append("false_empty_on_known_capability")
        elif re.search(r"(?i)there are 0 pricechangeevent|0 products with garment pricechangeevent", text):
            status = "FAIL"
            notes.append("zero_events_unexpected")
        elif re.search(r"(?i)pricechangeevent|price change events?", question) and not re.search(
            r"(?i)\b\d+\s+pricechangeevent|\bpricechangeevent nodes|price change events? nodes",
            text,
        ):
            if intention.get("intent") == "system_meta" or re.search(
                r"(?i)registered knowledge graphs", text
            ):
                status = "FAIL"
                notes.append("event_count_routed_system_meta")
            elif not re.search(r"(?i)\b\d+\b", text):
                status = "FAIL"
                notes.append("event_count_wrong_shape")
        elif re.search(r"(?i)05919713|satin shorts", question) and not re.search(
            r"(?i)85\.9|39\.95|-53", text
        ):
            status = "PARTIAL" if status == "PASS" else status
            notes.append("known_product_delta_missing")

    if section in {"A", "B"} and not err and not text.strip():
        status = "PARTIAL"
        notes.append("empty_catalog_answer")

    if ask:
        ans = str(ask.get("answer") or "")
        if ask.get("error"):
            status = "FAIL" if status == "PASS" else status
            notes.append(f"ask_error={ask.get('error')}")
        elif not ans.strip():
            status = "PARTIAL" if status == "PASS" else status
            notes.append("empty_llm_answer")
        elif re.search(r"(?i)don't have any data on historical price", ans):
            status = "FAIL"
            notes.append("llm_denied_price_history")

    if not thinking.strip():
        notes.append("thinking_empty")
        if status == "PASS":
            status = "PARTIAL"

    return {
        "id": qid,
        "section": section,
        "question": question,
        "status": status,
        "routes": routes,
        "price_history_mode": mode,
        "plan_filters": filters,
        "plan_group_by": plan.get("group_by"),
        "intention": {
            "intent": intention.get("intent"),
            "filters": intention.get("filters"),
            "group_by": intention.get("group_by"),
            "analysis": intention.get("analysis"),
            "thinking": (thinking or "")[:500],
        },
        "retrieve_chars": len(text),
        "retrieve_preview": text[:400],
        "evidence": len(evidence),
        "caveats": caveats[:4],
        "unavailable": unavailable,
        "ask_preview": (str((ask or {}).get("answer") or "")[:300] if ask else None),
        "notes": notes,
    }


def _run_one(platform, principal, qid: str, section: str, question: str, *, full_ask: bool) -> dict:
    from fashion_brain.agents._impl import IntentionAgent, ThinkingBrain
    from fashion_brain.ask import plan_routes, resolve_catalog_plan
    from knowledge_graph.retrieval.ecom import price_history_mode

    t0 = time.perf_counter()
    routes = plan_routes(question)
    plan = resolve_catalog_plan(question, None, carry=False)
    mode = price_history_mode(question)
    summary = ThinkingBrain().read(platform, question, history=None)
    intention_obj = IntentionAgent().understand(platform, question, history=None, summary=summary)
    intention = {
        "intent": getattr(intention_obj, "intent", None),
        "filters": dict(getattr(intention_obj, "filters", None) or {}),
        "group_by": getattr(intention_obj, "group_by", None),
        "analysis": bool(getattr(intention_obj, "analysis", False)),
        "suggest": bool(getattr(intention_obj, "suggest", False)),
        "limit": getattr(intention_obj, "limit", None),
    }
    thinking = str(getattr(summary, "text", "") or getattr(intention_obj, "thinking", "") or "")

    payload = {
        "question": question,
        "intent": intention["intent"] or routes[0],
        "filters": {k: str(v) for k, v in (intention["filters"] or {}).items() if v is not None},
        "group_by": intention["group_by"],
        "limit": intention["limit"],
        "suggest": intention["suggest"],
        "analysis": intention["analysis"],
        "graph_id": "ecom_products",
    }
    # Prefer live KG HTTP so eval matches production path.
    retrieve = _http_json(f"{KG_URL}/brain/retrieve", payload, timeout=90)
    ask = None
    if full_ask:
        ask = _http_json(f"{FB_URL}/ask", {"question": question}, timeout=180)

    row = _score_row(
        qid=qid,
        section=section,
        question=question,
        plan=plan,
        routes=routes,
        mode=mode,
        thinking=thinking,
        intention=intention,
        retrieve=retrieve,
        ask=ask,
    )
    row["elapsed_s"] = round(time.perf_counter() - t0, 3)
    row["variant"] = "canonical" if not qid.endswith("+") else "paraphrase"
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(ROOT / "questions.txt"))
    parser.add_argument("--ids", default="", help="Comma list like D3,D10,G3")
    parser.add_argument("--full-ask", action="store_true")
    parser.add_argument("--no-paraphrase", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    from knowledge_graph.access import Principal
    from knowledge_graph.config import load_settings
    from knowledge_graph.pipeline.service import Platform

    settings = load_settings()
    platform = Platform(settings)
    principal = Principal(subject="eval", clearance="internal")

    cases = _parse_questions(Path(args.questions))
    if args.ids:
        want = {x.strip() for x in args.ids.split(",") if x.strip()}
        cases = [c for c in cases if c["id"] in want]
    if args.limit:
        cases = cases[: args.limit]

    work: list[dict[str, str]] = []
    for case in cases:
        work.append(case)
        if not args.no_paraphrase:
            for i, para in enumerate(_paraphrases(case["id"], case["question"]), start=1):
                work.append(
                    {
                        "id": f"{case['id']}+{i}",
                        "section": case["section"],
                        "question": para,
                    }
                )

    print(f"Evaluating {len(work)} prompts ({len(cases)} canonical + paraphrases)…")
    rows: list[dict] = []
    for idx, case in enumerate(work, start=1):
        print(f"[{idx}/{len(work)}] {case['id']}: {case['question'][:70]}")
        try:
            row = _run_one(
                platform,
                principal,
                case["id"],
                case["section"],
                case["question"],
                full_ask=bool(args.full_ask) and "+" not in case["id"],
            )
        except Exception as exc:  # noqa: BLE001
            row = {
                "id": case["id"],
                "section": case["section"],
                "question": case["question"],
                "status": "FAIL",
                "notes": [f"exception={exc}"],
                "error": str(exc),
            }
        rows.append(row)
        print(f"    → {row.get('status')} {row.get('notes')}")

    summary = {
        "PASS": sum(1 for r in rows if r.get("status") == "PASS"),
        "PARTIAL": sum(1 for r in rows if r.get("status") == "PARTIAL"),
        "FAIL": sum(1 for r in rows if r.get("status") == "FAIL"),
        "total": len(rows),
    }
    by_section: dict[str, dict[str, int]] = {}
    for r in rows:
        sec = str(r.get("section") or "?")
        by_section.setdefault(sec, {"PASS": 0, "PARTIAL": 0, "FAIL": 0})
        by_section[sec][str(r.get("status"))] = by_section[sec].get(str(r.get("status")), 0) + 1

    failures = [r for r in rows if r.get("status") == "FAIL"]
    partials = [r for r in rows if r.get("status") == "PARTIAL"]
    report = {
        "summary": summary,
        "by_section": by_section,
        "failures": [{"id": r["id"], "question": r["question"], "notes": r.get("notes"), "plan_filters": r.get("plan_filters"), "retrieve_preview": r.get("retrieve_preview")} for r in failures],
        "partials": [{"id": r["id"], "question": r["question"], "notes": r.get("notes")} for r in partials],
        "rows": rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("by_section", json.dumps(by_section, indent=2))
    print(f"Wrote {OUT_JSON}")
    return 0 if summary["FAIL"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
