#!/usr/bin/env python3
"""Run Fashion Brain pattern benchmark (50 Qs) against the live API and validate vs DB.

Uses:
  - HTTP Fashion Brain conversation API (session sticky filters)
  - kg_stats for single-filter / facet oracles
  - Neo4j ``cypher_count_products`` for multi-filter oracles

Usage (repo root, APIs up):
  python scripts/eval_pattern_benchmark_50.py
  python scripts/eval_pattern_benchmark_50.py --limit 10
  python scripts/eval_pattern_benchmark_50.py --ids P01,P11,P21
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
WS_URL = FB_URL.replace("https://", "wss://").replace("http://", "ws://") + "/ws/ask"
QUESTIONS = ROOT / "data" / "generated" / "eval" / "fashion_brain_pattern_benchmark_50.json"
OUT_DIR = ROOT / "data" / "generated" / "eval"
OUT_JSON = OUT_DIR / "pattern_benchmark_50_results.json"
OUT_MD = OUT_DIR / "pattern_benchmark_50_results.md"
OUT_CSV = OUT_DIR / "pattern_benchmark_50_results.csv"

# Follow-ups that need prior turns in the same conversation.
CHAIN_SETUP: dict[str, list[str]] = {
    "P20": ["P11"],  # Primark sweater → of those … red
    "P30": ["P21"],  # full analysis → yes
    "P50": ["P41"],  # generate spec → regenerate for other brand
}


def _http_json(
    url: str,
    payload: dict | None = None,
    *,
    method: str = "POST",
    timeout: float = 600.0,
) -> dict[str, Any]:
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
        return {"error": f"HTTP {exc.code}", "body": body[:1200]}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _extract_int(text: str) -> int | None:
    """Prefer the headline count; ignore competing tops / 'which has N' burnish."""

    raw = text or ""
    # Strip source footer before scanning.
    raw = re.sub(r"(?im)^sources?\s*:.*$", "", raw)
    # Drop Assortment Overview heading only (keep the count body under it).
    raw = re.sub(r"(?im)^#{1,3}\s*assortment overview[^\n]*\n?", "", raw)
    # Cut burnish sections that inject competing tops (Black=5433, Girl=4757).
    raw = re.split(
        r"(?im)^#{0,3}\s*(?:related catalog facts|key metrics|merchandising reading|"
        r"product distribution|data notes)\s*$",
        raw,
        maxsplit=1,
    )[0]
    raw = re.sub(r"(?i)`?top\s*:[^`\n]*`?", "", raw)
    # Drop markdown bold/italic so "There are **2,292** products" matches cleanly.
    raw = re.sub(r"[*_`]+", "", raw)
    # Digits with thousand separators, no trailing comma greed (avoids "4757, Men").
    num = r"(\d{1,3}(?:,\d{3})+|\d+)"
    for pat in (
        rf"\bthere are\s+{num}\b",
        rf"(?m)^[^*\n]*?:\s*{num}\b",
        rf"\b{num}\s+products?\s+(?:with|in|for|under|from)\b",
        rf"products\s*=\s*{num}",
        rf"count\s*[:=]\s*{num}",
        # Generic "N products" last — later burnish lines often say "which has 5,433 products".
        rf"\b{num}\s+(?:products?|sweaters?|listings?|styles?|items?)\b",
    ):
        m = re.search(pat, raw, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1).replace(",", ""))
            except ValueError:
                pass
    # First number in the first sentence (before burnish / meta totals).
    first = (raw.split(".") or [raw])[0]
    nums = re.findall(r"\d{1,3}(?:,\d{3})+|\d+", first)
    if nums:
        try:
            return int(nums[0].replace(",", ""))
        except ValueError:
            pass
    nums = re.findall(r"\d{1,3}(?:,\d{3})+|\d+", raw)
    if not nums:
        return None
    try:
        vals = [int(n.replace(",", "")) for n in nums]
        positive = [v for v in vals if v > 0]
        return positive[0] if positive else vals[0]
    except ValueError:
        return None


def _route_blob(routes: list[Any]) -> str:
    return " ".join(str(r).casefold() for r in (routes or []))


def _evidence_blob(evidence: list[Any]) -> str:
    bits: list[str] = []
    for item in evidence or []:
        if isinstance(item, dict):
            bits.append(str(item.get("record_id") or ""))
            bits.append(str(item.get("excerpt") or "")[:200])
            bits.append(str(item.get("source_path") or item.get("graph") or ""))
        else:
            bits.append(str(getattr(item, "record_id", "") or ""))
            bits.append(str(getattr(item, "excerpt", "") or "")[:200])
    return " ".join(bits).casefold()


def _create_conversation(title: str) -> str | None:
    raw = _http_json(f"{FB_URL}/v1/conversations", {"title": title}, timeout=30)
    if raw.get("error"):
        return None
    return str(raw.get("id") or "") or None


def _empty_chatbot(*, ms: float = 0.0, error: str | None = None) -> dict[str, Any]:
    return {
        "ok": False,
        "answer": "",
        "thinking": "",
        "reasoning": "",
        "status_log": [],
        "references": [],
        "evidence": [],
        "routes": [],
        "caveats": [],
        "unavailable": [],
        "events": [],
        "ms": ms,
        "error": error,
        "chatbot": {
            "thinking": "",
            "reasoning": "",
            "response": "",
            "references": [],
            "status": [],
        },
    }


def _ask_chatbot_ws(
    question: str,
    *,
    conversation_id: str | None = None,
    graph_id: str | None = None,
    timeout: float = 600.0,
) -> dict[str, Any]:
    """Same stream the UI uses: thinking / reasoning / answer / source / done."""
    import websocket

    t0 = time.perf_counter()
    thinking = ""
    reasoning = ""
    answer = ""
    status_log: list[str] = []
    references: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    answer_obj: dict[str, Any] = {}
    err: str | None = None
    done = False

    def _on_message(_ws, message: str) -> None:
        nonlocal thinking, reasoning, answer, answer_obj, done, err
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return
        if data.get("error"):
            err = str(data.get("message") or data.get("error"))
            done = True
            try:
                _ws.close()
            except Exception:  # noqa: BLE001
                pass
            return
        kind = str(data.get("type") or "")
        text = str(data.get("text") or "")
        # Keep a compact event trace (not full answer chunks).
        if kind in {"status", "thinking", "reasoning", "source", "done", "error"}:
            slim = {"type": kind}
            if text and kind != "answer":
                slim["text"] = text if len(text) < 500 else text[:500] + "…"
            if kind == "source" and isinstance(data.get("item"), dict):
                slim["item"] = data.get("item")
            if kind == "done":
                slim["routes"] = list((data.get("answer") or {}).get("routes") or [])
            events.append(slim)
        if kind == "status" and text:
            status_log.append(text)
        elif kind == "thinking" and text:
            thinking += text
        elif kind == "reasoning" and text:
            reasoning += text
        elif kind == "answer":
            if data.get("replace"):
                answer = text
            else:
                answer += text
        elif kind == "source":
            item = data.get("item") or {}
            if isinstance(item, dict) and item:
                references.append(item)
        elif kind == "done":
            answer_obj = data.get("answer") if isinstance(data.get("answer"), dict) else {}
            done_answer = str(answer_obj.get("answer") or "")
            if done_answer and (not answer or len(done_answer) >= len(answer)):
                answer = done_answer
            done_ev = list(answer_obj.get("evidence") or [])
            if not references and done_ev:
                references.extend([x for x in done_ev if isinstance(x, dict)])
            done = True
            try:
                _ws.close()
            except Exception:  # noqa: BLE001
                pass

    def _on_error(_ws, error: Any) -> None:
        nonlocal err
        err = str(error)

    def _on_open(ws) -> None:
        ws.send(
            json.dumps(
                {
                    "question": question,
                    "conversation_id": conversation_id,
                    "graph_id": graph_id,
                }
            )
        )

    import threading

    ws_holder: dict[str, Any] = {"ws": None}

    def _watchdog() -> None:
        time.sleep(timeout)
        if not done:
            nonlocal err
            err = err or f"timeout after {timeout:.0f}s"
            sock = ws_holder.get("ws")
            if sock is not None:
                try:
                    sock.close()
                except Exception:  # noqa: BLE001
                    pass

    try:
        ws = websocket.WebSocketApp(
            WS_URL,
            on_open=_on_open,
            on_message=_on_message,
            on_error=_on_error,
        )
        ws_holder["ws"] = ws
        threading.Thread(target=_watchdog, daemon=True).start()
        ws.run_forever(ping_interval=20, ping_timeout=10)
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"

    ms = round((time.perf_counter() - t0) * 1000.0, 1)
    if not done and not err:
        err = "websocket closed before done"
    evidence = references or list(answer_obj.get("evidence") or [])
    routes = list(answer_obj.get("routes") or [])
    caveats = list(answer_obj.get("caveats") or [])
    unavailable = list(answer_obj.get("unavailable") or [])
    return {
        "ok": bool(answer.strip()) or bool(routes),
        "answer": answer,
        "thinking": thinking,
        "reasoning": reasoning,
        "status_log": status_log,
        "references": evidence,
        "evidence": evidence,
        "routes": routes,
        "caveats": caveats,
        "unavailable": unavailable,
        "events": events,
        "ms": ms,
        "error": err,
        "chatbot": {
            "thinking": thinking,
            "reasoning": reasoning,
            "response": answer,
            "references": evidence,
            "status": status_log,
        },
        "raw_done": answer_obj or None,
    }


def _ask_conversation(conversation_id: str, question: str) -> dict[str, Any]:
    """Prefer chatbot WebSocket stream; fall back to HTTP conversation message."""
    got = _ask_chatbot_ws(question, conversation_id=conversation_id)
    if got.get("ok") or not got.get("error"):
        return got
    # HTTP fallback (no thinking/reasoning panels).
    t0 = time.perf_counter()
    raw = _http_json(
        f"{FB_URL}/v1/conversations/{conversation_id}/messages",
        {"question": question},
        timeout=600,
    )
    ms = round((time.perf_counter() - t0) * 1000.0, 1)
    if raw.get("error"):
        out = _empty_chatbot(ms=ms, error=str(raw.get("error")))
        out["body"] = raw.get("body")
        return out
    answer_obj = raw.get("answer") if isinstance(raw.get("answer"), dict) else {}
    text = str((answer_obj or {}).get("answer") or "")
    evidence = list((answer_obj or {}).get("evidence") or [])
    return {
        "ok": bool(text.strip()),
        "answer": text,
        "thinking": "",
        "reasoning": "",
        "status_log": [],
        "references": evidence,
        "evidence": evidence,
        "routes": list((answer_obj or {}).get("routes") or []),
        "caveats": list((answer_obj or {}).get("caveats") or []),
        "unavailable": list((answer_obj or {}).get("unavailable") or []),
        "events": [{"type": "http_fallback"}],
        "ms": ms,
        "error": None,
        "chatbot": {
            "thinking": "",
            "reasoning": "",
            "response": text,
            "references": evidence,
            "status": ["http_fallback"],
        },
        "raw_done": answer_obj or None,
    }


def _ask_http(question: str) -> dict[str, Any]:
    return _ask_chatbot_ws(question)


def _db_oracle(qid: str, platform) -> dict[str, Any]:
    """Ground-truth facts from kg_stats / Neo4j for the concrete examples."""
    from knowledge_graph.retrieval.ecom import cypher_count_products
    from knowledge_graph.retrieval.kg_stats import (
        answer_simple_count,
        dimension_bucket,
        product_total,
        unique_count,
    )

    oracle: dict[str, Any] = {"qid": qid, "facts": {}, "source": None}

    def meta_count(graph: str, filters: dict | None = None) -> int | None:
        hit = answer_simple_count(graph, filters=filters or {})
        return int(hit["n"]) if hit else None

    def neo_count(graph: str, filters: dict) -> int | None:
        try:
            n = cypher_count_products(platform, graph, filters)
            return int(n) if n is not None else None
        except Exception as exc:  # noqa: BLE001
            oracle["neo_error"] = f"{type(exc).__name__}: {exc}"
            return None

    def neo_group_by(graph: str, field: str, filters: dict | None = None) -> list[dict[str, Any]]:
        from knowledge_graph.retrieval.ecom import cypher_group_by

        try:
            rows = cypher_group_by(platform, graph, field, dict(filters or {})) or []
        except Exception as exc:  # noqa: BLE001
            oracle["neo_error"] = f"{type(exc).__name__}: {exc}"
            return []
        empty = {"", "(empty)", "(unspecified)", "n/a", "na", "none", "null", "-", "—"}
        out: list[dict[str, Any]] = []
        for name, count in rows:
            label = str(name or "").strip()
            if not label or label.casefold() in empty:
                continue
            out.append({"name": label, "count": int(count)})
            if len(out) >= 5:
                break
        return out

    # Concrete example oracles aligned with fashion_brain_pattern_benchmark_50.json
    if qid == "P01":
        oracle.update(source="kg_stats", facts={"n": product_total("plmex_products"), "graph": "plmex_products"})
    elif qid == "P02":
        n = meta_count("ecom_products", {"brand": "primark"}) or meta_count("plmex_products", {"brand": "primark"})
        oracle.update(source="kg_stats", facts={"n": n, "filter": {"brand": "primark"}})
    elif qid == "P03":
        oracle.update(
            source="kg_stats",
            facts={"n": meta_count("plmex_products", {"buyer": "matalan"}), "filter": {"buyer": "matalan"}},
        )
    elif qid == "P04":
        oracle.update(
            source="kg_stats",
            facts={"n": meta_count("plmex_products", {"category": "sweater"}), "filter": {"category": "sweater"}},
        )
    elif qid == "P05":
        oracle.update(source="kg_stats", facts={"n": unique_count("plmex_products", "brand"), "facet": "brand"})
    elif qid == "P06":
        bucket = dimension_bucket("plmex_products", "brand") or dimension_bucket("ecom_products", "brand") or {}
        empty = {"", "(empty)", "(unspecified)", "n/a", "na", "none", "null", "-", "—"}
        ranked = sorted(
            ((str(k), int(v)) for k, v in bucket.items() if str(k).strip().casefold() not in empty and int(v) > 0),
            key=lambda item: (-item[1], item[0].casefold()),
        )[:10]
        oracle.update(source="kg_stats", facts={"rows": [{"name": a, "count": b} for a, b in ranked], "facet": "brand"})
    elif qid == "P07":
        oracle.update(
            source="kg_stats",
            facts={"n": meta_count("plmex_products", {"color": "white"}), "filter": {"color": "white"}},
        )
    elif qid == "P08":
        a = meta_count("ecom_products", {"brand": "primark"}) or meta_count("plmex_products", {"brand": "primark"})
        b = meta_count("ecom_products", {"brand": "zara"}) or meta_count("plmex_products", {"brand": "zara"})
        oracle.update(source="kg_stats", facts={"primark": a, "zara": b, "winner": "primark" if (a or 0) >= (b or 0) else "zara"})
    elif qid == "P09":
        oracle.update(
            source="kg_stats",
            facts={"n": meta_count("plmex_products", {"gender": "women"}), "filter": {"gender": "women"}},
        )
    elif qid == "P10":
        oracle.update(source="kg_stats", facts={"n": 0, "filter": {"brand": "__missing_brand__"}})
    elif qid == "P11":
        n = neo_count("plmex_products", {"brand": "primark", "category": "sweater"})
        oracle.update(source="neo4j", facts={"n": n, "filters": {"brand": "primark", "category": "sweater"}})
    elif qid == "P12":
        n = neo_count("plmex_products", {"brand": "primark", "category": "sweater", "color": "white"})
        oracle.update(
            source="neo4j",
            facts={"n": n, "filters": {"brand": "primark", "category": "sweater", "color": "white"}},
        )
    elif qid == "P13":
        n = neo_count("plmex_products", {"buyer": "matalan", "category": "sweater", "gender": "women"})
        oracle.update(
            source="neo4j",
            facts={"n": n, "filters": {"buyer": "matalan", "category": "sweater", "gender": "women"}},
        )
    elif qid == "P16":
        # availability wording varies; try common values
        n = None
        for val in ("in stock", "instock", "available", "in_stock"):
            n = neo_count("ecom_products", {"brand": "primark", "availability": val})
            if n:
                break
        if n is None:
            n = meta_count("ecom_products", {"brand": "primark"})
            oracle["note"] = "availability multi-filter unavailable; brand total only"
        oracle.update(source="neo4j", facts={"n": n, "filters": {"brand": "primark", "availability": "in stock?"}})
    elif qid == "P18":
        n = neo_count(
            "plmex_products",
            {"brand": "primark", "category": "sweater", "gender": "women", "color": "white"},
        )
        oracle.update(
            source="neo4j",
            facts={
                "n": n,
                "filters": {"brand": "primark", "category": "sweater", "gender": "women", "color": "white"},
            },
        )
    elif qid == "P19":
        a = neo_count("plmex_products", {"brand": "primark", "category": "sweater"})
        b = neo_count("plmex_products", {"brand": "zara", "category": "sweater"})
        oracle.update(source="neo4j", facts={"primark_sweater": a, "zara_sweater": b})
    elif qid == "P20":
        n = neo_count("plmex_products", {"brand": "primark", "category": "sweater", "color": "red"})
        oracle.update(
            source="neo4j",
            facts={"n": n, "filters": {"brand": "primark", "category": "sweater", "color": "red"}},
        )
    elif qid == "P22":
        # Show Primark category breakdown — scoped groups, not global meta tops.
        rows = neo_group_by("ecom_products", "category", {"brand": "primark"})
        n = neo_count("ecom_products", {"brand": "primark"})
        oracle.update(
            source="neo4j",
            facts={
                "n": n,
                "rows": rows,
                "filters": {"brand": "primark"},
                "facet": "category",
                "require_groups": True,
            },
        )
    elif qid == "P24":
        # Which brand has the most sweaters?
        rows = neo_group_by("ecom_products", "brand", {"category": "sweater"})
        if not rows:
            rows = neo_group_by("ecom_products", "brand", {"category": "sweaters"})
        n = neo_count("ecom_products", {"category": "sweater"})
        if n is None:
            n = neo_count("ecom_products", {"category": "sweaters"})
        oracle.update(
            source="neo4j",
            facts={
                "n": n,
                "rows": rows,
                "filters": {"category": "sweater"},
                "facet": "brand",
                "require_groups": True,
            },
        )
    else:
        oracle.update(source="route_check", facts={})
    return oracle


def _validate(row: dict[str, Any], got: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    qid = str(row.get("id") or "")
    category = str(row.get("category") or "")
    expected_source = str(row.get("expected_source") or "")
    answer = str(got.get("answer") or "")
    routes = list(got.get("routes") or [])
    route_s = _route_blob(routes)
    evid_s = _evidence_blob(list(got.get("evidence") or []))
    notes: list[str] = []
    status = "VALID"
    got_n = _extract_int(answer)
    facts = dict(oracle.get("facts") or {})
    low = answer.casefold()

    if got.get("error"):
        return {
            "status": "ERROR",
            "notes": [str(got.get("error"))],
            "got_n": got_n,
            "db_n": facts.get("n"),
            "route_ok": False,
            "db_match": False,
        }

    if not answer.strip():
        notes.append("empty_answer")
        status = "INVALID"

    # Route expectations
    route_ok = True
    if category == "basic_meta" or expected_source == "kg_stats":
        if "kg_stats" not in route_s and "kg_stats" not in evid_s:
            # compare asks may still be ok if numbers match
            if qid not in {"P08"}:
                route_ok = False
                notes.append("expected_kg_stats_route")
    elif category == "multi_filter_neo4j":
        if "kg_stats" in route_s and route_s.count("kg_stats") and len(facts.get("filters") or {}) >= 2:
            # multi-filter must not be answered solely by single-filter meta
            if got_n is not None and facts.get("n") is not None and int(got_n) != int(facts["n"]):
                notes.append("possible_meta_collapse")
        if not any(x in route_s for x in ("product_count", "products", "product_details", "breakdown", "catalog")):
            route_ok = False
            notes.append("missing_catalog_route")
    elif category == "analysis_neo4j":
        if qid == "P30":
            if not answer.strip():
                route_ok = False
                notes.append("confirm_empty")
        elif not any(x in route_s for x in ("analysis", "breakdown", "catalog")) and "clarif" not in route_s:
            # preview/clarify is acceptable for full analysis
            if "yes" not in answer.casefold() and "confirm" not in answer.casefold() and "accurate" not in answer.casefold():
                if len(answer) < 80:
                    route_ok = False
                    notes.append("thin_analysis_answer")
        # False refusals are never VALID when oracle (or routes) expect analysis data.
        if re.search(
            r"(?i)(breakdown\s+is\s+not\s+available|does\s+not\s+provide\s+a\s+breakdown|"
            r"not\s+available.*breakdown|no\s+totals?\s+or\s+breakdown|"
            r"do\s+not\s+have\s+the\s+necessary\s+data)",
            answer or "",
        ):
            notes.append("false_refusal_in_answer")
            status = "INVALID"
    elif category == "price":
        if any(x in route_s for x in ("product_count",)) and not any(
            x in route_s or x in answer.casefold() for x in ("price", "trend", "€", "$", "£", "eur", "gbp", "usd")
        ):
            route_ok = False
            notes.append("price_collapsed_to_count")
        if "price" not in route_s and "price" not in answer.casefold() and "clarif" not in route_s:
            # currency clarify is ok
            if "currency" not in answer.casefold() and "usd" not in answer.casefold() and "gbp" not in answer.casefold():
                notes.append("weak_price_signal")
                if status == "VALID":
                    status = "PARTIAL"
    elif category == "product_spec":
        markers = ("product specification", "specification", "composition", "construction", "material", "care")
        if not any(m in answer.casefold() for m in markers) and "specification" not in route_s:
            notes.append("missing_spec_markers")
            if status == "VALID":
                status = "PARTIAL"

    # Numeric DB match
    db_match: bool | None = None
    db_n = facts.get("n")
    if db_n is not None and category in {"basic_meta", "multi_filter_neo4j"}:
        if got_n is None:
            db_match = False
            notes.append(f"no_number_in_answer want={db_n}")
            status = "INVALID"
        else:
            db_match = int(got_n) == int(db_n)
            notes.append(f"db_n={db_n} answer_n={got_n}")
            if not db_match:
                status = "INVALID"
            else:
                notes.append("db_match")
    elif qid == "P08":
        a, b = facts.get("primark"), facts.get("zara")
        if a is not None and b is not None:
            has_a = str(a) in re.sub(r"[^\d]", "", answer) or a == got_n or _extract_int(answer) in {a, b}
            # softer: both numbers present
            nums = set(int(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", answer))
            db_match = a in nums and b in nums
            notes.append(f"primark={a} zara={b} found_nums={sorted(nums)[:8]}")
            if not db_match:
                status = "PARTIAL" if (a in nums or b in nums) else "INVALID"
    elif qid == "P06":
        rows = list(facts.get("rows") or [])
        hit = 0
        for r in rows[:5]:
            name = str(r.get("name") or "")
            cnt = int(r.get("count") or 0)
            if name and name.casefold() in answer.casefold() and str(cnt) in re.sub(r"[^\d]", " ", answer):
                hit += 1
        db_match = hit >= 3
        notes.append(f"top_rows_matched={hit}/5")
        if not db_match:
            status = "PARTIAL" if hit >= 1 else "INVALID"
    elif qid == "P19":
        a, b = facts.get("primark_sweater"), facts.get("zara_sweater")
        nums = set(int(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", answer))
        if a is not None and b is not None:
            db_match = a in nums and b in nums
            notes.append(f"primark_sweater={a} zara_sweater={b}")
            if not db_match:
                status = "PARTIAL" if (a in nums or b in nums) else "INVALID"
    elif facts.get("require_groups") or (facts.get("rows") and category == "analysis_neo4j"):
        rows = list(facts.get("rows") or [])
        hit = 0.0
        for r in rows[:5]:
            name = str(r.get("name") or "").strip()
            cnt = int(r.get("count") or 0)
            if not name:
                continue
            name_ok = name.casefold() in low
            count_ok = str(cnt) in re.sub(r"[^\d]", " ", answer)
            if name_ok and (count_ok or cnt == 0):
                hit += 1.0
            elif name_ok:
                hit += 0.5
        need = 1.0 if len(rows) == 1 else float(min(2, len(rows)))
        db_match = hit >= need
        notes.append(f"group_rows_matched={hit}/{len(rows[:5])} need>={need}")
        if rows and not db_match:
            status = "PARTIAL" if hit >= 0.5 else "INVALID"
        elif rows and db_match:
            notes.append("group_match")
        elif facts.get("require_groups") and not rows:
            notes.append("oracle_groups_empty")
            if status == "VALID":
                status = "PARTIAL"

    must = str(row.get("must") or "").strip()
    if must and category == "analysis_neo4j" and "clarif" not in route_s:
        # Soft content guard from benchmark JSON (not a free-text match of the whole must).
        if "scoped facet" in must.casefold() or "ranked parties" in must.casefold():
            if not facts.get("rows") and status == "VALID" and "false_refusal" not in " ".join(notes):
                notes.append("must_needs_groups")
        if "not global meta" in must.casefold():
            # Global Clothes=1464 style alone is not enough when scoped groups exist.
            if facts.get("rows") and db_match is False:
                notes.append("must_scoped_distribution_miss")

    if not route_ok and status == "VALID":
        status = "PARTIAL"
        notes.append("route_partial")
    if not route_ok and status == "PARTIAL":
        notes.append("route_weak")

    # Soft pass for clarify-first analysis/price when non-empty
    if status == "INVALID" and ("clarif" in route_s or "currency" in answer.casefold() or "confirm" in answer.casefold()):
        if answer.strip():
            status = "PARTIAL"
            notes.append("clarify_ok")

    # Communication checks (do not only score db_match).
    low = answer.casefold()
    oracle_filters = {
        str(k).casefold(): str(v).casefold()
        for k, v in dict(facts.get("filters") or facts.get("filter") or {}).items()
        if v not in (None, "")
    }
    missing_filters = [
        f"{k}={v}"
        for k, v in oracle_filters.items()
        if v and v not in low and not (v.endswith("s") and v[:-1] in low) and not (v + "s" in low)
    ]
    if missing_filters and category in {"basic_meta", "multi_filter_neo4j"} and "clarif" not in route_s:
        notes.append("missing_filters:" + ",".join(missing_filters))
        if status == "VALID":
            status = "PARTIAL"
    if re.search(r"(?is)^\s*(we\s+found\b|there\s+are\s+[\d,]+)", answer or ""):
        notes.append("robotic_opener")
        if status == "VALID":
            status = "PARTIAL"
    has_footer = bool(re.search(r"(?im)^sources?\s*:", answer or ""))
    if category in {"basic_meta", "multi_filter_neo4j"} and answer.strip() and not has_footer and "clarif" not in route_s:
        notes.append("missing_source_footer")
        if status == "VALID":
            status = "PARTIAL"
    # Source attribution: PLM-only graph must not answer as e-commerce-only.
    expected_graph = str((facts.get("graph") or "")).casefold()
    if expected_graph.startswith("plmex") and "e-commerce" in low and "plm" not in low:
        notes.append("wrong_source_attribution")
        if status == "VALID":
            status = "PARTIAL"

    return {
        "status": status,
        "notes": notes,
        "got_n": got_n,
        "db_n": db_n,
        "route_ok": route_ok,
        "db_match": db_match,
    }


def _write_outputs(report: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Pattern benchmark 50 — answers + DB validation",
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
        f"- pass_rate (VALID): {report['summary']['pass_rate_valid']}%",
        f"- usable_rate (VALID+PARTIAL): {report['summary']['pass_rate_usable']}%",
        "",
        "## By category",
        "",
    ]
    for cat, stats in (report.get("by_category") or {}).items():
        lines.append(
            f"- **{cat}**: {stats['VALID']}/{stats['total']} VALID · "
            f"{stats['PARTIAL']} PARTIAL · {stats['INVALID']} INVALID · {stats['ERROR']} ERROR"
        )
    lines.append("")
    lines.append("## Rows")
    lines.append("")
    for row in report["rows"]:
        lines.append(f"### {row['id']} — {row['validation_status']} · {row['category']}")
        lines.append("")
        lines.append(f"**Pattern:** `{row.get('pattern')}`")
        lines.append("")
        lines.append(f"**Q:** {row['question']}")
        lines.append("")
        lines.append(f"**Routes:** {' · '.join(row.get('routes') or []) or '—'}")
        lines.append("")
        lines.append(f"**DB oracle:** `{json.dumps(row.get('db_oracle') or {}, ensure_ascii=False)[:400]}`")
        lines.append("")
        lines.append(f"**Validation notes:** {', '.join(row.get('validation_notes') or []) or '—'}")
        lines.append("")
        chat = row.get("chatbot") or {}
        thinking = str(chat.get("thinking") or row.get("thinking") or "").strip()
        reasoning = str(chat.get("reasoning") or row.get("reasoning") or "").strip()
        response = str(chat.get("response") or row.get("answer") or "").strip() or "_(empty)_"
        refs = list(chat.get("references") or row.get("references") or row.get("evidence") or [])
        if len(thinking) > 2500:
            thinking = thinking[:2500].rstrip() + "…"
        if len(reasoning) > 2500:
            reasoning = reasoning[:2500].rstrip() + "…"
        if len(response) > 3000:
            response = response[:3000].rstrip() + "…"
        lines.append("#### Thinking")
        lines.append("")
        lines.append(thinking or "_(none)_")
        lines.append("")
        lines.append("#### Reasoning")
        lines.append("")
        lines.append(reasoning or "_(none)_")
        lines.append("")
        lines.append("#### Response")
        lines.append("")
        lines.append(response)
        lines.append("")
        lines.append(f"#### References ({len(refs)})")
        lines.append("")
        if not refs:
            lines.append("_(none)_")
        else:
            for src in refs[:20]:
                title = (src or {}).get("title") or (src or {}).get("record_id") or "source"
                graph = (src or {}).get("graph") or (src or {}).get("source_id") or ""
                excerpt = str((src or {}).get("excerpt") or "").strip()
                bit = f"- **{title}**"
                if graph:
                    bit += f" (`{graph}`)"
                if excerpt:
                    bit += f" — {excerpt[:180]}"
                lines.append(bit)
            if len(refs) > 20:
                lines.append(f"- … +{len(refs) - 20} more")
        lines.append("")
        lines.append("---")
        lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    import csv

    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "id",
                "category",
                "validation_status",
                "elapsed_ms",
                "db_n",
                "answer_n",
                "db_match",
                "route_ok",
                "question",
                "routes",
                "validation_notes",
                "thinking",
                "reasoning",
                "response",
                "references_count",
                "error",
            ]
        )
        for row in report["rows"]:
            chat = row.get("chatbot") or {}
            w.writerow(
                [
                    row.get("id"),
                    row.get("category"),
                    row.get("validation_status"),
                    row.get("elapsed_ms"),
                    row.get("db_n"),
                    row.get("answer_n"),
                    row.get("db_match"),
                    row.get("route_ok"),
                    row.get("question"),
                    "|".join(row.get("routes") or []),
                    "; ".join(row.get("validation_notes") or []),
                    chat.get("thinking") or row.get("thinking") or "",
                    chat.get("reasoning") or row.get("reasoning") or "",
                    chat.get("response") or row.get("answer") or "",
                    len(chat.get("references") or row.get("references") or row.get("evidence") or []),
                    row.get("error") or "",
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", default="")
    parser.add_argument(
        "--category",
        default="",
        help="Run one section: basic_meta | multi_filter_neo4j | analysis_neo4j | price | product_spec",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge this run into existing pattern_benchmark_50_results.json (keep other sections)",
    )
    parser.add_argument("--no-conversation", action="store_true", help="Use POST /ask only (no sticky session)")
    parser.add_argument("--progress-every", type=int, default=5)
    args = parser.parse_args()

    payload = json.loads(args.questions.read_text(encoding="utf-8"))
    rows = list(payload.get("questions") or [])
    want_cat = str(args.category or "").strip()
    if want_cat:
        known = set((payload.get("categories") or {}).keys())
        if known and want_cat not in known:
            print(f"Unknown --category {want_cat!r}. Choose from: {', '.join(sorted(known))}")
            return 2
        rows = [r for r in rows if str(r.get("category") or "") == want_cat]
    want = {x.strip() for x in args.ids.split(",") if x.strip()}
    if want:
        rows = [r for r in rows if str(r.get("id")) in want]
        # include chain setup parents
        extra_ids = set()
        for r in rows:
            for parent in CHAIN_SETUP.get(str(r.get("id")), []):
                extra_ids.add(parent)
        if extra_ids:
            by_id = {str(q["id"]): q for q in payload["questions"]}
            parents = [by_id[i] for i in extra_ids if i in by_id and i not in {str(r["id"]) for r in rows}]
            rows = parents + rows
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        print("No questions selected.")
        return 2

    health = _http_json(f"{FB_URL}/health", method="GET", timeout=10)
    if health.get("error") or health.get("status") != "ok":
        print(f"Fashion Brain health failed: {health}")
        return 2

    from knowledge_graph.access import Principal
    from knowledge_graph.config import load_settings
    from knowledge_graph.pipeline.service import Platform

    platform = Platform(load_settings())
    _ = Principal(subject="pattern_bench", clearance="internal")

    by_id = {str(r["id"]): r for r in payload["questions"]}
    prior_by_id: dict[str, dict[str, Any]] = {}
    if args.merge and OUT_JSON.exists():
        try:
            prior = json.loads(OUT_JSON.read_text(encoding="utf-8"))
            for row in prior.get("rows") or []:
                if isinstance(row, dict) and row.get("id"):
                    prior_by_id[str(row["id"])] = row
            print(f"Merging into existing results ({len(prior_by_id)} prior rows)", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"Could not load prior results for --merge: {exc}", flush=True)
    results: list[dict[str, Any]] = []
    mode = "ws/ask-chatbot" if args.no_conversation else "ws/ask-chatbot+conversation"
    section = f" category={want_cat}" if want_cat else ""
    print(f"Running {len(rows)} questions{section} via {mode} @ {WS_URL}")

    for i, row in enumerate(rows, start=1):
        qid = str(row.get("id") or "")
        question = str(row.get("example") or row.get("question") or "")
        category = str(row.get("category") or "")

        setup_answers: list[dict[str, Any]] = []
        conversation_id = None
        if not args.no_conversation:
            conversation_id = _create_conversation(f"pattern-bench-{qid}")
            # chain setup in same conversation
            for parent_id in CHAIN_SETUP.get(qid, []):
                parent = by_id.get(parent_id)
                if not parent:
                    continue
                pq = str(parent.get("example") or "")
                if conversation_id:
                    setup = _ask_conversation(conversation_id, pq)
                else:
                    setup = _ask_http(pq)
                setup_answers.append(
                    {
                        "id": parent_id,
                        "question": pq,
                        "answer": setup.get("answer"),
                        "thinking": setup.get("thinking"),
                        "reasoning": setup.get("reasoning"),
                        "routes": setup.get("routes"),
                    }
                )
                print(f"  setup {parent_id} → {(setup.get('answer') or '')[:80]!r}", flush=True)

        if conversation_id and not args.no_conversation:
            got = _ask_conversation(conversation_id, question)
        else:
            got = _ask_http(question)

        oracle = _db_oracle(qid, platform)
        verdict = _validate(row, got, oracle)

        record = {
            "id": qid,
            "category": category,
            "pattern": row.get("pattern"),
            "question": question,
            "expected_source": row.get("expected_source"),
            "expected_route": row.get("expected_route"),
            "must": row.get("must"),
            "conversation_id": conversation_id,
            "setup": setup_answers,
            "answer": got.get("answer") or "",
            "thinking": got.get("thinking") or "",
            "reasoning": got.get("reasoning") or "",
            "references": got.get("references") or got.get("evidence") or [],
            "chatbot": got.get("chatbot")
            or {
                "thinking": got.get("thinking") or "",
                "reasoning": got.get("reasoning") or "",
                "response": got.get("answer") or "",
                "references": got.get("references") or got.get("evidence") or [],
                "status": got.get("status_log") or [],
            },
            "status_log": got.get("status_log") or [],
            "events": got.get("events") or [],
            "routes": got.get("routes") or [],
            "evidence": got.get("evidence") or [],
            "caveats": got.get("caveats") or [],
            "unavailable": got.get("unavailable") or [],
            "elapsed_ms": got.get("ms"),
            "error": got.get("error"),
            "db_oracle": oracle,
            "validation_status": verdict["status"],
            "validation_notes": verdict["notes"],
            "answer_n": verdict.get("got_n"),
            "db_n": verdict.get("db_n"),
            "db_match": verdict.get("db_match"),
            "route_ok": verdict.get("route_ok"),
        }
        results.append(record)
        print(
            f"[{i}/{len(rows)}] {qid} {verdict['status']} "
            f"db_n={verdict.get('db_n')} ans_n={verdict.get('got_n')} "
            f"{got.get('ms')}ms :: {question[:70]}",
            flush=True,
        )

        if args.progress_every > 0 and (i % args.progress_every == 0 or i == len(rows)):
            checkpoint_rows = results
            if prior_by_id:
                merged = dict(prior_by_id)
                for row in results:
                    merged[str(row["id"])] = row
                order = [str(q["id"]) for q in payload.get("questions") or []]
                checkpoint_rows = [merged[qid] for qid in order if qid in merged] + [
                    merged[qid] for qid in merged if qid not in set(order)
                ]
            summary = _summary(checkpoint_rows)
            report = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "mode": mode,
                "fb_url": FB_URL,
                "summary": summary,
                "by_category": _by_category(checkpoint_rows),
                "rows": checkpoint_rows,
                "partial": True,
            }
            _write_outputs(report)
            print(f"checkpoint → {OUT_JSON}", flush=True)

    if prior_by_id:
        merged = dict(prior_by_id)
        for row in results:
            merged[str(row["id"])] = row
        # Preserve benchmark order from the questions file.
        order = [str(q["id"]) for q in payload.get("questions") or []]
        results = [merged[i] for i in order if i in merged] + [
            merged[i] for i in merged if i not in set(order)
        ]

    summary = _summary(results)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "fb_url": FB_URL,
        "summary": summary,
        "by_category": _by_category(results),
        "rows": results,
        "partial": bool(want_cat or want) and len(results) < len(payload.get("questions") or []),
    }
    _write_outputs(report)
    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_CSV}")
    return 0 if summary["INVALID"] == 0 and summary["ERROR"] == 0 else 1


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"VALID": 0, "PARTIAL": 0, "INVALID": 0, "ERROR": 0}
    for r in results:
        counts[str(r.get("validation_status") or "ERROR")] = counts.get(str(r.get("validation_status") or "ERROR"), 0) + 1
    total = len(results)
    usable = counts["VALID"] + counts["PARTIAL"]
    return {
        "total": total,
        **counts,
        "pass_rate_valid": round(100.0 * counts["VALID"] / total, 2) if total else 0.0,
        "pass_rate_usable": round(100.0 * usable / total, 2) if total else 0.0,
    }


def _by_category(results: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for r in results:
        cat = str(r.get("category") or "unknown")
        bucket = out.setdefault(cat, {"total": 0, "VALID": 0, "PARTIAL": 0, "INVALID": 0, "ERROR": 0})
        bucket["total"] += 1
        st = str(r.get("validation_status") or "ERROR")
        bucket[st] = bucket.get(st, 0) + 1
    return out


if __name__ == "__main__":
    raise SystemExit(main())
