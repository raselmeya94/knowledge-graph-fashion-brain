#!/usr/bin/env python3
"""Conversation-style questionnaire: multi-turn asks, store every turn, score the system.

Runs scripted chat flows through Fashion Brain catalog orchestration with a live
``conversation_state`` (same session memory as a chat id). Each turn is stored;
a final analysis reports pass/fail against expectations.

Usage (repo root):
  $env:PYTHONPATH = "."
  python scripts/eval_conversation_questionnaire.py
  python scripts/eval_conversation_questionnaire.py --script meta_switch
  python scripts/eval_conversation_questionnaire.py --no-llm-response
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data" / "generated" / "conversation_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Expect:
    """Optional checks for one turn."""

    routes_any: list[str] = field(default_factory=list)
    routes_all: list[str] = field(default_factory=list)
    routes_none: list[str] = field(default_factory=list)
    text_contains: list[str] = field(default_factory=list)
    text_regex: str | None = None
    clarification: bool | None = None
    kg_stats: bool | None = None
    last_graph_id: str | None = None
    understanding_pending: str | None = None
    max_seconds: float | None = None
    # Soft turns report failures but do not fail the script (known slow paths).
    soft: bool = False
    notes: str = ""


@dataclass
class Turn:
    user: str
    expect: Expect = field(default_factory=Expect)


@dataclass
class Script:
    id: str
    title: str
    turns: list[Turn]


SCRIPTS: list[Script] = [
    Script(
        id="meta_switch",
        title="Simple totals across catalogs (session sticky + named source)",
        turns=[
            Turn(
                user="How many products are in the PLM data?",
                expect=Expect(
                    routes_all=["kg_stats", "plmex_products"],
                    text_contains=["27,694"],
                    kg_stats=True,
                    clarification=False,
                    last_graph_id="plmex_products",
                    max_seconds=15.0,
                    notes="PLM total from kg_stats",
                ),
            ),
            Turn(
                user="how many products are in e-commerce data?",
                expect=Expect(
                    routes_all=["kg_stats", "ecom_products"],
                    text_contains=["18,183"],
                    kg_stats=True,
                    clarification=False,
                    last_graph_id="ecom_products",
                    max_seconds=20.0,
                    notes="Named e-com must beat sticky PLM; still meta-fast",
                ),
            ),
            Turn(
                user="And how many in PLM again?",
                expect=Expect(
                    routes_all=["kg_stats", "plmex_products"],
                    text_contains=["27,694"],
                    kg_stats=True,
                    clarification=False,
                    last_graph_id="plmex_products",
                    max_seconds=20.0,
                    notes="Switch back to PLM by name",
                ),
            ),
        ],
    ),
    Script(
        id="clarify_primark",
        title="Named analysis → accuracy confirm → yes → full report",
        turns=[
            Turn(
                user="Analyze PRIMARK",
                expect=Expect(
                    routes_any=["clarification"],
                    clarification=True,
                    understanding_pending="final_confirm",
                    text_regex=r"accurate|reply \"?yes\"?",
                    notes="Named brand analysis shows preview + accuracy confirm (not a dimension quiz)",
                ),
            ),
            Turn(
                user="yes",
                expect=Expect(
                    clarification=False,
                    routes_any=["analysis", "analytical_tool", "qu_ready", "breakdown", "product_count"],
                    max_seconds=180.0,
                    soft=True,
                    notes="Yes after final_confirm runs the full Primark analysis",
                ),
            ),
        ],
    ),
    Script(
        id="merchant_dialogue",
        title="Merchant chat: basic → continuous → analysis → follow-ups (9 turns)",
        turns=[
            Turn(
                user="How many products does Primark have?",
                expect=Expect(
                    routes_any=["kg_stats", "product_count", "ecom_products"],
                    clarification=False,
                    text_regex=r"primark|4[,.]?\d{3}|products",
                    max_seconds=45.0,
                    soft=True,
                    notes="T1 basic Primark total",
                ),
            ),
            Turn(
                user="And how many of them are sweaters?",
                expect=Expect(
                    clarification=False,
                    routes_none=["clarification"],
                    text_regex=r"sweater|knit|primark|\d",
                    max_seconds=90.0,
                    soft=True,
                    notes="T2 continuous Primark + sweater; sticky prior catalog",
                ),
            ),
            Turn(
                user="Of those Primark sweaters, how many are red?",
                expect=Expect(
                    clarification=False,
                    routes_none=["clarification", "plmex_products"],
                    text_regex=r"red|sweater|primark|\d",
                    max_seconds=90.0,
                    soft=True,
                    notes="T3 same-catalog Primark sweater + red (no PLM fan-out)",
                ),
            ),
            Turn(
                user="Overall, who has more products — Primark or Zara?",
                expect=Expect(
                    clarification=False,
                    text_regex=r"primark|zara|\d",
                    max_seconds=90.0,
                    soft=True,
                    notes="T4 overall brand compare (not sticky red)",
                ),
            ),
            Turn(
                user="Give me a full analysis of Primark",
                expect=Expect(
                    clarification=True,
                    routes_any=["clarification"],
                    understanding_pending="final_confirm",
                    text_regex=r"accurate|yes|Confirmed so far|Resolved ask",
                    notes="T5 analysis preview + accuracy confirm",
                ),
            ),
            Turn(
                user="yes",
                expect=Expect(
                    clarification=False,
                    routes_any=["analysis", "analytical_tool", "breakdown", "product_count", "qu_ready"],
                    max_seconds=240.0,
                    soft=True,
                    notes="T6 confirm → full Primark analysis",
                ),
            ),
            Turn(
                user="Show Primark category breakdown",
                expect=Expect(
                    clarification=False,
                    routes_none=["clarification"],
                    text_regex=r"categor|primark|\d",
                    max_seconds=120.0,
                    soft=True,
                    notes="T7 Primark category breakdown",
                ),
            ),
            Turn(
                user="What about color breakdown for them?",
                expect=Expect(
                    clarification=False,
                    text_regex=r"colou?r|primark|\d",
                    max_seconds=120.0,
                    soft=True,
                    notes="T8 color follow-up for them = Primark",
                ),
            ),
            Turn(
                user="How many products does Zara have?",
                expect=Expect(
                    clarification=False,
                    text_regex=r"zara|\d",
                    max_seconds=60.0,
                    soft=True,
                    notes="T9 switch party to Zara in same session",
                ),
            ),
        ],
    ),
    Script(
        id="long_story_ecom",
        title="Long narrative that ends in an e-commerce size ask",
        turns=[
            Turn(
                user=(
                    "I've been reviewing our retail channels and the team keeps asking "
                    "about the online catalog size. After looking at warehouse notes and "
                    "some old spreadsheets, what I really need is a clear sense of how "
                    "large the e-commerce product catalog is right now."
                ),
                expect=Expect(
                    routes_any=["kg_stats", "ecom_products", "product_count"],
                    text_regex=r"18[,.]?183|There are .*products",
                    kg_stats=True,
                    clarification=False,
                    last_graph_id="ecom_products",
                    max_seconds=45.0,
                    notes="QU (LLM or heuristic fallback) extracts count+ecom from the story",
                ),
            ),
        ],
    ),
    Script(
        id="followup_same_session",
        title="Count then named brand analysis in the same session",
        turns=[
            Turn(
                user="How many products are in the PLM data?",
                expect=Expect(
                    kg_stats=True,
                    text_contains=["27,694"],
                    last_graph_id="plmex_products",
                    clarification=False,
                ),
            ),
            Turn(
                user="Analyze ZARA",
                expect=Expect(
                    clarification=True,
                    routes_any=["clarification"],
                    understanding_pending="final_confirm",
                    notes="Named analysis after a count — accuracy confirm, do not reuse PLM total",
                ),
            ),
        ],
    ),
]


def _evidence_ids(retrieved) -> list[str]:
    out: list[str] = []
    for item in getattr(retrieved, "evidence", None) or []:
        rid = getattr(item, "record_id", None) or (item.get("record_id") if isinstance(item, dict) else None)
        if rid:
            out.append(str(rid))
    return out


def _run_qu_only_turn(platform, *, question: str, history: list[dict], state: dict) -> dict[str, Any]:
    """Score Query Understanding + session memory without full Neo4j retrieve."""

    from fashion_brain.agents.contracts import ReadyResult
    from fashion_brain.agents.query_understanding import (
        QueryUnderstandingAgent,
        store_understanding,
    )

    t0 = time.perf_counter()
    result = QueryUnderstandingAgent().evaluate(
        platform,
        question,
        history=history,
        conversation_state=state,
    )
    elapsed = round(time.perf_counter() - t0, 3)
    if isinstance(result, ReadyResult):
        validated = result.intent
        state.pop("clarification", None)
        state["validated_intent"] = validated.to_dict()
        store_understanding(
            state,
            validated=validated,
            summary=validated.effective_question or validated.objective,
        )
        text = (
            f"QU ready: depth={validated.analysis_depth} "
            f"group_by={validated.group_by} filters={validated.filters} "
            f"metrics={validated.metrics}"
        )
        return {
            "question": question,
            "elapsed_seconds": elapsed,
            "routes": ["qu_ready", str(validated.analysis_depth or "unset")],
            "text": text,
            "answer": text,
            "clarification": False,
            "evidence_ids": [],
            "intention": {
                "intent": "product_count" if "count" in (validated.metrics or []) else "analysis",
                "filters": dict(validated.filters or {}),
                "group_by": validated.group_by,
                "preferred_graphs": [validated.source] if validated.source else [],
                "retrieve_mode": None,
                "metrics": list(validated.metrics or []),
            },
            "context": {},
            "choice": {},
            "session": {
                "last_graph_id": state.get("last_graph_id"),
                "active_filters": dict(state.get("active_filters") or {}),
                "clarification": dict(state.get("clarification") or {}),
                "validated_intent": dict(state.get("validated_intent") or {}),
                "understanding": dict(state.get("understanding") or {}),
            },
            "response_error": None,
            "timed_out": False,
            "qu_only": True,
        }

    # ClarifyResult
    state["clarification"] = result.state.to_dict()
    store_understanding(
        state,
        clarification=result.state,
        summary=f"Awaiting clarification ({result.slot})",
        pending_slot=result.slot,
    )
    return {
        "question": question,
        "elapsed_seconds": elapsed,
        "routes": ["clarification", result.slot],
        "text": result.question,
        "answer": result.question,
        "clarification": True,
        "evidence_ids": [],
        "intention": {"intent": "clarification", "filters": {}, "group_by": None, "preferred_graphs": [], "retrieve_mode": None, "metrics": []},
        "context": {},
        "choice": {},
        "session": {
            "last_graph_id": state.get("last_graph_id"),
            "active_filters": dict(state.get("active_filters") or {}),
            "clarification": dict(state.get("clarification") or {}),
            "validated_intent": dict(state.get("validated_intent") or {}),
            "understanding": dict(state.get("understanding") or {}),
        },
        "response_error": None,
        "timed_out": False,
        "qu_only": True,
    }


def _run_turn(
    platform,
    principal,
    *,
    question: str,
    history: list[dict],
    state: dict,
    compose_response: bool,
    qu_only: bool = False,
) -> dict[str, Any]:
    from fashion_brain.agents.query_understanding import clarification_awaiting
    from fashion_brain.orchestration.catalog_pipeline import draft_catalog_orchestrated

    if qu_only:
        return _run_qu_only_turn(platform, question=question, history=history, state=state)

    t0 = time.perf_counter()
    retrieved, routes, intention, context, choice = draft_catalog_orchestrated(
        platform,
        question,
        principal,
        history=history,
        conversation_state=state,
    )
    text = str(getattr(retrieved, "text", "") or "")
    clarifying = clarification_awaiting(retrieved)
    answer_text = text
    if compose_response and not clarifying:
        try:
            from fashion_brain.agents import ResponseAgent

            answer_text = ResponseAgent().compose(platform, question, retrieved) or text
        except Exception as exc:
            answer_text = text
            response_error = f"{type(exc).__name__}: {exc}"
        else:
            response_error = None
    else:
        response_error = None
    elapsed = round(time.perf_counter() - t0, 3)
    understanding = dict(state.get("understanding") or {})
    return {
        "question": question,
        "elapsed_seconds": elapsed,
        "routes": list(routes or []),
        "text": text,
        "answer": answer_text,
        "clarification": clarifying,
        "evidence_ids": _evidence_ids(retrieved),
        "intention": {
            "intent": getattr(intention, "intent", None),
            "filters": dict(getattr(intention, "filters", None) or {}),
            "group_by": getattr(intention, "group_by", None),
            "preferred_graphs": list(getattr(intention, "preferred_graphs", None) or []),
            "retrieve_mode": getattr(intention, "retrieve_mode", None),
            "metrics": list(getattr(intention, "metrics", None) or []),
        },
        "context": {
            "decision": getattr(context, "decision", None),
            "is_contextual": bool(getattr(context, "is_contextual", False)),
        },
        "choice": {
            "graph": getattr(choice, "graph", None) if choice else None,
            "selected": list(getattr(choice, "selected_graphs", None) or []) if choice else [],
        },
        "session": {
            "last_graph_id": state.get("last_graph_id"),
            "active_filters": dict(state.get("active_filters") or {}),
            "clarification": dict(state.get("clarification") or {}),
            "validated_intent": dict(state.get("validated_intent") or {}),
            "understanding": understanding,
        },
        "response_error": response_error,
        "timed_out": False,
    }


def _write_dialogue_transcript(script: Script, turns_out: list[dict[str, Any]]) -> Path:
    """Human-readable User / Assistant dialogue with stored answers."""

    lines = [
        f"# {script.title}",
        "",
        f"Script id: `{script.id}`",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "---",
        "",
    ]
    for row in turns_out:
        idx = row.get("turn")
        q = str(row.get("question") or "").strip()
        a = str(row.get("answer") or row.get("text") or "").strip()
        routes = ", ".join(str(r) for r in (row.get("routes") or []))
        elapsed = row.get("elapsed_seconds")
        status = "PASS" if row.get("passed") else ("SOFT" if row.get("soft") else "FAIL")
        lines.append(f"## Turn {idx} — {status}")
        if row.get("expect_notes"):
            lines.append(f"_{row.get('expect_notes')}_")
        lines.append("")
        lines.append(f"**You:** {q}")
        lines.append("")
        lines.append("**Assistant:**")
        lines.append("")
        lines.append(a if a else "_(empty)_")
        lines.append("")
        lines.append(f"- routes: `{routes}`")
        lines.append(f"- elapsed: `{elapsed}s`")
        if row.get("failures"):
            lines.append(f"- failures: {row.get('failures')}")
        lines.append("")
        lines.append("---")
        lines.append("")
    path = OUT_DIR / f"{script.id}_transcript.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _score_turn(result: dict[str, Any], expect: Expect) -> tuple[bool, list[str]]:
    fails: list[str] = []
    routes = [str(r) for r in (result.get("routes") or [])]
    text = str(result.get("answer") or result.get("text") or "")
    session = result.get("session") or {}
    understanding = session.get("understanding") or {}

    for token in expect.routes_all:
        if token not in routes:
            fails.append(f"missing route '{token}' (got {routes})")
    if expect.routes_any and not any(t in routes for t in expect.routes_any):
        fails.append(f"expected one of routes {expect.routes_any} (got {routes})")
    for token in expect.routes_none:
        if token in routes:
            fails.append(f"unexpected route '{token}'")

    for needle in expect.text_contains:
        if needle.casefold() not in text.casefold():
            fails.append(f"text missing '{needle}'")
    if expect.text_regex and not re.search(expect.text_regex, text, re.I):
        fails.append(f"text regex mismatch /{expect.text_regex}/")

    if expect.clarification is True and not result.get("clarification"):
        fails.append("expected clarification turn")
    if expect.clarification is False and result.get("clarification"):
        fails.append("did not expect clarification")

    if expect.kg_stats is True:
        if "kg_stats" not in routes and "kg_stats" not in (result.get("evidence_ids") or []):
            fails.append("expected kg_stats evidence/route")
    if expect.kg_stats is False and "kg_stats" in routes:
        fails.append("unexpected kg_stats route")

    if expect.last_graph_id and session.get("last_graph_id") != expect.last_graph_id:
        fails.append(
            f"last_graph_id={session.get('last_graph_id')!r} expected {expect.last_graph_id!r}"
        )

    if expect.understanding_pending is not None:
        pending = understanding.get("pending_slot") or (session.get("clarification") or {}).get(
            "pending_slot"
        )
        if pending != expect.understanding_pending:
            fails.append(
                f"pending_slot={pending!r} expected {expect.understanding_pending!r}"
            )

    if expect.max_seconds is not None and float(result.get("elapsed_seconds") or 0) > expect.max_seconds:
        fails.append(
            f"too slow {result.get('elapsed_seconds')}s > {expect.max_seconds}s"
        )
    if result.get("timed_out"):
        fails.append(result.get("response_error") or "turn timed out")

    return (len(fails) == 0), fails


def _run_script(
    platform,
    principal,
    script: Script,
    *,
    compose_response: bool,
) -> dict[str, Any]:
    history: list[dict[str, str]] = []
    state: dict[str, Any] = {}
    turns_out: list[dict[str, Any]] = []
    passed = 0

    for idx, turn in enumerate(script.turns, start=1):
        # Optional QU-only shortcut for scripts that only need understanding readiness.
        qu_only = False
        result = _run_turn(
            platform,
            principal,
            question=turn.user,
            history=list(history),
            state=state,
            compose_response=compose_response,
            qu_only=qu_only,
        )
        ok, fails = _score_turn(result, turn.expect)
        soft = bool(turn.expect.soft)
        # Soft failures are reported but do not fail the script score.
        hard_ok = ok or soft
        record = {
            "turn": idx,
            "expect_notes": turn.expect.notes,
            "passed": ok,
            "soft": soft,
            "hard_ok": hard_ok,
            "failures": fails,
            **result,
        }
        turns_out.append(record)
        if hard_ok:
            passed += 1

        # Persist chat history like a real session.
        history.append({"role": "user", "content": turn.user})
        history.append(
            {
                "role": "assistant",
                "content": str(result.get("answer") or result.get("text") or ""),
            }
        )

        # Store every turn immediately.
        turn_path = OUT_DIR / f"{script.id}_turn_{idx:02d}.json"
        turn_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "script_id": script.id,
        "title": script.title,
        "turns": len(script.turns),
        "passed": passed,
        "failed": len(script.turns) - passed,
        "ok": passed == len(script.turns),
        "final_session": {
            "last_graph_id": state.get("last_graph_id"),
            "understanding": state.get("understanding"),
            "clarification": state.get("clarification"),
            "validated_intent": state.get("validated_intent"),
            "active_filters": state.get("active_filters"),
        },
        "results": turns_out,
    }
    script_path = OUT_DIR / f"{script.id}_session.json"
    script_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_dialogue_transcript(script, turns_out)
    return summary


def _print_analysis(report: dict[str, Any]) -> None:
    print("\n=== Conversation questionnaire analysis ===")
    print(f"Started: {report.get('started_at')}")
    print(f"Scripts: {report['totals']['scripts']}  "
          f"turns: {report['totals']['turns']}  "
          f"passed: {report['totals']['passed']}  "
          f"failed: {report['totals']['failed']}")
    print(f"Overall: {'PASS' if report['totals']['ok'] else 'FAIL'}")
    print(f"Stored under: {OUT_DIR}")
    print()
    for script in report["scripts"]:
        mark = "PASS" if script["ok"] else "FAIL"
        print(f"[{mark}] {script['script_id']} — {script['title']} "
              f"({script['passed']}/{script['turns']})")
        for turn in script["results"]:
            if turn.get("passed"):
                tmark = "ok"
            elif turn.get("soft"):
                tmark = "SOFT"
            else:
                tmark = "FAIL"
            routes = ",".join(turn.get("routes") or [])
            preview = (turn.get("answer") or turn.get("text") or "").replace("\n", " ")[:90]
            print(
                f"  turn {turn['turn']}: {tmark}  {turn['elapsed_seconds']}s  "
                f"routes=[{routes}]  {preview}"
            )
            for fail in turn.get("failures") or []:
                print(f"    - {fail}")
        print()

    # Human-readable analysis file for the latest run.
    analysis_lines = [
        "# Conversation questionnaire analysis",
        "",
        f"- Started: {report.get('started_at')}",
        f"- Finished: {report.get('finished_at')}",
        f"- Result: **{'PASS' if report['totals']['ok'] else 'FAIL'}** "
        f"({report['totals']['passed']}/{report['totals']['turns']} hard-ok turns)",
        f"- Verdict: {report.get('verdict')}",
        "",
        "## Scripts",
        "",
    ]
    for script in report["scripts"]:
        analysis_lines.append(
            f"### {script['script_id']} — {script['title']} "
            f"({'PASS' if script['ok'] else 'FAIL'} {script['passed']}/{script['turns']})"
        )
        analysis_lines.append("")
        for turn in script["results"]:
            status = "pass" if turn.get("passed") else ("soft-fail" if turn.get("soft") else "fail")
            analysis_lines.append(
                f"- Turn {turn['turn']} [{status}] `{turn['question'][:80]}` "
                f"→ routes={turn.get('routes')} ({turn.get('elapsed_seconds')}s)"
            )
            if turn.get("failures"):
                for fail in turn["failures"]:
                    analysis_lines.append(f"  - {fail}")
        analysis_lines.append("")
    analysis_path = OUT_DIR / "questionnaire_analysis_latest.md"
    analysis_path.write_text("\n".join(analysis_lines) + "\n", encoding="utf-8")
    print(f"Analysis: {analysis_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--script",
        action="append",
        dest="scripts",
        help="Run only this script id (repeatable). Default: all.",
    )
    parser.add_argument(
        "--no-llm-response",
        action="store_true",
        help="Skip ResponseAgent compose; score retrieval/clarification text only.",
    )
    parser.add_argument(
        "--fast-fallback",
        action="store_true",
        help=(
            "Force model calls to fail immediately so heuristic / kg_stats paths "
            "are scored without waiting on Ollama timeouts."
        ),
    )
    args = parser.parse_args()

    from knowledge_graph.access import Principal
    from knowledge_graph.config import load_settings
    from knowledge_graph.pipeline.service import Platform

    wanted = set(args.scripts or [])
    scripts = [s for s in SCRIPTS if not wanted or s.id in wanted]
    if not scripts:
        print(f"No matching scripts. Known: {[s.id for s in SCRIPTS]}", file=sys.stderr)
        return 2

    settings = load_settings()
    platform = Platform(settings)
    principal = Principal(subject="conversation-eval", clearance="internal")
    compose = not args.no_llm_response

    # Avoid 12–30s hangs per agent when the local model is unreachable.
    if args.fast_fallback:
        from knowledge_graph.errors import ModelUnavailableError
        from unittest.mock import patch

        class _DeadProvider:
            def complete(self, *a, **k):
                raise ModelUnavailableError("Model provider is unavailable: EvalFastFallback.")

            def chat(self, *a, **k):
                raise ModelUnavailableError("Model provider is unavailable: EvalFastFallback.")

        # Prefer deterministic templates over Text2Cypher schema probes in this mode.
        try:
            object.__setattr__(settings, "text2cypher_enabled", False)
        except Exception:
            try:
                settings.text2cypher_enabled = False  # type: ignore[attr-defined]
            except Exception:
                pass

        _patches = [
            patch("fashion_brain.llm.build_model_provider", return_value=_DeadProvider()),
            patch("fashion_brain.agents.query_understanding.build_model_provider", return_value=_DeadProvider()),
            patch("fashion_brain.agents.conversation_context.build_model_provider", return_value=_DeadProvider()),
            patch("fashion_brain.agents._impl.build_model_provider", return_value=_DeadProvider()),
        ]
        for p in _patches:
            p.start()
        print("fast-fallback: model unavailable + text2cypher off (heuristic/meta/templates)")

    started = datetime.now(timezone.utc).isoformat()
    script_reports: list[dict[str, Any]] = []
    for script in scripts:
        print(f"\n--- Running script: {script.id} ({script.title}) ---")
        report = _run_script(platform, principal, script, compose_response=compose)
        script_reports.append(report)
        print(
            f"  -> {report['passed']}/{report['turns']} passed "
            f"({'PASS' if report['ok'] else 'FAIL'})"
        )

    totals_turns = sum(r["turns"] for r in script_reports)
    totals_passed = sum(r["passed"] for r in script_reports)
    report = {
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "compose_response": compose,
        "model": {
            "provider": getattr(settings, "model_provider", None),
            "name": getattr(settings, "model_name", None),
            "answer_mode": getattr(settings, "answer_mode", None),
        },
        "totals": {
            "scripts": len(script_reports),
            "turns": totals_turns,
            "passed": totals_passed,
            "failed": totals_turns - totals_passed,
            "ok": totals_passed == totals_turns,
        },
        "scripts": script_reports,
        "verdict": (
            "System working for covered conversation flows."
            if totals_passed == totals_turns
            else "System partially failing — see per-turn failures."
        ),
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = OUT_DIR / f"questionnaire_report_{stamp}.json"
    latest = OUT_DIR / "questionnaire_report_latest.json"
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    out_path.write_text(payload, encoding="utf-8")
    latest.write_text(payload, encoding="utf-8")

    _print_analysis(report)
    print(f"Report: {out_path}")
    return 0 if report["totals"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
