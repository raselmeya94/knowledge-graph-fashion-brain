"""Export session Q&A + non-duplicate eval benchmark questions to Excel."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "generated" / "fashion_brain" / "app.db"
DEFAULT_BENCH = ROOT / "data" / "generated" / "eval" / "fashion_brain_pattern_benchmark_50.json"
DEFAULT_RESULTS = ROOT / "data" / "generated" / "eval" / "pattern_benchmark_50_results.json"
DEFAULT_OUT = ROOT / "data" / "generated" / "eval" / "session_qa_merged_benchmark.xlsx"


def _norm_q(text: str) -> str:
    t = (text or "").strip().casefold()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[?!.]+$", "", t).strip()
    return t


def load_session_pairs(db_path: Path, session_id: str) -> list[dict]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    exists = con.execute("SELECT 1 FROM conversations WHERE id=?", (session_id,)).fetchone()
    if not exists:
        con.close()
        raise SystemExit(f"Session not found: {session_id}")

    msgs = con.execute(
        """
        SELECT role, content, created_at
        FROM messages
        WHERE conversation_id=?
        ORDER BY created_at, rowid
        """,
        (session_id,),
    ).fetchall()
    con.close()

    pairs: list[dict] = []
    pending_q: str | None = None
    for m in msgs:
        role = (m["role"] or "").strip().lower()
        content = (m["content"] or "").strip()
        if not content:
            continue
        if role == "user":
            pending_q = content
        elif role == "assistant" and pending_q is not None:
            pairs.append(
                {
                    "question": pending_q,
                    "answer": content,
                    "rating": "",
                    "source": "session",
                }
            )
            pending_q = None
    return pairs


def load_benchmark_pairs(bench_path: Path, results_path: Path) -> list[dict]:
    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    by_id = {r.get("id"): r for r in (results.get("rows") or []) if r.get("id")}

    rows: list[dict] = []
    for q in bench.get("questions") or []:
        qid = q.get("id") or ""
        question = (q.get("example") or q.get("pattern") or "").strip()
        if not question:
            continue
        result = by_id.get(qid) or {}
        answer = (result.get("response") or result.get("answer") or "").strip()
        rows.append(
            {
                "question": question,
                "answer": answer,
                "rating": "",
                "source": f"benchmark:{qid}",
                "benchmark_id": qid,
                "category": q.get("category") or "",
            }
        )
    return rows


def merge_unique(session_rows: list[dict], bench_rows: list[dict]) -> list[dict]:
    seen = {_norm_q(r["question"]) for r in session_rows}
    out = [
        {"question": r["question"], "answer": r["answer"], "rating": r.get("rating", "")}
        for r in session_rows
    ]
    for r in bench_rows:
        key = _norm_q(r["question"])
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "question": r["question"],
                "answer": r["answer"],
                "rating": r.get("rating", ""),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--benchmark", type=Path, default=DEFAULT_BENCH)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    session_rows = load_session_pairs(args.db, args.session_id)
    bench_rows = load_benchmark_pairs(args.benchmark, args.results)
    merged = merge_unique(session_rows, bench_rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(merged, columns=["question", "answer", "rating"])
    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="qa", index=False)
        # Extra trace sheet for audit
        session_df = pd.DataFrame(
            [
                {
                    "question": r["question"],
                    "answer": r["answer"],
                    "rating": "",
                    "source": "session",
                }
                for r in session_rows
            ]
        )
        added = [
            {
                "question": r["question"],
                "answer": r["answer"],
                "rating": "",
                "source": r["source"],
                "benchmark_id": r.get("benchmark_id", ""),
                "category": r.get("category", ""),
            }
            for r in bench_rows
            if _norm_q(r["question"])
            not in {_norm_q(s["question"]) for s in session_rows}
        ]
        pd.DataFrame(session_df).to_excel(writer, sheet_name="session_only", index=False)
        pd.DataFrame(added).to_excel(writer, sheet_name="benchmark_added", index=False)

    print(f"session_pairs={len(session_rows)}")
    print(f"benchmark_total={len(bench_rows)}")
    print(f"benchmark_added={len(added)}")
    print(f"final_rows={len(merged)}")
    print(f"wrote={args.out}")


if __name__ == "__main__":
    main()
