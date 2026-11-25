"""Refresh merchant_dialogue_qa.json and print a short scoreboard."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SESSION = ROOT / "data" / "generated" / "conversation_eval" / "merchant_dialogue_session.json"
QA = ROOT / "data" / "generated" / "conversation_eval" / "merchant_dialogue_qa.json"


def main() -> None:
    data = json.loads(SESSION.read_text(encoding="utf-8"))
    qa = []
    for row in data.get("results") or []:
        qa.append(
            {
                "turn": row.get("turn"),
                "kind": (row.get("expect_notes") or "").split(" ", 1)[0],
                "question": row.get("question"),
                "answer": row.get("answer") or row.get("text"),
                "routes": row.get("routes"),
                "clarification": row.get("clarification"),
                "filters": (row.get("intention") or {}).get("filters"),
                "group_by": (row.get("intention") or {}).get("group_by"),
                "elapsed_seconds": row.get("elapsed_seconds"),
                "passed": row.get("passed"),
                "soft": row.get("soft"),
                "failures": row.get("failures"),
            }
        )
    QA.write_text(json.dumps({"script": "merchant_dialogue", "turns": qa}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Harness: {data.get('passed')}/{data.get('turns')} ok={data.get('ok')}")
    for row in qa:
        ans = str(row.get("answer") or "").replace("\n", " ")
        if len(ans) > 110:
            ans = ans[:110] + "…"
        print(
            f"T{row['turn']}: passed={row.get('passed')} soft={row.get('soft')} "
            f"filters={row.get('filters')} group={row.get('group_by')}"
        )
        print(f"     {ans}")


if __name__ == "__main__":
    main()
