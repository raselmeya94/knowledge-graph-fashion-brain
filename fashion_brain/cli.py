"""Command line for the chatbot."""

from __future__ import annotations

import argparse
import json
import sys

from fashion_brain.agent import answer_question
from knowledge_graph.access import Principal
from knowledge_graph.config import load_settings
from knowledge_graph.logging_config import configure_logging
from knowledge_graph.pipeline.service import Platform

DEMO_QUESTIONS = [
    "Which materials are used by our demo products?",
    "Which products share a material?",
    "How many synthetic demo products are in the PLM data?",
]


def run_demo(settings) -> int:
    # Demo rebuilds a local sample graph; it does not call the KG HTTP API.
    local = settings.model_copy(update={"kg_url": ""})
    platform = Platform(local)
    reports = platform.rebuild()
    principal = Principal(subject="demo", clearance="internal")
    print(json.dumps([report.model_dump() for report in reports], indent=2))
    answers = []
    for question in DEMO_QUESTIONS:
        answer = answer_question(platform, question, principal)
        answers.append(answer.model_dump())
        print("\nQUESTION:", question)
        print(answer.answer)
    materials = answers[0]
    shared = answers[1]
    ok = (
        "PLMEX-DEMO-001" in materials["answer"]
        and "fashion:material:cotton" in materials["answer"]
        and "PLMEX-DEMO-001" in shared["answer"]
        and "PLMEX-DEMO-005" in shared["answer"]
        and all(not report.errors for report in reports)
    )
    if not ok:
        print("Demo acceptance check failed.", file=sys.stderr)
        return 1
    print("\nDemo acceptance check passed.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fashion_brain")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo")
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--clearance", default="internal")
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)
    settings = load_settings()
    configure_logging(settings.log_level, settings.log_json)
    if args.command == "demo":
        return run_demo(settings)
    if args.command == "ask":
        from fashion_brain.kg_client import uses_kg_api
        from fashion_brain.load import ensure_ask_data

        platform = Platform(settings)
        if uses_kg_api(settings):
            # Evidence comes from the knowledge-graph service. No local snapshot load.
            pass
        else:
            ensure_ask_data(platform, args.question)
        answer = answer_question(platform, args.question, Principal(subject="cli", clearance=args.clearance))
        print(answer.model_dump_json(indent=2))
        return 0
    if args.command == "serve":
        import uvicorn

        from fashion_brain.api.app import create_app
        from fashion_brain.llm import format_active_model

        line = format_active_model(settings)
        print(f"Fashion Brain active model: {line}", flush=True)
        uvicorn.run(
            create_app(),
            host=args.host or settings.api_host,
            port=args.port or settings.api_port,
            timeout_graceful_shutdown=5,
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
