"""Command line for the knowledge-graph construction service."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from knowledge_graph.api import create_app, open_platform
from knowledge_graph.config import load_settings
from knowledge_graph.construction.pipeline import pipeline_for
from knowledge_graph.logging_config import configure_logging


def _print(payload) -> None:
    print(json.dumps(payload, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="knowledge_graph")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Start the knowledge-graph API.")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)

    sub.add_parser("validate", help="Check the product registry, manifest, and ontology.")
    ingest = sub.add_parser("ingest", help="Build one source into the knowledge graph.")
    ingest.add_argument("--own_data", action="store_true", help="PLM product data. Writes the own_products graph.")
    ingest.add_argument("--document", action="store_true", help="PDF, Word, or text. Queues a review.")
    ingest.add_argument("--customer", action="store_true", help="Customer JSON. Queues a review.")
    ingest.add_argument("--email", action="store_true", help="Email file. Queues a review.")
    ingest.add_argument("--file", default=None)
    ingest.add_argument("--text", default=None)
    ingest.add_argument("--model", action="store_true", help="Ask Ollama for additional relationships.")
    ingest.add_argument("--commit", action="store_true", help="Approve and save a document, customer, or email review.")
    ingest.add_argument("--source", action="append", dest="sources")

    sub.add_parser("tools", help="List construction tools.")

    ecom_ping = sub.add_parser("ecom-ping", help="Test e-comm Postgres (master_products counts).")
    ecom_sync = sub.add_parser("ecom-sync", help="Sync master_products → Neo4j.")
    ecom_sync.add_argument("--brand", default=None, help="Only one brand, e.g. primark.")
    ecom_sync.add_argument("--full", action="store_true", help="Ignore watermark; sync all matching rows.")
    ecom_sync.add_argument("--limit", type=int, default=None, help="Max rows for this run.")

    pipeline = sub.add_parser("pipeline", help="Extract one source and queue a review.")
    pipeline.add_argument("--kind", default=None)
    pipeline.add_argument("--file", default=None)
    pipeline.add_argument("--text", default=None)
    pipeline.add_argument("--model", action="store_true", help="Ask Ollama for additional relationships.")
    pipeline.add_argument("--commit", action="store_true", help="Approve and save the proposals.")

    sub.add_parser("reviews", help="List queued reviews.")

    approve = sub.add_parser("approve", help="Approve every pending relationship in one review.")
    approve.add_argument("review_id")

    commit = sub.add_parser("commit", help="Write approved relationships into the graph.")
    commit.add_argument("review_id")

    args = parser.parse_args(argv)
    settings = load_settings()
    configure_logging(settings.log_level, settings.log_json)

    if args.command == "serve":
        import uvicorn

        # timeout_graceful_shutdown: one Ctrl+C must exit even if a job thread is mid-ingest.
        uvicorn.run(
            create_app(),
            host=args.host or settings.kg_api_host,
            port=args.port or settings.kg_api_port,
            timeout_graceful_shutdown=5,
        )
        return 0

    if args.command == "validate":
        errors = validate_raw(settings)
        if errors:
            _print(errors)
            return 1
        print("Product data, registry, and ontology validated.")
        return 0
    if args.command == "ecom-ping":
        from knowledge_graph.pipeline.ecom_sync import ping_ecom_db

        try:
            _print(ping_ecom_db(settings))
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 1
        return 0
    if args.command == "ecom-sync":
        from knowledge_graph.pipeline.ecom_sync import sync_ecom_to_neo4j
        from knowledge_graph.pipeline.service import Platform

        platform = Platform(settings)
        try:
            report = sync_ecom_to_neo4j(
                platform,
                brand=args.brand,
                full=bool(args.full),
                limit=args.limit,
            )
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 1
        finally:
            closer = getattr(platform.graph, "close", None)
            if callable(closer):
                closer()
        _print(report)
        return 0 if report.get("ok") else 1
    if args.command == "ingest":
        return _ingest(settings, args)
    if args.command == "commit":
        return _commit(settings, args.review_id)

    load_sources = _construction_sources(args) if args.command == "pipeline" else None
    platform = open_platform(
        settings,
        load_snapshot=args.command in {"pipeline", "serve"},
        sources=load_sources,
    )
    construction = pipeline_for(platform)
    if args.command == "tools":
        _print(construction.list_tools())
        return 0
    if args.command == "reviews":
        _print([item.model_dump() for item in construction.store.list()])
        return 0
    if args.command == "pipeline":
        payload = _pipeline_payload(args)
        _print(construction.run(payload))
        return 0
    if args.command == "approve":
        _print(construction.run_tool("update_review", {"review_id": args.review_id, "decision": "approved"}))
        return 0
    return 2


def validate_raw(settings) -> list[str]:
    import knowledge_graph.adapters  # noqa: F401

    from knowledge_graph.adapters.base import create_adapter
    from knowledge_graph.ontology import load_ontology
    from knowledge_graph.registry import load_registry

    registry = load_registry(settings.raw_dir / "source_registry.json")
    ontology = load_ontology(settings.ontology_file)
    errors: list[str] = []
    for source in registry.enabled_sources():
        try:
            adapter = create_adapter(source, settings.raw_dir / source.source_id)
        except Exception as exc:
            errors.append(str(exc))
            continue
        binder = getattr(adapter, "bind_runtime", None)
        if binder:
            binder(ontology, "deterministic", None, settings.request_timeout_seconds)
        discovered = adapter.discover()
        errors.extend(discovered.errors)
        for record in discovered.records:
            for message in adapter.validate(record):
                errors.append(f"{source.source_id}:{record.record_id}: {message}")
    return errors


def _ingest(settings, args) -> int:
    selected = [name for name in ("own_data", "document", "customer", "email") if getattr(args, name)]
    if len(selected) > 1:
        print("Choose one source: --own_data, --document, --customer, or --email.", file=sys.stderr)
        return 2
    if args.own_data or not selected:
        from knowledge_graph.pipeline.service import Platform

        sources = ["own_products"] if args.own_data else args.sources
        reports = Platform(settings).rebuild(sources)
        _print([report.model_dump() for report in reports])
        return 1 if any(report.errors for report in reports) else 0
    platform = open_platform(settings, load_snapshot=True, sources=_construction_sources(args))
    payload = _pipeline_payload(args)
    if args.customer:
        payload["kind"] = "json"
    elif args.email:
        payload["kind"] = "email"
    _print(pipeline_for(platform).run(payload))
    return 0


def _commit(settings, review_id: str) -> int:
    from knowledge_graph.construction.identity import NEWS_KINDS
    from knowledge_graph.pipeline.service import Platform

    platform = Platform(settings)
    construction = pipeline_for(platform)
    record = construction.store.get(review_id)
    source_id = "news"
    platform._restore_memory_snapshot([source_id])
    _print(construction.run_tool("ingest_graph", {"review_id": review_id}))
    return 0


def _construction_sources(args) -> list[str]:
    if getattr(args, "own_data", False):
        return ["own_products"]
    return ["news"]


def _pipeline_payload(args) -> dict:
    payload = {
        "kind": getattr(args, "kind", None),
        "use_model": bool(getattr(args, "model", False)),
        "auto_commit": bool(getattr(args, "commit", False)),
    }
    if args.file:
        path = Path(args.file)
        if not path.is_file():
            print(f"File not found: {path}", file=sys.stderr)
            raise SystemExit(1)
        payload["filename"] = path.name
        payload["data"] = path.read_bytes()
    if args.text is not None:
        payload["text"] = args.text
    if "data" not in payload and "text" not in payload:
        print("Provide --text or --file.", file=sys.stderr)
        raise SystemExit(1)
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
