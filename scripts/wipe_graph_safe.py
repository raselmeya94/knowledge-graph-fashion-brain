"""Memory-safe Neo4j wipe: relationships first, then nodes, tiny batches."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def wipe_one(platform, graph_id: str, *, rel_batch: int = 50, node_batch: int = 50) -> dict:
    rel_total = 0
    node_total = 0
    print(f"=== wipe {graph_id} ===", flush=True)

    # Pass 1: delete relationships touching this graph's nodes
    while True:
        with platform.graph._session() as session:
            row = session.run(
                """
                MATCH (n:Entity {graph: $g})-[r]-()
                WITH r LIMIT $limit
                DELETE r
                RETURN count(*) AS c
                """,
                {"g": graph_id, "limit": rel_batch},
            ).single()
            c = int(row["c"]) if row else 0
        rel_total += c
        if rel_total % 2000 == 0 or c == 0:
            print(f"  rels batch={c} total={rel_total}", flush=True)
        if c == 0:
            break

    # Pass 2: delete remaining nodes (should be degree-0 now)
    while True:
        with platform.graph._session() as session:
            row = session.run(
                """
                MATCH (n:Entity {graph: $g})
                WITH n LIMIT $limit
                DELETE n
                RETURN count(*) AS c
                """,
                {"g": graph_id, "limit": node_batch},
            ).single()
            c = int(row["c"]) if row else 0
        node_total += c
        if node_total % 2000 == 0 or c == 0:
            print(f"  nodes batch={c} total={node_total}", flush=True)
        if c == 0:
            break

    return {"graph": graph_id, "rels": rel_total, "nodes": node_total}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graphs",
        nargs="+",
        default=["ecom_products", "plmex_products", "fashion_brain"],
    )
    parser.add_argument("--rel-batch", type=int, default=50)
    parser.add_argument("--node-batch", type=int, default=50)
    args = parser.parse_args()

    from knowledge_graph.config import load_settings
    from knowledge_graph.pipeline.service import Platform

    settings = load_settings()
    platform = Platform(settings)
    try:
        print("before", platform.graph.counts(), flush=True)
        results = []
        for gid in args.graphs:
            for attempt in range(5):
                try:
                    results.append(
                        wipe_one(
                            platform,
                            gid,
                            rel_batch=args.rel_batch,
                            node_batch=args.node_batch,
                        )
                    )
                    break
                except Exception as exc:
                    print(f"attempt {attempt+1} failed for {gid}: {exc}", flush=True)
                    time.sleep(2 + attempt)
                    if attempt == 4:
                        raise
        print("results", results, flush=True)
        print("after", platform.graph.counts(), flush=True)

        # clear sync states
        for name in ("ecom_sync_state.json", "plmex_products_sync_state.json"):
            path = Path(settings.generated) / name
            if path.is_file():
                path.unlink()
                print("removed", path, flush=True)
        return 0
    finally:
        closer = getattr(platform.graph, "close", None)
        if callable(closer):
            closer()


if __name__ == "__main__":
    raise SystemExit(main())
