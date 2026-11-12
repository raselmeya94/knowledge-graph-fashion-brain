#!/usr/bin/env python3
"""One-off: set price_num / original_price_num on existing Product nodes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="ecom_products")
    parser.add_argument("--batch", type=int, default=2000)
    args = parser.parse_args()

    from knowledge_graph.config import load_settings
    from knowledge_graph.pipeline.service import Platform

    settings = load_settings()
    settings = settings
    platform = Platform(settings)
    session_factory = getattr(platform.graph, "_session", None)
    if not callable(session_factory):
        print("Neo4j session unavailable", file=sys.stderr)
        return 1

    # Stamp price_num_migrated so unparseable rows do not re-enter the batch loop.
    set_query = """
    MATCH (p:Entity:Product {graph:$g})
    WHERE p.price IS NOT NULL AND coalesce(p.price_num_migrated, false) = false
    WITH p LIMIT $batch
    WITH p, toFloat(replace(replace(replace(replace(replace(
      coalesce(p.price, ''), '€', ''), '$', ''), '£', ''), ',', ''), ' ', '')) AS num
    SET p.price_num = num, p.price_num_migrated = true
    RETURN count(p) AS n
    """
    set_orig = """
    MATCH (p:Entity:Product {graph:$g})
    WHERE p.original_price IS NOT NULL AND coalesce(p.original_price_num_migrated, false) = false
    WITH p LIMIT $batch
    WITH p, toFloat(replace(replace(replace(replace(replace(
      coalesce(p.original_price, ''), '€', ''), '$', ''), '£', ''), ',', ''), ' ', '')) AS num
    SET p.original_price_num = num, p.original_price_num_migrated = true
    RETURN count(p) AS n
    """

    total_price = 0
    total_orig = 0
    with session_factory() as session:
        while True:
            row = session.run(set_query, {"g": args.graph, "batch": args.batch}).single()
            n = int(row["n"]) if row else 0
            total_price += n
            print(f"price_num batch={n} cumulative={total_price}")
            if n == 0:
                break
        while True:
            row = session.run(set_orig, {"g": args.graph, "batch": args.batch}).single()
            n = int(row["n"]) if row else 0
            total_orig += n
            print(f"original_price_num batch={n} cumulative={total_orig}")
            if n == 0:
                break
        ok = session.run(
            "MATCH (p:Entity:Product {graph:$g}) WHERE p.price_num IS NOT NULL RETURN count(p) AS n",
            {"g": args.graph},
        ).single()
        print(f"products_with_price_num={ok['n'] if ok else 0}")

    print(f"Done. touched_price={total_price} touched_original={total_orig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
