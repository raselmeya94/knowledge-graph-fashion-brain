"""Hard-delete ecom_products nodes/edges so a full ecom-sync can rewrite clean names."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from knowledge_graph.config import load_settings
    from knowledge_graph.pipeline.service import Platform

    settings = load_settings()
    platform = Platform(settings)
    gid = "ecom_products"
    print("before", platform.graph.counts())

    batch = 5000
    deleted = 0
    with platform.graph._session() as session:
        while True:
            row = session.run(
                "MATCH (n:Entity {graph:$g}) WITH n LIMIT $limit "
                "DETACH DELETE n RETURN count(*) AS c",
                {"g": gid, "limit": batch},
            ).single()
            c = int(row["c"]) if row else 0
            deleted += c
            print(f"deleted_batch={c} total={deleted}")
            if c == 0:
                break

    state = Path(settings.generated) / "ecom_sync_state.json"
    if state.is_file():
        state.unlink()
        print("removed", state)

    print("after", platform.graph.counts())
    closer = getattr(platform.graph, "close", None)
    if callable(closer):
        closer()
    print("wipe_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
