"""One snapshot file per source, plus a master file of shared links for queries."""

from __future__ import annotations

import json
from pathlib import Path

from knowledge_graph.logging_config import get_logger

LOGGER = get_logger("knowledge_graph.graph.snapshots")

MASTER = "master"
SHARED = "shared"
# Named graph_snapshots (not "graphs") so it is not confused with the graph/ code package.
SNAPSHOTS_DIR = "graph_snapshots"
_LEGACY_SNAPSHOTS_DIR = "graphs"


def snapshots_root(generated: Path) -> Path:
    modern = generated / SNAPSHOTS_DIR
    legacy = generated / _LEGACY_SNAPSHOTS_DIR
    if modern.is_dir() or not legacy.is_dir():
        return modern
    return legacy


def source_snapshot_path(generated: Path, source_id: str, *, for_write: bool = False) -> Path:
    modern = generated / SNAPSHOTS_DIR / source_id / "graph_snapshot.json"
    if for_write:
        return modern
    legacy = generated / _LEGACY_SNAPSHOTS_DIR / source_id / "graph_snapshot.json"
    if modern.is_file() or not legacy.is_file():
        return modern
    return legacy


def master_snapshot_path(generated: Path, *, for_write: bool = False) -> Path:
    modern = generated / SNAPSHOTS_DIR / MASTER / "graph_snapshot.json"
    if for_write:
        return modern
    legacy = generated / _LEGACY_SNAPSHOTS_DIR / MASTER / "graph_snapshot.json"
    if modern.is_file() or not legacy.is_file():
        return modern
    return legacy


def legacy_snapshot_path(generated: Path) -> Path:
    return generated / "graph_snapshot.json"


def save_source_snapshots(graph, generated: Path, source_ids: list[str]) -> list[Path]:
    """Write only the named source graphs. Other source files stay as they are."""

    if not hasattr(graph, "find_nodes"):
        return []
    written: list[Path] = []
    schema = list(getattr(graph, "applied_schema", []))
    for source_id in source_ids:
        payload = {
            "source_id": source_id,
            "nodes": [node.model_dump() for node in graph.find_nodes(graph=source_id, include_deleted=True)],
            "edges": [edge.model_dump() for edge in graph.find_edges(graph=source_id, include_deleted=True)],
            "schema": schema,
        }
        path = source_snapshot_path(generated, source_id, for_write=True)
        _write(path, payload)
        LOGGER.info("wrote %s snapshot nodes=%s edges=%s", source_id, len(payload["nodes"]), len(payload["edges"]))
        written.append(path)
    return written


def save_master_snapshot(graph, generated: Path) -> Path | None:
    """Shared nodes and cross-source links. Source facts stay in the source files."""

    if not hasattr(graph, "find_nodes"):
        return None
    nodes = [node.model_dump() for node in graph.find_nodes(graph=SHARED, include_deleted=True)]
    edges = [edge.model_dump() for edge in graph.find_edges(graph=SHARED, include_deleted=True)]
    loaded = set(loaded_source_ids(graph))
    missing = set(_source_ids_on_disk(generated)) - loaded
    if missing:
        nodes, edges = _keep_other_sources(generated, nodes, edges, loaded)
    payload = {
        "role": MASTER,
        "sources": _source_ids_on_disk(generated),
        "nodes": nodes,
        "edges": edges,
        "schema": list(getattr(graph, "applied_schema", [])),
    }
    path = master_snapshot_path(generated, for_write=True)
    _write(path, payload)
    LOGGER.info("wrote master snapshot nodes=%s edges=%s sources=%s", len(payload["nodes"]), len(payload["edges"]), payload["sources"])
    return path


def restore_snapshots(graph, generated: Path, source_ids: list[str] | None = None) -> bool:
    """Load source snapshots, then the master links. A named source loads only that file."""

    load = getattr(graph, "load_snapshot", None)
    if load is None:
        return False
    already = set(loaded_source_ids(graph))
    has_shared = _has_shared(graph)
    if source_ids is None and already:
        return True
    paths = []
    for path in _restore_paths(generated, source_ids):
        name = path.parent.name
        if name == MASTER:
            if has_shared:
                continue
        elif name in already:
            continue
        paths.append(path)
    if not paths:
        if source_ids and not any(source in already or source == MASTER for source in source_ids):
            LOGGER.info("no snapshot for %s; starting an empty graph", ", ".join(source_ids))
            return False
        return bool(already) or bool(source_ids)
    merge = getattr(graph, "merge_snapshot", None) or load
    LOGGER.info("restoring graph snapshots files=%s", [_display(path, generated) for path in paths])
    if already or has_shared:
        for path in paths:
            merge(json.loads(path.read_text(encoding="utf-8")))
    else:
        load(json.loads(paths[0].read_text(encoding="utf-8")))
        for path in paths[1:]:
            merge(json.loads(path.read_text(encoding="utf-8")))
    counts = getattr(graph, "counts", None)
    if callable(counts):
        tallies = counts()
        node_n = int(tallies.get("nodes", 0))
        edge_n = int(tallies.get("edges", 0))
    else:
        node_n = len(getattr(graph, "nodes", {}) or {})
        edge_n = len(getattr(graph, "edges", {}) or {})
    LOGGER.info("restored graph snapshot nodes=%s edges=%s", node_n, edge_n)
    return True


def loaded_source_ids(graph) -> list[str]:
    distinct = getattr(graph, "distinct_source_graphs", None)
    if callable(distinct):
        try:
            return list(distinct())
        except Exception:
            pass
    nodes = getattr(graph, "nodes", None)
    if not isinstance(nodes, dict):
        return []
    return sorted({node.graph for node in nodes.values() if node.graph != SHARED})


def _has_shared(graph) -> bool:
    nodes = getattr(graph, "nodes", None)
    if isinstance(nodes, dict):
        return any(node.graph == SHARED for node in nodes.values())
    finder = getattr(graph, "find_nodes", None)
    if not callable(finder):
        return False
    try:
        return bool(finder(graph=SHARED, include_deleted=True))
    except Exception:
        return False


def _source_ids_on_disk(generated: Path) -> list[str]:
    root = snapshots_root(generated)
    if not root.is_dir():
        return []
    return sorted(
        path.parent.name
        for path in root.glob("*/graph_snapshot.json")
        if path.parent.name != MASTER and path.stat().st_size > 0
    )


def _restore_paths(generated: Path, source_ids: list[str] | None = None) -> list[Path]:
    if source_ids is not None:
        return [
            path
            for source_id in source_ids
            if (path := source_snapshot_path(generated, source_id)).is_file() and path.stat().st_size > 0
        ]
    sources = [
        source_snapshot_path(generated, source_id)
        for source_id in _source_ids_on_disk(generated)
    ]
    paths = [path for path in sources if path.is_file() and path.stat().st_size > 0]
    legacy = legacy_snapshot_path(generated)
    if legacy.is_file() and legacy.stat().st_size > 0 and "own_products" not in _source_ids_on_disk(generated):
        paths.insert(0, legacy)
    master = master_snapshot_path(generated)
    if master.is_file() and master.stat().st_size > 0:
        paths.append(master)
    return paths


def _keep_other_sources(generated: Path, nodes: list[dict], edges: list[dict], loaded: set[str]) -> tuple[list[dict], list[dict]]:
    """A news write must not drop product links that were not loaded into memory."""

    path = master_snapshot_path(generated)
    if not path.is_file() or path.stat().st_size == 0:
        return nodes, edges
    existing = json.loads(path.read_text(encoding="utf-8"))
    node_ids = {node["id"] for node in nodes}
    edge_ids = {edge["id"] for edge in edges}
    kept_nodes = [
        raw
        for raw in existing.get("nodes", [])
        if raw.get("id") not in node_ids and raw.get("properties", {}).get("prov_source_id") not in loaded
    ]
    kept_edges = [
        raw
        for raw in existing.get("edges", [])
        if raw.get("id") not in edge_ids and raw.get("properties", {}).get("prov_source_id") not in loaded
    ]
    return [*kept_nodes, *nodes], [*kept_edges, *edges]


def _display(path: Path, generated: Path) -> str:
    try:
        return path.relative_to(generated).as_posix()
    except ValueError:
        return path.name


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
