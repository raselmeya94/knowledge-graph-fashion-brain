"""Fill product manifest hashes after sample files change. Not used during ingestion."""

from pathlib import Path

from knowledge_graph.manifests import refresh_manifest_hashes

if __name__ == "__main__":
    from knowledge_graph.config import DATA_ROOT

    root = DATA_ROOT / "raw"
    refresh_manifest_hashes(root)
    print(f"Updated manifests under {root}")
