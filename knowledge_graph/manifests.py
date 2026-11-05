"""Developer helper that fills manifest hashes. Ingestion never calls this."""

from __future__ import annotations

import json
from pathlib import Path

from knowledge_graph.adapters.products import load_products_from_file, product_hash
from knowledge_graph.hashing import sha256_bytes


def refresh_manifest_hashes(raw_root: Path) -> None:
    for manifest_path in raw_root.glob("*/manifest.json"):
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_root = manifest_path.parent
        for entry in data["records"]:
            path = source_root / entry["path"]
            if entry["hash_method"] == "file":
                entry["content_hash"] = sha256_bytes(path.read_bytes())
            elif entry["hash_method"] == "canonical_product":
                products = load_products_from_file(path)
                product = next(item for item in products if item["product_id"] == entry["record_id"])
                entry["content_hash"] = product_hash(product)
            else:
                raise ValueError(f"Unknown hash method {entry['hash_method']}")
            entry["ingestion_status"] = "registered"
        manifest_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
