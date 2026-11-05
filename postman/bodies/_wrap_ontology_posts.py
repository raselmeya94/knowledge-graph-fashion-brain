"""One-shot helper: wrap ontology JSON into POST /ontologies bodies."""

from __future__ import annotations

import json
from pathlib import Path

bodies = Path(__file__).resolve().parent
specs = [
    ("ontology_ecom.json", "ecom", "E-commerce products"),
    ("ontology_product.json", "product", "PLMEX products"),
    ("ontology_news.json", "news", "News and documents"),
    ("ontology_system.json", "system", "System / registry metadata"),
]
for src_name, oid, default_name in specs:
    raw = json.loads((bodies / src_name).read_text(encoding="utf-8"))
    body = {
        "id": oid,
        "name": raw.get("name") or default_name,
        "description": raw.get("description") or "",
        "version": raw.get("version") or "1.0.0",
        "entity_types": raw.get("entity_types") or [],
        "relationship_allowlist": raw.get("relationship_allowlist") or {},
        "canonical_entities": raw.get("canonical_entities") or [],
    }
    out = bodies / f"POST_ontologies_{oid}.json"
    out.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    print(out.name, out.stat().st_size)
