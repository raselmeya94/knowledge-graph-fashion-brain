"""First-time knowledge-graph bootstrap (ontology + graph + source + mapping).

Idempotent: already-registered pieces are reused. Optional sync kicks off after scaffold.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from knowledge_graph.config import DATA_ROOT, Settings
from knowledge_graph.errors import NotFoundError, ValidationFailed
from knowledge_graph.kg_registry.models import GraphCreate, OntologyRecord
from knowledge_graph.logging_config import get_logger
from knowledge_graph.models import SourceConfig, SourceRegistry
from knowledge_graph.ontology import load_ontology
from knowledge_graph.registry import load_registry, validate_registry
from knowledge_graph.sources.registry_ops import SourceRegisterBody, register_source

LOGGER = get_logger("knowledge_graph.kg_build")

ECOM_PROFILE = "ecom_products"
MAPPING_NAME = "ecom_master_products_mapping.json"


def build_new_kg(
    platform,
    *,
    profile: str = ECOM_PROFILE,
    sync: bool = False,
    full: bool = False,
    brand: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Scaffold a product KG for first use. Does not wipe existing Neo4j data."""

    profile_id = (profile or ECOM_PROFILE).strip().casefold()
    if profile_id != ECOM_PROFILE:
        raise ValidationFailed(
            f"Unsupported profile '{profile}'. Currently only '{ECOM_PROFILE}' is supported."
        )

    steps: dict[str, Any] = {}
    settings: Settings = platform.settings

    steps["mapping"] = ensure_ecom_mapping(settings)
    steps["ontology"] = ensure_ecom_ontology(platform)
    steps["graph"] = ensure_ecom_graph(platform)
    steps["source"] = ensure_ecom_source(platform)

    from knowledge_graph.retrieval.schema_hooks import on_schema_changed

    steps["schema_hooks"] = on_schema_changed(
        platform,
        graph_id="ecom_products",
        ontology_id="ecom",
    )

    report: dict[str, Any] = {
        "ok": True,
        "profile": ECOM_PROFILE,
        "graph_id": "ecom_products",
        "ontology_id": "ecom",
        "source_id": "ecom_products",
        "table": "master_products",
        "database": settings.postgres_db,
        "steps": steps,
        "sync_requested": bool(sync),
    }

    if sync:
        from knowledge_graph.pipeline.ecom_sync import sync_ecom_to_neo4j

        LOGGER.info("kg build: starting ecom sync brand=%s full=%s limit=%s", brand, full, limit)
        report["sync"] = sync_ecom_to_neo4j(
            platform,
            brand=brand,
            full=full,
            limit=limit,
            refresh_generated=True,
        )
        report["ok"] = bool(report["sync"].get("ok", False)) if isinstance(report["sync"], dict) else True
    else:
        report["next"] = {
            "db_check": "GET /api/db/connection?table=master_products",
            "sync": "POST /api/sources/ecom_products/sync",
            "or_rebuild": "POST /api/kg/build with {\"sync\": true}",
        }
    return report


def ensure_ecom_mapping(settings: Settings) -> dict[str, Any]:
    """Ensure data/schemas/ecom_master_products_mapping.json exists (required by sync)."""

    target = settings.path(Path("schemas") / MAPPING_NAME)
    if target.is_file():
        return {"status": "exists", "path": str(target)}

    candidates = [
        DATA_ROOT / "schemas" / MAPPING_NAME,
        Path(__file__).resolve().parents[2] / "postman" / "bodies" / MAPPING_NAME,
        *_backup_mapping_candidates(),
    ]
    source = next((path for path in candidates if path.is_file() and path.resolve() != target.resolve()), None)
    if source is None:
        raise ValidationFailed(
            f"Missing {MAPPING_NAME}. Place it under data/schemas/ "
            f"(or restore from postman/bodies/{MAPPING_NAME})."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    LOGGER.info("kg build: restored mapping from %s -> %s", source, target)
    return {"status": "restored", "path": str(target), "from": str(source)}


def ensure_ecom_ontology(platform) -> dict[str, Any]:
    """Register ecom ontology from disk if needed."""

    from knowledge_graph.kg_registry.models import OntologyCreate
    from knowledge_graph.kg_registry.store import ontology_registry_path

    registry = platform.kg_registry
    settings: Settings = platform.settings
    target = settings.path(Path("schemas/ontologies/ecom.json"))

    try:
        record = registry.get_ontology("ecom")
        path = registry.ontology_file("ecom")
        if path.is_file():
            load_ontology(path)
            return {"status": "exists", "id": record.id, "path": str(path)}
    except Exception:
        pass

    seed = target if target.is_file() else (
        Path(__file__).resolve().parents[2] / "postman" / "bodies" / "POST_ontologies_ecom.json"
    )
    if not seed.is_file():
        raise ValidationFailed(
            "E-com ontology file not found. POST /ontologies first or restore "
            "data/schemas/ontologies/ecom.json."
        )

    data = json.loads(seed.read_text(encoding="utf-8"))
    if "entity_types" not in data and isinstance(data.get("definition"), dict):
        data = data["definition"]

    # Prefer registering an already-written ontology file without rewriting it.
    if target.is_file():
        ontology = load_ontology(target)
        record = OntologyRecord(
            id="ecom",
            name=ontology.name or "ecom-products-ontology",
            description=ontology.description or "",
            path=ontology_registry_path(settings, "ecom", target),
            builtin=False,
        )
        bag = registry.load()
        bag.ontologies = [item for item in bag.ontologies if item.id != "ecom"]
        bag.ontologies.append(record)
        registry.save(bag)
        return {"status": "registered", "id": "ecom", "path": str(target)}

    body = OntologyCreate.model_validate(
        {
            "id": data.get("id") or "ecom",
            "name": data.get("name") or "ecom-products-ontology",
            "description": data.get("description") or "",
            "version": data.get("version") or "1.2.0",
            "entity_types": data.get("entity_types") or [],
            "relationship_allowlist": data.get("relationship_allowlist") or {},
            "canonical_entities": data.get("canonical_entities") or [],
        }
    )
    try:
        record, _ontology = registry.create_ontology(body)
        return {"status": "created", "id": record.id, "path": record.path}
    except ValidationFailed as exc:
        if "already exists" not in str(exc).lower():
            raise
        record = registry.get_ontology("ecom")
        return {"status": "exists", "id": record.id, "path": record.path}

def ensure_ecom_graph(platform) -> dict[str, Any]:
    try:
        record = platform.kg_registry.get_graph("ecom_products")
        return {"status": "exists", "id": record.id, "ontology_id": record.ontology_id}
    except NotFoundError:
        pass

    body = GraphCreate(
        id="ecom_products",
        name="E-Commerce Knowledge Graph",
        ontology_id="ecom",
        kind="product",
        description=(
            "Contains products and brand information from multiple e-commerce sources "
            "(Primark, Zara, Matalan, C&A, Walmart)."
        ),
        source_ids=["ecom_products"],
    )
    try:
        record = platform.kg_registry.create_graph(body)
        return {"status": "created", "id": record.id, "ontology_id": record.ontology_id}
    except ValidationFailed as exc:
        if "already exists" in str(exc).lower():
            record = platform.kg_registry.get_graph("ecom_products")
            return {"status": "exists", "id": record.id, "ontology_id": record.ontology_id}
        raise


def ensure_ecom_source(platform) -> dict[str, Any]:
    for source in platform.registry.sources:
        if source.source_id == "ecom_products":
            return {"status": "exists", "source_id": source.source_id, "enabled": source.enabled}

    body = SourceRegisterBody(
        source_id="ecom_products",
        source_type="ecom_products",
        connector="authorized_sql",
        supported_formats=["sql"],
        ingestion_mode="incremental",
        update_strategy="content_hash",
        permissions_model="classification",
        enabled=True,
        description=(
            "Centralized e-commerce master_products (plan SRC_002). "
            "Brands such as Primark, Zara, Matalan, C&A, and Walmart live inside this one KG."
        ),
        future_connectors=["authorized_api"],
    )
    try:
        record = register_source(platform, body)
        return {"status": "created", "source_id": record["source_id"], "enabled": record.get("enabled")}
    except ValidationFailed as exc:
        if "already exists" not in str(exc).lower():
            raise
        # Race / file already had the source — reload and report exists.
        path = platform.settings.raw_dir / "source_registry.json"
        if path.exists():
            platform.registry = load_registry(path)
        for source in platform.registry.sources:
            if source.source_id == "ecom_products":
                return {"status": "exists", "source_id": source.source_id, "enabled": source.enabled}
        # Last resort: write-merge if registry lacked it despite message.
        registry = load_registry(path) if path.exists() else SourceRegistry(version=1, sources=[])
        record = SourceConfig(**body.model_dump())
        registry.sources = [item for item in registry.sources if item.source_id != "ecom_products"]
        registry.sources.append(record)
        validate_registry(registry)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(registry.model_dump(), indent=2) + "\n", encoding="utf-8")
        platform.registry = load_registry(path)
        return {"status": "created", "source_id": "ecom_products", "enabled": True}


def _backup_mapping_candidates() -> list[Path]:
    root = Path(__file__).resolve().parents[2]
    matches: list[Path] = []
    for path in sorted(root.glob("data_backup_*/schemas/" + MAPPING_NAME), reverse=True):
        matches.append(path)
    return matches
