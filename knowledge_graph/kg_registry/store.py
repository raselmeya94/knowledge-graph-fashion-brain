"""File-backed registry of ontologies and graphs."""

from __future__ import annotations

import json
import re
import shutil
import threading
from pathlib import Path

from knowledge_graph.config import DATA_ROOT, Settings
from knowledge_graph.graph.snapshots import SNAPSHOTS_DIR
from knowledge_graph.errors import NotFoundError, ValidationFailed
from knowledge_graph.graph.build import build_envelopes, register_builder
from knowledge_graph.ids import register_source_identity, slug
from knowledge_graph.kg_registry.models import (
    GraphCreate,
    GraphRecord,
    KgRegistry,
    OntologyCreate,
    OntologyRecord,
    OntologyReplace,
)
from knowledge_graph.ontology import Ontology, combine_ontologies, load_ontology, validate_ontology

_LOCK = threading.Lock()

_PRODUCT_CAPABILITIES = [
    "product_search",
    "material_analysis",
    "buyer_analysis",
    "specification",
]
_ECOM_CAPABILITIES = [
    "product_search",
    "brand_comparison",
    "material_analysis",
    "price_analysis",
]

def registry_path(settings: Settings) -> Path:
    return settings.generated / "kg_registry.json"


def ontologies_dir(settings: Settings) -> Path:
    """Approved custom ontologies live beside builtin schemas (not under generated/)."""

    return settings.ontology_file.parent / "ontologies"


def ontology_registry_path(settings: Settings, ontology_id: str, absolute: Path) -> str:
    """Prefer a data/-relative path; fall back to absolute (test temp dirs)."""

    try:
        return absolute.resolve().relative_to(DATA_ROOT.resolve()).as_posix()
    except ValueError:
        return str(absolute.resolve())


def _id_slug(value: str) -> str:
    """Normalize registry ids.

    Explicit ids like ``ecom_products`` / ``own_products`` must keep underscores —
    ``slug()`` would turn them into ``ecom-products`` and break kind=product create.
    """

    text = value.strip().casefold()
    if re.match(r"^[a-z0-9]+(?:[_-][a-z0-9]+)*$", text):
        return text
    text = slug(value)
    if not text or not re.match(r"^[a-z0-9]+(?:-[a-z0-9]+)*$", text):
        raise ValidationFailed(
            f"Invalid id {value!r}. Use letters, numbers, hyphens, or underscores."
        )
    return text


def default_graph_for_kind(kind: str) -> str:
    """Default individual graph for construction when the admin does not pick one."""

    if kind in {"ecom", "ecom_products"}:
        return "ecom_products"
    if kind in {"product", "own_products", "catalog", "plmex", "plmex_products"}:
        return "own_products"
    # Email/HTML/JSON/PDF all land on news unless admin selects another graph.
    return "news"


def enrich_graph_record(record: GraphRecord, ontology: Ontology | None = None) -> GraphRecord:
    """Fill entities/relationships/capabilities from ontology when missing."""

    updated = record.model_copy(deep=True)
    if ontology is not None:
        if not updated.entities:
            updated.entities = list(ontology.entity_types)
        if not updated.relationships:
            updated.relationships = sorted(ontology.relationship_allowlist.keys())
    if not updated.capabilities:
        if updated.kind == "product":
            if updated.id == "ecom_products" or "ecom" in updated.id:
                updated.capabilities = list(_ECOM_CAPABILITIES)
            else:
                updated.capabilities = list(_PRODUCT_CAPABILITIES)
        elif updated.kind == "documents":
            updated.capabilities = ["document_search", "news_facts"]
        elif updated.kind == "system":
            updated.capabilities = [
                "source_inventory",
                "graph_overview",
                "attribute_discovery",
                "registry_lookup",
                "shared_interconnect",
                "system_metadata",
            ]
        else:
            updated.capabilities = ["graph_query"]
    return updated


class KgRegistryStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.path = registry_path(settings)
        self._cache: KgRegistry | None = None

    def load(self) -> KgRegistry:
        with _LOCK:
            if self._cache is not None:
                return self._cache.model_copy(deep=True)
            if self.path.exists():
                data = json.loads(self.path.read_text(encoding="utf-8"))
                registry = KgRegistry.model_validate(data)
                registry = self._normalize_ontology_paths(registry)
                # Persist path normalization only — never re-seed builtin graphs/ontologies.
                self._write(registry)
            else:
                # Empty start for full POST-driven setup (ontology → graph → source → sync).
                registry = KgRegistry(version=1, ontologies=[], graphs=[])
            self._cache = registry
            return registry.model_copy(deep=True)

    def save(self, registry: KgRegistry) -> KgRegistry:
        with _LOCK:
            self._write(registry)
            self._cache = registry.model_copy(deep=True)
            return self._cache.model_copy(deep=True)

    def reload(self) -> KgRegistry:
        with _LOCK:
            self._cache = None
        return self.load()

    def _write(self, registry: KgRegistry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(registry.model_dump(), indent=2) + "\n"
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(payload, encoding="utf-8")
        # Windows often denies Path.replace while another process (FB/KG) holds the json open.
        try:
            temporary.replace(self.path)
        except PermissionError:
            self.path.write_text(payload, encoding="utf-8")
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def ontology_file(self, ontology_id: str) -> Path:
        """Resolve ontology JSON path.

        API-created ontologies live under ``schemas/ontologies/<id>.json`` and that
        path is stored in the registry. Prefer the registry path whenever the file
        exists — do not force classic builtin filenames (``ecom-products-ontology.json``)
        which break POST-driven empty boots.
        """

        from knowledge_graph.config import KG_ROOT

        record = self.get_ontology(ontology_id)
        path = Path(record.path)
        if path.is_absolute():
            if path.exists():
                return path
            registered = path
        else:
            data_candidate = DATA_ROOT / path
            if data_candidate.exists():
                return data_candidate
            legacy = KG_ROOT / path
            if legacy.exists():
                return legacy
            registered = data_candidate

        # Fall back to classic seeded filenames only when the registry file is missing.
        classics: list[Path] = []
        if ontology_id == "ecom":
            classics = [self.settings.ontology_file]
        elif ontology_id == "news":
            classics = [self.settings.news_ontology_file]
        elif ontology_id == "product":
            classics = [
                DATA_ROOT / "schemas" / "product-knowledge-ontology.json",
                self.settings.ontology_file,
            ]
        elif ontology_id == "system":
            classics = [DATA_ROOT / "schemas" / "system-meta-ontology.json"]
        for candidate in classics:
            if candidate.exists():
                return candidate
        return registered

    def _normalize_ontology_paths(self, registry: KgRegistry) -> KgRegistry:
        """Keep custom ontologies under data/schemas/ontologies with relative registry paths.

        Older commits wrote approved ontologies into generated/ontologies/ (runtime junk).
        On load we migrate those files into schemas/ontologies/ so confirmed schemas stay
        versionable next to news/product.
        """

        changed = False
        target_dir = ontologies_dir(self.settings)
        target_dir.mkdir(parents=True, exist_ok=True)
        legacy_dir = self.settings.generated / "ontologies"
        from knowledge_graph.config import KG_ROOT

        for item in registry.ontologies:
            if item.builtin or item.status != "active":
                continue
            desired_abs = target_dir / f"{item.id}.json"
            desired_path = ontology_registry_path(self.settings, item.id, desired_abs)
            current = Path(item.path)
            current_abs = current if current.is_absolute() else (
                DATA_ROOT / current if (DATA_ROOT / current).exists() else KG_ROOT / current
            )
            if not desired_abs.exists():
                candidates = []
                if current_abs.exists():
                    candidates.append(current_abs)
                legacy = legacy_dir / f"{item.id}.json"
                if legacy.exists():
                    candidates.append(legacy)
                if candidates:
                    desired_abs.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(candidates[0], desired_abs)
                    changed = True
            if item.path != desired_path and desired_abs.exists():
                item.path = desired_path
                changed = True
            legacy = legacy_dir / f"{item.id}.json"
            if legacy.exists() and desired_abs.exists() and legacy.resolve() != desired_abs.resolve():
                try:
                    legacy.unlink()
                    changed = True
                except OSError:
                    pass
        return registry

    def get_ontology(self, ontology_id: str, *, include_deleted: bool = False) -> OntologyRecord:
        for item in self.load().ontologies:
            if item.id != ontology_id:
                continue
            if not include_deleted and item.status == "deleted":
                break
            return item
        raise NotFoundError(f"Ontology '{ontology_id}' was not found.")

    def get_graph(self, graph_id: str, *, include_deleted: bool = False) -> GraphRecord:
        for item in self.load().graphs:
            if item.id != graph_id:
                continue
            if not include_deleted and item.status == "deleted":
                break
            return item
        raise NotFoundError(f"Graph '{graph_id}' was not found.")

    def list_ontologies(self, *, include_deleted: bool = False) -> list[OntologyRecord]:
        rows = self.load().ontologies
        if include_deleted:
            return rows
        return [item for item in rows if item.status == "active"]

    def list_graphs(self, *, include_deleted: bool = False) -> list[GraphRecord]:
        rows = self.load().graphs
        if include_deleted:
            return rows
        return [item for item in rows if item.status == "active"]

    def load_ontology_object(self, ontology_id: str) -> Ontology:
        return load_ontology(self.ontology_file(ontology_id))

    def union_ontology(self) -> Ontology:
        from knowledge_graph.ontology import empty_ontology

        rows = self.list_ontologies()
        if not rows:
            return empty_ontology()
        ontologies: list[Ontology] = []
        for item in rows:
            try:
                ontologies.append(self.load_ontology_object(item.id))
            except Exception:
                continue
        if not ontologies:
            return empty_ontology()
        combined = ontologies[0]
        for other in ontologies[1:]:
            combined = combine_ontologies(combined, other)
        return combined

    def create_ontology(self, body: OntologyCreate) -> tuple[OntologyRecord, Ontology]:
        ontology_id = _id_slug(body.id or body.name)
        registry = self.load()
        if any(item.id == ontology_id and item.status == "active" for item in registry.ontologies):
            raise ValidationFailed(f"Ontology '{ontology_id}' already exists.")
        if body.clone_from:
            data = json.loads(self.ontology_file(body.clone_from).read_text(encoding="utf-8"))
            data["name"] = body.name or data.get("name") or ontology_id
            if body.description:
                data["description"] = body.description
        else:
            if not body.entity_types:
                raise ValidationFailed("entity_types is required unless clone_from is set.")
            data = {
                "name": body.name,
                "version": body.version,
                "description": body.description,
                "entity_types": body.entity_types,
                "relationship_allowlist": body.relationship_allowlist,
                "canonical_entities": body.canonical_entities,
            }
        ontology = Ontology.model_validate(data)
        validate_ontology(ontology)
        absolute = ontologies_dir(self.settings) / f"{ontology_id}.json"
        absolute.parent.mkdir(parents=True, exist_ok=True)
        absolute.write_text(json.dumps(ontology.model_dump(), indent=2) + "\n", encoding="utf-8")
        # Drop any stale generated copy so schemas/ stays the source of truth.
        legacy = self.settings.generated / "ontologies" / f"{ontology_id}.json"
        if legacy.exists() and legacy.resolve() != absolute.resolve():
            legacy.unlink()
        record = OntologyRecord(
            id=ontology_id,
            name=body.name,
            description=body.description or ontology.description,
            path=ontology_registry_path(self.settings, ontology_id, absolute),
            builtin=False,
        )
        registry.ontologies = [item for item in registry.ontologies if item.id != ontology_id]
        registry.ontologies.append(record)
        self.save(registry)
        return record, ontology

    def replace_ontology(self, ontology_id: str, body: OntologyReplace) -> Ontology:
        record = self.get_ontology(ontology_id)
        path = self.ontology_file(ontology_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        if body.name is not None:
            data["name"] = body.name
            record.name = body.name
        if body.description is not None:
            data["description"] = body.description
            record.description = body.description
        if body.version is not None:
            data["version"] = body.version
        if body.entity_types is not None:
            data["entity_types"] = body.entity_types
        if body.relationship_allowlist is not None:
            data["relationship_allowlist"] = body.relationship_allowlist
        if body.canonical_entities is not None:
            data["canonical_entities"] = body.canonical_entities
        ontology = Ontology.model_validate(data)
        validate_ontology(ontology)
        path.write_text(json.dumps(ontology.model_dump(), indent=2) + "\n", encoding="utf-8")
        registry = self.load()
        registry.ontologies = [record if item.id == ontology_id else item for item in registry.ontologies]
        # refresh record from loop
        for index, item in enumerate(registry.ontologies):
            if item.id == ontology_id:
                registry.ontologies[index] = record
        self.save(registry)
        return ontology

    def add_pair(self, ontology_id: str, rel_type: str, start_type: str, end_type: str) -> Ontology:
        from knowledge_graph.ontology import allow_relationship

        ontology = self.load_ontology_object(ontology_id)
        path = self.ontology_file(ontology_id)
        allow_relationship(ontology, path, rel_type.upper(), start_type, end_type)
        return self.load_ontology_object(ontology_id)

    def remove_pair(self, ontology_id: str, rel_type: str, start_type: str, end_type: str) -> Ontology:
        path = self.ontology_file(ontology_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        allowlist = data.setdefault("relationship_allowlist", {})
        key = rel_type.upper()
        pairs = allowlist.get(key) or []
        target = [start_type, end_type]
        if target not in pairs:
            raise NotFoundError(f"{key} {start_type} -> {end_type} is not on the allowlist.")
        allowlist[key] = [pair for pair in pairs if pair != target]
        if not allowlist[key]:
            del allowlist[key]
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return load_ontology(path)

    def create_graph(self, body: GraphCreate) -> GraphRecord:
        graph_id = _id_slug(body.id or body.name)
        self.get_ontology(body.ontology_id)
        registry = self.load()
        if any(item.id == graph_id and item.status == "active" for item in registry.graphs):
            raise ValidationFailed(f"Graph '{graph_id}' already exists.")
        if body.kind == "product" and graph_id not in {"own_products", "ecom_products"}:
            raise ValidationFailed(
                "kind=product is only allowed for graph id 'own_products' or 'ecom_products' "
                f"(got '{graph_id}'). Send body.id exactly as ecom_products or own_products."
            )
        record = GraphRecord(
            id=graph_id,
            name=body.name,
            ontology_id=body.ontology_id,
            kind=body.kind,
            description=body.description,
            builtin=False,
            source_ids=list(body.source_ids or []),
        )
        try:
            ontology = self.load_ontology_object(body.ontology_id)
        except Exception:
            ontology = None
        record = enrich_graph_record(record, ontology)
        registry.graphs = [item for item in registry.graphs if item.id != graph_id]
        registry.graphs.append(record)
        self.save(registry)
        ensure_graph_identity(record)
        snap_dir = self.settings.generated / SNAPSHOTS_DIR / graph_id
        snap_dir.mkdir(parents=True, exist_ok=True)
        snap = snap_dir / "graph_snapshot.json"
        if not snap.exists():
            snap.write_text(
                json.dumps({"source_id": graph_id, "nodes": [], "edges": [], "schema": []}, indent=2) + "\n",
                encoding="utf-8",
            )
        return record

    def soft_delete_graph(self, graph_id: str, *, force: bool = False) -> GraphRecord:
        record = self.get_graph(graph_id)
        if record.builtin and not force:
            raise ValidationFailed(f"Built-in graph '{graph_id}' cannot be deleted without force=true.")
        registry = self.load()
        for item in registry.graphs:
            if item.id == graph_id:
                item.status = "deleted"
                record = item
        self.save(registry)
        return record

    def register_all_identities(self) -> None:
        for graph in self.list_graphs():
            ensure_graph_identity(graph)

    def unified_registry(self, *, node_counts: dict[str, tuple[int, int]] | None = None) -> list:
        """Source + graph + ontology summary for routers and Fashion Brain."""

        from knowledge_graph.kg_registry.models import UnifiedRegistryEntry

        counts = node_counts or {}
        rows: list[UnifiedRegistryEntry] = []
        for graph in self.list_graphs():
            try:
                ontology = self.load_ontology_object(graph.ontology_id)
            except Exception:
                ontology = None
            enriched = enrich_graph_record(graph, ontology)
            nodes, edges = counts.get(graph.id, (0, 0))
            rows.append(
                UnifiedRegistryEntry(
                    graph_id=enriched.id,
                    source_ids=list(enriched.source_ids),
                    graph_name=enriched.name,
                    description=enriched.description,
                    kind=enriched.kind,
                    ontology_id=enriched.ontology_id,
                    entities=list(enriched.entities),
                    relationships=list(enriched.relationships),
                    capabilities=list(enriched.capabilities),
                    status=enriched.status,
                    node_count=nodes,
                    edge_count=edges,
                )
            )
        return rows


def ensure_graph_identity(graph: GraphRecord) -> None:
    """Register id prefix + envelope builder for a construction or product graph."""

    if graph.kind == "product" or graph.id in {"own_products", "ecom_products"}:
        if graph.id == "ecom_products":
            register_source_identity("ecom_products", "ecom", "product", "Product")
        return
    if graph.id in {"news", "documents"}:
        return
    prefix = f"g{slug(graph.id).replace('-', '')[:16]}" or "gcustom"
    register_source_identity(graph.id, prefix, "document", "Document")

    def _builder(envelopes, repo, ontology, expected=graph.id):
        return build_envelopes(envelopes, repo, ontology, expected)

    register_builder(graph.id, _builder)


def clone_ontology_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, destination)
