"""Runtime OntologyRegistry: scan ontology JSONs for domains + canonical/alias index.

Discovery uses ``FASHIONGPT_ONTOLOGY_GLOB`` when set, else
``{DATA_ROOT}/schemas/ontologies/*.json`` and registry-listed paths.
No ontology filenames or domain names are hardcoded as primary logic.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from knowledge_graph.config import DATA_ROOT
from knowledge_graph.ontology import normalize_name

LOGGER = logging.getLogger("knowledge_graph.ontology_registry")

_LOCK = threading.Lock()
_CACHE: OntologyRegistry | None = None
_CACHE_TS: float = 0.0
_CACHE_TTL_SEC = 600.0


@dataclass
class DomainProfile:
    """One ontology JSON as a routing / prune profile."""

    ontology_id: str
    name: str
    description: str = ""
    version: str = ""
    entity_types: list[str] = field(default_factory=list)
    relationship_allowlist: dict[str, list[list[str]]] = field(default_factory=dict)
    path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def tokens(self) -> set[str]:
        bag: set[str] = set()
        for text in (self.name, self.description, self.ontology_id):
            bag.update(_tokens(str(text or "")))
        return bag


@dataclass
class CanonicalIndexEntry:
    surface: str
    canonical_id: str
    canonical_name: str
    entity_type: str
    ontology_name: str
    ontology_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OntologyRegistry:
    """In-memory snapshot of all discovered ontology JSON files."""

    domains: list[DomainProfile] = field(default_factory=list)
    canonical_index: dict[str, list[CanonicalIndexEntry]] = field(default_factory=dict)
    built_at: float = 0.0
    source_paths: list[str] = field(default_factory=list)

    def list_domains(self) -> list[DomainProfile]:
        return list(self.domains)

    def domain_by_id(self, ontology_id: str) -> DomainProfile | None:
        want = str(ontology_id or "").casefold()
        for domain in self.domains:
            if domain.ontology_id.casefold() == want or domain.name.casefold() == want:
                return domain
        return None

    def allowlist_for_domains(self, domain_keys: list[str] | None = None) -> dict[str, list[list[str]]]:
        """Union relationship_allowlist for selected domains (or all)."""

        keys = {str(k).casefold() for k in (domain_keys or []) if k}
        merged: dict[str, list[list[str]]] = {}
        for domain in self.domains:
            if keys and domain.ontology_id.casefold() not in keys and domain.name.casefold() not in keys:
                continue
            for rel, pairs in (domain.relationship_allowlist or {}).items():
                bucket = merged.setdefault(str(rel).upper(), [])
                for pair in pairs or []:
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        item = [str(pair[0]), str(pair[1])]
                        if item not in bucket:
                            bucket.append(item)
        return merged

    def entity_types_for_domains(self, domain_keys: list[str] | None = None) -> set[str]:
        keys = {str(k).casefold() for k in (domain_keys or []) if k}
        out: set[str] = set()
        for domain in self.domains:
            if keys and domain.ontology_id.casefold() not in keys and domain.name.casefold() not in keys:
                continue
            out.update(str(t) for t in (domain.entity_types or []))
        return out

    def domain_for_question(self, question: str, *, limit: int = 3) -> list[DomainProfile]:
        """Score question tokens against ontology name+description (no hardcoded domains)."""

        q_tokens = _tokens(question)
        if not self.domains:
            return []
        scored: list[tuple[int, DomainProfile]] = []
        for domain in self.domains:
            overlap = len(q_tokens & domain.tokens())
            if overlap:
                scored.append((overlap, domain))
        scored.sort(key=lambda item: (-item[0], item[1].ontology_id))
        if scored:
            return [d for _, d in scored[: max(1, int(limit))]]
        # Fail-open: return first N domains when nothing scores.
        return list(self.domains[: max(1, int(limit))])

    def to_dict(self) -> dict[str, Any]:
        return {
            "domains": [d.to_dict() for d in self.domains],
            "canonical_surfaces": len(self.canonical_index),
            "source_paths": list(self.source_paths),
            "built_at": self.built_at,
        }


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9_]+", (text or "").casefold()) if len(t) >= 2}


def _discover_ontology_paths(platform=None) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()

    def _add(path: Path) -> None:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        key = str(resolved)
        if key in seen or not path.is_file():
            return
        seen.add(key)
        paths.append(path)

    glob_pat = (os.getenv("FASHIONGPT_ONTOLOGY_GLOB") or "").strip()
    if glob_pat:
        for match in glob.glob(glob_pat, recursive=True):
            _add(Path(match))
    else:
        default_dir = Path(DATA_ROOT) / "schemas" / "ontologies"
        if default_dir.is_dir():
            for match in sorted(default_dir.glob("*.json")):
                _add(match)

    try:
        registry = getattr(platform, "kg_registry", None) if platform is not None else None
        if registry is not None:
            lister = getattr(registry, "list_ontologies", None)
            rows = list(lister() or []) if callable(lister) else []
            for row in rows:
                oid = getattr(row, "id", None) or (row.get("id") if isinstance(row, dict) else None)
                if not oid:
                    continue
                loader = getattr(registry, "ontology_file", None)
                if callable(loader):
                    try:
                        _add(Path(loader(str(oid))))
                    except Exception:
                        pass
            settings = getattr(platform, "settings", None)
            if settings is not None:
                try:
                    from knowledge_graph.kg_registry.store import ontologies_dir

                    odir = ontologies_dir(settings)
                    if odir.is_dir():
                        for match in sorted(odir.glob("*.json")):
                            _add(match)
                except Exception:
                    pass
    except Exception as exc:
        LOGGER.debug("registry ontology discovery skipped: %s", exc)

    return paths


def build_ontology_registry(platform=None) -> OntologyRegistry:
    """Scan and parse ontology JSON files into a registry snapshot."""

    from knowledge_graph.ontology import load_ontology

    reg = OntologyRegistry(built_at=time.time())
    for path in _discover_ontology_paths(platform):
        try:
            ontology = load_ontology(path)
        except Exception as exc:
            LOGGER.warning("skip ontology file %s: %s", path, exc)
            continue
        ontology_id = path.stem
        cleaned_allow: dict[str, list[list[str]]] = {}
        for rel, pairs in (ontology.relationship_allowlist or {}).items():
            bucket: list[list[str]] = []
            for pair in pairs or []:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    bucket.append([str(pair[0]), str(pair[1])])
            cleaned_allow[str(rel).upper()] = bucket
        profile = DomainProfile(
            ontology_id=ontology_id,
            name=str(ontology.name or ontology_id),
            description=str(ontology.description or ""),
            version=str(ontology.version or ""),
            entity_types=list(ontology.entity_types or []),
            relationship_allowlist=cleaned_allow,
            path=str(path),
        )
        reg.domains.append(profile)
        reg.source_paths.append(str(path))

        for entity in ontology.canonical_entities or []:
            for surface in [entity.name, *(entity.aliases or [])]:
                nn = normalize_name(str(surface or ""))
                if not nn or len(nn) < 2:
                    continue
                entry = CanonicalIndexEntry(
                    surface=nn,
                    canonical_id=str(entity.id),
                    canonical_name=str(entity.name),
                    entity_type=str(entity.type),
                    ontology_name=profile.name,
                    ontology_id=ontology_id,
                )
                reg.canonical_index.setdefault(nn, []).append(entry)
    return reg


def invalidate_ontology_registry_cache() -> None:
    global _CACHE, _CACHE_TS
    with _LOCK:
        _CACHE = None
        _CACHE_TS = 0.0


def get_ontology_registry(platform=None, *, force: bool = False) -> OntologyRegistry:
    """Return cached OntologyRegistry, refreshing when stale."""

    global _CACHE, _CACHE_TS
    now = time.monotonic()
    with _LOCK:
        if not force and _CACHE is not None and (now - _CACHE_TS) < _CACHE_TTL_SEC:
            return _CACHE
    snap = build_ontology_registry(platform)
    with _LOCK:
        _CACHE = snap
        _CACHE_TS = time.monotonic()
    return snap


def warm_ontology_registry(platform=None) -> dict[str, Any]:
    """Force-build registry (KG API lifespan)."""

    snap = get_ontology_registry(platform, force=True)
    return {
        "domains": len(snap.domains),
        "canonical_surfaces": len(snap.canonical_index),
        "paths": list(snap.source_paths),
    }
