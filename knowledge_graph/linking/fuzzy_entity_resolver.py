"""Fuzzy entity resolve against OntologyRegistry canonical/alias index.

Uses ``rapidfuzz`` when installed, otherwise ``difflib``. Emits exact
canonical id/name hints for Text2Cypher — never invents ids.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from knowledge_graph.ontology import normalize_name

LOGGER = logging.getLogger("knowledge_graph.fuzzy_entity_resolver")

DEFAULT_THRESHOLD = 85.0


@dataclass
class FuzzyHit:
    surface: str
    canonical_id: str
    canonical_name: str
    entity_type: str
    ontology: str
    ontology_id: str = ""
    score: float = 0.0
    matched_surface: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FuzzyResolveResult:
    hits: list[FuzzyHit] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)

    def to_resolved_payload(self) -> dict[str, Any]:
        by_type: dict[str, dict[str, Any]] = {}
        for hit in self.hits:
            key = str(hit.entity_type or "entity").casefold()
            bucket = by_type.setdefault(
                key,
                {
                    "ids": [],
                    "names": [],
                    "status": "matched",
                    "hits": [],
                    "source": "ontology_fuzzy",
                },
            )
            if hit.canonical_id and hit.canonical_id not in bucket["ids"]:
                bucket["ids"].append(hit.canonical_id)
            if hit.canonical_name and hit.canonical_name not in bucket["names"]:
                bucket["names"].append(hit.canonical_name)
            bucket["hits"].append(hit.to_dict())
        return {
            "by_type": by_type,
            "fuzzy_hits": [h.to_dict() for h in self.hits],
            "unresolved": list(self.unresolved),
        }


def _ngrams(tokens: list[str], *, max_n: int = 3) -> list[str]:
    out: list[str] = []
    n = len(tokens)
    for size in range(min(max_n, n), 0, -1):
        for i in range(0, n - size + 1):
            gram = " ".join(tokens[i : i + size])
            if len(gram) >= 2:
                out.append(gram)
    seen: set[str] = set()
    unique: list[str] = []
    for g in out:
        if g in seen:
            continue
        seen.add(g)
        unique.append(g)
    return unique


def _question_tokens(question: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (question or "").casefold()) if len(t) >= 2]


def _score(a: str, b: str) -> float:
    try:
        from rapidfuzz import fuzz

        return float(fuzz.WRatio(a, b))
    except Exception:
        pass
    import difflib

    return difflib.SequenceMatcher(None, a, b).ratio() * 100.0


class FuzzyEntityResolver:
    """Match question n-grams against ontology canonical names/aliases."""

    def __init__(self, registry=None, *, threshold: float = DEFAULT_THRESHOLD):
        self.registry = registry
        self.threshold = float(threshold)

    def _index(self, platform=None) -> dict[str, list]:
        if self.registry is not None:
            return getattr(self.registry, "canonical_index", {}) or {}
        from knowledge_graph.ontology_registry import get_ontology_registry

        reg = get_ontology_registry(platform)
        self.registry = reg
        return reg.canonical_index or {}

    def resolve(
        self,
        question: str,
        *,
        platform=None,
        limit: int = 8,
    ) -> FuzzyResolveResult:
        index = self._index(platform)
        if not index:
            return FuzzyResolveResult(unresolved=_question_tokens(question)[:12])

        tokens = _question_tokens(question)
        grams = _ngrams(tokens, max_n=3)
        # Generic function words only — never brand/domain names.
        stop = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "that",
            "this",
            "what",
            "which",
            "how",
            "many",
            "show",
            "list",
            "have",
            "does",
            "about",
        }

        surfaces = list(index.keys())
        hits: list[FuzzyHit] = []
        used_ids: set[str] = set()
        matched_grams: set[str] = set()

        for gram in grams:
            if gram in stop and " " not in gram:
                continue
            nn = normalize_name(gram)
            if not nn:
                continue
            exact = index.get(nn) or []
            best_score = 100.0 if exact else 0.0
            candidates = list(exact)
            if not candidates:
                scored: list[tuple[float, str]] = []
                for key in surfaces:
                    sc = _score(nn, key)
                    if sc >= self.threshold:
                        scored.append((sc, key))
                scored.sort(key=lambda item: (-item[0], item[1]))
                for sc, key in scored[:3]:
                    for entry in index.get(key) or []:
                        candidates.append(entry)
                        best_score = max(best_score, sc)
            for entry in candidates:
                cid = str(getattr(entry, "canonical_id", "") or "")
                if not cid or cid in used_ids:
                    continue
                used_ids.add(cid)
                matched_grams.add(nn)
                hits.append(
                    FuzzyHit(
                        surface=gram,
                        canonical_id=cid,
                        canonical_name=str(getattr(entry, "canonical_name", "") or ""),
                        entity_type=str(getattr(entry, "entity_type", "") or ""),
                        ontology=str(getattr(entry, "ontology_name", "") or ""),
                        ontology_id=str(getattr(entry, "ontology_id", "") or ""),
                        score=float(best_score),
                        matched_surface=str(getattr(entry, "surface", "") or nn),
                    )
                )
                if len(hits) >= limit:
                    break
            if len(hits) >= limit:
                break

        unresolved = [
            g
            for g in grams
            if " " not in g and g not in stop and normalize_name(g) not in matched_grams
        ]
        return FuzzyResolveResult(hits=hits, unresolved=unresolved[:20])


def resolve_fuzzy_entities(
    question: str,
    *,
    platform=None,
    registry=None,
    threshold: float = DEFAULT_THRESHOLD,
    limit: int = 8,
) -> FuzzyResolveResult:
    return FuzzyEntityResolver(registry=registry, threshold=threshold).resolve(
        question, platform=platform, limit=limit
    )
