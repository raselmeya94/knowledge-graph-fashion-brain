"""Resolve user terms to Canonical / SourceEntity / Product / Brand / Buyer / Material.

Uses full-text indexes when present, then normalized_name / aliases CONTAINS fallback.
Also follows (:SourceEntity)-[:RESOLVES_TO]->(:Canonical) when available.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from knowledge_graph.ontology import normalize_name

LOGGER = logging.getLogger("knowledge_graph.canonical_resolve")

_DEFAULT_LABELS = (
    "Canonical",
    "SourceEntity",
    "Product",
    "Brand",
    "Buyer",
    "Material",
)


def _fulltext_index_name(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "", label).lower()
    return f"{safe}_name_ft"


def _lucene_fuzzy_query(term: str) -> str:
    """Build a Lucene fulltext query with ``token~`` fuzzy markers."""

    cleaned = re.sub(r'[+\-&|!(){}\[\]^"~*?:\\/]', " ", str(term or "")).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        return ""
    parts = [f"{t}~" if len(t) > 2 else t for t in cleaned.split()]
    return " ".join(parts)


@dataclass
class CanonicalHit:
    label: str
    id: str
    name: str
    normalized_name: str
    score: float
    source: str  # fulltext|normalized|alias|resolves_to
    canonical_id: str | None = None
    canonical_name: str | None = None
    graph: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CanonicalResolveResult:
    term: str
    status: str  # matched|ambiguous|no_match|skipped
    hits: list[CanonicalHit] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "term": self.term,
            "status": self.status,
            "hits": [h.to_dict() for h in self.hits],
            "ids": [h.id for h in self.hits],
            "names": [h.name for h in self.hits if h.name],
            "canonical_ids": [h.canonical_id for h in self.hits if h.canonical_id],
            "canonical_names": [h.canonical_name for h in self.hits if h.canonical_name],
        }


def resolve_canonical_entity(
    platform,
    term: str,
    *,
    labels: list[str] | None = None,
    graph_id: str | None = None,
    limit: int = 8,
) -> CanonicalResolveResult:
    """Resolve a free-text term to graph entities (never invent ids)."""

    text = str(term or "").strip()
    if not text or len(text) < 2:
        return CanonicalResolveResult(term=text, status="skipped")

    session_factory = getattr(getattr(platform, "graph", None), "_session", None)
    if not callable(session_factory):
        return CanonicalResolveResult(term=text, status="no_match")

    wanted = [str(lab) for lab in (labels or _DEFAULT_LABELS) if lab]
    nn = normalize_name(text)
    hits: list[CanonicalHit] = []
    seen: set[str] = set()

    def _add(hit: CanonicalHit) -> None:
        key = f"{hit.label}:{hit.id}"
        if key in seen or not hit.id:
            return
        seen.add(key)
        hits.append(hit)

    try:
        with session_factory() as session:
            for label in wanted:
                # 1) Full-text when index exists (exact, then Lucene term~ fuzzy)
                try:
                    index = _fulltext_index_name(label)
                    q = (
                        f"CALL db.index.fulltext.queryNodes($index, $q) "
                        f"YIELD node, score "
                        f"WHERE $label IN labels(node) "
                        f"AND ($g IS NULL OR coalesce(node.graph,'') = $g OR node.graph IS NULL) "
                        f"RETURN node.id AS id, coalesce(node.name, node.id) AS name, "
                        f"coalesce(node.normalized_name, '') AS nn, score, "
                        f"coalesce(node.graph, '') AS graph "
                        f"ORDER BY score DESC LIMIT $limit"
                    )
                    before = len(hits)
                    for query_text, source in (
                        (text, "fulltext"),
                        (_lucene_fuzzy_query(text), "fulltext_fuzzy"),
                    ):
                        if not query_text:
                            continue
                        if source == "fulltext_fuzzy" and len(hits) > before:
                            break
                        try:
                            for row in session.run(
                                q,
                                {
                                    "index": index,
                                    "q": query_text,
                                    "label": label,
                                    "g": graph_id,
                                    "limit": max(1, min(int(limit), 20)),
                                },
                            ):
                                _add(
                                    CanonicalHit(
                                        label=label,
                                        id=str(row.get("id") or ""),
                                        name=str(row.get("name") or ""),
                                        normalized_name=str(row.get("nn") or ""),
                                        score=float(row.get("score") or 0.0),
                                        source=source,
                                        graph=str(row.get("graph") or "") or None,
                                    )
                                )
                        except Exception:
                            continue
                except Exception:
                    pass

                # 2) normalized_name / aliases CONTAINS fallback
                try:
                    q2 = (
                        f"MATCH (n:`{label}`) "
                        f"WHERE ($g IS NULL OR coalesce(n.graph,'') = $g OR n.graph IS NULL) "
                        f"AND ("
                        f"toLower(coalesce(n.normalized_name,'')) CONTAINS $nn "
                        f"OR toLower(coalesce(n.name,'')) CONTAINS $q "
                        f"OR ANY(a IN coalesce(n.aliases, []) WHERE toLower(toString(a)) CONTAINS $q)"
                        f") "
                        f"RETURN n.id AS id, coalesce(n.name, n.id) AS name, "
                        f"coalesce(n.normalized_name, '') AS nn, "
                        f"coalesce(n.graph, '') AS graph "
                        f"LIMIT $limit"
                    )
                    for row in session.run(
                        q2,
                        {
                            "g": graph_id,
                            "nn": nn,
                            "q": text.casefold(),
                            "limit": max(1, min(int(limit), 20)),
                        },
                    ):
                        name = str(row.get("name") or "")
                        row_nn = str(row.get("nn") or "")
                        source = "normalized"
                        if nn and row_nn == nn:
                            score = 1.0
                        elif nn and nn in row_nn:
                            score = 0.7
                        else:
                            score = 0.5
                            source = "alias"
                        _add(
                            CanonicalHit(
                                label=label,
                                id=str(row.get("id") or ""),
                                name=name,
                                normalized_name=row_nn,
                                score=score,
                                source=source,
                                graph=str(row.get("graph") or "") or None,
                            )
                        )
                except Exception as exc:
                    LOGGER.debug("contains fallback failed for %s: %s", label, exc)

            # 3) Attach RESOLVES_TO canonical when SourceEntity matched
            source_ids = [h.id for h in hits if h.label == "SourceEntity"]
            if source_ids:
                try:
                    for row in session.run(
                        "MATCH (s:SourceEntity)-[:RESOLVES_TO]->(c:Canonical) "
                        "WHERE s.id IN $ids "
                        "RETURN s.id AS sid, c.id AS cid, coalesce(c.name, c.id) AS cname",
                        {"ids": source_ids},
                    ):
                        sid = str(row.get("sid") or "")
                        for hit in hits:
                            if hit.id == sid and hit.label == "SourceEntity":
                                hit.canonical_id = str(row.get("cid") or "") or None
                                hit.canonical_name = str(row.get("cname") or "") or None
                                if hit.source != "resolves_to":
                                    hit.source = "resolves_to"
                except Exception as exc:
                    LOGGER.debug("RESOLVES_TO enrich failed: %s", exc)

            # Also reverse: if Canonical hit, keep as-is; if Product/Brand matched, done
    except Exception as exc:
        LOGGER.warning("resolve_canonical_entity failed: %s", exc)
        return CanonicalResolveResult(term=text, status="no_match")

    hits.sort(key=lambda h: (-h.score, h.label, h.name))
    top = hits[: max(1, min(int(limit), 12))]
    if not top:
        return CanonicalResolveResult(term=text, status="no_match")
    # Ambiguous when several different names at similar score
    names = {normalize_name(h.canonical_name or h.name) for h in top[:3]}
    status = "ambiguous" if len(names) > 1 and top[0].score - top[min(1, len(top) - 1)].score < 0.15 else "matched"
    return CanonicalResolveResult(term=text, status=status, hits=top)


def resolve_dynamic_entity(
    platform,
    term: str,
    *,
    labels: list[str] | None = None,
    graph_id: str | None = None,
    limit: int = 8,
) -> CanonicalResolveResult:
    """Neo4j fulltext with Lucene ``~`` + CONTAINS fallback (not ontology-only)."""

    return resolve_canonical_entity(
        platform, term, labels=labels, graph_id=graph_id, limit=limit
    )


def resolve_terms(
    platform,
    terms: list[str],
    *,
    graph_id: str | None = None,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Resolve many surface terms; return a dict suitable for Text2Cypher <resolved>."""

    out: dict[str, Any] = {}
    for term in terms:
        key = str(term or "").strip()
        if not key:
            continue
        result = resolve_dynamic_entity(platform, key, graph_id=graph_id, labels=labels)
        if result.status in {"matched", "ambiguous"} and result.hits:
            payload = result.to_dict()
            out[key] = payload
    return out
