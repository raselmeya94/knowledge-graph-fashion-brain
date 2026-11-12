"""Resolve user terms to real Neo4j facet node ids (never invent names)."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from knowledge_graph.ontology import normalize_name

LOGGER = logging.getLogger("knowledge_graph.linking.entity_linker")

_MATERIAL_DEMOTE = ("lining", "sole", "upper", "sock", "insole", "fill", "padding")


def _fulltext_index_name(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "", label).lower()
    return f"{safe}_name_ft"


def _resolve_graph_id(platform, graph_id: str | None) -> str | None:
    if graph_id:
        return graph_id
    from knowledge_graph.retrieval.runtime_schema import default_product_graph_id

    return default_product_graph_id(platform)


def _field_to_label(platform, field: str, graph_id: str | None) -> str | None:
    """Map extract field key → ontology facet label via RuntimeSchemaContext."""

    from knowledge_graph.retrieval.runtime_schema import build_runtime_schema

    ctx = build_runtime_schema(platform, graph_id)
    key = str(field or "").casefold()
    if key in {"brand", "buyer"} and ctx.brand_label:
        return ctx.brand_label
    # Direct match on facet filter keys / labels
    for label in ctx.facet_types:
        fk = label.casefold()
        if key == fk or key == fk.rstrip("s"):
            return label
        # GarmentCategory ↔ category
        if key in {"category", "garment", "garment_type"} and "category" in fk:
            return label
        if key in {"color", "colour"} and "color" in fk:
            return label
        if key == "material" and "material" in fk:
            return label
        if key == "availability" and "availability" in fk:
            return label
    aliases = {
        "category": ("GarmentCategory", "Category"),
        "garment": ("GarmentCategory", "Category"),
        "garment_type": ("GarmentCategory", "Category"),
        "material": ("Material",),
        "color": ("Color", "Colour"),
        "availability": ("Availability",),
        "brand": ("Brand",),
        "country": ("Country",),
    }
    for candidate in aliases.get(key, ()):
        if candidate in ctx.facet_types or candidate in ctx.entity_types:
            return candidate
    return None


@dataclass
class LinkHit:
    label: str
    id: str
    name: str
    normalized_name: str
    score: float
    source: str  # exact|fulltext|alias


@dataclass
class LinkResult:
    field: str
    term: str
    status: str  # matched|ambiguous|no_match|skipped
    hits: list[LinkHit] = field(default_factory=list)

    def ids(self) -> list[str]:
        return [h.id for h in self.hits]

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "term": self.term,
            "status": self.status,
            "hits": [asdict(h) for h in self.hits],
            "ids": self.ids(),
        }


def _aliases_path(platform=None) -> Path:
    settings = getattr(platform, "settings", None) if platform is not None else None
    generated = getattr(settings, "generated", None) if settings is not None else None
    if generated:
        candidate = Path(generated).parent / "schemas" / "entity_aliases.json"
        if candidate.is_file():
            return candidate
    root = Path(__file__).resolve().parents[2]
    return root / "data" / "schemas" / "entity_aliases.json"


def load_aliases(platform=None) -> dict[str, str]:
    path = _aliases_path(platform)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload.get("aliases") if isinstance(payload, dict) else {}
        out: dict[str, str] = {}
        for key, value in (raw or {}).items():
            k = str(key).strip().casefold()
            v = str(value or "").strip()
            if k:
                out[k] = v
        return out
    except Exception:
        return {}


def apply_alias(term: str, aliases: dict[str, str] | None = None) -> str:
    text = str(term or "").strip()
    if not text:
        return ""
    table = aliases if aliases is not None else {}
    mapped = table.get(text.casefold())
    if mapped is None:
        return text
    return mapped


def _plural_variants(term: str) -> list[str]:
    nn = normalize_name(term)
    if not nn:
        # Non-Latin: keep original token only.
        return [term.strip()] if term.strip() else []
    variants = [nn]
    if nn.endswith("ies") and len(nn) > 4:
        variants.append(nn[:-3] + "y")
    elif nn.endswith("s") and len(nn) > 3:
        variants.append(nn[:-1])
    else:
        variants.append(nn + "s")
    # unique preserve order
    seen: set[str] = set()
    out: list[str] = []
    for item in variants:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _session(platform):
    factory = getattr(platform.graph, "_session", None)
    return factory if callable(factory) else None


def list_live_values(platform, label: str, graph_id: str | None = None) -> list[dict[str, str]]:
    """List active facet nodes for any ontology label."""

    gid = _resolve_graph_id(platform, graph_id)
    session_factory = _session(platform)
    if not session_factory or not gid or not label:
        return []
    # Label interpolated only after stripping unsafe chars.
    safe = re.sub(r"[^A-Za-z0-9_]", "", label)
    if safe != label:
        return []
    query = (
        f"MATCH (n:Entity:{safe} {{graph:$g}}) "
        "WHERE coalesce(n.lifecycle,'active') <> 'deleted' "
        "RETURN n.id AS id, coalesce(n.name,'') AS name, coalesce(n.normalized_name,'') AS nn "
        "ORDER BY nn"
    )
    with session_factory() as session:
        return [
            {"id": str(r["id"]), "name": str(r["name"]), "normalized_name": str(r["nn"])}
            for r in session.run(query, {"g": gid})
        ]


def list_live_brands(platform, graph_id: str | None = None) -> list[dict[str, str]]:
    from knowledge_graph.retrieval.runtime_schema import build_runtime_schema

    gid = _resolve_graph_id(platform, graph_id)
    ctx = build_runtime_schema(platform, gid)
    label = ctx.brand_label or "Brand"
    return list_live_values(platform, label, gid)


def list_live_availability(platform, graph_id: str | None = None) -> list[dict[str, str]]:
    label = _field_to_label(platform, "availability", graph_id) or "Availability"
    return list_live_values(platform, label, graph_id)


def _fulltext_query(term: str) -> str:
    # Escape Lucene specials lightly; append fuzzy marker.
    cleaned = re.sub(r'[+\-&|!(){}\[\]^"~*?:\\/]', " ", term).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        return ""
    tokens = cleaned.split()
    parts = [f"{t}~" if len(t) > 2 else t for t in tokens]
    return " ".join(parts)


def _exact_and_fulltext(
    platform,
    *,
    graph_id: str,
    label: str,
    term: str,
    top_k: int,
) -> list[LinkHit]:
    session_factory = _session(platform)
    if not session_factory:
        return []
    variants = _plural_variants(term)
    index_name = _fulltext_index_name(label)
    hits: dict[str, LinkHit] = {}

    with session_factory() as session:
        for variant in variants:
            if not variant:
                continue
            # Exact normalized_name
            rows = session.run(
                f"MATCH (n:Entity:{label} {{graph:$g}}) "
                "WHERE coalesce(n.lifecycle,'active') <> 'deleted' "
                "AND n.normalized_name = $nn "
                "RETURN n.id AS id, coalesce(n.name,'') AS name, coalesce(n.normalized_name,'') AS nn "
                "LIMIT $k",
                {"g": graph_id, "nn": variant, "k": top_k},
            )
            for row in rows:
                node_id = str(row["id"])
                hits[node_id] = LinkHit(
                    label=label,
                    id=node_id,
                    name=str(row["name"]),
                    normalized_name=str(row["nn"]),
                    score=1.0,
                    source="exact",
                )
            # Prefix (jacket → jackets) when exact missed
            if not hits:
                rows = session.run(
                    f"MATCH (n:Entity:{label} {{graph:$g}}) "
                    "WHERE coalesce(n.lifecycle,'active') <> 'deleted' "
                    "AND (n.normalized_name STARTS WITH $nn OR n.normalized_name CONTAINS $nn) "
                    "RETURN n.id AS id, coalesce(n.name,'') AS name, coalesce(n.normalized_name,'') AS nn "
                    "ORDER BY size(n.normalized_name) ASC LIMIT $k",
                    {"g": graph_id, "nn": variant, "k": top_k},
                )
                for row in rows:
                    node_id = str(row["id"])
                    nn = str(row["nn"])
                    score = 0.92 if nn == variant or nn.startswith(variant) else 0.75
                    prev = hits.get(node_id)
                    if prev is None or score > prev.score:
                        hits[node_id] = LinkHit(
                            label=label,
                            id=node_id,
                            name=str(row["name"]),
                            normalized_name=nn,
                            score=score,
                            source="exact",
                        )

        if index_name:
            ft_term = _fulltext_query(term) or _fulltext_query(" ".join(variants))
            if ft_term:
                try:
                    rows = session.run(
                        "CALL db.index.fulltext.queryNodes($index, $q) YIELD node, score "
                        "WHERE node.graph = $g AND coalesce(node.lifecycle,'active') <> 'deleted' "
                        "RETURN node.id AS id, coalesce(node.name,'') AS name, "
                        "coalesce(node.normalized_name,'') AS nn, score "
                        "ORDER BY score DESC LIMIT $k",
                        {"index": index_name, "q": ft_term, "g": graph_id, "k": top_k},
                    )
                    for row in rows:
                        node_id = str(row["id"])
                        # Normalize lucene score into ~0.4–0.95 band
                        raw = float(row["score"] or 0.0)
                        score = min(0.95, 0.4 + raw / 10.0)
                        prev = hits.get(node_id)
                        if prev is None or score > prev.score:
                            hits[node_id] = LinkHit(
                                label=label,
                                id=node_id,
                                name=str(row["name"]),
                                normalized_name=str(row["nn"]),
                                score=score,
                                source="fulltext",
                            )
                except Exception as exc:
                    LOGGER.debug("fulltext query failed (%s): %s", index_name, exc)

    return sorted(hits.values(), key=lambda h: (-h.score, h.normalized_name))


def _term_supported(nn: str, term_nn: str) -> bool:
    if not term_nn or not nn:
        return False
    variants = set(_plural_variants(term_nn))
    variants.add(term_nn)
    tokens = set(nn.split())
    if nn in variants or any(v in tokens for v in variants):
        return True
    return any(nn.startswith(v + " ") or nn.endswith(" " + v) for v in variants)


def _prefer_term_support(hits: list[LinkHit], term_nn: str) -> list[LinkHit]:
    """If a strong exact/prefix hit exists, drop fuzzy neighbors that lack the term."""

    if not hits or not term_nn:
        return hits
    strong = [h for h in hits if h.score >= 0.95 and _term_supported(h.normalized_name, term_nn)]
    if not strong:
        return hits
    filtered = [h for h in hits if _term_supported(h.normalized_name, term_nn)]
    return filtered or hits


def _adjust_material_score(hit: LinkHit, term_nn: str) -> LinkHit:
    nn = hit.normalized_name
    score = hit.score
    if term_nn and nn == term_nn:
        score = max(score, 0.99)
    if re.search(r"\b\d+\b", nn) and term_nn and term_nn not in nn.split():
        score *= 0.7
    if any(token in nn for token in _MATERIAL_DEMOTE) and term_nn and term_nn not in ("lining", "sole"):
        score *= 0.55
    if term_nn and nn.startswith(term_nn + " "):
        score *= 0.85
    return LinkHit(hit.label, hit.id, hit.name, hit.normalized_name, score, hit.source)


def _select_hits(
    hits: list[LinkHit],
    *,
    top_k: int,
    min_score: float,
    score_delta: float,
) -> tuple[str, list[LinkHit]]:
    if not hits:
        return "no_match", []
    best = hits[0].score
    if best < min_score:
        return "no_match", []
    within = [h for h in hits if (best - h.score) <= score_delta]
    # Too many near-ties → ask user instead of OR-ing a huge set.
    if len(within) > 5:
        return "no_match", []
    chosen = within[: max(1, top_k)]
    if len(chosen) == 1:
        return "matched", chosen
    return "ambiguous", chosen


def _link_availability(platform, graph_id: str, term: str, *, top_k: int, min_score: float, score_delta: float) -> LinkResult:
    aliases = load_aliases(platform)
    mapped = apply_alias(term, aliases) or term
    token = re.sub(r"[^a-z0-9]+", "", mapped.casefold())
    rows = list_live_availability(platform, graph_id)
    hits: list[LinkHit] = []
    for row in rows:
        nn = re.sub(r"[^a-z0-9]+", "", (row.get("normalized_name") or "").casefold())
        name = re.sub(r"[^a-z0-9]+", "", (row.get("name") or "").casefold())
        score = 0.0
        if token and token in {nn, name}:
            score = 1.0
        elif token and (token in nn or nn in token or token in name):
            score = 0.85
        if score > 0:
            hits.append(
                LinkHit(
                    label="Availability",
                    id=row["id"],
                    name=row["name"],
                    normalized_name=row["normalized_name"],
                    score=score,
                    source="exact",
                )
            )
    hits.sort(key=lambda h: (-h.score, h.normalized_name))
    status, chosen = _select_hits(hits, top_k=top_k, min_score=min_score, score_delta=score_delta)
    return LinkResult(field="availability", term=term, status=status, hits=chosen)


def _link_brand(platform, graph_id: str, term: str) -> LinkResult:
    brands = list_live_brands(platform, graph_id)
    aliases = load_aliases(platform)
    mapped = apply_alias(term, aliases) or term
    key = normalize_name(mapped) or mapped.casefold()
    compact = re.sub(r"[^a-z0-9]+", "", key)
    hits: list[LinkHit] = []
    for row in brands:
        nn = row.get("normalized_name") or ""
        name = row.get("name") or ""
        if key in {nn, normalize_name(name), name.casefold()} or compact == re.sub(r"[^a-z0-9]+", "", name.casefold()):
            hits.append(
                LinkHit(
                    label="Brand",
                    id=row["id"],
                    name=name,
                    normalized_name=nn,
                    score=1.0,
                    source="exact",
                )
            )
    if not hits:
        return LinkResult(field="brand", term=term, status="no_match", hits=[])
    return LinkResult(field="brand", term=term, status="matched", hits=hits[:1])


def link_one(
    platform,
    field: str,
    term: str,
    *,
    graph_id: str | None = None,
) -> LinkResult:
    settings = getattr(platform, "settings", None)
    top_k = int(getattr(settings, "entity_linker_top_k", 5) or 5)
    min_score = float(getattr(settings, "entity_linker_min_score", 0.35) or 0.35)
    score_delta = float(getattr(settings, "entity_linker_score_delta", 0.08) or 0.08)
    graph_id = _resolve_graph_id(platform, graph_id)

    raw = str(term or "").strip()
    if not raw or not graph_id:
        return LinkResult(field=field, term=term, status="skipped", hits=[])

    if field == "brand":
        return _link_brand(platform, graph_id, raw)
    if field == "availability":
        return _link_availability(
            platform, graph_id, raw, top_k=top_k, min_score=min_score, score_delta=score_delta
        )

    label = _field_to_label(platform, field, graph_id)
    if not label:
        return LinkResult(field=field, term=term, status="skipped", hits=[])

    aliases = load_aliases(platform)
    mapped = apply_alias(raw, aliases)
    if mapped == "":
        return LinkResult(field=field, term=term, status="skipped", hits=[])
    search_term = mapped or raw
    alias_applied = search_term.casefold() != raw.casefold()

    hits = _exact_and_fulltext(platform, graph_id=graph_id, label=label, term=search_term, top_k=top_k * 2)

    term_nn = normalize_name(search_term)
    if label == "Material":
        hits = [_adjust_material_score(h, term_nn) for h in hits]
        hits.sort(key=lambda h: (-h.score, h.normalized_name))
    hits = _prefer_term_support(hits, term_nn)

    if alias_applied:
        for i, hit in enumerate(hits):
            if hit.source != "alias":
                hits[i] = LinkHit(hit.label, hit.id, hit.name, hit.normalized_name, hit.score, "alias")

    status, chosen = _select_hits(hits, top_k=top_k, min_score=min_score, score_delta=score_delta)
    # Plan: if k>5 after delta → clarify (no_match)
    if status == "no_match" and len(chosen) > 5:
        return LinkResult(field=field, term=term, status="no_match", hits=[])
    return LinkResult(field=field, term=term, status=status, hits=chosen)


def link_by_label(
    platform,
    label: str,
    term: str,
    *,
    graph_id: str | None = None,
) -> LinkResult:
    """KG API helper: resolve a term against one Neo4j facet/entity label (no LLM)."""

    settings = getattr(platform, "settings", None)
    top_k = int(getattr(settings, "entity_linker_top_k", 5) or 5)
    min_score = float(getattr(settings, "entity_linker_min_score", 0.35) or 0.35)
    score_delta = float(getattr(settings, "entity_linker_score_delta", 0.08) or 0.08)
    graph_id = _resolve_graph_id(platform, graph_id)
    raw = str(term or "").strip()
    safe = re.sub(r"[^A-Za-z0-9_]", "", str(label or ""))
    if not raw or not graph_id or not safe:
        return LinkResult(field=str(label or ""), term=term, status="skipped", hits=[])
    hits = _exact_and_fulltext(platform, graph_id=graph_id, label=safe, term=raw, top_k=top_k * 2)
    status, chosen = _select_hits(hits, top_k=top_k, min_score=min_score, score_delta=score_delta)
    return LinkResult(field=str(label or ""), term=term, status=status, hits=chosen)


def link_entities(
    platform,
    entities: dict[str, Any],
    *,
    graph_id: str | None = None,
) -> list[LinkResult]:
    """Link extracted entity fields. ``entities`` values may be str or list[str]."""

    graph_id = _resolve_graph_id(platform, graph_id)
    try:
        from fashion_brain.agents.entity_extract import is_catalog_scope_term
    except Exception:  # pragma: no cover

        def is_catalog_scope_term(term: str | None, scope_tokens=None) -> bool:  # type: ignore[misc]
            return False

    from knowledge_graph.retrieval.runtime_schema import build_runtime_schema

    scope = build_runtime_schema(platform, graph_id).scope_tokens if graph_id else []
    results: list[LinkResult] = []
    for field in ("brand", "category", "material", "color", "availability", "country"):
        value = entities.get(field)
        if value in (None, "", []):
            continue
        values = value if isinstance(value, list) else [value]
        for item in values:
            text = str(item or "").strip()
            if not text:
                continue
            try:
                scoped = is_catalog_scope_term(text, scope_tokens=scope)
            except TypeError:
                scoped = is_catalog_scope_term(text)
            if scoped:
                continue
            results.append(link_one(platform, field, text, graph_id=graph_id))
    return results


def resolved_payload(results: list[LinkResult]) -> dict[str, Any]:
    """Compact block for Text2Cypher prompts."""

    out: dict[str, Any] = {}
    for result in results:
        if result.status in {"matched", "ambiguous"} and result.hits:
            out[result.field] = {
                "term": result.term,
                "status": result.status,
                "ids": result.ids(),
                "names": [h.name for h in result.hits],
                "normalized_names": [h.normalized_name for h in result.hits],
            }
        elif result.status == "no_match":
            out[result.field] = {"term": result.term, "status": "no_match", "ids": []}
    return out


def clarification_message(
    results: list[LinkResult],
    *,
    require_hard_match: bool = False,
    schema=None,
) -> str | None:
    """Ask user to rephrase only for hard named-entity misses.

    Default is soft: unmatched filters are dropped by the caller and Text2Cypher continues.
    Hard clarify only when require_hard_match=True (user named a specific catalog value).
    """

    try:
        from fashion_brain.agents.entity_extract import is_catalog_scope_term, is_reserved_term
    except Exception:  # pragma: no cover

        def is_catalog_scope_term(term: str | None) -> bool:  # type: ignore[misc]
            return False

        def is_reserved_term(term: str | None, _schema=None) -> bool:  # type: ignore[misc]
            return False

    misses = [
        r
        for r in results
        if r.status == "no_match"
        and not is_catalog_scope_term(r.term)
        and not is_reserved_term(r.term, schema)
    ]
    if not misses:
        return None
    if not require_hard_match:
        # Soft path: pipeline drops these filters and answers without aborting.
        return None
    parts = []
    for miss in misses:
        parts.append(f"I could not match {miss.field} '{miss.term}' to a known catalog value.")
    parts.append("Please rephrase using a brand/category/material/color from the catalog, or pick a closer spelling.")
    return " ".join(parts)


class EntityLinker:
    """Facade used by Fashion Brain / KG API."""

    def link(self, platform, entities: dict[str, Any], *, graph_id: str | None = None) -> list[LinkResult]:
        return link_entities(platform, entities, graph_id=graph_id)

    def brands(self, platform, graph_id: str | None = None) -> list[dict[str, str]]:
        return list_live_brands(platform, graph_id)
