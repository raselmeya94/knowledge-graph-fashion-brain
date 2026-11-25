"""Live-schema grounding helpers — no hardcoded brands, materials, or graph ids.

Uses runtime registry / live Neo4j schema / ontology tokens to decide whether a
question belongs in the Knowledge Graph versus general-knowledge ``direct``.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from knowledge_graph.ontology import normalize_name

# Structural Cypher noise — never treat as domain catalog hits by themselves.
_STRUCTURAL_LABELS = frozenset(
    {
        "entity",
        "sourceentity",
        "canonical",
        "node",
        "nodes",
        "relationship",
        "relationships",
    }
)

_TOKEN_STOP = frozenset(
    normalize_name(
        """
        how many are is a an the of in on our with and or for from at that this be any all
        every each number count total what which who when where list has have had was were
        been do we you your me please could would them those these their same also just show
        give want let us again still about as can around roughly approx approximately most
        common usually typically often frequently common usual usualy details detail
        """
    ).split()
)


def allowed_product_graph_ids(platform: Any | None) -> list[str]:
    """Active product-kind graph ids from the live KG registry (dynamic)."""

    if platform is None:
        return []
    try:
        from knowledge_graph.retrieval.ecom import registry_product_graphs

        return list(registry_product_graphs(platform) or [])
    except Exception:
        try:
            graphs = list(platform.kg_registry.list_graphs())
        except Exception:
            return []
        return [
            str(item.id)
            for item in graphs
            if getattr(item, "kind", None) == "product"
            and getattr(item, "status", "active") == "active"
            and getattr(item, "id", None)
        ]


def sanitize_graph_param(value: Any, allowed: Iterable[str]) -> str | None:
    """Return value only when it is an exact allowed product-graph id."""

    text = str(value or "").strip()
    if not text:
        return None
    allow = {str(item) for item in allowed if item}
    return text if text in allow else None


def question_matches_catalog_schema(
    question: str,
    platform: Any | None = None,
    *,
    graph_ids: list[str] | None = None,
) -> bool:
    """True when any question token/phrase overlaps live labels, rels, or properties.

    Used to block ``direct`` (no-KG) routing for domain attribute asks.
    """

    text = normalize_name(question or "")
    if not text.strip():
        return False
    tokens = [tok for tok in text.split() if tok and tok not in _TOKEN_STOP and len(tok) > 2]
    if not tokens:
        return False
    catalog = _schema_token_catalog(platform, tuple(graph_ids or ()))
    if not catalog:
        # Fail closed toward catalog when registry/schema unavailable but question
        # already looks like a product-domain ask via property-ish stems.
        return bool(
            re.search(
                r"\b(product|products|catalog|assortment|brand|brands|buyer|buyers|"
                r"material|materials|fabric|price|prices|availability|category|"
                r"categories|construction|fit|composition|color|colour|garment)\b",
                text,
            )
        )
    # Exact token hits.
    for tok in tokens:
        if tok in catalog or (tok.endswith("s") and tok[:-1] in catalog):
            return True
        if len(tok) > 4 and any(tok in key or key in tok for key in catalog if len(key) >= 4):
            # Soft substring for multi-word props collapsed by normalize_name.
            return True
    # Fuzzy / n-gram style: join adjacent tokens and test against catalog keys.
    for width in (3, 2):
        for i in range(0, max(0, len(tokens) - width + 1)):
            phrase = " ".join(tokens[i : i + width])
            compact = phrase.replace(" ", "")
            if phrase in catalog or compact in catalog:
                return True
            for key in catalog:
                if len(key) < 4:
                    continue
                if phrase in key or key in phrase or compact in key.replace(" ", ""):
                    return True
                if _fuzzy_hit(phrase, key) or _fuzzy_hit(compact, key.replace(" ", "")):
                    return True
    return False


def _fuzzy_hit(a: str, b: str, *, threshold: float = 0.86) -> bool:
    if not a or not b or abs(len(a) - len(b)) > max(3, len(b) // 3):
        return False
    try:
        from rapidfuzz import fuzz

        return float(fuzz.ratio(a, b)) >= threshold * 100.0
    except Exception:
        import difflib

        return difflib.SequenceMatcher(None, a, b).ratio() >= threshold


def _schema_token_catalog(platform: Any | None, graph_ids: tuple[str, ...]) -> frozenset[str]:
    """Build a frozenset of normalized schema tokens for matching."""

    tokens: set[str] = set()
    ids = list(graph_ids) or allowed_product_graph_ids(platform)
    # Live schema snapshot (labels / props / rels).
    try:
        from knowledge_graph.retrieval.live_schema import get_live_schema

        snap = get_live_schema(platform) if platform is not None else None
        if snap is not None:
            for lab in getattr(snap, "labels", None) or []:
                _add_schema_token(tokens, lab)
            for rel in getattr(snap, "relationship_types", None) or []:
                _add_schema_token(tokens, rel)
                # HAS_BRAND → brand
                if "_" in str(rel):
                    parts = str(rel).split("_")
                    if parts:
                        _add_schema_token(tokens, parts[-1])
            props = getattr(snap, "node_properties", None) or {}
            if isinstance(props, dict):
                for lab, fields in props.items():
                    _add_schema_token(tokens, lab)
                    for field in fields or []:
                        _add_schema_token(tokens, field)
            for gid in getattr(snap, "graph_values", None) or []:
                _add_schema_token(tokens, gid)
    except Exception:
        pass
    # Runtime schema cards per product graph.
    for gid in ids[:8]:
        try:
            from fashion_brain.planning.runtime_schema import get_runtime_schema

            schema = get_runtime_schema(platform, gid)
            for raw in list(getattr(schema, "labels", None) or []) + list(
                getattr(schema, "relationship_types", None) or []
            ):
                _add_schema_token(tokens, raw)
            for raw in list(getattr(schema, "attributes", None) or []) + list(
                getattr(schema, "facet_types", None) or []
            ):
                _add_schema_token(tokens, raw)
        except Exception:
            continue
    # Ontology registry entity types / surface aliases (not hardcoded value names).
    try:
        from knowledge_graph.ontology_registry import get_ontology_registry

        reg = get_ontology_registry(platform)
        for entry_list in (getattr(reg, "canonical_index", None) or {}).values():
            for entry in entry_list or []:
                _add_schema_token(tokens, getattr(entry, "entity_type", None))
        allow = getattr(reg, "allowlist_for_domains", None)
        if callable(allow):
            for rel, pairs in (allow() or {}).items():
                _add_schema_token(tokens, rel)
                for pair in pairs or []:
                    if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                        _add_schema_token(tokens, pair[0])
                        _add_schema_token(tokens, pair[1])
    except Exception:
        pass
    return frozenset(tokens)


def _add_schema_token(bucket: set[str], raw: Any) -> None:
    text = normalize_name(str(raw or ""))
    if not text or text in _STRUCTURAL_LABELS:
        return
    bucket.add(text)
    bucket.add(text.replace(" ", ""))
    bucket.add(text.replace("_", " "))
    bucket.add(text.replace("_", ""))
    # Split snake / camel stems.
    for part in re.split(r"[_\s]+", text):
        if part and part not in _STRUCTURAL_LABELS and len(part) > 2:
            bucket.add(part)
            if part.endswith("s") and len(part) > 3:
                bucket.add(part[:-1])


# Query-shape classifiers (schema-agnostic linguistic shapes; values come from schema later).


def detect_query_shape(question: str, *, schema_attrs: Iterable[str] | None = None) -> str | None:
    """Return a planner shape id or None.

    Shapes:
      - facet_listing: products filtered by a target entity/attribute value
      - entity_frequency: attributes appearing on >N products / shared values
      - category_rollup: price/event rollup by category hierarchy
      - volatility: products ranked by absolute price-change magnitude
      - state_transition: from-state → to-state over snapshot/event history
      - shared_overlap: cross-catalog shared attribute values / linked products
    """

    text = normalize_name(question or "")
    if not text:
        return None
    attrs = {normalize_name(a) for a in (schema_attrs or []) if a}

    if re.search(r"\b(share|shared|across (catalogs?|sources?|graphs?)|between .+ and|interconnect)\b", text):
        if re.search(r"\b(canonical|ontology)\b", text) and re.search(
            r"\b(brand|buyer|party)\b", text
        ):
            # Canonical linking metadata — not shared domain attribute values.
            pass
        else:
            return "shared_overlap"

    if re.search(r"\b(from)\b.+\b(to)\b", text) and re.search(
        r"\b(availability|stock|price|status|state|snapshot|history)\b", text
    ):
        return "state_transition"

    if re.search(
        r"\b(swing|swings|swinging|volatil|fluctuat|up and down|largest (moves?|changes?)|"
        r"most (volatile|unstable)|price (moves?|movement|movements))\b",
        text,
    ):
        return "volatility"

    if re.search(r"\b(categor(?:y|ies)|family|families|section|department)\b", text) and re.search(
        r"\b(cheaper|cheapening|getting cheaper|price (drop|drops|decrease|decreases|trend|trends)|"
        r"decreasing|dropping)\b",
        text,
    ):
        return "category_rollup"

    if re.search(
        r"\b(more than one|across .+ products?|appear on|shows? up on|shared by|"
        r"common (?:\/ )?canonical|most (common|often)|frequency|co[- ]?occur)\b",
        text,
    ) and (
        re.search(r"\b(material|materials|attribute|attributes|value|values|facet|facets)\b", text)
        or (attrs and any(a in text for a in attrs))
    ):
        return "entity_frequency"

    if re.search(r"\b(which|what|show|list|find)\b.+\bproducts?\b", text) and (
        re.search(r"\b(mainly|mostly|primarily|made of|using|with|share|shared)\b", text)
        or (attrs and any(a in text for a in attrs))
    ):
        return "facet_listing"

    return None
