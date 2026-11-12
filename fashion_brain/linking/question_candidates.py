"""Fashion Brain: question n-grams → KG graph lookup candidates.

Uses KG public APIs only (``link_one``, ``link_by_label``). No LLM here.
"""

from __future__ import annotations

import re
from typing import Any

from knowledge_graph.linking.entity_linker import LinkResult, link_by_label, link_one
from knowledge_graph.ontology import normalize_name


def _facet_field_key(label: str) -> str:
    key = str(label or "").casefold()
    if "brand" in key:
        return "brand"
    if "categor" in key or "garment" in key:
        return "category"
    if "material" in key or "fabric" in key:
        return "material"
    if "color" in key or "colour" in key:
        return "color"
    if "avail" in key:
        return "availability"
    if "famil" in key and "sub" in key:
        return "subfamily"
    if "famil" in key:
        return "family"
    if "country" in key:
        return "country"
    if "section" in key:
        return "section"
    return key.replace(" ", "_")


def link_question_candidates(
    platform,
    question: str,
    *,
    graph_id: str | None = None,
    max_terms: int = 12,
) -> list[LinkResult]:
    """N-grams from the question → KG full-text/exact hits (schema-dynamic)."""

    from fashion_brain.planning.runtime_schema import get_runtime_schema, resolve_product_graph_id

    graph_id = graph_id or resolve_product_graph_id(platform)
    if not graph_id or not (question or "").strip():
        return []

    ctx = get_runtime_schema(platform, graph_id)
    reserved: set[str] = set(normalize_name(t) for t in (ctx.scope_tokens or []) if t)
    for key in ctx.filter_keys or []:
        nn = normalize_name(str(key))
        if nn:
            reserved.add(nn)
            reserved.update(nn.split("_"))
            reserved.add(nn + "s")
    for label in list(ctx.facet_types or []) + list(ctx.entity_types or []):
        for part in re.findall(r"[A-Za-z]+", str(label)):
            p = part.casefold()
            if len(p) >= 3:
                reserved.add(p)
                reserved.add(p + "s")
    for prop in ctx.product_properties or []:
        nn = normalize_name(str(prop))
        reserved.update(p for p in nn.replace("-", "_").split("_") if len(p) >= 3)
    reserved |= {
        "how", "many", "the", "a", "an", "and", "or", "in", "of", "to", "for",
        "from", "with", "is", "are", "do", "we", "what", "which", "who", "when",
        "where", "show", "list", "all", "any", "each", "per", "by", "on", "at",
    }

    tokens = [t for t in normalize_name(question).split() if t]
    terms: list[str] = []
    seen: set[str] = set()
    for n in (3, 2, 1):
        for i in range(len(tokens) - n + 1):
            chunk = tokens[i : i + n]
            if any(t in reserved for t in chunk):
                continue
            if n == 1 and len(chunk[0]) < 3:
                continue
            phrase = " ".join(chunk)
            if phrase in seen:
                continue
            seen.add(phrase)
            terms.append(phrase)
            if len(terms) >= max_terms:
                break
        if len(terms) >= max_terms:
            break

    # Map schema labels → (field_key_or_label, use_link_one?)
    core_fields = {"brand", "category", "material", "color", "availability"}
    label_targets: list[tuple[str, str]] = []  # (key, neo4j_label_or_field)
    for label in ctx.facet_types or []:
        fk = _facet_field_key(label)
        label_targets.append((fk, fk if fk in core_fields else str(label)))
    for label in ctx.entity_types or []:
        fk = _facet_field_key(label)
        if fk in {t[0] for t in label_targets}:
            continue
        if fk in {"family", "subfamily", "country", "section", "brand"}:
            label_targets.append((fk, str(label)))

    results: list[LinkResult] = []
    matched_keys: set[str] = set()
    for term in terms:
        for key, target in label_targets:
            if key in matched_keys and key != "brand":
                continue
            if target in core_fields:
                result = link_one(platform, target, term, graph_id=graph_id)
            else:
                result = link_by_label(platform, target, term, graph_id=graph_id)
                result = LinkResult(field=key, term=result.term, status=result.status, hits=result.hits)
            if result.status in {"matched", "ambiguous"} and result.hits:
                results.append(result)
                matched_keys.add(key)
                break
    return results


def candidates_payload(results: list[LinkResult]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in results:
        if r.status not in {"matched", "ambiguous"} or not r.hits:
            continue
        out.append(
            {
                "field": r.field,
                "term": r.term,
                "status": r.status,
                "ids": r.ids(),
                "names": [h.name for h in r.hits],
                "labels": [h.label for h in r.hits],
            }
        )
    return out
