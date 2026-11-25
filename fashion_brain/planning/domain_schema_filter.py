"""Question-scoped schema pruning from live Neo4j + registry tokens (zero hardcoding)."""

from __future__ import annotations

import re
from typing import Any

from fashion_brain.planning.schema_cache import get_cached, set_cached


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9_]+", (text or "").casefold()) if len(t) >= 2}


def _live_domain_tokens(platform, live) -> dict[str, set[str]]:
    """Map graph_id / domain name → descriptive tokens from registry + live values."""

    buckets: dict[str, set[str]] = {}
    for gid in list(getattr(live, "graph_values", None) or []):
        buckets[str(gid)] = _tokens(str(gid).replace("_", " ").replace("-", " "))
    for name in list(getattr(live, "domain_names", None) or []):
        buckets.setdefault(f"domain:{name}", set()).update(_tokens(str(name)))

    # Enrich from kg registry records when available (capabilities / description).
    try:
        registry = getattr(platform, "kg_registry", None)
        rows = []
        if registry is not None:
            lister = getattr(registry, "list_graphs", None) or getattr(registry, "graphs", None)
            if callable(lister):
                rows = list(lister() or [])
            elif isinstance(lister, list):
                rows = list(lister)
        for row in rows:
            if hasattr(row, "__dict__") and not isinstance(row, dict):
                data = {
                    "graph_id": getattr(row, "graph_id", None) or getattr(row, "id", None),
                    "name": getattr(row, "name", None),
                    "description": getattr(row, "description", None),
                    "capabilities": getattr(row, "capabilities", None),
                    "kind": getattr(row, "kind", None),
                }
            else:
                data = dict(row or {})
            gid = str(data.get("graph_id") or data.get("id") or "").strip()
            if not gid:
                continue
            bag = buckets.setdefault(gid, set())
            bag.update(_tokens(str(data.get("name") or "")))
            bag.update(_tokens(str(data.get("description") or "")))
            bag.update(_tokens(str(data.get("kind") or "")))
            for cap in data.get("capabilities") or []:
                bag.update(_tokens(str(cap)))
    except Exception:
        pass
    return buckets


def classify_ontology_domains(platform, question: str, *, limit: int = 3) -> list[Any]:
    """Score question against ontology JSON name+description (runtime discovery)."""

    try:
        from knowledge_graph.ontology_registry import get_ontology_registry

        return get_ontology_registry(platform).domain_for_question(question, limit=limit)
    except Exception:
        return []


def classify_question_domains(
    platform,
    question: str,
    *,
    preferred_graphs: list[str] | None = None,
) -> list[str]:
    """Return relevant graph/domain keys for this question.

    Prefers ontology JSON name+description when a registry is available, then
    falls back to live Neo4j / registry graph tokens (existing behaviour).
    """

    out: list[str] = []
    for gid in preferred_graphs or []:
        if gid and gid not in out:
            out.append(str(gid))

    for profile in classify_ontology_domains(platform, question, limit=3):
        oid = getattr(profile, "ontology_id", None) or ""
        if oid and oid not in out:
            out.append(str(oid))

    from knowledge_graph.retrieval.live_schema import get_live_schema

    live = get_live_schema(platform)
    if live is None:
        return out or list(preferred_graphs or [])

    q_tokens = _tokens(question)
    buckets = _live_domain_tokens(platform, live)
    scored: list[tuple[int, str]] = []
    for key, bag in buckets.items():
        overlap = len(q_tokens & bag)
        # Prefer preferred graphs even with weak overlap
        if preferred_graphs and key in preferred_graphs:
            overlap += 3
        if overlap:
            scored.append((overlap, key))
    scored.sort(key=lambda item: (-item[0], item[1]))
    for _, key in scored[:4]:
        if key.startswith("domain:"):
            continue
        if key not in out:
            out.append(key)
    if preferred_graphs:
        for gid in preferred_graphs:
            if gid not in out:
                out.insert(0, gid)
    if not out:
        out = list(live.graph_values or preferred_graphs or [])[:3]
    return out


def _label_relevance(label: str, question_tokens: set[str], domain_keys: list[str]) -> bool:
    lab = label.casefold()
    lab_tokens = _tokens(re.sub(r"([a-z])([A-Z])", r"\1 \2", label))
    if question_tokens & lab_tokens:
        return True
    # Always keep structural / product hub labels
    if lab in {
        "entity",
        "product",
        "brand",
        "buyer",
        "material",
        "canonical",
        "sourceentity",
        "domain",
        "pricechangeevent",
        "pricesnapshot",
        "availabilitysnapshot",
    }:
        return True
    # Keep labels that mention selected domain/graph tokens
    for key in domain_keys:
        if _tokens(key.replace("domain:", " ")) & lab_tokens:
            return True
    # Price / availability / inventory / shipment / certification cues from question
    cues = {
        "price": ("price", "snapshot", "change", "listing"),
        "stock": ("avail", "inventory", "stock"),
        "ship": ("ship", "order", "buyer"),
        "material": ("material", "compos", "fabric", "canonical", "source"),
        "plm": ("plm", "gauge", "knit", "machine", "sample"),
        "cert": ("cert", "complian", "standard"),
    }
    for cue, needles in cues.items():
        if cue in question_tokens or any(n in question_tokens for n in needles):
            if any(n in lab for n in needles) or cue in lab:
                return True
    # Default: keep common retail facets when question is broad
    if lab in {"color", "colour", "category", "garmentcategory", "availability", "family"}:
        return True
    return False


def _rel_relevance(rel: str, question_tokens: set[str], keep_labels: set[str]) -> bool:
    r = rel.casefold()
    if r in {"resolves_to", "has_brand", "has_buyer", "has_material", "belongs_to_category"}:
        return True
    if any(tok in r for tok in question_tokens if len(tok) >= 4):
        return True
    if "price" in question_tokens and "price" in r:
        return True
    if "avail" in r or "stock" in r:
        return "stock" in question_tokens or "avail" in question_tokens or "availability" in question_tokens
    # Keep rels that typically connect kept labels (heuristic by name)
    return any(lab.casefold() in r for lab in keep_labels if len(lab) >= 4)


def prune_live_schema_for_question(
    platform,
    question: str,
    *,
    preferred_graphs: list[str] | None = None,
) -> dict[str, Any]:
    """Return pruned label/rel/prop sets + compact card text."""

    from knowledge_graph.retrieval.live_schema import get_live_schema

    live = get_live_schema(platform)
    domains = classify_question_domains(platform, question, preferred_graphs=preferred_graphs)
    cache_key = f"prune:{'|'.join(domains)}:{hash(question) & 0xFFFFFFFF}"
    hit = get_cached(cache_key, platform)
    if isinstance(hit, dict) and hit.get("card"):
        return hit

    if live is None:
        payload = {
            "domains": domains,
            "labels": [],
            "relationship_types": [],
            "card": "",
            "source": "unavailable",
        }
        return payload

    q_tokens = _tokens(question)
    keep_labels = {lab for lab in live.labels if _label_relevance(lab, q_tokens, domains)}
    # Always include Product + Entity if present
    for required in ("Product", "Entity", "Brand", "Material", "Canonical", "SourceEntity"):
        if required in live.labels:
            keep_labels.add(required)
    keep_rels = {rel for rel in live.relationship_types if _rel_relevance(rel, q_tokens, keep_labels)}

    ontology_profiles = classify_ontology_domains(platform, question, limit=3)
    ontology_ids = [getattr(p, "ontology_id", "") for p in ontology_profiles if getattr(p, "ontology_id", "")]
    allowlist: dict[str, list[list[str]]] = {}
    try:
        from knowledge_graph.ontology_registry import get_ontology_registry

        reg = get_ontology_registry(platform)
        keys = ontology_ids or None
        ont_labels = reg.entity_types_for_domains(keys)
        allowlist = reg.allowlist_for_domains(keys)
        if ont_labels:
            # Intersect live ∩ ontology entity types (additive keep of hubs already done)
            intersected = {lab for lab in keep_labels if lab in ont_labels}
            if len(intersected) >= 3:
                keep_labels = intersected
                for required in ("Product", "Entity", "Brand", "Material", "Canonical", "SourceEntity"):
                    if required in live.labels:
                        keep_labels.add(required)
        if allowlist:
            allow_rels = {r.upper() for r in allowlist.keys()}
            intersected_rels = {
                rel for rel in keep_rels if rel.upper() in allow_rels or rel == "RESOLVES_TO"
            }
            if intersected_rels:
                keep_rels = intersected_rels
                if "RESOLVES_TO" in live.relationship_types:
                    keep_rels.add("RESOLVES_TO")
    except Exception:
        allowlist = {}

    # Fail-open: if pruning too aggressive, keep more labels (existing behaviour)
    source = "live+ontology" if ontology_ids else "live"
    if len(keep_labels) < 5:
        keep_labels = set(live.labels[:40])
        keep_rels = set(live.relationship_types[:40])
        source = "live_fail_open"

    card = live.compact_card(keep_labels=keep_labels, keep_rels=keep_rels)
    header_bits: list[str] = []
    if ontology_profiles:
        names = [
            f"{getattr(p, 'name', '')} ({getattr(p, 'ontology_id', '')})" for p in ontology_profiles
        ]
        header_bits.append("Ontology domains: " + ", ".join(names))
    if domains:
        header_bits.append("Relevant graphs/domains for this ask: " + ", ".join(domains))
    if header_bits:
        card = "\n".join(header_bits) + "\n" + card

    payload = {
        "domains": domains,
        "ontology_domains": ontology_ids,
        "labels": sorted(keep_labels),
        "relationship_types": sorted(keep_rels),
        "card": card,
        "source": source,
        "allowlist": allowlist,
    }
    set_cached(cache_key, payload)
    return payload


def pruned_schema_card(
    platform,
    question: str,
    *,
    preferred_graphs: list[str] | None = None,
    fallback_card: str = "",
) -> str:
    """Schema card for Text2Cypher: pruned live schema + optional ontology fallback."""

    try:
        pruned = prune_live_schema_for_question(
            platform, question, preferred_graphs=preferred_graphs
        )
        card = str(pruned.get("card") or "").strip()
        if card and fallback_card:
            return card + "\n\nOntology/runtime hints:\n" + fallback_card[:2500]
        if card:
            return card
    except Exception:
        pass
    return fallback_card or ""
