"""Existing-relationship lookup and new-relationship proposals."""

from __future__ import annotations

import hashlib
import json
import re

from knowledge_graph.construction.identity import NEWS_KINDS
from knowledge_graph.construction.news_facts import (
    extract_trade_facts,
    normalize_news_relationship,
    ontology_gap_warning,
    reject_news_relationship,
)
from knowledge_graph.errors import ValidationFailed


def parse_model_json(text: str):
    raise ValidationFailed("LLM JSON parse removed from Knowledge Graph.")
from knowledge_graph.errors import ModelUnavailableError
from knowledge_graph.kg_registry import default_graph_for_kind
from knowledge_graph.ontology import Ontology, normalize_name

_TYPE_ALIASES = {
    "person": "Person",
    "people": "Person",
    "author": "Person",
    "organization": "Organization",
    "org": "Organization",
    "company": "Company",
    "brand": "Brand",
    "supplier": "Supplier",
    "topic": "Topic",
    "subject": "Topic",
    "concept": "Topic",
    "theme": "Topic",
    "role": "Person",
    "actor": "Person",
    "consumer": "Segment",
    "customers": "Segment",
    "industry": "Organization",
    "location": "Location",
    "place": "Location",
    "event": "Event",
    "product": "Product",
    "material": "Material",
    "fabric": "Fabric",
    "document": "Document",
    "email": "Document",
    "article": "Document",
    "newsletter": "Document",
    "season": "Season",
    "category": "GarmentCategory",
    "trend": "Trend",
    "policy": "Policy",
    "country": "Country",
    "region": "Region",
    "market": "Market",
    "publisher": "Publisher",
}

_FROM = re.compile(r"(?im)^from:\s*(.+?)\s*$")
_TO = re.compile(r"(?im)^to:\s*(.+?)\s*$")
_PUBLISHED = re.compile(r"(?i)\bpublished by\s+([^\n.]{2,80})")
_AUTHORED = re.compile(r"(?i)\b(?:authored|written) by\s+([^\n.]{2,80})")
_ADDRESS = re.compile(r"^(.*?)\s*<([^>]+)>\s*$")

_JSON_FIELDS = {
    "author": ("Person", "AUTHORED_BY"),
    "from": ("Person", "SENT_BY"),
    "sender": ("Person", "SENT_BY"),
    "to": ("Person", "SENT_TO"),
    "recipient": ("Person", "SENT_TO"),
    "publisher": ("Organization", "PUBLISHED_BY"),
    "brand": ("Organization", "PUBLISHED_BY"),
    "topic": ("Topic", "ABOUT"),
    "about": ("Topic", "ABOUT"),
    "location": ("Location", "MENTIONS"),
    "city": ("Location", "MENTIONS"),
    "material": ("Material", "MENTIONS"),
    "organization": ("Organization", "MENTIONS"),
    "company": ("Organization", "MENTIONS"),
}


class ListRelationshipsTool:
    name = "list_relationships"
    description = "Read relationships already stored in the knowledge graph."

    def run(self, payload: dict, ctx) -> dict:
        return {"relationships": list_existing(ctx.graph, payload.get("graph"))}


class IdentifyRelationshipsTool:
    name = "identify_relationships"
    description = (
        "Compare extracted content with existing graph relationships and propose new relationships. "
        "Allowlisted edges are ready to approve; unlisted but evidenced edges are flagged for ontology review. "
        "Uses Ollama when it is reachable, then keeps the deterministic proposals."
    )

    def run(self, payload: dict, ctx) -> dict:
        title = str(payload.get("title") or "Untitled document").strip()
        text = str(payload.get("text") or "")
        kind = str(payload.get("kind") or payload.get("source_kind") or "text")
        graph_id = str(payload.get("graph_id") or "").strip() or default_graph_for_kind(kind)
        ontology = ctx.ontology_for(graph_id)
        ontology_id = ctx.platform.kg_registry.get_graph(graph_id).ontology_id
        news = ontology_id == "news" or "EXPORTS_TO" in ontology.relationship_allowlist or kind in NEWS_KINDS
        warnings: list[str] = []
        existing = payload.get("existing")
        if existing is None:
            existing = list_existing(ctx.graph, graph_id)
        relationship_names = (
            sorted(ontology.relationship_allowlist) if news else _relationship_names(existing)
        )
        proposals = _deterministic(kind, title, text, payload.get("structured"), ontology)
        if news:
            for item in proposals:
                if item["type"] == "PUBLISHED_BY" and not _pair_allowed(
                    ontology, item["type"], item["start_type"], item["end_type"]
                ):
                    item["end_type"] = "Publisher"
            proposals.extend(extract_trade_facts(title, text))
        if payload.get("use_model", True) and text.strip():
            try:
                proposals.extend(_from_ollama(ctx, title, text, ontology, relationship_names, news=news))
            except ModelUnavailableError as exc:
                warnings.append(exc.message)
        known = {
            (item["type"], normalize_name(item.get("start_name") or ""), normalize_name(item.get("end_name") or ""))
            for item in existing
        }
        merged: list[dict] = []
        seen: set[tuple[str, str, str]] = set()
        for item in proposals:
            # Trade-fact rejects apply to news articles, not email/document header edges
            # (evidence is often "From header" / "To header", not a body quote).
            if news and item.get("start_type") != "Document":
                normalize_news_relationship(item)
                reason = reject_news_relationship(item, ontology)
                if reason:
                    warnings.append(reason)
                    continue
            gap = ontology_gap_warning(item, ontology)
            if gap:
                item["needs_ontology_approval"] = True
                warnings.append(gap)
            else:
                item["needs_ontology_approval"] = False
            key = (item["type"], normalize_name(item["start_name"]), normalize_name(item["end_name"]))
            if not key[1] or not key[2] or key[1] == key[2] or key in seen:
                continue
            seen.add(key)
            item["status"] = "existing" if key in known else "new"
            item["id"] = _rel_id(key)
            merged.append(item)
        return {
            "relationships": merged,
            "warnings": warnings,
            "graph_id": graph_id,
            "ontology_id": ontology_id,
        }


def list_existing(graph, graph_name: str | None = None) -> list[dict]:
    nodes = {node.id: node for node in graph.find_nodes(include_deleted=False)}
    rows = []
    for edge in graph.find_edges(include_deleted=False):
        if graph_name and edge.graph != graph_name:
            continue
        start = nodes.get(edge.start_id)
        end = nodes.get(edge.end_id)
        rows.append(
            {
                "id": edge.id,
                "type": edge.type,
                "start_id": edge.start_id,
                "end_id": edge.end_id,
                "start_name": _node_name(start),
                "end_name": _node_name(end),
                "start_type": _node_type(start),
                "end_type": _node_type(end),
                "graph": edge.graph,
            }
        )
    return rows


def _node_name(node) -> str:
    if node is None:
        return ""
    return str(node.properties.get("name") or node.properties.get("title") or "")


def _node_type(node) -> str:
    if node is None:
        return ""
    return str(node.properties.get("entity_type") or "")


def _deterministic(kind: str, title: str, text: str, structured, ontology: Ontology) -> list[dict]:
    found: list[dict] = []
    if kind == "email":
        sender = _party(_first(_FROM, text))
        if sender:
            found.append(_doc_rel("SENT_BY", title, sender, "Person", "From header", "deterministic"))
        for raw in _split_parties(_first(_TO, text)):
            party = _party(raw)
            if party:
                found.append(_doc_rel("SENT_TO", title, party, "Person", "To header", "deterministic"))
    for match in _PUBLISHED.finditer(text):
        name = _clean_name(match.group(1))
        if name:
            found.append(_doc_rel("PUBLISHED_BY", title, name, "Organization", match.group(0).strip(), "deterministic"))
    for match in _AUTHORED.finditer(text):
        name = _clean_name(match.group(1))
        if name:
            found.append(_doc_rel("AUTHORED_BY", title, name, "Person", match.group(0).strip(), "deterministic"))
    if isinstance(structured, dict):
        found.extend(_from_structured(title, structured, ontology))
    return found


def _from_structured(title: str, data: dict, ontology: Ontology) -> list[dict]:
    found: list[dict] = []
    rows = [data]
    for item in data.get("items") or []:
        if isinstance(item, dict):
            rows.append(item)
    for row in rows:
        for key, (entity_type, rel_type) in _JSON_FIELDS.items():
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                found.append(_doc_rel(rel_type, title, value.strip(), entity_type, f"{key}: {value.strip()}", "deterministic"))
    for item in data.get("relationships") or []:
        if not isinstance(item, dict):
            continue
        rel_type = str(item.get("type") or "").strip().upper()
        end_name = str(item.get("target") or item.get("end") or item.get("end_name") or "").strip()
        end_type = _canonical_type(str(item.get("target_type") or item.get("end_type") or "Topic"), ontology)
        start_name = str(item.get("source") or item.get("start") or item.get("start_name") or "document")
        if start_name in {"document", "doc", title}:
            start_type = "Document"
            start_name = title
        else:
            start_type = _canonical_type(str(item.get("source_type") or item.get("start_type") or "Organization"), ontology)
        if rel_type and end_name and end_type:
            found.append(
                {
                    "type": rel_type,
                    "start_name": start_name,
                    "start_type": start_type,
                    "end_name": end_name,
                    "end_type": end_type,
                    "evidence": str(item.get("evidence") or ""),
                    "method": "deterministic",
                }
            )
    return found


def _relationship_names(existing: list[dict]) -> list[str]:
    return sorted({str(item.get("type") or "").strip() for item in existing if str(item.get("type") or "").strip()})


_NEWS_PROPERTIES = (
    "product",
    "value",
    "share_pct",
    "previous_share",
    "period",
    "yoy_pct",
    "rank",
    "market",
    "pct",
    "direction",
    "amount",
    "cert",
    "date",
    "claim",
    "season",
    "title",
)


def _from_ollama(
    ctx,
    title: str,
    text: str,
    ontology: Ontology,
    relationship_names: list[str],
    *,
    news: bool = False,
) -> list[dict]:
    pairs = []
    for rel_type in relationship_names:
        allowed = ontology.relationship_allowlist.get(rel_type, [])
        rendered = ", ".join(f"{pair[0]} -> {pair[1]}" for pair in allowed)
        pairs.append(f"{rel_type}: {rendered}" if rendered else rel_type)
    listed = "\n".join(pairs) if pairs else "(none yet)"
    if news:
        system = (
            "This is a garment-industry news article, not a PLM product row. "
            "Return a JSON object with keys entities and relationships. "
            "Each relationship has type, start, start_type, end, end_type, evidence, and properties. "
            "Prefer an allowlisted triple below. If a real fact in the text does not fit any allowlisted "
            "triple, still extract it with a clear UPPER_SNAKE type and correct entity types; a person "
            "will review it before it is added to the ontology. "
            "For each exporter or importer, extract one EXPORTS_TO or IMPORTS_FROM edge. "
            "Put product, value, share_pct, previous_share, yoy_pct, and period on that edge. "
            "A period is a property, never a node, and never the end of OCCURRED_IN. "
            "OCCURRED_IN is only Event -> Location. "
            "A rank is RANKED_IN to a Market, with rank and product. "
            "A quote is STATED_IN from the person to the Document. Speaking at an event is SPOKE_AT. "
            "PUBLISHED_BY ends at a Publisher. A data source is CITES_DATA_FROM from the Document. "
            "The evidence sentence must contain both entity names and every number you put in properties. "
            "Do not invent facts that are not in the text.\n"
            "Allowed triples:\n"
            f"{listed}"
        )
    else:
        system = (
            "You extract knowledge-graph relationships. Return a JSON object with keys entities and relationships. "
            "Each relationship has type, start, start_type, end, end_type, and evidence. "
            "Use a relationship name already stored in this source when it fits the text. "
            "If none of those names fit, create a new UPPER_SNAKE relationship name. "
            "The source document is start \"document\" with start_type Document. "
            "Do not invent facts that are not in the text.\n"
            "Relationship names for this source:\n"
            f"{listed}"
        )
    prompt = f"Document title: {title}\n\n{text[:12000]}"
    payload = parse_model_json(ctx.ollama.complete(prompt, system=system, json_mode=True))
    entities = {}
    for entity in payload.get("entities") or []:
        if not isinstance(entity, dict):
            continue
        name = str(entity.get("name") or "").strip()
        entity_type = _entity_type(str(entity.get("type") or ""), ontology, strict=news)
        if name and entity_type:
            entities[str(entity.get("id") or name)] = (name, entity_type)
            entities[normalize_name(name)] = (name, entity_type)
    found = []
    for rel in payload.get("relationships") or []:
        if not isinstance(rel, dict):
            continue
        rel_type = _relation_name(str(rel.get("type") or ""))
        start = _endpoint(
            rel.get("start") or rel.get("source") or "document",
            entities,
            title,
            ontology,
            str(rel.get("start_type") or rel.get("source_type") or ""),
            strict=news,
        )
        end = _endpoint(
            rel.get("end") or rel.get("target"),
            entities,
            title,
            ontology,
            str(rel.get("end_type") or rel.get("target_type") or ""),
            strict=news,
        )
        if not rel_type or start is None or end is None:
            continue
        properties = _relation_properties(rel)
        found.append(
            {
                "type": rel_type,
                "start_name": start[0],
                "start_type": start[1],
                "end_name": end[0],
                "end_type": end[1],
                "evidence": str(rel.get("evidence") or "")[:500],
                "method": "ollama",
                "properties": properties,
            }
        )
    return found


def _relation_properties(rel: dict) -> dict:
    raw = rel.get("properties") if isinstance(rel.get("properties"), dict) else rel
    properties = {}
    for key in _NEWS_PROPERTIES:
        value = raw.get(key)
        if value not in (None, ""):
            properties[key] = value
    return properties


def _relation_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").upper()


def _endpoint(token, entities: dict, title: str, ontology: Ontology, hinted_type: str = "", *, strict: bool = False) -> tuple[str, str] | None:
    if token is None:
        return None
    if isinstance(token, dict):
        name = str(token.get("name") or token.get("id") or "").strip()
        entity_type = _entity_type(str(token.get("type") or hinted_type), ontology, strict=strict)
        if name in {"document", "doc"}:
            return title, "Document"
        if name and entity_type:
            return name, entity_type
        return None
    key = str(token).strip()
    if key in {"document", "doc"}:
        return title, "Document"
    if key in entities:
        return entities[key]
    named = entities.get(normalize_name(key))
    if named:
        return named
    entity_type = _entity_type(hinted_type, ontology, strict=strict)
    if key and entity_type:
        return key, entity_type
    return None


def _entity_type(value: str, ontology: Ontology, *, strict: bool = False) -> str:
    cleaned = value.strip()
    if not cleaned:
        return ""
    # Models often emit ALLCAPS types (CONCEPT, ROLE); normalize before alias lookup.
    lookup = cleaned.casefold()
    if cleaned.isupper() and "_" not in cleaned:
        cleaned = cleaned.title()
    mapped = _TYPE_ALIASES.get(lookup, cleaned[:1].upper() + cleaned[1:])
    if mapped in ontology.entity_types:
        return mapped
    if not strict and mapped and mapped[0].isupper() and mapped.replace("_", "").isalnum():
        # Prefer a known ontology type when the model invents a close synonym.
        for candidate in ontology.entity_types:
            if candidate.casefold() == mapped.casefold():
                return candidate
        return mapped
    return ""


def _canonical_type(value: str, ontology: Ontology) -> str:
    cleaned = value.strip()
    mapped = _TYPE_ALIASES.get(cleaned.casefold(), cleaned[:1].upper() + cleaned[1:] if cleaned else "")
    if mapped in ontology.entity_types:
        return mapped
    return ""


def _pair_allowed(ontology: Ontology, rel_type: str, start_type: str, end_type: str) -> bool:
    return [start_type, end_type] in ontology.relationship_allowlist.get(rel_type, [])


def _doc_rel(rel_type: str, title: str, end_name: str, end_type: str, evidence: str, method: str) -> dict:
    return {
        "type": rel_type,
        "start_name": title,
        "start_type": "Document",
        "end_name": end_name,
        "end_type": end_type,
        "evidence": evidence,
        "method": method,
    }


def _first(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def _split_parties(value: str) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _party(value: str) -> str:
    if not value:
        return ""
    match = _ADDRESS.match(value.strip())
    if match and match.group(1).strip():
        return _clean_name(match.group(1))
    if "@" in value and "<" not in value:
        return ""
    return _clean_name(value)


def _clean_name(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" \t\"'`.,;")


def _rel_id(key: tuple[str, str, str]) -> str:
    digest = hashlib.sha256("|".join(key).encode("utf-8")).hexdigest()[:16]
    return f"rel_{digest}"


def propose_scraping_chunk(
    ctx,
    *,
    title: str,
    chunk: dict,
    file_id: str,
    ontology: Ontology,
    ontology_id: str,
    graph_id: str,
    use_model: bool = True,
) -> tuple[list[dict], list[str]]:
    """Propose relationships for one paragraph. Known types are hints; new types are allowed."""

    text = str(chunk.get("text") or "")
    chunk_id = str(chunk.get("chunk_id") or "")
    chunk_index = int(chunk.get("index") or 0)
    warnings: list[str] = []
    if not text.strip():
        return [], warnings
    relationship_names = sorted(ontology.relationship_allowlist)
    known_types = set(relationship_names)
    proposals = _deterministic("text", title, text, None, ontology)
    if "EXPORTS_TO" in ontology.relationship_allowlist:
        proposals.extend(extract_trade_facts(title, text))
    model_attempted = False
    model_error = ""
    if use_model:
        model_attempted = True
        try:
            proposals.extend(
                _from_ollama_scraping(ctx, title, text, ontology, relationship_names)
            )
            if not proposals:
                # One simpler retry — models sometimes return empty JSON on the first pass.
                proposals.extend(
                    _from_ollama_scraping(
                        ctx,
                        title,
                        text,
                        ontology,
                        relationship_names,
                        simple=True,
                    )
                )
        except ModelUnavailableError as exc:
            model_error = exc.message
            warnings.append(f"chunk {chunk_index}: Ollama unavailable — {exc.message}")
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            model_error = str(exc)
            warnings.append(f"chunk {chunk_index}: Ollama response unusable — {exc}")
    # MENTIONS fallback only when the model was off or failed hard — never as a silent
    # replacement for a successful empty LLM call (that hid the real problem).
    if not proposals and (not model_attempted or model_error):
        proposals.extend(_scraping_mention_fallback(title, text, ontology))
        if proposals:
            reason = "model assists off" if not model_attempted else f"Ollama failed ({model_error})"
            warnings.append(
                f"chunk {chunk_index}: {reason}; added Document MENTIONS fallback for review."
            )
    elif not proposals and model_attempted:
        warnings.append(
            f"chunk {chunk_index}: Ollama returned no relationships for this paragraph."
        )
    merged: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for item in proposals:
        # Map invented entity types toward ontology aliases before gap checks.
        item["start_type"] = _entity_type(str(item.get("start_type") or ""), ontology, strict=False) or item.get("start_type") or "Topic"
        item["end_type"] = _entity_type(str(item.get("end_type") or ""), ontology, strict=False) or item.get("end_type") or "Topic"
        normalize_news_relationship(item)
        hard = reject_news_relationship(item, ontology)
        # Soft path for scraping: keep almost everything for admin review.
        # Only drop placeholders / empty endpoints. Ontology gaps become novelty=new.
        if hard and "placeholder" in hard:
            warnings.append(hard)
            continue
        if hard:
            warnings.append(hard)
        key = (item["type"], normalize_name(item["start_name"]), normalize_name(item["end_name"]))
        if not key[1] or not key[2] or key[1] == key[2] or key in seen:
            continue
        seen.add(key)
        allowed = ontology.relationship_allowlist.get(item["type"], [])
        pair_ok = [item["start_type"], item["end_type"]] in allowed
        novelty = "existing" if item["type"] in known_types and pair_ok else "new"
        item["novelty"] = novelty
        item["needs_ontology_approval"] = novelty == "new" or not pair_ok
        item["chunk_id"] = chunk_id
        item["chunk_index"] = chunk_index
        item["file_id"] = file_id
        item["graph_id"] = graph_id
        item["ontology_id"] = ontology_id
        item["status"] = "new"
        item["id"] = _rel_id(key) + f"_{chunk_index}"
        item["properties"] = {
            **dict(item.get("properties") or {}),
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "file_id": file_id,
            "novelty": novelty,
        }
        if not item.get("evidence"):
            item["evidence"] = text[:500]
        merged.append(item)
    return merged, warnings


def _from_ollama_scraping(
    ctx,
    title: str,
    text: str,
    ontology: Ontology,
    relationship_names: list[str],
    *,
    simple: bool = False,
) -> list[dict]:
    pairs = []
    for rel_type in relationship_names[:40]:
        allowed = ontology.relationship_allowlist.get(rel_type, [])
        rendered = ", ".join(f"{pair[0]} -> {pair[1]}" for pair in allowed[:4])
        pairs.append(f"{rel_type}: {rendered}" if rendered else rel_type)
    listed = "\n".join(pairs) if pairs else "(none yet)"
    entity_listed = ", ".join(ontology.entity_types) if ontology.entity_types else "(any)"
    if simple:
        system = (
            "Return JSON only: {\"entities\":[...],\"relationships\":[...]}. "
            "Each relationship: type, start, start_type, end, end_type, evidence. "
            "Extract every clear fact from the paragraph as a relationship. "
            "Prefer allowlisted types when they fit; otherwise invent UPPER_SNAKE types. "
            f"Entity types to prefer: {entity_listed}. "
            "Named people, brands, places, products, events, and media must become edges. "
            "Do not return an empty relationships array when names appear in the text."
        )
    else:
        system = (
            "Extract knowledge-graph relationships from one paragraph of scraped garment/buyer/fashion text. "
            "Return JSON with keys entities and relationships. "
            "Each relationship has type, start, start_type, end, end_type, evidence, properties. "
            "Prefer allowlisted relationship triples below when they fit the paragraph. "
            "Also extract NEW relationships that are evidenced in the text but not yet in the allowlist: "
            "use a clear UPPER_SNAKE type and entity types from the ontology list when possible. "
            "Admin will review every proposed edge and approve new types into the ontology for future runs. "
            "Prefer these entity types: "
            f"{entity_listed}. "
            "If the paragraph names brands, companies, countries, products, materials, people, topics, or trends, "
            "you MUST return at least one relationship. Prefer specific types (LAUNCHES, HOLDS_ROLE, "
            "OCCURRED_IN, LOCATED_IN, PARTNERS_WITH, MENTIONS, …) over inventing vague ones. "
            "Do not invent facts absent from the paragraph. "
            "Evidence must quote or paraphrase the paragraph and name both endpoints.\n"
            "Allowed triples (hints):\n"
            f"{listed}"
        )
    prompt = f"Document title: {title}\n\nParagraph:\n{text[:4000]}"
    payload = parse_model_json(ctx.ollama.complete(prompt, system=system, json_mode=True))
    entities = {}
    for entity in payload.get("entities") or []:
        if not isinstance(entity, dict):
            continue
        name = str(entity.get("name") or "").strip()
        entity_type = _entity_type(str(entity.get("type") or ""), ontology, strict=False)
        if name and entity_type:
            entities[str(entity.get("id") or name)] = (name, entity_type)
            entities[normalize_name(name)] = (name, entity_type)
    found = []
    for rel in payload.get("relationships") or []:
        if not isinstance(rel, dict):
            continue
        rel_type = _relation_name(str(rel.get("type") or ""))
        start = _endpoint(
            rel.get("start") or rel.get("source") or "document",
            entities,
            title,
            ontology,
            str(rel.get("start_type") or rel.get("source_type") or ""),
            strict=False,
        )
        end = _endpoint(
            rel.get("end") or rel.get("target"),
            entities,
            title,
            ontology,
            str(rel.get("end_type") or rel.get("target_type") or ""),
            strict=False,
        )
        if not rel_type or start is None or end is None:
            continue
        found.append(
            {
                "type": rel_type,
                "start_name": start[0],
                "start_type": start[1],
                "end_name": end[0],
                "end_type": end[1],
                "evidence": str(rel.get("evidence") or text)[:500],
                "method": "ollama",
                "properties": _relation_properties(rel),
            }
        )
    return found


def _scraping_mention_fallback(title: str, text: str, ontology: Ontology) -> list[dict]:
    """When the model returns nothing, still surface Document MENTIONS for admin review."""

    from knowledge_graph.construction.news_facts import _GEOS, _lookup_geo

    found: list[dict] = []
    seen: set[str] = set()
    mentions_pairs = ontology.relationship_allowlist.get("MENTIONS", [])

    def pick_end_type(name: str) -> str:
        geo = _lookup_geo(name) or _GEOS.get(name.casefold())
        if geo and geo[1] in ontology.entity_types:
            return geo[1]
        preferred = ("Topic", "Trend", "Brand", "Company", "Organization", "Product", "Material", "Person", "Country")
        for candidate in preferred:
            if candidate in ontology.entity_types and (
                not mentions_pairs or ["Document", candidate] in mentions_pairs
            ):
                return candidate
        return next(iter(ontology.entity_types), "Topic")

    for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", text):
        name = _clean_name(match.group(1))
        key = normalize_name(name)
        if len(name) < 3 or key in seen or key == normalize_name(title):
            continue
        if name.casefold() in {"the", "and", "for", "with", "from", "this", "that", "january", "february"}:
            continue
        seen.add(key)
        end_name = name
        geo = _lookup_geo(name) or _GEOS.get(name.casefold())
        if geo:
            end_name = geo[0]
        found.append(
            _doc_rel("MENTIONS", title, end_name, pick_end_type(name), match.group(0), "deterministic")
        )
        if len(found) >= 8:
            break
    return found
