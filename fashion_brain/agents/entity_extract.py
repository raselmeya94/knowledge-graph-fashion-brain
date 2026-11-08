"""LLM step: extract structured entities from the question (no Cypher).

Filter keys and scope tokens come from RuntimeSchemaContext (registry/ontology),
not from hard-coded ecom brand or commerce word lists.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from fashion_brain.llm import build_model_provider

LOGGER = logging.getLogger("fashion_brain.agents.entity_extract")

# Minimal English glue words — never ontology-specific.
_GENERIC_SCOPE = frozenset(
    {
        "how",
        "many",
        "the",
        "a",
        "an",
        "and",
        "or",
        "in",
        "of",
        "to",
        "for",
        "from",
        "with",
        "our",
        "is",
        "are",
        "found",
        "product",
        "products",
        "total",
        "number",
        "count",
        "show",
        "list",
        "all",
        "any",
        "there",
        "have",
        "has",
        "do",
        "we",
        "what",
        "which",
        # Shared-material / interconnect phrasing — never invent as category/color.
        "share",
        "shares",
        "shared",
        "sharing",
        "same",
        "across",
        "common",
        "interconnect",
        "material",
        "materials",
        "fabric",
        "fabrics",
        "brand",
        "brands",
        "color",
        "colors",
        "colour",
        "colours",
        "category",
        "categories",
    }
)


@dataclass
class ExtractedEntities:
    brand: str | None = None
    category: str | None = None
    material: str | None = None
    color: str | None = None
    availability: str | None = None
    price_min: float | None = None
    price_max: float | None = None
    currency: str | None = None
    limit: int | None = None
    group_by: str | None = None
    count_entity: str | None = None
    intent: str = "other"
    raw: dict[str, Any] = field(default_factory=dict)

    def as_link_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in ("brand", "category", "material", "color", "availability"):
            value = getattr(self, key)
            if value:
                out[key] = value
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "brand": self.brand,
            "category": self.category,
            "material": self.material,
            "color": self.color,
            "availability": self.availability,
            "price_min": self.price_min,
            "price_max": self.price_max,
            "currency": self.currency,
            "limit": self.limit,
            "group_by": self.group_by,
            "count_entity": self.count_entity,
            "intent": self.intent,
        }


def build_extract_system(schema) -> str:
    filter_keys = list(getattr(schema, "filter_keys", None) or []) or [
        "brand",
        "category",
        "material",
        "color",
        "availability",
    ]
    facets = list(getattr(schema, "facet_types", None) or [])
    hints = []
    for item in getattr(schema, "canonical_hints", None) or []:
        name = item.get("name")
        if name:
            hints.append(str(name))
    hint_line = ", ".join(hints[:12]) if hints else "(from ontology canonical_entities when present)"
    scope = ", ".join(list(getattr(schema, "scope_tokens", None) or [])[:20])
    return f"""\
You extract catalog entities from a user question for a product knowledge graph.
Return ONE JSON object only (no markdown, no prose):
{{
  "brand": null or string,
  "category": null or string,
  "material": null or string,
  "color": null or string,
  "availability": null or string,
  "price_min": null or number,
  "price_max": null or number,
  "currency": null or string,
  "limit": null or integer,
  "group_by": null or string,
  "count_entity": null or one of {facets + ["product"]},
  "intent": "count" | "list" | "breakdown" | "parent" | "other"
}}

Rules:
- Allowed filter fields (use only when the user names a value): {filter_keys}.
- Canonical hint values (examples, not an exclusive list): {hint_line}.
- Graph/catalog scope tokens (NOT filters): {scope or "graph name/description tokens"}.
- Whole-catalog count ("how many products", "how many in <graph>") → filters null, intent count, count_entity product.
- "how many <FacetType>s" (not "per facet") → count_entity = that facet type name, filters null.
- Never write Cypher. Never invent node ids.
"""


def is_catalog_scope_term(term: str | None, scope_tokens: list[str] | None = None) -> bool:
    """True when the term names the catalog/graph itself, not a facet value."""

    if not term:
        return False
    from knowledge_graph.ontology import normalize_name

    nn = normalize_name(term)
    compact = nn.replace(" ", "")
    scope = set(scope_tokens or []) | _GENERIC_SCOPE
    if nn in scope or compact in scope:
        return True
    tokens = set(nn.split())
    if tokens and tokens <= scope:
        return True
    return False


def sanitize_extracted(
    entities: ExtractedEntities,
    *,
    scope_tokens: list[str] | None = None,
    facet_types: list[str] | None = None,
    question: str | None = None,
) -> ExtractedEntities:
    for key in ("brand", "category", "material", "color", "availability"):
        value = getattr(entities, key)
        if is_catalog_scope_term(value, scope_tokens):
            setattr(entities, key, None)
        elif value and str(value).casefold() in _GENERIC_SCOPE:
            setattr(entities, key, None)
    # "which products share a material" must not invent White/share as category.
    if question and _looks_like_shared_material(question):
        for key in ("brand", "category", "material", "color", "availability"):
            setattr(entities, key, None)
        entities.count_entity = None
    if entities.count_entity:
        ce = entities.count_entity.casefold()
        # Normalize plurals / aliases against live facet types.
        facets = {t.casefold(): t for t in (facet_types or [])}
        if ce in {"brands", "buyer", "buyers"}:
            ce = "brand"
        elif ce in {"colors", "colours", "colour"}:
            ce = "color"
        elif ce in {"materials", "fabric", "fabrics"}:
            ce = "material"
        elif ce in {"categories", "garment", "garments"}:
            ce = "category"
        elif ce in {"products"}:
            ce = "product"
        if ce == "product":
            entities.count_entity = "product"
        elif ce in facets:
            entities.count_entity = facets[ce]
        elif ce.rstrip("s") in facets:
            entities.count_entity = facets[ce.rstrip("s")]
        else:
            # Keep lower key if unknown — Text2Cypher may still map it.
            entities.count_entity = ce
        if entities.count_entity and entities.count_entity.casefold() != "product":
            # Clear filter that duplicates the counted facet word.
            for key in ("brand", "category", "material", "color"):
                val = getattr(entities, key)
                if val and val.casefold() in {
                    entities.count_entity.casefold(),
                    entities.count_entity.casefold() + "s",
                    key,
                    key + "s",
                }:
                    setattr(entities, key, None)
    return entities


class EntityExtractAgent:
    """LLM extracts entities; EntityLinker resolves them to graph ids."""

    def extract(
        self,
        platform,
        question: str,
        *,
        brands: list[dict[str, str]] | None = None,
        schema=None,
    ) -> ExtractedEntities:
        if schema is None:
            try:
                from fashion_brain.planning.runtime_schema import get_runtime_schema

                schema = get_runtime_schema(platform)
            except Exception:
                schema = None
        scope = list(getattr(schema, "scope_tokens", None) or [])
        facets = list(getattr(schema, "facet_types", None) or [])

        if platform.settings.answer_mode != "llm":
            return sanitize_extracted(
                _heuristic_extract(question, schema=schema),
                scope_tokens=scope,
                facet_types=facets,
                question=question,
            )
        provider = build_model_provider(platform.settings)
        brand_lines = ""
        if brands:
            brand_lines = "Live brand nodes:\n" + "\n".join(
                f"- {b.get('name')} (id={b.get('id')})" for b in brands[:30]
            )
        system = build_extract_system(schema) if schema is not None else build_extract_system(_EmptySchema())
        prompt = f"{brand_lines}\n\nQuestion:\n{question}\n\nExtract the JSON now."
        try:
            from fashion_brain.stage_timing import timed

            with timed("entity_extract_llm"):
                raw = provider.complete(
                    prompt.strip(),
                    timeout=_timeout(platform.settings.request_timeout_seconds),
                    system=system,
                )
        except Exception as exc:
            LOGGER.warning("entity extract LLM failed: %s", exc)
            return sanitize_extracted(
                _heuristic_extract(question, schema=schema),
                scope_tokens=scope,
                facet_types=facets,
                question=question,
            )
        parsed = _parse(raw)
        if parsed is None:
            return sanitize_extracted(
                _heuristic_extract(question, schema=schema),
                scope_tokens=scope,
                facet_types=facets,
                question=question,
            )
        parsed = sanitize_extracted(
            parsed, scope_tokens=scope, facet_types=facets, question=question
        )
        if not any(
            [
                parsed.brand,
                parsed.category,
                parsed.material,
                parsed.color,
                parsed.availability,
                parsed.price_min,
                parsed.price_max,
                parsed.count_entity,
            ]
        ):
            if _looks_like_shared_material(question):
                parsed.intent = "other"
                return parsed
            if _looks_like_catalog_count(question, scope):
                parsed.intent = "count"
                parsed.count_entity = "product"
                return parsed
            return sanitize_extracted(
                _heuristic_extract(question, schema=schema),
                scope_tokens=scope,
                facet_types=facets,
                question=question,
            )
        return parsed


class _EmptySchema:
    filter_keys: list[str] = []
    facet_types: list[str] = []
    canonical_hints: list = []
    scope_tokens: list[str] = []


def _timeout(seconds: float | None) -> float:
    if seconds is None or seconds <= 0:
        return 8.0
    return min(8.0, float(seconds))


def _parse(raw: str) -> ExtractedEntities | None:
    from fashion_brain.ask import _extract_json_object

    data = _extract_json_object(raw or "")
    if not isinstance(data, dict):
        return None

    def _str(key: str) -> str | None:
        value = data.get(key)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _num(key: str) -> float | None:
        value = data.get(key)
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _int(key: str) -> int | None:
        value = data.get(key)
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    intent = str(data.get("intent") or "other").strip().lower()
    if intent not in {"count", "list", "breakdown", "parent", "other"}:
        intent = "other"
    return ExtractedEntities(
        brand=_str("brand"),
        category=_str("category"),
        material=_str("material"),
        color=_str("color"),
        availability=_str("availability"),
        price_min=_num("price_min"),
        price_max=_num("price_max"),
        currency=_str("currency"),
        limit=_int("limit"),
        group_by=_str("group_by"),
        count_entity=_str("count_entity"),
        intent=intent,
        raw=data,
    )


def _looks_like_catalog_count(question: str, scope_tokens: list[str] | None = None) -> bool:
    text = (question or "").casefold()
    if not re.search(r"\bhow\s+many\b|\bcount\b|\btotal\b|\bnumber\s+of\b", text):
        return False
    # If a live canonical/hint brand token appears, treat as filtered ask.
    return True


def _looks_like_shared_material(question: str) -> bool:
    text = (question or "").casefold()
    if not re.search(r"\b(share|shared|sharing|same)\b", text):
        return False
    return bool(re.search(r"\b(material|materials|fabric|fabrics)\b", text))


def _facet_filter_key_local(label: str) -> str:
    key = str(label or "").casefold()
    if "category" in key:
        return "category"
    return key


def facet_entity_count_from_schema(question: str, schema) -> str | None:
    """Dynamic 'how many brands/colors/…' against ontology facet_types."""

    from knowledge_graph.ontology import normalize_name

    text = normalize_name(question)
    if not re.search(r"\b(how many|number of|count of|total)\b", text):
        return None
    if re.search(r"\b(per|by|wise|grouped|breakdown|each)\b", text):
        return None
    tokens = set(text.split())
    for label in getattr(schema, "facet_types", None) or []:
        key = label.casefold()
        variants = {key, key + "s", key.rstrip("s")}
        # GarmentCategory → also match category/categories
        if "category" in key:
            variants |= {"category", "categories", "garment", "garments"}
        if "colour" in key or key == "color":
            variants |= {"color", "colors", "colour", "colours"}
        if tokens & variants:
            return label
    if tokens & {"product", "products"}:
        return "product"
    return None


def _heuristic_extract(question: str, *, schema=None) -> ExtractedEntities:
    from knowledge_graph.ontology import normalize_name

    text = question or ""
    nn = normalize_name(text)
    tokens = set(nn.split()) if nn else set()
    scope = list(getattr(schema, "scope_tokens", None) or [])

    brand = None
    for item in getattr(schema, "canonical_hints", None) or []:
        if str(item.get("type") or "").casefold() != "brand":
            continue
        names = [str(item.get("name") or "")] + [str(a) for a in (item.get("aliases") or [])]
        for name in names:
            n = normalize_name(name)
            if n and (n in nn or n.replace(" ", "") in nn.replace(" ", "")):
                brand = str(item.get("name") or name)
                break
        if brand:
            break

    count_entity = facet_entity_count_from_schema(text, schema) if schema is not None else None
    category = material = color = None
    availability = None
    if re.search(r"\binstock\b|\bin[\s_-]?stock\b", text, re.I):
        availability = "InStock"

    # Only invent a free-text category when not a catalog/facet/shared-material ask.
    if (
        not count_entity
        and not _looks_like_catalog_count(text, scope)
        and not _looks_like_shared_material(text)
        and nn
    ):
        stop = set(_GENERIC_SCOPE) | set(scope)
        for label in getattr(schema, "facet_types", None) or []:
            stop.add(label.casefold())
            stop.add(_facet_filter_key_local(label))
        for token in nn.split():
            if len(token) >= 5 and token not in stop:
                category = token
                break

    price_max = None
    currency = None
    m = re.search(r"under\s+(\d+(?:\.\d+)?)\s*(eur|euro|usd|gbp|£|€|\$)?", text, re.I)
    if m:
        price_max = float(m.group(1))
        cur = (m.group(2) or "").lower()
        if cur in {"eur", "euro", "€"}:
            currency = "EUR"
        elif cur in {"usd", "$"}:
            currency = "USD"
        elif cur in {"gbp", "£"}:
            currency = "GBP"
    limit = None
    m2 = re.search(r"\b(?:show|list|top)\s+(\d+)\b", text, re.I)
    if m2:
        limit = int(m2.group(1))
    if count_entity == "product" or (not count_entity and _looks_like_catalog_count(text, scope)):
        count_entity = count_entity or "product"
        category = None

    return ExtractedEntities(
        brand=brand,
        category=category,
        material=material,
        color=color,
        availability=availability,
        price_min=None,
        price_max=price_max,
        currency=currency,
        limit=limit,
        group_by=None,
        count_entity=count_entity,
        intent="list" if limit else ("count" if ("how many" in text.casefold() or count_entity) else "other"),
    )
