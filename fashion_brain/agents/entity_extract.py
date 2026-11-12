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

# Minimal English glue + ranking/metric operators — never ontology-specific catalog VALUES.
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
        "largest",
        "biggest",
        "highest",
        "lowest",
        "most",
        "least",
        "average",
        "avg",
        "mean",
        "median",
        "overall",
        "typical",
        "decrease",
        "decreases",
        "decreased",
        "increase",
        "increases",
        "increased",
        "drop",
        "drops",
        "rise",
        "rises",
        "change",
        "changes",
        "trend",
        "trends",
        "history",
        "historical",
        "fluctuation",
        "fluctuations",
        "cheaper",
        "expensive",
        "had",
        "which",
        "what",
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
    country: str | None = None
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
        for key in ("brand", "category", "material", "color", "availability", "country"):
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
            "country": self.country,
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
  "country": null or string,
  "price_min": null or number,
  "price_max": null or number,
  "currency": null or string,
  "limit": null or integer,
  "group_by": null or string,
  "count_entity": null or one of {facets + ["product"]},
  "intent": "count" | "list" | "breakdown" | "parent" | "other"
}}

Rules:
- Allowed filter fields (use only when the user names a concrete catalog VALUE like primark, cotton, dresses): {filter_keys}.
- Canonical hint values (examples, not an exclusive list): {hint_line}.
- Graph/catalog scope tokens (NOT filters): {scope or "graph name/description tokens"}.
- Country of origin ("made in Bangladesh", "manufactured in …", "country of origin …") → country=<place>.
  NEVER put a country/place into category/garment/brand.
- Follow-ups that keep prior constraints (e.g. "REQUIRED: MADE_IN country=…") must copy country (and other REQUIRED filters) into the JSON.
- "check for all brands" / "by brand" with a prior country → group_by="brand", keep country set, brand null.
- Whole-catalog count ("how many products", "how many in <graph>") → filters null, intent count, count_entity product.
- "how many <FacetType>s" (not "per facet") → count_entity = that facet type name, filters null.
- Analysis / positioning questions ("how are brands positioned on price", "price by family", "compare brands within family"):
  set intent=breakdown (or other), group_by=family (or brand), leave brand/category/material/color NULL.
  NEVER put verbs like positioned/compare/pricing into category or brand.
- Brand/buyer analysis or profile asks ("analyze ZARA", "comprehensive analysis of … brand"):
  set brand (or buyer) only; category/material/color/garment MUST be null.
  Never copy task phrasing or intensifiers from the question into category/garment — those are not product types.
- Retailer / label names that appear in the Live brand nodes list are ALWAYS brand (or buyer), NEVER category.
  Examples of shape (not an exclusive list): "how many products does <Brand> have", "products for <Brand>",
  "<Brand> t-shirts" → brand=<Brand>, category only if a product type is also named.
- Product types (t-shirts, dresses, sweaters, …) go in category — never confuse them with brand names.
- "in each family / per family / by family" → group_by="family" (not a category filter named family).
- Price words without a numeric bound are NOT price_min/price_max filters.
- Never write Cypher. Never invent node ids.
"""


def normalize_or_empty(term: str | None) -> str:
    from knowledge_graph.ontology import normalize_name

    return normalize_name(term or "") if term else ""


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


def reserved_dimension_tokens(schema=None) -> set[str]:
    """Dynamic reserved words from live ontology/schema — never filter VALUES.

    Built from filter_keys, facet_types, entity type labels, and product property names.
    No hand-maintained analysis-verb lists.
    """

    from knowledge_graph.ontology import normalize_name

    out: set[str] = set(_GENERIC_SCOPE)
    if schema is None:
        return out
    for key in getattr(schema, "filter_keys", None) or []:
        nn = normalize_name(str(key))
        if nn:
            out.add(nn)
            out.add(nn + "s")
            out.update(nn.split())
    for label in getattr(schema, "facet_types", None) or []:
        nn = normalize_name(str(label))
        if not nn:
            continue
        out.add(nn)
        out.add(nn + "s")
        # GarmentCategory → also reserve "garment", "category"
        for part in re.findall(r"[A-Za-z]+", str(label)):
            p = part.casefold()
            if len(p) >= 3:
                out.add(p)
                out.add(p + "s")
    for label in getattr(schema, "entity_types", None) or []:
        nn = normalize_name(str(label))
        if nn:
            out.add(nn)
            out.update(nn.split())
    for prop in getattr(schema, "product_properties", None) or []:
        nn = normalize_name(str(prop))
        if not nn:
            continue
        out.add(nn)
        # price_num → price
        out.update(p for p in nn.split("_") if len(p) >= 3)
    for tok in getattr(schema, "scope_tokens", None) or []:
        nn = normalize_name(str(tok))
        if nn:
            out.add(nn)
    return out


def is_reserved_term(term: str | None, schema=None) -> bool:
    """True when term names a dimension/property/scope word, not a catalog value."""

    if not term:
        return False
    from knowledge_graph.ontology import normalize_name

    nn = normalize_name(term)
    if not nn:
        return False
    reserved = reserved_dimension_tokens(schema)
    if nn in reserved:
        return True
    tokens = set(nn.split())
    return bool(tokens) and tokens <= reserved


def matches_canonical_hint(term: str | None, schema=None, *, types: set[str] | None = None) -> bool:
    """True when term equals/aliases a live canonical hint (schema-grounded value)."""

    if not term or schema is None:
        return False
    from knowledge_graph.ontology import normalize_name

    nn = normalize_name(term)
    if not nn:
        return False
    want = {t.casefold() for t in (types or set())}
    for item in getattr(schema, "canonical_hints", None) or []:
        kind = str(item.get("type") or "").casefold()
        if want and kind not in want:
            continue
        names = [str(item.get("name") or "")] + [str(a) for a in (item.get("aliases") or [])]
        for name in names:
            n = normalize_name(name)
            if n and (n == nn or n.replace(" ", "") == nn.replace(" ", "")):
                return True
    return False


# Back-compat alias used by linker / pipeline scrubbers.
def is_analysis_stopword(term: str | None, schema=None) -> bool:
    return is_reserved_term(term, schema)


def detect_group_by_from_schema(question: str | None, schema=None) -> str | None:
    """Pick group_by from 'each/by/per <dimension>' using schema filter/facet keys."""

    if not question:
        return None
    from knowledge_graph.ontology import normalize_name

    text = normalize_name(question)
    tokens = text.split()
    dims: list[tuple[str, set[str]]] = []
    for key in getattr(schema, "filter_keys", None) or []:
        k = str(key).casefold()
        dims.append((k, {k, k + "s", k.replace("_", " ")}))
    for label in getattr(schema, "facet_types", None) or []:
        raw = str(label)
        k = raw.casefold()
        variants = {k, k + "s", normalize_name(raw)}
        if "family" in k:
            variants |= {"family", "families"}
            key_out = "family" if k.endswith("family") and not k.startswith("sub") else k
            if "subfamily" in k or k.endswith("sub family"):
                key_out = "subfamily"
                variants |= {"subfamily", "subfamilies"}
            dims.append((key_out, variants))
        elif "categor" in k:
            dims.append(("category", variants | {"category", "categories"}))
        elif "brand" in k:
            dims.append(("brand", variants | {"brand", "brands"}))
        elif "color" in k or "colour" in k:
            dims.append(("color", variants | {"color", "colors", "colour", "colours"}))
        elif "material" in k:
            dims.append(("material", variants | {"material", "materials"}))
        else:
            dims.append((normalize_name(raw).replace(" ", "_") or k, variants))
    # Ontology Family/Subfamily entity types even if not in filter_keys yet.
    for label in getattr(schema, "entity_types", None) or []:
        k = str(label).casefold()
        if k == "family":
            dims.append(("family", {"family", "families"}))
        elif k == "subfamily":
            dims.append(("subfamily", {"subfamily", "subfamilies"}))
    for i, tok in enumerate(tokens):
        # "by|per|each|all|every <schema_dimension>" — dimension set is live schema only.
        if tok in {"by", "per", "each", "all", "every", "within", "across", "among"} and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            for key_out, variants in dims:
                if nxt in variants:
                    return key_out
        if tok in {"wise"} and i > 0:
            prev = tokens[i - 1]
            for key_out, variants in dims:
                if prev in variants:
                    return key_out
    return None


def looks_like_metric_breakdown(question: str | None, schema=None) -> bool:
    """Aggregation over a schema dimension + product metric — not a named-entity filter ask."""

    if not question:
        return False
    from knowledge_graph.ontology import normalize_name

    text = normalize_name(question)
    tokens = set(text.split())
    # Metrics: from live product_properties (price_num → price) plus tiny universal agg words.
    metric_tokens = {"avg", "average", "mean", "median", "min", "max", "sum", "total"}
    for prop in getattr(schema, "product_properties", None) or []:
        nn = normalize_name(str(prop))
        metric_tokens.update(p for p in nn.replace("-", "_").split("_") if len(p) >= 3)
    if not (tokens & metric_tokens):
        # Still allow "positioned on <metric>" when group_by dimension is present.
        if not detect_group_by_from_schema(question, schema):
            return False
    # Need a breakdown dimension (schema facet / each|by|per).
    if detect_group_by_from_schema(question, schema):
        return True
    facet_words: set[str] = set()
    for label in getattr(schema, "facet_types", None) or []:
        facet_words.update(re.findall(r"[a-z]+", str(label).casefold()))
        facet_words.add(str(label).casefold())
    for key in getattr(schema, "filter_keys", None) or []:
        facet_words.add(str(key).casefold())
        facet_words.add(str(key).casefold() + "s")
    return bool(tokens & facet_words) and bool(tokens & metric_tokens)


def looks_like_price_family_analysis(question: str | None, schema=None) -> bool:
    """Back-compat wrapper → schema-grounded metric breakdown."""

    return looks_like_metric_breakdown(question, schema)


def sanitize_extracted(
    entities: ExtractedEntities,
    *,
    scope_tokens: list[str] | None = None,
    facet_types: list[str] | None = None,
    question: str | None = None,
    schema=None,
) -> ExtractedEntities:
    reserved = reserved_dimension_tokens(schema)
    for key in ("brand", "category", "material", "color", "availability", "country"):
        value = getattr(entities, key)
        if is_catalog_scope_term(value, scope_tokens):
            setattr(entities, key, None)
        elif value and normalize_or_empty(value) in reserved:
            setattr(entities, key, None)
        elif is_reserved_term(value, schema):
            setattr(entities, key, None)
    # If a schema facet value was also stuffed into category, prefer the facet field.
    if schema is not None:
        from fashion_brain.planning.active_constraints import schema_dimension_keys

        dims = schema_dimension_keys(schema)
        for key in dims:
            if not hasattr(entities, key):
                continue
            val = getattr(entities, key)
            if not val:
                continue
            cat = entities.category
            if cat and normalize_or_empty(cat) == normalize_or_empty(val) and key != "category":
                entities.category = None
    # "which products share a material" must not invent White/share as category.
    if question and _looks_like_shared_material(question):
        for key in ("brand", "category", "material", "color", "availability"):
            setattr(entities, key, None)
        entities.count_entity = None
    if question and looks_like_metric_breakdown(question, schema):
        for key in ("brand", "category", "material", "color", "availability"):
            setattr(entities, key, None)
        entities.price_min = None
        entities.price_max = None
        entities.count_entity = None
        if not entities.group_by:
            entities.group_by = detect_group_by_from_schema(question, schema) or entities.group_by
        if entities.intent in {"other", "count", "list"}:
            entities.intent = "breakdown"
    if question and looks_like_facet_product_breakdown(question, schema):
        entities.count_entity = None
        if not entities.group_by:
            entities.group_by = detect_group_by_from_schema(question, schema) or entities.group_by
        if entities.intent in {"other", "count", "list"}:
            entities.intent = "breakdown"
        # Do not filter to a brand named "brand".
        for key in ("brand", "category", "material", "color"):
            val = getattr(entities, key)
            if val and normalize_or_empty(val) in reserved:
                setattr(entities, key, None)
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
                schema=schema,
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
                schema=schema,
            )
        parsed = _parse(raw)
        if parsed is None:
            return sanitize_extracted(
                _heuristic_extract(question, schema=schema),
                scope_tokens=scope,
                facet_types=facets,
                question=question,
                schema=schema,
            )
        parsed = sanitize_extracted(
            parsed, scope_tokens=scope, facet_types=facets, question=question, schema=schema
        )
        if not any(
            [
                parsed.brand,
                parsed.category,
                parsed.material,
                parsed.color,
                parsed.availability,
                parsed.country,
                parsed.price_min,
                parsed.price_max,
                parsed.count_entity,
                parsed.group_by,
            ]
        ):
            if looks_like_metric_breakdown(question, schema):
                parsed.intent = "breakdown"
                parsed.group_by = detect_group_by_from_schema(question, schema) or parsed.group_by
                return parsed
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
                schema=schema,
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
        country=_str("country"),
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


def looks_like_facet_product_breakdown(question: str | None, schema=None) -> bool:
    """True when the user wants products counted *per* facet value (list/table), not distinct facets.

    Examples: "list all brands and total products", "products per brand", "brand wise product count".
    """

    if not question:
        return False
    from knowledge_graph.ontology import normalize_name

    text = normalize_name(question)
    has_products = bool(re.search(r"\b(products?|items?|skus?|styles?)\b", text))
    has_list = bool(re.search(r"\b(list|table|show|breakdown|ranking|rank)\b", text))
    has_counts = bool(re.search(r"\b(count|counts|total|totals|how many)\b", text))
    dim = detect_group_by_from_schema(question, schema)
    if dim and has_products and (has_list or has_counts):
        return True
    # "brand wise product count" / "products for each brand" without needing list verbs.
    if dim and has_products and re.search(r"\b(wise|each|per|by|all|every|across|among)\b", text):
        return True
    return False


def facet_entity_count_from_schema(question: str, schema) -> str | None:
    """Dynamic 'how many brands/colors/…' against ontology facets + filter/property keys."""

    from knowledge_graph.ontology import normalize_name

    text = normalize_name(question)
    # Product-per-facet lists are group_by breakdowns — never distinct-facet counts.
    if looks_like_facet_product_breakdown(question, schema):
        return None
    if not re.search(r"\b(how many|number of|count of)\b", text):
        # Bare "total brands" is ok; "total products … brand" is handled above as breakdown.
        if not re.search(r"\btotal\b", text):
            return None
        if re.search(r"\btotal\s+(products?|items?|skus?|styles?)\b", text):
            return None
    if re.search(r"\b(per|by|wise|grouped|breakdown|each)\b", text):
        return None
    tokens = set(text.split())
    labels: list[str] = []
    seen: set[str] = set()
    for raw in (
        list(getattr(schema, "facet_types", None) or [])
        + list(getattr(schema, "filter_keys", None) or [])
        + list(getattr(schema, "product_properties", None) or [])
    ):
        label = str(raw or "").strip()
        key = label.casefold()
        if not label or key in seen or key in {"id", "name", "graph", "lifecycle", "product"}:
            continue
        seen.add(key)
        labels.append(label)
    for label in labels:
        key = label.casefold().replace(" ", "_").replace("-", "_")
        # Variants derived from this key only (plural / compacted) — no fixed fashion nouns.
        variants = {key, key + "s", key.replace("_", ""), key.replace("_", "") + "s"}
        if key.endswith("y") and len(key) > 2:
            variants.add(key[:-1] + "ies")
        if key.endswith("s") and len(key) > 3:
            variants.add(key[:-1])
        # CamelCase facet type GarmentCategory → also expose filter key stem.
        stem = key.split("_")[-1] if "_" in key else key
        if stem and stem != key:
            variants |= {stem, stem + "s"}
            if stem.endswith("y") and len(stem) > 2:
                variants.add(stem[:-1] + "ies")
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
    country = None
    # Schema-gated origin: only if live schema exposes a `country` filter/facet.
    if schema is not None:
        from fashion_brain.planning.active_constraints import schema_dimension_keys

        if "country" in schema_dimension_keys(schema):
            m = re.search(
                r"\b(?:made|manufactured|produced|sourced)\s+in\s+([A-Za-z][A-Za-z]*(?:\s+[A-Za-z]+){0,3})",
                text,
                re.I,
            )
            if m:
                country = m.group(1).strip().casefold()
    if re.search(r"\binstock\b|\bin[\s_-]?stock\b", text, re.I):
        availability = "InStock"

    metric_breakdown = looks_like_metric_breakdown(text, schema)

    # Schema-grounded only: never invent free-text category from grammar words.
    if not count_entity and not metric_breakdown and not _looks_like_shared_material(text) and nn:
        for item in getattr(schema, "canonical_hints", None) or []:
            kind = str(item.get("type") or "").casefold()
            names = [str(item.get("name") or "")] + [str(a) for a in (item.get("aliases") or [])]
            hit = None
            for name in names:
                n = normalize_name(name)
                if n and (n in nn or n.replace(" ", "") in nn.replace(" ", "")):
                    hit = str(item.get("name") or name)
                    break
            if not hit:
                continue
            if kind in {"category", "garmentcategory"} and not category:
                category = hit
            elif kind in {"material"} and not material:
                material = hit
            elif kind in {"color", "colour"} and not color:
                color = hit

    price_max = None
    currency = None
    m = re.search(r"under\s+(\d+(?:\.\d+)?)\s*(eur|euro|usd|gbp|£|€|\$)?", text, re.I)
    if m and not metric_breakdown:
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

    group_by = None
    intent = "list" if limit else ("count" if ("how many" in text.casefold() or count_entity) else "other")
    if metric_breakdown:
        brand = None
        category = None
        material = None
        color = None
        count_entity = None
        price_max = None
        group_by = detect_group_by_from_schema(text, schema)
        intent = "breakdown"

    return ExtractedEntities(
        brand=brand,
        category=category,
        material=material,
        color=color,
        availability=availability,
        country=country,
        price_min=None,
        price_max=price_max,
        currency=currency,
        limit=limit,
        group_by=group_by,
        count_entity=count_entity,
        intent=intent,
    )
