"""Product chatbot. Answers are assembled only from knowledge-graph evidence.

Planning contract
-----------------
The language model decides what the user means; code only validates it.

* ``plan_catalog_question`` is the entry point for planning. It asks the model for a
  structured plan, then grounds every filter value in the user's own words, normalises
  season names and fixes buyer spelling against the catalog.
* ``resolve_catalog_plan`` is the keyword planner. It is the fallback when there is no
  API key, the call times out, or the model returns invalid JSON.
"""

from __future__ import annotations

import json
import re
from typing import Any

from knowledge_graph.access import Principal, allows  # noqa: F401  (re-exported)
from knowledge_graph.adapters.products import SqlProductConnector  # noqa: F401  (re-exported)
from knowledge_graph.errors import ProductBackendUnavailable  # noqa: F401  (re-exported)
from knowledge_graph.models import EvidenceItem, GroundedAnswer
from knowledge_graph.logging_config import get_logger
from knowledge_graph.ontology import normalize_name
from knowledge_graph.pipeline.service import Platform
from knowledge_graph.retrieval.news import news_question, search_news  # noqa: F401  (re-exported)
from knowledge_graph.retrieval.products import CATALOG_FIELDS, field_text
from knowledge_graph.retrieval.queries import product_materials, products_sharing_material  # noqa: F401

from fashion_brain.llm import build_model_provider

LOGGER = get_logger("fashion_brain.ask")

DEMO_CAVEAT = "These figures describe the synthetic demo products, not a live PLMEX database."
CATALOG_CAVEAT = ""


def _words(text: str) -> frozenset[str]:
    return frozenset(text.split())


# --------------------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------------------

_COLOR_WORDS = _words("red blue green black white yellow pink purple grey gray brown orange beige navy burgundy")
_COLOR_MARKERS = _words("color colour colors colours")
_MATERIAL_WORDS = _words("cotton linen elastane polyester nylon wool viscose acrylic silk")

# Words that can never be a filter value (question grammar, field names, polite filler).
_COUNT_STOPWORDS = _words(
    """
    how many product products are is a an the of in our catalog catalogue plm data synthetic demo there do we have
    with and or for from at that this be any all every each number count total visible clearance by gender wise per
    breakdown analysis analyze grouped distribution split brands brand buyer buyers season seasons category
    categories material materials department departments designer designers year now can you your make me
    please could would them those these their same also just show give want let us again still instead about
    what which who when where list has have had was were been contain contains containing made using use uses top bottom
    highest lowest leading rank ranking first last garments type types kind kinds normally usually typical
    typically often frequently frequent frequency common commonly mostly making makes does did produce produces producing manufacture
    manufactures if then upcomming upcoming coming coming seasion sesion suggest suggestion suggestions
    specification specifications spec specs recommend recommendation recommendations sample samples new
    used shared share same interconnect interconnection found matched match result results currently available
    across among amongst only also present available registered registry metadata attribute attributes
    unique values description descriptions knowledge graph graphs kg kgs source sources system basic info
    information overview stored inside within ecommerce ecommerces commerce commerces ecom ecomm
    node nodes event events snapshot snapshots exist exists
    """
)
_RANK_WORDS = _words("top bottom highest lowest leading most largest biggest smallest fewest")
# Comparison / change language — never a catalog facet VALUE (garment/category/…).
_METRIC_OPERATOR_WORDS = _words(
    """
    decrease decreases decreased decreasing increase increases increased increasing
    drop drops dropped dropping rise rises rose rising fall falls fallen falling
    change changes changed changing trend trends trending history historical historically
    fluctuation fluctuations mover movers move moves moved cheaper expensive costly
    percent percentage delta difference diff vs versus compared compare comparing comparison
    over time window period months month weeks week days day ago since until
    average avg mean median overall typical aggregate aggregates summary
    significant multiple large larger largest small smaller greater than less equal
    threshold recent recently ignore ignoring switch switches switched currency
    old new percent pct
    """
)
# Task / relation phrasing — never a brand, buyer, or garment value.
_TASK_WORDS = _words(
    """
    compare compared comparing comparison similar similarly same across against
    between among amongst side beside relative relatively respectively
    historical historically recent recently prior previous past
    move moves moved mover movers
    appear appears appeared appearing often frequently frequent frequency common commonly typical typically
    exist exists existing stored store storage node nodes event events snapshot snapshots
    instruction instructions detail details composition compositions outline outlines
    region regions country countries area areas uk eu us
    decide decides deciding decision study studies studying studies
    """
)
# Measure / property names that are not garment types when used alone in ranking asks.
_MEASURE_PROPERTY_WORDS = _words(
    "price prices cost costs amount amounts rating ratings discount discounts "
    "margin margins stock quantity qty average avg mean median "
    "availability availabilities care construction constructions fit fits "
    "supplier suppliers origin origins season seasons"
)
_MAKE_VERBS = _words("make makes making produce produces producing manufacture manufactures manufacturing")
_MAKE_ADVERBS = _words("normally usually typically often frequently frequent frequency commonly mostly")
# Catalog / graph / system words: never a garment, material, or buyer/brand filter value.
_SYSTEM_WORDS = _words(
    "plm plmex plm-ex system database catalog catalogue ecom ecomm ecommerce commerce "
    "neo4j postgres sql kg graph graphs source sources knowledge "
    "pricechangeevent availabilitysnapshot entity entities"
)
# Legacy soft tokens used only to strip graph-scope noise from party filters.
# Preferred graph ids come from the live registry (see preferred_graphs_from_question).
_GRAPH_SCOPE_NOISE = _words(
    "ecom ecomm ecommerce ecommerces commerce commerces plm plmex own"
)
# Grammar / generic catalog words — never decide which product graph to use.
# Distinctive scope tokens (plm, ecom, ecommerce, …) stay outside this set.
_GRAPH_SCORE_STOPWORDS = _words(
    """
    how many are is a an the of in on our with and or for from at that this be any all
    every each number count total what which who when where list has have had was were
    been contain contains containing do we you your me please could would them those
    these their same also just show give want let us again still about product products
    catalog catalogue data information info overview knowledge graph graphs kg source
    sources system database node nodes available visible there here inside within
    """
)
# A filter value containing any of these is the question pasted back, not a catalog value.
_FILTER_NOISE = _words(
    "normally usually typical typically making make makes made type types kind kinds which what "
    "product products does new sample samples suggest specification specifications "
    "found matched match result results currently available total number count "
    "largest biggest highest lowest most least fewest smallest top "
    "decrease decreases decreased increase increases increased drop drops rise rose "
    "change changes trend trends trending history historical fluctuation fluctuations "
    "analyse analyze analysis attribute wise breakdown takeaway leading "
    "cheaper expensive mover movers"
)
_SUGGEST_WORDS = _words(
    "suggest suggestion suggestions recommend recommendation recommendations"
)
_ANALYSIS_WORDS = _words("analysis analyze analysing analyzing analyse profile review overview study")
_HISTORY_WORDS = _words("past previous prior historic historical history earlier old existing")
_UPCOMING_WORDS = _words("upcoming upcomming coming comming next")
_BY_WORDS = _words("by per each")
_BREAKDOWN_HEADS = _words("breakdown distribution split")
_BREAKDOWN_SUFFIXES = _BREAKDOWN_HEADS | {"wise"}

_SPEC_FIELDS = (
    "garment_type",
    "category",
    "subcategory",
    "plm_main_category",
    "plm_sub_sub_category",
    "material",
    "color",
    "gender",
    "design_for",
    "gauge",
    "gsm",
    "weight",
    "knit_machine_type",
    "knitting_time",
    "design_feature",
    "style_feature",
    "department",
    "shopping_department",
    "unit",
    "cuttable_width",
    "sample_collection_type",
)
# Named seasons map to an autumn/winter style code, not to a value in the season-year column.
_SEASON_WORDS = {
    "winter": "winter",
    "winer": "winter",
    "autumn": "winter",
    "fall": "winter",
    "aw": "winter",
    "spring": "spring",
    "summer": "summer",
}
_SEASON_NAMES = frozenset(_SEASON_WORDS.values())
_TYPE_PATTERNS = (
    ("type", "of", "products"),
    ("type", "of", "product"),
    ("types", "of", "products"),
    ("types", "of", "product"),
    ("type", "of", "garments"),
    ("type", "of", "garment"),
    ("types", "of", "garments"),
    ("kind", "of", "products"),
    ("kind", "of", "product"),
    ("kinds", "of", "products"),
    ("kind", "of", "garments"),
    ("product", "type"),
    ("product", "types"),
)
_UNNAMED_GROUPS = frozenset({"", "unspecified", "null", "none", "n/a", "na", "-"})
_GENDER_VALUES = {
    "women": "women",
    "woman": "women",
    "ladies": "women",
    "female": "women",
    "men": "men",
    "man": "men",
    "male": "men",
    "girl": "girl",
    "girls": "girl",
    "boy": "boy",
    "boys": "boy",
    "unisex": "unisex",
}

_FIELD_ALIASES = {alias: field.key for field in CATALOG_FIELDS for alias in field.aliases}
_FILTER_LABELS = {field.key: field.label.casefold() for field in CATALOG_FIELDS}
_FILTER_LABELS["garment"] = "category or garment type"
_GROUP_LABELS = {field.key: field.label for field in CATALOG_FIELDS}
_PLAN_FILTERS = tuple(field.key for field in CATALOG_FIELDS)
_PLAN_GROUPS = set(_PLAN_FILTERS)
_TYPE_FILTER_KEYS = ("category", "subcategory", "garment_type", "plm_main_category", "plm_sub_sub_category")
_PLAN_FIELD_LIST = ", ".join((*_PLAN_FILTERS, "garment"))
_PLAN_LIMIT_MAX = 100

_PLAN_SYSTEM = (
    "Convert the PLM product question into one JSON object and nothing else. "
    f"Allowed filter keys: {_PLAN_FIELD_LIST}. "
    "Allowed group_by values: any of those keys except garment, or null. "
    'Shape: {"intent":"count","filters":{},"group_by":null,"limit":null,"season_name":null,"upcoming":false,'
    '"references_prior_chat":false}. '
    'intent is "count", "analysis" (type-wise top breakdown of past products), '
    '"specification" (suggest a new product spec) or "unsupported". '
    "Use analysis when the user says analysis, analyze, profile, breakdown, review, or overview of a buyer's products. "
    "Do not use specification for analysis questions. "
    "Filter values must be concrete catalog names (brands, materials, colors, categories) — never verbs or metric words. "
    "When the user asks for price/positioning by family or brand, set group_by to that dimension and leave filters empty. "
    "For garment or product-type words such as sweater, jumper, dress, or cardigan, put the value in "
    '"garment" (searches category and garment type together). Do not put those words in garment_type alone. '
    "limit is an integer only for top-N or bottom-N questions, else null. "
    'season_name is "winter", "spring", "summer" or null; upcoming is true only when the user says upcoming or next season. '
    "Copy filter values only from the user's words. Use intent unsupported when the question is not about catalog products. "
    "Decide yourself whether the latest question depends on earlier chat. "
    "Set references_prior_chat true only when the latest question refers back to earlier turns "
    "(anaphora, continuation, or pointing at a previous selection). "
    "Set it false for any standalone question that is complete on its own. "
    "When references_prior_chat is false, extract filters only from the latest question. "
    "When true, you may keep filters from earlier user questions. "
    "Never copy filters from assistant answers. Do not rely on a fixed word list — judge from meaning."
)


# --------------------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------------------


def _is_price_history_question(question: str) -> bool:
    """Catalog price movers / trends / history (ecom PriceChangeEvent path)."""

    text = normalize_name(question)
    if re.search(r"\bpricechangeevent\b", text):
        return True
    # "got cheaper" / currency-switch asks may omit the word "price".
    if re.search(r"\b(cheaper|currency\s+switch|currency\s+switches|same[- ]currency)\b", text):
        return True
    if not re.search(r"\bprices?\b", text):
        return False
    return bool(
        re.search(
            r"\b(decrease|decreases|decrease[sd]|increase|increases|increased|"
            r"drop|drops|dropped|rise|rises|rose|rising|fallen|falling|"
            r"change|changes|changed|trend|trends|history|historical|"
            r"fluctuation|fluctuations|move|moves|moved|mover|movers|"
            r"cheaper|expensive|"
            r"over time|last\s+\d+\s+(day|days|month|months|week|weeks))\b",
            text,
        )
    ) or bool(re.search(r"\b(largest|biggest|highest|lowest)\s+price\b", text))


def temporal_count_entity(question: str) -> str | None:
    """Schema temporal node counts — not Product garment filters."""

    text = normalize_name(question)
    if not re.search(r"\b(how many|count|number|exist|exists|stored)\b", text):
        return None
    if re.search(r"\b(price\s*change\s*events?|pricechangeevent)\b", text):
        return "PriceChangeEvent"
    if re.search(r"\b(availability\s*(change\s*)?(snapshots?|events?)|availabilitysnapshot)\b", text):
        return "AvailabilitySnapshot"
    return None


def product_ref_from_question(question: str) -> str | None:
    """Extract an explicit product id / SKU token from the question text."""

    text = str(question or "")
    m = re.search(r"\b(ecom:[a-z0-9_.:-]+)\b", text, flags=re.I)
    if m:
        return m.group(1)
    m = re.search(r"\((\d{6,})\)", text)
    if m:
        return m.group(1)
    m = re.search(r"\bproduct\s+(?:ecom:[a-z0-9_.:-]+|(\d{6,}))\b", text, flags=re.I)
    if m:
        return m.group(0).split()[-1]
    m = re.search(r"\b(\d{6,})\b", text)
    return m.group(1) if m else None


def _is_availability_history_question(question: str) -> bool:
    text = normalize_name(question)
    if not re.search(r"\bavailability\b", text):
        return False
    return bool(
        re.search(
            r"\b(history|historical|changed|change|changes|recent|recently|"
            r"snapshot|snapshots|instock|mixed|outofstock)\b",
            text,
        )
    )


def availability_history_mode(question: str) -> str | None:
    """Classify availability-history asks (AvailabilitySnapshot path)."""

    if not _is_availability_history_question(question):
        return None
    text = normalize_name(question)
    if temporal_count_entity(question) == "AvailabilitySnapshot":
        return "count_snapshots"
    # Product listing / recent-change asks win over brand-ranking phrasing that
    # conversation refine may carry forward from a prior turn.
    if re.search(r"\b(products?|items?|skus?)\b", text) and re.search(
        r"\b(recent|recently|changed|change|changes)\b", text
    ):
        return "recent"
    if re.search(r"\b(brands?|buyers?)\b", text) and re.search(
        r"\b(most|largest|biggest|highest|top|rank|ranking)\b", text
    ):
        return "events_by_brand"
    if re.search(r"\b(recent|recently|changed|history|historical)\b", text):
        return "recent"
    return "events_by_brand" if re.search(r"\b(brands?|buyers?)\b", text) else "recent"


def is_composition_compare_question(question: str) -> bool:
    """Compare materials/composition of a garment across named brands."""

    text = normalize_name(question or "")
    if not re.search(r"\b(composition|compositions|material|materials|fabric|fabrics)\b", text):
        return False
    if not re.search(r"\b(compare|comparison|versus|vs|across|between|similar)\b", text):
        return False
    return len(_compare_party_names(question)) >= 2


def garment_type_from_question(question: str) -> str | None:
    """Pull a product-type phrase for composition/spec compares (e.g. t-shirts)."""

    text = normalize_name(question or "")
    # Prefer multi-word / hyphenated types before generic "shirts".
    patterns = (
        r"\b(t[\s-]?shirts?)\b",
        r"\b(sweat[\s-]?shirts?)\b",
        r"\b(polo[\s-]?shirts?)\b",
        r"\b(crop[\s-]?tops?)\b",
        r"\b(tank[\s-]?tops?)\b",
        r"\b(hoodies?)\b",
        r"\b(sweaters?|jumpers?|cardigans?)\b",
        r"\b(dresses?)\b",
        r"\b(jackets?|coats?)\b",
        r"\b(jeans|trousers|pants)\b",
        # Never bare "top"/"tops" — ranking language ("top counts") is not a garment.
        r"\b(shirts?)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        raw = re.sub(r"\s+", "-", match.group(1).strip())
        if raw in {"jeans", "pants", "trousers"}:
            return raw
        if raw.endswith("ss"):
            return raw
        if raw.endswith("s") and len(raw) > 3:
            return raw[:-1]
        return raw
    return None


def catalog_metric_family(question: str) -> str:
    """Coarse metric family for conversation guards + Thinking labels.

    Distinct families must never REUSE each other's answers across turns.
    """

    text = str(question or "").strip()
    if not text:
        return "catalog"
    if temporal_count_entity(text):
        return "temporal_count"
    try:
        from knowledge_graph.retrieval.ecom import price_history_mode

        if price_history_mode(text):
            return "price"
    except Exception:
        if _is_price_history_question(text):
            return "price"
    if availability_history_mode(text):
        return "availability"
    if _is_average_price_question(text):
        return "avg_price"
    if is_composition_compare_question(text):
        return "composition"
    return "catalog"


def _is_average_price_question(question: str) -> bool:
    """Current-state average price (not historical delta %)."""

    text = normalize_name(question)
    if not re.search(r"\b(average|avg|mean)\b", text):
        return False
    if not re.search(r"\bprices?\b", text):
        return False
    if _is_price_history_question(question) and re.search(
        r"\b(change|changes|trend|trends|history|historical|delta|mover|movers)\b", text
    ):
        return False
    return True


def _is_unsupported_ecom_facet_question(question: str) -> str | None:
    """Facets not present on ecom Product — return a short closed reason."""

    text = normalize_name(question)
    if re.search(r"\b(suppliers?|vendor|vendors|factory|factories)\b", text):
        return "Supplier / vendor entities are not in the ecom product schema."
    if re.search(r"\bseasons?\b", text) and re.search(r"\b(breakdown|distribution|by|wise)\b", text):
        return "Season is not populated on ecom products (all values unspecified unless derived)."
    return None


def is_open_product_listing(question: str, *, limit: int | None = None) -> bool:
    """True for open 'which/show products …' asks without an explicit top-N.

    These should answer with total + facet mix + a few examples — not dump dozens
    of product-name rows just because Cypher used LIMIT 50.
    """

    if limit is not None:
        return False
    text = normalize_name(question)
    if re.search(r"\b(top|bottom|first|last|show|list)\s+\d+\b", text):
        return False
    if re.search(r"\b\d+\s+(products?|items?|styles?)\b", text):
        return False
    if not re.search(r"\bproducts?\b", text):
        return False
    return bool(
        re.search(
            r"\b(which|what|show|list|find|using|use|uses|with|have|has|made)\b",
            text,
        )
    )


def plan_routes(question: str) -> list[str]:
    text = question.lower()
    tokens = set(normalize_name(question).split())
    if "live" in tokens and "plmex" in text:
        return ["live_sql"]
    if temporal_count_entity(question):
        return ["product_count"]
    if _is_price_history_question(question):
        return ["product_count"]
    if _is_availability_history_question(question):
        return ["product_count"]
    if _is_average_price_question(question):
        return ["product_count"]
    if _is_analysis(question):
        return ["product_count"]
    if _is_product_details_request(question):
        return ["product_details"]
    if _is_suggestion(question):
        return ["specification"]
    if re.search(r"\b(top|bottom|highest|lowest|most)\s+\d+\b", normalize_name(question)):
        return ["product_count"]
    if _buyer_ranking_question(question):
        return ["product_count"]
    # "how many <facet>s" — product KG facet entity count, not shared interconnect.
    if _facet_entity_count(question):
        return ["product_count"]
    if ("how many" in text and "product" in text) or _group_field(question) or _product_type_breakdown(question):
        return ["product_count"]
    if news_question(question):
        return ["news"]
    if any(word in text for word in ("share", "shared", "same material", "interconnect")):
        # Shared *products* / materials — system shared overview is selected via kg_system grounding.
        # Do not treat the English verb "found" / "are found" as shared.
        return ["shared_products"]
    if tokens & {"product", "products", "material", "materials", "fabric"}:
        return ["products"]
    if _asked_dimension(question):
        return ["product_count"]
    # Bare count/list asks default to the product catalog — never the shared interconnect.
    if tokens & {"how", "many", "count", "total", "number", "list", "show"}:
        return ["product_count"]
    return ["product_count"]


def _facet_entity_count(question: str, schema=None) -> str | None:
    """'How many <facet>s' → ontology facet label (or product), else None."""

    if schema is not None:
        from fashion_brain.agents.entity_extract import (
            facet_entity_count_from_schema,
            looks_like_facet_product_breakdown,
        )

        if looks_like_facet_product_breakdown(question, schema):
            return None
        hit = facet_entity_count_from_schema(question, schema)
        if hit:
            return hit
    text = normalize_name(question)
    # Breakdown / list of products per facet is group_by, not distinct-facet count.
    if re.search(r"\b(per|by|wise|grouped|breakdown|each|all|every)\b", text) and re.search(
        r"\b(products?|items?|skus?|styles?|list|table|show)\b", text
    ):
        return None
    if not re.search(r"\b(how many|number of|count of)\b", text):
        if not re.search(r"\btotal\b", text):
            return None
        if re.search(r"\btotal\s+(products?|items?|skus?|styles?)\b", text):
            return None
    if re.search(r"\b(per|by|wise|grouped|breakdown|each)\b", text):
        return None
    tokens = set(text.split())
    # Without schema: treat plural facet-ish tokens as product-count routing signal.
    # Concrete label comes from RuntimeSchemaContext when platform is available.
    for token in tokens:
        if token in {"product", "products"}:
            return "product"
        if token.endswith("s") and len(token) > 3 and token not in _SYSTEM_WORDS | _GRAPH_SCOPE_NOISE:
            # brands → brand, colors → color
            return token[:-1] if token.endswith("s") else token
    return None

def _is_analysis(question: str) -> bool:
    """Product-mix analysis on the selected product KG (PLM, ecom, or other) — not ecom-only."""

    tokens = set(normalize_name(question).split())
    if not tokens & _ANALYSIS_WORDS:
        return False
    # "breakdown by gauge" stays a single-field count.
    if _group_field(question):
        return False
    return True


def _buyer_ranking_question(question: str) -> bool:
    """Which buyers make the most products in the PLM data."""

    tokens = set(normalize_name(question).split())
    if not tokens & {"buyer", "buyers"}:
        return False
    return bool(tokens & _RANK_WORDS) or bool(tokens & _MAKE_VERBS)


# --------------------------------------------------------------------------------------
# Answering
# --------------------------------------------------------------------------------------


def answer_question(
    platform: Platform,
    question: str,
    principal: Principal,
    history: list | None = None,
    graph_id: str | None = None,
    conversation_state: dict | None = None,
) -> GroundedAnswer:
    """Route an LLM ask through the agent graph. Template mode stays on the catalog."""

    from fashion_brain.trace import current_trace, end_trace, start_trace

    owns_trace = current_trace() is None
    if owns_trace:
        start_trace(question, source="http/ask")
    try:
        # Conversation session memory (last_graph_id / filters) requires catalog orchestration.
        if conversation_state is not None:
            answer = answer_from_catalog(
                platform,
                question,
                principal,
                history,
                graph_id=graph_id,
                conversation_state=conversation_state,
            )
        elif _use_agent_graph(platform):
            from fashion_brain.flow import ask_with_graph

            answer = ask_with_graph(platform, question, principal, history, graph_id=graph_id)
        else:
            answer = answer_from_catalog(platform, question, principal, history, graph_id=graph_id)
        if owns_trace:
            end_trace(status="ok", routes=getattr(answer, "routes", None), chars=len(getattr(answer, "answer", "") or ""))
        return answer
    except Exception:
        if owns_trace:
            end_trace(status="error")
        raise


def _use_agent_graph(platform: Platform) -> bool:
    settings = platform.settings
    if settings.answer_mode != "llm":
        return False
    provider = str(settings.model_provider or "").strip().casefold()
    if provider in {"", "none", "null"}:
        return False
    if provider == "ollama":
        return True
    from fashion_brain.llm import resolve_model_api_key

    return bool(resolve_model_api_key(settings))


def answer_from_catalog(
    platform: Platform,
    question: str,
    principal: Principal,
    history: list | None = None,
    graph_id: str | None = None,
    conversation_state: dict | None = None,
) -> GroundedAnswer:
    """Intention agent, graph router, knowledge-graph fetch, response agent."""

    from fashion_brain.agents import ResponseAgent

    retrieved, routes, _intention = draft_catalog(
        platform,
        question,
        principal,
        history,
        graph_id=graph_id,
        conversation_state=conversation_state,
    )
    LOGGER.info("step 4/4 response agent started")
    answer = ResponseAgent().compose(platform, question, retrieved)
    LOGGER.info("step 4/4 response agent finished chars=%s", len(answer))
    return _grounded(question, routes, answer, retrieved, platform)


def draft_catalog(
    platform: Platform,
    question: str,
    principal: Principal,
    history: list | None = None,
    summary=None,
    graph_id: str | None = None,
    conversation_state: dict | None = None,
):
    """Run the catalog steps through retrieval and return the retrieved notes plus routes."""

    from fashion_brain.orchestration.catalog_pipeline import draft_catalog_orchestrated

    retrieved, routes, intention, _context, _choice = draft_catalog_orchestrated(
        platform,
        question,
        principal,
        history,
        summary=summary,
        graph_id=graph_id,
        conversation_state=conversation_state,
    )
    LOGGER.info(
        "catalog pipeline finished evidence=%s chars=%s routes=%s",
        len(retrieved.evidence),
        len(retrieved.text),
        routes,
    )
    return retrieved, routes, intention


def _grounded(question: str, routes: list[str], answer: str, retrieved, platform: Platform) -> GroundedAnswer:
    synthetic = getattr(retrieved, "synthetic_data", False)
    if not synthetic:
        products = getattr(platform, "products", None)
        envelopes = getattr(products, "envelopes", None) or []
        synthetic = any(getattr(envelope, "synthetic", False) for envelope in envelopes)
    return GroundedAnswer(
        question=question,
        routes=routes,
        answer=answer,
        evidence=_dedupe_evidence(retrieved.evidence),
        caveats=retrieved.caveats,
        unavailable=retrieved.unavailable,
        synthetic_data=synthetic,
    )


def enrich_ecom_question_filters(question: str, filters: dict | None = None) -> dict:
    """Fill availability / price / section cues the LLM often omits."""

    out = {key: value for key, value in dict(filters or {}).items() if value not in (None, "")}
    text = (question or "").casefold()
    if "availability" not in out and re.search(
        r"\b(in[\s_-]?stock|instock|available now|currently available)\b", text
    ):
        out["availability"] = "InStock"
    if "price_max" not in out:
        match = re.search(
            r"(?:under|below|less than|cheaper than|<)\s*[€$£]?\s*(\d+(?:\.\d+)?)",
            text,
        )
        if match:
            out["price_max"] = match.group(1)
    if "currency" not in out and (" eur" in f" {text}" or "€" in (question or "") or "euro" in text):
        out["currency"] = "EUR"
    if "section" not in out and re.search(r"\bwomen\b", text):
        out["section"] = "women"
    # Composition/spec compares need the named garment kept even when EntityExtract is thin.
    if is_composition_compare_question(question) and not any(
        out.get(k) for k in ("garment", "garment_type", "category")
    ):
        garment = garment_type_from_question(question)
        if garment:
            out["garment"] = garment
    parties = _compare_party_names(question)
    if len(parties) >= 2:
        out["brand"] = " | ".join(parties)
    return out


def parent_category_name(question: str) -> str | None:
    """Extract 'Jackets' from 'parent category of Jackets' / 'what is the parent of X'."""

    text = question or ""
    match = re.search(
        r"(?:parent\s+category\s+of|parent\s+of|belongs\s+under)\s+([A-Za-z][\w\s\-&']{1,40})",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    name = match.group(1).strip().rstrip("?.!,")
    return name or None


def _retrieve(
    platform: Platform,
    principal: Principal,
    intention,
    question: str = "",
    graph_id: str | None = None,
    graph_ids: list[str] | None = None,
):
    from fashion_brain.agents import Retrieved
    from fashion_brain.execution.kg_client import retrieve_remote, uses_kg_api
    from fashion_brain.trace import trace_debug, trace_step
    from knowledge_graph.errors import GraphBackendUnavailable
    from knowledge_graph.retrieval.brain import BrainRetrieveRequest, retrieve as local_retrieve

    filters = enrich_ecom_question_filters(question, dict(intention.filters or {}))
    parent_of = parent_category_name(question)
    if parent_of:
        filters = {**filters, "parent_of": parent_of}

    request = BrainRetrieveRequest(
        question=question,
        intent="parent_category" if parent_of else intention.intent,
        filters=filters,
        group_by=intention.group_by,
        unsupported=bool(intention.unsupported),
        limit=intention.limit,
        suggest=(bool(intention.suggest) or intention.intent == "specification") and not bool(getattr(intention, "analysis", False)),
        analysis=bool(getattr(intention, "analysis", False)),
        season_name=intention.season_name,
        upcoming=bool(intention.upcoming),
        corrections=list(intention.corrections or []),
        count_entity=getattr(intention, "count_entity", None),
        graph_id=graph_id or None,
        graph_ids=list(graph_ids) if graph_ids else None,
    )
    remote = uses_kg_api(platform.settings)
    trace_step(
        "kg",
        "Brain retrieve request",
        mode="remote" if remote else "local",
        intent=request.intent,
        filters=request.filters,
        group_by=request.group_by,
        analysis=request.analysis,
        graph_id=request.graph_id,
        graph_ids=request.graph_ids,
    )
    try:
        if remote:
            payload = retrieve_remote(platform.settings, principal.clearance, request)
        else:
            payload = local_retrieve(platform, principal, request)
    except GraphBackendUnavailable as exc:
        # Fail fast with a usable message instead of an empty Fashion Brain bubble.
        message = (
            f"The knowledge graph retrieve timed out or failed ({exc}). "
            "Try a narrower question (one brand or one field), or retry shortly."
        )
        trace_step("kg", "Brain retrieve failed fast", error=str(exc)[:200])
        return Retrieved(
            text=message,
            evidence=[],
            caveats=["Knowledge graph retrieve did not finish in time."],
            unavailable=["knowledge_graph_timeout"],
            synthetic_data=False,
        )
    trace_debug(
        "Brain retrieve response",
        chars=len(payload.text or ""),
        evidence=len(payload.evidence or []),
        caveats=list(payload.caveats or [])[:3],
        unavailable=list(payload.unavailable or [])[:3],
    )
    return Retrieved(
        text=payload.text,
        evidence=list(payload.evidence),
        caveats=list(payload.caveats),
        unavailable=list(payload.unavailable),
        synthetic_data=bool(payload.synthetic_data),
    )


# --------------------------------------------------------------------------------------
# Keyword planner (fallback)
# --------------------------------------------------------------------------------------


def _alias_at(tokens: list[str], index: int) -> tuple[str, int] | None:
    """Match the longest catalog field name starting at this token."""

    found: tuple[str, int] | None = None
    for length in range(1, 7):
        if index + length > len(tokens):
            break
        field = _FIELD_ALIASES.get(" ".join(tokens[index : index + length]))
        if field:
            found = (field, length)
    return found


def _group_marker_at(tokens: list[str], index: int) -> tuple[str, int] | None:
    """A breakdown phrase at this token. Returns (field, index after the phrase).

    Recognised shapes: "by|per|each <field>", "<field> wise|breakdown|distribution|split",
    and "breakdown|distribution|split [by] <field>".
    """

    token = tokens[index]
    if token in _BY_WORDS:
        found = _alias_at(tokens, index + 1)
        if found:
            return found[0], index + 1 + found[1]
    found = _alias_at(tokens, index)
    if found:
        after = index + found[1]
        if after < len(tokens) and tokens[after] in _BREAKDOWN_SUFFIXES:
            return found[0], after + 1
    if token in _BREAKDOWN_HEADS:
        start = index + 1
        if start < len(tokens) and tokens[start] == "by":
            start += 1
        found = _alias_at(tokens, start)
        if found:
            return found[0], start + found[1]
    return None


def _group_field(question: str) -> str | None:
    tokens = normalize_name(question).split()
    for index in range(len(tokens)):
        marker = _group_marker_at(tokens, index)
        if marker:
            return marker[0]
    return None


def _count_dimension(question: str) -> str | None:
    """The catalog column a question wants counted, including 'which type of products'."""

    return _group_field(question) or _asked_dimension(question) or _product_type_breakdown(question)


def _product_type_breakdown(question: str) -> str | None:
    """'Which type of products' asks for a garment-type ranking, not a garment named type."""

    tokens = normalize_name(question).split()
    for index in range(len(tokens)):
        for pattern in _TYPE_PATTERNS:
            if tuple(tokens[index : index + len(pattern)]) == pattern:
                return "garment_type"
    return None


def _is_suggestion(question: str) -> bool:
    if _is_analysis(question):
        return False
    tokens = normalize_name(question).split()
    if set(tokens) & _SUGGEST_WORDS:
        return True
    text = " ".join(tokens)
    if any(phrase in text for phrase in ("new product", "new sample", "new style", "sample for")):
        return True
    has_spec_term = bool(set(tokens) & {"spec", "specs", "specification", "specifications"})
    has_creation_verb = bool(set(tokens) & {"make", "makes", "create", "creates", "design", "develop", "prepare", "draft"})
    return has_spec_term and has_creation_verb


def _is_product_details_request(question: str) -> bool:
    if _is_suggestion(question):
        return False
    text = normalize_name(question)
    spec_term = r"(?:specifications?|specs?|details?|attributes?)"
    return bool(
        re.search(rf"\b(?:full|complete|detailed|exact)\b.{{0,32}}\b{spec_term}\b", text)
        or re.search(rf"\b{spec_term}\s+(?:for|of)\b", text)
    )


def _is_party_token(token: str) -> bool:
    """A buyer/brand name, not a material, color, field, metric, or task word."""

    if (
        token in _MAKE_VERBS
        or token in _MAKE_ADVERBS
        or token in _COUNT_STOPWORDS
        or token in _SYSTEM_WORDS
        or token in _ANALYSIS_WORDS
        or token in _HISTORY_WORDS
        or token in _METRIC_OPERATOR_WORDS
        or token in _MEASURE_PROPERTY_WORDS
        or token in _TASK_WORDS
        or token in _RANK_WORDS
        or token in _MATERIAL_WORDS
        or token in _COLOR_WORDS
        or token in _GENDER_VALUES
        or token in _SEASON_WORDS
    ):
        return False
    if token.isdigit() or re.fullmatch(r"(19|20)\d{2}", token):
        return False
    if token in _FIELD_ALIASES:
        return False
    return len(token) > 1


def _take_party_name(tokens: list[str], index: int) -> list[str]:
    """One party token only — product title words must not become multi-brand filters."""

    if index < len(tokens) and _is_party_token(tokens[index]):
        return [tokens[index]]
    return []


def _make_verb_follows(tokens: list[str], index: int) -> int | None:
    """Index of a make-verb at or after ``index``, skipping adverbs like 'normally'."""

    while index < len(tokens) and tokens[index] in _MAKE_ADVERBS:
        index += 1
    if index < len(tokens) and tokens[index] in _MAKE_VERBS:
        return index
    return None


def _reliable_filter(value: str) -> bool:
    """Reject a model filter that pasted the question instead of a catalog value."""

    tokens = normalize_name(value).split()
    if not tokens or len(tokens) > 4:
        return False
    if set(tokens) & _SYSTEM_WORDS:
        return False
    if set(tokens) & _METRIC_OPERATOR_WORDS:
        return False
    if set(tokens) & _TASK_WORDS:
        return False
    if set(tokens) & _RANK_WORDS:
        return False
    # Bare measure names ("price") are dimensions, not product types.
    if set(tokens) <= _MEASURE_PROPERTY_WORDS:
        return False
    # Multi-party filters must be real party tokens only (e.g. "zara | primark").
    if "|" in str(value):
        parts = [normalize_name(p) for p in str(value).split("|")]
        if not parts or any(not _is_party_token(p) for p in parts if p):
            return False
        return True
    return not (set(tokens) & _FILTER_NOISE)


def _asked_dimension(question: str) -> str | None:
    tokens = normalize_name(question).split()
    if not tokens or tokens[0] not in {"what", "which", "list", "show"}:
        return None
    found = _alias_at(tokens, 1)
    return found[0] if found else None


def _compare_party_names(question: str) -> list[str]:
    """Named parties in a compare ask: 'Primark or Zara', 'A vs B', 'A and B'."""

    text = str(question or "").strip()
    if not text:
        return []
    patterns = (
        r"(?i)\b([A-Za-z][\w&'-]{1,40})\s+or\s+([A-Za-z][\w&'-]{1,40})\s*[?.!]?\s*$",
        r"(?i)\b([A-Za-z][\w&'-]{1,40})\s+vs\.?\s+([A-Za-z][\w&'-]{1,40})\b",
        r"(?i)\b([A-Za-z][\w&'-]{1,40})\s+versus\s+([A-Za-z][\w&'-]{1,40})\b",
        r"(?i)\bbetween\s+([A-Za-z][\w&'-]{1,40})\s+and\s+([A-Za-z][\w&'-]{1,40})\b",
        # "Zara and Primark" inside a longer sentence (compare / similar / price asks).
        r"(?i)\b([A-Za-z][\w&'-]{1,40})\s+and\s+([A-Za-z][\w&'-]{1,40})\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            left = match.group(1).strip()
            right = match.group(2).strip()
            a = normalize_name(left)
            b = normalize_name(right)
            if not a or not b or a == b:
                continue
            if not (_is_party_token(a) and _is_party_token(b)):
                continue
            # Keep original casing for display; filters normalize later.
            return [left, right]
    return []


def split_multi_party_filter(value: str | None) -> list[str]:
    """Split brand/buyer compare values encoded as 'Primark | Zara'."""

    text = str(value or "").strip()
    if not text:
        return []
    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    return [text]


def _is_filter_term(token: str) -> bool:
    return (
        token not in _COUNT_STOPWORDS
        and token not in _COLOR_MARKERS
        and token not in _SYSTEM_WORDS
        and token not in _ANALYSIS_WORDS
        and token not in _HISTORY_WORDS
        and token not in _RANK_WORDS
        and token not in _METRIC_OPERATOR_WORDS
        and token not in _MEASURE_PROPERTY_WORDS
        and token not in _TASK_WORDS
        and token not in _MAKE_VERBS
        and len(token) > 1
    )


def _is_value_token(token: str) -> bool:
    return token.isdigit() or _is_filter_term(token)


def resolve_catalog_plan(question: str, history: list | None = None, *, carry: bool | None = None) -> dict:
    """Keyword plan from the latest question. Earlier turns merge only when carry=True (agent decision)."""

    from fashion_brain.trace import trace_debug

    current = _catalog_plan(question)
    trace_debug(
        "Keyword plan",
        filters=current.get("filters"),
        group_by=current.get("group_by"),
        analysis=current.get("analysis"),
        preferred=current.get("preferred_graphs"),
        carry=bool(carry),
    )
    if not carries_history_filters(question, history, carry=carry):
        return {
            "filters": dict(current["filters"]),
            "group_by": current["group_by"],
            "unsupported": False,
            "limit": current.get("limit"),
            "suggest": bool(current.get("suggest")),
            "analysis": bool(current.get("analysis")),
            "season_name": current.get("season_name"),
            "upcoming": bool(current.get("upcoming")),
            "preferred_graphs": list(current.get("preferred_graphs") or []),
        }

    running_filters: dict[str, str] = {}
    running_group: str | None = None
    running_limit: int | None = None
    for text in _user_questions(history):
        step = _catalog_plan(text)
        running_filters.update(step["filters"])
        if step.get("group_by"):
            running_group = step["group_by"]
        if step.get("limit"):
            running_limit = step["limit"]
    running_filters.update(current["filters"])
    if current.get("group_by"):
        running_group = current["group_by"]
    if current.get("limit"):
        running_limit = current["limit"]
    return {
        "filters": running_filters,
        "group_by": running_group,
        "unsupported": False,
        "limit": running_limit,
        "suggest": bool(current.get("suggest")),
        "analysis": bool(current.get("analysis")),
        "season_name": current.get("season_name"),
        "upcoming": bool(current.get("upcoming")),
        "preferred_graphs": list(current.get("preferred_graphs") or []),
    }


def catalog_focus(intention) -> str:
    """Short thinking line for the filters that will be sent to the graph."""

    if getattr(intention, "intent", None) == "system_meta":
        return "Reading registered knowledge graphs, source descriptions, attributes, and the shared interconnect."
    parts = []
    phrase = _filter_phrase(getattr(intention, "filters", {}) or {})
    if phrase:
        parts.append(phrase[0].upper() + phrase[1:] + ".")
    group_by = getattr(intention, "group_by", None)
    if group_by:
        parts.append(f"Grouped by {group_by.replace('_', ' ')}.")
    season_name = getattr(intention, "season_name", None)
    if season_name:
        parts.append(f"Season {season_name}.")
    if getattr(intention, "analysis", False):
        parts.append("Full product analysis across key attributes; presentation chosen by the response.")
    if getattr(intention, "suggest", False):
        parts.append("Suggesting specifications from the styles this buyer already has.")
    if getattr(intention, "intent", None) == "product_details":
        parts.append("Retrieving full specifications from matching catalog products.")
    for note in getattr(intention, "corrections", []) or []:
        parts.append(note)
    return " ".join(parts)


def _user_questions(history: list | None) -> list[str]:
    questions = []
    for turn in history or []:
        if isinstance(turn, dict):
            role = turn.get("role")
            content = turn.get("content")
        else:
            role = getattr(turn, "role", None)
            content = getattr(turn, "content", None)
        if role != "user":
            continue
        text = str(content or "").strip()
        if text:
            questions.append(text)
    return questions


def carries_history_filters(question: str, history: list | None = None, *, carry: bool | None = None) -> bool:
    """Whether earlier-turn filters may merge into this question.

    ``carry`` must come from the thinking/planning agent (references_prior_chat).
    There is no static word list — without an agent decision, history is not applied.
    """

    if not history:
        return False
    if carry is None:
        return False
    return bool(carry)


def _question_tokens(question: str) -> list[str]:
    """Normalize and glue graph phrases like 'e comm' / 'e-commerces' → ecomm/ecommerce."""

    tokens = normalize_name(question).split()
    glued: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        nxt = tokens[index + 1] if index + 1 < len(tokens) else ""
        if token == "e" and nxt in {"comm", "com"}:
            glued.append("ecomm")
            index += 2
            continue
        if token == "e" and nxt in {"commerce", "commerces"}:
            glued.append("ecommerce")
            index += 2
            continue
        if token in {"ecommerce", "ecommerces"}:
            glued.append("ecommerce")
            index += 1
            continue
        glued.append(token)
        index += 1
    return glued


def _graph_scope_signal_tokens(tokens: set[str]) -> set[str]:
    """Tokens that can name a catalog world — drop grammar and generic 'products/data'."""

    return {tok for tok in tokens if len(tok) >= 2 and tok not in _GRAPH_SCORE_STOPWORDS}


def _id_part_scope_hits(signal: set[str], id_parts: set[str]) -> set[str]:
    """Exact or prefix id-fragment hits (plm→plmex, ecom→ecom_products)."""

    hits: set[str] = set()
    for part in id_parts:
        if len(part) < 3:
            continue
        if part in signal:
            hits.add(part)
            continue
        for tok in signal:
            if len(tok) < 3:
                continue
            if part.startswith(tok) or tok.startswith(part):
                hits.add(part)
                break
    return hits


def preferred_graphs_from_question(question: str, platform=None) -> list[str]:
    """Match question tokens to live registry name/description/capabilities/id.

    Never hard-codes graph ids. When no token matches, returns [] so callers
    fall back to catalog.preferred_graphs() / first active product graph.

    Returns at most one product graph unless the question is clearly cross-source
    (compare / across / both), so "PLM data" does not also select e-comm.
    """

    tokens = set(_question_tokens(question))
    signal = _graph_scope_signal_tokens(tokens)
    text = " ".join(_question_tokens(question))
    wants_cross = any(
        hint in f" {text} "
        for hint in (
            " compare ",
            " versus ",
            " vs ",
            " across ",
            " both sources ",
            " cross source ",
            " cross-source ",
            " using both ",
            " from both ",
            " and plm ",
            " and plmex ",
            " and ecom ",
            " and ecomm ",
            " and ecommerce ",
            " ecom and ",
            " ecomm and ",
            " ecommerce and ",
            " plm and ",
            " plmex and ",
        )
    )
    scored: list[tuple[int, str]] = []
    graphs = _registry_graph_rows(platform)
    for row in graphs:
        graph_id = str(row.get("graph_id") or row.get("id") or "")
        if not graph_id:
            continue
        kind = str(row.get("kind") or "")
        if kind == "interconnect":
            continue
        # Strong identity fields first — ignore negated "do not use for …" prose.
        identity = normalize_name(
            " ".join(
                [
                    graph_id,
                    str(row.get("name") or row.get("graph_name") or ""),
                    " ".join(str(x) for x in (row.get("source_ids") or [])),
                    " ".join(str(x) for x in (row.get("capabilities") or [])),
                ]
            )
        )
        description = str(row.get("description") or "")
        # Drop negative clauses so "do not use for plm" cannot claim a PLM question.
        description = re.sub(
            r"\b(?:do not use|don't use|not for|never for|excluding|except)\b.*?(?:\.|$)",
            " ",
            description,
            flags=re.IGNORECASE,
        )
        desc = normalize_name(description)
        id_parts = {normalize_name(part) for part in re.split(r"[_\-\s]+", graph_id) if part}
        identity_tokens = set(identity.split()) | id_parts
        desc_tokens = set(desc.split())
        score = 0
        # Distinctive tokens only — "products"/"in" must not let e-comm beat PLM.
        scope_hits = _id_part_scope_hits(signal, id_parts)
        hit_id = signal & identity_tokens
        hit_desc = signal & desc_tokens
        if scope_hits:
            score += 20 * len(scope_hits)
        elif hit_id:
            score += 10 * len(hit_id)
        if hit_desc:
            score += 4 * len(hit_desc - scope_hits - hit_id)
        if score <= 0 and signal:
            for part in sorted(identity_tokens, key=len, reverse=True):
                if len(part) < 3 or part in _GRAPH_SCORE_STOPWORDS:
                    continue
                if part in text or any(part.startswith(tok) or tok.startswith(part) for tok in signal if len(tok) >= 3):
                    score += 12 if part in id_parts else 4
                    break
        if score > 0:
            scored.append((score, graph_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    if not scored:
        return []
    # Two+ distinctive catalog hits (ecom + plm) → multi-graph, e.g. study retail then PLM spec.
    distinctive = [gid for score, gid in scored if score >= 20]
    if wants_cross or len(distinctive) >= 2:
        return [gid for _score, gid in scored if _score > 0]
    # Single best match — keeps PLM vs e-comm answers scoped.
    best = scored[0][0]
    top = [gid for score, gid in scored if score == best]
    return top[:1]


def _registry_graph_rows(platform=None) -> list[dict]:
    if platform is None:
        return []
    try:
        from fashion_brain.planning.runtime_schema import get_runtime_schema

        ctx = get_runtime_schema(platform)
        return [
            {
                "graph_id": g.graph_id,
                "id": g.graph_id,
                "name": g.name,
                "kind": g.kind,
                "description": g.description,
                "capabilities": g.capabilities,
                "source_ids": g.source_ids,
                "status": g.status,
            }
            for g in ctx.graphs
            if g.status == "active"
        ]
    except Exception:
        return []


def _remap_buyer_to_brand_if_needed(filters: dict, group_by: str | None, preferred: list[str], platform=None) -> tuple[dict, str | None]:
    """If preferred product graphs expose Brand but not Buyer, remap buyer→brand."""

    if not preferred:
        return filters, group_by
    has_brand = False
    has_buyer = False
    for gid in preferred:
        try:
            from fashion_brain.planning.runtime_schema import get_runtime_schema

            ctx = get_runtime_schema(platform, gid) if platform is not None else None
            if ctx is None:
                continue
            keys = {k.casefold() for k in (ctx.filter_keys or [])}
            facets = {t.casefold() for t in (ctx.facet_types or [])}
            if "brand" in keys or "brand" in facets or (ctx.brand_label or "").casefold() == "brand":
                has_brand = True
            if "buyer" in keys or "buyer" in facets:
                has_buyer = True
        except Exception:
            continue
    if has_brand and not has_buyer:
        out = dict(filters or {})
        if group_by == "buyer":
            group_by = "brand"
        if out.get("buyer") and not out.get("brand"):
            out["brand"] = out.pop("buyer")
        else:
            out.pop("buyer", None)
        return out, group_by
    return filters, group_by


def _all_parties_dimension(tokens: list[str]) -> str | None:
    """'all buyers' / 'all brands' → group by that field, with no party filter value."""

    for index, token in enumerate(tokens):
        if token != "all":
            continue
        after = tokens[index + 1] if index + 1 < len(tokens) else ""
        if after in {"buyer", "buyers"}:
            return "buyer"
        if after in {"brand", "brands"}:
            return "brand"
    return None


def _catalog_plan(question: str) -> dict:
    """Read filters and one breakdown field from the question words. Counts are not guessed here."""

    tokens = _question_tokens(question)
    filters: dict[str, list[str]] = {}
    group_by = _count_dimension(question)
    limit: int | None = None
    season_name = None
    suggest = _is_suggestion(question)
    analysis = _is_analysis(question)
    upcoming = bool(set(tokens) & _UPCOMING_WORDS)
    all_parties = _all_parties_dimension(tokens)
    if all_parties:
        group_by = all_parties
    preferred_graphs = preferred_graphs_from_question(question)
    # Resolve compare parties first so leftover tokens never become garment/brand noise.
    compare_parties = _compare_party_names(question)
    compare_set = {normalize_name(p) for p in compare_parties}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if compare_set and normalize_name(token) in compare_set:
            index += 1
            continue

        if (
            token in _ANALYSIS_WORDS
            or token in _HISTORY_WORDS
            or token in _METRIC_OPERATOR_WORDS
            or token in _MEASURE_PROPERTY_WORDS
            or token in _TASK_WORDS
        ):
            index += 1
            continue
        # Ranking adjectives without "top N" are not garment/category values.
        if token in _RANK_WORDS and not (
            index + 1 < len(tokens) and tokens[index + 1].isdigit()
        ):
            index += 1
            continue

        # "top 5 [field]" / "show 5 …"
        if token in (_RANK_WORDS | {"show", "list", "give"}) and index + 1 < len(tokens) and tokens[index + 1].isdigit():
            limit = int(tokens[index + 1])
            found = _alias_at(tokens, index + 2)
            if found:
                group_by = found[0]
                index += 2 + found[1]
            else:
                index += 2
            continue

        # breakdown phrases were already read into group_by; skip their words
        marker = _group_marker_at(tokens, index)
        if marker:
            index = marker[1]
            continue

        found = _alias_at(tokens, index)

        # "<field> value [value ...]" — but plural facet words used as sentence subjects
        # ("how are brands positioned…") are NOT filter keys. Prefer "brand primark".
        def _field_value_ok(at: int) -> bool:
            if at >= len(tokens):
                return False
            tok = tokens[at]
            nxt = tokens[at + 1] if at + 1 < len(tokens) else None
            if _is_value_token(tok):
                return True
            # Hyphen splits: "t-shirts" → "t" "shirts"
            return (
                len(tok) == 1
                and nxt is not None
                and _is_value_token(nxt)
                and nxt not in _FIELD_ALIASES
            )

        if found and _field_value_ok(index + found[1]):
            surface = " ".join(tokens[index : index + found[1]])
            plural_subject = found[1] == 1 and surface.endswith("s") and surface in _FIELD_ALIASES
            if plural_subject:
                index += found[1]
                continue
            values: list[str] = []
            index += found[1]
            while index < len(tokens) and len(values) < 3 and _field_value_ok(index):
                tok = tokens[index]
                nxt = tokens[index + 1] if index + 1 < len(tokens) else None
                if nxt in _COLOR_MARKERS:
                    break
                if tok in _FIELD_ALIASES and nxt in _FIELD_ALIASES:
                    break
                values.append(tok)
                index += 1
            if values and found[0] not in filters:
                filters[found[0]] = values
            continue

        if token in _COLOR_MARKERS:
            if index + 1 < len(tokens) and _is_filter_term(tokens[index + 1]):
                filters.setdefault("color", []).append(tokens[index + 1])
                index += 2
            else:
                index += 1
            continue

        if token.isdigit() and index + 1 < len(tokens) and tokens[index + 1] in _MATERIAL_WORDS:
            filters.setdefault("material", []).append(f"{token}% {tokens[index + 1]}")
            index += 2
            continue

        if found:
            index += found[1]
            continue

        # "which types does <buyer> [normally] make"
        if group_by == "garment_type" and token in _MAKE_VERBS:
            name = _take_party_name(tokens, index + 1)
            if name:
                filters.setdefault("buyer", []).extend(name)
                index += 1 + len(name)
            else:
                index += 1
            continue
        if group_by == "garment_type" and _is_party_token(token):
            verb_at = _make_verb_follows(tokens, index + 1)
            if verb_at is not None:
                filters.setdefault("buyer", []).append(token)
                index = verb_at + 1
                continue

        mapped_season = _SEASON_WORDS.get(token)
        if mapped_season:
            season_name = mapped_season
            index += 1
            continue

        # Origin: "made|manufactured|produced|sourced in <country>" — never garment.
        if token in {"made", "manufactured", "produced", "sourced"} and index + 1 < len(tokens) and tokens[index + 1] == "in":
            place = _take_party_name(tokens, index + 2)
            if place:
                filters.setdefault("country", []).extend(place)
                index += 2 + len(place)
                continue

        # "for|at|by <brand|buyer>" — never treat the party name as garment.
        # Covers: "products at zara", "for matalan". ("from" is ambiguous with origin.)
        if token in {"for", "at", "by"}:
            name = _take_party_name(tokens, index + 1)
            if name:
                filters.setdefault("brand", []).extend(name)
                filters.setdefault("buyer", []).extend(name)
                index += 1 + len(name)
                continue
        if token == "from":
            # Prefer brand only when clearly a party cue; else skip (origin handled above).
            name = _take_party_name(tokens, index + 1)
            if name and (index + 1 + len(name) < len(tokens) and tokens[index + 1 + len(name)] in {"brand", "brands", "buyer", "buyers"}):
                filters.setdefault("brand", []).extend(name)
                filters.setdefault("buyer", []).extend(name)
                index += 1 + len(name)
                continue
            if name and any(t in {"brand", "brands"} for t in tokens):
                filters.setdefault("brand", []).extend(name)
                index += 1 + len(name)
                continue
            index += 1
            continue

        if not _is_filter_term(token):
            index += 1
            continue

        # "<party> brand|buyer" — e.g. "zara brand".
        # Not "to brand Matalan" / "belong to brand X" — those are "brand <value>" field forms.
        if index + 1 < len(tokens) and tokens[index + 1] in {"brand", "brands", "buyer", "buyers"}:
            role_at = index + 1
            value_at = index + 2
            if value_at < len(tokens) and _is_value_token(tokens[value_at]) and _is_party_token(tokens[value_at]):
                index += 1
                continue
            if _is_party_token(token):
                role = tokens[role_at]
                if role.startswith("brand"):
                    filters.setdefault("brand", []).append(token)
                    filters.pop("buyer", None)
                else:
                    filters.setdefault("buyer", []).append(token)
                    filters.pop("brand", None)
                index += 2
                continue

        if index + 1 < len(tokens) and tokens[index + 1] in _COLOR_MARKERS:
            filters.setdefault("color", []).append(token)
            index += 2
            continue

        if token in _COLOR_WORDS:
            filters.setdefault("color", []).append(token)
        elif token in _GENDER_VALUES and group_by != "gender":
            filters.setdefault("gender", []).append(_GENDER_VALUES[token])
        elif token in _MATERIAL_WORDS:
            filters.setdefault("material", []).append(token)
        elif re.fullmatch(r"(19|20)\d{2}", token):
            filters.setdefault("season", []).append(token)
        elif token in _FIELD_ALIASES:
            # Dimension names (family, brand, …) are group keys, never garment values.
            index += 1
            continue
        elif group_by:
            # Breakdown asks: leftover grammar is not a garment filter.
            index += 1
            continue
        elif analysis:
            # Analysis / profile asks: leftover task words are never garment values.
            # Party comes from "X brand" / analysis subject; facets only if explicitly named.
            index += 1
            continue
        elif filters.get("country"):
            # Origin ask already grounded — do not invent garment from leftover place words.
            index += 1
            continue
        else:
            # Do NOT invent garment/brand from free tokens here.
            # Ambiguous names (Primark vs sweater) are grounded by EntityExtract + live Brand nodes.
            index += 1
            continue
        index += 1

    if analysis:
        suggest = False
        if limit is None:
            limit = 5
        # Task phrasing must never survive as garment/category on analysis asks.
        filters.pop("garment", None)
        filters.pop("garment_type", None)
        field_wise = bool(group_by) and (
            "wise" in tokens or bool(set(tokens) & _BREAKDOWN_HEADS) or "by" in tokens
        )
        if all_parties:
            # "analysis all buyers in ecomm" → breakdown of every party, no single-party filter.
            group_by = all_parties
            filters.pop("buyer", None)
            filters.pop("brand", None)
        elif field_wise:
            # "analysis … material wise" → keep the column breakdown; not a single-party profile.
            analysis = False
        else:
            group_by = None
            party = _analysis_party_name(tokens)
            if party and party not in _SYSTEM_WORDS and party not in _GRAPH_SCOPE_NOISE:
                explicit_buyer = bool(set(tokens) & {"buyer", "buyers"})
                explicit_brand = bool(set(tokens) & {"brand", "brands"})
                if explicit_brand and not explicit_buyer:
                    filters["brand"] = [party]
                    filters.pop("buyer", None)
                elif explicit_buyer and not explicit_brand:
                    filters["buyer"] = [party]
                    filters.pop("brand", None)
                else:
                    # Unqualified party (e.g. "analysis for X"): keep both until
                    # live metadata picks brand vs buyer for the selected KG.
                    filters["brand"] = [party]
                    filters["buyer"] = [party]

    # Named "A and/or/vs B" parties are always brands — never garments/task leftovers.
    if compare_parties:
        filters["brand"] = list(compare_parties)
        filters.pop("buyer", None)
        filters.pop("garment", None)
        filters.pop("garment_type", None)
        # Keep an explicit category if the user named one; drop invented leftovers.
        cat_vals = filters.get("category") or []
        if cat_vals and not _reliable_filter(" ".join(str(v) for v in cat_vals)):
            filters.pop("category", None)
        brand_compare = (
            group_by in {"brand", "buyer"}
            or any(tok in {"brand", "brands", "buyer", "buyers"} for tok in tokens)
            or bool(re.search(r"\b(more|fewer|less|most|least)\b", " ".join(tokens)))
        )
        # Price-history compares keep brand filters but do not force a category breakdown.
        if brand_compare and not _is_price_history_question(question):
            group_by = group_by if group_by in {"brand", "buyer"} else "brand"

    flat: dict[str, str] = {}
    for key, values in filters.items():
        if key in {"brand", "buyer"} and len(values) > 1:
            flat[key] = " | ".join(values)
        else:
            flat[key] = " ".join(values)
    flat = sanitize_party_filters(flat)
    flat = _drop_group_by_as_filter(flat, group_by, question)
    flat, group_by = _remap_buyer_to_brand_if_needed(flat, group_by, preferred_graphs)
    # Ranking/metric phrasing must never survive as garment/category values.
    _drop_unreliable_garment(flat)
    for key in ("garment", "garment_type", "category", "family", "subfamily", "material", "color"):
        value = flat.get(key)
        if value and not _reliable_filter(str(value)):
            flat.pop(key, None)

    # Explicit product refs belong on product_id — never as garment leftovers.
    pref = product_ref_from_question(question)
    if pref:
        flat["product_id"] = pref
        for junk in ("garment", "garment_type"):
            flat.pop(junk, None)
    # Temporal node counts are not facet filters.
    if temporal_count_entity(question):
        for junk in ("garment", "garment_type", "category", "family", "subfamily"):
            flat.pop(junk, None)
        group_by = None
    # Price / availability history asks: drop invented facet noise.
    if _is_price_history_question(question) or _is_availability_history_question(question):
        for junk in ("garment", "garment_type"):
            flat.pop(junk, None)

    if limit is None and group_by == "garment_type" and flat.get("buyer"):
        limit = 10
    if limit is None and group_by in {"buyer", "brand"} and (set(tokens) & _RANK_WORDS):
        limit = 10
    return {
        "filters": flat,
        "group_by": group_by,
        "unsupported": False,
        "limit": limit,
        "suggest": suggest,
        "analysis": analysis,
        "season_name": season_name,
        "upcoming": upcoming,
        "preferred_graphs": preferred_graphs,
        "temporal_count": temporal_count_entity(question),
        "average_price": _is_average_price_question(question),
    }


def _party_filter_core(value: str) -> str:
    """Strip prepositions / 'brand'|'buyer' wrappers so 'at zara' → 'zara'."""

    text = normalize_name(str(value))
    text = re.sub(r"^(at|from|for|by|of)\s+", "", text)
    text = re.sub(r"\b(brands?|buyers?)\b", "", text)
    return re.sub(r"\s+", " ", text).strip()


def promote_live_brand_filters(filters: dict | None, brand_names: list[str] | set[str] | None) -> dict:
    """If garment/category equals a live Brand name, promote to brand (schema-grounded)."""

    out = {k: v for k, v in (filters or {}).items() if v not in (None, "")}
    names = {normalize_name(str(n)) for n in (brand_names or []) if n}
    if not names:
        return out
    for key in ("garment", "garment_type", "category", "family", "subfamily"):
        value = out.get(key)
        if not value:
            continue
        want = normalize_name(str(value))
        if _is_grammar_party_name(want):
            continue
        # Exact brand match only — never fuzzy-promote sweater→brand.
        if want in names:
            out.pop(key, None)
            out.setdefault("brand", str(value))
            out.setdefault("buyer", str(value))
    brand = normalize_name(str(out.get("brand") or "").split("|")[0].strip())
    if brand:
        for key in ("garment", "garment_type", "category"):
            if out.get(key) and normalize_name(str(out.get(key))) == brand:
                out.pop(key, None)
    return out


def live_brand_names(platform, graph_id: str | None = None, principal=None) -> list[str]:
    """Brand names from the live product graph (cached) + schema canonical hints."""

    names: list[str] = []
    try:
        from fashion_brain.planning.runtime_schema import get_runtime_schema, resolve_product_graph_id
        from fashion_brain.planning.schema_cache import get_brands

        gid = graph_id or resolve_product_graph_id(platform)
        if gid:
            for row in get_brands(platform, gid, principal) or []:
                name = str((row or {}).get("name") or "").strip()
                if name:
                    names.append(name)
        schema = get_runtime_schema(platform, gid)
        for item in getattr(schema, "canonical_hints", None) or []:
            if str(item.get("type") or "").casefold() != "brand":
                continue
            name = str(item.get("name") or "").strip()
            if name:
                names.append(name)
            for alias in item.get("aliases") or []:
                if alias:
                    names.append(str(alias))
    except Exception:
        pass
    # De-dupe preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        key = normalize_name(name)
        if key and key not in seen:
            seen.add(key)
            out.append(name)
    return out


def _is_grammar_party_name(name: str) -> bool:
    """Reject catalog values that collide with question grammar (e.g. buyer 'frequently')."""

    key = normalize_name(str(name or ""))
    if not key:
        return True
    tokens = set(key.split())
    if not tokens:
        return True
    grammar = (
        _COUNT_STOPWORDS
        | _GRAPH_SCORE_STOPWORDS
        | _SYSTEM_WORDS
        | _GRAPH_SCOPE_NOISE
        | _MAKE_ADVERBS
        | _MAKE_VERBS
        | _TASK_WORDS
        | _METRIC_OPERATOR_WORDS
        | _ANALYSIS_WORDS
        | _RANK_WORDS
        | _MEASURE_PROPERTY_WORDS
    )
    # Whole value is stopword grammar — never a real brand/buyer filter from the question.
    if tokens <= grammar:
        return True
    if len(tokens) == 1 and next(iter(tokens)) in grammar:
        return True
    return False


def ground_filters_from_live_brands(
    question: str,
    filters: dict | None,
    brand_names: list[str] | None,
) -> dict:
    """Link question tokens to live Brand nodes — not a hardcoded retailer list.

    Used when the keyword planner leaves party names unassigned so EntityExtract /
    Intention can ground against the actual catalog.
    """

    out = promote_live_brand_filters(filters, brand_names)
    if out.get("brand") or out.get("buyer"):
        return out
    text = normalize_name(question or "")
    if not text or not brand_names:
        return out
    # Longest brand name first so "calvin klein" wins over a bare token clash.
    ranked = sorted(
        {normalize_name(str(n)): str(n) for n in brand_names if n}.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    hits: list[str] = []
    for key, surface in ranked:
        if len(key) < 2 or _is_grammar_party_name(key):
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", text):
            hits.append(surface)
    if len(hits) == 1:
        out["brand"] = hits[0]
        out["buyer"] = hits[0]
    elif len(hits) >= 2:
        out["brand"] = " | ".join(hits[:4])
    return out


def sanitize_party_filters(filters: dict) -> dict:
    """Brand/buyer names must not also appear as garment/category filters."""

    cleaned = dict(filters or {})
    for key in ("brand", "buyer"):
        value = cleaned.get(key)
        if not value:
            continue
        parts = split_multi_party_filter(str(value)) or [str(value)]
        kept = [part for part in parts if part and not _is_grammar_party_name(part)]
        if not kept:
            cleaned.pop(key, None)
        elif len(kept) == 1:
            cleaned[key] = kept[0]
        else:
            cleaned[key] = " | ".join(kept)
    parties: list[str] = []
    for key in ("brand", "buyer"):
        value = cleaned.get(key)
        if not value:
            continue
        parties.append(normalize_name(str(value)))
        for part in split_multi_party_filter(str(value)):
            parties.append(normalize_name(part))
    party_set = {p for p in parties if p}
    graph_noise = set(_GRAPH_SCOPE_NOISE) | {
        "stored",
        "inside",
        "within",
        "source",
        "sources",
        "e",
        "comm",
        "com",
    }
    for key in ("garment", "garment_type", "category"):
        value = cleaned.get(key)
        if not value:
            continue
        raw = normalize_name(str(value))
        core = _party_filter_core(raw)
        tokens = set(raw.split())
        if raw in party_set or core in party_set or (tokens & party_set):
            cleaned.pop(key, None)
            continue
        # Drop values that are only graph/source phrasing ("stored inside commerces").
        content = {tok for tok in tokens if tok not in graph_noise and tok not in _COUNT_STOPWORDS}
        if not content:
            cleaned.pop(key, None)
    return cleaned


def _drop_group_by_as_filter(filters: dict, group_by: str | None, question: str) -> dict:
    """Do not turn 'material wise' into material=<invented value>."""

    cleaned = dict(filters or {})
    if not group_by or group_by not in cleaned:
        return cleaned
    raw = str(cleaned.get(group_by) or "")
    value = normalize_name(raw)
    tokens = set(_question_tokens(question))
    # Keep intentional multi-party compares ("Primark | Zara") when each name appears.
    if "|" in raw:
        parts = [normalize_name(p) for p in raw.split("|") if p.strip()]
        if parts and all(part in tokens for part in parts):
            return cleaned
    # Keep only when the user clearly named that value apart from the field/wise words.
    if not value or value == normalize_name(group_by) or value not in tokens:
        cleaned.pop(group_by, None)
    return cleaned


def _analysis_party_name(tokens: list[str]) -> str | None:
    """Party name in an analysis question: 'PRIMARK buyer', 'buyer PRIMARK', or 'analysis primark'."""

    # "all buyers" / "all brands" is a breakdown, not a named party.
    if _all_parties_dimension(tokens):
        return None
    for index, token in enumerate(tokens):
        if token not in {"buyer", "buyers", "brand", "brands"}:
            continue
        before = tokens[index - 1] if index > 0 else ""
        after = tokens[index + 1] if index + 1 < len(tokens) else ""
        if before == "all":
            return None
        if before and _is_party_token(before) and before not in _GRAPH_SCOPE_NOISE:
            return before
        if after and _is_party_token(after) and after not in _GRAPH_SCOPE_NOISE:
            return after
    # Sole party token is the analysis subject.
    parties = [
        token
        for token in tokens
        if _is_party_token(token) and token not in _GRAPH_SCOPE_NOISE and token not in _SYSTEM_WORDS
    ]
    if len(parties) == 1:
        return parties[0]
    return None


def _analysis_buyer_name(tokens: list[str]) -> str | None:
    """Back-compat alias for older call sites."""

    return _analysis_party_name(tokens)


# --------------------------------------------------------------------------------------
# Model planner (primary) with code validation
# --------------------------------------------------------------------------------------


def plan_catalog_question(
    platform: Platform,
    question: str,
    history: list | None = None,
    *,
    carry: bool | None = None,
) -> dict:
    """The model decides, code validates. Falls back to the keyword planner.

    ``carry`` is the agent's references_prior_chat decision. When omitted, the planner
    model decides it; keyword fallback never invents history carry.

    Always grounds filters against each registered KG's unique columns and live values.
    """

    from fashion_brain.planning.kg_catalog import build_field_catalog, ground_filters_with_catalog

    catalog = build_field_catalog(platform, question)
    model_plan = _plan_with_model(platform, question, history, catalog=catalog)
    if model_plan is None:
        plan = resolve_catalog_plan(question, history, carry=bool(carry))
        plan["source"] = "keywords"
        plan["references_prior_chat"] = bool(carry)
        plan["filters"] = ground_filters_with_catalog(
            sanitize_party_filters(dict(plan.get("filters") or {})),
            catalog,
            question=question,
        )
        plan["preferred_graphs"] = (
            list(plan.get("preferred_graphs") or [])
            or preferred_graphs_from_question(question, platform)
            or catalog.preferred_graphs()
        )
        plan["available_attributes"] = catalog.all_attributes()
        return plan
    if carry is None:
        carry = bool(model_plan.get("references_prior_chat"))
    else:
        carry = bool(carry)
    model_plan["references_prior_chat"] = carry
    if model_plan["unsupported"]:
        return {
            **model_plan,
            "filters": {},
            "group_by": None,
            "source": "model",
            "preferred_graphs": catalog.preferred_graphs(),
            "available_attributes": catalog.all_attributes(),
        }
    texts = [*_user_questions(history), question] if carry else [question]
    model_plan["filters"] = _ground_filters(model_plan["filters"], texts)
    if not carry:
        model_plan["filters"] = _ground_filters(model_plan["filters"], [question])
    _promote_type_filters(model_plan["filters"])
    model_plan["filters"] = sanitize_party_filters(model_plan["filters"])
    _normalize_named_season(model_plan)
    keyword = resolve_catalog_plan(question, history, carry=carry)
    if not model_plan["filters"] and keyword.get("filters"):
        model_plan["filters"] = dict(keyword["filters"])
    # Keyword party cues beat a model that stuffed the same name under garment/category.
    kw_filters = dict(keyword.get("filters") or {})
    if kw_filters.get("brand") or kw_filters.get("buyer"):
        for key in ("brand", "buyer"):
            if kw_filters.get(key):
                model_plan["filters"][key] = kw_filters[key]
        model_plan["filters"] = sanitize_party_filters(model_plan["filters"])
    if _is_analysis(question):
        model_plan["suggest"] = False
        field_wise = bool(keyword.get("group_by")) or bool(model_plan.get("group_by"))
        if field_wise:
            # "analysis … material wise" → one-column breakdown, not a full multi-field profile.
            model_plan["analysis"] = False
            model_plan["group_by"] = keyword.get("group_by") or model_plan.get("group_by")
        else:
            model_plan["analysis"] = True
            model_plan["group_by"] = None
            if model_plan.get("limit") is None:
                model_plan["limit"] = 5
            if keyword.get("filters", {}).get("buyer") and not model_plan["filters"].get("buyer"):
                model_plan["filters"]["buyer"] = keyword["filters"]["buyer"]
            if keyword.get("filters", {}).get("brand") and not model_plan["filters"].get("brand"):
                model_plan["filters"]["brand"] = keyword["filters"]["brand"]
    model_plan["filters"] = ground_filters_with_catalog(model_plan["filters"], catalog, question=question)
    model_plan["filters"] = _drop_group_by_as_filter(model_plan.get("filters") or {}, model_plan.get("group_by"), question)
    preferred = (
        list(model_plan.get("preferred_graphs") or [])
        or preferred_graphs_from_question(question, platform)
        or catalog.preferred_graphs()
    )
    model_plan["preferred_graphs"] = preferred
    model_plan["available_attributes"] = catalog.all_attributes()
    filters, group_by = _remap_buyer_to_brand_if_needed(
        dict(model_plan.get("filters") or {}),
        model_plan.get("group_by"),
        preferred,
        platform,
    )
    if re.search(r"\ball\s+(buyers?|brands?)\b", question.casefold()):
        filters.pop("brand", None)
        filters.pop("buyer", None)
        group_by = group_by or "brand"
    model_plan["filters"] = filters
    model_plan["group_by"] = group_by
    model_plan["source"] = "model"
    return model_plan


def _plan_timeout(seconds: float | None) -> float | None:
    if seconds is None or seconds <= 0:
        return 10.0
    return min(10.0, seconds)


def _plan_with_model(
    platform: Platform,
    question: str,
    history: list | None = None,
    *,
    catalog=None,
) -> dict | None:
    provider = build_model_provider(platform.settings)
    earlier = _user_questions(history)
    prompt = question
    if earlier:
        lines = "\n".join(f"- {item}" for item in earlier)
        prompt = f"Earlier user questions:\n{lines}\n\nLatest question:\n{question}"
    system = _PLAN_SYSTEM
    if catalog is not None:
        from fashion_brain.planning.kg_catalog import planner_system_prompt

        system = planner_system_prompt(catalog)
    try:
        raw = provider.complete(
            prompt,
            timeout=_plan_timeout(platform.settings.request_timeout_seconds),
            system=system,
        )
    except Exception:
        LOGGER.warning("model planner failed; using keyword planner", exc_info=True)
        return None
    plan = _parse_model_plan(raw)
    if plan is None:
        LOGGER.warning("model planner returned no usable JSON; using keyword planner")
    return plan


def _extract_json_object(text: str) -> dict | None:
    """First JSON object in the text. Tolerates code fences and trailing prose or braces."""

    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            value, _end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(value, dict):
            return value
        start = text.find("{", start + 1)
    return None


def _field_key(raw: Any) -> str | None:
    key = str(raw).strip().lower()
    if key in _PLAN_GROUPS or key == "garment":
        return key
    return _FIELD_ALIASES.get(key)


def _parse_model_plan(text: str) -> dict | None:
    data = _extract_json_object(text or "")
    if data is None:
        return None

    raw_filters = data.get("filters") if isinstance(data.get("filters"), dict) else {}
    filters: dict[str, str] = {}
    for raw_key, value in raw_filters.items():
        key = _field_key(raw_key)
        if key and isinstance(value, str) and value.strip() and _reliable_filter(value):
            filters[key] = " ".join(value.split())

    group_by = data.get("group_by")
    if isinstance(group_by, str):
        group_by = _FIELD_ALIASES.get(group_by.strip().lower(), group_by.strip().lower())
    if group_by == "garment":
        group_by = "garment_type"
    if group_by not in _PLAN_GROUPS:
        group_by = None

    limit = data.get("limit")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _PLAN_LIMIT_MAX:
        limit = None

    season_name = _SEASON_WORDS.get(str(data.get("season_name") or "").strip().lower())
    intent = data.get("intent")
    analysis = intent == "analysis" or data.get("analysis") is True
    suggest = (intent == "specification" or data.get("suggest") is True) and not analysis
    preferred_graphs: list[str] = []
    raw_preferred = data.get("preferred_graphs")
    if isinstance(raw_preferred, list):
        preferred_graphs = [str(item).strip() for item in raw_preferred if str(item).strip()]
    elif isinstance(raw_preferred, str) and raw_preferred.strip():
        preferred_graphs = [raw_preferred.strip()]
    # Keep group_by when the model asked for a single-field wise/breakdown analysis.
    if analysis and group_by:
        analysis = False
    return {
        "filters": filters,
        "group_by": group_by,
        "unsupported": intent == "unsupported" and not filters and group_by is None and not analysis,
        "limit": limit,
        "suggest": suggest,
        "analysis": analysis,
        "season_name": season_name,
        "upcoming": data.get("upcoming") is True,
        "references_prior_chat": data.get("references_prior_chat") is True,
        "preferred_graphs": preferred_graphs,
    }


def _token_in_words(token: str, words: frozenset[str]) -> bool:
    if token in words:
        return True
    if len(token) < 4 or token.isdigit():
        return False
    allowed = 2 if len(token) >= 6 else 1
    return any(_edit_distance(token, word) <= allowed for word in words)


def _ground_filters(filters: dict[str, str], texts: list[str]) -> dict[str, str]:
    """Keep only filter values the user actually said (typos allowed). Stops invented filters."""

    words = frozenset(normalize_name(" ".join(texts)).split())
    gender_values = set(_GENDER_VALUES.values())
    kept: dict[str, str] = {}
    for key, value in filters.items():
        tokens = normalize_name(value).split()
        # "in plm" / "in plmex" names the catalog system, not a garment or material.
        if set(tokens) & _SYSTEM_WORDS:
            LOGGER.info("dropped system-word planner filter %s=%s", key, value)
            continue
        # "buyer=frequently" / "brand=often" — adverb/task words that appear in the question
        # are still not party names (grounding alone is not enough).
        if key in {"brand", "buyer"} and _is_grammar_party_name(value):
            LOGGER.info("dropped grammar party filter %s=%s", key, value)
            continue
        if key == "gender" and value in gender_values:
            kept[key] = value
        elif tokens and all(_token_in_words(token, words) for token in tokens):
            kept[key] = value
        else:
            LOGGER.info("dropped ungrounded planner filter %s=%s", key, value)
    return kept


# --------------------------------------------------------------------------------------
# Buyer spelling
# --------------------------------------------------------------------------------------

_NAME_CACHE: dict[tuple[int, int, str], list[str]] = {}
_NAME_CACHE_MAX = 32


def _edit_distance(left: str, right: str) -> int:
    if abs(len(left) - len(right)) > 2:
        return 3
    previous = list(range(len(right) + 1))
    for index, char in enumerate(left, start=1):
        current = [index]
        for other_index, other in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[other_index] + 1, previous[other_index - 1] + (char != other)))
        previous = current
    return previous[-1]


def _catalog_names(platform, field: str) -> list[str]:
    """Distinct catalog values of one field. Cached per envelope list and size."""

    products = getattr(platform, "products", None)
    envelopes = getattr(products, "envelopes", None) or []
    key = (id(envelopes), len(envelopes), field)
    cached = _NAME_CACHE.get(key)
    if cached is not None:
        return cached
    found: dict[str, str] = {}
    for envelope in envelopes:
        metadata = getattr(envelope, "metadata", {}) or {}
        product = metadata.get("product", {}) if isinstance(metadata, dict) else {}
        raw = field_text(product, field).strip()
        if raw:
            found.setdefault(normalize_name(raw), raw)
    names = list(found.values())
    if len(_NAME_CACHE) >= _NAME_CACHE_MAX:
        _NAME_CACHE.clear()
    _NAME_CACHE[key] = names
    return names


def _nearest_name(typed: str, names: list[str]) -> str | None:
    """Closest catalog name within 1-2 edits. A tie between two names returns None."""

    target = normalize_name(typed)
    if len(target) < 4 or not names:
        return None
    best_name: str | None = None
    best_distance = 99
    second = 99
    for name in names:
        folded = normalize_name(name)
        if folded == target:
            return name
        distance = _edit_distance(target, folded)
        if distance < best_distance:
            second, best_distance, best_name = best_distance, distance, name
        elif distance < second:
            second = distance
    limit = 2 if len(target) >= 6 else 1
    if best_name is None or best_distance > limit or second == best_distance:
        return None
    return best_name


def apply_buyer_spelling(platform, holder) -> str:
    """Replace near-miss buyer/brand spelling with catalog names. Return the note."""

    filters = getattr(holder, "filters", None)
    if not isinstance(filters, dict) or platform is None:
        return ""
    notes: list[str] = []
    for key, label in (("buyer", "buyer"), ("brand", "brand")):
        if not filters.get(key):
            continue
        typed = str(filters[key])
        typed_key = normalize_name(typed)
        # Never fuzzy-match graph/source words ("ecomm", "comm", "plm") onto parties.
        if (
            typed_key in _SYSTEM_WORDS
            or typed_key in _GRAPH_SCOPE_NOISE
            or typed_key in {"comm", "com", "data", "all"}
        ):
            filters.pop(key, None)
            continue
        names = _catalog_names(platform, key)
        if not names:
            # Prefer live metadata values when the local catalog index is empty (ecom brands).
            try:
                from knowledge_graph.retrieval.metadata import discover_values

                payload = discover_values(platform, attribute=key, q=typed, limit=50)
                names = [str(item) for item in (payload.get("values") or [])]
            except Exception:
                names = []
        if not names:
            continue
        exact = next((name for name in names if normalize_name(name) == normalize_name(typed)), None)
        if exact:
            filters[key] = exact
            continue
        # Short tokens are too dangerous to fuzzy-match (comm → primark).
        if len(typed_key) < 5:
            filters.pop(key, None)
            continue
        nearest = _nearest_name(typed, names)
        if nearest is None:
            continue
        filters[key] = nearest
        note = f"{typed} is not a {label} name. Using the closest product {label}, {nearest}."
        notes.append(note)
        corrections = getattr(holder, "corrections", None)
        if isinstance(corrections, list):
            corrections.append(note)
    return notes[0] if notes else ""


# --------------------------------------------------------------------------------------
# Catalog answer (template mode)
# --------------------------------------------------------------------------------------


def _drop_unreliable_garment(filters: dict) -> None:
    garment = filters.get("garment")
    if garment and not _reliable_filter(str(garment)):
        filters.pop("garment", None)


def _promote_type_filters(filters: dict) -> None:
    """Move category/garment_type filters onto garment so Sweater and Jumper both match."""

    promoted: list[str] = []
    for key in _TYPE_FILTER_KEYS:
        value = filters.pop(key, None)
        if value:
            promoted.append(str(value))
    if not promoted:
        return
    if filters.get("garment"):
        return
    seen: set[str] = set()
    parts: list[str] = []
    for value in promoted:
        folded = value.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        parts.append(value)
    filters["garment"] = parts[0] if len(parts) == 1 else " ".join(parts)


def _normalize_named_season(plan: dict) -> None:
    """Winter is an autumn/winter style code, not a value in the season-year column."""

    filters = plan.setdefault("filters", {})
    mapped = _SEASON_WORDS.get(normalize_name(str(filters.get("season") or "")))
    if mapped:
        filters.pop("season", None)
        plan["season_name"] = plan.get("season_name") or mapped
    _promote_type_filters(filters)
    _drop_unreliable_garment(filters)


def _query_filters(plan: dict) -> dict:
    filters = {key: value for key, value in plan["filters"].items() if value}
    mapped = _SEASON_WORDS.get(normalize_name(str(filters.get("season") or "")))
    if mapped:
        filters.pop("season", None)
        filters["season_name"] = plan.get("season_name") or mapped
    elif plan.get("season_name"):
        filters["season_name"] = plan["season_name"]
    if plan.get("upcoming"):
        filters["upcoming"] = "yes"
    _promote_type_filters(filters)
    _drop_unreliable_garment(filters)
    return filters


def _append_catalog_answer(platform, principal, plan, lines, evidence, caveats, unavailable) -> None:
    if plan.get("unsupported"):
        lines.append(_outside_catalog_message())
        return
    _normalize_named_season(plan)
    if plan.get("suggest"):
        _append_specification(platform, principal, plan, lines, evidence, caveats, unavailable)
        return
    visible = platform.products.count(principal)
    hidden = len(platform.products.envelopes) - visible
    filters = _query_filters(plan)
    group_by = plan.get("group_by")
    if not visible:
        lines.append("The PLM data contains 0 products.")
    elif filters or group_by:
        matched, total = platform.products.search(principal, limit=5, **filters)
        phrase = _filter_phrase(filters)
        lines.append(f"The PLM data contains {total} products" + (f" {phrase}." if phrase else "."))
        if group_by:
            groups = platform.products.count_by(principal, group_by, **filters)
            label = _GROUP_LABELS.get(group_by, group_by)
            if plan.get("limit"):
                lines.extend(_ranking_lines(groups, label, int(plan["limit"])))
            else:
                lines.append(f"{label} counts for that selection: {_group_text(groups)}.")
        evidence.extend(_evidence_from_product(envelope) for envelope in matched)
    else:
        lines.append(f"The PLM data contains {visible} products.")
    if CATALOG_CAVEAT:
        caveats.append(CATALOG_CAVEAT)
    if hidden:
        unavailable.append(f"{hidden} products are hidden by access control.")


def _profile_matches(platform, principal, filters: dict) -> tuple[int, dict, str | None]:
    products = platform.products
    code = products.prepare_filters(principal, filters).get("season_code") if hasattr(products, "prepare_filters") else None
    total, ranked = products.profile(principal, _SPEC_FIELDS, **filters)
    return total, ranked, code


def _usable_spec_value(value: str) -> bool:
    folded = value.strip().casefold()
    if folded in _UNNAMED_GROUPS:
        return False
    return re.fullmatch(r"0+(?:\.0+)?", folded.replace(" ", "")) is None


def _append_specification(platform, principal, plan, lines, evidence, caveats, unavailable) -> None:
    """Buyer first, then the named season, then the most common value of every specification field."""

    filters = _query_filters(plan)
    lines.extend(plan.get("corrections") or [])
    buyer = filters.get("buyer") or "PLM data"
    buyer_only = {key: value for key, value in filters.items() if key not in {"season_name", "season_code", "upcoming"}}
    buyer_total = platform.products.count(principal, **buyer_only)
    lines.append(f"Step 1. Buyer {buyer}: {buyer_total} products.")
    # Upcoming without winter/summer → resolve the latest AW/SS code for this buyer.
    if filters.get("upcoming") and not filters.get("season_name") and hasattr(platform.products, "prepare_filters"):
        resolved = platform.products.prepare_filters(principal, {**buyer_only, "upcoming": "yes"})
        if resolved.get("season_name"):
            filters["season_name"] = resolved["season_name"]
            plan["season_name"] = resolved["season_name"]
    filters.pop("upcoming", None)
    filters.pop("season_code", None)
    season_name = filters.get("season_name")
    if not season_name:
        total, ranked, _code = _profile_matches(platform, principal, buyer_only)
        scope = f"all {buyer_total} products for that buyer"
    else:
        season_label = str(season_name).capitalize()
        total, ranked, _code = _profile_matches(platform, principal, filters)
        codes = []
        if hasattr(platform.products, "season_code_counts"):
            codes = platform.products.season_code_counts(principal, season_name, **buyer_only)
        code_text = ", ".join(f"{code} {count}" for code, count in codes[:6])
        lines.append(
            f"Step 2. {season_label} styles for that buyer, using season codes in the style reference: {total} products."
            + (f" Season codes: {code_text}." if code_text else "")
        )
        scope = f"those {total} {season_name} products"
        if total == 0:
            total, ranked, _code = _profile_matches(platform, principal, buyer_only)
            lines.append(f"No {season_name} style codes were stored, so the specification uses all {total} products for that buyer.")
            scope = f"all {total} products for that buyer"
    if not total:
        lines.append(f"The PLM data has no styles for {buyer}, so there is no specification to build.")
        if CATALOG_CAVEAT:
            caveats.append(CATALOG_CAVEAT)
        return
    lines.append(f"Step 3. On {scope}, the most common value of each specification field is the suggested value.")
    lines.append("Product specification:")
    lines.append(f"Buyer: {buyer}")
    if season_name:
        lines.append(f"Season: {season_name}")
    for field in _SPEC_FIELDS:
        choices = [item for item in (ranked.get(field) or []) if _usable_spec_value(item[0])]
        if not choices:
            continue
        value, count = choices[0]
        label = _GROUP_LABELS.get(field, field.replace("_", " "))
        lines.append(f"{label}: {value} ({count} of {total})")
    lines.append("Each count is how many of the filtered products use that value. The sheet is drawn from those products only.")
    if CATALOG_CAVEAT:
        caveats.append(CATALOG_CAVEAT)


def _ranking_lines(groups: list[tuple[str, int]], label: str, limit: int) -> list[str]:
    """Named leaders first. Blank buyer names stay out of the ranking."""

    named = [(name, count) for name, count in groups if name.strip().casefold() not in _UNNAMED_GROUPS]
    missing = [(name, count) for name, count in groups if name.strip().casefold() in _UNNAMED_GROUPS]
    named.sort(key=lambda item: (-item[1], item[0].casefold()))
    shown = named[: max(limit, 1)]
    lines = [f"Top {len(shown)} {label.casefold()} by product count:"]
    lines.extend(f"{rank}. {name}: {count}" for rank, (name, count) in enumerate(shown, start=1))
    if missing:
        hidden = ", ".join(f"{name or 'blank'} {count}" for name, count in missing)
        lines.append(f"Records with no {label.casefold()} name are kept separate: {hidden}.")
    return lines


def _group_text(groups: list[tuple[str, int]]) -> str:
    if not groups:
        return "no visible products"
    ordered = sorted(groups, key=lambda item: (-item[1], item[0].casefold()))
    shown = ordered[:30]
    text = ", ".join(f"{label} {count}" for label, count in shown)
    rest = sum(count for _label, count in ordered[30:])
    if rest:
        text = f"{text}, other values {rest}"
    return text


def _outside_catalog_message() -> str:
    names = ", ".join(field.label.casefold() for field in CATALOG_FIELDS)
    return "No count was estimated. Ask about product counts or a breakdown by any stored field: " + names + "."


def _filter_phrase(filters: dict[str, str]) -> str:
    parts = [f"{_FILTER_LABELS[key]} {value}" for key, value in filters.items() if value and key in _FILTER_LABELS]
    if not parts:
        return ""
    return "with " + " and ".join(parts)


# --------------------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------------------


def _evidence_from_product(envelope) -> EvidenceItem:
    product = envelope.metadata.get("product", {})
    excerpt = (
        f"color={product.get('color', '')}; "
        f"category={product.get('category', '')}; "
        f"garment_type={product.get('garment_type', '')}; "
        f"gender={product.get('gender', '')}"
    )
    return EvidenceItem(
        source_id=envelope.source_id,
        record_id=envelope.record_id,
        title=envelope.title,
        excerpt=excerpt,
        field_ref="color",
        source_path=envelope.source_path,
        content_hash=envelope.content_hash,
        verification_status="observed",
        synthetic=envelope.synthetic,
    )


def _evidence_from_edge(edge, node, relationship: str) -> EvidenceItem:
    props = edge.properties
    return EvidenceItem(
        source_id=str(props.get("prov_source_id") or ""),
        record_id=str(props.get("prov_record_id") or ""),
        title=node.properties.get("title") or node.properties.get("name"),
        excerpt=str(props.get("prov_evidence_text") or ""),
        field_ref=props.get("prov_field_ref"),
        section_ref=props.get("prov_section_ref"),
        source_path=props.get("prov_source_path"),
        url=props.get("prov_url"),
        content_hash=props.get("prov_content_hash"),
        verification_status=str(props.get("prov_verification_status") or "observed"),
        graph=edge.graph,
        node_id=node.id,
        relationship=relationship,
        synthetic=bool(props.get("prov_synthetic", True)),
    )


def _dedupe_evidence(items: list[EvidenceItem]) -> list[EvidenceItem]:
    seen = set()
    unique = []
    for item in items:
        key = (item.source_id, item.record_id, item.relationship, item.field_ref, item.section_ref, item.excerpt)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique
