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
    with and or for from that this be any all every each number count total visible clearance by gender wise per
    breakdown analysis analyze grouped distribution split brands brand buyer buyers season seasons category
    categories material materials department departments designer designers year now can you your make me
    please could would them those these their same also just show give want let us again still instead about
    what which who when where list has have had contain contains containing made using use uses top bottom
    highest lowest leading rank ranking first last garments type types kind kinds normally usually typical
    typically often common commonly mostly making makes does did produce produces producing manufacture
    manufactures if then upcomming upcoming coming coming seasion sesion suggest suggestion suggestions
    specification specifications spec specs recommend recommendation recommendations sample samples new
    used shared share same interconnect interconnection found matched match result results currently available
    across among amongst only also present available registered registry metadata attribute attributes
    unique values description descriptions knowledge graph graphs kg kgs source sources system basic info
    information overview
    """
)
_RANK_WORDS = _words("top bottom highest lowest leading most")
_MAKE_VERBS = _words("make makes making produce produces producing manufacture manufactures manufacturing")
_MAKE_ADVERBS = _words("normally usually typically often commonly mostly")
# Catalog / graph / system words: never a garment, material, or buyer/brand filter value.
_SYSTEM_WORDS = _words(
    "plm plmex plm-ex system database catalog catalogue ecom ecomm ecommerce commerce "
    "neo4j postgres sql kg graph graphs source sources"
)
# Question phrases that name a product KG (not a filter value).
_GRAPH_SOURCE_ALIASES = {
    "ecom": "ecom_products",
    "ecomm": "ecom_products",
    "ecommerce": "ecom_products",
    "commerce": "ecom_products",
    "plm": "own_products",
    "plmex": "own_products",
    "own": "own_products",
}
# A filter value containing any of these is the question pasted back, not a catalog value.
_FILTER_NOISE = _words(
    "normally usually typical typically making make makes made type types kind kinds which what "
    "product products does new sample samples suggest specification specifications "
    "found matched match result results currently available total number count"
)
_SUGGEST_WORDS = _words(
    "suggest suggestion suggestions specification specifications spec specs recommend recommendation recommendations"
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
    "For garment or product-type words such as sweater, jumper, dress, or cardigan, put the value in "
    '"garment" (searches category and garment type together). Do not put those words in garment_type alone. '
    "limit is an integer only for top-N or bottom-N questions, else null. "
    'season_name is "winter", "spring", "summer" or null; upcoming is true only when the user says upcoming or next season. '
    "Copy filter values only from the user's words. Use intent unsupported when the question is not about our products. "
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


def plan_routes(question: str) -> list[str]:
    text = question.lower()
    tokens = set(normalize_name(question).split())
    if "live" in tokens and "plmex" in text:
        return ["live_sql"]
    if _is_analysis(question):
        return ["product_count"]
    if _is_suggestion(question):
        return ["specification"]
    if re.search(r"\b(top|bottom|highest|lowest|most)\s+\d+\b", normalize_name(question)):
        return ["product_count"]
    if _buyer_ranking_question(question):
        return ["product_count"]
    if ("how many" in text and "product" in text) or _group_field(question) or _product_type_breakdown(question):
        return ["product_count"]
    if news_question(question):
        return ["news"]
    if any(word in text for word in ("share", "shared", "same material", "interconnect")):
        # Shared *products* / materials — system shared overview is selected via kg_system grounding.
        return ["shared_products"]
    if tokens & {"product", "products", "material", "materials", "fabric"}:
        return ["products"]
    if _asked_dimension(question):
        return ["product_count"]
    return ["shared"]


def _is_analysis(question: str) -> bool:
    """Buyer / product analysis: type-wise tops, not a suggested new spec sheet."""

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
) -> GroundedAnswer:
    """Route an LLM ask through the agent graph. Template mode stays on the catalog."""

    from fashion_brain.trace import current_trace, end_trace, start_trace

    owns_trace = current_trace() is None
    if owns_trace:
        start_trace(question, source="http/ask")
    try:
        if _use_agent_graph(platform):
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
    if settings.model_provider in {"", "none", "null"}:
        return False
    from fashion_brain.llm import resolve_model_api_key

    return bool(resolve_model_api_key(settings))


def answer_from_catalog(
    platform: Platform,
    question: str,
    principal: Principal,
    history: list | None = None,
    graph_id: str | None = None,
) -> GroundedAnswer:
    """Intention agent, graph router, knowledge-graph fetch, response agent."""

    from fashion_brain.agents import ResponseAgent

    retrieved, routes, _intention = draft_catalog(platform, question, principal, history, graph_id=graph_id)
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


def _retrieve(
    platform: Platform,
    principal: Principal,
    intention,
    question: str = "",
    graph_id: str | None = None,
    graph_ids: list[str] | None = None,
):
    from fashion_brain.agents import Retrieved
    from fashion_brain.kg_client import retrieve_remote, uses_kg_api
    from fashion_brain.trace import trace_debug, trace_step
    from knowledge_graph.retrieval.brain import BrainRetrieveRequest, retrieve as local_retrieve

    request = BrainRetrieveRequest(
        question=question,
        intent=intention.intent,
        filters=dict(intention.filters or {}),
        group_by=intention.group_by,
        unsupported=bool(intention.unsupported),
        limit=intention.limit,
        suggest=(bool(intention.suggest) or intention.intent == "specification") and not bool(getattr(intention, "analysis", False)),
        analysis=bool(getattr(intention, "analysis", False)),
        season_name=intention.season_name,
        upcoming=bool(intention.upcoming),
        corrections=list(intention.corrections or []),
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
    if remote:
        payload = retrieve_remote(platform.settings, principal.clearance, request)
    else:
        payload = local_retrieve(platform, principal, request)
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
    return any(phrase in text for phrase in ("new product", "new sample", "new style", "product specification", "sample for"))


def _is_party_token(token: str) -> bool:
    """A buyer name, not a material, color, field, or question word."""

    if (
        token in _MAKE_VERBS
        or token in _COUNT_STOPWORDS
        or token in _SYSTEM_WORDS
        or token in _ANALYSIS_WORDS
        or token in _HISTORY_WORDS
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
    name: list[str] = []
    while index < len(tokens) and _is_party_token(tokens[index]) and len(name) < 3:
        name.append(tokens[index])
        index += 1
    return name


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
    return not (set(tokens) & _FILTER_NOISE)


def _asked_dimension(question: str) -> str | None:
    tokens = normalize_name(question).split()
    if not tokens or tokens[0] not in {"what", "which", "list", "show"}:
        return None
    found = _alias_at(tokens, 1)
    return found[0] if found else None


def _is_filter_term(token: str) -> bool:
    return (
        token not in _COUNT_STOPWORDS
        and token not in _COLOR_MARKERS
        and token not in _SYSTEM_WORDS
        and token not in _ANALYSIS_WORDS
        and token not in _HISTORY_WORDS
        and token not in _RANK_WORDS
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
        parts.append("Type-wise top analysis of past products.")
    if getattr(intention, "suggest", False):
        parts.append("Suggesting specifications from the styles this buyer already has.")
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
    """Normalize and glue graph phrases like 'e comm' / 'e-comm' → ecomm."""

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
        if token == "e" and nxt == "commerce":
            glued.append("ecommerce")
            index += 2
            continue
        glued.append(token)
        index += 1
    return glued


def preferred_graphs_from_question(question: str) -> list[str]:
    """Map 'in ecomm' / 'plm data' phrasing to registry graph ids."""

    tokens = set(_question_tokens(question))
    preferred: list[str] = []
    for token, graph_id in _GRAPH_SOURCE_ALIASES.items():
        if token in tokens and graph_id not in preferred:
            preferred.append(graph_id)
    return preferred


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
    index = 0
    while index < len(tokens):
        token = tokens[index]

        if token in _ANALYSIS_WORDS or token in _HISTORY_WORDS:
            index += 1
            continue

        # "top 5 [field]"
        if token in _RANK_WORDS and index + 1 < len(tokens) and tokens[index + 1].isdigit():
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

        # "<field> value [value ...]"
        if found and index + found[1] < len(tokens) and _is_value_token(tokens[index + found[1]]):
            values: list[str] = []
            index += found[1]
            while index < len(tokens) and _is_value_token(tokens[index]) and len(values) < 3:
                nxt = tokens[index + 1] if index + 1 < len(tokens) else None
                if nxt in _COLOR_MARKERS:
                    break
                if tokens[index] in _FIELD_ALIASES and nxt in _FIELD_ALIASES:
                    break
                values.append(tokens[index])
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

        # "for <brand|buyer>" — e-comm uses brand; PLM used buyer. Set both; KG maps buyer→brand.
        if token == "for":
            name = _take_party_name(tokens, index + 1)
            if name:
                filters.setdefault("brand", []).extend(name)
                filters.setdefault("buyer", []).extend(name)
                index += 1 + len(name)
                continue

        if not _is_filter_term(token):
            index += 1
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
        else:
            # Unknown content words default to garment/category search.
            # Brand/buyer come from "for X", make-verb patterns, analysis party,
            # or live metadata value grounding — not from guessing every token is a party.
            filters.setdefault("garment", []).append(token)
        index += 1

    if analysis:
        suggest = False
        if limit is None:
            limit = 5
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
            if party and party not in _SYSTEM_WORDS and party not in _GRAPH_SOURCE_ALIASES:
                explicit_buyer = bool(set(tokens) & {"buyer", "buyers"})
                explicit_brand = bool(set(tokens) & {"brand", "brands"})
                if explicit_brand and not explicit_buyer:
                    filters["brand"] = [party]
                    filters.pop("buyer", None)
                elif explicit_buyer and not explicit_brand:
                    filters["buyer"] = [party]
                    filters.pop("brand", None)
                else:
                    # Unqualified party (e.g. "analysis primark data"): keep both until
                    # live metadata picks brand vs buyer for the selected KG.
                    filters["brand"] = [party]
                    filters["buyer"] = [party]

    # E-comm KGs store Brand, not PLM Buyer.
    if preferred_graphs == ["ecom_products"] or (
        "ecom_products" in preferred_graphs and "own_products" not in preferred_graphs
    ):
        if group_by == "buyer":
            group_by = "brand"
        if filters.get("buyer") and not filters.get("brand"):
            filters["brand"] = filters.pop("buyer")
        else:
            filters.pop("buyer", None)

    flat = {key: " ".join(values) for key, values in filters.items()}
    flat = sanitize_party_filters(flat)
    flat = _drop_group_by_as_filter(flat, group_by, question)

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
    }


def sanitize_party_filters(filters: dict) -> dict:
    """Brand/buyer names must not also appear as garment/category filters."""

    cleaned = dict(filters or {})
    parties: list[str] = []
    for key in ("brand", "buyer"):
        value = cleaned.get(key)
        if value:
            parties.append(normalize_name(str(value)))
    if not parties:
        return cleaned
    for key in ("garment", "garment_type", "category"):
        value = cleaned.get(key)
        if value and normalize_name(str(value)) in parties:
            cleaned.pop(key, None)
    return cleaned


def _drop_group_by_as_filter(filters: dict, group_by: str | None, question: str) -> dict:
    """Do not turn 'material wise' into material=<invented value>."""

    cleaned = dict(filters or {})
    if not group_by or group_by not in cleaned:
        return cleaned
    value = normalize_name(str(cleaned.get(group_by) or ""))
    tokens = set(_question_tokens(question))
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
        if before and _is_party_token(before) and before not in _GRAPH_SOURCE_ALIASES:
            return before
        if after and _is_party_token(after) and after not in _GRAPH_SOURCE_ALIASES:
            return after
    # "analysis primark data" / "study zara" — sole party token is the subject.
    parties = [
        token
        for token in tokens
        if _is_party_token(token) and token not in _GRAPH_SOURCE_ALIASES and token not in _SYSTEM_WORDS
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
        plan["preferred_graphs"] = list(plan.get("preferred_graphs") or []) or preferred_graphs_from_question(question) or catalog.preferred_graphs()
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
        or preferred_graphs_from_question(question)
        or catalog.preferred_graphs()
    )
    model_plan["preferred_graphs"] = preferred
    model_plan["available_attributes"] = catalog.all_attributes()
    # E-comm: buyer → brand for filters/group_by.
    if preferred == ["ecom_products"] or (
        "ecom_products" in preferred and "own_products" not in preferred
    ):
        if model_plan.get("group_by") == "buyer":
            model_plan["group_by"] = "brand"
        filters = dict(model_plan.get("filters") or {})
        if filters.get("buyer") and not filters.get("brand"):
            filters["brand"] = filters.pop("buyer")
        else:
            filters.pop("buyer", None)
        if re.search(r"\ball\s+(buyers?|brands?)\b", question.casefold()):
            filters.pop("brand", None)
            filters.pop("buyer", None)
            model_plan["group_by"] = model_plan.get("group_by") or "brand"
        model_plan["filters"] = filters
    model_plan["source"] = "model"
    return model_plan


def _plan_timeout(seconds: float | None) -> float | None:
    if seconds is None:
        return None
    return min(12, seconds)


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
        # Never fuzzy-match graph/source words ("ecomm", "comm", "plm") onto Primark etc.
        if (
            typed_key in _SYSTEM_WORDS
            or typed_key in _GRAPH_SOURCE_ALIASES
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
