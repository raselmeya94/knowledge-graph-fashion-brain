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
# Product-type words — never brand/buyer parties in "A or B" compare / party filters.
_GARMENT_TYPE_WORDS = _words(
    """
    sweater sweaters jumper jumpers cardigan cardigans dress dresses jacket jackets
    coat coats shirt shirts tshirt tshirts hoodie hoodies jean jeans trouser trousers
    pant pants knitwear knit knits top tops blouse blouses skirt skirts tee tees
    polo polos shorts sock socks scarf scarves bag bags shoe shoes sneaker sneakers
    blazer blazers swimwear loungewear pyjama pyjamas pajama pajamas
    """
)
# Singular query tokens used in CONTAINS matching (tops↔top cardigan on PLMEX).
_GARMENT_MATCH_ALIASES: dict[str, tuple[str, ...]] = {
    "top": ("top", "tops"),
    "tops": ("top", "tops"),
    "dress": ("dress", "dresses"),
    "dresses": ("dress", "dresses"),
    "dresse": ("dress", "dresses"),  # bad stem guard
    "blouse": ("blouse", "blouses"),
    "blouses": ("blouse", "blouses"),
    "blous": ("blouse", "blouses"),  # bad stem guard
    "jacket": ("jacket", "jackets"),
    "jackets": ("jacket", "jackets"),
    "skirt": ("skirt", "skirts"),
    "skirts": ("skirt", "skirts"),
    "jumper": ("jumper", "jumpers"),
    "jumpers": ("jumper", "jumpers"),
    "sweater": ("sweater", "sweaters"),
    "sweaters": ("sweater", "sweaters"),
    "hoodie": ("hoodie", "hoodies"),
    "hoodies": ("hoodie", "hoodies"),
}

# Words that can never be a filter value (question grammar, field names, polite filler).
_COUNT_STOPWORDS = _words(
    """
    how many product products are is a an the of in our catalog catalogue plm data synthetic demo there do we have
    with and or for from at that this be any all every each number count total visible clearance by gender wise per
    breakdown analysis analyze analyzing analyse summary summarize summarise summarization summarisation
    grouped distribution split brands brand buyer buyers season seasons category
    categories material materials department departments designer designers year now can you your make me
    please could would them those these their same also just show give want let us again still instead about
    possible possibly impossible maybe perhaps able unable capability capable feasible
    tell tells telling told say says saying said ask asks asking asked explain explains explaining
    explore exploring explored focus focusing focused narrow narrowing confirm confirming confirmed
    accurate accuracy reply report reports dimension dimensions aspect aspects complete
    yes yeah yep ok okay sure proceed correct fine alright y
    what which who when where list has have had was were been contain contains containing made using use uses top bottom
    name names named naming highest lowest leading rank ranking first last garments type types kind kinds normally usually typical
    typically often frequently frequent frequency common commonly mostly making makes does did produce produces producing manufacture
    manufactures if then upcomming upcoming coming coming seasion sesion suggest suggestion suggestions
    specification specifications spec specs recommend recommendation recommendations sample samples new
    used shared share same interconnect interconnection found matched match result results currently available
    across among amongst only also present available registered registry metadata attribute attributes
    unique values description descriptions knowledge graph graphs kg kgs source sources system basic info
    information overview stored inside within ecommerce ecommerces commerce commerces ecom ecomm
    node nodes event events snapshot snapshots exist exists
    up range ranges assortment assortments catalog catalogs catalogue catalogues
    break breaks broke down dominate dominates mix mixes mixed mixture mixtures blend blends
    listing listings ops brief briefs online concentrated supports join
    see seen seeing detail details detailled construction constructions fit fits
    """
)
_RANK_WORDS = _words("top bottom highest lowest leading most more largest biggest smallest fewest")
# Comparison / change language — never a catalog facet VALUE (garment/category/…).
_METRIC_OPERATOR_WORDS = _words(
    """
    decrease decreases decreased decreasing increase increases increased increasing
    drop drops dropped dropping rise rises rose rising fall falls fallen falling
    change changes changed changing trend trends trending history historical historically
    fluctuation fluctuations mover movers move moves moved moving cheaper expensive costly
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
    move moves moved moving mover movers
    appear appears appeared appearing often frequently frequent frequency common commonly typical typically
    exist exists existing stored store storage node nodes event events snapshot snapshots
    instruction instructions detail details composition compositions outline outlines
    region regions country countries area areas uk eu us
    decide decides deciding decision study studies studying studies
    sell sells selling sold buy buys buying
    tell tells telling told say says saying said ask asks asking asked
    explain explains explaining explained
    explore exploring explored focus focusing focused narrow narrowing
    confirm confirming confirmed accurate reply report dimension dimensions
    aspect aspects yes yeah yep ok okay sure proceed correct fine alright
    getting got gone recorded recording ignore ignoring than actually
    show shows showing shown like likes careful false friend friends
    merge merging merged blindly separate separately role roles rather
    give gives giving source sources count counts value values
    up how versus vs blind
    """
)
# Modal / capability language — never a brand or buyer ("is that possible…?").
_MODAL_WORDS = _words(
    """
    possible possibly impossible maybe perhaps perhapsly able unable capability
    capable capabilities feasible feasibly feasibility potential potentially
    might maybe can could would should may
    """
)
# Measure / property names that are not garment types when used alone in ranking asks.
_MEASURE_PROPERTY_WORDS = _words(
    "price prices pricing cost costs amount amounts rating ratings discount discounts "
    "margin margins stock quantity qty average avg mean median "
    "availability availabilities care construction constructions fit fits "
    "supplier suppliers origin origins season seasons"
)
# Stock-status labels (and report English around them) — never brand/buyer.
# They may be availability VALUES only when the user clearly filters to that status.
_AVAILABILITY_STATUS_WORDS = _words(
    "instock in-stock instocked mixed outofstock out-of-stock "
    "available unavailable stockstatus"
)
_MAKE_VERBS = _words("make makes making produce produces producing manufacture manufactures manufacturing")
_MAKE_ADVERBS = _words("normally usually typically often frequently frequent frequency commonly mostly")
# Catalog / graph / system words: never a garment, material, or buyer/brand filter value.
_SYSTEM_WORDS = _words(
    "plm plmex plm-ex system database catalog catalogue ecom ecomm ecommerce commerce "
    "neo4j postgres sql kg graph graphs source sources knowledge "
    "pricechangeevent availabilitysnapshot entity entities "
    # Product / app name — never a brand, garment, or category filter.
    "fashion brain fashionbrain fashiongpt bettex assistant chatbot app"
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
    tell tells telling told say says saying said ask asks asking asked explain explains
    explore exploring explored focus focusing focused narrow narrowing confirm confirming
    accurate reply report dimension dimensions aspect aspects
    catalog catalogue data information info overview knowledge graph graphs kg source
    sources system database node nodes available visible there here inside within
    example examples sample samples as can could around roughly approx approximately
    """
)
# A filter value containing any of these is the question pasted back, not a catalog value.
_FILTER_NOISE = _words(
    "normally usually typical typically making make makes made type types kind kinds which what "
    "product products does new sample samples suggest specification specifications "
    "generate generates generating build builds building write writes writing draft drafts "
    "create creates creating provide provides propose proposes invent invents formulate "
    "found matched match result results currently available total number count "
    "largest biggest highest lowest most least fewest smallest top "
    "decrease decreases decreased increase increases increased drop drops rise rose "
    "change changes trend trends trending history historical fluctuation fluctuations "
    "analyse analyze analysis attribute wise breakdown takeaway leading "
    "cheaper expensive mover movers more than getting recorded "
    "mix mixes mixture mixtures blend blends common commonly often "
    "listing listings ops brief briefs online concentrated supports join "
    "break breaks broke down dominate dominates "
    "tell tells telling told say says saying said ask asks asking asked "
    "explore exploring explored focus focusing focused narrow narrowing "
    "confirm confirming confirmed accurate reply report dimension dimensions "
    "yes yeah yep ok okay sure proceed correct fine alright "
    "possible possibly impossible maybe perhaps able unable capability capable "
    "feasible feasibility potential potentially might "
    "toward relationship relationships use used using via label labels "
    "garmentcategory required schema hint hints filter filters key keys"
)
_SUGGEST_WORDS = _words(
    "suggest suggestion suggestions recommend recommendation recommendations"
)
# Verbs that mean "build / invent a sample-ready sheet" (paired with spec terms).
# Keep separate from _MAKE_VERBS (assortment "what do they make") so we do not broaden counts.
_SPEC_CREATE_VERBS = _words(
    """
    make makes create creates design designs develop develops prepare prepares draft drafts
    generate generates generating build builds building write writes writing
    propose proposes proposing provide provides providing produce produces producing
    invent invents inventing formulate formulates
    """
)
# Mode-of-catalog cues: user wants a *typical* sheet, not one SKU's stored props.
_SPEC_MODE_CUES = _words(
    "typical typically usual usually common commonly sample samples new"
)
_ANALYSIS_WORDS = _words("analysis analyze analysing analyzing analyse profile review overview study")
# UI / agent status words and rewrite leftovers — never brand/buyer values.
_AGENT_UI_NOISE = _words(
    """
    thinking thought thoughts hidden providing provided provide occurrence
    occurrences categorized categorised differences difference explaining
    explained carefully careful blindly merging merge merged roles role
    source sources selected catalogs catalog looking resolved ask
    """
)
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
    'Use intent "specification" when the user wants a product/sample specification sheet suggested from '
    "catalog modes — any wording (generate/create/make/draft/build/write/provide/propose a spec, "
    "typical/usual/common sweater spec, product specification for a brand+garment without a style/SKU id). "
    'Use intent "unsupported" never for those; do not treat them as count. '
    "Only avoid specification when they ask for the full/exact stored details of a named style/SKU. "
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


def is_trending_product_ask(question: str) -> bool:
    """Which products are hot/trending now (activity ranking) — not brand avg trend.

    Examples: "which product is more trending on Primark", "hot products now".
    """

    text = normalize_name(question or "")
    if not text:
        return False
    # "which color is most running" is a facet ranking — not product hotness.
    if is_dominant_facet_ask(question):
        return False
    if not re.search(r"\b(trend|trends|trending|hot|popular|buzzing)\b", text):
        return False
    # Explicit analysis / overall brand trend language stays on PriceChangeEvent avg path.
    if re.search(r"\b(analysis|analyse|analyze|overall|average|avg|summary)\b", text):
        return False
    if re.search(r"\bprice\s+trend\b", text):
        return False
    productish = bool(re.search(r"\b(products?|items?|styles?|skus?)\b", text))
    whichish = bool(re.search(r"\b(which|what|top|most|more|currently|now|present|right now)\b", text))
    if productish and whichish:
        return True
    if re.search(r"\b(trending|hot|popular)\s+(products?|items?|styles?)\b", text):
        return True
    return False


_FACET_RANK_TOKEN = (
    r"colou?rs?|materials?|fabrics?|categor(?:y|ies)|famil(?:y|ies)|"
    r"sections?|brands?|buyers?|garment(?:\s+types?)?"
)


def dominant_facet_from_question(question: str) -> str | None:
    """Return facet key for 'which color is most…' asks, else None."""

    text = normalize_name(question or "")
    if not text:
        return None
    mapping = (
        ("color", r"colou?rs?"),
        ("material", r"materials?|fabrics?"),
        ("category", r"categor(?:y|ies)"),
        ("family", r"famil(?:y|ies)"),
        ("section", r"sections?"),
        ("brand", r"brands?"),
        ("buyer", r"buyers?"),
        ("garment_type", r"garment(?:\s+types?)?"),
    )
    for key, pat in mapping:
        if re.search(
            rf"\bwhich\s+(?:{pat})\b.{{0,60}}\b(most|top|popular|common|running|dominant|frequent)\b",
            text,
        ):
            return key
        if re.search(rf"\b(most|top|popular|common|dominant|frequent)\s+(?:{pat})\b", text):
            return key
        if re.search(rf"\bgrouped by\s+(?:{pat})\b", text) or re.search(
            rf"\bby\s+(?:{pat})\b.{{0,40}}\b(count|share|breakdown)\b", text
        ):
            return key
    return None


def is_dominant_facet_ask(question: str) -> bool:
    """True for 'which color is most running' / 'most common material'."""

    return dominant_facet_from_question(question) is not None


def is_price_trend_analysis_ask(question: str) -> bool:
    """True for 'make trending analysis', 'trend analysis', 'price trend', etc.

    In Fashion Brain catalog chat these mean PriceChangeEvent trends — not a
    static product-mix / attribute-wise assortment profile.
    """

    text = normalize_name(question or "")
    if not text:
        return False
    # "which product is trending" is product hotness — not brand average trend.
    if is_trending_product_ask(question):
        return False
    # Explicit price + trend/history language.
    if re.search(r"\bprices?\b", text) and re.search(
        r"\b(trend|trends|trending|history|historical|change|changes|over time)\b",
        text,
    ):
        return True
    # "make trending analysis" / "do a trend analysis" — no "price" word required.
    if re.search(
        r"\b("
        r"(?:make|run|do|show|give|want|need|please)?\s*"
        r"(?:a\s+|the\s+)?"
        r"(?:price\s+)?"
        r"(?:trend|trends|trending)"
        r"(?:\s+analysis|\s+analyse|\s+analyze)"
        r"|"
        r"trending\s+analysis|trend\s+analysis|price\s+trend"
        r")\b",
        text,
    ):
        return True
    return False


def _is_price_history_question(question: str) -> bool:
    """Catalog price movers / trends / history (ecom PriceChangeEvent path)."""

    text = normalize_name(question)
    # Current catalog ranking ("top 10 highest prices") is a product list, not history.
    if is_ranked_product_listing_ask(question):
        return False
    # "not history" / "current-state" are current-price asks, not PriceChangeEvent.
    text = re.sub(r"\bnot\s+history\b", " ", text)
    if re.search(
        r"\b(right now|current[- ]state|current average|current price|average price of)\b",
        text,
    ) and not re.search(
        r"\b(change|changes|changed|trend|trends|history|historical|over time|"
        r"decrease|increase|drop|rise|mover|movers|fluctuation)\b",
        text,
    ):
        return False
    if re.search(r"\bpricechangeevent\b", text):
        return True
    # "got cheaper" / currency-switch asks may omit the word "price".
    if re.search(r"\b(cheaper|currency\s+switch|currency\s+switches|same[- ]currency)\b", text):
        return True
    # "which product is trending" → product activity ranking (still temporal path).
    if is_trending_product_ask(question):
        return True
    # "make trending analysis" → price-change trends (not assortment profile).
    if is_price_trend_analysis_ask(question):
        return True
    if not re.search(r"\bprices?\b", text):
        return False
    # Explicit change/history language only — not "highest price" product rankings.
    return bool(
        re.search(
            r"\b(decrease|decreases|decrease[sd]|increase|increases|increased|"
            r"drop|drops|dropped|rise|rises|rose|rising|fallen|falling|"
            r"change|changes|changed|trend|trends|history|historical|"
            r"fluctuation|fluctuations|move|moves|moved|moving|mover|movers|"
            r"cheaper|"
            r"over time|last\s+\d+\s+(day|days|month|months|week|weeks))\b",
            text,
        )
    )


def temporal_count_entity(question: str) -> str | None:
    """Schema temporal node counts — not Product garment filters."""

    text = normalize_name(question)
    if not re.search(r"\b(how many|count|number|exist|exists|stored)\b", text):
        return None
    if re.search(r"\b(price\s*change\s*events?|pricechangeevent)\b", text):
        return "PriceChangeEvent"
    # "availability changes/snapshots/events" and AvailabilitySnapshot label forms.
    if re.search(
        r"\b(availability\s*(changes?|snapshots?|events?)|availabilitysnapshot)\b",
        text,
    ):
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


def mentions_stock_status_comparison(question: str) -> bool:
    """True when stock statuses are analysis axes (mix / A vs B), not a single filter."""

    text = normalize_name(question or "")
    if not text:
        return False
    if re.search(
        r"\b(stock[\s_-]?status|availability)\b.{0,40}\b(mix|mixes|mixture|breakdown|profile|brief)\b",
        text,
    ):
        return True
    if re.search(
        r"\b(in[\s_-]?stock|instock)\b.{0,40}\b(vs|versus|or)\b.{0,40}\b(mixed|outofstock|out of stock)\b",
        text,
    ):
        return True
    if re.search(
        r"\b(mixed|outofstock|out of stock)\b.{0,40}\b(vs|versus|or)\b.{0,40}\b(in[\s_-]?stock|instock)\b",
        text,
    ):
        return True
    return False


def is_availability_price_ops_brief_ask(question: str) -> bool:
    """Availability + pricing ops brief (stock mix + price events by stock status).

    Not a full attribute-wise assortment profile (A16-style asks).
    """

    text = normalize_name(question or "")
    if not text:
        return False
    has_avail = bool(
        re.search(r"\b(availability|stock[\s_-]?status|instock|in stock|mixed listings)\b", text)
        or mentions_stock_status_comparison(question)
    )
    has_price = bool(
        re.search(r"\b(price|prices|pricing|pricechangeevent)\b", text)
        and re.search(r"\b(event|events|change|changes|join|concentrated)\b", text)
    )
    has_ops = bool(
        re.search(r"\b(ops|brief|combine|join)\b", text)
        or mentions_stock_status_comparison(question)
    )
    return bool(has_avail and has_price and has_ops)


# Attribute words that appear as a *list* in product briefs (not a single group_by).
_BRIEF_FACET_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bcompositions?\b", "material"),
    (r"\bmaterials?\b", "material"),
    (r"\bconstructions?\b", "construction"),
    (r"\bfits?\b", "fit"),
    (r"\bcare(?:\s+instructions?)?\b", "care_instruction"),
    (r"\bcolou?rs?\b", "color"),
    (r"\bprices?\b", "price"),
)


def named_product_brief_facets(question: str) -> list[str]:
    """Catalog facets named in a product brief (deduped, order preserved)."""

    text = normalize_name(question or "")
    if not text:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for pattern, field in _BRIEF_FACET_PATTERNS:
        if re.search(pattern, text):
            if field in seen:
                continue
            seen.add(field)
            out.append(field)
    return out


def analysis_fields_for_product_brief(question: str) -> list[str]:
    """Retrieve facets for a C1-style brief — only what the user named.

    Maps price → price_band (numeric price stays in overview metrics).
    Composition also pulls construction on ecom, where body blends are stored there.
    """

    text = normalize_name(question or "")
    named = named_product_brief_facets(question)
    out: list[str] = []
    seen: set[str] = set()

    def _add(field: str) -> None:
        key = str(field or "").strip()
        if not key or key in seen:
            return
        seen.add(key)
        out.append(key)

    for field in named:
        if field == "price":
            _add("price_band")
            continue
        if field == "material":
            _add("material")
            # Ecom PDPs put body composition under construction.
            if re.search(r"\bcompositions?\b", text):
                _add("construction")
            continue
        _add(field)
    return out


def is_product_attribute_brief_ask(question: str) -> bool:
    """Brand/garment product brief across several attributes (C1-style).

    Example: \"full Primark sweater brief — composition, construction, fit, care, colours, and price\".
    Not an ops/stock brief, not a single-dimension material breakdown, not a suggest-sheet.

    Must not call ``_is_suggestion`` / ``_is_analysis`` (those call back into this helper).
    """

    if is_availability_price_ops_brief_ask(question):
        return False
    text = normalize_name(question or "")
    if not text:
        return False
    tokens = set(text.split())
    # Inline suggest-sheet exclusion (avoid recursion with _is_suggestion → _is_analysis).
    if tokens & _SUGGEST_WORDS:
        return False
    has_spec_term = bool(tokens & {"spec", "specs", "specification", "specifications"})
    if has_spec_term and (
        tokens & _SPEC_CREATE_VERBS
        or tokens & _SPEC_MODE_CUES
        or re.search(r"\b(?:product\s+)?(?:specifications?|specs?)\s+(?:for|of)\b", text)
    ):
        return False
    facets = named_product_brief_facets(question)
    if len(facets) < 2:
        return False
    has_brief_cue = bool(
        re.search(r"\bbriefs?\b", text)
        or re.search(r"\b(?:full|complete|detailed)\b.{0,40}\b(?:specifications?|specs?|details?)\b", text)
        or re.search(
            r"\b(?:composition|construction|fit|care).{0,40}\b(?:colour|color|price)\b",
            text,
        )
    )
    if not has_brief_cue:
        return False
    # Need a party and/or garment so this is a scoped product brief, not a bare facet list.
    token_list = [t for t in text.split() if t]
    has_party = any(_is_party_token(tok) for tok in token_list)
    has_scope = bool(garment_type_from_question(question) or has_party)
    return has_scope


def _is_availability_history_question(question: str) -> bool:
    text = normalize_name(question)
    if not re.search(r"\bavailability\b", text):
        return False
    # Ops brief / stock-status mix is current assortment + price join, not snapshot history.
    if mentions_stock_status_comparison(question):
        return False
    if re.search(
        r"\b(history|historical|changed|change|changes|recent|recently|snapshot|snapshots)\b",
        text,
    ):
        return True
    # Status labels alone only count with an explicit transition ("from … to …").
    return bool(
        re.search(r"\bfrom\b.+\bto\b", text)
        and re.search(r"\b(instock|in stock|mixed|outofstock|out of stock)\b", text)
    )


def availability_history_mode(question: str) -> str | None:
    """Classify availability-history asks (AvailabilitySnapshot path)."""

    if not _is_availability_history_question(question):
        return None
    text = normalize_name(question)
    if temporal_count_entity(question) == "AvailabilitySnapshot":
        return "count_snapshots"
    # State transition: "from <state> to <state>" over snapshot history.
    if re.search(r"\bfrom\b.+\bto\b", text) and re.search(
        r"\b(availability|stock|instock|in stock|mixed|outofstock|out of stock)\b", text
    ):
        return "transition"
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
        r"\b(sweaters?|jumpers?|cardigans?|knitwear)\b",
        r"\b(dresses?)\b",
        r"\b(jackets?|coats?)\b",
        r"\b(jeans|trousers|pants)\b",
        r"\b(skirts?)\b",
        r"\b(blouses?)\b",
        r"\b(blazers?)\b",
        r"\b(shirts?)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        raw = re.sub(r"\s+", "-", match.group(1).strip())
        if raw in {"jeans", "pants", "trousers"}:
            return raw
        # Irregular / -es plurals (do not strip "ses" from blouses → blou).
        irregular = {
            "dresses": "dress",
            "blouses": "blouse",
            "skies": "sky",
        }
        if raw in irregular:
            return irregular[raw]
        if raw.endswith("ss"):
            return raw
        if raw.endswith("s") and len(raw) > 3:
            return raw[:-1]
        return raw
    # Ranking "top/tops <schema field|measure>" is never a garment.
    # Objects come from CATALOG_FIELDS aliases + measure/property keys — not an English guess list.
    cleaned = _strip_ranking_top_collocations(text)
    # Bare "tops" / contextual "top" is a garment after ranking phrases are removed.
    if re.search(r"\btops\b", cleaned):
        return "tops"
    if re.search(r"\btop\b", cleaned):
        if re.search(
            r"\b(in|for|among|across|of)\s+\w[\w&'-]*\s+top\b|\b\w[\w&'-]*\s+top\b",
            cleaned,
        ):
            return "tops"
    return None


def _strip_ranking_top_collocations(text: str) -> str:
    """Drop ``top/tops <facet>`` ranking phrases using schema aliases, not noun lists."""

    tokens = normalize_name(text or "").split()
    if not tokens:
        return ""
    # Non-schema ranking heads (still grammar, not catalog values).
    rank_heads = frozenset(
        {
            "products",
            "items",
            "counts",
            "count",
            "movers",
            "results",
            "values",
            "n",
            "shares",
            "sellers",
            "styles",
            "attributes",
            "fields",
        }
    )
    out: list[str] = []
    index = 0
    while index < len(tokens):
        tok = tokens[index]
        if tok in {"top", "tops"}:
            nxt = tokens[index + 1] if index + 1 < len(tokens) else ""
            if nxt.isdigit():
                index += 2
                continue
            alias = _alias_at(tokens, index + 1)
            if alias:
                index += 1 + alias[1]
                continue
            if nxt in _MEASURE_PROPERTY_WORDS or nxt in rank_heads:
                index += 2
                continue
            if nxt in _FIELD_ALIASES:
                index += 2
                continue
        out.append(tok)
        index += 1
    return " ".join(out)


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
    if is_availability_price_ops_brief_ask(text):
        return "availability_price"
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
    text = re.sub(r"\bnot\s+history\b", " ", text)
    if not re.search(r"\b(average|avg|mean)\b", text):
        return False
    if not re.search(r"\bprices?\b", text):
        return False
    if _is_price_history_question(question):
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


def is_product_count_ask(question: str) -> bool:
    """True when the user is asking for a quantity (not a listing or analysis).

    Fallback helper for Query Understanding when the LLM path did not run or
    failed, and for listing guards. LLM Query Understanding is authoritative
    for long/narrative asks; this classifier must not be the primary gate in
    the orchestrator.
    """

    text = normalize_name(question)
    if not text:
        return False
    # Explicit quantity cues (including paraphrases of "how many").
    if re.search(
        r"\b("
        r"how many|number of|count of|total number|how much|"
        r"how large|how big|size of|volume of|quantity of|"
        r"product count|item count|sku count|"
        r"total (?:products?|items?|skus?|styles?|brands?|categories)|"
        r"(?:products?|items?|skus?|styles?|brands?|categories) total"
        r")\b",
        text,
    ):
        return True
    if re.search(r"\b(count|total|quantity)\b", text) and re.search(
        r"\b(products?|items?|skus?|styles?|brands?|categories|buyers?)\b",
        text,
    ):
        return True
    return False


def is_capability_followup_ask(question: str) -> bool:
    """True for 'is that possible…?', 'can you…?' — capability, not a new party name."""

    text = normalize_name(question or "")
    if not text:
        return False
    return bool(
        re.search(
            r"\b("
            r"is that possible|is it possible|possible to|"
            r"can you|could you|would you|are you able|"
            r"able to|feasible to"
            r")\b",
            text,
        )
    )


def is_catalog_dataset_total_ask(question: str) -> bool:
    """Whole catalog/dataset product total — must not inherit a prior brand/buyer.

    Examples: "How many products are in the e-comm data?",
    "How many products are in the PLM data?"
    Not: "How many products does Primark have?"
    Not: "How many sweaters in plmex" (scoped type, even with typos).
    """

    if not is_product_count_ask(question):
        return False
    text = normalize_name(question)
    graph_scope = re.search(
        r"\b(?:in|across|from|within)\s+(?:the\s+)?(?:"
        r"e[\s-]?comm(?:erce)?|ecom|plm(?:ex)?|catalog|assortment|"
        r"data(?:set|base)?|product\s+(?:data|catalog|graph|source)|"
        r"knowledge\s+graph"
        r")\b",
        text,
    )
    if not graph_scope:
        return False
    # Require a generic countable noun — "how many X in plmex" where X is a
    # type/typo (sweaters / swweaters) is a filtered count, not the catalog total.
    return bool(
        re.search(
            r"\b(?:"
            r"how\s+many\s+(?:(?:total|overall)\s+)?(?:products?|items?|skus?|styles?|records?|rows?)\b|"
            r"(?:number|count|total)\s+(?:of\s+)?(?:products?|items?|skus?|styles?|records?|rows?)\b|"
            r"(?:products?|items?|skus?|styles?)\s+(?:are\s+)?(?:in|across|from|within)\b|"
            r"(?:product|item|sku|style)\s+count\b"
            r")",
            text,
        )
    )


def is_facet_value_ranking_ask(question: str, graph_id: str | None = None) -> tuple[str, int] | None:
    """'list top 10 brand names' / 'top 10 buyers' → (facet_key, n), else None.

    These are kg_stats filter rankings, not ranked product price listings.
    """

    text = normalize_name(question or "")
    if not text:
        return None
    # Must look like a top-N ask (allow "garment type(s)" multi-word facets).
    m = re.search(
        r"\b(?:list|show|give|name|what(?:'s| is| are)?)\s+(?:me\s+)?(?:the\s+)?"
        r"top\s+(\d+)\s+([a-z][a-z0-9_]*(?:\s+[a-z][a-z0-9_]*){0,2})(?:\s+names?)?\b",
        text,
    )
    if not m:
        m = re.search(
            r"\btop\s+(\d+)\s+([a-z][a-z0-9_]*(?:\s+[a-z][a-z0-9_]*){0,2})(?:\s+names?)?\b",
            text,
        )
    if not m:
        return None
    n = max(1, min(int(m.group(1)), 50))
    raw = str(m.group(2) or "").strip().casefold()
    # Drop trailing filler words that are not part of the facet name.
    raw = re.sub(r"\b(name|names|value|values|in|on|from|for)\b.*$", "", raw).strip()
    if raw in {"product", "products", "item", "items", "sku", "skus", "style", "styles", "price", "prices"}:
        return None
    try:
        from knowledge_graph.retrieval.ecom import _fold_facet_token

        folded = _fold_facet_token(raw.replace(" ", "_")) or raw.replace(" ", "_")
    except Exception:
        folded = raw.replace(" ", "_")
        if folded.endswith("s") and len(folded) > 3:
            folded = folded[:-1]
    # Confirm the facet exists in kg_stats (prefer sticky/named graph).
    keys = _facet_keys_from_kg_stats(graph_id)
    token_set = {
        folded,
        raw,
        raw.replace(" ", "_"),
        folded + "s",
        raw + "s",
        *raw.split(),
        *folded.split("_"),
    }
    hit = _match_facet_in_tokens(token_set, keys) if keys else folded
    if not hit or hit in {"product", "products"}:
        # Still accept common party facets even if stats temporarily unavailable.
        if folded in {"brand", "buyer", "category", "family", "section", "garment_type"}:
            return folded, n
        if "garment" in token_set and "type" in token_set:
            return "garment_type", n
        return None
    return hit, n


def question_has_catalog_deixis(question: str) -> bool:
    """True when the ask points at prior catalog scope ('of them', 'those', 'same'…)."""

    text = normalize_name(question or "")
    if not text:
        return False
    if re.search(r"\b(of them|among them|from them|in them|those|these|same (one|catalog|source|graph|data)|that (catalog|source|graph|data))\b", text):
        return True
    return bool(re.search(r"\b(them|those|these)\b", text))


def is_ranked_product_listing_ask(question: str) -> bool:
    """True for 'top 10 products…' / 'highest prices among them' — a ranked product list.

    Not a product count, not PriceChangeEvent history, and not 'top 10 brand names'.
    """

    text = normalize_name(question or "")
    if not text:
        return False
    # Facet value rankings ("top 10 brands") stay on kg_stats — never product price lists.
    if is_facet_value_ranking_ask(question) is not None:
        return False
    has_products = bool(re.search(r"\b(products?|items?|skus?|styles?)\b", text))
    if re.search(r"\b(top|bottom|first|last)\s+\d+\b", text) and has_products:
        return True
    if has_products and re.search(
        r"\b(highest|lowest|most\s+expensive|cheapest)\s+prices?\b",
        text,
    ):
        return True
    if has_products and re.search(
        r"\bproducts?\b.+\b(highest|lowest|most\s+expensive|cheapest)\b",
        text,
    ):
        return True
    # "list top 10 …" only counts as a product listing when products/prices are named.
    if re.search(r"\b(show|list|give)\s+(?:me\s+)?(?:the\s+)?top\s+\d+\b", text):
        return bool(re.search(r"\b(products?|items?|skus?|styles?|prices?)\b", text))
    return False


def ranked_product_listing_limit(question: str) -> int | None:
    """Top-N for a ranked product listing, capped for retrieval."""

    text = normalize_name(question or "")
    m = re.search(r"\b(?:top|bottom|first|last|show|list)\s+(\d+)\b", text)
    if m:
        return max(1, min(int(m.group(1)), 50))
    if is_ranked_product_listing_ask(question):
        return 10
    return None


def product_price_sort_direction(question: str) -> str | None:
    """'desc' / 'asc' when the user ranks products by current price."""

    if not is_ranked_product_listing_ask(question):
        text = normalize_name(question or "")
        if not (
            re.search(r"\bprices?\b", text)
            and re.search(r"\b(highest|lowest|expensive|cheap)\b", text)
            and re.search(r"\b(products?|items?)\b", text)
        ):
            return None
    text = normalize_name(question or "")
    if re.search(r"\b(lowest|cheapest|cheap|bottom|least\s+expensive)\b", text):
        return "asc"
    if re.search(r"\b(highest|most\s+expensive|expensive|top)\b", text):
        return "desc"
    return None


def is_open_product_listing(question: str, *, limit: int | None = None) -> bool:
    """True for open 'which/show products …' asks without an explicit top-N.

    These should answer with total + facet mix + a few examples — not dump dozens
    of product-name rows just because Cypher used LIMIT 50.
    """

    if limit is not None:
        return False
    text = normalize_name(question)
    # Count asks stay on product_count — never demote to a sample listing.
    if is_product_count_ask(question):
        return False
    if is_ranked_product_listing_ask(question):
        return False
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
    # Planner shapes from live-schema-aware linguistic detectors (no hardcoded values).
    try:
        from fashion_brain.planning.schema_grounding import detect_query_shape

        shape = detect_query_shape(question)
    except Exception:
        shape = None
    if shape == "shared_overlap":
        return ["shared_products"]
    if shape in {"category_rollup", "volatility", "state_transition"}:
        return ["product_count"]
    if shape == "entity_frequency":
        return ["product_count"]
    if shape == "facet_listing":
        return ["products"]
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
    # Domain attribute / construction / fit asks stay on product_count (never direct).
    if tokens & {"construction", "constructions", "fit", "fits", "composition", "compositions"}:
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


def _facet_keys_from_kg_stats(graph_id: str | None = None) -> list[str]:
    """Indexed filter names from kg_stats (kg → filter → values). Stats-driven, not a noun list."""

    try:
        from knowledge_graph.retrieval.kg_stats import load_graph_stats, stats_available
    except Exception:
        return []
    gids: list[str] = []
    if graph_id:
        gids.append(str(graph_id))
    for alt in ("ecom_products", "plmex_products", "own_products"):
        if alt not in gids:
            gids.append(alt)
    keys: list[str] = []
    seen: set[str] = set()
    for gid in gids:
        try:
            if not stats_available(gid):
                continue
            stats = load_graph_stats(gid) or {}
        except Exception:
            continue
        for raw in list((stats.get("unique_counts") or {}).keys()) + list(
            (stats.get("value_counts") or {}).keys()
        ):
            key = str(raw or "").strip().casefold().replace(" ", "_")
            if not key or key in seen:
                continue
            seen.add(key)
            keys.append(key)
    return keys


def _match_facet_in_tokens(tokens: set[str], facet_keys: list[str]) -> str | None:
    """Map question tokens onto an indexed filter key (buyer/brand/…)."""

    try:
        from knowledge_graph.retrieval.ecom import _fold_facet_token
    except Exception:
        _fold_facet_token = None  # type: ignore[assignment]
    for key in facet_keys:
        folded = (_fold_facet_token(key) if _fold_facet_token else None) or key
        variants = {
            folded,
            folded + "s",
            folded.replace("_", ""),
            folded.replace("_", "") + "s",
            key,
            key + "s",
        }
        if folded.endswith("y") and len(folded) > 2:
            variants.add(folded[:-1] + "ies")
        stem = folded.split("_")[-1] if "_" in folded else folded
        if stem and stem != folded:
            variants |= {stem, stem + "s"}
            if stem.endswith("y") and len(stem) > 2:
                variants.add(stem[:-1] + "ies")
        if tokens & {str(v).casefold() for v in variants if v}:
            return folded
    return None


def _facet_entity_count(question: str, schema=None) -> str | None:
    """'How many <facet>s' → ontology / kg_stats filter key (or product), else None.

    Lookup shape: ``kg_stats[<graph>].value_counts[<filter>]`` — never invent random plurals.
    """

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
    # Placeholder / missing-value probes are product filters, not distinct-facet counts.
    if re.search(
        r"\b(missing|unspecified|unknown|null|__\w+__)\b",
        text,
    ) and re.search(r"\b(brand|buyer|color|category|material)\b", text):
        return None
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
    # Prefer the named product KG's indexed filters (plmex / ecom / …).
    graph_hint = None
    try:
        prefs = preferred_graphs_from_question(question)
        graph_hint = prefs[0] if prefs else None
    except Exception:
        graph_hint = None
    if not graph_hint:
        if tokens & {"plm", "plmex"}:
            graph_hint = "plmex_products"
        elif tokens & {"ecom", "ecomm", "ecommerce", "commerce"}:
            graph_hint = "ecom_products"
    # "how many products does buyer X have" counts products, not distinct buyers.
    # Distinct-facet only when the facet is the counted noun: "how many brands/buyers".
    if tokens & {"product", "products", "item", "items", "sku", "skus", "style", "styles"}:
        if not re.search(
            r"\b(how many|number of|count of)\s+(brands?|buyers?|colors?|colours?|"
            r"categories|materials?|seasons?|genders?)\b",
            text,
        ):
            return "product"
    hit = _match_facet_in_tokens(tokens, _facet_keys_from_kg_stats(graph_hint))
    if hit and hit not in {"product", "products"}:
        return hit
    for token in tokens:
        if token in {"product", "products"}:
            return "product"
    return None

def _is_analysis(question: str) -> bool:
    """Product-mix / assortment analysis (full or focused) — same analytical engine.

    Full: \"full analysis of Primark\" → every facet with distinct values.
    Focused: \"Primark category breakdown\" / \"analysis material wise\" → one facet.
    Compare assortment profiles also belong here.

    Price-trend / PriceChangeEvent asks are NOT assortment analysis even when the
    user says \"price-trend analysis\".
    """

    if is_price_trend_analysis_ask(question) or _is_price_history_question(question):
        return False
    # Stock-status + price-events ops brief is NOT full assortment analysis.
    if is_availability_price_ops_brief_ask(question):
        return False
    # Multi-attribute product brief (composition + construction + …) uses the same engine.
    if is_product_attribute_brief_ask(question):
        return True
    tokens = set(normalize_name(question).split())
    if tokens & _ANALYSIS_WORDS:
        return True
    if "compare" in tokens and tokens & {"assortment", "profile", "profiles", "mix"}:
        return True
    # Breakdown / wise → same engine (focused when a dimension is named).
    if tokens & _BREAKDOWN_HEADS or "wise" in tokens:
        return True
    return False


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
    """Single ask entry for HTTP /ask and conversation messages.

    Uses the same ``iter_ask_events`` router as WebSocket chat (LLM route +
    fashion-scope safety), so definition asks like ``what is GSM?`` never fall
    straight into catalog entity clarification.
    """

    from fashion_brain.flow import iter_ask_events
    from fashion_brain.trace import current_trace, end_trace, start_trace
    from fashion_brain.stage_timing import current_timing, end_timing, log_timing, start_timing, timed

    owns_timing = current_timing() is None
    if owns_timing:
        start_timing()
    owns_trace = current_trace() is None
    if owns_trace:
        start_trace(question, source="http/ask")
    try:
        with timed("ask_pipeline"):
            state = conversation_state if conversation_state is not None else {}
            final: dict | None = None
            for event in iter_ask_events(
                platform,
                question,
                principal,
                history,
                graph_id=graph_id,
                conversation_state=state,
            ):
                if event.get("type") == "done" and isinstance(event.get("answer"), dict):
                    final = event["answer"]
            if final is not None:
                answer = GroundedAnswer.model_validate(final)
            else:
                answer = answer_from_catalog(
                    platform,
                    question,
                    principal,
                    history,
                    graph_id=graph_id,
                    conversation_state=state,
                )
        if owns_trace:
            end_trace(
                status="ok",
                routes=getattr(answer, "routes", None),
                chars=len(getattr(answer, "answer", "") or ""),
            )
        return answer
    except Exception:
        if owns_trace:
            end_trace(status="error")
        raise
    finally:
        if owns_timing:
            bundle = end_timing()
            if bundle is not None:
                log_timing(bundle, LOGGER)


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
    from fashion_brain.agents.query_understanding import clarification_awaiting

    if clarification_awaiting(retrieved):
        LOGGER.info("step 4/4 clarification — skipping response agent")
        return _grounded(question, routes, retrieved.text or "", retrieved, platform)
    if _is_simple_count_result(retrieved, _intention, question):
        LOGGER.info("step 4/4 answer-card writer for count")
        raw = _write_from_answer_card(
            platform,
            question,
            retrieved,
            routes=routes,
            intention=_intention,
        )
        return _grounded(question, routes, raw, retrieved, platform)
    scope_note_text = ""
    if conversation_state is not None:
        scope_note_text = str(conversation_state.pop("_response_scope_note", "") or "")
    LOGGER.info("step 4/4 response agent started")
    answer = ResponseAgent().compose(
        platform,
        question,
        retrieved,
        scope_note=scope_note_text,
        routes=routes,
    )
    LOGGER.info("step 4/4 response agent finished chars=%s", len(answer))
    return _grounded(question, routes, answer, retrieved, platform)


def _write_from_answer_card(
    platform: Platform,
    question: str,
    retrieved,
    *,
    routes: list[str] | None = None,
    intention=None,
) -> str:
    """Build an answer card and write via LLM (or safe fallback when answer_mode != llm)."""

    from fashion_brain.agents.answer_card import (
        WRITER_PROMPT,
        build_answer_card,
        safe_fallback,
        write_answer,
    )
    from fashion_brain.llm import build_model_provider

    resolved = dict(getattr(retrieved, "resolved", None) or {})
    filters = {
        str(k): str(v)
        for k, v in (
            resolved.get("filters")
            or getattr(intention, "filters", None)
            or {}
        ).items()
        if v not in (None, "")
    }
    # Persist filters for ResponseAgent / later turns.
    resolved["filters"] = filters
    if routes:
        resolved["routes"] = list(routes)
    if getattr(intention, "group_by", None):
        resolved["group_by"] = intention.group_by
    if getattr(intention, "limit", None):
        resolved["limit"] = intention.limit
    retrieved.resolved = resolved

    card = build_answer_card(
        question,
        routes or resolved.get("routes") or [],
        filters,
        retrieved,
        response_type=getattr(retrieved, "response_type", None),
        group_by=resolved.get("group_by"),
        limit=resolved.get("limit"),
    )
    settings = getattr(platform, "settings", None)
    answer_mode = str(getattr(settings, "answer_mode", "") or "")
    model_provider = str(getattr(settings, "model_provider", "") or "").strip().casefold()
    if answer_mode != "llm" or model_provider in {"", "none", "null"}:
        body = safe_fallback(card)
        footer = card.footer()
        return f"{body}\n\n{footer}".strip() if footer else body

    provider = build_model_provider(platform.settings)
    configured_timeout = getattr(platform.settings, "request_timeout_seconds", None)
    if configured_timeout is None or configured_timeout <= 0:
        configured_timeout = getattr(platform.settings, "ollama_timeout_seconds", None)
    response_timeout = min(float(configured_timeout), 120.0) if configured_timeout else 90.0

    def llm_call(prompt: str) -> str:
        return str(
            provider.complete(
                prompt,
                timeout=response_timeout,
                system="",
                temperature=0.3,
            )
            or ""
        )

    try:
        return write_answer(card, llm_call, WRITER_PROMPT)
    except Exception:
        body = safe_fallback(card)
        footer = card.footer()
        return f"{body}\n\n{footer}".strip() if footer else body


def _is_simple_count_result(retrieved, intention, question: str) -> bool:
    """True when a count is already fully formatted and needs no response-model rewrite."""

    metadata_count = (
        getattr(retrieved, "retrieve_mode", None) == "template"
        and getattr(retrieved, "response_type", None) in {"stats", "concise"}
        and any(
            str(getattr(item, "record_id", "") or "") == "kg_stats"
            for item in (getattr(retrieved, "evidence", None) or [])
        )
    )
    if metadata_count:
        return True
    if (
        str(getattr(intention, "intent", "") or "").casefold() not in {"count", "product_count"}
        or getattr(intention, "group_by", None)
        or bool(getattr(intention, "analysis", False))
        or bool(getattr(intention, "suggest", False))
        or getattr(intention, "limit", None) not in (None, 0)
    ):
        return False
    if not is_product_count_ask(question):
        return False
    if is_ranked_product_listing_ask(question) or is_open_product_listing(question):
        return False
    try:
        from knowledge_graph.retrieval.ecom import price_history_mode

        if _is_price_history_question(question) or is_price_trend_analysis_ask(question) or price_history_mode(question):
            return False
    except Exception:
        return False
    return True


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
    stock_compare = mentions_stock_status_comparison(question)
    # Mix / vs asks talk ABOUT stock statuses — do not pin availability to one side.
    if stock_compare:
        out.pop("availability", None)
    # Filter TO in-stock only when that is the ask — never when comparing stock statuses.
    elif "availability" not in out and re.search(
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
    # Do not invent section=women when gender is already set — that turns a
    # single-filter kg_stats hit into a multi-filter Cypher miss (P09).
    if "section" not in out and "gender" not in out and re.search(r"\bwomen\b", text):
        out["section"] = "women"
    # Composition/spec compares and product-type listings need the named garment kept.
    if not any(out.get(k) for k in ("garment", "garment_type", "category")):
        garment = garment_type_from_question(question)
        if garment and (
            is_composition_compare_question(question)
            or bool(
                re.search(
                    r"\b(t[\s-]?shirts?|sweaters?|knitwear|dresses?|jeans|hoodies?)\b",
                    (question or "").casefold(),
                )
            )
        ):
            out["garment"] = garment
    # Brand compares only — never promote stock-status "in-stock vs mixed" into brand=.
    parties = _compare_party_names(question)
    if len(parties) >= 2 and not stock_compare:
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
    # Spec / season asks: keep season on the filter payload AND request.season_name.
    if getattr(intention, "season_name", None) and not filters.get("season_name"):
        filters["season_name"] = str(intention.season_name)
    # Preserve the caller's requested scope for provenance on :8001 evidence.
    requested_filters = dict(filters)

    request = BrainRetrieveRequest(
        question=question,
        intent="parent_category" if parent_of else intention.intent,
        filters=filters,
        group_by=intention.group_by,
        unsupported=bool(intention.unsupported),
        limit=intention.limit,
        suggest=(bool(intention.suggest) or intention.intent == "specification") and not bool(getattr(intention, "analysis", False)),
        analysis=bool(getattr(intention, "analysis", False)),
        season_name=intention.season_name or filters.get("season_name"),
        upcoming=bool(intention.upcoming),
        corrections=list(intention.corrections or []),
        count_entity=getattr(intention, "count_entity", None),
        graph_id=graph_id or None,
        graph_ids=list(graph_ids) if graph_ids else None,
        requested_filters=requested_filters,
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


def _is_breakdown_phrase_at(tokens: list[str], index: int) -> int | None:
    """Return index after a breakdown phrase starting at ``index``, else None."""

    if index >= len(tokens):
        return None
    token = tokens[index]
    if token in _BREAKDOWN_SUFFIXES:
        return index + 1
    # Spoken form: "break down" / "breaks down"
    if token in {"break", "breaks", "broke"} and index + 1 < len(tokens) and tokens[index + 1] == "down":
        return index + 2
    return None


def _group_marker_at(tokens: list[str], index: int) -> tuple[str, int] | None:
    """A breakdown phrase at this token. Returns (field, index after the phrase).

    Recognised shapes: "by|per|each <field>", "<field> wise|breakdown|distribution|split|break down",
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
        end = _is_breakdown_phrase_at(tokens, after)
        if end is not None:
            return found[0], end
    head_end = _is_breakdown_phrase_at(tokens, index)
    if head_end is not None or token in _BREAKDOWN_HEADS:
        start = head_end if head_end is not None else index + 1
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
    """'Which type of products' asks for a garment-type ranking, not a garment named type.

    Do not fire on narrative phrases like "sweater-like product types appear in
    e-commerce" — that is compare/scope language, not a breakdown ask.
    """

    tokens = normalize_name(question).split()
    rank_cues = frozenset(
        {
            "which",
            "what",
            "list",
            "show",
            "rank",
            "top",
            "breakdown",
            "break",
            "wise",
            "by",
            "per",
        }
    )
    for index in range(len(tokens)):
        for pattern in _TYPE_PATTERNS:
            if tuple(tokens[index : index + len(pattern)]) != pattern:
                continue
            window = tokens[max(0, index - 4) : index + len(pattern) + 2]
            if set(window) & rank_cues:
                return "garment_type"
    return None


def _is_suggestion(question: str) -> bool:
    """True when the user wants a *suggested* product/sample specification sheet.

    Wording varies (generate / create / typical / product specification for brand+garment).
    Do not require one fixed sentence — only exclude analysis and explicit retrieve-existing cues.
    """

    if _is_analysis(question):
        return False
    tokens = normalize_name(question).split()
    if set(tokens) & _SUGGEST_WORDS:
        return True
    text = " ".join(tokens)
    has_new_product = any(
        phrase in text
        for phrase in (
            "new product",
            "new sample",
            "new style",
            "sample for",
            "product development",
            "development recommendation",
        )
    )
    if has_new_product:
        return True
    has_spec_term = bool(set(tokens) & {"spec", "specs", "specification", "specifications"})
    if not has_spec_term:
        return False
    # Named style/SKU detail asks are retrieve-existing, not mode suggestion.
    if product_ref_from_question(question):
        return False
    has_creation_verb = bool(set(tokens) & _SPEC_CREATE_VERBS)
    has_mode_cue = bool(set(tokens) & _SPEC_MODE_CUES)
    # "product specification for Primark sweater" (no style id) = build a sample-ready sheet.
    has_spec_for_scope = bool(
        re.search(r"\b(?:product\s+)?(?:specifications?|specs?)\s+(?:for|of)\b", text)
    )
    # Create/generate/draft a (detailed) Winter … specification → NEW sheet.
    if has_creation_verb or has_mode_cue:
        return True
    # "Give me the full/detailed specification for …" without create verbs → retrieve existing.
    if re.search(
        r"\b(?:full|complete|detailed|exact)\b.{0,32}\b(?:specifications?|specs?|details?|attributes?)\b",
        text,
    ):
        return False
    return has_spec_for_scope


def _is_product_details_request(question: str) -> bool:
    if _is_suggestion(question):
        return False
    text = normalize_name(question)
    # Explicit style / SKU asks ("Zara style 08965901 made of", "tell me more about … 04786296").
    if product_ref_from_question(question) and re.search(
        r"\b(style|sku|product|made of|material|materials|composition|"
        r"colour|color|price|availability|detail|details|tell me more|about)\b",
        text,
    ):
        return True
    spec_term = r"(?:specifications?|specs?|details?|attributes?)"
    # Retrieve stored props only when the user marks it as full/exact/complete detail,
    # or asks details/attributes for/of a scope without creation/suggest language.
    if re.search(rf"\b(?:full|complete|detailed|exact)\b.{{0,32}}\b{spec_term}\b", text):
        return True
    # Bare "details for/of …" / "attributes for …" still means look up existing records.
    # Do not treat bare "specification for …" as product_details — that is suggest-sheet intent.
    return bool(
        re.search(rf"\b(?:details?|attributes?)\s+(?:for|of)\b", text)
    )


def _is_party_token(token: str) -> bool:
    """A buyer/brand name, not a material, color, field, metric, or task word."""

    raw = str(token or "").strip()
    if not raw:
        return False
    # Multi-word / hyphenated labels: every piece must itself be a party token
    # (stops "in stock" / "in-stock" surviving as a brand compare side).
    parts = [p for p in normalize_name(raw).split() if p]
    if len(parts) > 1:
        return all(_is_party_token(part) for part in parts)
    token = parts[0] if parts else normalize_name(raw)
    if (
        token in _MAKE_VERBS
        or token in _MAKE_ADVERBS
        or token in _COUNT_STOPWORDS
        or token in _SYSTEM_WORDS
        or token in _ANALYSIS_WORDS
        or token in _AGENT_UI_NOISE
        or token in _HISTORY_WORDS
        or token in _METRIC_OPERATOR_WORDS
        or token in _MEASURE_PROPERTY_WORDS
        or token in _AVAILABILITY_STATUS_WORDS
        or token in _TASK_WORDS
        or token in _MODAL_WORDS
        or token in _RANK_WORDS
        or token in _MATERIAL_WORDS
        or token in _COLOR_WORDS
        or token in _COLOR_MARKERS
        or token in _GARMENT_TYPE_WORDS
        or token in _GENDER_VALUES
        or token in _SEASON_WORDS
        or token in _FILTER_NOISE
        or token in _GRAPH_SCORE_STOPWORDS
        or token in _BREAKDOWN_HEADS
        or token in _BREAKDOWN_SUFFIXES
    ):
        return False
    if token.isdigit() or re.fullmatch(r"(19|20)\d{2}", token):
        return False
    if token in _FIELD_ALIASES:
        return False
    # Hyphenated garment forms (t-shirt, sweat-shirt) are never parties.
    if garment_type_from_question(token):
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


def _is_schema_prose_filter(value: str) -> bool:
    """True when a filter value is schema-hint text, not a catalog facet value."""

    text = normalize_name(value or "")
    if not text:
        return True
    if ":" in str(value or ""):  # e.g. ":GarmentCategory" from relationship hints
        return True
    tokens = set(text.split())
    prose = {
        "toward",
        "relationship",
        "relationships",
        "required",
        "schema",
        "hint",
        "hints",
        "garmentcategory",
        "allowlist",
        "via",
    }
    return bool(tokens & prose)


def _reliable_filter(value: str) -> bool:
    """Reject a model filter that pasted the question instead of a catalog value."""

    if _is_schema_prose_filter(value):
        return False
    tokens = normalize_name(value).split()
    if not tokens or len(tokens) > 4:
        return False
    if set(tokens) & _SYSTEM_WORDS:
        return False
    if set(tokens) & _METRIC_OPERATOR_WORDS:
        return False
    if set(tokens) & _TASK_WORDS:
        return False
    if set(tokens) & _MODAL_WORDS:
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
    if not tokens:
        return None
    # Mid-sentence too: "In Zara tops, which material mixes…"
    for index, token in enumerate(tokens):
        if token not in {"what", "which", "list", "show"}:
            continue
        found = _alias_at(tokens, index + 1)
        if found:
            return found[0]
        # "which material mixes / blends" → material breakdown, not material=<mixes>.
        nxt = tokens[index + 1] if index + 1 < len(tokens) else ""
        if nxt in {"material", "materials", "fabric", "fabrics", "composition", "compositions"}:
            return "material" if nxt.startswith("material") or nxt.startswith("fabric") else "composition"
    return None


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
            # Stock-status axes ("in-stock vs mixed") are not brand compares.
            if a in _AVAILABILITY_STATUS_WORDS or b in _AVAILABILITY_STATUS_WORDS:
                continue
            if set(a.split()) & _AVAILABILITY_STATUS_WORDS or set(b.split()) & _AVAILABILITY_STATUS_WORDS:
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


def resolve_catalog_plan(
    question: str,
    history: list | None = None,
    *,
    carry: bool | None = None,
    brand_names: list[str] | None = None,
) -> dict:
    """Keyword plan from the latest question. Earlier turns merge only when carry=True (agent decision)."""

    from fashion_brain.trace import trace_debug

    current = _catalog_plan(question, brand_names=brand_names)
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
        step = _catalog_plan(text, brand_names=brand_names)
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
        # C&A / C and A → c-and-a (normalize_name splits on &).
        if token == "c" and nxt == "a":
            glued.append("c-and-a")
            index += 2
            continue
        if token == "c" and nxt == "and" and index + 2 < len(tokens) and tokens[index + 2] == "a":
            glued.append("c-and-a")
            index += 3
            continue
        glued.append(token)
        index += 1
    return glued


def _facet_attr_tokens_for_graph_scoring() -> set[str]:
    """Filter/facet attribute names must not select a product KG (brand ≠ ecom)."""

    keys = set(_facet_keys_from_kg_stats())
    out: set[str] = set()
    for key in keys:
        out.add(key)
        out.add(key + "s")
        stem = key.split("_")[-1]
        if len(stem) >= 4:
            out.add(stem)
            out.add(stem + "s")
        if key.endswith("y") and len(key) > 4:
            out.add(key[:-1] + "ies")
        if stem.endswith("y") and len(stem) > 4:
            out.add(stem[:-1] + "ies")
    # Always exclude the most common party/mix facet nouns even before stats load.
    out |= {
        "brand",
        "brands",
        "buyer",
        "buyers",
        "category",
        "categories",
        "family",
        "families",
        "section",
        "sections",
        "color",
        "colors",
        "colour",
        "colours",
        "material",
        "materials",
        "garment",
        "garments",
    }
    return out


def _graph_scope_signal_tokens(tokens: set[str]) -> set[str]:
    """Tokens that can name a catalog world — drop grammar, facets, and generic 'products/data'."""

    facet_noise = _facet_attr_tokens_for_graph_scoring()
    return {
        tok
        for tok in tokens
        if len(tok) >= 2
        and tok not in _GRAPH_SCORE_STOPWORDS
        and tok not in facet_noise
        and not tok.isdigit()
    }


def question_names_catalog_explicitly(question: str, platform=None) -> bool:
    """True when the user names a catalog world (plm/ecom/…), not merely a facet like brand."""

    text = normalize_name(question or "")
    # Glued tokens so "e-commerce" / "e commerce" count as ecommerce (normalize splits on '-').
    glued = set(_question_tokens(question))
    # Distinctive catalog tokens even when registry prefs are unavailable.
    if glued & {"plm", "plmex"} or re.search(r"\bplm(?:ex)?\b", text):
        return True
    if glued & {"ecom", "ecomm", "ecommerce", "ecommerces"} or re.search(
        r"\be[\s\-]?comm?(erce)?s?\b", text
    ):
        return True
    named = preferred_graphs_from_question(question, platform)
    if not named:
        return False
    # Generic id fragments that appear in almost every product ask — never treat
    # as an explicit catalog mention (ecom_products → "products" was matching all).
    generic = {
        "products",
        "product",
        "items",
        "item",
        "catalog",
        "data",
        "graph",
        "source",
        "sources",
    }
    # Require a distinctive catalog token / id fragment in the question.
    for gid in named:
        parts = {
            p
            for p in re.split(r"[_\-\s]+", str(gid))
            if len(p) >= 3 and p not in generic
        }
        tokens = set(text.split()) | glued
        if any(part in tokens for part in parts):
            return True
        if "plm" in str(gid) and (
            "plm" in glued or "plmex" in glued or re.search(r"\bplm(?:ex)?\b", text)
        ):
            return True
        if "ecom" in str(gid) and (
            glued & {"ecom", "ecomm", "ecommerce", "ecommerces", "commerce", "commerces"}
            or re.search(r"\be[\s\-]?comm?(erce)?s?\b", text)
        ):
            return True
    return False


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


def _catalog_plan(question: str, *, brand_names: list[str] | None = None) -> dict:
    """Read filters and one breakdown field from the question words. Counts are not guessed here."""

    tokens = _question_tokens(question)
    live_brand_keys = {
        normalize_name(str(n)) for n in (brand_names or []) if n
    }
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
                # "material mixes/blends" is a breakdown ask, not material=mixes.
                if found[0] in {"material", "composition"} and tok in {
                    "mix",
                    "mixes",
                    "mixture",
                    "mixtures",
                    "blend",
                    "blends",
                }:
                    group_by = group_by or found[0]
                    index += 1
                    values = []
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
        # "in <brand> tops/t-shirts/…" — brand scope + garment, not country-of-origin.
        if token == "in" and index + 1 < len(tokens):
            party = tokens[index + 1]
            garment_tok = tokens[index + 2] if index + 2 < len(tokens) else ""
            if _is_party_token(party) and party not in _GRAPH_SCOPE_NOISE and party not in _SYSTEM_WORDS:
                # "in Zara tops" OR "in Zara, which tops…" (garment later).
                if garment_tok in _GARMENT_TYPE_WORDS or garment_type_from_question(garment_tok) or garment_tok in {
                    "tops",
                    "top",
                }:
                    filters.setdefault("brand", []).append(party)
                    filters.setdefault("buyer", []).append(party)
                    if not filters.get("garment") and not filters.get("garment_type"):
                        g = garment_type_from_question(garment_tok) or (
                            "tops" if garment_tok in {"top", "tops"} else garment_tok
                        )
                        filters.setdefault("garment", []).append(g)
                    index += 3
                    continue
                if garment_tok in {"which", "what", "whose", "how"} or not garment_tok:
                    filters.setdefault("brand", []).append(party)
                    filters.setdefault("buyer", []).append(party)
                    index += 2
                    continue
        # "<brand> jackets/tops/…" without a preposition.
        if (
            _is_party_token(token)
            and token not in _GARMENT_TYPE_WORDS
            and index + 1 < len(tokens)
            and (
                tokens[index + 1] in _GARMENT_TYPE_WORDS
                or garment_type_from_question(tokens[index + 1])
                or tokens[index + 1] in {"tops", "top"}
            )
        ):
            filters.setdefault("brand", []).append(token)
            filters.setdefault("buyer", []).append(token)
            g = garment_type_from_question(tokens[index + 1]) or (
                "tops" if tokens[index + 1] in {"top", "tops"} else tokens[index + 1]
            )
            if not filters.get("garment") and not filters.get("garment_type"):
                filters.setdefault("garment", []).append(g)
            index += 2
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
            # "zara colours break down" → brand=zara + group_by=color — never color=zara.
            if _is_party_token(token) and token not in _GARMENT_TYPE_WORDS:
                filters.setdefault("brand", []).append(token)
                if not group_by:
                    group_by = "color"
                index += 2
                end = _is_breakdown_phrase_at(tokens, index)
                if end is not None:
                    index = end
                continue
            # "red colour" / "navy colors" → color filter.
            if token in _COLOR_WORDS or not _is_party_token(token):
                filters.setdefault("color", []).append(token)
                index += 2
                continue

        # "zara prices" / "primark availability" — party before a measure word is the brand.
        nxt = tokens[index + 1] if index + 1 < len(tokens) else ""
        if (
            _is_party_token(token)
            and token not in _GARMENT_TYPE_WORDS
            and (
                nxt in _MEASURE_PROPERTY_WORDS
                or nxt in {"prices", "price", "availability", "stock"}
            )
        ):
            filters.setdefault("brand", []).append(token)
            index += 1
            continue

        if token in _COLOR_WORDS:
            filters.setdefault("color", []).append(token)
        elif token in _GENDER_VALUES and group_by != "gender":
            filters.setdefault("gender", []).append(_GENDER_VALUES[token])
        elif token in _MATERIAL_WORDS:
            filters.setdefault("material", []).append(token)
        elif (
            token in _GARMENT_TYPE_WORDS
            and token not in {"shirt", "shirts", "top", "tops", "short"}  # prefer phrase detect
            and not filters.get("garment")
            and not filters.get("garment_type")
        ):
            # Do not promote a closed lexicon hit alone (unknown English ≠ catalog value).
            # Only keep it when the full-question phrase detector agrees on a product type.
            detected = garment_type_from_question(question)
            det_key = normalize_name(detected or "")
            tok_key = normalize_name(token)
            if detected and (
                tok_key == det_key
                or tok_key.rstrip("s") == det_key
                or det_key.rstrip("s") == tok_key
                or tok_key in det_key
                or det_key in tok_key
            ):
                filters.setdefault("garment", []).append(detected)
        elif re.fullmatch(r"(19|20)\d{2}", token):
            filters.setdefault("season", []).append(token)
        elif token in _FIELD_ALIASES:
            # Dimension names (family, brand, …) are group keys, never garment values.
            index += 1
            continue
        elif group_by:
            # Breakdown / ranking: leftover party token may be the brand
            # ("Primark colours…"). Never vacuum report English into brand=A|B|C.
            if (
                _is_party_token(token)
                and token not in _GARMENT_TYPE_WORDS
                and (not live_brand_keys or normalize_name(token) in live_brand_keys)
            ):
                filters.setdefault("brand", []).append(token)
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
        # Task phrasing must never survive as garment/category on analysis asks.
        filters.pop("garment", None)
        filters.pop("garment_type", None)
        spoken_break_down = any(
            tokens[i] in {"break", "breaks", "broke"}
            and i + 1 < len(tokens)
            and tokens[i + 1] == "down"
            for i in range(len(tokens))
        )
        field_wise = bool(group_by) and (
            "wise" in tokens
            or bool(set(tokens) & _BREAKDOWN_HEADS)
            or "by" in tokens
            or spoken_break_down
        )
        # Full multi-field analysis: modest per-field sample. Focused breakdowns
        # (e.g. "Zara colours break down") keep limit=None so every value is shown —
        # never invent top-5 when the user did not ask for N.
        if limit is None and not field_wise:
            limit = 5
        if all_parties:
            # "analysis all buyers in ecomm" → breakdown of every party, no single-party filter.
            group_by = all_parties
            filters.pop("buyer", None)
            filters.pop("brand", None)
        elif field_wise:
            # Focused analysis: keep analysis=True + group_by so the same engine
            # runs one facet (unique-value breakdown) instead of a count stub.
            pass
        else:
            group_by = None
            party = _analysis_party_name(tokens)
            if party and party not in _SYSTEM_WORDS and party not in _GRAPH_SCOPE_NOISE:
                if live_brand_keys and normalize_name(party) not in live_brand_keys:
                    party = None
            # "full analysis of Primark" — party after "of" may miss keyword cues;
            # fall back to live brand tokens present in the question.
            if not party and live_brand_keys:
                for tok in tokens:
                    key = normalize_name(tok)
                    if key in live_brand_keys and key not in _SYSTEM_WORDS:
                        party = tok
                        break
            if party:
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
        # Drop invented garment leftovers, then restore a real product-type phrase below.
        filters.pop("garment", None)
        filters.pop("garment_type", None)
        named_garment = garment_type_from_question(question)
        if named_garment:
            filters["garment"] = [named_garment]
        # Keep an explicit category if the user named one; drop invented leftovers.
        cat_vals = filters.get("category") or []
        if cat_vals and not _reliable_filter(" ".join(str(v) for v in cat_vals)):
            filters.pop("category", None)
        brand_compare = (
            group_by in {"brand", "buyer"}
            or any(tok in {"brand", "brands", "buyer", "buyers"} for tok in tokens)
            or bool(re.search(r"\b(more|fewer|less|most|least)\b", " ".join(tokens)))
        )
        # Composition/spec compares keep the garment filter — do not force brand breakdown.
        if is_composition_compare_question(question):
            brand_compare = False
        token_set = set(tokens)
        assortment_compare = bool(
            token_set & _ANALYSIS_WORDS
            or token_set & {"assortment", "profile", "profiles", "mix", "overview"}
            or (
                "compare" in token_set
                and not bool(
                    re.search(
                        r"\b(more|fewer|less|most|least|how many|count)\b",
                        " ".join(tokens),
                    )
                )
            )
        )
        if assortment_compare:
            # Side-by-side full attribute profiles (same engine as full analysis).
            analysis = True
            group_by = None
            if limit is None:
                limit = 5
        elif brand_compare and not _is_price_history_question(question):
            # Count-style "who has more" → brand breakdown.
            group_by = group_by if group_by in {"brand", "buyer"} else "brand"

    # Long briefs dump many leftover English tokens into brand under group_by.
    # Real multi-party compares come from _compare_party_names (exactly 2) or live inventory.
    for party_key in ("brand", "buyer"):
        vals = list(filters.get(party_key) or [])
        if len(vals) <= 1:
            continue
        if live_brand_keys:
            kept = [v for v in vals if normalize_name(str(v)) in live_brand_keys]
            if kept:
                filters[party_key] = kept
            else:
                filters.pop(party_key, None)
        elif len(vals) > 2:
            filters.pop(party_key, None)

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
    # Price / availability history asks: drop invented facet noise + assortment flag.
    if _is_price_history_question(question) or _is_availability_history_question(question):
        for junk in ("garment", "garment_type"):
            flat.pop(junk, None)
        analysis = False
        group_by = None

    # Listing / ranking asks that name a product type: keep garment filter.
    # Prefer full-phrase detect (t-shirt) over raw token leftovers (shirts).
    detected_garment = garment_type_from_question(question)
    if detected_garment and (
        group_by in {"brand", "buyer", "category", "material", "color", "garment_type"}
        or bool(
            set(tokens)
            & {
                "have",
                "has",
                "show",
                "list",
                "which",
                "what",
                "most",
                "products",
                "examples",
                "compare",
                "composition",
                "compositions",
                "mix",
                "mixes",
                "common",
                "breakdown",
            }
        )
        or any(flat.get(k) for k in ("garment", "garment_type", "brand", "buyer"))
        or is_composition_compare_question(question)
    ):
        flat["garment"] = detected_garment
        flat.pop("garment_type", None)
    elif not any(flat.get(k) for k in ("garment", "garment_type", "category")):
        pass

    # Multi-attribute product briefs list composition among other fields — not group_by.
    attribute_brief = is_product_attribute_brief_ask(question)
    if attribute_brief:
        group_by = None
        analysis = True
    # "composition mixes" / "material mixes" without a prior group marker.
    if not attribute_brief and group_by is None and re.search(
        r"\b((material|materials|composition|compositions)\s+(mix|mixes|blend|blends)|"
        r"(mix|mixes|blend|blends)\s+of\s+(material|materials))\b",
        " ".join(tokens),
    ):
        group_by = "material"
    if not attribute_brief and group_by is None and re.search(
        r"\b(most common|common)\s+(material|materials|composition|compositions)\b",
        " ".join(tokens),
    ):
        group_by = "material"
    if not attribute_brief and group_by is None and re.search(
        r"\b(material|materials|composition|compositions)\s+(dominate|dominant|breakdown)\b",
        " ".join(tokens),
    ):
        group_by = "material"

    # "most common … garment type" / "which garment types" / "by garment type"
    # Not: "explain using garment_type / subfamily roles" or "counts per source".
    if group_by is None and re.search(
        r"\b(?:"
        r"(?:which|what|list|show|rank|top|most\s+common|common)\s+garment\s+types?\b|"
        r"\bgarment\s+types?\s+(?:are\s+)?(?:most\s+)?common\b|"
        r"\b(?:by|per)\s+garment\s+types?\b|"
        r"\bgarment\s+types?\s+wise\b"
        r")",
        " ".join(tokens),
    ):
        group_by = "garment_type"
    # Construction / fit attribute frequency ("what construction … most common").
    if (
        not attribute_brief
        and group_by is None
        and re.search(
            r"\b(construction|constructions|fit|fits)\b",
            " ".join(tokens),
        )
        and re.search(
            r"\b(common|usually|typical|most|details?|see|seen)\b",
            " ".join(tokens),
        )
    ):
        group_by = (
            "construction"
            if any(t.startswith("construction") for t in tokens)
            else "fit"
        )

    # Ground leftover content tokens onto live kg_stats values (category/garment/…)
    # before inventing a brand — typos like "swweaters" → category=sweater.
    flat = ground_free_tokens_against_stats(
        question,
        flat,
        preferred_graphs=preferred_graphs,
    )
    # "PLMEX Zara …" / "ecom Primark …" — catalog token then party name.
    if not flat.get("brand") and not flat.get("buyer"):
        for index, token in enumerate(tokens[:-1]):
            if token not in _GRAPH_SCOPE_NOISE and token not in _SYSTEM_WORDS:
                continue
            nxt = tokens[index + 1]
            if _is_party_token(nxt) and nxt not in _GARMENT_TYPE_WORDS:
                # Only attach when the next token is a live brand (never a typo garment).
                if live_brand_keys and normalize_name(nxt) not in live_brand_keys:
                    continue
                flat["brand"] = nxt
                flat["buyer"] = nxt
                break
    # Lone party in "how many Primark products…" when no brand was attached yet.
    # Unknown tokens (swweaters) are NOT brands — stats grounding above owns those.
    if not flat.get("brand") and not flat.get("buyer") and live_brand_keys:
        parties = [
            tok
            for tok in tokens
            if _is_party_token(tok)
            and tok not in _GARMENT_TYPE_WORDS
            and tok not in _GRAPH_SCOPE_NOISE
            and tok not in _SYSTEM_WORDS
            and normalize_name(tok) in live_brand_keys
        ]
        if len(parties) == 1:
            flat["brand"] = parties[0]
            flat["buyer"] = parties[0]

    # "about 10" / "show me 10" examples when digit was not already captured after show/list.
    if limit is None:
        about_n = re.search(r"\b(?:about|around|roughly|approx(?:imately)?|top|show|list|give)\s+(\d{1,3})\b", " ".join(tokens))
        if about_n and re.search(r"\b(example|examples|products?|items?|styles?)\b", " ".join(tokens)):
            limit = max(1, min(int(about_n.group(1)), 50))

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


def _strip_app_scope_phrases(question: str) -> str:
    """Drop product/app name phrases so they never become catalog filters."""

    text = str(question or "")
    text = re.sub(r"(?i)\bfashion\s*brains?\b", " ", text)
    text = re.sub(r"(?i)\bfashion\s*gpts?\b", " ", text)
    text = re.sub(r"(?i)\bbettex\b", " ", text)
    return text


def ground_free_tokens_against_stats(
    question: str,
    filters: dict | None = None,
    *,
    preferred_graphs: list[str] | None = None,
    generated=None,
) -> dict:
    """Map free question tokens onto live kg_stats values (schema-driven).

    Used when the keyword planner missed a typo/near-miss ("swweaters" → sweater)
    so we ground against indexed catalog values instead of inventing a brand.
    Prefer type/hierarchy dims over brand/buyer so typos do not become parties.
    """

    out = {k: v for k, v in (filters or {}).items() if v not in (None, "")}
    # Distinct-facet asks ("how many brands…") must not invent garment/brand values
    # from leftover English (e.g. "fashion" → "young fashion").
    try:
        facet = _facet_entity_count(question)
        if facet and str(facet).casefold() not in {"product", "products", "item", "items"}:
            return out
    except Exception:
        pass
    if any(
        out.get(k)
        for k in (
            "garment",
            "garment_type",
            "category",
            "family",
            "subfamily",
            "brand",
            "buyer",
            "color",
            "material",
            "gender",
        )
    ):
        return out
    try:
        from knowledge_graph.retrieval.kg_stats import find_value_in_stats, stats_available
    except Exception:
        return out

    gids: list[str] = []
    for gid in preferred_graphs or []:
        if gid and str(gid) not in gids:
            gids.append(str(gid))
    # Prefer graphs the question named (plm / ecom); avoid importing QU (cycle risk).
    scope_question = _strip_app_scope_phrases(question)
    low = normalize_name(scope_question)
    if re.search(r"\b(plm|plmex)\b", low) and "plmex_products" not in gids:
        gids.insert(0, "plmex_products")
    if re.search(r"\b(ecom|ecomm|ecommerce|commerce)\b", low) and "ecom_products" not in gids:
        gids.insert(0, "ecom_products")
    if not gids:
        gids = ["plmex_products", "ecom_products"]

    prefer = (
        "category",
        "garment_type",
        "family",
        "subfamily",
        "plm_main_category",
        "plm_sub_category",
        "plm_sub_sub_category",
        "section",
    )
    type_attrs = set(prefer)
    # Report / ranking English — never treat as a misspelled product type.
    # ("a short note", "top constructions", "fibre blends dominate", …)
    ground_skip = (
        _SYSTEM_WORDS
        | _GRAPH_SCOPE_NOISE
        | _GRAPH_SCORE_STOPWORDS
        | _FILTER_NOISE
        | _GARMENT_TYPE_WORDS
        | _words(
            "short thin long full every note notes fibre fiber blend blends "
            "composition compositions construction constructions concentrated "
            "produce assortment scrape report reports mix mixes dominate dominant "
            "material materials commerce scrape brands brand fashion brain"
        )
    )
    tokens = [
        tok
        for tok in _question_tokens(scope_question)
        if tok
        and len(tok) >= 4
        and _is_party_token(tok)  # leftover content word, not grammar
        and tok not in ground_skip
    ]
    best: tuple[float, str, str, str] | None = None  # score, gid, attr, canon
    for tok in tokens:
        tok_key = normalize_name(tok)
        for gid in gids:
            if not stats_available(gid, generated):
                continue
            hit = find_value_in_stats(
                gid, tok, generated=generated, prefer_attrs=prefer, min_score=0.86
            )
            if hit is None:
                continue
            attr, canon, _n = hit
            if attr not in type_attrs:
                continue
            canon_key = normalize_name(canon)
            # Exact / simple-plural hits belong to enrich/lexicon — this helper is
            # for typos ("swweaters"). Exact "short"↔category short is report noise.
            if tok_key == canon_key or tok_key.rstrip("s") == canon_key.rstrip("s"):
                continue
            from difflib import SequenceMatcher

            score = float(SequenceMatcher(None, tok_key, canon_key).ratio())
            if best is None or score > best[0]:
                best = (score, gid, attr, canon)
    if best is None:
        return out
    _score, _gid, attr, canon = best
    if attr in {"category", "garment_type", "family", "subfamily"}:
        # Planner / retrieve treat product-type as garment; stats resolve maps later.
        out["garment"] = canon
    else:
        out[attr] = canon
    if _gid and _gid not in (preferred_graphs or []):
        # Caller may pick preferred_graphs from the plan; leave that to preferred_graphs_from_question.
        pass
    return out


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
    # Prefer kg_stats brand inventory too (covers graphs with no brand-node cache).
    try:
        names.extend(live_party_values(platform, "brand", graph_id=graph_id, include_brand_cache=False))
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


def live_party_values(
    platform,
    attribute: str,
    *,
    graph_id: str | None = None,
    include_brand_cache: bool = True,
) -> list[str]:
    """Live facet values (brand/buyer/…) from kg_stats (+ brand cache for brand).

    Used to reject invented party filters like buyer=possible / brand=impossible.
    """

    attr = str(attribute or "").strip().casefold().replace(" ", "_")
    if not attr or platform is None:
        return []
    names: list[str] = []
    if attr == "brand" and include_brand_cache:
        try:
            # Avoid recursion: brand cache path without kg_stats re-entry.
            from fashion_brain.planning.runtime_schema import get_runtime_schema, resolve_product_graph_id
            from fashion_brain.planning.schema_cache import get_brands

            gid = graph_id or resolve_product_graph_id(platform)
            if gid:
                for row in get_brands(platform, gid, None) or []:
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
        except Exception:
            pass
    try:
        from knowledge_graph.retrieval.ecom import registry_product_graphs
        from knowledge_graph.retrieval.kg_stats import dimension_bucket

        generated = getattr(getattr(platform, "settings", None), "generated", None)
        gids: list[str] = []
        if graph_id:
            gids.append(str(graph_id))
        for gid in registry_product_graphs(platform) or []:
            if gid and str(gid) not in gids:
                gids.append(str(gid))
        for gid in gids:
            bucket = dimension_bucket(gid, attr, generated) or {}
            for key in bucket:
                text = str(key or "").strip()
                if text:
                    names.append(text)
    except Exception:
        pass
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        key = normalize_name(name)
        if key and key not in seen and not _is_grammar_party_name(key):
            seen.add(key)
            out.append(name)
    return out


def match_live_party_value(raw: str, allowed: list[str] | set[str] | None) -> str | None:
    """Return a canonical live party name if ``raw`` matches metadata; else None."""

    want = normalize_name(str(raw or ""))
    if not want or _is_grammar_party_name(want):
        return None
    if not allowed:
        return None
    by_norm = {normalize_name(str(n)): str(n) for n in allowed if n}
    if want in by_norm:
        return by_norm[want]
    compact = want.replace(" ", "").replace("-", "").replace("_", "")
    for key, surface in by_norm.items():
        if key.replace(" ", "").replace("-", "").replace("_", "") == compact:
            return surface
    return None


def ground_filters_against_live_metadata(
    platform,
    filters: dict | None,
    *,
    question: str = "",
) -> dict:
    """Grammar-sanitize then keep only categorical values present in live metadata.

    Applies to every strict field (brand, buyer, color, material, availability, …),
    not only parties. Call after LLM/keyword extraction so invented values never
    reach Scope or Cypher.
    """

    cleaned = sanitize_party_filters(dict(filters or {}))
    if platform is None:
        return cleaned
    try:
        from fashion_brain.planning.metadata_resolver import (
            ground_filters_against_live_metadata as _meta_ground,
        )

        return _meta_ground(platform, cleaned, question=question)
    except Exception:
        LOGGER.debug("ground_filters_against_live_metadata skipped", exc_info=True)
        return cleaned


def ground_party_filters_against_metadata(
    filters: dict | None,
    platform,
    *,
    graph_id: str | None = None,
) -> dict:
    """Keep brand/buyer only when the value exists in live KG metadata.

    Grammar sanitize first, then metadata check. Unknown tokens (possible,
    impossible, random English) are dropped — not treated as parties.

    On PLM graphs, when a party exists as buyer (and optionally also as brand),
    prefer buyer — e.g. Matalan sweaters for women (P13) are buyer-scoped.
    On e-commerce graphs, prefer brand when both match.
    """

    out = sanitize_party_filters(dict(filters or {}))
    if platform is None:
        return out
    for key in ("brand", "buyer"):
        raw = out.get(key)
        if not raw:
            continue
        allowed = live_party_values(platform, key, graph_id=graph_id)
        if not allowed:
            # Metadata unavailable for this dimension — do not invent; keep
            # grammar-clean values only (already sanitized).
            LOGGER.debug("party metadata empty for %s; skipped live grounding", key)
            continue
        parts = split_multi_party_filter(str(raw)) or [str(raw)]
        kept: list[str] = []
        for part in parts:
            hit = match_live_party_value(part, allowed)
            if hit:
                kept.append(hit)
            else:
                LOGGER.info(
                    "dropped ungrounded party filter %s=%s (not in live metadata)",
                    key,
                    part,
                )
        if not kept:
            out.pop(key, None)
        elif len(kept) == 1:
            out[key] = kept[0]
        else:
            out[key] = " | ".join(dict.fromkeys(kept))

    # Cross-role: value set as brand but only live as buyer → buyer (P13 Matalan).
    # When both roles match on PLM, prefer buyer; on e-commerce prefer brand.
    # Never role-flip a multi-party compare string ("Primark | Zara") as one token.
    gid = str(graph_id or "").casefold()
    brand_raw = out.get("brand")
    buyer_raw = out.get("buyer")
    plm_like = "plmex" in gid or gid in {"plm", "plm records"} or "plm" in gid
    ecom_like = "ecom" in gid or "commerce" in gid
    multi_brand = len(split_multi_party_filter(str(brand_raw or ""))) >= 2
    multi_buyer = len(split_multi_party_filter(str(buyer_raw or ""))) >= 2
    if brand_raw and not buyer_raw and not multi_brand:
        buyer_allowed = live_party_values(platform, "buyer", graph_id=graph_id) or []
        brand_allowed = live_party_values(platform, "brand", graph_id=graph_id) or []
        buyer_hit = match_live_party_value(str(brand_raw), buyer_allowed)
        brand_hit = match_live_party_value(str(brand_raw), brand_allowed)
        if buyer_hit and (not brand_hit or plm_like):
            out["buyer"] = buyer_hit
            out.pop("brand", None)
            LOGGER.info(
                "party role flip brand→buyer value=%s graph=%s brand_hit=%s",
                buyer_hit,
                graph_id,
                bool(brand_hit),
            )
    elif (
        brand_raw
        and buyer_raw
        and not multi_brand
        and not multi_buyer
        and normalize_name(str(brand_raw)) == normalize_name(str(buyer_raw))
    ):
        if ecom_like:
            out.pop("buyer", None)
        elif plm_like:
            out.pop("brand", None)
    return out


def ensure_compare_party_filters(filters: dict | None, question: str = "") -> dict:
    """Keep multi-brand compare scope (Primark | Zara). Recover from the question if dropped."""

    out = dict(filters or {})
    parties = split_multi_party_filter(str(out.get("brand") or out.get("buyer") or ""))
    if len(parties) >= 2:
        return out
    named = _compare_party_names(question or "")
    if len(named) >= 2:
        # Prefer brand encoding; graph aligner remaps per catalog later.
        out["brand"] = " | ".join(named)
        out.pop("buyer", None)
        LOGGER.info("recovered compare parties from question: %s", out["brand"])
    return out


# ISO currency codes / money words — never brand or buyer (e.g. user reply "USD").
_CURRENCY_PARTY_NOISE = frozenset(
    {
        "usd",
        "gbp",
        "eur",
        "sek",
        "dkk",
        "nok",
        "pln",
        "czk",
        "chf",
        "jpy",
        "cny",
        "aud",
        "cad",
        "dollar",
        "dollars",
        "pound",
        "pounds",
        "sterling",
        "euro",
        "euros",
        "currency",
        "currencies",
        "both",
    }
)


def _is_grammar_party_name(name: str) -> bool:
    """Reject catalog values that collide with question grammar (e.g. buyer 'frequently')."""

    key = normalize_name(str(name or ""))
    if not key:
        return True
    tokens = set(key.split())
    if not tokens:
        return True
    # Bare currency reply after a price-trend clarify must never become buyer=usd.
    if key in _CURRENCY_PARTY_NOISE or tokens <= _CURRENCY_PARTY_NOISE:
        return True
    grammar = (
        _COUNT_STOPWORDS
        | _GRAPH_SCORE_STOPWORDS
        | _SYSTEM_WORDS
        | _GRAPH_SCOPE_NOISE
        | _MAKE_ADVERBS
        | _MAKE_VERBS
        | _TASK_WORDS
        | _MODAL_WORDS
        | _METRIC_OPERATOR_WORDS
        | _ANALYSIS_WORDS
        | _AGENT_UI_NOISE
        | _RANK_WORDS
        | _MEASURE_PROPERTY_WORDS
        | _AVAILABILITY_STATUS_WORDS
        | _FILTER_NOISE
        | _CURRENCY_PARTY_NOISE
    )
    # Whole value is stopword grammar — never a real brand/buyer filter from the question.
    if tokens <= grammar:
        return True
    if len(tokens) == 1 and next(iter(tokens)) in grammar:
        return True
    return False


def live_brands_mentioned(question: str, brand_names: list[str] | None) -> list[str]:
    """Return live Brand names mentioned in the question (longest match first, de-duped)."""

    text = normalize_name(question or "")
    if not text or not brand_names:
        return []
    ranked = sorted(
        {normalize_name(str(n)): str(n) for n in brand_names if n}.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    hits: list[str] = []
    seen: set[str] = set()
    for key, surface in ranked:
        if len(key) < 2 or _is_grammar_party_name(key) or key in seen:
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", text):
            hits.append(surface)
            seen.add(key)
    return hits


def question_has_party_deixis(question: str) -> bool:
    """True when the ask refers to a prior party ('of them', 'those', 'same brand', …)."""

    return bool(
        re.search(
            r"\b(them|those|these|same|it|again)\b|\bof\s+them\b|\bfor\s+them\b|\bthat\s+brand\b",
            question or "",
            flags=re.IGNORECASE,
        )
    )


def ground_filters_from_live_brands(
    question: str,
    filters: dict | None,
    brand_names: list[str] | None,
    *,
    platform=None,
    graph_id: str | None = None,
) -> dict:
    """Link question tokens to live Brand nodes — not a hardcoded retailer list.

    When the latest question names one live brand, that party **replaces** any
    prior multi-party sticky filter (e.g. after a Primark vs Zara compare).
    Brand/buyer values that are not in live metadata are dropped.
    """

    # Drop grammar leftovers (e.g. brand=tell from "tell me") before treating
    # the filter as already grounded. Real parties still match live Brand nodes below.
    out = sanitize_party_filters(promote_live_brand_filters(filters, brand_names))

    def _is_placeholder_party(value: str) -> bool:
        low = normalize_name(str(value or ""))
        return bool(
            re.search(r"\b(missing|unspecified|unknown|null|none)\b", low)
            or re.fullmatch(r"__\w+__", str(value or "").strip())
        )

    if platform is not None:
        # Keep explicit missing/placeholder brands for honest zero answers.
        placeholders = {
            k: v
            for k, v in out.items()
            if k in {"brand", "buyer"} and _is_placeholder_party(str(v))
        }
        out = ground_party_filters_against_metadata(out, platform, graph_id=graph_id)
        out.update(placeholders)
    else:
        # Still reject brand values that are not in the provided live brand list.
        allowed = list(brand_names or [])
        if allowed and out.get("brand"):
            parts = split_multi_party_filter(str(out["brand"])) or [str(out["brand"])]
            kept = []
            for part in parts:
                if _is_placeholder_party(part):
                    kept.append(part)
                    continue
                hit = match_live_party_value(part, allowed)
                if hit:
                    kept.append(hit)
            if not kept:
                out.pop("brand", None)
            elif len(kept) == 1:
                out["brand"] = kept[0]
            else:
                out["brand"] = " | ".join(dict.fromkeys(kept))
    hits = live_brands_mentioned(question, brand_names)
    if len(hits) == 1:
        explicit_buyer = bool(re.search(r"\bbuyers?\b", question, re.IGNORECASE)) and not bool(
            re.search(r"\bbrands?\b", question, re.IGNORECASE)
        )
        party_key = "buyer" if explicit_buyer else "brand"
        other_key = "brand" if explicit_buyer else "buyer"
        out[party_key] = hits[0]
        out.pop(other_key, None)
        if platform is not None:
            out = ground_party_filters_against_metadata(out, platform, graph_id=graph_id)
        return out
    if len(hits) >= 2:
        out["brand"] = " | ".join(hits[:4])
        out.pop("buyer", None)
        if platform is not None:
            out = ground_party_filters_against_metadata(out, platform, graph_id=graph_id)
        return out
    # No party named in this question — keep prior sanitized party (of-them carry).
    return out


def enrich_scope_from_question(
    question: str,
    filters: dict | None = None,
    *,
    brand_names: list[str] | None = None,
    platform=None,
) -> tuple[dict, str | None]:
    """Dynamic scope for the latest turn: live brands + keyword facets + deixis carry.

    Returns ``(filters, group_by)``. Never hard-codes retailer names.
    """

    # "how many brands in Fashion Brain" is a distinct-facet count — no product filters.
    try:
        facet = _facet_entity_count(question)
        if facet and str(facet).casefold() not in {"product", "products", "item", "items"}:
            plan = _catalog_plan(_strip_app_scope_phrases(question))
            return {}, plan.get("group_by")
    except Exception:
        pass

    prior = sanitize_party_filters(dict(filters or {}))
    plan = _catalog_plan(_strip_app_scope_phrases(question))
    plan_filters = sanitize_party_filters(dict(plan.get("filters") or {}))
    out = dict(prior)

    # Facets / parties mentioned in THIS question always win.
    for key in (
        "garment",
        "garment_type",
        "category",
        "color",
        "gender",
        "material",
        "season",
        "family",
        "subfamily",
        "brand",
        "buyer",
        "availability",
    ):
        if plan_filters.get(key):
            out[key] = plan_filters[key]

    names = list(brand_names or [])
    if not names and platform is not None:
        try:
            names = live_brand_names(platform)
        except Exception:
            names = []
    out = ground_filters_from_live_brands(
        question, out, names, platform=platform
    )
    if platform is not None:
        def _placeholder_party(value: str) -> bool:
            low = normalize_name(str(value or ""))
            return bool(
                re.search(r"\b(missing|unspecified|unknown|null|none)\b", low)
                or re.fullmatch(r"__\w+__", str(value or "").strip())
            )

        placeholders = {
            k: v
            for k, v in out.items()
            if k in {"brand", "buyer"} and _placeholder_party(str(v))
        }
        out = ground_party_filters_against_metadata(out, platform)
        out.update(placeholders)

    hits = live_brands_mentioned(question, names)
    # An explicitly named party starts a fresh scope; keep only facets named this turn.
    if hits:
        for key in ("garment", "garment_type", "category", "color", "material", "gender", "season", "family", "subfamily"):
            if not plan_filters.get(key):
                out.pop(key, None)

    # Broader analysis / overall totals: never keep prior restrictive facets.
    low_q = normalize_name(question or "")
    broader_scope = bool(
        _is_analysis(question)
        or re.search(r"\b(overall|in total|altogether|full analysis|complete analysis)\b", low_q)
    )
    if broader_scope:
        for key in ("garment", "garment_type", "category", "color", "material", "gender", "season", "family", "subfamily"):
            if not plan_filters.get(key):
                out.pop(key, None)

    # Whole-catalog / dataset totals: never keep a prior brand/buyer.
    if is_catalog_dataset_total_ask(question):
        if not plan_filters.get("brand"):
            out.pop("brand", None)
        if not plan_filters.get("buyer"):
            out.pop("buyer", None)

    # Deixis / short follow-up with no new party: keep prior brand/buyer.
    if not hits and not (out.get("brand") or out.get("buyer")):
        if question_has_party_deixis(question) and (prior.get("brand") or prior.get("buyer")):
            if prior.get("brand"):
                out["brand"] = prior["brand"]
            if prior.get("buyer"):
                out["buyer"] = prior["buyer"]


    group_by = plan.get("group_by")
    # Multi-attribute product briefs list composition/colour/… as report sections —
    # never collapse that list into a single group_by=material (or colour).
    # Spec / season-filter asks: "Winter season product specification" is NOT group_by=season.
    if is_product_attribute_brief_ask(question) or _is_suggestion(question):
        group_by = None
    elif not group_by:
        # Dimension words in the ask (category / color / …) from live-ish aliases.
        low = normalize_name(question or "")
        season_is_filter = bool(
            plan_filters.get("season_name")
            or plan_filters.get("season")
            or plan.get("season_name")
            or re.search(r"\b(winter|spring|summer|autumn|fall)\b", low)
        )
        for dim in (
            "category",
            "garment_type",
            "color",
            "gender",
            "material",
            "composition",
            "season",
            "family",
            "subfamily",
            "brand",
            "buyer",
        ):
            if dim == "season" and season_is_filter:
                continue
            label = dim.replace("_", " ")
            if re.search(rf"\b{re.escape(label)}\b", low) or re.search(rf"\b{re.escape(dim)}\b", low):
                group_by = "material" if dim == "composition" else dim
                break
    # Dimension-only follow-up ("explore category"): clear stale facets not in this ask.
    if group_by and not any(
        plan_filters.get(k) for k in ("garment", "garment_type", "category", "color", "material")
    ):
        for key in ("garment", "garment_type", "color", "material", "gender"):
            if key != group_by:
                out.pop(key, None)

    out = sanitize_party_filters(out)
    return out, (str(group_by) if group_by else None)


def sanitize_party_filters(filters: dict) -> dict:
    """Brand/buyer names must not also appear as garment/category filters."""

    cleaned = dict(filters or {})
    for key in ("brand", "buyer"):
        value = cleaned.get(key)
        if not value:
            continue
        parts = split_multi_party_filter(str(value)) or [str(value)]
        kept = [
            part
            for part in parts
            if part
            and not _is_grammar_party_name(part)
            and _reliable_filter(part)
            and normalize_name(part) not in _GARMENT_TYPE_WORDS
            and not garment_type_from_question(part)
        ]
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
    state: dict | None = None,
) -> dict:
    """The model decides, code validates. Falls back to the keyword planner.

    ``carry`` is the agent's references_prior_chat decision. When omitted, the planner
    model decides it; keyword fallback never invents history carry.

    Always grounds filters against each registered KG's unique columns and live values.
    """

    from fashion_brain.planning.kg_catalog import build_field_catalog, ground_filters_with_catalog
    from fashion_brain.planning.metadata_resolver import (
        build_value_index,
        check_understanding,
        repair_filters,
    )

    catalog = build_field_catalog(platform, question)
    brand_names = live_brand_names(platform)
    model_plan = _plan_with_model(platform, question, history, catalog=catalog)
    if model_plan is None:
        LOGGER.info("catalog planner keyword fallback used (model plan unavailable)")
        plan = resolve_catalog_plan(
            question, history, carry=bool(carry), brand_names=brand_names
        )
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
        chk = check_understanding(platform, question, history, plan, state)
        if chk.proceed:
            plan["filters"] = chk.filters
        else:
            plan["filters"] = chk.filters
            plan["needs_clarification"] = True
            plan["clarify_text"] = chk.clarify_text
            plan["clarify_options"] = chk.options
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
    # User said it (_ground_filters) + it exists in live metadata for that field.
    try:
        model_plan["filters"] = ground_filters_against_live_metadata(
            platform, model_plan.get("filters") or {}, question=question
        )
    except Exception:
        LOGGER.debug("meta ground after _ground_filters skipped", exc_info=True)
        try:
            idx_fields = list(
                dict.fromkeys(
                    [
                        *(catalog.all_attributes() or []),
                        *(model_plan.get("filters") or {}).keys(),
                    ]
                )
            )
            index = build_value_index(platform, idx_fields) if idx_fields else {}
            if index:
                model_plan["filters"], _unresolved = repair_filters(
                    model_plan["filters"], index
                )
        except Exception:
            LOGGER.debug("repair_filters after _ground_filters skipped", exc_info=True)
    _promote_type_filters(model_plan["filters"])
    model_plan["filters"] = sanitize_party_filters(model_plan["filters"])
    _normalize_named_season(model_plan)
    keyword = resolve_catalog_plan(
        question, history, carry=carry, brand_names=brand_names
    )
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
        model_plan["analysis"] = True
        field_wise = bool(keyword.get("group_by")) or bool(model_plan.get("group_by"))
        if field_wise:
            # Focused: one facet via the same analytical engine.
            model_plan["group_by"] = keyword.get("group_by") or model_plan.get("group_by")
            # Leave limit unset unless the user/model named an explicit N.
        else:
            # Full: every facet with distinct values — clear single-field group_by.
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
    chk = check_understanding(platform, question, history, model_plan, state)
    if chk.proceed:
        model_plan["filters"] = chk.filters
    else:
        model_plan["filters"] = chk.filters
        model_plan["needs_clarification"] = True
        model_plan["clarify_text"] = chk.clarify_text
        model_plan["clarify_options"] = chk.options
    return model_plan


def _plan_timeout(seconds: float | None) -> float | None:
    if seconds is None or seconds <= 0:
        return 20.0
    return min(20.0, float(seconds))


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
    # Focused analysis keeps both analysis=True and group_by (same engine, one facet).
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
        # Keep the typed value (e.g. zara, c&a) so count filters still apply —
        # never delete a short brand just because fuzzy match is unsafe.
        if len(typed_key) < 5:
            continue
        nearest = _nearest_name(typed, names)
        if nearest is None:
            continue
        # Reject weak fuzzy hits for unknown parties (tesco → random catalog name).
        typed_nn = normalize_name(typed)
        near_nn = normalize_name(nearest)
        if not typed_nn or not near_nn:
            continue
        if typed_nn not in near_nn and near_nn not in typed_nn:
            # Keep the user's spelling; retrieve should return empty/honest miss.
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
        lines.append("The PLMex data contains 0 products.")
    elif filters or group_by:
        matched, total = platform.products.search(principal, limit=5, **filters)
        phrase = _filter_phrase(filters)
        lines.append(f"The PLMex data contains {total} products" + (f" {phrase}." if phrase else "."))
        if group_by:
            groups = platform.products.count_by(principal, group_by, **filters)
            label = _GROUP_LABELS.get(group_by, group_by)
            if plan.get("limit"):
                lines.extend(_ranking_lines(groups, label, int(plan["limit"])))
            else:
                lines.append(f"{label} counts for that selection: {_group_text(groups)}.")
        evidence.extend(_evidence_from_product(envelope) for envelope in matched)
    else:
        lines.append(f"The PLMex data contains {visible} products.")
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
        lines.append(f"The PLMex data has no styles for {buyer}, so there is no specification to build.")
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
