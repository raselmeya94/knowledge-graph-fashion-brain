"""Answer card → writer LLM → validator → source footer.

Retrieval builds an AnswerCard (structured data + full tables/findings).
The LLM writes rich, well-presented prose from it with no artificial length cap.
Code validates accuracy, retries once, then appends the footer from the card.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

LOGGER = logging.getLogger("fashion_brain.answer_card")

SOURCE_LABELS = {
    "plmex_products": "PLM records",
    "ecom_products": "e-commerce listings",
    "own_products": "our own product data",
}

INTENT_BY_ROUTE = {
    "product_count": "count",
    "kg_stats": "count",
    "count": "count",
    "breakdown": "breakdown",
    "ranking": "ranking",
    "list": "examples",
    "products": "analysis",
    "analysis": "analysis",
    "analytical_tool": "analysis",
    "product_details": "spec_brief",
    "specification": "spec_brief",
    "product_specification": "spec_brief",
    "product_spec": "spec_brief",
    "price_trend": "price_trend",
    "price_history": "price_history",
    "trending": "price_trend",
    "clarification": "clarify",
    "shared_products": "compare",
    "compare": "compare",
    "comparison": "compare",
}

FILTER_SYNONYMS = {
    "instock": ["in stock", "available"],
    "outofstock": ["out of stock", "unavailable"],
    "women": ["women", "women's", "womens", "ladies"],
    "men": ["men", "men's", "mens"],
    "kids": ["kids", "kid's", "children"],
    "sweater": ["sweater", "sweaters"],
    "sweaters": ["sweater", "sweaters"],
}

BANNED = [
    r"knowledge\s*graph",
    # Do not flag measurement units like "kg" before a number (e.g. "12 kg").
    r"\bkg\b(?!\s*\d)",
    r"ecom_products",
    r"plmex_products",
    r"\bplmex\b",
    r"catalog\s*[12]",
    r"source\s*[12]\b",
    r"\bnull\b",
    r"leading value\s*:",
    r"distinct values\s*:",
    r"one visualization per attribute",
    r"\[?\s*chart\s+placeholder\s*\]?",
    r"a typical online example",
    r"recorded\s+[\d,]+\s+times",
    r"by\s+a\s+gap\s+of",
    r"lead(?:s)?\s+.{0,60}?\s+by\s+a\s+gap",
    r"the\s+returned\s+(?:results|breakdowns)\s+cover",
    r"do\s+not\s+have\s+the\s+necessary\s+data",
    r"no\s+totals?\s+or\s+breakdown",
    r"no\s+totals?\s+(?:or|and)\s+breakdown\s+metrics?\s+were\s+provided",
]

_NULLISH = {"", "—", "-", "–", "null", "none", "n/a", "na", "unspecified"}
_MALFORMED_PRICE_RE = re.compile(r"(?i)(?:£|€|\$)\s*(?:19|20)\d{2}\b|(?:£|€|\$)\s*$")
_ROBOTIC_COUNT_RE = re.compile(
    r"(?i)there\s+are\s+(?P<n>[\d,]+)\s+products(?P<phrase>\s+(?:with|in)\s+.+?)?(?=\.|$)"
)
_ROBOTIC_OPENER_RE = re.compile(r"(?is)^\s*(?:we\s+found\b|there\s+are\s+[\d,]+)")
_WE_FOUND_OPENER_RE = re.compile(r"(?is)^\s*we\s+found\b")
_SECTION_HEAD_RE = re.compile(r"^(#{2,3})\s+(.+?)\s*$")
_PRODUCTS_EQ_RE = re.compile(r"(?i)products?\s*=\s*(\d[\d,]*)")

# No upper word-count cap for any intent — depth and clarity guide length.
_INTENT_WORD_CAPS: dict[str, int] = {}

# Minimum depth threshold for analytical/structured briefings when rich card data exists.
_ANALYSIS_MIN_WORDS = 80

INTENT_HINTS = {
    "count": (
        "INTENT HINT: SIMPLE PRODUCT COUNT — answer in 1–3 short sentences. "
        "Lead with the exact count and every active filter (e.g. '2,292 products with color white "
        "in PLM records.'). If two sources exist, give both counts in one breath. "
        "Do NOT write Assortment Overview, Key Metrics, Product Distribution, Merchandising Reading, "
        "or any breakdown/ranking tables. Do NOT quote competing facet tops (other colours, genders, "
        "brands) from card.context — those steal the headline number. "
        "Ignore optional related-facts context unless the user asked for more detail. "
        "Do not invent beyond the card."
    ),
    "breakdown": (
        "INTENT HINT: This is a focused assortment analysis (one dimension). "
        "Lead with the headline total and dominant values (counts + shares). "
        "Present a clean Markdown table of unique values from card.tables / breakdown, "
        "cover each source separately if multi-source, then short merchandising takeaways "
        "and any data limitations. Same quality bar as a full analysis — just one facet."
    ),
    "ranking": (
        "INTENT HINT: Lead with the top-ranked leader and key insight, then present a clean "
        "Markdown table with the ranked items (all requested N rows, or all available rows if fewer), "
        "followed by a concise takeaway highlighting concentration or notable standouts."
    ),
    "compare": (
        "INTENT HINT: Present a complete, side-by-side comparison report with clear headings:\n"
        "1) **Executive Summary & Scope** — brands/sources/filters resolved\n"
        "2) **Side-by-Side Metrics** — each party/source's totals and key KPIs (never merge distinct brands into one total)\n"
        "3) **Mix & Pattern Breakdown** — clean Markdown tables comparing dominant categories, materials, colours, pricing, or availability from card.tables / card.findings\n"
        "4) **Merchandising Reading** — what the differences show (strictly grounded in the data)\n"
        "5) **Data Notes & Limitations** — missing facets, currency differences, or source gaps.\n"
        "No length limit; prioritize completeness and readability."
    ),
    "examples": (
        "INTENT HINT: Open with a clear summary (how many examples are shown and the overall total if present). "
        "Present a well-formatted Markdown table including all available clean attributes (Name, Price, Availability, "
        "and Brand/Category/Material if present in card.tables or card.findings). Note briefly if fewer usable "
        "items exist than requested."
    ),
    "analysis": (
        "INTENT HINT: PRODUCT ANALYSIS REPORT — not a thin assortment summary. "
        "OPENING (required): start with 1–2 natural sentences a buyer would read — name the brand/scope, "
        "headline product count, and price cue if present. Do NOT open with labeled blocks like "
        "'### Scope', '### Core metrics', or raw lines such as 'metric: Average price=…'.\n"
        "Full ask → cover EVERY attribute field in the card with its own heading and value table "
        "(unique values + counts + shares). Do NOT collapse to a single 'leading values' table. "
        "Focused ask (one dimension in filters/group_by) → deep-dive that field only, same depth.\n"
        "After the opening prose, use:\n"
        "1) **Attribute breakdown** — one ### section per field with its full value table "
        "(preserve ```chart fences when present)\n"
        "2) **What stands out** — short merchandising reading grounded in the tables\n"
        "3) **Notes** — missing attributes, outliers, multi-source caveats when needed.\n"
        "Compare (brand=A|B): one full block per brand, then a short contrast — never merge totals.\n"
        "CRITICAL: Attribute names (Category, Material, Color, Department, …) are NOT brands. "
        "Never write 'side-by-side comparison for Category vs Material'."
    ),
    "spec_brief": (
        "INTENT HINT: NEW PRODUCT SPECIFICATION (buyer intelligence) — not a raw mode dump. "
        "Write a professional report with:\n"
        "1) **Executive Summary** — party, season, objective, how many historical products informed this\n"
        "2) **Historical Data Overview** — short (scope + source)\n"
        "3) **Key Attribute Findings** — compact table of the most important historical leaders only\n"
        "4) **Important Patterns** — a few bullets on compatible combinations\n"
        "5) **Proposed New Product Specification** — clean Field | Proposed value | Evidence basis | Confidence table "
        "(or Field: value sheet). Identity = Pending PLMEX creation/assignment. Never reuse historical IDs.\n"
        "6) **Buyer Confirmation Requirements** — pending / weak-evidence fields\n"
        "7) **Final Recommendations** — short next steps.\n"
        "CRITICAL: Do NOT paste the raw Decision Engine / long reason walls. "
        "A counts-only historical dump without a proposed sheet FAILS this intent."
    ),
    "report": (
        "INTENT HINT: FULL MERCHANDISING REPORT — same comprehensive structure as analysis: "
        "Executive Summary → Scope & Metrics → Attribute Breakdown Tables → Business Reading → Limitations. "
        "No length limit."
    ),
    "price_trend": (
        "INTENT HINT: COMPREHENSIVE PRICE TREND REPORT — no length limit; NEVER truncate mid-table. "
        "Preserve EVERY section present in card.findings (metrics, period table, ```chart fence, "
        "drivers, example moves). Use clear headings:\n"
        "1) **Trend Headline** — overall price direction, magnitude (%), event counts, products, direction mix\n"
        "2) **Period / Segment Breakdown** — full Markdown period table from findings (all rows shown)\n"
        "3) **Chart** — copy the complete ```chart … ``` block from findings unchanged when present\n"
        "4) **Notable Price Movers & Examples** — keep every example product line from findings\n"
        "5) **Data Caveats** — currency switches / same-currency notes from card.caveats / findings.\n"
        "Do not stop after the first period row. Do not omit the chart or examples."
    ),
    "price_history": (
        "INTENT HINT: COMPREHENSIVE PRICE HISTORY REPORT — same rich structure as price_trend: "
        "Headline movement → Chronological / Segment Table → Notable Examples → Currency & Data Caveats. "
        "Never mix currencies."
    ),
    "follow_up": (
        "INTENT HINT: Answer directly and naturally while briefly restating the carried scope "
        "(e.g., \"Among those Primark sweaters, 30 are red.\"). Include any supporting table, "
        "breakdown, or context from the card so the follow-up is genuinely helpful."
    ),
    "clarify": (
        "INTENT HINT: Ask one clear, friendly clarification question when scope is truly missing."
    ),
    "chitchat": (
        "INTENT HINT: Reply in 1-2 friendly sentences explaining what you can help with; no footer."
    ),
}

WRITER_PROMPT = """
# ROLE: Fashion Brain — Universal Merchandising & Product Analytics Writer

You are the final response agent for Fashion Brain, an intelligent fashion product,
merchandising, specification, and business analytics assistant. Your audience includes
merchandisers, buyers, designers, sourcing teams, and business decision-makers.

Transform the user's question and the Answer Card (including its tables, findings,
context, and caveats) into a clear, well-structured, accurate, and genuinely helpful
response. There is NO arbitrary line or word limit — make the presentation as thorough,
scannable, and satisfying as the evidence supports.

## 1. INPUT AUTHORITY
Use ONLY facts present in the Answer Card (`results`, `tables`, `context`, `derived`,
`findings`, `data_notes`, `caveats`, `filters`). Do NOT invent numbers, brands,
products, trends, or external market claims.
You own: structure, headings, readability, Markdown tables, ordering, and clear
merchandising takeaways.
Do NOT write a `Source:` or `Sources:` footer — code appends it automatically.
If `required_filters` is present, weave every filter value naturally into the opening
(once each, without awkward synonym repetition).

## 2. ADAPTIVE & SATISFYING PRESENTATION
- **Match the ask**: Simple count / lookup → 1–3 sentences with the number; never inflate
  into an assortment report. Analysis, comparison, price trend, and product specification
  → full structured depth (never shrink those into 3–5 lines).
- **Headings & scannability**: Use clean Markdown headings (`###`), bold key figures,
  bullet points, and well-aligned Markdown tables whenever there are 3+ rows or multiple
  attributes/metrics — skip headings for plain counts.
- **Tables & charts**:
  - Present tables from `card.tables` or `card.results` cleanly with human-readable
    column headers and polished attribute names (e.g. `In stock`, `T-Shirt`, `100% Cotton`).
  - If `card.findings` contains a ready ```chart fence that fits the answer, you may
    preserve it cleanly (never invent fake chart values or `[chart placeholder]`).
- **Label polishing**: Prefer polished facet labels already in the card (from meta
  ``display_values`` / Neo4j names). If a value is still scrubbed lowercase, Title Case it
  and preserve any `&` `/` `-` shown in the card — do not invent new product names.
- **Every section adds value**: Do not pad with fluff or repeat the same sentence three
  times, and never include robotic filler closers ("Would you like...", "Let me know if...").

## 3. MULTI-SOURCE & COMPARISON DISCIPLINE
When multiple sources exist in the card, distinguish **PLM records** from **e-commerce
listings** cleanly. Do not merge counts from different sources or different brands into
one combined total unless the card explicitly provides a combined figure. Keep the primary
requested result prominent.

## 4. ANALYTICAL & REPORT FLOW (Analysis / Compare / Price Trend / Spec)
Follow this clear structure whenever the card carries rich data:
1. **Executive Summary & Scope** — directly answer the core ask and state resolved filters/scope.
2. **Key Metrics** — totals, price statistics, or overall direction first.
3. **Detailed Breakdown / Evidence** — Markdown tables and bullet insights across categories,
   materials, colours, brands, time periods, or specification fields.
4. **Merchandising Reading** — clear, grounded observations about what the numbers show.
5. **Data Notes & Limitations** — honestly note missing fields, outliers, mixed currencies,
   or sparse records from `data_notes` / `caveats` / `findings`.

## 5. WRITING STYLE & ACCURACY
- Write like an expert fashion business analyst: warm, direct, natural, non-technical.
- Never mention internal identifiers or plumbing (`knowledge graph`, `KG`, `Neo4j`,
  `Cypher`, `ecom_products`, `plmex_products`, `Catalog 1/2`, `Source 1/2`, `null`,
  `routes`, or internal colon-prefixed IDs).
- Never open with `"We found"`, `"Based on the context"`, `"According to"`, or robotic
  phrases like `"recorded N times"` / `"lead by a gap"`. Plain `"There are N …"` is fine
  for simple count asks.
- Zero matches: clearly state that no matching records exist under those conditions —
  never claim the brand/product does not exist at all.
- Reply in the user's language while preserving entity names and exact numbers.
- Return ONLY the final user-facing response.
""".strip()


def depth_from_question(question: str, *, intent: str = "count") -> str:
    """User words override defaults: brief / normal / full."""

    q = (question or "").casefold()
    if re.search(r"\b(in\s+2\s+lines?|one\s+line|one\s+sentence|briefly|tl;?dr|very\s+short)\b", q):
        return "brief"
    if intent in {"clarify", "chitchat"}:
        return "brief"
    if intent in {
        "analysis",
        "spec_brief",
        "report",
        "compare",
        "price_trend",
        "price_history",
        "breakdown",
        "ranking",
        "examples",
    }:
        return "full"
    if re.search(
        r"\b(full|detailed|in\s+detail|explain|analyze|analyse|report|comprehensive|"
        r"assortment|profile|profiles|briefing|compare|trend|specification)\b",
        q,
    ):
        return "full"
    return "normal"


def _concept_stem(value: str) -> str:
    """Normalize a filter value so sweater / sweaters / sweater garments collide."""

    v = re.sub(r"[^a-z0-9 ]", " ", (value or "").casefold())
    v = re.sub(r"\b(?:garments?|products?|items?|records?|listings?)\b", " ", v)
    v = re.sub(r"\s+", " ", v).strip()
    if v.endswith("ies") and len(v) > 4:
        v = v[:-3] + "y"
    elif v.endswith("ses") and len(v) > 4:
        v = v[:-2]
    elif v.endswith("s") and not v.endswith("ss") and len(v) > 3:
        v = v[:-1]
    return v


def _norm_match_text(text: str) -> str:
    return re.sub(r"[\s_\-]+", " ", (text or "").casefold()).strip()


def _derived_facts(
    results: list[SourceResult],
    filters: dict[str, str],
    *,
    intent: str = "count",
) -> list[str]:
    """Code-owned shares / ranks — gaps only for explicit compare intents."""

    out: list[str] = []
    for result in results:
        label = SOURCE_LABELS.get(result.source_id, "our data")
        total = result.total
        if result.breakdown:
            ranked = sorted(result.breakdown, key=lambda x: int(x.get("count") or 0), reverse=True)
            denom = total if (total and total > 0) else sum(int(x.get("count") or 0) for x in ranked)
            for i, item in enumerate(ranked[:10], start=1):
                name = str(item.get("value") or "").strip()
                count = int(item.get("count") or 0)
                if not name or count <= 0:
                    continue
                share = item.get("share_pct")
                if share is None and denom > 0:
                    share = round(100.0 * count / denom, 1)
                if share is not None and "share_pct" not in item:
                    item["share_pct"] = share
                share_str = f" ({share}%)" if share is not None else ""
                out.append(f"{label} rank {i}: {name}={count}{share_str}")
        if total == 0:
            desc = " ".join(str(v) for v in filters.values() if v) or "these filters"
            out.append(f"{label}: zero matches for {desc}")
    if (intent or "").casefold().strip() == "compare":
        with_totals = [(r, r.total) for r in results if r.total is not None]
        if len(with_totals) == 2:
            (a, ta), (b, tb) = with_totals
            if ta != tb:
                bigger = a if ta >= tb else b
                smaller = b if ta >= tb else a
                gap = abs(int(ta) - int(tb))
                out.append(
                    f"difference: {SOURCE_LABELS.get(bigger.source_id, 'a')} has {gap:,} more than "
                    f"{SOURCE_LABELS.get(smaller.source_id, 'b')}"
                )
    return out[:24]


def _filter_values(card: "AnswerCard") -> list[str]:
    """Stable display order for filter values that must appear in the answer."""

    order = (
        "brand",
        "buyer",
        "color",
        "colour",
        "gender",
        "material",
        "garment",
        "category",
        "availability",
        "season",
        "family",
    )
    seen: set[str] = set()
    seen_stems: set[str] = set()
    out: list[str] = []

    def _add(val: str) -> None:
        v = str(val or "").strip()
        if not v:
            return
        low = v.casefold()
        if low in seen:
            return
        stem = _concept_stem(v)
        if stem and stem in seen_stems:
            return
        seen.add(low)
        if stem:
            seen_stems.add(stem)
        out.append(v)

    for key in order:
        _add((card.filters or {}).get(key) or "")
    for key, val in (card.filters or {}).items():
        if key in order:
            continue
        _add(val)
    return out


def intent_hint(intent: str, card: "AnswerCard | None" = None) -> str:
    """One intent-specific hint injected after the master writer prompt."""

    key = (intent or "count").casefold().strip()
    filter_bits = _filter_values(card) if card else []
    must = (
        f" REQUIRED FILTERS TO MENTION NATURALLY: {', '.join(filter_bits)}."
        if filter_bits
        else ""
    )
    depth = getattr(card, "depth", None) or "normal"
    if key == "count" and card is not None and len(card.results) > 1:
        return (
            f"INTENT HINT (depth={depth}): SIMPLE COUNT — state both source totals in 1–3 "
            f"sentences; distinguish PLM records from e-commerce listings; keep the primary "
            f"result first.{must} No assortment headings, no competing facet tops, no related-facts dump."
        )
    if key == "count":
        return (
            f"INTENT HINT (depth={depth}): SIMPLE PRODUCT COUNT — 1–3 sentences with the exact "
            f"number and filters only.{must} No Assortment Overview / Key Metrics / Merchandising "
            "Reading / related facet tops. Do not invent beyond the card."
        )
    if key == "examples" and card is not None and card.requested_n:
        return (
            f"INTENT HINT: The user asked for {card.requested_n} examples. Open with a clear summary "
            f"(including totals if available), then present a clean Markdown table with up to "
            f"{card.requested_n} usable rows (and note plainly if fewer usable rows exist).{must}"
        )
    base = INTENT_HINTS.get(key, INTENT_HINTS["count"])
    if key in {"analysis", "report", "spec_brief", "compare", "price_trend", "price_history"}:
        depth_label = (getattr(card, "depth", None) if card is not None else None) or "full"
        base = f"{base} (card.depth={depth_label}; no length limit)."
    return f"{base}{must}" if must else base


@dataclass
class SourceResult:
    source_id: str  # plmex_products | ecom_products
    total: int | None = None
    breakdown: list[dict] = field(default_factory=list)  # value, count, share_pct
    examples: list[dict] = field(default_factory=list)  # name, price, availability


@dataclass
class AnswerCard:
    question: str
    intent: str
    filters: dict[str, str]
    results: list[SourceResult]
    requested_n: int | None = None
    data_notes: list[str] = field(default_factory=list)
    context: list[str] = field(default_factory=list)
    derived: list[str] = field(default_factory=list)
    depth: str = "normal"
    tables: list[dict[str, Any]] = field(default_factory=list)
    findings: str = ""
    caveats: list[str] = field(default_factory=list)

    def footer(self) -> str:
        labels: list[str] = []
        for r in self.results:
            lab = SOURCE_LABELS.get(r.source_id)
            if lab and lab not in labels:
                labels.append(lab[:1].upper() + lab[1:] if lab else lab)
        if not labels or self.intent in {"clarify", "chitchat"}:
            return ""
        head = "Sources" if len(labels) > 1 else "Source"
        return f"{head}: " + " · ".join(labels)

    def prompt_json(self) -> str:
        d = asdict(self)
        for r in d["results"]:
            r["source"] = SOURCE_LABELS.get(r.pop("source_id"), "our data")
        required = _filter_values(self)
        if required:
            d["required_filters"] = required
        for key in ("context", "data_notes", "derived", "tables", "findings", "caveats"):
            if not d.get(key):
                d.pop(key, None)
        return json.dumps(d, ensure_ascii=False, indent=1)


def _mentions(answer_low: str, value: str) -> bool:
    norm_ans = _norm_match_text(answer_low)
    v = re.sub(r"[^a-z0-9 ]", "", value.casefold()).strip()
    v_norm = _norm_match_text(value)
    forms = FILTER_SYNONYMS.get(v.replace(" ", ""), []) + [
        value.casefold(),
        v,
        v_norm,
    ]
    return any(f and (f in answer_low or f in norm_ans) for f in forms)


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def validate(answer: str, card: AnswerCard) -> list[str]:
    issues: list[str] = []
    low = answer.casefold()
    norm_low = _norm_match_text(answer)

    if card.intent != "clarify":
        filter_vals = [str(v).strip() for v in (card.filters or {}).values() if str(v or "").strip()]
        for key, value in card.filters.items():
            if not value:
                continue
            # Multi-party filter e.g. "Primark | Zara" — verify each party is mentioned
            parts = [p.strip() for p in re.split(r"\s*\|\s*", str(value)) if p.strip()]
            if len(parts) > 1:
                if all(_mentions(low, p) for p in parts):
                    continue
            elif _mentions(low, value):
                continue
            stem = _concept_stem(value)
            covered = False
            if stem:
                if stem in norm_low:
                    covered = True
                elif key in {"garment", "category", "garment_type"}:
                    for other in filter_vals:
                        if _concept_stem(other) == stem and _mentions(low, other):
                            covered = True
                            break
            if not covered:
                issues.append(f"missing_filter:{key}={value}")

    # Required totals from the card must appear (formatted or plain).
    answer_nums = {n.replace(",", "") for n in re.findall(r"\d[\d,]*", answer)}
    for result in card.results:
        if result.total is not None and result.total > 0:
            if str(result.total) not in answer_nums:
                issues.append(f"missing_value:total={result.total}")
        if card.intent in {"breakdown", "ranking", "analysis"}:
            for item in (result.breakdown or [])[:3]:
                name = str(item.get("value") or "").strip()
                if not name or name.casefold() in _NULLISH:
                    continue
                if name.casefold() not in low and _norm_match_text(name) not in norm_low:
                    issues.append(f"missing_value:name={name}")

    for pat in BANNED:
        if re.search(pat, low):
            issues.append(f"banned:{pat}")

    has_usable = (
        any((r.total or 0) > 0 for r in card.results)
        or any(r.breakdown for r in card.results)
        or any(r.examples for r in card.results)
        or bool(card.context)
        or bool(card.tables)
        or bool(card.findings.strip())
    )
    if has_usable and re.search(
        r"(?i)(do\s+not\s+have\s+the\s+necessary\s+data|no\s+totals?\s+or\s+breakdown|"
        r"not\s+available.*(?:breakdown|total|metric)|"
        r"(?:breakdown|total|metric|distribution).{0,60}not\s+available|"
        r"does\s+not\s+provide\s+a\s+breakdown|"
        r"available\s+data\s+does\s+not\s+provide|"
        r"breakdown\s+is\s+not\s+available)",
        answer or "",
    ):
        issues.append("format:false_refusal")
    # Attribute facets mistaken for brand compare parties.
    if re.search(
        r"(?i)side-by-side\s+compar(?:e|ison)\s+for\s+(?:category|material|color|colour|"
        r"department|gender|construction|plm)\b",
        answer or "",
    ):
        issues.append("format:fake_attribute_compare")

    # "There are N …" is the natural opener for simple counts; block only "We found".
    if card.intent in {"count", "product_count", "kg_stats"}:
        if _WE_FOUND_OPENER_RE.search(answer or ""):
            issues.append("format:robotic_opener")
    elif _ROBOTIC_OPENER_RE.search(answer or ""):
        issues.append("format:robotic_opener")

    lines = [ln.strip() for ln in answer.splitlines()]
    if any(ln.startswith("|") for ln in lines):
        data_rows = [
            ln
            for ln in lines
            if ln.startswith("|") and not re.match(r"^\|\s*:?-{3,}", ln)
        ][1:]
        if not data_rows:
            issues.append("table:no_rows")
        if card.requested_n and len(data_rows) < min(card.requested_n, 10):
            available = 0
            for result in card.results:
                available = max(available, len(result.breakdown), len(result.examples))
            if available >= card.requested_n and len(data_rows) < card.requested_n:
                issues.append(f"table:expected_{card.requested_n}_got_{len(data_rows)}")

    # Ground numbers against card JSON + findings (allowing <=100 percentages/ranks, years, and light rounding).
    grounding_blob = f"{card.prompt_json()}\n{card.findings or ''}".replace(",", "")
    allowed = set(re.findall(r"\d[\d.]*", grounding_blob))
    allowed_floats: list[float] = []
    for tok in allowed:
        try:
            allowed_floats.append(float(tok))
        except ValueError:
            continue

    for number in re.findall(r"\d[\d,]*\.?\d*", answer):
        cleaned = number.replace(",", "").rstrip(".")
        if not cleaned or cleaned in allowed:
            continue
        if re.fullmatch(r"(?:19|20)\d{2}", cleaned):
            continue
        try:
            val = float(cleaned)
        except ValueError:
            continue
        if val <= 100:
            continue
        if any(abs(val - known) <= max(5.0, abs(known) * 0.05) for known in allowed_floats):
            continue
        issues.append(f"number_not_in_card:{cleaned}")

    if re.search(r"(?im)^sources?\s*:", answer):
        issues.append("format:writer_wrote_footer")

    # Ensure rich analytical / report / price-trend / spec responses don't shrink into tiny stubs.
    if card.intent in {"analysis", "report", "compare", "spec_brief", "price_trend", "price_history"} and has_usable:
        rich = (
            len(card.context or []) >= 2
            or len(card.tables or []) >= 1
            or any(r.breakdown for r in card.results)
            or any((r.total or 0) > 0 for r in card.results)
            or len((card.findings or "").strip()) >= 180
        )
        if rich and _word_count(answer) < _ANALYSIS_MIN_WORDS:
            issues.append(f"format:too_brief:{_word_count(answer)}<{_ANALYSIS_MIN_WORDS}")
    # Simple counts stay short — do not require Related catalog facts / assortment padding.
    if card.intent in {"count", "product_count", "kg_stats"}:
        low_ans = (answer or "").casefold()
        if re.search(
            r"(?i)\b(assortment overview|merchandising reading|key metrics|"
            r"product distribution|color diversity|related catalog facts)\b",
            low_ans,
        ):
            issues.append("format:count_inflated_to_report")
    # Truncated / unclosed chart JSON must not ship to the UI.
    fence_depth = 0
    for ln in (answer or "").splitlines():
        s = ln.strip()
        if re.match(r"(?i)^```(?:chart|fbchart)\s*$", s):
            fence_depth += 1
        elif s == "```" and fence_depth > 0:
            fence_depth -= 1
    if fence_depth > 0:
        issues.append("format:unclosed_chart")
    if re.search(
        r'(?is)\{\s*"type"\s*:\s*"(?:bar|pie|line)".{0,400}$',
        (answer or "").rstrip(),
    ):
        tail = (answer or "").rstrip()
        start = tail.rfind("{")
        blob = tail[start:] if start >= 0 else ""
        try:
            import json as _json

            _json.loads(blob)
        except Exception:
            # Incomplete bare chart JSON at end of draft (max-token cut).
            if "```" not in blob:
                issues.append("format:truncated_chart")
    # Price-trend drafts must keep chart + period rows + examples when findings have them.
    if card.intent in {"price_trend", "price_history"}:
        findings = card.findings or ""
        low_find = findings.casefold()
        low_ans = (answer or "").casefold()
        if "```chart" in findings and "```chart" not in (answer or ""):
            issues.append("format:missing_chart")
        find_period_rows = [
            ln
            for ln in findings.splitlines()
            if ln.strip().startswith("|")
            and not re.match(r"^\|\s*:?-{3,}", ln.strip())
            and "period" not in ln.casefold()
            and "change %" not in ln.casefold()
        ]
        ans_period_rows = [
            ln
            for ln in (answer or "").splitlines()
            if ln.strip().startswith("|")
            and not re.match(r"^\|\s*:?-{3,}", ln.strip())
            and "period" not in ln.casefold()
            and "change %" not in ln.casefold()
        ]
        if len(find_period_rows) >= 2 and len(ans_period_rows) < min(2, len(find_period_rows)):
            issues.append(
                f"format:incomplete_period_table:{len(ans_period_rows)}<{len(find_period_rows)}"
            )
        if "example moves" in low_find and "example" not in low_ans:
            issues.append("format:missing_examples")
        # Mid-cut after a lone period cell (classic max-token truncate).
        if re.search(r"(?im)^\|\s*\d{1,2}-\d{2}[^\n|]*\|\s*$", answer or ""):
            issues.append("format:truncated_mid_table")
    return issues


# Attribute / viz headings must NEVER be treated as brand compare parties.
_FACET_HEADING_NOISE = frozenset(
    {
        "metric",
        "metrics",
        "overview",
        "query 1",
        "query 2",
        "category",
        "color",
        "colour",
        "material",
        "gender",
        "department",
        "season",
        "season year",
        "family",
        "subfamily",
        "construction",
        "availability",
        "garment type",
        "garment",
        "fit",
        "section",
        "buyer",
        "brand",
        "country",
        "closure",
        "certification",
        "care instruction",
        "color family",
        "plm main category",
        "plm sub category",
        "plm sub sub category",
        "main category",
        "sub category",
        "sub sub category",
        "patterns",
        "scope",
        "limitations",
        "reading",
        "e-commerce",
        "plmex",
        "plm",
    }
)


def _is_facet_heading(name: str) -> bool:
    low = re.sub(r"\s+", " ", (name or "").casefold().replace("_", " ")).strip()
    if not low or low in _FACET_HEADING_NOISE:
        return True
    if low.startswith("plm "):
        return True
    if low.endswith((" category", " type", " family", " year")):
        return True
    return False


def _compare_parties_from_card(card: AnswerCard) -> list[str]:
    """Brand/buyer parties for side-by-side compare — never attribute facet headings."""

    for key in ("brand", "buyer"):
        raw = str((card.filters or {}).get(key) or "")
        if "|" in raw:
            parts = [p.strip() for p in raw.split("|") if p.strip()]
            if len(parts) >= 2:
                return parts
    # Only recover from "Product analysis for X" lines — not ### Category / ### Material.
    text = "\n".join([*(card.context or []), card.findings or ""])
    found: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(
        r"(?im)(?:overview:\s*|Product analysis for\s+)([A-Za-z][A-Za-z0-9 &.’'-]{1,40}?)(?:\s+products\s*=|\s*:\s*\*?)",
        text,
    ):
        name = match.group(1).strip().rstrip(":")
        low = name.casefold()
        if not name or low in seen or _is_facet_heading(name):
            continue
        if low in {"this selection", "selection", "the assortment"}:
            continue
        seen.add(low)
        found.append(name)
    return found[:4] if len(found) >= 2 else []


def _leading_bits_from_card(card: AnswerCard, *, limit: int = 8) -> list[str]:
    group_bits: list[str] = []
    for r in card.results:
        for item in (r.breakdown or [])[:limit]:
            name = str(item.get("value") or "").strip()
            count = item.get("count")
            if not name or count is None:
                continue
            share = item.get("share_pct")
            if share is not None:
                group_bits.append(f"{name} ({count:,}, {share}%)")
            else:
                group_bits.append(f"{name} ({count:,})")
    if group_bits:
        return group_bits[:limit]
    leads: list[str] = []
    for line in card.context or []:
        low = line.casefold()
        if low.startswith("leading ") or low.startswith("leading value:"):
            bit = line.split(":", 1)[1].strip() if ":" in line else line.strip()
            if bit:
                leads.append(bit)
        elif low.startswith("products_total=") or re.search(r"(?i)\bunique\s*=", line):
            leads.append(line.strip())
        elif re.search(r"(?i)\btop\s*:", line):
            # Surface compact key=value tops from kg_stats meta lines.
            top = line.split("top:", 1)[-1].strip() if "top:" in low else line.strip()
            if top:
                leads.append(top if ":" not in line[:24] else f"{line.split(':', 1)[0].strip()}: {top}")
        if len(leads) >= limit:
            break
    return leads[:limit]


def _metric_bits_from_card(card: AnswerCard) -> list[str]:
    parts: list[str] = []
    for r in card.results:
        if r.total is None:
            continue
        label = SOURCE_LABELS.get(r.source_id, "our data")
        if r.total == 0:
            parts.append(f"none in {label}")
        elif card.intent in {"analysis", "report", "compare", "spec_brief"}:
            parts.append(f"{r.total:,} products in {label}")
        else:
            parts.append(f"{r.total:,} in {label}")
    return parts


def _humanize_price_cue(raw: str) -> str:
    """Turn 'metric: Average price=30.1' into a readable price cue."""

    text = str(raw or "").strip()
    text = re.sub(r"(?i)^metric:\s*", "", text).strip()
    m = re.search(
        r"(?i)(?:average|avg|mean)\s*price\s*[=:]\s*([\d.,]+)",
        text,
    )
    if m:
        avg = m.group(1).rstrip("0").rstrip(".") if "." in m.group(1) else m.group(1)
        bits = [f"average price about {avg}"]
        mn = re.search(r"(?i)min(?:imum)?\s*(?:price)?\s*[=:]\s*([\d.,]+)", text)
        mx = re.search(r"(?i)max(?:imum)?\s*(?:price)?\s*[=:]\s*([\d.,]+)", text)
        if mn:
            bits.append(f"from {mn.group(1)}")
        if mx:
            bits.append(f"up to {mx.group(1)}")
        if len(bits) == 1:
            return bits[0]
        if mn and mx:
            return f"average price about {avg} (range {mn.group(1)}–{mx.group(1)})"
        return ", ".join(bits)
    if "price range" in text.casefold():
        return text
    return text


def _price_line_from_context(context: list[str]) -> str | None:
    for line in context or []:
        low = line.casefold()
        if "average price" in low or "price range" in low or "min price" in low:
            return _humanize_price_cue(line)
        m = re.search(
            r"(?i)(?:average|avg|mean)\s+price\s*[=:]\s*([\d.,]+).*?"
            r"(?:min(?:imum)?\s*[=:]\s*([\d.,]+))?.*?"
            r"(?:max(?:imum)?\s*[=:]\s*([\d.,]+))?",
            line,
        )
        if m:
            return _humanize_price_cue(line)
    return None


def _analysis_opening_paragraph(card: AnswerCard) -> str:
    """Natural briefing opener — brand/scope + size + price, no labeled Scope dump."""

    scope = _human_scope_phrase(card)
    metrics = _metric_bits_from_card(card)
    price = _price_line_from_context(card.context or [])
    # Prefer a clean Title Case brand/scope for the headline.
    title = scope[:1].upper() + scope[1:] if scope else "This selection"

    if metrics:
        size = "; ".join(metrics)
        if price:
            return (
                f"**{title}** shows **{size}**, with {price}. "
                "Here's how the catalog breaks down by attribute."
            )
        return (
            f"**{title}** shows **{size}**. "
            "Here's how the catalog breaks down by attribute."
        )
    if price:
        return (
            f"Here's a product analysis of **{title}** ({price}). "
            "Attribute breakdowns follow below."
        )
    return (
        f"Here's a product analysis of **{title}**. "
        "Attribute breakdowns follow below."
    )


def _render_md_table(headers: list[str], rows: list[list[str]]) -> str:
    if not headers or not rows:
        return ""
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def _render_tables_from_card(card: AnswerCard) -> list[str]:
    blocks: list[str] = []
    for tbl in card.tables or []:
        headers = tbl.get("headers") or []
        rows = tbl.get("rows") or []
        if not headers or not rows:
            continue
        title = str(tbl.get("title") or "").strip()
        md = _render_md_table(headers, rows)
        if title:
            blocks.append(f"### {title}\n{md}")
        else:
            blocks.append(md)
    if blocks:
        return blocks
    # Synthesize from SourceResult breakdowns / examples if no explicit table was captured.
    for r in card.results:
        label = SOURCE_LABELS.get(r.source_id, "Results")
        if r.breakdown:
            rows = []
            try:
                from knowledge_graph.retrieval.ecom import _display_label as _polish_label
            except Exception:  # pragma: no cover
                _polish_label = lambda x: str(x)
            for item in r.breakdown:
                val = _polish_label(str(item.get("value") or ""))
                cnt = f"{int(item['count']):,}" if item.get("count") is not None else "—"
                sh = f"{item['share_pct']}%" if item.get("share_pct") is not None else "—"
                rows.append([val, cnt, sh])
            blocks.append(
                f"### {label.capitalize()} Breakdown\n"
                + _render_md_table(["Value", "Products", "Share"], rows)
            )
        if r.examples:
            ex_rows = [
                [
                    str(ex.get("name") or "—"),
                    str(ex.get("price") or "—"),
                    str(ex.get("availability") or "—"),
                ]
                for ex in r.examples
            ]
            blocks.append(
                f"### {label.capitalize()} Examples\n"
                + _render_md_table(["Product", "Price", "Availability"], ex_rows)
            )
    return blocks


def _side_by_side_compare_fallback(card: AnswerCard, parties: list[str]) -> str | None:
    if len(parties) < 2:
        return None
    blob = "\n".join([*(card.context or []), card.findings or ""])
    brand_blocks: list[str] = []
    for party in parties:
        pat = re.compile(
            rf"(?is)(?:###\s*{re.escape(party)}\b.*?Product analysis for\s+{re.escape(party)}\s*:\s*\*\*([\d,]+)\*\*"
            rf"|Product analysis for\s+{re.escape(party)}\s*:\s*\*\*([\d,]+)\*\*)",
        )
        m = pat.search(blob)
        n = (m.group(1) or m.group(2) or "").replace(",", "") if m else None
        leads: list[str] = []
        section = re.search(rf"(?is)###\s*{re.escape(party)}\b(.*?)(?=^###\s|\Z)", blob)
        chunk = section.group(1) if section else blob
        for line in chunk.splitlines():
            low = line.casefold()
            if low.startswith("leading value:") or low.startswith("leading "):
                leads.append(line.split(":", 1)[1].strip() if ":" in line else line.strip())
            if len(leads) >= 6:
                break
        lines = [f"### {party}"]
        if n:
            lines.append(f"**Metrics:** {int(n):,} products on the resolved brand filter.")
        if leads:
            lines.append("**Patterns:** " + "; ".join(leads[:6]) + ".")
        if len(lines) > 1:
            brand_blocks.append("\n".join(lines))

    tables = _render_tables_from_card(card)
    if len(brand_blocks) < 2 and not tables:
        if card.findings.strip():
            return card.findings.strip()
        return (
            f"Comparing **{'** vs **'.join(parties)}** on separate brand filters.\n\n"
            "### Notes\nPer-brand totals could not be separated cleanly from the retrieved context."
        )
    opener = (
        f"Side-by-side look at **{'** and **'.join(parties)}** — "
        "each brand is summarized separately below."
    )
    parts = [opener, *brand_blocks, *tables]
    if card.data_notes or card.caveats:
        parts.append("### Notes\n" + " ".join([*card.data_notes[:4], *card.caveats[:3]]))
    return "\n\n".join(p for p in parts if p)


def _human_scope_phrase(card: AnswerCard) -> str:
    """Natural scope line: 'Primark sweaters' rather than 'primark, sweater, primark'."""

    brand = str((card.filters or {}).get("brand") or (card.filters or {}).get("buyer") or "").strip()
    if "|" in brand:
        brand = " vs ".join(p.strip() for p in brand.split("|") if p.strip())
    garment = str(
        (card.filters or {}).get("garment")
        or (card.filters or {}).get("garment_type")
        or (card.filters or {}).get("category")
        or ""
    ).strip()
    bits: list[str] = []
    if brand:
        bits.append(brand[:1].upper() + brand[1:] if brand else brand)
    if garment:
        g = garment.casefold()
        if not g.endswith("s"):
            garment = garment + "s" if not g.endswith(("ss", "us")) else garment
        bits.append(garment.casefold())
    if bits:
        return " ".join(bits)
    other = [v for k, v in (card.filters or {}).items() if k not in {"brand", "buyer", "garment", "garment_type", "category"} and v]
    return ", ".join(str(v) for v in other) if other else "this assortment"


def _facet_leaders_from_context(card: AnswerCard, *, limit: int = 8) -> list[tuple[str, str]]:
    """Parse 'leading Color: Black=600 (14.8%)' style context into (facet, leader)."""

    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line in card.context or []:
        m = re.match(
            r"(?i)^leading\s+([^:]+):\s*(.+)$",
            line.strip(),
        )
        if not m:
            continue
        facet = re.sub(r"(?i)^plm\s+", "", m.group(1).strip().replace("_", " "))
        facet = re.sub(r"\s{2,}", " ", facet).strip()
        # Friendlier labels for PLM hierarchy noise.
        facet = {
            "sub sub category": "Knit / product type",
            "sub category": "Product group",
            "main category": "Main category",
        }.get(facet.casefold(), facet)
        leader = m.group(2).strip()
        key = facet.casefold()
        if not facet or key in seen:
            continue
        # Skip party self-echo (leading Brand: Primark=…).
        party = str((card.filters or {}).get("brand") or (card.filters or {}).get("buyer") or "")
        if key in {"brand", "buyer"} and party and party.casefold().split("|")[0].strip() in leader.casefold():
            continue
        seen.add(key)
        out.append((facet[:1].upper() + facet[1:], leader))
        if len(out) >= limit:
            break
    return out


def _field_sections_from_findings(findings: str) -> list[str]:
    """Keep per-attribute ### blocks (tables/charts) from analysis findings."""

    text = str(findings or "").strip()
    if not text:
        return []
    # Strip internal viz markers; keep headings, tables, charts.
    cleaned_lines: list[str] = []
    for ln in text.splitlines():
        if ln.strip().startswith("<!--fb:viz:"):
            continue
        cleaned_lines.append(ln)
    body = "\n".join(cleaned_lines).strip()
    if "### " not in body:
        return []
    # Prefer from first field heading onward (drop duplicate overview prose above).
    idx = body.find("\n### ")
    if idx < 0:
        idx = 0 if body.startswith("### ") else -1
    if idx < 0:
        return []
    chunk = body[idx:].lstrip("\n") if idx > 0 else body
    # Drop trailing sample-only noise if present without a heading.
    return [chunk] if chunk.strip() else []


def _findings_look_like_product_specification_report(findings: str) -> bool:
    """True when retrieve already built a proposed NEW product specification package."""

    text = str(findings or "").strip()
    if len(text) < 120:
        return False
    low = text.casefold()
    has_identity = "pending plmex" in low
    has_sheet = (
        "proposed new product specification" in low
        or "final product specification" in low
        or '"approval_status"' in low
    )
    has_analysis = (
        "product specification decision engine" in low
        or "historical data overview" in low
        or "historical data analysis" in low
        or "### attribute analysis" in low
        or "## 1. user requirements" in low
    )
    return has_identity and has_sheet and has_analysis


def _specification_report_fallback(card: AnswerCard) -> str:
    """Ship the buyer brief as-is when retrieve already built the 4-section report."""

    findings = str(card.findings or "").strip()
    if not findings:
        return "No product specification evidence was retrieved."

    # Prefer the markdown buyer brief when present.
    for marker in (
        "# ",
        "## 1. User Requirements",
        "Historical Data Analysis",
        "Final Product Specification",
        "Product specification:",
        "Proposed new product specification",
    ):
        idx = findings.find(marker)
        if idx >= 0 and marker.startswith("#"):
            # Start at the titled brief; drop scope chatter above the title.
            start = findings[idx:].strip()
            notes = [*(card.data_notes or [])[:2], *(card.caveats or [])[:2]]
            find_low = start.casefold()
            extra = [n for n in notes if n and str(n).casefold() not in find_low]
            if extra:
                start = start + "\n\n### Notes\n" + "\n".join(f"- {n}" for n in extra)
            return start
        if idx >= 0 and marker == "## 1. User Requirements":
            start = findings[idx:].strip()
            # Keep a preceding H1 title if present.
            head = findings[:idx].rstrip().splitlines()
            if head and head[-1].startswith("# "):
                start = head[-1] + "\n\n" + start
            notes = [*(card.data_notes or [])[:2], *(card.caveats or [])[:2]]
            find_low = start.casefold()
            extra = [n for n in notes if n and str(n).casefold() not in find_low]
            if extra:
                start = start + "\n\n### Notes\n" + "\n".join(f"- {n}" for n in extra)
            return start

    # Legacy fallback: overview / proposed sheet slices.
    def _section(start_mark: str, *stop_marks: str) -> str:
        idx = findings.find(start_mark)
        if idx < 0:
            return ""
        chunk = findings[idx:]
        for stop in stop_marks:
            cut = chunk.find(stop, len(start_mark))
            if cut > 0:
                chunk = chunk[:cut]
                break
        return chunk.strip()

    parts: list[str] = []
    for line in findings.splitlines()[:8]:
        if line.lower().startswith("scope for "):
            parts.append(line.strip())
            break
    overview = _section(
        "Historical data overview:",
        "\nImportant relationships and patterns:",
        "\nEvidence note:",
        "\nProduct specification:",
    )
    if overview:
        parts.append(overview)
    proposed = _section(
        "Product specification:",
        "\nBuyer confirmation requirements:",
        "\n### Buyer confirmation requirements",
        "\nData gaps:",
        "\nHistorical benchmark note:",
    )
    if not proposed:
        proposed = _section(
            "Proposed new product specification",
            "\nBuyer confirmation requirements:",
            "\nHistorical benchmark note:",
        )
    if proposed:
        parts.append(proposed)
    confirm = _section(
        "Buyer confirmation requirements:",
        "\nData gaps:",
        "\nHistorical benchmark note:",
    ) or _section(
        "### Buyer confirmation requirements",
        "\nHistorical benchmark note:",
    )
    if confirm:
        parts.append(confirm)
    bench = _section("Historical benchmark note:")
    if bench:
        parts.append(bench)
    start = "\n\n".join(p for p in parts if p).strip() or findings
    notes = [*(card.data_notes or [])[:2], *(card.caveats or [])[:2]]
    find_low = start.casefold()
    extra = [n for n in notes if n and str(n).casefold() not in find_low]
    if extra:
        start = start + "\n\n### Notes\n" + "\n".join(f"- {n}" for n in extra)
    return start


def _findings_look_like_complete_analysis(findings: str) -> bool:
    """True when retrieve already has attribute tables — do not LLM-rewrite duplicates."""

    text = str(findings or "").strip()
    if len(text) < 80:
        return False
    # Spec Decision Engine packages are not assortment analysis dumps.
    if _findings_look_like_product_specification_report(text):
        return False
    low = text.casefold()
    has_table = "|" in text and "---" in text
    has_field = "### " in text or "leading value:" in low
    has_scope = (
        "product analysis for" in low
        or "share %" in low
        or "share%" in low
        or ("count" in low and "share" in low)
    )
    return bool(has_table and (has_field or has_scope))


def _findings_look_like_complete_price_trend(findings: str) -> bool:
    """True when retrieve already produced a full trend package (metrics + evidence)."""

    text = str(findings or "").strip()
    if len(text) < 160:
        return False
    # Attribute assortment tables are not price-trend packages.
    if _findings_look_like_complete_analysis(text) and not any(
        tok in text.casefold()
        for tok in (
            "same-currency",
            "same currency",
            "price change",
            "trending products",
            "most active / trending",
            "```chart",
        )
    ):
        return False
    low = text.casefold()
    has_metrics = (
        ("average" in low and "change" in low)
        or "same-currency" in low
        or "same currency" in low
        or "trending products" in low
        or "most active" in low
    )
    has_evidence = (
        "```chart" in text
        or "example moves" in low
        or "| period |" in low
        or "largest same-currency" in low
        or "stock snaps" in low
        or "| product |" in low
    )
    return bool(has_metrics and has_evidence)


def _price_trend_fallback(card: AnswerCard) -> str:
    """Ship the complete retrieve package — never a truncated LLM rewrite."""

    findings = str(card.findings or "").strip()
    # Mis-routed assortment / color tables must not get a Price-trend wrapper + reprint.
    if _findings_look_like_complete_analysis(findings) and not _findings_look_like_complete_price_trend(
        findings
    ):
        return _analysis_report_fallback(card)
    if _findings_look_like_complete_price_trend(findings):
        sections = [findings]
        notes = [*(card.data_notes or [])[:3], *(card.caveats or [])[:3]]
        # Avoid duplicating caveats already present in findings.
        find_low = findings.casefold()
        extra = [n for n in notes if n and str(n).casefold() not in find_low]
        if extra:
            sections.append("### Notes\n" + "\n".join(f"- {n}" for n in extra))
        return "\n\n".join(sections)

    scope = _human_scope_phrase(card)
    low_find = findings.casefold()
    # Product-count stubs are not a price-trend answer — say so instead of reprinting totals.
    count_only = bool(findings) and (
        "products with brand" in low_find
        or "product analysis for" in low_find
        or (
            re.search(r"\bthere are\b.*\bproducts?\b", low_find)
            and not any(
                tok in low_find
                for tok in (
                    "price change",
                    "same-currency",
                    "same currency",
                    "trending",
                    "```chart",
                    "currency",
                )
            )
        )
    )
    if count_only or not findings:
        notes = [*(card.data_notes or [])[:3], *(card.caveats or [])[:3]]
        body = (
            f"No comparable price-trend evidence was available for {scope} "
            "(assortment product totals are not a substitute for PriceChangeEvent trends)."
        )
        if notes:
            body += "\n\n### Notes\n" + "\n".join(f"- {n}" for n in notes if n)
        return f"### Price trend · {scope}\n\n{body}"

    parts = [f"### Price trend · {scope}"]
    metrics = _metric_bits_from_card(card)
    # Drop catalog product-count metrics from a price-trend card.
    metrics = [
        m
        for m in (metrics or [])
        if "product" not in str(m).casefold() or "price" in str(m).casefold()
    ]
    if metrics:
        parts.append("**Key metrics:** " + "; ".join(metrics))
    # If findings already include the tables, do not print card.tables again.
    if findings and ("|" in findings and "---" in findings):
        parts.append(findings)
    else:
        tables = _render_tables_from_card(card)
        if tables:
            parts.extend(tables)
        if findings:
            parts.append(findings)
    notes = [*(card.data_notes or [])[:4], *(card.caveats or [])[:3]]
    if notes:
        find_low = findings.casefold()
        extra = [n for n in notes if n and str(n).casefold() not in find_low]
        if extra:
            parts.append("### Notes\n" + "\n".join(f"- {n}" for n in extra))
    return "\n\n".join(p for p in parts if p) if len(parts) > 1 else (
        f"No price-trend evidence was available for {scope}."
    )


def _analysis_report_fallback(card: AnswerCard) -> str:
    """Field-wise product analysis briefing — ship retrieve text once, no reprint."""

    findings = str(card.findings or "").strip()
    scope = _human_scope_phrase(card)
    # Complete retrieve package already has overview + field tables — print once.
    if _findings_look_like_complete_analysis(findings):
        sections = [findings]
        notes = [*(card.data_notes or [])[:5], *(card.caveats or [])[:3]]
        find_low = findings.casefold()
        extra = []
        for n in notes:
            t = str(n).replace("PLMex", "PLM").replace("plmex", "PLM")
            if t and t.casefold() not in find_low:
                extra.append(t)
        if extra:
            sections.append("### Notes\n" + "\n".join(f"- {n}" for n in extra))
        return "\n\n".join(sections)

    metrics = _metric_bits_from_card(card)
    field_blocks = _field_sections_from_findings(findings)
    sections: list[str] = [_analysis_opening_paragraph(card)]

    if field_blocks:
        # Field ### headings already in findings — don't wrap in another cold label.
        sections.append("\n\n".join(field_blocks))
    else:
        table_blocks = _render_tables_from_card(card)
        if table_blocks:
            sections.append("### Attribute breakdown\n" + "\n\n".join(table_blocks))
        else:
            leaders = _facet_leaders_from_context(card, limit=16)
            if leaders:
                rows = [[facet, leader] for facet, leader in leaders]
                sections.append(
                    "### Attribute breakdown\n"
                    "Leading value per attribute (full value tables were not available):\n\n"
                    + _render_md_table(["Field", "Leading value"], rows)
                )

    if metrics or field_blocks or card.tables or card.context:
        sections.append(
            "### What stands out\n"
            f"The tables above show how **{scope}** is weighted across each attribute. "
            "Read category, material, colour, and garment type together — "
            "and keep PLM vs e-commerce totals separate when both appear."
        )
    if card.data_notes or card.caveats:
        notes = [*card.data_notes[:5], *card.caveats[:3]]
        cleaned = []
        for n in notes:
            t = str(n).replace("PLMex", "PLM").replace("plmex", "PLM")
            cleaned.append(t)
        sections.append("### Notes\n" + "\n".join(f"- {n}" for n in cleaned))
    elif not metrics and not field_blocks and not card.tables:
        sections.append(
            f"### Notes\nNo matching totals or field breakdowns were available for {scope}."
        )
    return "\n\n".join(s for s in sections if s)


def safe_fallback(card: AnswerCard) -> str:
    """Rich, structured fallback using all data in the card when the LLM writer fails."""

    # Multi-brand compare ONLY when filters encode A|B (or recovered analysis-for parties).
    parties = _compare_parties_from_card(card)
    multi_brand = bool(parties) and any(
        "|" in str((card.filters or {}).get(k) or "") for k in ("brand", "buyer")
    )
    if multi_brand and card.intent in {"analysis", "compare", "report"}:
        side = _side_by_side_compare_fallback(card, parties)
        if side:
            return side

    if card.intent in {"price_trend", "price_history"}:
        return _price_trend_fallback(card)

    if card.intent in {"analysis", "report", "compare"}:
        return _analysis_report_fallback(card)

    filt_bits = _filter_values(card)
    desc = " ".join(filt_bits) if filt_bits else "this request"
    parts = _metric_bits_from_card(card)
    lead_bits = _leading_bits_from_card(card, limit=8)
    tables = _render_tables_from_card(card)

    sections: list[str] = []
    if parts:
        joined = "; ".join(parts)
        head = f"{desc[:1].upper() + desc[1:]}: {joined}." if filt_bits else joined[:1].upper() + joined[1:] + "."
        sections.append(head)
    elif filt_bits:
        sections.append(f"### Scope\n{desc[:1].upper() + desc[1:]}")

    count_like = card.intent in {"count", "product_count", "kg_stats"}
    # Plain counts: headline only — never related-facts / assortment padding.
    if count_like and sections:
        return "\n\n".join(sections)
    if lead_bits and not tables:
        sections.append("**Key highlights:**\n" + "\n".join(f"- {b}" for b in lead_bits))
    if tables:
        sections.extend(tables)
    elif card.context:
        ctx_lines = [c for c in card.context[:10] if c]
        if ctx_lines:
            sections.append("### Additional Details\n" + "\n".join(f"- {c}" for c in ctx_lines))
    if not tables and not parts and card.findings.strip():
        sections.append(card.findings.strip())
    if card.data_notes or card.caveats:
        notes = [*card.data_notes[:4], *card.caveats[:3]]
        sections.append("### Notes\n" + "\n".join(f"- {n}" for n in notes))
    if sections:
        return "\n\n".join(sections)
    return f"No matching data turned up for {desc}."


_COUNT_INFLATION_SECTION_RE = re.compile(
    r"(?ims)^#{1,3}\s*(?:assortment overview|key metrics|merchandising reading|"
    r"product distribution|related catalog facts|data notes|color diversity)\b.*?"
    r"(?=^#{1,3}\s|\Z)"
)


def _strip_count_inflation(text: str) -> str:
    """Remove report-template sections that leak into simple product counts."""

    out = _COUNT_INFLATION_SECTION_RE.sub("", text or "")
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


def write_answer(
    card: AnswerCard,
    llm: Callable[[str], str],
    writer_prompt: str = WRITER_PROMPT,
    max_tries: int = 2,
) -> str:
    hint = intent_hint(card.intent, card)
    card_json = card.prompt_json()
    LOGGER.info(
        "answer_card write intent=%s depth=%s filters=%s results=%s tables=%s context_n=%s derived_n=%s",
        card.intent,
        card.depth,
        list(card.filters),
        [(r.source_id, r.total) for r in card.results],
        len(card.tables or []),
        len(card.context or []),
        len(card.derived or []),
    )
    LOGGER.debug("answer_card payload=%s", card_json)
    # Price-trend retrieve packages are already complete — prefer them over a
    # truncated LLM rewrite (common mid-table / missing-chart cutoffs).
    if card.intent in {"price_trend", "price_history"} and _findings_look_like_complete_price_trend(
        card.findings or ""
    ):
        fallback = _price_trend_fallback(card)
        footer = card.footer()
        final = f"{fallback}\n\n{footer}".strip() if footer else fallback
        LOGGER.info("answer_card price_trend shipped complete findings (%s chars)", len(final))
        return final
    # New product-spec Decision Engine packages are already report-shaped — ship once.
    if card.intent == "spec_brief" and _findings_look_like_product_specification_report(
        card.findings or ""
    ):
        fallback = _specification_report_fallback(card)
        footer = card.footer()
        final = f"{fallback}\n\n{footer}".strip() if footer else fallback
        LOGGER.info("answer_card spec_brief shipped proposed-spec findings (%s chars)", len(final))
        return final
    # Assortment / color breakdowns: ship retrieve tables once — LLM rewrites reprint them.
    if card.intent in {"analysis", "report", "compare", "breakdown", "ranking"} and (
        _findings_look_like_complete_analysis(card.findings or "")
    ):
        fallback = _analysis_report_fallback(card)
        footer = card.footer()
        final = f"{fallback}\n\n{footer}".strip() if footer else fallback
        LOGGER.info(
            "answer_card %s shipped complete findings (%s chars)",
            card.intent,
            len(final),
        )
        return final
    # Mis-tagged price_trend intent with assortment findings → analysis once.
    if card.intent in {"price_trend", "price_history"} and _findings_look_like_complete_analysis(
        card.findings or ""
    ):
        fallback = _analysis_report_fallback(card)
        footer = card.footer()
        final = f"{fallback}\n\n{footer}".strip() if footer else fallback
        LOGGER.info("answer_card misrouted price_trend shipped analysis once (%s chars)", len(final))
        return final
    feedback = ""
    last_draft = ""
    last_issues: list[str] = []
    for attempt in range(max_tries):
        prompt = f"{writer_prompt}\n\n{hint}\n\nANSWER CARD:\n{card_json}{feedback}"
        LOGGER.debug("answer_card prompt_attempt=%s chars=%s", attempt + 1, len(prompt))
        draft = str(llm(prompt) or "").strip()
        last_draft = draft
        if not draft:
            last_issues = ["empty_draft"]
            break
        try:
            from fashion_brain.agents.answer_format import repair_chart_blocks

            draft = repair_chart_blocks(draft)
        except Exception:
            pass
        issues = validate(draft, card)
        last_issues = issues
        LOGGER.info(
            "answer_card draft_attempt=%s ok=%s issues=%s",
            attempt + 1,
            not issues,
            issues,
        )
        if not issues:
            try:
                from fashion_brain.agents.answer_format import repair_chart_blocks

                draft = repair_chart_blocks(draft)
            except Exception:
                pass
            if card.intent in {"count", "product_count", "kg_stats"}:
                draft = _strip_count_inflation(draft)
            footer = card.footer()
            final = f"{draft}\n\n{footer}".strip() if footer else draft
            LOGGER.debug("answer_card final=%s", final[:500])
            return final
        extra = ""
        if any("count_inflated_to_report" in i for i in issues):
            # Deterministic cleanup beats another LLM retry that may re-inflate.
            cleaned = _strip_count_inflation(draft)
            if cleaned and "count_inflated_to_report" not in " ".join(validate(cleaned, card)):
                footer = card.footer()
                return f"{cleaned}\n\n{footer}".strip() if footer else cleaned
            extra = (
                " Rewrite as a simple count: 1–3 sentences with the asked number and filters only. "
                "Remove Assortment Overview, Key Metrics, Merchandising Reading, and competing tops."
            )
        elif any("too_brief" in i or "missing_report_structure" in i for i in issues):
            if card.intent in {"count", "product_count", "kg_stats"}:
                extra = (
                    " Keep it a short count answer (1–3 sentences). Do not inflate into a report."
                )
            elif card.intent in {"price_trend", "price_history"}:
                extra = (
                    " Include the full period table, the complete ```chart block from findings, "
                    "drivers, and every Example moves line. Never truncate mid-table."
                )
            else:
                extra = (
                    " Open with 1–2 natural sentences (brand/scope + size + price), then "
                    "attribute sections with a table per field, then a short 'What stands out'. "
                    "Do not open with '### Scope' / '### Core metrics' or raw 'metric:' lines. "
                    "Do not collapse to a single leading-values table. Do not answer in 2-3 lines."
                )
        if any(
            key in i
            for i in issues
            for key in (
                "unclosed_chart",
                "truncated_chart",
                "missing_chart",
                "incomplete_period_table",
                "missing_examples",
                "truncated_mid_table",
            )
        ):
            extra += (
                " Preserve the full period table, complete ```chart fence, and Example moves "
                "from findings. Close every chart fence. Never leave a half-written table row."
            )
        feedback = (
            "\n\nYour last draft failed these checks, fix them: "
            + "; ".join(issues)
            + extra
        )
    # Prefer a stripped LLM draft for counts when fallback would lose the spoken number.
    if card.intent in {"count", "product_count", "kg_stats"} and last_draft:
        cleaned = _strip_count_inflation(last_draft)
        if cleaned and not validate(cleaned, card):
            footer = card.footer()
            return f"{cleaned}\n\n{footer}".strip() if footer else cleaned
    fallback = safe_fallback(card)
    try:
        from fashion_brain.agents.answer_format import repair_chart_blocks

        fallback = repair_chart_blocks(fallback)
    except Exception:
        pass
    if card.intent in {"count", "product_count", "kg_stats"}:
        fallback = _strip_count_inflation(fallback)
    footer = card.footer()
    final = f"{fallback}\n\n{footer}".strip() if footer else fallback
    LOGGER.warning(
        "answer_card fallback used draft=%r issues=%s final=%s",
        (last_draft or "")[:240],
        last_issues,
        final[:240],
    )
    return final


def pluralize_garment(name: str) -> str:
    raw = (name or "").strip()
    if not raw:
        return raw
    low = raw.casefold()
    if low.endswith("s") and not low.endswith(("ss", "us", "is")):
        return raw
    if low.endswith(("s", "ss", "sh", "ch", "x", "z")):
        return raw + "es"
    if low.endswith("y") and len(raw) > 1 and raw[-2].casefold() not in "aeiou":
        return raw[:-1] + "ies"
    return raw + "s"


def clean_example_rows(rows: list[dict[str, Any]], *, limit: int = 15) -> list[dict[str, str]]:
    """Keep only displayable product examples for the card."""

    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows or []:
        name = str(row.get("name") or row.get("product name") or "").strip()
        price = str(row.get("price") or "").strip()
        avail = str(row.get("availability") or "").strip()
        if name.casefold() in _NULLISH:
            continue
        if re.fullmatch(r"\d{6,}", name) or re.fullmatch(r"[A-Z]{0,3}\d{5,}", name):
            continue
        if price and _MALFORMED_PRICE_RE.search(price):
            price = ""
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "name": name,
                "price": price,
                "availability": avail if avail.casefold() not in _NULLISH else "",
            }
        )
        if len(out) >= limit:
            break
    return out


_INTENT_PRIORITY = (
    "clarify",
    "price_trend",
    "price_history",
    "spec_brief",
    "analysis",
    "compare",
    "ranking",
    "breakdown",
    "examples",
    "count",
)


def intent_from_routes(
    routes: list[str] | None,
    *,
    question: str = "",
    response_type: str | None = None,
    group_by: str | None = None,
) -> str:
    q = (question or "").casefold()
    rt = str(response_type or "").casefold()
    route_keys = {str(r or "").casefold().strip() for r in (routes or [])}

    if "clarification" in route_keys or rt == "clarification":
        return "clarify"

    # Single-facet "which color is most…" beats sticky price_trend / full analysis routes.
    try:
        from fashion_brain.ask import is_dominant_facet_ask
    except Exception:
        is_dominant_facet_ask = lambda _q: False  # type: ignore
    if group_by or is_dominant_facet_ask(question):
        return "breakdown"
    if any(
        tok in q
        for tok in (
            "color",
            "colour",
            "material",
            "fabric",
            "composition",
            "mix",
            "breakdown",
            "distribution",
            "split",
            "wise",
        )
    ) and not (
        "price" in q
        and any(tok in q for tok in ("trend", "history", "change", "drop", "rise"))
    ):
        return "breakdown"

    # Price trend / history must beat generic "product_count" route mapping.
    if route_keys & {"price_trend", "price_history", "trending"} or rt == "trend":
        return "price_trend"
    if "price" in q and any(
        tok in q
        for tok in (
            "trend",
            "trending",
            "moving",
            "movement",
            "history",
            "historical",
            "changed",
            "change",
            "fluctuat",
            "drop",
            "dropped",
            "rise",
            "rose",
            "increase",
            "decrease",
        )
    ):
        return "price_trend"

    # Product specification / detail briefs must beat generic "product_count".
    if route_keys & {"specification", "product_specification", "product_spec", "product_details"} or rt == "specification":
        return "spec_brief"
    if any(
        tok in q
        for tok in (
            "specification",
            "product spec",
            "construction details",
            "suggest a ",
            "suggest style",
            "sample spec",
            "tech pack",
            "/specification",
            "/spec",
        )
    ):
        return "spec_brief"

    if any(tok in q for tok in ("example", "show me", "list ", "show ")):
        if "example" in q or "show me" in q or re.search(r"\bshow\s+\d+", q):
            return "examples"

    if "ranking" in route_keys or "top " in q or "ranking" in q:
        return "ranking"

    analysis_ask = (
        rt in {"analysis_report", "analysis"}
        or bool(route_keys & {"analysis", "analytical_tool", "products"})
        or any(
            tok in q
            for tok in (
                "assortment",
                "profile",
                "profiles",
                "full analysis",
                "comprehensive",
                "deep dive",
                "analyse",
                "analyze",
                "analysis",
                "overview",
            )
        )
    )
    if analysis_ask:
        return "analysis"

    if rt == "comparison" or bool(route_keys & {"compare", "comparison", "shared_products"}) or (
        "compare" in q or " vs " in f" {q} " or " versus " in f" {q} "
    ):
        return "compare"

    mapped: list[str] = []
    for route in routes or []:
        key = str(route or "").casefold().strip()
        if key in INTENT_BY_ROUTE:
            mapped.append(INTENT_BY_ROUTE[key])
    for preferred in _INTENT_PRIORITY:
        if preferred in mapped:
            return preferred

    if rt in {"stats", "concise"}:
        return "breakdown" if group_by else "count"
    return "count"


def _source_id_from_label(label: str) -> str | None:
    low = (label or "").casefold()
    if "plmex" in low or low in {"plm", "plm records", "our plm records"} or "plm" in low:
        return "plmex_products"
    if "ecom" in low or "commerce" in low or "online" in low:
        return "ecom_products"
    if "own" in low:
        return "own_products"
    return None


def _parse_filter_phrase(phrase: str) -> dict[str, str]:
    body = re.sub(r"(?i)^(?:with|in)\s+", "", (phrase or "").strip()).strip()
    if not body:
        return {}
    out: dict[str, str] = {}
    known = {
        "brand",
        "buyer",
        "category",
        "garment",
        "garment type",
        "gender",
        "color",
        "colour",
        "season",
        "material",
        "family",
        "availability",
    }
    for part in re.split(r"\s+and\s+", body):
        part = part.strip()
        if not part:
            continue
        matched = False
        for key in sorted(known, key=len, reverse=True):
            if part.casefold().startswith(key + " "):
                out[key.replace(" ", "_")] = part[len(key) :].strip()
                matched = True
                break
        if not matched:
            bits = part.split(None, 1)
            if len(bits) == 2:
                out[bits[0].casefold()] = bits[1]
    return out


def _split_sections(text: str) -> list[tuple[str, str]]:
    lines = (text or "").splitlines()
    if not any(_SECTION_HEAD_RE.match(ln.strip()) for ln in lines):
        return [("", text or "")]
    sections: list[tuple[str, str]] = []
    heading = ""
    buf: list[str] = []
    for ln in lines:
        match = _SECTION_HEAD_RE.match(ln.strip())
        if match:
            if heading or any(x.strip() for x in buf):
                sections.append((heading, "\n".join(buf).strip()))
            heading = match.group(2).strip()
            buf = []
            continue
        buf.append(ln)
    if heading or any(x.strip() for x in buf):
        sections.append((heading, "\n".join(buf).strip()))
    return sections


def _parse_md_tables(text: str) -> list[list[dict[str, str]]]:
    lines = (text or "").splitlines()
    tables: list[list[dict[str, str]]] = []
    i = 0
    in_fence = False
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            i += 1
            continue
        if in_fence or not stripped.startswith("|"):
            i += 1
            continue
        block: list[str] = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            block.append(lines[i].strip())
            i += 1
        if len(block) < 2:
            continue
        headers = [c.strip().casefold() for c in block[0].strip("|").split("|")]
        rows: list[dict[str, str]] = []
        for ln in block[1:]:
            if re.match(r"^\|\s*:?-{3,}", ln):
                continue
            cells = [c.strip() for c in ln.strip("|").split("|")]
            if len(cells) != len(headers):
                continue
            rows.append({headers[j]: cells[j] for j in range(len(headers))})
        if rows:
            tables.append(rows)
    return tables


def _extract_all_tables(text: str, *, max_rows_per_table: int = 40) -> list[dict[str, Any]]:
    """Preserve every Markdown table from findings along with its section heading."""

    lines = (text or "").splitlines()
    tables: list[dict[str, Any]] = []
    current_heading = ""
    in_fence = False
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            i += 1
            continue
        if in_fence:
            i += 1
            continue
        h_match = _SECTION_HEAD_RE.match(stripped)
        if h_match:
            current_heading = h_match.group(2).strip()
            i += 1
            continue
        if not stripped.startswith("|"):
            if stripped and not stripped.startswith(("-", "*")) and len(stripped) <= 80 and ":" not in stripped:
                # Light subtitle tracking when a bold title sits right above a table
                if stripped.startswith("**") and stripped.endswith("**"):
                    current_heading = stripped.strip("* ").strip()
            i += 1
            continue
        block: list[str] = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            block.append(lines[i].strip())
            i += 1
        if len(block) < 2:
            continue
        headers = [c.strip() for c in block[0].strip("|").split("|")]
        if not any(headers):
            continue
        rows: list[list[str]] = []
        for ln in block[1:]:
            if re.match(r"^\|\s*:?-{3,}", ln):
                continue
            cells = [c.strip() for c in ln.strip("|").split("|")]
            if len(cells) != len(headers):
                continue
            rows.append(cells)
            if len(rows) >= max_rows_per_table:
                break
        if rows:
            tables.append(
                {
                    "title": current_heading,
                    "headers": headers,
                    "rows": rows,
                }
            )
    return tables


def _is_ranking_or_breakdown_table(keys: set[str]) -> bool:
    if "rank" in keys:
        return True
    if any(k in keys for k in ("count", "share %", "share", "product_count", "total")):
        return True
    if "products" in keys and len(keys) <= 4:
        return True
    return False


def _extract_breakdown(body: str) -> list[dict]:
    out: list[dict] = []
    count_candidates = ("count", "products", "product_count", "total", "styles", "n")
    skip_label_cols = set(count_candidates) | {"rank", "share %", "share", "share_pct", "%"}
    for table in _parse_md_tables(body):
        if not table:
            continue
        ordered_keys = list(table[0].keys())
        keys = set(ordered_keys)
        count_key = next((k for k in count_candidates if k in keys), None)
        if not count_key:
            continue
        label_key = next(
            (k for k in ("name", "brand", "value", "category", "material", "color", "colour", "family") if k in keys),
            None,
        )
        if not label_key:
            label_key = next((k for k in ordered_keys if k not in skip_label_cols), None)
        if not label_key:
            continue
        for row in table:
            name = str(row.get(label_key) or "").strip()
            if name.casefold() in _NULLISH:
                continue
            try:
                count = int(str(row.get(count_key) or "0").replace(",", ""))
            except ValueError:
                continue
            share_raw = row.get("share %") or row.get("share") or ""
            try:
                share = float(str(share_raw).replace("%", "").strip()) if share_raw else None
            except ValueError:
                share = None
            item: dict[str, Any] = {"value": name, "count": count}
            if share is not None:
                item["share_pct"] = share
            out.append(item)
        if out:
            break
    return out


_ANALYSIS_TOTAL_RE = re.compile(
    r"Product analysis for\s+(?P<who>.+?):\s*\*?\*?(?P<n>[\d,]+)\*?\*?\s*products",
    re.IGNORECASE,
)
_LEADING_VALUE_RE = re.compile(
    r"^Leading value:\s*(?P<name>.+?)\s*\((?P<count>[\d,]+)\s*,\s*(?P<share>[\d.]+)\s*%\)\s*\.?\s*$",
    re.IGNORECASE,
)
_LEADING_FACET_RE = re.compile(
    r"^Leading\s+(?P<facet>[A-Za-z][A-Za-z0-9 _/-]{0,40}):\s*(?P<name>.+?)\s*"
    r"\((?P<count>[\d,]+)(?:\s+of\s+[\d,]+)?(?:\s*,\s*(?P<share>[\d.]+)\s*%)?\)\s*\.?\s*$",
    re.IGNORECASE,
)
_LEADING_ARROW_RE = re.compile(
    r"(?P<facet>[A-Za-z][A-Za-z0-9 _/-]{0,40})→(?P<name>[^(]+?)\s*\((?P<count>[\d,]+)(?:\s*,\s*(?P<share>[\d.]+)\s*%)?\)"
)


def _analysis_context_from_findings(text: str) -> list[str]:
    """Lift analysis overview, metrics, and leading attributes into card.context."""

    out: list[str] = []
    seen: set[str] = set()

    def _add(line: str) -> None:
        key = line.casefold().strip()
        if not key or key in seen:
            return
        seen.add(key)
        out.append(line)

    for match in _ANALYSIS_TOTAL_RE.finditer(text or ""):
        who = re.sub(r"\*+", "", match.group("who")).strip()
        _add(f"overview: {who} products={match.group('n').replace(',', '')}")

    in_metrics = False
    for ln in (text or "").splitlines():
        low = ln.casefold()
        if ln.startswith("|") and "metric" in low and "value" in low:
            in_metrics = True
            continue
        if in_metrics:
            if not ln.startswith("|"):
                in_metrics = False
                continue
            if "---" in ln:
                continue
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if len(cells) >= 2 and cells[0].casefold() not in {"metric", ""}:
                _add(f"metric: {cells[0]}={cells[1]}")

    section: str | None = None
    in_chart = False
    for ln in (text or "").splitlines():
        if ln.startswith("```"):
            in_chart = not in_chart
            continue
        if in_chart:
            continue
        if ln.startswith("### "):
            section = ln[4:].strip()
            continue
        lead = _LEADING_VALUE_RE.match(ln.strip())
        if lead and section:
            _add(
                f"leading {section}: {lead.group('name').strip()}="
                f"{lead.group('count').replace(',', '')} ({lead.group('share')}%)"
            )
            section = None
            continue
        facet_lead = _LEADING_FACET_RE.match(ln.strip())
        if facet_lead and facet_lead.group("facet").casefold() != "value":
            share = facet_lead.group("share")
            share_bit = f" ({share}%)" if share else ""
            _add(
                f"leading {facet_lead.group('facet').strip()}: "
                f"{facet_lead.group('name').strip()}="
                f"{facet_lead.group('count').replace(',', '')}{share_bit}"
            )
            continue
        for arrow in _LEADING_ARROW_RE.finditer(ln):
            share = arrow.group("share")
            share_bit = f" ({share}%)" if share else ""
            _add(
                f"leading {arrow.group('facet').strip()}: {arrow.group('name').strip()}="
                f"{arrow.group('count').replace(',', '')}{share_bit}"
            )
    return out[:32]


def _extract_examples(body: str) -> list[dict]:
    rows: list[dict] = []
    for table in _parse_md_tables(body):
        keys = set(table[0].keys()) if table else set()
        if _is_ranking_or_breakdown_table(keys):
            continue
        if "name" not in keys and "product name" not in keys and "title" not in keys:
            continue
        if not (keys & {"price", "availability", "brand", "category", "material", "color", "colour"}):
            continue
        for row in table:
            rows.append(
                {
                    "name": row.get("name") or row.get("product name") or row.get("title") or "",
                    "price": row.get("price") or "",
                    "availability": row.get("availability") or "",
                }
            )
    return clean_example_rows(rows)


def _requested_n(question: str) -> int | None:
    match = re.search(r"\btop\s+(\d+)\b", question or "", flags=re.I)
    if match:
        return int(match.group(1))
    match = re.search(r"\b(\d+)\s+example", question or "", flags=re.I)
    if match:
        return int(match.group(1))
    return None


def _context_from_evidence(
    evidence: list[Any],
    *,
    filters: dict[str, str] | None = None,
    result_totals: set[int] | None = None,
) -> list[str]:
    out: list[str] = []
    totals = {int(t) for t in (result_totals or set()) if t is not None}
    filt = {str(k).casefold(): str(v).casefold() for k, v in (filters or {}).items() if v}
    filt_keys = set(filt)
    for item in evidence or []:
        record_id = str(getattr(item, "record_id", "") or "").strip().casefold()
        title = str(getattr(item, "title", "") or "").strip().casefold()
        is_meta = record_id in {"optional_metadata", "meta"} or title.startswith("kg filter meta")
        if not is_meta:
            continue
        excerpt = str(getattr(item, "excerpt", "") or "").strip()
        if not excerpt:
            continue
        bits = [b.strip() for b in excerpt.split(";") if b.strip()]
        for bit in bits or [excerpt]:
            low = bit.casefold()
            attr_prefix = ""
            attr_m = re.match(r"^([a-z0-9_ ]+)\s*:", bit, re.I)
            if attr_m:
                attr_prefix = attr_m.group(1).strip().casefold().replace(" ", "_")
            same_dim = bool(
                attr_prefix
                and filt_keys
                and (
                    attr_prefix in filt_keys
                    or any(
                        attr_prefix == k
                        or attr_prefix.endswith("_" + k)
                        or k.endswith("_" + attr_prefix)
                        for k in filt_keys
                    )
                )
            )
            # Never surface facet top-lists beside a filtered count — competing values
            # (Black=5433 next to white=2292) inflate the answer and steal the headline.
            if re.search(r"(?i)\btop\s*:", bit):
                continue
            # Drop unrelated global mix shards; keep products_total / unique= only.
            if filt and attr_prefix and not same_dim and "products_total" not in low:
                if "unique" not in low:
                    continue
            for match in re.finditer(r"(?i)(?:^|[,\s])([a-z0-9][\w'’&./ -]{0,40}?)=(\d[\d,]*)", bit):
                name, raw_n = match.group(1).strip(), match.group(2).replace(",", "")
                try:
                    n = int(raw_n)
                except ValueError:
                    continue
                if n in totals or n <= 10:
                    continue
                brand = filt.get("brand") or filt.get("buyer")
                if brand and brand in name.casefold() and "unique" not in low:
                    bit = ""
                    break
            if not bit:
                continue
            if bit not in out:
                out.append(bit)
    return out[:16]


def _example_from_evidence_item(item: Any) -> dict[str, str] | None:
    record_id = str(getattr(item, "record_id", "") or "").strip()
    if record_id in {
        "kg_stats",
        "count",
        "group_by",
        "breakdown",
        "meta",
        "optional_metadata",
        "kg filter meta reference",
        "text2cypher",
    } or record_id.startswith("text2cypher_"):
        return None
    title = str(getattr(item, "title", "") or "").strip()
    excerpt = str(getattr(item, "excerpt", "") or "").strip()
    if title.casefold() in _NULLISH or title.casefold().startswith("kg filter"):
        title = ""
    name = title
    price = ""
    avail = ""
    parts = [p.strip() for p in excerpt.split("|")] if "|" in excerpt else []
    if len(parts) >= 2:
        candidate = parts[1] if len(parts) > 1 else parts[0]
        if candidate and candidate.casefold() not in _NULLISH:
            name = name or candidate
        for part in parts[2:]:
            if re.search(r"[$€£]|USD|EUR|GBP|\d+\.\d{2}", part) and not price:
                price = part
            elif part.casefold() in {"instock", "in stock", "outofstock", "out of stock"}:
                avail = part
    if not name:
        name = excerpt.split("|")[0].strip() if excerpt else ""
    if not name or name.casefold() in _NULLISH:
        return None
    if re.fullmatch(r"\d{6,}", name) or re.fullmatch(r"[A-Z]{0,3}\d{5,}", name):
        return None
    return {"name": name, "price": price, "availability": avail}


def _results_from_evidence(evidence: list[Any], *, example_limit: int = 15) -> list[SourceResult]:
    by_source: dict[str, SourceResult] = {}
    raw_examples: dict[str, list[dict[str, str]]] = {}
    for item in evidence or []:
        source_id = str(getattr(item, "graph", None) or getattr(item, "source_id", None) or "").strip()
        if source_id not in SOURCE_LABELS:
            continue
        excerpt = str(getattr(item, "excerpt", "") or "")
        record_id = str(getattr(item, "record_id", "") or "")
        title = str(getattr(item, "title", "") or "")
        total = None
        match = _PRODUCTS_EQ_RE.search(excerpt)
        if match:
            total = int(match.group(1).replace(",", ""))
        else:
            match = re.search(r"(?i)(?:→|->|=)\s*(\d[\d,]*)", excerpt)
            if match and record_id in {"kg_stats", "count", "group_by"}:
                total = int(match.group(1).replace(",", ""))
        slot = by_source.setdefault(source_id, SourceResult(source_id=source_id))
        if total is not None:
            slot.total = total
        if not slot.breakdown and (
            record_id in {"kg_stats", "count", "group_by", "breakdown"}
            or "breakdown" in title.casefold()
        ):
            bd = _usable_breakdown(_extract_breakdown(excerpt))
            if bd:
                slot.breakdown = bd
        example = _example_from_evidence_item(item)
        if example:
            raw_examples.setdefault(source_id, []).append(example)
    for source_id, rows in raw_examples.items():
        slot = by_source.setdefault(source_id, SourceResult(source_id=source_id))
        if not slot.examples:
            slot.examples = clean_example_rows(rows, limit=example_limit)
    return list(by_source.values())


def _results_from_findings_text(text: str) -> list[SourceResult]:
    """Extract source totals, breakdowns, and examples without dropping non-source headings."""

    sections = _split_sections(text)
    results: list[SourceResult] = []
    used_ids: set[str] = set()
    for heading, body in sections:
        source_id = _source_id_from_label(heading) if heading else None
        count_hit = _ROBOTIC_COUNT_RE.search(body or "")
        total = None
        if count_hit:
            total = int(count_hit.group("n").replace(",", ""))
        else:
            match = _PRODUCTS_EQ_RE.search(body or "")
            if match:
                total = int(match.group(1).replace(",", ""))
        if source_id is None:
            body_low = (body or "").casefold()
            if "plmex" in body_low or re.search(r"\bplm\b", body_low):
                source_id = "plmex_products"
            elif "e-commerce" in body_low or "ecommerce" in body_low or "ecom_" in body_low:
                source_id = "ecom_products"
            else:
                source_id = ""

        breakdown = _usable_breakdown(_extract_breakdown(body))
        examples = _extract_examples(body)

        if not source_id:
            if total is None and not breakdown and not examples:
                continue
            existing_empty = next((r for r in results if not r.source_id), None)
            if existing_empty is not None:
                if total is not None and existing_empty.total is None:
                    existing_empty.total = total
                if breakdown and not existing_empty.breakdown:
                    existing_empty.breakdown = breakdown
                if examples and not existing_empty.examples:
                    existing_empty.examples = examples
            else:
                results.append(
                    SourceResult(
                        source_id="",
                        total=total,
                        breakdown=breakdown,
                        examples=examples,
                    )
                )
            continue

        if source_id in used_ids:
            existing = next(r for r in results if r.source_id == source_id)
            if total is not None and existing.total is None:
                existing.total = total
            if not existing.breakdown and breakdown:
                existing.breakdown = breakdown
            if not existing.examples and examples:
                existing.examples = examples
            continue

        used_ids.add(source_id)
        results.append(
            SourceResult(
                source_id=source_id,
                total=total,
                breakdown=breakdown,
                examples=examples,
            )
        )

    if len(results) == 2 and results[0].total == results[1].total == 0:
        if results[0].source_id == results[1].source_id:
            return results[:1]
    return results


def _preferred_source_id(routes: list[str], evidence: list[Any] | None = None) -> str:
    for route_name in routes or []:
        low = str(route_name).casefold()
        if "ecom" in low:
            return "ecom_products"
        if "plmex" in low or low == "plm" or low.endswith("_plm"):
            return "plmex_products"
    for item in evidence or []:
        gid = str(getattr(item, "graph", None) or getattr(item, "source_id", None) or "")
        if gid in SOURCE_LABELS:
            return gid
    return ""


def _usable_breakdown(rows: list[dict]) -> list[dict]:
    if len(rows) == 1 and rows[0].get("share_pct") == 100:
        return []
    return list(rows)


def _merge_orphan_breakdown(
    results: list[SourceResult],
    findings_text: str,
    *,
    routes: list[str],
    evidence: list[Any] | None = None,
) -> None:
    if not findings_text or not results:
        return
    if all(r.breakdown for r in results):
        return
    orphan = _usable_breakdown(_extract_breakdown(findings_text))
    if not orphan:
        return
    preferred = _preferred_source_id(routes, evidence)
    target = next((r for r in results if r.source_id == preferred and not r.breakdown), None)
    if target is None:
        target = next((r for r in results if not r.breakdown), None)
    if target is None:
        return
    if not target.source_id and preferred:
        target.source_id = preferred
    target.breakdown = orphan


def _bind_empty_source_ids(
    results: list[SourceResult],
    *,
    routes: list[str],
    evidence: list[Any] | None = None,
) -> None:
    preferred = _preferred_source_id(routes, evidence)
    if not preferred:
        return
    for row in results:
        if not row.source_id:
            row.source_id = preferred


def build_answer_card(
    question: str,
    route: str | list[str] | None,
    filters: dict[str, str] | None,
    retrieval_results: Any,
    *,
    response_type: str | None = None,
    group_by: str | None = None,
    limit: int | None = None,
) -> AnswerCard:
    """Build a rich structured AnswerCard preserving findings, tables, and caveats."""

    routes = route if isinstance(route, list) else ([route] if route else [])
    filt = {str(k): str(v) for k, v in (filters or {}).items() if v not in (None, "")}
    findings_text = ""
    evidence: list[Any] = []
    caveats: list[str] = []
    resolved: dict = {}

    if hasattr(retrieval_results, "text"):
        findings_text = str(getattr(retrieval_results, "text", "") or "")
        evidence = list(getattr(retrieval_results, "evidence", None) or [])
        caveats = [
            str(c).strip()
            for c in (getattr(retrieval_results, "caveats", None) or [])
            if str(c or "").strip()
            and "Retrieved via LLM Text-to-Cypher" not in str(c)
        ]
        resolved = dict(getattr(retrieval_results, "resolved", None) or {})
        response_type = response_type or getattr(retrieval_results, "response_type", None)
        if not filt:
            filt = {
                str(k): str(v)
                for k, v in (resolved.get("filters") or {}).items()
                if v not in (None, "")
            }
    elif isinstance(retrieval_results, str):
        findings_text = retrieval_results
    elif isinstance(retrieval_results, list) and retrieval_results and isinstance(retrieval_results[0], SourceResult):
        intent = intent_from_routes(routes, question=question, response_type=response_type, group_by=group_by)
        results_list = list(retrieval_results)
        return AnswerCard(
            question=question,
            intent=intent,
            filters=filt,
            results=results_list,
            requested_n=limit or _requested_n(question),
            derived=_derived_facts(results_list, filt, intent=intent),
            depth=depth_from_question(question, intent=intent),
        )

    for _heading, body in _split_sections(findings_text):
        hit = _ROBOTIC_COUNT_RE.search(body or "")
        if hit and hit.group("phrase"):
            parsed_filt = _parse_filter_phrase(hit.group("phrase"))
            for key, value in parsed_filt.items():
                if value and not filt.get(key):
                    filt[key] = value
            if not filt:
                filt.update(parsed_filt)
            break

    results = _results_from_evidence(evidence)
    if not results:
        results = _results_from_findings_text(findings_text)
        _bind_empty_source_ids(results, routes=routes, evidence=evidence)
    elif findings_text:
        parsed = _results_from_findings_text(findings_text)
        by_id = {r.source_id: r for r in results if r.source_id}
        for parsed_row in parsed:
            if not parsed_row.source_id:
                preferred = _preferred_source_id(routes, evidence)
                slot = by_id.get(preferred) if preferred else None
                if slot is None and len(results) == 1:
                    slot = results[0]
                if slot is None:
                    if preferred:
                        parsed_row.source_id = preferred
                        results.append(parsed_row)
                        by_id[preferred] = parsed_row
                    continue
                if slot.total is None and parsed_row.total is not None:
                    slot.total = parsed_row.total
                if not slot.breakdown and parsed_row.breakdown:
                    slot.breakdown = parsed_row.breakdown
                if not slot.examples and parsed_row.examples:
                    slot.examples = parsed_row.examples
                continue
            slot = by_id.get(parsed_row.source_id)
            if slot is None:
                results.append(parsed_row)
                by_id[parsed_row.source_id] = parsed_row
                continue
            if slot.total is None:
                slot.total = parsed_row.total
            if not slot.breakdown:
                slot.breakdown = parsed_row.breakdown
            if not slot.examples:
                slot.examples = parsed_row.examples
        if all(r.total is None for r in results):
            hit = _ROBOTIC_COUNT_RE.search(findings_text)
            if hit:
                n = int(hit.group("n").replace(",", ""))
                preferred = _preferred_source_id(routes, evidence)
                target = next((r for r in results if r.source_id == preferred), None) if preferred else None
                if target is None and len(results) == 1:
                    target = results[0]
                if target is not None:
                    target.total = n

    if results and findings_text:
        _merge_orphan_breakdown(results, findings_text, routes=routes, evidence=evidence)

    if not results and findings_text:
        hit = _ROBOTIC_COUNT_RE.search(findings_text)
        total = int(hit.group("n").replace(",", "")) if hit else None
        if total is None:
            match = _PRODUCTS_EQ_RE.search(findings_text)
            total = int(match.group(1).replace(",", "")) if match else None
        source_id = ""
        for item in evidence:
            gid = str(getattr(item, "graph", None) or getattr(item, "source_id", None) or "")
            if gid in SOURCE_LABELS:
                source_id = gid
                break
        if not source_id:
            source_id = _preferred_source_id(routes, evidence)
        if not source_id:
            body_low = findings_text.casefold()
            if "plm" in body_low or "plmex" in body_low:
                source_id = "plmex_products"
            elif "e-commerce" in body_low or "ecommerce" in body_low:
                source_id = "ecom_products"
            else:
                source_id = "plmex_products"
        results = [SourceResult(source_id=source_id, total=total)]

    data_notes: list[str] = []
    for result in results:
        for ex in result.examples:
            price = str(ex.get("price") or "")
            if _MALFORMED_PRICE_RE.search(price):
                data_notes.append(f"price {price} looks like a data error")
                ex["price"] = ""

    intent = intent_from_routes(
        routes,
        question=question,
        response_type=response_type,
        group_by=group_by,
    )
    requested = limit or _requested_n(question)
    if intent == "examples" and requested and evidence:
        enriched = _results_from_evidence(evidence, example_limit=max(requested, 10))
        by_id = {r.source_id: r for r in results}
        for slot in enriched:
            existing = by_id.get(slot.source_id)
            if existing is None:
                results.append(slot)
                continue
            if slot.examples and len(slot.examples) > len(existing.examples or []):
                existing.examples = slot.examples
            if existing.total is None and slot.total is not None:
                existing.total = slot.total

    result_totals = {int(r.total) for r in results if r.total is not None}
    all_zero = bool(results) and all(r.total == 0 for r in results if r.total is not None)
    context = [] if all_zero else _context_from_evidence(
        evidence,
        filters=filt,
        result_totals=result_totals,
    )
    if findings_text:
        for line in _analysis_context_from_findings(findings_text):
            if line not in context:
                context.append(line)
        if intent in {"analysis", "compare", "breakdown", "report", "spec_brief"}:
            overview_totals = list(_ANALYSIS_TOTAL_RE.finditer(findings_text))
            if overview_totals and (
                not results
                or all((r.total is None or not r.breakdown) for r in results)
            ):
                for match in overview_totals:
                    n = int(match.group("n").replace(",", ""))
                    who = re.sub(r"\*+", "", match.group("who")).strip()
                    target = None
                    who_low = who.casefold()
                    for r in results:
                        label = SOURCE_LABELS.get(r.source_id, "").casefold()
                        if who_low in label or label in who_low:
                            target = r
                            break
                    if target is None and results:
                        target = results[0]
                    if target is None:
                        source_id = _preferred_source_id(routes, evidence) or "ecom_products"
                        target = SourceResult(source_id=source_id, total=n)
                        results.append(target)
                    elif target.total is None:
                        target.total = n

    tables = _extract_all_tables(findings_text) if findings_text else []
    trimmed_findings = findings_text.strip()
    if len(trimmed_findings) > 16000:
        trimmed_findings = trimmed_findings[:16000].rstrip() + "\n...(truncated)"

    derived = _derived_facts(results, filt, intent=intent)
    depth = depth_from_question(question, intent=intent)
    return AnswerCard(
        question=question,
        intent=intent,
        filters=filt,
        results=results,
        requested_n=requested,
        data_notes=data_notes,
        context=context,
        derived=derived,
        depth=depth,
        tables=tables,
        findings=trimmed_findings,
        caveats=caveats[:8],
    )