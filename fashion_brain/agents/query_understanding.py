"""Context-aware query understanding with progressive, metadata-grounded clarification.

Runs before Intention / retrieval. LLM reads long or ambiguous asks (ChatGPT-style),
asks at most one follow-up per turn when needed, and persists understanding on the
conversation session. Precise short asks can pass through without an LLM call.
Heuristic paths remain as fallback when the model is unavailable.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from knowledge_graph.ontology import normalize_name

from fashion_brain.agents.contracts import (
    ClarificationState,
    ClarifyResult,
    ContextDecision,
    IntentSummary,
    Intention,
    ReadyResult,
    ValidatedIntent,
)
from fashion_brain.llm import build_model_provider

LOGGER = logging.getLogger("fashion_brain.agents.query_understanding")

# Prefer these analytical dimensions when present in source metadata.
_PREFERRED_DIMENSIONS = (
    "category",
    "garment_type",
    "color",
    "gender",
    "material",
    "composition",
    "season",
    "price",
    "product_type",
    "family",
    "subfamily",
    "department",
    "gauge",
)

# Require an analysis/report noun after full/complete — bare "full price" must not match.
_FULL_ANALYSIS_RE = re.compile(
    r"\b(full|complete|comprehensive|overall|entire|whole)\s+"
    r"(analysis|overview|profile|report|study)\b"
    r"|\b(overview|everything|all\s+aspects)\b"
    r"|\bmake\s+(?:a\s+)?(?:full\s+)?(?:analysis|report|overview)\b"
    r"|\b(?:final|full)\s+report\b"
    r"|\brun\s+(?:the\s+)?(?:analysis|report)\b"
    r"|\bgo\s+(?:ahead|through)\b"
    r"|\bgenerate\s+(?:the\s+)?(?:analysis|report)\b",
    re.IGNORECASE,
)

_ANALYSIS_RE = re.compile(
    r"\b(analy[sz]e|analy[sz]ing|analysis|profile|review|overview|study)\b",
    re.IGNORECASE,
)

_PRECISE_METRIC_RE = re.compile(
    r"\b(how\s+many|count|number\s+of|average|avg|mean|sum|total|"
    r"list|show\s+me|top\s+\d+|rank|compare|vs\.?|versus|"
    r"percentage|percent|distribution\s+of|"
    r"breakdown|break\s*down|\w[\w-]*\s*-?\s*wise)\b",
    re.IGNORECASE,
)

_SCOPE_HINTS = {
    "category": ("category", "categories"),
    "garment_type": ("garment", "garment type", "product type", "type"),
    "color": ("color", "colour", "colors", "colours"),
    "gender": ("gender", "men", "women", "unisex", "kids"),
    "material": ("material", "fabric", "composition", "fibre", "fiber"),
    "season": ("season", "seasonal", "ss", "aw", "fw"),
    "price": ("price", "pricing", "cost"),
    "department": ("department", "section"),
    "family": ("family", "subfamily"),
    "gauge": ("gauge",),
}

_STOP_ENTITY = frozenset(
    normalize_name(w)
    for w in (
        "analyze analysis analysing analyzing analyse profile review overview study "
        "the a an of for about please show me what which how many products product "
        "data dataset brand buyer category are is was were be been being in on at to "
        "from with without into onto over under within across count number total list "
        "give tell find get check see look share shares shared using use used "
        "possible possibly impossible maybe perhaps able unable capability capable "
        "feasible potential might could would should can that this price trend "
        "trends trending history historical analysis"
    ).split()
)

# Catalog / source names — never brands. "PLM data" means plmex_products, not brand=plm.
_GRAPH_SCOPE_NOISE = frozenset(
    normalize_name(w)
    for w in (
        "plm plmex plm-ex ecom ecomm ecommerce commerce commerces own own_products "
        "ecom_products plmex_products catalog catalogue database graph graphs "
        "knowledge kg source sources system"
    ).split()
)

_ALLOWED_SLOTS = frozenset(
    {
        "entity",
        "scope_or_direction",
        "analysis_depth",
        "focused_dimension",
        "source",
        "final_confirm",
    }
)

QU_SYSTEM = """\
You are the Query Understanding agent for a fashion Knowledge Graph assistant.
Read the latest user message (which may be a long story), conversation history, and
session memory. Decide whether the analytical intent is ready to execute or needs
ONE clarification question.

Return ONE JSON object only:
{
  "status": "ready" | "clarify",
  "source": "ecom_products" | "plmex_products" | "own_products" | null,
  "entity": "brand or buyer name or null",
  "filters": {"brand":"...","category":"..."},
  "objective": "short objective",
  "metrics": ["count"] or [],
  "group_by": null or attribute name,
  "analysis_depth": "unset" | "focused" | "full",
  "comparison": [],
  "effective_question": "self-contained question for retrieval",
  "summary": "one sentence of what you understood",
  "clarification_slot": "entity|scope_or_direction|analysis_depth|focused_dimension|final_confirm|source|null",
  "clarification_question": "one metadata-grounded question or null",
  "offered_options": ["category","color"]
}

Rules:
- Prefer clarifying only when a critical ambiguity would change the result.
- Never invent attributes not listed in available_dimensions / metadata.
- Graph-scope words (PLM, ecom, ecommerce, catalog) select source — never brand.
- Quantity asks (how many, totals, size of catalog, product count) → metrics:["count"], status ready when source/filters are clear.
- Clear "analyze X" / "analysis X" with a named brand/buyer → status ready, analysis_depth=full.
  Do NOT ask final_confirm / "reply yes" / "if this looks accurate". Run the analysis in one shot.
  Do NOT ask "what would you like to explore" / invent a dimension quiz when the party is already clear.
- Only clarify entity when the analysis target is missing. Only use scope_or_direction when the user is mid-trail adding filters.
- When filters were collected and the user says "make analysis" / "final report" / "go ahead" /
  "yes" → status ready with analysis_depth=full (no second confirmation question).
- When continuing a pending clarification, interpret free-form answers against that slot and MERGE into filters.
- clarification_question should briefly restate the resolved ask so far, then ask the next choice.
- Ask at most one question. No markdown outside JSON.
"""


def _tokens(text: str) -> list[str]:
    return [t for t in normalize_name(text or "").split() if t]


def _is_graph_scope_token(token: str) -> bool:
    t = normalize_name(token)
    if not t:
        return False
    if t in _GRAPH_SCOPE_NOISE:
        return True
    return any(t == noise or t.startswith(noise + "_") for noise in ("plm", "plmex", "ecom", "own"))


def _detect_graph_id(question: str) -> str | None:
    """Map question tokens like 'PLM data' to a product graph id."""

    tokens = set(_tokens(question))
    if tokens & {"plm", "plmex"} or "plmex_products" in tokens:
        return "plmex_products"
    if tokens & {"ecom", "ecomm", "ecommerce", "commerce"} or "ecom_products" in tokens:
        return "ecom_products"
    if "own_products" in tokens or (tokens & {"own"} and "products" in tokens):
        return "own_products"
    return None


def _label(attr: str) -> str:
    return str(attr or "").replace("_", " ").strip()


def _match_option(text: str, options: list[str]) -> str | None:
    raw = (text or "").strip()
    if not raw:
        return None
    low = raw.casefold()
    norm = normalize_name(raw)
    for opt in options:
        if not opt:
            continue
        if opt.casefold() == low or normalize_name(opt) == norm:
            return opt
    for opt in options:
        on = normalize_name(opt)
        if on and (on in norm or norm in on):
            return opt
    return None


def _is_full_analysis(text: str) -> bool:
    return bool(_FULL_ANALYSIS_RE.search(text or ""))


def _is_confirm_yes(text: str) -> bool:
    from fashion_brain.agents.qu_fast import classify_reply

    kind, _ = classify_reply(text)
    return kind == "yes"


def _is_confirm_no(text: str) -> bool:
    from fashion_brain.agents.qu_fast import classify_reply

    kind, _ = classify_reply(text)
    return kind == "no"


def _append_filter_trail(confirmed: dict, *, turn: int, updates: dict) -> list[dict]:
    """Accumulate step-by-step filter confirmations for the analysis thread."""

    trail = list(confirmed.get("filter_trail") or [])
    for key, value in (updates or {}).items():
        if value in (None, ""):
            continue
        key_s = str(key)
        val_s = str(value)
        # Deduplicate consecutive identical entries.
        if trail and trail[-1].get("key") == key_s and str(trail[-1].get("value")) == val_s:
            continue
        trail.append({"turn": int(turn or 0), "key": key_s, "value": val_s})
    confirmed["filter_trail"] = trail[-30:]
    return confirmed["filter_trail"]


def _restated_analysis_ask(
    *,
    original: str = "",
    filters: dict | None = None,
    group_by: str | None = None,
    analysis_depth: str = "unset",
) -> str:
    """Human restatement of the growing analysis ask (original + confirmed Scope)."""

    filt = {str(k): v for k, v in (filters or {}).items() if v not in (None, "")}
    party = filt.get("brand") or filt.get("buyer")
    bits: list[str] = []
    if party:
        bits.append(str(party))
    for key, value in filt.items():
        if key in {"brand", "buyer"}:
            continue
        bits.append(f"{_label(key)}={value}")
    if group_by:
        bits.append(f"by {_label(str(group_by))}")
    if analysis_depth == "full":
        prefix = "full analysis of"
    elif analysis_depth == "focused":
        prefix = "focused analysis of"
    else:
        prefix = "analyze"
    if bits:
        return f"Resolved ask: {prefix} " + "; ".join(bits) + "."
    ask = " ".join(str(original or "").split())
    if ask:
        return f"Resolved ask: {ask}."
    return "Resolved ask: catalog analysis."


def _format_filter_trail(confirmed: dict | None) -> str:
    trail = list((confirmed or {}).get("filter_trail") or [])
    filters = dict((confirmed or {}).get("filters") or {})
    lines: list[str] = []
    if trail:
        for step in trail[-12:]:
            key = _label(str(step.get("key") or ""))
            val = step.get("value")
            if key and val not in (None, ""):
                lines.append(f"- {key} = {val}")
    elif filters:
        for key, val in filters.items():
            if val not in (None, ""):
                lines.append(f"- {_label(str(key))} = {val}")
    if not lines:
        return ""
    return "Confirmed so far:\n" + "\n".join(lines)


def _basic_entity_brief(
    platform,
    *,
    entity: str | None,
    filters: dict | None = None,
    graph_id: str | None = None,
) -> str:
    """One short metadata-backed basics line (count / scope) for progressive analysis."""

    from fashion_brain.agents.qu_fast import cached

    filt = {str(k): v for k, v in (filters or {}).items() if v not in (None, "")}
    if entity and "brand" not in filt and "buyer" not in filt:
        filt = {**filt, "brand": entity}
    gid = graph_id
    if not gid:
        try:
            from fashion_brain.ask import preferred_graphs_from_question

            named = preferred_graphs_from_question(
                " ".join(str(v) for v in filt.values()), platform
            ) or []
            gid = named[0] if named else "ecom_products"
        except Exception:
            gid = "ecom_products"
    cache_key = ("brief", str(gid), str(entity or ""), tuple(sorted(filt.items())))

    def _compute() -> str:
        try:
            from knowledge_graph.retrieval.kg_stats import answer_simple_count, stats_available

            generated = getattr(getattr(platform, "settings", None), "generated", None)
            use_gid = str(gid)
            if not stats_available(use_gid, generated):
                for alt in ("ecom_products", "plmex_products"):
                    if stats_available(alt, generated):
                        use_gid = alt
                        break
            # Prefer a single known filter for the brief; else global for that entity brand.
            probe = dict(filt)
            if len(probe) > 1:
                party_key = (
                    "brand" if probe.get("brand") else ("buyer" if probe.get("buyer") else None)
                )
                probe = {party_key: probe[party_key]} if party_key else {}
            meta = answer_simple_count(use_gid, filters=probe, generated=generated)
            if meta is None and probe:
                meta = answer_simple_count(use_gid, filters={}, generated=generated)
            if meta is None:
                return ""
            n = int(meta.get("n") or 0)
            who = entity or (probe.get("brand") or probe.get("buyer") or "this collection")
            if meta.get("kind") == "value_count":
                return f"{who} currently has {n:,} products."
            if meta.get("kind") == "product_total":
                return f"There are {n:,} products in this collection."
            return f"There are {n:,} products."
        except Exception:
            return ""

    return cached(cache_key, _compute, ttl=300)


def _compose_progressive_clarify(
    *,
    platform,
    next_question: str,
    entity: str | None,
    filters: dict,
    confirmed: dict,
    original: str = "",
    group_by: str | None = None,
    analysis_depth: str = "unset",
    graph_id: str | None = None,
    include_basics: bool = True,
) -> str:
    """Prefer one scope line + the next question (LLM text already in user language)."""

    blocks: list[str] = []
    restated = _restated_analysis_ask(
        original=original,
        filters=filters,
        group_by=group_by or confirmed.get("focused_dimension"),
        analysis_depth=analysis_depth
        or str(confirmed.get("analysis_depth") or "unset"),
    )
    if include_basics:
        brief = _basic_entity_brief(
            platform, entity=entity, filters=filters, graph_id=graph_id
        )
        # One scope line: prefer restated ask (includes filters); else count brief.
        if restated and (
            len(filters) > 1
            or group_by
            or confirmed.get("focused_dimension")
            or not brief
        ):
            blocks.append(restated)
        elif brief:
            blocks.append(brief)
        elif restated:
            blocks.append(restated)
    ask = str(next_question or "").strip()
    if ask:
        blocks.append(ask)
    return "\n\n".join(b for b in blocks if b)


def _is_vague_analysis(text: str) -> bool:
    if _is_price_trend_ask(text):
        return False
    if not _ANALYSIS_RE.search(text or ""):
        return False
    if _PRECISE_METRIC_RE.search(text or ""):
        return False
    if re.search(r"\bby\s+\w+", text or "", re.IGNORECASE):
        return False
    return True


def _is_price_trend_ask(text: str) -> bool:
    """PriceChangeEvent trend asks — never a full product-mix analysis confirm."""

    try:
        from fashion_brain.ask import is_price_trend_analysis_ask, _is_price_history_question

        return bool(is_price_trend_analysis_ask(text) or _is_price_history_question(text))
    except Exception:
        return bool(
            re.search(
                r"\b(price\s+trend|trend(?:ing)?\s+analysis|prices?\s+(?:trend|change|history))\b",
                text or "",
                flags=re.IGNORECASE,
            )
        )


def _is_named_party_analysis(text: str) -> bool:
    """True for clear 'analysis primark' / 'analyze Zara' — run full report, no quiz."""

    # "make price trend analysis for Zara" is NOT a full assortment analysis.
    if _is_price_trend_ask(text):
        return False
    if not _is_vague_analysis(text):
        return False
    return _extract_entity_candidate(text) is not None


def _is_precise_ask(text: str, filters: dict) -> bool:
    if _PRECISE_METRIC_RE.search(text or ""):
        return True
    if not _is_vague_analysis(text):
        if filters and not _ANALYSIS_RE.search(text or ""):
            return True
        if not _ANALYSIS_RE.search(text or ""):
            return True
    return False


def _is_long_or_narrative(text: str) -> bool:
    """True for multi-sentence / long stories that need LLM understanding."""

    raw = (text or "").strip()
    if not raw:
        return False
    if len(raw) >= 280:
        return True
    tokens = raw.split()
    if len(tokens) >= 40:
        return True
    # Two or more sentence-like breaks.
    if len(re.findall(r"[.!?]+", raw)) >= 2 and len(tokens) >= 20:
        return True
    return False


def _needs_llm_understanding(question: str, *, pending: bool, confirmed: dict) -> bool:
    if pending:
        return True
    if _is_vague_analysis(question):
        return True
    if _is_long_or_narrative(question):
        return True
    # Short precise asks stay on the cheap path.
    if _is_precise_ask(question, confirmed) or not _is_vague_analysis(question):
        return False
    return True


def _extract_entity_candidate(text: str) -> str | None:
    """Heuristic brand/entity from 'Analyze PRIMARK' style asks only."""

    try:
        from fashion_brain.ask import _is_grammar_party_name, is_capability_followup_ask
    except Exception:
        _is_grammar_party_name = None  # type: ignore[assignment]
        is_capability_followup_ask = None  # type: ignore[assignment]

    # "is that possible price trend analysis?" is capability language — not a brand.
    if is_capability_followup_ask is not None and is_capability_followup_ask(text):
        return None

    cleaned = re.sub(
        r"\b(analy[sz]e|analy[sz]ing|analysis|profile|review|overview|study|"
        r"please|about|for|the|a|an|of|brand|buyer|"
        r"possible|possibly|impossible|maybe|perhaps|able|unable|"
        r"price|trend|trends|trending|history|historical)\b",
        " ",
        text or "",
        flags=re.IGNORECASE,
    )
    parts = [
        p
        for p in _tokens(cleaned)
        if p not in _STOP_ENTITY
        and not _is_graph_scope_token(p)
        and len(p) > 1
        and not (_is_grammar_party_name is not None and _is_grammar_party_name(p))
    ]
    if len(parts) == 1:
        return parts[0]
    return None


def _detect_focused_dimension(text: str, available: list[str]) -> str | None:
    low = (text or "").casefold()
    norm = normalize_name(text or "")
    for attr in available:
        label = _label(attr).casefold()
        if attr in norm or label in low or attr.replace("_", "") in norm.replace(" ", ""):
            return attr
        for hint in _SCOPE_HINTS.get(attr, ()):
            if hint in low:
                return attr
    alias_map = {
        "colour": "color",
        "pricing": "price",
        "materials": "material",
        "composition": "material",
        "categories": "category",
        "sweater": None,
    }
    for alias, target in alias_map.items():
        if alias in low and target and target in available:
            return target
    return None


def _detect_product_scope(
    text: str,
    value_options: list[str],
    *,
    platform=None,
    graph_id: str | None = None,
) -> tuple[str | None, str | None]:
    """Return (filter_key, value) when the user names a product category/type."""

    matched = _match_option(text, value_options)
    if matched:
        return "category", matched
    stripped = re.sub(
        r"\b(category|categories|products?|items?|garments?)\b",
        "",
        text or "",
        flags=re.IGNORECASE,
    ).strip(" .,")
    if stripped and stripped.casefold() != (text or "").casefold():
        matched = _match_option(stripped, value_options)
        if matched:
            return "category", matched
    # Metadata-driven match — no hardcoded garment lexicon.
    if platform is not None:
        try:
            from fashion_brain.planning.metadata_resolver import build_value_index, match_spans

            fields = ["category", "garment_type", "family", "product_type"]
            index = build_value_index(platform, fields)
            spans = match_spans(text or "", index)
            for span in spans:
                if span.field in fields and span.value:
                    return span.field if span.field != "product_type" else "category", span.value
        except Exception:
            pass
    # Fall back to offered value list as an n-gram index.
    if value_options:
        try:
            from fashion_brain.planning.metadata_resolver import match_spans

            index = {
                "category": [
                    (v, normalize_name(v)) for v in value_options if v
                ]
            }
            spans = match_spans(text or "", index)
            if spans:
                return "category", spans[0].value
        except Exception:
            pass
    return None, None


def _format_history(history: list | None, *, limit: int = 8) -> str:
    lines: list[str] = []
    for item in (history or [])[-limit:]:
        if isinstance(item, dict):
            role = str(item.get("role") or "user")
            content = str(item.get("content") or item.get("text") or "").strip()
        else:
            role = str(getattr(item, "role", "user") or "user")
            content = str(getattr(item, "content", "") or getattr(item, "text", "") or "").strip()
        if content:
            lines.append(f"{role}: {content[:500]}")
    return "\n".join(lines)


def _qu_timeout(seconds: float | None) -> float:
    if seconds is None or seconds <= 0:
        return 12.0
    return min(20.0, float(seconds))


def store_understanding(
    state: dict,
    *,
    validated: ValidatedIntent | None = None,
    clarification: ClarificationState | dict | None = None,
    summary: str | None = None,
    pending_slot: str | None = None,
) -> None:
    """Write durable understanding blob onto conversation session state."""

    clar_dict: dict = {}
    if isinstance(clarification, ClarificationState):
        clar_dict = clarification.to_dict()
    elif isinstance(clarification, dict):
        clar_dict = clarification
    elif isinstance(state.get("clarification"), dict):
        clar_dict = dict(state.get("clarification") or {})

    pending = pending_slot
    if pending is None:
        pending = clar_dict.get("pending_slot")

    intent_dict = validated.to_dict() if validated is not None else state.get("validated_intent")
    if not summary and isinstance(intent_dict, dict):
        summary = str(intent_dict.get("effective_question") or intent_dict.get("objective") or "")
    if not summary and clar_dict:
        summary = f"Awaiting clarification ({pending})" if pending else ""

    state["understanding"] = {
        "summary": summary or "",
        "validated_intent": intent_dict if isinstance(intent_dict, dict) else None,
        "pending_slot": pending,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


class QueryUnderstandingAgent:
    """Metadata-aware progressive clarification before query execution."""

    AWAITING_CAVEAT = "Awaiting clarification"

    def evaluate(
        self,
        platform,
        question: str,
        history: list | None = None,
        context: ContextDecision | None = None,
        conversation_state: dict | None = None,
        graph_id: str | None = None,
    ) -> ClarifyResult | ReadyResult:
        state = dict(conversation_state or {})
        clar = ClarificationState.from_dict(state.get("clarification"))
        ctx_filters = dict((context.active_filters if context else None) or {})
        session_filters = dict(state.get("active_filters") or {})
        pending = bool(clar and clar.pending_slot)
        try:
            from fashion_brain.ask import (
                is_capability_followup_ask,
                is_catalog_dataset_total_ask,
                sanitize_party_filters,
            )

            dataset_total = is_catalog_dataset_total_ask(question)
            capability_followup = is_capability_followup_ask(question)
        except Exception:
            dataset_total = False
            capability_followup = False
            sanitize_party_filters = None  # type: ignore[assignment]
        if not pending:
            if dataset_total or (
                context is not None
                and str(getattr(context, "intent_type", "") or "") == "new_independent_question"
                and not capability_followup
            ):
                carry_session_filters = False
                ctx_filters = {}
            elif capability_followup and (
                session_filters.get("brand") or session_filters.get("buyer") or ctx_filters
            ):
                # "is that possible price trend?" after Primark → keep prior party.
                carry_session_filters = True
            elif context is not None:
                carry_session_filters = bool(ctx_filters)
            else:
                from fashion_brain.ask import question_has_party_deixis

                carry_session_filters = question_has_party_deixis(question)
            if not carry_session_filters:
                session_filters = {}

        # Carry confirmed filters from session when continuing an analysis thread.
        confirmed: dict[str, Any] = {}
        for key in ("brand", "buyer", "category", "garment_type", "color", "gender", "material", "season"):
            for src in (ctx_filters, session_filters):
                val = src.get(key)
                if val and key not in confirmed:
                    confirmed[key] = val if isinstance(val, str) else (
                        val[0] if isinstance(val, list) and val else str(val)
                    )
        if sanitize_party_filters is not None:
            confirmed = sanitize_party_filters(confirmed)
        for key in ("brand", "buyer"):
            raw = confirmed.get(key)
            if raw and all(_is_graph_scope_token(tok) for tok in _tokens(str(raw))):
                confirmed.pop(key, None)

        # Resume pending clarification — short yes/no/option replies skip the LLM.
        if pending and clar is not None:
            from fashion_brain.agents.qu_fast import classify_reply

            kind, picked = classify_reply(question, clar.offered_options)
            # final_confirm after a price-trend original ask → run price trend, not full mix.
            if (
                clar.pending_slot == "final_confirm"
                and kind == "yes"
                and _is_price_trend_ask(str(clar.original_question or ""))
            ):
                filt = dict((clar.confirmed or {}).get("filters") or {}) or dict(confirmed)
                party = (
                    filt.get("brand")
                    or filt.get("buyer")
                    or (clar.confirmed or {}).get("entity")
                    or ""
                )
                ask = (
                    f"Summarize average price change trends for {party}"
                    if party
                    else "Summarize average price change trends"
                )
                clar.pending_slot = None
                clar.confirmed = {
                    **dict(clar.confirmed or {}),
                    "filters": filt,
                    "analysis_depth": "focused",
                    "understanding_summary": ask,
                }
                return ReadyResult(
                    intent=self._ready_intent(
                        ask,
                        filt,
                        analysis_depth="focused",
                        objective="execute",
                        available_dimensions=self._available_dimensions(
                            platform, graph_id=graph_id, question=ask
                        ),
                        entity=party or None,
                        metrics=["price"],
                    ),
                    state=clar,
                )
            if clar.pending_slot == "final_confirm" and kind == "yes":
                return self._continue(platform, "yes", clar, graph_id=graph_id)
            if clar.pending_slot == "final_confirm" and kind == "no":
                clar.pending_slot = "scope_or_direction"
                clar.turn_count = int(clar.turn_count or 0) + 1
                return ClarifyResult(
                    question=(
                        "Okay. What should I change: brand, product type, or another attribute?"
                    ),
                    state=clar,
                    slot="scope_or_direction",
                )
            if kind == "option" and picked:
                return self._continue(platform, picked, clar, graph_id=graph_id)
            if kind in {"yes", "no"} and clar.pending_slot != "final_confirm":
                # Map to a canonical token so _continue stays deterministic.
                return self._continue(
                    platform, "yes" if kind == "yes" else "no", clar, graph_id=graph_id
                )
            # Only edit / other free-form replies go to the LLM.
            # Budget cap is enforced inside _continue after scope/focus parsing fails.
            if kind in {"edit", "other"}:
                # Clarification resume contract (no per-query shortcuts):
                #   • yes / no / option → answer the pending slot (handled above)
                #   • price-trend ask → NEW turn on PriceChangeEvent (never full_analysis)
                #   • live brand hit with no party yet → answer entity via _continue
                #   • otherwise executable catalog plan with no party → NEW turn
                #   • party already confirmed → keep LLM/_continue (refinement thread)
                conf = dict(clar.confirmed or {})
                conf_filt = dict(conf.get("filters") or {})
                has_party = bool(
                    conf.get("entity") or conf_filt.get("brand") or conf_filt.get("buyer")
                )
                if _is_price_trend_ask(question):
                    fresh_state = dict(state)
                    fresh_state.pop("clarification", None)
                    fresh_state.pop("understanding", None)
                    return self.evaluate(
                        platform,
                        question,
                        history=history,
                        context=context,
                        conversation_state=fresh_state,
                        graph_id=graph_id,
                    )
                if not has_party:
                    from fashion_brain.agents.qu_fast import resolve_entity

                    brand_hit, _sugs = resolve_entity(platform, question, graph_id)
                    if brand_hit:
                        return self._continue(platform, question, clar, graph_id=graph_id)
                    try:
                        from fashion_brain.ask import enrich_scope_from_question

                        _scoped, group_by = enrich_scope_from_question(question, {})
                    except Exception:
                        group_by = None
                    if (
                        group_by
                        or _detect_graph_id(question)
                        or _PRECISE_METRIC_RE.search(question or "")
                    ):
                        fresh_state = dict(state)
                        fresh_state.pop("clarification", None)
                        fresh_state.pop("understanding", None)
                        return self.evaluate(
                            platform,
                            question,
                            history=history,
                            context=context,
                            conversation_state=fresh_state,
                            graph_id=graph_id,
                        )
                llm_hit = self._from_model(
                    platform,
                    question,
                    history,
                    state=state,
                    context=context,
                    graph_id=graph_id,
                    clar=clar,
                    confirmed=confirmed,
                )
                if llm_hit is not None:
                    return llm_hit
            return self._continue(platform, question, clar, graph_id=graph_id)

        # Turn planner already decided this turn → skip a second LLM call.
        prior_plan = state.get("turn_plan") if isinstance(state.get("turn_plan"), dict) else None
        if prior_plan and prior_plan.get("llm_ok") and not pending:
            filt = dict(prior_plan.get("filters") or confirmed or {})
            metric = str(prior_plan.get("metric") or "").casefold()
            depth = str(prior_plan.get("analysis_depth") or "unset")
            ask = str(prior_plan.get("rewritten_question") or question)
            metrics = [metric] if metric and metric not in {"other", "list", "analysis"} else []
            if metric == "analysis" or depth == "full":
                depth = "full"
            return ReadyResult(
                intent=self._ready_intent(
                    ask,
                    filt,
                    analysis_depth=depth if depth in {"full", "focused", "unset"} else "unset",
                    objective="full_analysis" if depth == "full" else "execute",
                    available_dimensions=self._available_dimensions(
                        platform, graph_id=graph_id, question=ask
                    ),
                    entity=filt.get("brand") or filt.get("buyer"),
                    group_by=prior_plan.get("group_by"),
                    metrics=metrics,
                ),
                state=ClarificationState(
                    pending_slot=None,
                    confirmed={
                        "entity": filt.get("brand") or filt.get("buyer"),
                        "filters": filt,
                        "analysis_depth": depth,
                        "understanding_summary": ask,
                    },
                    original_question=question,
                ),
            )

        # Price-trend asks with a named brand → ready immediately (PriceChangeEvent path).
        # Unscoped price-trend falls through so turn planner / clarify can ask for a party.
        if _is_price_trend_ask(question) and not pending:
            from fashion_brain.agents.qu_fast import resolve_entity

            filt = dict(confirmed)
            live_brand, _sugs = resolve_entity(platform, question, graph_id)
            named = live_brand or _extract_entity_candidate(question)
            if named and "brand" not in filt and "buyer" not in filt:
                filt["brand"] = named
            if sanitize_party_filters is not None:
                filt = sanitize_party_filters(filt)
            party = filt.get("brand") or filt.get("buyer") or named or ""
            if party:
                ask = f"Summarize average price change trends for {party}"
                return ReadyResult(
                    intent=self._ready_intent(
                        ask,
                        filt,
                        analysis_depth="focused",
                        objective="execute",
                        available_dimensions=self._available_dimensions(
                            platform, graph_id=graph_id, question=ask
                        ),
                        entity=party,
                        metrics=["price"],
                    ),
                    state=ClarificationState(
                        pending_slot=None,
                        confirmed={
                            "entity": party,
                            "filters": filt,
                            "analysis_depth": "focused",
                            "understanding_summary": ask,
                        },
                        original_question=question,
                    ),
                )

        # Long / vague / narrative → LLM first; heuristic fallback on failure.
        if _needs_llm_understanding(question, pending=False, confirmed=confirmed):
            llm_hit = self._from_model(
                platform,
                question,
                history,
                state=state,
                context=context,
                graph_id=graph_id,
                clar=None,
                confirmed=confirmed,
            )
            if llm_hit is not None:
                return llm_hit

        return self._evaluate_heuristic(
            platform,
            question,
            confirmed=confirmed,
            graph_id=graph_id,
        )

    def _evaluate_heuristic(
        self,
        platform,
        question: str,
        *,
        confirmed: dict,
        graph_id: str | None,
    ) -> ClarifyResult | ReadyResult:
        from fashion_brain.agents.qu_fast import (
            AUTO_RUN_NAMED_PARTY_ANALYSIS,
            SKIP_ANALYSIS_FINAL_CONFIRM,
            resolve_entity,
        )

        # Precise / non-analysis asks: still ground live brands + facets from the question.
        # (Previously returned empty filters → kg_stats answered the global catalog total.)
        if _is_precise_ask(question, confirmed) or not _is_vague_analysis(question):
            metrics: list[str] = []
            try:
                from fashion_brain.ask import enrich_scope_from_question, is_product_count_ask

                # Fallback classifier only when LLM did not run / failed.
                if is_product_count_ask(question):
                    metrics = ["count"]
                scoped, group_by = enrich_scope_from_question(
                    question,
                    dict(confirmed),
                    platform=platform,
                )
            except Exception:
                scoped, group_by = dict(confirmed), None
            # Gibberish / chitchat with no catalog signal → clarify, not a global total.
            # group_by alone is enough signal (e.g. "gender wise breakdown e-commerce data").
            if (
                not scoped
                and not metrics
                and not group_by
                and not _PRECISE_METRIC_RE.search(question or "")
            ):
                try:
                    from fashion_brain.planning.metadata_resolver import check_understanding

                    und = check_understanding(
                        platform,
                        question,
                        None,
                        {
                            "filters": {},
                            "available_attributes": [],
                            "source": "keywords",
                        },
                        None,
                    )
                    if not und.proceed:
                        clar = ClarificationState(
                            pending_slot="entity",
                            confirmed={},
                            offered_options=[
                                str(o.get("restated_question") or "")
                                for o in (und.options or [])[:3]
                                if o
                            ],
                            turn_count=1,
                            original_question=question,
                        )
                        return ClarifyResult(
                            question=str(und.clarify_text or "").strip()
                            or (
                                "What would you like to know from the catalog? "
                                "Name a brand, product type, or metric."
                            ),
                            state=clar,
                            slot="entity",
                        )
                except Exception:
                    pass
                if len((question or "").split()) <= 3 and not confirmed:
                    clar = ClarificationState(
                        pending_slot="entity",
                        confirmed={},
                        offered_options=[],
                        turn_count=1,
                        original_question=question,
                    )
                    return ClarifyResult(
                        question=(
                            "What would you like to know from the catalog? "
                            "Name a brand, product type, or metric (for example how many products)."
                        ),
                        state=clar,
                        slot="entity",
                    )
            ready = self._ready_intent(
                question,
                scoped,
                analysis_depth="unset",
                objective="execute",
                available_dimensions=[],
                metrics=metrics,
                group_by=group_by,
            )
            ready.source = _detect_graph_id(question)
            return ReadyResult(intent=ready)

        filters = dict(confirmed)
        brand, suggestions = resolve_entity(platform, question, graph_id)
        entity = brand or filters.get("brand") or filters.get("buyer")
        # Live-brand replace: "analysis primark" must not keep Primark|Zara from a prior compare.
        try:
            from fashion_brain.ask import enrich_scope_from_question, live_brand_names

            names = live_brand_names(platform)
            filters, _gb = enrich_scope_from_question(
                question, filters, brand_names=names, platform=platform
            )
            if brand:
                entity = brand
                filters["brand"] = brand
            elif filters.get("brand") and "|" not in str(filters.get("brand")):
                entity = str(filters.get("brand"))
        except Exception:
            if brand:
                filters["brand"] = brand
                entity = brand

        dimensions = self._available_dimensions(platform, graph_id=graph_id, question=question)
        if not entity:
            hint = ""
            if suggestions:
                hint = f" Did you mean {', '.join(suggestions)}?"
            clar = ClarificationState(
                pending_slot="entity",
                confirmed={},
                offered_options=list(suggestions[:3]),
                turn_count=1,
                original_question=question,
            )
            return ClarifyResult(
                question=(
                    f"Which brand or buyer should I analyze? "
                    f"Name one that exists in the catalog.{hint}"
                ),
                state=clar,
                slot="entity",
            )

        if entity and "brand" not in filters and "buyer" not in filters:
            filters["brand"] = entity

        # Named party analysis → run full report now (no "reply yes" confirm loop).
        if (
            (AUTO_RUN_NAMED_PARTY_ANALYSIS or SKIP_ANALYSIS_FINAL_CONFIRM)
            and entity
            and _is_named_party_analysis(question)
            and not _is_price_trend_ask(question)
        ):
            if "brand" not in filters and "buyer" not in filters:
                filters["brand"] = entity
            return ReadyResult(
                intent=self._ready_intent(
                    question,
                    filters,
                    analysis_depth="full",
                    objective="full_analysis",
                    available_dimensions=dimensions,
                    entity=entity,
                ),
                state=ClarificationState(
                    pending_slot=None,
                    confirmed={
                        "entity": entity,
                        "filters": filters,
                        "analysis_depth": "full",
                        "understanding_summary": f"Analysing {entity}, all products",
                        "auto_ran": True,
                    },
                    original_question=question,
                ),
            )

        # Only when final-confirm is enabled and entity is still fuzzy.
        if SKIP_ANALYSIS_FINAL_CONFIRM and entity:
            if "brand" not in filters and "buyer" not in filters:
                filters["brand"] = entity
            return ReadyResult(
                intent=self._ready_intent(
                    question,
                    filters,
                    analysis_depth="full",
                    objective="full_analysis",
                    available_dimensions=dimensions,
                    entity=entity,
                ),
                state=ClarificationState(
                    pending_slot=None,
                    confirmed={
                        "entity": entity,
                        "filters": filters,
                        "analysis_depth": "full",
                        "understanding_summary": f"Analysing {entity}, all products",
                        "auto_ran": True,
                    },
                    original_question=question,
                ),
            )

        confirmed_state = {
            "entity": entity,
            "filters": filters,
            "analysis_depth": "full",
        }
        _append_filter_trail(confirmed_state, turn=1, updates=dict(filters))
        clar = ClarificationState(
            pending_slot="final_confirm",
            confirmed=confirmed_state,
            offered_options=list(dimensions[:8]),
            turn_count=1,
            original_question=question,
        )
        return ClarifyResult(
            question=self._final_confirm_question(
                platform,
                entity=entity,
                filters=filters,
                confirmed=confirmed_state,
                original=question,
                dimensions=dimensions,
                graph_id=graph_id,
            ),
            state=clar,
            slot="final_confirm",
        )

    def _from_model(
        self,
        platform,
        question: str,
        history: list | None,
        *,
        state: dict,
        context: ContextDecision | None,
        graph_id: str | None,
        clar: ClarificationState | None,
        confirmed: dict,
    ) -> ClarifyResult | ReadyResult | None:
        if platform is None or getattr(getattr(platform, "settings", None), "answer_mode", None) != "llm":
            return None
        prior_q = (clar.original_question if clar else None) or question
        dimensions = self._available_dimensions(
            platform,
            graph_id=graph_id or _detect_graph_id(prior_q),
            question=question,
        )
        meta_snip = self._metadata_snippet(platform, question=question, graph_id=graph_id, dimensions=dimensions)
        pending = (clar.pending_slot if clar else None) or None
        prior_understanding = state.get("understanding") or {}
        prior_validated = state.get("validated_intent") or prior_understanding.get("validated_intent")
        prompt = (
            f"Available dimensions (metadata-grounded): {dimensions[:10]}\n"
            f"Selective source metadata:\n{meta_snip}\n\n"
            f"Session last_graph_id: {state.get('last_graph_id') or graph_id or 'none'}\n"
            f"Session active_filters: {state.get('active_filters') or confirmed or {}}\n"
            f"Prior validated_intent: {prior_validated or {}}\n"
            f"Prior understanding summary: {(prior_understanding.get('summary') if isinstance(prior_understanding, dict) else '') or '(none)'}\n"
            f"Pending clarification slot: {pending or 'none'}\n"
            f"Pending clarification confirmed: {(clar.confirmed if clar else None) or {}}\n"
            f"Original question in thread: {(clar.original_question if clar else '') or '(none)'}\n"
            f"Context decision: {getattr(context, 'decision', None) or 'n/a'}\n\n"
            f"Earlier chat:\n{_format_history(history) or '(none)'}\n\n"
            f"Latest user message:\n{question}\n\n"
            "Return the JSON decision now."
        )
        try:
            provider = build_model_provider(platform.settings)
            raw = provider.complete(
                prompt,
                timeout=_qu_timeout(getattr(platform.settings, "request_timeout_seconds", None)),
                system=QU_SYSTEM,
            )
        except Exception as exc:
            LOGGER.warning("query understanding LLM failed; using heuristic: %s", exc)
            return None
        parsed = self._parse_llm(raw)
        if not parsed:
            LOGGER.warning("query understanding LLM returned unusable JSON; using heuristic")
            return None
        return self._apply_llm_result(
            platform,
            question,
            parsed,
            clar=clar,
            dimensions=dimensions,
            confirmed=confirmed,
            graph_id=graph_id,
        )

    def _parse_llm(self, raw: str) -> dict[str, Any] | None:
        from fashion_brain.ask import _extract_json_object

        data = _extract_json_object(raw or "")
        if not isinstance(data, dict):
            return None
        status = str(data.get("status") or "").casefold().strip()
        if status not in {"ready", "clarify"}:
            return None
        return data

    def _apply_llm_result(
        self,
        platform,
        question: str,
        data: dict[str, Any],
        *,
        clar: ClarificationState | None,
        dimensions: list[str],
        confirmed: dict,
        graph_id: str | None,
    ) -> ClarifyResult | ReadyResult:
        status = str(data.get("status") or "").casefold()
        filters = dict(confirmed or {})
        raw_filters = data.get("filters") if isinstance(data.get("filters"), dict) else {}
        for key, val in (raw_filters or {}).items():
            if val in (None, ""):
                continue
            k = str(key)
            # Never accept graph-scope tokens as brand/buyer.
            if k in {"brand", "buyer"} and all(
                _is_graph_scope_token(tok) for tok in _tokens(str(val))
            ):
                continue
            filters[k] = val if isinstance(val, str) else str(val)

        source = str(data.get("source") or "").strip() or None
        if source and source not in {"ecom_products", "plmex_products", "own_products"}:
            # Allow only known product graphs; fall back to detector.
            source = _detect_graph_id(question) or _detect_graph_id(
                str(data.get("effective_question") or "")
            )
        if not source:
            source = _detect_graph_id(question) or graph_id

        from fashion_brain.agents.qu_fast import (
            AUTO_RUN_NAMED_PARTY_ANALYSIS,
            SKIP_ANALYSIS_FINAL_CONFIRM,
            resolve_entity,
        )

        entity = data.get("entity")
        if entity is not None:
            entity = str(entity).strip() or None
        if entity and _is_graph_scope_token(entity):
            entity = None
        # Validate LLM entity against live brands — never accept free text.
        live_brand, live_suggestions = resolve_entity(
            platform, entity or question, graph_id or source
        )
        if entity and not live_brand:
            # Drop invented brands; keep suggestions for clarify.
            entity = None
            filters.pop("brand", None)
        elif live_brand:
            entity = live_brand
            filters["brand"] = live_brand
        elif entity and "brand" not in filters and "buyer" not in filters:
            filters.setdefault("brand", entity)

        metrics = [str(m) for m in (data.get("metrics") or []) if m]
        depth = str(data.get("analysis_depth") or "unset").casefold()
        if depth not in {"unset", "focused", "full"}:
            depth = "unset"
        group_by = data.get("group_by")
        group_by = str(group_by).strip() if group_by else None
        if group_by and dimensions and group_by not in dimensions:
            # Keep only if it looks like a known attr; else drop.
            if group_by.casefold() not in {d.casefold() for d in dimensions}:
                group_by = _detect_focused_dimension(group_by, dimensions)

        comparison = [str(x) for x in (data.get("comparison") or []) if x]
        objective = str(data.get("objective") or "execute").strip() or "execute"
        effective = str(data.get("effective_question") or question).strip() or question
        summary = str(data.get("summary") or "").strip()

        if status == "clarify":
            slot = str(data.get("clarification_slot") or "").strip()
            if slot not in _ALLOWED_SLOTS:
                slot = (clar.pending_slot if clar else None) or "scope_or_direction"
            # Prefer LLM clarification text (user language); English templates are fallback.
            ask = str(data.get("clarification_question") or "").strip()
            # Clear "analysis <brand>" — price-trend → PriceChangeEvent; exact brand → auto-run.
            named = live_brand or entity
            if _is_price_trend_ask(question) and named:
                if "brand" not in filters and "buyer" not in filters:
                    filters["brand"] = named
                price_ask = f"Summarize average price change trends for {named}"
                return ReadyResult(
                    intent=self._ready_intent(
                        price_ask,
                        filters,
                        analysis_depth="focused",
                        objective="execute",
                        available_dimensions=dimensions,
                        entity=named,
                        metrics=["price"],
                    ),
                    state=ClarificationState(
                        pending_slot=None,
                        confirmed={
                            "entity": named,
                            "filters": filters,
                            "analysis_depth": "focused",
                            "understanding_summary": price_ask,
                        },
                        original_question=question,
                    ),
                )
            if (
                not clar
                and named
                and _is_named_party_analysis(question)
                and (AUTO_RUN_NAMED_PARTY_ANALYSIS or SKIP_ANALYSIS_FINAL_CONFIRM)
                and (live_brand or SKIP_ANALYSIS_FINAL_CONFIRM)
            ):
                if "brand" not in filters and "buyer" not in filters:
                    filters["brand"] = named
                return ReadyResult(
                    intent=self._ready_intent(
                        question,
                        filters,
                        analysis_depth="full",
                        objective="full_analysis",
                        available_dimensions=dimensions,
                        entity=named,
                    ),
                    state=ClarificationState(
                        pending_slot=None,
                        confirmed={
                            "entity": named,
                            "filters": filters,
                            "analysis_depth": "full",
                            "understanding_summary": summary
                            or f"Analysing {named}, all products",
                            "auto_ran": True,
                        },
                        original_question=question,
                    ),
                )
            # Skip the "reply yes" confirm loop when disabled.
            if (
                SKIP_ANALYSIS_FINAL_CONFIRM
                and not clar
                and named
                and slot in {"scope_or_direction", "analysis_depth", "final_confirm", ""}
                and _is_named_party_analysis(question)
            ):
                if "brand" not in filters and "buyer" not in filters:
                    filters["brand"] = named
                return ReadyResult(
                    intent=self._ready_intent(
                        question,
                        filters,
                        analysis_depth="full",
                        objective="full_analysis",
                        available_dimensions=dimensions,
                        entity=named,
                    ),
                    state=ClarificationState(
                        pending_slot=None,
                        confirmed={
                            "entity": named,
                            "filters": filters,
                            "analysis_depth": "full",
                            "understanding_summary": summary
                            or f"Analysing {named}, all products",
                            "auto_ran": True,
                        },
                        original_question=question,
                    ),
                )
            if (
                not clar
                and not SKIP_ANALYSIS_FINAL_CONFIRM
                and slot in {"scope_or_direction", "analysis_depth", "final_confirm", ""}
                and named
                and _is_named_party_analysis(question)
            ):
                if "brand" not in filters and "buyer" not in filters:
                    filters["brand"] = named
                conf = {
                    "entity": named,
                    "filters": filters,
                    "analysis_depth": "full",
                    "understanding_summary": summary,
                }
                _append_filter_trail(conf, turn=1, updates=dict(filters))
                base = ClarificationState(
                    pending_slot="final_confirm",
                    confirmed=conf,
                    offered_options=list(dimensions[:8]),
                    turn_count=1,
                    original_question=question,
                )
                return ClarifyResult(
                    question=self._final_confirm_question(
                        platform,
                        entity=named,
                        filters=filters,
                        confirmed=conf,
                        original=question,
                        dimensions=dimensions,
                        graph_id=graph_id or source,
                    ),
                    state=base,
                    slot="final_confirm",
                )
            if not named and live_suggestions and slot in {"entity", "scope_or_direction", ""}:
                slot = "entity"
                ask = ask or (
                    "I couldn't find that brand. Did you mean "
                    + ", ".join(live_suggestions)
                    + "?"
                )
            dim_set = set(dimensions or [])
            raw_opts = data.get("offered_options") or list(dimensions[:8])
            offered = [
                str(o)
                for o in raw_opts
                if o and (not dim_set or str(o) in dim_set or str(o) in live_suggestions)
            ]
            if not offered:
                offered = list(live_suggestions[:3] or dimensions[:8])
            if not ask:
                ask = (
                    f"What would you like to focus on: "
                    f"{self._format_dimension_options(offered)}?"
                )
            base = clar or ClarificationState(
                pending_slot=slot,
                confirmed={},
                offered_options=offered,
                turn_count=0,
                original_question=question,
            )
            base.pending_slot = slot
            base.offered_options = offered
            base.turn_count = int(base.turn_count or 0) + 1
            base.original_question = base.original_question or question
            conf = dict(base.confirmed or {})
            if entity:
                conf["entity"] = entity
            conf["filters"] = filters
            conf["analysis_depth"] = depth
            if group_by:
                conf["focused_dimension"] = group_by
            if summary:
                conf["understanding_summary"] = summary
            _append_filter_trail(conf, turn=base.turn_count, updates=dict(filters))
            base.confirmed = conf
            # Prefer LLM clarification_question; compose adds at most one scope line.
            composed = _compose_progressive_clarify(
                platform=platform,
                next_question=ask,
                entity=entity or conf.get("entity"),
                filters=filters,
                confirmed=conf,
                original=base.original_question or question,
                group_by=group_by,
                analysis_depth=depth,
                graph_id=graph_id or source,
                include_basics=True,
            )
            return ClarifyResult(question=composed, state=base, slot=slot)

        # ready
        if not metrics:
            try:
                from fashion_brain.ask import is_product_count_ask

                if is_product_count_ask(effective) or is_product_count_ask(question):
                    metrics = ["count"]
            except Exception:
                pass
        intent = self._ready_intent(
            effective,
            filters,
            analysis_depth=depth,
            objective=objective,
            available_dimensions=dimensions,
            entity=entity,
            group_by=group_by,
            metrics=metrics,
        )
        intent.source = source
        intent.comparison = comparison
        intent.effective_question = effective
        if summary:
            # Keep summary available via ambiguities channel is wrong; attach in objective note.
            intent.objective = objective
        # Clear pending clarification on ready.
        ready_state = None
        if clar is not None:
            clar.pending_slot = None
            clar.confirmed = {
                **(clar.confirmed or {}),
                "filters": filters,
                "analysis_depth": depth,
                "understanding_summary": summary,
            }
            ready_state = clar
        return ReadyResult(intent=intent, state=ready_state)

    def _metadata_snippet(
        self,
        platform,
        *,
        question: str,
        graph_id: str | None,
        dimensions: list[str],
    ) -> str:
        gid = graph_id or _detect_graph_id(question)
        lines = [f"Likely source: {gid or 'unspecified'}"]
        try:
            from knowledge_graph.retrieval.kg_stats import meta_overview, stats_available

            generated = getattr(getattr(platform, "settings", None), "generated", None)
            for candidate in [gid, "ecom_products", "plmex_products"]:
                if not candidate or not stats_available(candidate, generated):
                    continue
                overview = meta_overview(candidate, generated) or {}
                totals = overview.get("totals") or {}
                cols = overview.get("columns") or []
                col_keys = [str(c.get("key")) for c in cols[:8] if c.get("key")]
                lines.append(
                    f"- {candidate}: products={totals.get('products')}, columns={col_keys}"
                )
                if len(lines) >= 4:
                    break
        except Exception:
            pass
        if dimensions:
            lines.append(f"Dimensions for grounding: {dimensions[:8]}")
        return "\n".join(lines) if lines else "(no kg_stats metadata)"

    def _continue(
        self,
        platform,
        answer: str,
        clar: ClarificationState,
        *,
        graph_id: str | None = None,
    ) -> ClarifyResult | ReadyResult:
        slot = clar.pending_slot or ""
        confirmed = dict(clar.confirmed or {})
        filters = dict(confirmed.get("filters") or {})
        entity = confirmed.get("entity") or filters.get("brand") or filters.get("buyer")
        dimensions = list(clar.offered_options or []) or self._available_dimensions(
            platform, graph_id=graph_id, question=clar.original_question or answer
        )
        clar.turn_count = int(clar.turn_count or 0) + 1
        original = clar.original_question or answer

        from fashion_brain.agents.qu_fast import (
            SKIP_ANALYSIS_FINAL_CONFIRM,
            budget_exhausted,
            resolve_entity,
        )

        def _ready_full(now: dict, *, group_by: str | None = None) -> ReadyResult:
            now = dict(now)
            now["analysis_depth"] = "full"
            now["filters"] = dict(now.get("filters") or filters)
            clar.confirmed = now
            clar.pending_slot = None
            return ReadyResult(
                intent=self._ready_intent(
                    original,
                    dict(now.get("filters") or {}),
                    analysis_depth="full",
                    objective="full_analysis",
                    available_dimensions=dimensions,
                    entity=now.get("entity") or entity,
                    group_by=group_by or now.get("focused_dimension"),
                ),
                state=clar,
            )

        if slot == "final_confirm":
            if _is_confirm_no(answer) and not _is_full_analysis(answer):
                clar.pending_slot = "scope_or_direction"
                return ClarifyResult(
                    question=(
                        "Okay. What should I change: brand, product type, or another attribute?"
                    ),
                    state=clar,
                    slot="scope_or_direction",
                )
            if _is_confirm_yes(answer) or _is_full_analysis(answer):
                # Price-trend originals must run PriceChangeEvent, not full assortment.
                if _is_price_trend_ask(str(original or "")):
                    party = entity or filters.get("brand") or filters.get("buyer") or ""
                    ask = (
                        f"Summarize average price change trends for {party}"
                        if party
                        else "Summarize average price change trends"
                    )
                    confirmed["analysis_depth"] = "focused"
                    confirmed["filters"] = filters
                    confirmed["understanding_summary"] = ask
                    clar.confirmed = confirmed
                    clar.pending_slot = None
                    return ReadyResult(
                        intent=self._ready_intent(
                            ask,
                            filters,
                            analysis_depth="focused",
                            objective="execute",
                            available_dimensions=dimensions,
                            entity=entity,
                            metrics=["price"],
                        ),
                        state=clar,
                    )
                confirmed["analysis_depth"] = "full"
                confirmed["filters"] = filters
                clar.confirmed = confirmed
                clar.pending_slot = None
                return ReadyResult(
                    intent=self._ready_intent(
                        original,
                        filters,
                        analysis_depth="full",
                        objective="full_analysis",
                        available_dimensions=dimensions,
                        entity=entity,
                        group_by=confirmed.get("focused_dimension"),
                    ),
                    state=clar,
                )
            # User added another dimension instead of confirming.
            focused = _detect_focused_dimension(answer, dimensions)
            scope_key, scope_val = _detect_product_scope(
                answer,
                self._category_values(platform, graph_id=graph_id),
                platform=platform,
                graph_id=graph_id,
            )
            if scope_key and scope_val:
                filters[scope_key] = scope_val
                _append_filter_trail(confirmed, turn=clar.turn_count, updates={scope_key: scope_val})
                confirmed["filters"] = filters
                clar.confirmed = confirmed
                if SKIP_ANALYSIS_FINAL_CONFIRM:
                    return _ready_full(confirmed)
                clar.pending_slot = "final_confirm"
                return ClarifyResult(
                    question=self._final_confirm_question(
                        platform,
                        entity=entity,
                        filters=filters,
                        confirmed=confirmed,
                        original=original,
                        dimensions=dimensions,
                        graph_id=graph_id,
                    ),
                    state=clar,
                    slot="final_confirm",
                )
            if focused:
                confirmed["focused_dimension"] = focused
                _append_filter_trail(
                    confirmed, turn=clar.turn_count, updates={"group_by": focused}
                )
                clar.confirmed = confirmed
                if SKIP_ANALYSIS_FINAL_CONFIRM:
                    return _ready_full(confirmed, group_by=focused)
                clar.pending_slot = "final_confirm"
                return ClarifyResult(
                    question=self._final_confirm_question(
                        platform,
                        entity=entity,
                        filters=filters,
                        confirmed=confirmed,
                        original=original,
                        dimensions=dimensions,
                        graph_id=graph_id,
                        group_by=focused,
                    ),
                    state=clar,
                    slot="final_confirm",
                )
            # Cap endless re-asks — proceed with confirmed scope.
            if budget_exhausted(clar):
                confirmed["analysis_depth"] = "full"
                confirmed["filters"] = filters
                confirmed["understanding_summary"] = (
                    f"Proceeded with confirmed scope after {clar.turn_count} clarification turns"
                )
                clar.confirmed = confirmed
                clar.pending_slot = None
                return ReadyResult(
                    intent=self._ready_intent(
                        original,
                        filters,
                        analysis_depth="full",
                        objective="full_analysis",
                        available_dimensions=dimensions,
                        entity=entity,
                        group_by=confirmed.get("focused_dimension"),
                    ),
                    state=clar,
                )
            clar.pending_slot = "final_confirm"
            return ClarifyResult(
                question=self._final_confirm_question(
                    platform,
                    entity=entity,
                    filters=filters,
                    confirmed=confirmed,
                    original=original,
                    dimensions=dimensions,
                    graph_id=graph_id,
                ),
                state=clar,
                slot="final_confirm",
            )

        if slot == "entity":
            brand, suggestions = resolve_entity(platform, answer, graph_id)
            if not brand:
                hint = f" Did you mean {', '.join(suggestions)}?" if suggestions else ""
                clar.pending_slot = "entity"
                clar.offered_options = list(suggestions[:3])
                return ClarifyResult(
                    question=f"I couldn't find that brand.{hint}",
                    state=clar,
                    slot="entity",
                )
            entity = brand
            filters["brand"] = entity
            confirmed.update({"entity": entity, "filters": filters, "analysis_depth": "full"})
            _append_filter_trail(confirmed, turn=clar.turn_count, updates={"brand": entity})
            clar.confirmed = confirmed
            clar.offered_options = list(dimensions[:8])
            # Initially off: run full analysis as soon as the brand is known (no "reply yes").
            if SKIP_ANALYSIS_FINAL_CONFIRM:
                return _ready_full(confirmed)
            clar.pending_slot = "final_confirm"
            return ClarifyResult(
                question=self._final_confirm_question(
                    platform,
                    entity=entity,
                    filters=filters,
                    confirmed=confirmed,
                    original=original,
                    dimensions=dimensions,
                    graph_id=graph_id,
                ),
                state=clar,
                slot="final_confirm",
            )

        if slot == "scope_or_direction":
            # "make analysis" with whatever filters we have → full report (skip yes-confirm).
            if _is_full_analysis(answer):
                confirmed["filters"] = filters
                clar.confirmed = confirmed
                clar.offered_options = list(dimensions[:8])
                if SKIP_ANALYSIS_FINAL_CONFIRM:
                    return _ready_full(confirmed)
                clar.pending_slot = "final_confirm"
                return ClarifyResult(
                    question=self._final_confirm_question(
                        platform,
                        entity=entity,
                        filters=filters,
                        confirmed=confirmed,
                        original=original,
                        dimensions=dimensions,
                        graph_id=graph_id,
                    ),
                    state=clar,
                    slot="final_confirm",
                )

            focused = _detect_focused_dimension(answer, dimensions)
            scope_key, scope_val = _detect_product_scope(
                answer,
                self._category_values(platform, graph_id=graph_id),
                platform=platform,
                graph_id=graph_id,
            )
            if scope_key and scope_val:
                filters[scope_key] = scope_val
                if "garment_type" in dimensions and scope_key == "category":
                    filters.setdefault("garment_type", scope_val)
                confirmed["filters"] = filters
                confirmed["analysis_depth"] = "unset"
                _append_filter_trail(
                    confirmed, turn=clar.turn_count, updates={scope_key: scope_val}
                )
                clar.confirmed = confirmed
                clar.pending_slot = "analysis_depth"
                clar.offered_options = list(dimensions[:8])
                who = entity or "these products"
                scope_label = scope_val
                next_q = (
                    f"Would you like a complete analysis of {who}'s {scope_label} products, "
                    f"a specific aspect such as {self._format_dimension_options(dimensions)}, "
                    f'or say "make analysis" to run the final report with confirmed filters?'
                )
                return ClarifyResult(
                    question=_compose_progressive_clarify(
                        platform=platform,
                        next_question=next_q,
                        entity=entity,
                        filters=filters,
                        confirmed=confirmed,
                        original=original,
                        analysis_depth="unset",
                        graph_id=graph_id,
                    ),
                    state=clar,
                    slot="analysis_depth",
                )

            if focused:
                # Focused dimension alone (e.g. "category") → run that breakdown now.
                confirmed["analysis_depth"] = "focused"
                confirmed["focused_dimension"] = focused
                confirmed["filters"] = filters
                _append_filter_trail(
                    confirmed, turn=clar.turn_count, updates={"group_by": focused}
                )
                clar.confirmed = confirmed
                clar.pending_slot = None
                return ReadyResult(
                    intent=self._ready_intent(
                        original,
                        filters,
                        analysis_depth="focused",
                        objective=f"focused:{focused}",
                        available_dimensions=dimensions,
                        entity=entity,
                        group_by=focused,
                        metrics=["count"],
                    ),
                    state=clar,
                )

            clar.pending_slot = "scope_or_direction"
            options_txt = self._format_dimension_options(dimensions)
            next_q = (
                f"I can explore {options_txt} for {entity or 'this brand'}, "
                f'or say "make analysis" for a full report. Which would you like?'
            )
            return ClarifyResult(
                question=_compose_progressive_clarify(
                    platform=platform,
                    next_question=next_q,
                    entity=entity,
                    filters=filters,
                    confirmed=confirmed,
                    original=original,
                    analysis_depth="unset",
                    graph_id=graph_id,
                    include_basics=False,
                ),
                state=clar,
                slot="scope_or_direction",
            )

        if slot == "analysis_depth":
            if _is_full_analysis(answer) or normalize_name(answer) in {
                "full",
                "complete",
                "all",
                "everything",
                "overview",
            }:
                confirmed["filters"] = filters
                clar.confirmed = confirmed
                if SKIP_ANALYSIS_FINAL_CONFIRM:
                    return _ready_full(
                        confirmed, group_by=confirmed.get("focused_dimension")
                    )
                clar.pending_slot = "final_confirm"
                return ClarifyResult(
                    question=self._final_confirm_question(
                        platform,
                        entity=entity,
                        filters=filters,
                        confirmed=confirmed,
                        original=original,
                        dimensions=dimensions,
                        graph_id=graph_id,
                        group_by=confirmed.get("focused_dimension"),
                    ),
                    state=clar,
                    slot="final_confirm",
                )

            focused = _detect_focused_dimension(answer, dimensions)
            if focused:
                confirmed["analysis_depth"] = "focused"
                confirmed["focused_dimension"] = focused
                confirmed["filters"] = filters
                _append_filter_trail(
                    confirmed, turn=clar.turn_count, updates={"group_by": focused}
                )
                clar.confirmed = confirmed
                clar.pending_slot = None
                return ReadyResult(
                    intent=self._ready_intent(
                        original,
                        filters,
                        analysis_depth="focused",
                        objective=f"focused:{focused}",
                        available_dimensions=dimensions,
                        entity=entity,
                        group_by=focused,
                        metrics=["count"],
                    ),
                    state=clar,
                )

            clar.pending_slot = "analysis_depth"
            next_q = (
                "Please choose a complete analysis, one aspect such as "
                f"{self._format_dimension_options(dimensions)}, "
                f'or say "make analysis" to confirm the final report.'
            )
            return ClarifyResult(
                question=_compose_progressive_clarify(
                    platform=platform,
                    next_question=next_q,
                    entity=entity,
                    filters=filters,
                    confirmed=confirmed,
                    original=original,
                    analysis_depth="unset",
                    graph_id=graph_id,
                    include_basics=False,
                ),
                state=clar,
                slot="analysis_depth",
            )

        if slot == "focused_dimension":
            focused = _detect_focused_dimension(answer, dimensions) or _match_option(
                answer, dimensions
            )
            if not focused:
                clar.pending_slot = "focused_dimension"
                return ClarifyResult(
                    question=_compose_progressive_clarify(
                        platform=platform,
                        next_question=(
                            "Which dimension should I focus on: "
                            f"{self._format_dimension_options(dimensions)}?"
                        ),
                        entity=entity,
                        filters=filters,
                        confirmed=confirmed,
                        original=original,
                        analysis_depth="unset",
                        graph_id=graph_id,
                        include_basics=False,
                    ),
                    state=clar,
                    slot="focused_dimension",
                )
            confirmed["analysis_depth"] = "focused"
            confirmed["focused_dimension"] = focused
            confirmed["filters"] = filters
            _append_filter_trail(
                confirmed, turn=clar.turn_count, updates={"group_by": focused}
            )
            clar.confirmed = confirmed
            clar.pending_slot = None
            return ReadyResult(
                intent=self._ready_intent(
                    original,
                    filters,
                    analysis_depth="focused",
                    objective=f"focused:{focused}",
                    available_dimensions=dimensions,
                    entity=entity,
                    group_by=focused,
                    metrics=["count"],
                ),
                state=clar,
            )

        clar.pending_slot = None
        return ReadyResult(
            intent=self._ready_intent(
                original or answer,
                filters,
                analysis_depth=str(confirmed.get("analysis_depth") or "unset"),
                objective="execute",
                available_dimensions=dimensions,
                entity=entity,
            ),
            state=clar,
        )

    def _final_confirm_question(
        self,
        platform,
        *,
        entity: str | None,
        filters: dict,
        confirmed: dict,
        original: str,
        dimensions: list[str],
        graph_id: str | None = None,
        group_by: str | None = None,
    ) -> str:
        """Restate all accumulated params and ask once before the full analysis report."""

        next_q = (
            "If this looks accurate, reply \"yes\" to run the final analysis report. "
            "To narrow first, name a filter or dimension "
            f"({self._format_dimension_options(dimensions)})."
        )
        return _compose_progressive_clarify(
            platform=platform,
            next_question=next_q,
            entity=entity,
            filters=filters,
            confirmed=confirmed,
            original=original,
            group_by=group_by or confirmed.get("focused_dimension"),
            analysis_depth="full",
            graph_id=graph_id,
            include_basics=True,
        )

    def _ready_intent(
        self,
        question: str,
        filters: dict,
        *,
        analysis_depth: str,
        objective: str,
        available_dimensions: list[str],
        entity: str | None = None,
        group_by: str | None = None,
        metrics: list[str] | None = None,
    ) -> ValidatedIntent:
        ent = entity or filters.get("brand") or filters.get("buyer")
        effective = question
        if analysis_depth == "full" and ent:
            scope_bits = [
                str(filters[k])
                for k in ("category", "garment_type", "color", "gender", "material", "season")
                if filters.get(k)
            ]
            scope = " ".join(scope_bits)
            effective = f"Full analysis of {ent}" + (f" {scope}" if scope else "") + " products"
        elif analysis_depth == "focused" and group_by and ent:
            effective = f"{ent} {group_by} distribution" + (
                f" for {filters.get('category') or filters.get('garment_type')}"
                if filters.get("category") or filters.get("garment_type")
                else ""
            )
        return ValidatedIntent(
            source=None,
            entity=str(ent) if ent else None,
            filters=dict(filters or {}),
            objective=objective,
            metrics=list(metrics or []),
            group_by=group_by,
            comparison=[],
            output="analysis_report" if analysis_depth == "full" else None,
            analysis_depth=analysis_depth,
            available_dimensions=list(available_dimensions or []),
            ambiguities=[],
            effective_question=effective,
        )

    def _available_dimensions(
        self, platform, *, graph_id: str | None = None, question: str = ""
    ) -> list[str]:
        from fashion_brain.agents.qu_fast import cached

        gid = graph_id or _detect_graph_id(question)

        def _compute() -> list[str]:
            attrs: list[str] = []
            try:
                from knowledge_graph.retrieval.kg_stats import facet_inventory, meta_overview

                generated = getattr(getattr(platform, "settings", None), "generated", None)
                if gid:
                    overview = meta_overview(gid, generated)
                    if overview and overview.get("columns"):
                        attrs = [
                            str(col.get("key"))
                            for col in overview["columns"]
                            if col.get("key") and int(col.get("unique_count") or 0) > 0
                        ]
                    if not attrs:
                        inv = facet_inventory(gid, generated)
                        attrs = [
                            k for k, n in inv.items() if k != "products" and int(n or 0) > 0
                        ]
            except Exception:
                attrs = []
            if not attrs:
                try:
                    from fashion_brain.planning.kg_catalog import build_field_catalog

                    catalog = build_field_catalog(platform, question or "", probe_values=False)
                    if gid and gid in catalog.graphs:
                        attrs = list(catalog.graphs[gid].attributes or [])
                        analysis = list(
                            getattr(catalog.graphs[gid], "analysis_fields", None) or []
                        )
                        if analysis:
                            attrs = list(dict.fromkeys([*analysis, *attrs]))
                    else:
                        attrs = catalog.all_attributes()
                        for gf in catalog.graphs.values():
                            for item in getattr(gf, "analysis_fields", None) or []:
                                if item not in attrs:
                                    attrs.append(item)
                except Exception:
                    attrs = []

            ranked = [a for a in _PREFERRED_DIMENSIONS if a in attrs]
            for a in attrs:
                if a in ranked:
                    continue
                if a in {"brand", "buyer", "name", "id", "product_id", "sku", "description"}:
                    continue
                if len(ranked) >= 10:
                    break
                ranked.append(a)
            return ranked or list(_PREFERRED_DIMENSIONS[:6])

        return cached(("dims", gid), _compute, ttl=300)

    def _category_values(self, platform, *, graph_id: str | None = None) -> list[str]:
        from fashion_brain.agents.qu_fast import cached

        gid = graph_id

        def _compute() -> list[str]:
            values: list[str] = []
            try:
                from knowledge_graph.retrieval.kg_stats import load_graph_stats

                generated = getattr(getattr(platform, "settings", None), "generated", None)
                if gid:
                    stats = load_graph_stats(gid, generated) or {}
                    for attr in ("category", "garment_type", "family"):
                        bucket = (stats.get("value_counts") or {}).get(attr) or {}
                        for name, count in sorted(
                            bucket.items(),
                            key=lambda item: (-int(item[1] or 0), str(item[0])),
                        ):
                            text = str(name).strip()
                            if (
                                text
                                and text.casefold()
                                not in {"(empty)", "(unspecified)", "n/a", "na"}
                                and int(count or 0) > 0
                                and text not in values
                            ):
                                values.append(text)
                            if len(values) >= 40:
                                return values
            except Exception:
                pass
            if values:
                return values
            try:
                from knowledge_graph.retrieval.metadata import discover_values

                for attr in ("category", "garment_type", "family"):
                    try:
                        payload = discover_values(
                            platform, graph_id=graph_id, attribute=attr, limit=40
                        )
                        for row in payload.get("values") or []:
                            if isinstance(row, dict):
                                text = str(
                                    row.get("value") or row.get("name") or ""
                                ).strip()
                            else:
                                text = str(row).strip()
                            if text and text not in values:
                                values.append(text)
                    except Exception:
                        continue
            except Exception:
                return values
            return values

        return cached(("cats", gid), _compute, ttl=300)

    def _format_dimension_options(self, dimensions: list[str]) -> str:
        labels = [_label(d) for d in (dimensions or [])[:6] if d]
        if not labels:
            return "available product attributes"
        if len(labels) == 1:
            return labels[0]
        if len(labels) == 2:
            return f"{labels[0]} or {labels[1]}"
        return ", ".join(labels[:-1]) + f", or {labels[-1]}"


def apply_validated_intent(
    intention: Intention,
    validated: ValidatedIntent,
    *,
    summary: IntentSummary | None = None,
) -> Intention:
    """Seed Intention (and optional IntentSummary) from a ready ValidatedIntent."""

    filters = dict(validated.filters or {})
    # Already-resolved product specification / details must not be rewritten into
    # assortment analysis (that stole brand filters and skipped the Neo4j mode sheet).
    spec_locked = (
        str(intention.intent or "") in {"specification", "product_details"}
        or bool(getattr(intention, "suggest", False))
    )
    if validated.analysis_depth == "full" and not spec_locked:
        # Full analysis scope is authoritative — never merge sticky garment/color
        # from ThinkingBrain / prior turns onto a confirmed brand-only ask.
        intention.filters = dict(filters)
        intention.group_by = validated.group_by
    elif filters:
        intention.filters = {**(intention.filters or {}), **filters}
    if validated.source:
        prefs = list(intention.preferred_graphs or [])
        if validated.source not in prefs:
            intention.preferred_graphs = [validated.source, *prefs]
    if validated.group_by and validated.analysis_depth != "full" and not spec_locked:
        intention.group_by = validated.group_by
    if validated.analysis_depth == "full" and not spec_locked:
        intention.analysis = True
        intention.response_type = intention.response_type or "analysis_report"
        intention.retrieve_mode = intention.retrieve_mode or "analytical_tool"
        if intention.intent in {"other", "list", "count", "product_count", ""}:
            intention.intent = "analysis"
    elif validated.analysis_depth == "focused" and not spec_locked:
        # Focused breakdown uses the same analytical engine (one facet).
        intention.analysis = True
        intention.response_type = intention.response_type or "analysis_report"
        intention.retrieve_mode = intention.retrieve_mode or "analytical_tool"
        if validated.group_by and intention.intent in {
            "other",
            "list",
            "count",
            "product_count",
            "analysis",
            "",
        }:
            intention.intent = "analysis"
        if validated.metrics:
            intention.metrics = list(validated.metrics)
    if (
        validated.analysis_depth == "unset"
        and intention.intent in {"product_count", "count", "breakdown", "other", ""}
        and len(filters) <= 1
        and not intention.analysis
    ):
        intention.retrieve_mode = "template"
        if intention.intent in {"other", ""}:
            intention.intent = "product_count"
    if validated.available_dimensions:
        intention.available_attributes = list(
            dict.fromkeys(
                [*(intention.available_attributes or []), *validated.available_dimensions]
            )
        )
    if validated.metrics and not intention.metrics:
        intention.metrics = list(validated.metrics)
    if validated.effective_question:
        intention.thinking = (
            (intention.thinking + "\n" if intention.thinking else "")
            + f"Validated intent: {validated.effective_question}"
        ).strip()

    if summary is not None:
        summary.clarification_required = False
        if validated.analysis_depth == "full":
            summary.filters = dict(filters)
            summary.group_by = validated.group_by
        else:
            if filters:
                summary.filters = {**(summary.filters or {}), **filters}
            if validated.group_by:
                summary.group_by = validated.group_by
        if validated.effective_question:
            summary.actual_intent = validated.objective or summary.actual_intent
            summary.text = validated.effective_question
    return intention


def clarification_awaiting(retrieved) -> bool:
    """True when retrieval was short-circuited for a clarification question."""

    caveats = getattr(retrieved, "caveats", None) or []
    return QueryUnderstandingAgent.AWAITING_CAVEAT in caveats
