"""Conversation context agent: resolve current ask vs history (no permanent filters).

Policy source: ``conversation_context_policy.md``.
Current question has priority; history only resolves references / continuity.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from fashion_brain.agents.contracts import ContextDecision
from fashion_brain.llm import build_model_provider

LOGGER = logging.getLogger("fashion_brain.agents.conversation_context")

# Initially off: do not LLM-study prior chat turns. Treat each ask as standalone.
# Flip to True when multi-turn carry / "of them" continuity is needed again.
STUDY_PAST_HISTORY = False

CONTEXT_SYSTEM = """\
# CONVERSATION CONTEXT RESOLUTION

You determine whether the user's CURRENT question should inherit context from previous turns.
Conversation history resolves references and continuity. It must NEVER be treated as permanent filters.

## CORE PRINCIPLE
The current user question has priority over historical context.
Never automatically carry forward filters from previous questions.
Only inherit previous context when the current question clearly depends on it.
Users may change topics without saying "new topic".

## CLASSIFY intent_type (exactly one)

### continuation
User continues the immediately relevant previous question/result.
Examples: "And how many of them are sweaters?", "Of those, how many are red?", "What about color breakdown for them?", "Show the top five."
Action: preserve relevant prior filters; apply new conditions; do not add unrelated historical filters.

### modification
User modifies an existing analytical scope.
Examples: "What about Zara instead?", "Change the color to blue.", "Overall, who has more — Primark or Zara?"
Action: keep relevant intent; replace/remove only specified conditions; drop conflicting prior facets when the ask is broader (e.g. overall totals).

### new_independent_question
User introduces a question that can be understood alone.
Examples: "How many products does Primark have?", "How many products does Zara have?",
"Who has more products — Primark or Zara?" (after any prior color/garment count — do NOT keep that color),
"How many products are in the PLM data?", "How many products are in the e-comm data?", "What is the average product price?"
Action: new scope; filters ONLY from the current question; carry_filters=false; is_contextual=false (unless they only name a catalog switch).
Whole-catalog / dataset totals ("in the PLM/e-comm data") NEVER inherit a prior brand — even if the last turn was Primark.
Same-topic product counts without deixis ("of them"/"those"/"same") are new_independent — never inherit leftover facets.

### partial_context_reuse
Different analytical task that explicitly reuses an entity from earlier chat.
Examples: "Give me a full analysis of Primark.", "For the same brand, show price distribution.", "Let's analyze Zara instead."
Action: keep only the requested entity/subject; DISCARD restrictive prior filters (garment, color, material, …). Full analysis of Primark = brand=primark only — not prior red sweaters.

## FILTER INHERITANCE
1. Never inherit all filters from the whole conversation.
2. Prefer the immediately preceding relevant scope over older turns.
3. Inherit only when wording/intent supports it ("of them", "those", "same", "instead", "what about" — interpret in context).
4. A newly named brand/category/color overrides conflicting prior values.
5. Broader analysis / overall count → remove previous restrictive facets unless explicitly kept.
6. Never use prior result filters merely because they exist in history.
7. Clarification/confirm does not permanently lock filters.
8. If inheritance would materially change the result and intent is ambiguous, set needs_clarification=true.

## MEMORY VS QUERY FILTERS
Conversation memory (discussed brands/results) ≠ active query filters.
Only put conditions for the CURRENT request into active_filters / inherited_filters.

## OUTPUT — one JSON object only
{
  "intent_type": "continuation"|"modification"|"new_independent_question"|"partial_context_reuse",
  "is_contextual": true|false,
  "decision": "RETRIEVE"|"REFINE"|"REUSE",
  "carry_filters": true|false,
  "refined_question": "...",
  "group_by": null,
  "active_filters": {},
  "inherited_filters": {},
  "removed_filters": {},
  "inheritance_reason": "...",
  "needs_clarification": false,
  "reason": "..."
}

- carry_filters true only for continuation/modification that should keep prior facets/parties.
- For new_independent_question: carry_filters=false, active_filters={}, inherited_filters={}.
- For partial_context_reuse (e.g. full analysis): carry_filters=false for facets; active_filters may include only the named party.
- decision REUSE only when prior findings already contain the answer; else RETRIEVE/REFINE.
- refined_question: self-contained ask when contextual; else the latest question unchanged.
Do not invent facts. No markdown.
"""


class ConversationContextAgent:
    """Classify current ask vs history; decide RETRIEVE/REFINE/REUSE and filter carry."""

    def analyze(
        self,
        question: str,
        history: list | None = None,
        *,
        conversation_state: dict | None = None,
        available_attributes: list[str] | None = None,
        platform=None,
    ) -> ContextDecision:
        state = conversation_state or {}
        prior_result_id = state.get("last_result_id")
        prior_filters = dict(state.get("active_filters") or {})
        prior_attrs = list(available_attributes or state.get("available_attributes") or [])
        history = history or []

        # Past-history study commented off for now — no inherit / reuse / LLM context pass.
        if not STUDY_PAST_HISTORY:
            return ContextDecision(
                decision="RETRIEVE",
                is_contextual=False,
                refined_question=question,
                additional_retrieval_required=True,
                active_filters={},
                intent_type="new_independent_question",
                removed_filters=dict(prior_filters) if prior_filters else {},
                inheritance_reason="Past-history study disabled (STUDY_PAST_HISTORY=False).",
            )

        if not history and not prior_filters and not prior_result_id:
            return ContextDecision(
                decision="RETRIEVE",
                is_contextual=False,
                refined_question=question,
                additional_retrieval_required=True,
                intent_type="new_independent_question",
                inheritance_reason="No prior conversation state.",
            )

        if platform is not None:
            try:
                from fashion_brain.ask import preferred_graphs_from_question

                named = list(preferred_graphs_from_question(question, platform) or [])
                sticky = str(state.get("last_graph_id") or "").strip()
                if named and (not sticky or sticky not in named):
                    return ContextDecision(
                        decision="RETRIEVE",
                        is_contextual=False,
                        refined_question=question,
                        additional_retrieval_required=True,
                        active_filters={},
                        intent_type="new_independent_question",
                        removed_filters=dict(prior_filters),
                        inheritance_reason="User named a different product catalog.",
                    )
            except Exception:
                pass

        parsed = self._from_model(
            platform,
            question,
            history,
            prior_filters=prior_filters,
            prior_attrs=prior_attrs,
            prior_result_id=prior_result_id,
            last_graph_id=state.get("last_graph_id"),
            prior_findings=str(state.get("last_retrieved_text") or "") or None,
        )
        if parsed is None:
            parsed = _heuristic_context_resolution(question, prior_filters=prior_filters)

        return self._decision_from_parsed(
            question,
            history,
            state=state,
            prior_filters=prior_filters,
            prior_result_id=prior_result_id,
            parsed=parsed,
        )

    def _decision_from_parsed(
        self,
        question: str,
        history: list | None,
        *,
        state: dict,
        prior_filters: dict,
        prior_result_id: str | None,
        parsed: dict[str, Any],
    ) -> ContextDecision:
        intent_type = str(parsed.get("intent_type") or "").strip().casefold()
        if intent_type not in {
            "continuation",
            "modification",
            "new_independent_question",
            "partial_context_reuse",
        }:
            intent_type = "continuation" if parsed.get("is_contextual") else "new_independent_question"

        contextual = bool(parsed.get("is_contextual"))
        carry = bool(parsed.get("carry_filters", contextual and intent_type in {"continuation", "modification"}))
        if intent_type == "new_independent_question":
            contextual = False
            carry = False
        if intent_type == "partial_context_reuse":
            # Keep session/catalog stickiness; do not carry restrictive facets.
            contextual = True
            carry = False

        decision = str(parsed.get("decision") or "RETRIEVE").upper()
        if decision not in {"REUSE", "REFINE", "RETRIEVE"}:
            decision = "RETRIEVE"
        refined = str(parsed.get("refined_question") or question).strip() or question

        if decision == "REUSE" and not (prior_result_id or state.get("last_retrieved_text")):
            decision = "RETRIEVE"
        if decision == "REFINE" and not (prior_result_id or state.get("last_retrieved_text")):
            decision = "RETRIEVE"

        decision, contextual, carry = _guard_metric_shift(
            question,
            history,
            decision=decision,
            contextual=contextual,
            carry=carry,
            last_family=str(state.get("last_metric_family") or "") or None,
        )
        decision, contextual, carry, intent_type = _guard_standalone_filters(
            question,
            prior_filters=prior_filters,
            decision=decision,
            contextual=contextual,
            carry=carry,
            intent_type=intent_type,
        )
        decision, contextual, carry, intent_type = _guard_broader_scope_reset(
            question,
            prior_filters=prior_filters,
            decision=decision,
            contextual=contextual,
            carry=carry,
            intent_type=intent_type,
        )
        decision, contextual, carry, intent_type = _guard_catalog_dataset_total(
            question,
            prior_filters=prior_filters,
            decision=decision,
            contextual=contextual,
            carry=carry,
            intent_type=intent_type,
        )

        active, inherited, removed = _resolve_filter_sets(
            intent_type=intent_type,
            carry=carry,
            prior_filters=prior_filters,
            parsed=parsed,
            question=question,
        )

        reason = str(
            parsed.get("inheritance_reason") or parsed.get("reason") or ""
        ).strip()
        if not reason:
            reason = {
                "continuation": "Current ask continues the prior scope.",
                "modification": "Current ask modifies the prior scope.",
                "new_independent_question": "Current ask stands alone; no filter inheritance.",
                "partial_context_reuse": "Reusing only the named entity; restrictive facets cleared.",
            }.get(intent_type, "")

        return ContextDecision(
            decision=decision,
            is_contextual=contextual,
            refined_question=refined if contextual else question,
            reuse_previous_results=decision in {"REUSE", "REFINE"},
            previous_result_id=prior_result_id if contextual else None,
            additional_retrieval_required=decision == "RETRIEVE"
            or (
                decision == "REFINE"
                and not (prior_result_id or state.get("last_retrieved_text"))
            ),
            active_filters=active,
            intent_type=intent_type,
            inherited_filters=inherited,
            removed_filters=removed,
            inheritance_reason=reason,
            needs_clarification=bool(parsed.get("needs_clarification")),
        )

    def _from_model(
        self,
        platform,
        question: str,
        history: list | None,
        *,
        prior_filters: dict,
        prior_attrs: list[str],
        prior_result_id: str | None,
        last_graph_id: str | None = None,
        prior_findings: str | None = None,
    ) -> dict[str, Any] | None:
        if platform is None or getattr(getattr(platform, "settings", None), "answer_mode", None) != "llm":
            return None
        provider = build_model_provider(platform.settings)
        history_block = _format_history(history)
        findings_excerpt = (prior_findings or "").strip()
        if len(findings_excerpt) > 2800:
            findings_excerpt = findings_excerpt[:2800] + "…"
        prompt = (
            f"Prior structured filters (conversation memory — not automatic query filters): "
            f"{prior_filters or {}}\n"
            f"Prior available attributes: {prior_attrs or []}\n"
            f"Prior result id: {prior_result_id or 'none'}\n"
            f"Prior product catalog graph (sticky for this session): {last_graph_id or 'none'}\n"
            "When brand/buyer names exist in more than one catalog, prefer that prior graph "
            "unless the user names a different catalog (ecom / plmex / plm).\n\n"
            f"Earlier chat:\n{history_block or '(none)'}\n\n"
            f"<prior_findings>\n{findings_excerpt or '(none)'}\n</prior_findings>\n\n"
            f"Latest question:\n{question}\n\n"
            "Classify intent_type first. Current question has priority. "
            "If the latest ask can be answered from <prior_findings> alone "
            "(summarize / sort / explain / pick a listed row), choose REUSE.\n"
            "Return the JSON decision now."
        )
        try:
            raw = provider.complete(
                prompt,
                timeout=_timeout(getattr(platform.settings, "request_timeout_seconds", None)),
                system=CONTEXT_SYSTEM,
            )
        except Exception as exc:
            LOGGER.warning("conversation context LLM failed: %s", exc)
            return None
        return _parse_decision(raw)


def _heuristic_context_resolution(question: str, *, prior_filters: dict) -> dict[str, Any]:
    """Deterministic classifier when the context LLM is unavailable."""

    try:
        from fashion_brain.ask import (
            _is_analysis,
            question_has_party_deixis,
            resolve_catalog_plan,
            sanitize_party_filters,
        )
        from knowledge_graph.ontology import normalize_name
    except Exception:
        return {
            "intent_type": "new_independent_question",
            "is_contextual": False,
            "decision": "RETRIEVE",
            "carry_filters": False,
            "refined_question": question,
            "inheritance_reason": "Heuristic fallback (imports failed).",
        }

    low = normalize_name(question or "")
    deixis = question_has_party_deixis(question)
    plan = resolve_catalog_plan(question, None, carry=False)
    plan_filters = sanitize_party_filters(dict(plan.get("filters") or {}))
    named_party = bool(plan_filters.get("brand") or plan_filters.get("buyer"))
    broader = bool(
        re.search(r"\b(overall|in total|altogether|full analysis|complete analysis)\b", low)
        or (_is_analysis(question) and named_party)
    )

    if deixis and prior_filters:
        return {
            "intent_type": "continuation",
            "is_contextual": True,
            "decision": "RETRIEVE",
            "carry_filters": True,
            "refined_question": question,
            "inheritance_reason": "Deixis (them/those/same) continues prior party scope.",
        }

    if broader:
        party = {}
        if plan_filters.get("brand"):
            party["brand"] = plan_filters["brand"]
        if plan_filters.get("buyer"):
            party["buyer"] = plan_filters["buyer"]
        if not party:
            for key in ("brand", "buyer"):
                if prior_filters.get(key):
                    party[key] = prior_filters[key]
        removed = {
            k: v
            for k, v in prior_filters.items()
            if k not in party and k in {"garment", "garment_type", "category", "color", "material", "gender", "season", "family", "subfamily"}
        }
        return {
            "intent_type": "partial_context_reuse" if party else "new_independent_question",
            "is_contextual": bool(party),
            "decision": "RETRIEVE",
            "carry_filters": False,
            "active_filters": party,
            "inherited_filters": dict(party),
            "removed_filters": removed,
            "refined_question": question,
            "inheritance_reason": "Broader analysis/overall ask — keep party only, drop restrictive facets.",
        }

    if named_party and not deixis:
        return {
            "intent_type": "new_independent_question",
            "is_contextual": False,
            "decision": "RETRIEVE",
            "carry_filters": False,
            "active_filters": {},
            "removed_filters": dict(prior_filters),
            "refined_question": question,
            "inheritance_reason": "Current ask names its own party; no filter inheritance.",
        }

    try:
        from fashion_brain.ask import is_catalog_dataset_total_ask

        if is_catalog_dataset_total_ask(question):
            return {
                "intent_type": "new_independent_question",
                "is_contextual": False,
                "decision": "RETRIEVE",
                "carry_filters": False,
                "active_filters": {},
                "removed_filters": dict(prior_filters),
                "refined_question": question,
                "inheritance_reason": "Whole-catalog/dataset total — no prior brand/buyer carry.",
            }
    except Exception:
        pass

    # Short dimension / slice follow-up without a new party.
    if prior_filters and (
        plan.get("group_by")
        or plan_filters.get("garment")
        or plan_filters.get("color")
        or plan_filters.get("material")
        or re.search(r"\b(breakdown|of them|for them|and how many)\b", low)
    ):
        return {
            "intent_type": "continuation",
            "is_contextual": True,
            "decision": "RETRIEVE",
            "carry_filters": True,
            "refined_question": question,
            "inheritance_reason": "Follow-up facet/dimension on prior scope.",
        }

    sticky = bool(prior_filters)
    return {
        "intent_type": "continuation" if sticky else "new_independent_question",
        "is_contextual": sticky,
        "decision": "RETRIEVE",
        "carry_filters": False,
        "active_filters": {},
        "refined_question": question,
        "inheritance_reason": "Fail closed on filter carry; session may still stick to prior catalog.",
    }


def _resolve_filter_sets(
    *,
    intent_type: str,
    carry: bool,
    prior_filters: dict,
    parsed: dict[str, Any],
    question: str,
) -> tuple[dict, dict, dict]:
    """Build active / inherited / removed filter maps for this turn."""

    prior = {str(k): v for k, v in (prior_filters or {}).items() if v not in (None, "")}
    parsed_active = dict(parsed.get("active_filters") or {}) if isinstance(parsed.get("active_filters"), dict) else {}
    parsed_inherited = (
        dict(parsed.get("inherited_filters") or {})
        if isinstance(parsed.get("inherited_filters"), dict)
        else {}
    )
    parsed_removed = (
        dict(parsed.get("removed_filters") or {}) if isinstance(parsed.get("removed_filters"), dict) else {}
    )

    if intent_type == "new_independent_question":
        return {}, {}, dict(parsed_removed or prior)

    if intent_type == "partial_context_reuse":
        # Party-only from parsed or prior brand/buyer named in question.
        active = dict(parsed_active or parsed_inherited)
        if not active:
            try:
                from fashion_brain.ask import resolve_catalog_plan, sanitize_party_filters

                plan = sanitize_party_filters(
                    dict(resolve_catalog_plan(question, None, carry=False).get("filters") or {})
                )
                for key in ("brand", "buyer"):
                    if plan.get(key):
                        active[key] = plan[key]
                    elif prior.get(key):
                        active[key] = prior[key]
            except Exception:
                for key in ("brand", "buyer"):
                    if prior.get(key):
                        active[key] = prior[key]
        removed = {
            k: v
            for k, v in prior.items()
            if k not in active
            and k
            in {
                "garment",
                "garment_type",
                "category",
                "color",
                "material",
                "gender",
                "season",
                "family",
                "subfamily",
            }
        }
        removed.update(parsed_removed)
        return active, dict(active), removed

    if carry and prior:
        inherited = dict(parsed_inherited) if parsed_inherited else dict(prior)
        active = dict(parsed_active) if parsed_active else dict(inherited)
        removed = dict(parsed_removed)
        return active, inherited, removed

    return dict(parsed_active), dict(parsed_inherited), dict(parsed_removed or prior)


def _prior_user_question(history: list | None) -> str:
    for item in reversed(history or []):
        if isinstance(item, dict):
            role = str(item.get("role") or "").casefold()
            content = str(item.get("content") or item.get("text") or "").strip()
        else:
            role = str(getattr(item, "role", "") or "").casefold()
            content = str(getattr(item, "content", "") or getattr(item, "text", "") or "").strip()
        if role in {"user", "human"} and content:
            return content
    return ""


def _guard_metric_shift(
    question: str,
    history: list | None,
    *,
    decision: str,
    contextual: bool,
    carry: bool,
    last_family: str | None = None,
) -> tuple[str, bool, bool]:
    """Force RETRIEVE when the latest ask's metric family differs from the prior turn."""

    try:
        from fashion_brain.ask import (
            availability_history_mode,
            catalog_metric_family,
        )
        from knowledge_graph.retrieval.ecom import price_history_mode
    except Exception:
        return decision, contextual, carry

    current = catalog_metric_family(question)
    prior_q = _prior_user_question(history)
    prior = str(last_family or "").strip() or catalog_metric_family(prior_q)
    if current != "catalog" and decision == "REUSE":
        decision = "RETRIEVE"
        if prior and prior != current:
            contextual = False
            carry = False
    if prior and current != "catalog" and prior != "catalog" and prior != current:
        decision = "RETRIEVE"
        contextual = False
        carry = False
    if current == "availability" and prior == "availability":
        if availability_history_mode(question) != availability_history_mode(prior_q):
            decision = "RETRIEVE"
            contextual = False
            carry = False
    if current == "price" and prior == "price":
        if price_history_mode(question) != price_history_mode(prior_q):
            decision = "RETRIEVE"
            contextual = False
            carry = False
    return decision, contextual, carry


def _guard_standalone_filters(
    question: str,
    *,
    prior_filters: dict,
    decision: str,
    contextual: bool,
    carry: bool,
    intent_type: str,
) -> tuple[str, bool, bool, str]:
    """Drop prior brand/buyer carry when the latest ask is self-contained without those parties."""

    if not prior_filters or not (contextual or carry):
        return decision, contextual, carry, intent_type
    try:
        from fashion_brain.ask import (
            is_open_product_listing,
            question_has_party_deixis,
            resolve_catalog_plan,
            sanitize_party_filters,
            split_multi_party_filter,
        )
        from knowledge_graph.ontology import normalize_name
    except Exception:
        return decision, contextual, carry, intent_type

    latest = resolve_catalog_plan(question, None, carry=False)
    latest_filters = sanitize_party_filters(dict(latest.get("filters") or {}))
    deixis = question_has_party_deixis(question)
    if (latest_filters.get("brand") or latest_filters.get("buyer")) and not deixis:
        return "RETRIEVE", False, False, "new_independent_question"
    if deixis:
        return decision, True, True, "continuation" if intent_type == "new_independent_question" else intent_type
    qn = normalize_name(question)
    prior_parties: list[str] = []
    for key in ("brand", "buyer"):
        raw = str(prior_filters.get(key) or "").strip()
        if not raw:
            continue
        prior_parties.extend(split_multi_party_filter(raw) or [raw])
    parties_mentioned = any(
        normalize_name(p) and normalize_name(p) in qn for p in prior_parties if p
    )
    standalone = bool(latest_filters) or is_open_product_listing(question)
    if standalone and prior_parties and not parties_mentioned:
        if decision == "REUSE":
            decision = "RETRIEVE"
        return decision, False, False, "new_independent_question"
    return decision, contextual, carry, intent_type


def _guard_broader_scope_reset(
    question: str,
    *,
    prior_filters: dict,
    decision: str,
    contextual: bool,
    carry: bool,
    intent_type: str,
) -> tuple[str, bool, bool, str]:
    """Overall / full-analysis asks must not keep restrictive prior facets."""

    if not prior_filters:
        return decision, contextual, carry, intent_type
    try:
        from fashion_brain.ask import _is_analysis
        from knowledge_graph.ontology import normalize_name
    except Exception:
        return decision, contextual, carry, intent_type

    low = normalize_name(question or "")
    broader = bool(
        re.search(r"\b(overall|in total|altogether|full analysis|complete analysis)\b", low)
        or _is_analysis(question)
    )
    if not broader:
        return decision, contextual, carry, intent_type
    # Keep contextual for same-party analysis, but never carry garment/color/etc.
    return decision, True, False, "partial_context_reuse"


def _guard_catalog_dataset_total(
    question: str,
    *,
    prior_filters: dict,
    decision: str,
    contextual: bool,
    carry: bool,
    intent_type: str,
) -> tuple[str, bool, bool, str]:
    """Dataset/catalog totals never inherit a prior brand (e.g. after Primark → e-comm total)."""

    try:
        from fashion_brain.ask import is_catalog_dataset_total_ask
    except Exception:
        return decision, contextual, carry, intent_type
    if not is_catalog_dataset_total_ask(question):
        return decision, contextual, carry, intent_type
    if decision == "REUSE":
        decision = "RETRIEVE"
    return "RETRIEVE", False, False, "new_independent_question"


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
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _parse_decision(raw: str) -> dict[str, Any] | None:
    from fashion_brain.ask import _extract_json_object

    data = _extract_json_object(raw or "")
    return data if isinstance(data, dict) else None


def _timeout(seconds: float | None) -> float:
    if seconds is None or seconds <= 0:
        return 12.0
    return min(20.0, float(seconds))
