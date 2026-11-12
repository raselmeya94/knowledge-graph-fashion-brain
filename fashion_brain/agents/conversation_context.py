"""Conversation context agent: past chat + new question → LLM decides follow-up.

No regex/word-lists for “is this a follow-up?”. The model reads history and the
latest turn and returns structured JSON. Code only applies that decision.
"""

from __future__ import annotations

import logging
from typing import Any

from fashion_brain.agents.contracts import ContextDecision
from fashion_brain.llm import build_model_provider

LOGGER = logging.getLogger("fashion_brain.agents.conversation_context")

CONTEXT_SYSTEM = """\
You read earlier chat turns and the latest user question for a knowledge-graph Q&A system.

Return ONE JSON object only:
{"is_contextual":true|false,"decision":"RETRIEVE"|"REFINE"|"REUSE","carry_filters":true|false,"refined_question":"...","group_by":null,"reason":"..."}

- is_contextual: true when the latest turn depends on earlier chat in THIS session — same topic continued,
  follow-up, or the user points at the past (e.g. earlier / before / previous / as you said / continue /
  same brand / from that / based on past / use memory / same catalog). false only for a truly new standalone topic.
- carry_filters: true when prior brand/buyer/category filters should stay; false when the latest ask clearly
  replaces them. You can set is_contextual=true with carry_filters=false (keep session/catalog memory,
  but run a fresh filter set from the latest words).
- decision:
  - REUSE = only rephrase the previous answer with no new facts (e.g. "say that again", "summarize that").
  - REFINE = same metric/topic, narrower slice of the prior result.
  - RETRIEVE = run a new graph query (default when unsure). Still set is_contextual=true when the user
    is continuing the same session topic so prior catalog choice can stick.
- Metric/topic change → RETRIEVE. Keep is_contextual=true when they still refer to the same parties/session
  (e.g. Primark availability then Primark product types). Set is_contextual=false only when the subject
  clearly changes to an unrelated ask.
  Examples of different metrics: product count vs availability changes vs price trends vs average current price.
- A complete new ask that does NOT refer to prior parties/session is standalone.
  Example: after comparing Primark vs Zara, "Which products use cotton as the primary material?"
  with no link back → RETRIEVE, is_contextual=false, carry_filters=false — do NOT keep Primark|Zara.
- Only carry brand/buyer filters when the latest question still refers to those parties
  (or clearly says "same brands" / "for those" / "of them" / "continue with Primark").
- refined_question: if contextual, one self-contained question that includes what must stay from prior turns; otherwise the latest question unchanged.
- group_by: dimension to break down by if the user asked for one; else null.
Do not invent facts. No markdown.
"""


class ConversationContextAgent:
    """LLM decides REUSE / REFINE / RETRIEVE and whether to carry prior filters."""

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

        # No history → standalone; no LLM needed.
        if not history and not prior_filters and not prior_result_id:
            return ContextDecision(
                decision="RETRIEVE",
                is_contextual=False,
                refined_question=question,
                additional_retrieval_required=True,
            )

        parsed = self._from_model(
            platform,
            question,
            history,
            prior_filters=prior_filters,
            prior_attrs=prior_attrs,
            prior_result_id=prior_result_id,
            last_graph_id=state.get("last_graph_id"),
        )
        if parsed is None:
            # Fail closed on filter carry, but keep session context when prior graph/result exists
            # so the same conversation id can stick to ecom vs plmex.
            sticky = bool(prior_result_id or state.get("last_graph_id") or history)
            return ContextDecision(
                decision="RETRIEVE",
                is_contextual=sticky,
                refined_question=question,
                additional_retrieval_required=True,
                active_filters={},
            )

        contextual = bool(parsed.get("is_contextual"))
        carry = bool(parsed.get("carry_filters", contextual))
        decision = str(parsed.get("decision") or "RETRIEVE").upper()
        if decision not in {"REUSE", "REFINE", "RETRIEVE"}:
            decision = "RETRIEVE"
        refined = str(parsed.get("refined_question") or question).strip() or question

        # REUSE only when we actually have prior text/result.
        if decision == "REUSE" and not (prior_result_id or state.get("last_retrieved_text")):
            decision = "RETRIEVE"

        # REFINE without a cached result → retrieve with carry.
        if decision == "REFINE" and not prior_result_id:
            decision = "RETRIEVE"

        # Hard guard: price / availability / temporal metric shifts never reuse prior text.
        decision, contextual, carry = _guard_metric_shift(
            question,
            history,
            decision=decision,
            contextual=contextual,
            carry=carry,
            last_family=str(state.get("last_metric_family") or "") or None,
        )
        decision, contextual, carry = _guard_standalone_filters(
            question,
            prior_filters=prior_filters,
            decision=decision,
            contextual=contextual,
            carry=carry,
        )

        return ContextDecision(
            decision=decision,
            is_contextual=contextual,
            refined_question=refined if contextual else question,
            reuse_previous_results=decision in {"REUSE", "REFINE"},
            previous_result_id=prior_result_id if contextual else None,
            additional_retrieval_required=decision == "RETRIEVE"
            or (decision == "REFINE" and not prior_result_id),
            active_filters=prior_filters if (contextual and carry) else {},
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
    ) -> dict[str, Any] | None:
        if platform is None or getattr(getattr(platform, "settings", None), "answer_mode", None) != "llm":
            return None
        provider = build_model_provider(platform.settings)
        history_block = _format_history(history)
        prompt = (
            f"Prior structured filters (may be empty): {prior_filters or {}}\n"
            f"Prior available attributes: {prior_attrs or []}\n"
            f"Prior result id: {prior_result_id or 'none'}\n"
            f"Prior product catalog graph (sticky for this session): {last_graph_id or 'none'}\n"
            "When brand/buyer names exist in more than one catalog, prefer that prior graph "
            "unless the user names a different catalog (ecom / plmex / plm).\n\n"
            f"Earlier chat:\n{history_block or '(none)'}\n\n"
            f"Latest question:\n{question}\n\n"
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
    # Any explicit temporal/price/availability ask must query fresh data — never rephrase
    # a prior answer about a different (or even same) metric from cached prose.
    if current != "catalog" and decision == "REUSE":
        decision = "RETRIEVE"
        if prior and prior != current:
            contextual = False
            carry = False
    if prior and current != "catalog" and prior != "catalog" and prior != current:
        decision = "RETRIEVE"
        contextual = False
        carry = False
    # Same family, different shape (brand ranking vs recent product list, trend vs movers)
    # is still a new retrieve without carrying the prior breakdown wording.
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
) -> tuple[str, bool, bool]:
    """Drop prior brand/buyer carry when the latest ask is self-contained without those parties."""

    if not prior_filters or not (contextual or carry):
        return decision, contextual, carry
    try:
        from fashion_brain.ask import (
            is_open_product_listing,
            resolve_catalog_plan,
            split_multi_party_filter,
        )
        from knowledge_graph.ontology import normalize_name
    except Exception:
        return decision, contextual, carry

    latest = resolve_catalog_plan(question, None, carry=False)
    latest_filters = dict(latest.get("filters") or {})
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
    # Self-contained filtered / open listing ask that does not name prior parties.
    standalone = bool(latest_filters) or is_open_product_listing(question)
    if standalone and prior_parties and not parties_mentioned:
        if decision == "REUSE":
            decision = "RETRIEVE"
        return decision, False, False
    return decision, contextual, carry


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
