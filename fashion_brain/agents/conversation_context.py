"""Agent 3: conversation context — REUSE / REFINE / RETRIEVE."""

from __future__ import annotations

import re

from fashion_brain.agents.contracts import ContextDecision

_REFINE = re.compile(
    r"\b(only|just|filter|narrow|men'?s|women'?s|show me more|these|those|same|from that|of them)\b",
    re.IGNORECASE,
)
_NEW_TOPIC = re.compile(
    r"\b(instead|unrelated|new topic|different brand|forget|start over)\b",
    re.IGNORECASE,
)


class ConversationContextAgent:
    """Decide whether prior results answer the turn without a new KG query."""

    def analyze(
        self,
        question: str,
        history: list | None = None,
        *,
        conversation_state: dict | None = None,
        available_attributes: list[str] | None = None,
    ) -> ContextDecision:
        state = conversation_state or {}
        prior_result_id = state.get("last_result_id")
        prior_filters = dict(state.get("active_filters") or {})
        prior_attrs = list(available_attributes or state.get("available_attributes") or [])
        history = history or []

        standalone = not history or bool(_NEW_TOPIC.search(question))
        if standalone and not prior_result_id:
            return ContextDecision(
                decision="RETRIEVE",
                is_contextual=False,
                refined_question=question,
                additional_retrieval_required=True,
            )

        contextual = bool(state.get("references_prior_chat")) or bool(_REFINE.search(question))
        if not contextual:
            # Heuristic: short follow-ups after prior turns often continue the topic.
            if history and len(question.split()) <= 14:
                contextual = True

        if not contextual:
            return ContextDecision(
                decision="RETRIEVE",
                is_contextual=False,
                refined_question=question,
                additional_retrieval_required=True,
                active_filters=prior_filters,
            )

        # Contextual but no cached result id → still retrieve, carrying filters via IntentionAgent.
        if not prior_result_id:
            return ContextDecision(
                decision="RETRIEVE",
                is_contextual=True,
                refined_question=question,
                additional_retrieval_required=True,
                active_filters=prior_filters,
            )

        # REFINE when prior result exists and question looks like a filter tweak.
        needs_attrs = _guess_needed_attributes(question)
        missing = [name for name in needs_attrs if name and name not in prior_attrs]
        if missing:
            return ContextDecision(
                decision="RETRIEVE",
                is_contextual=True,
                refined_question=question,
                reuse_previous_results=False,
                previous_result_id=prior_result_id,
                additional_retrieval_required=True,
                active_filters=prior_filters,
            )

        # Full REUSE only when the question is a pure rewrite / repeat of prior answer.
        rewrite = bool(
            re.search(
                r"\b(single sentence|one sentence|shorter|shorten|rephrase|rewrite|translate|bullet)\b",
                question,
                re.IGNORECASE,
            )
        )
        if rewrite:
            return ContextDecision(
                decision="REUSE",
                is_contextual=True,
                refined_question=question,
                reuse_previous_results=True,
                previous_result_id=prior_result_id,
                additional_retrieval_required=False,
                active_filters=prior_filters,
            )

        return ContextDecision(
            decision="REFINE",
            is_contextual=True,
            refined_question=f"Filter the previously retrieved results: {question}",
            reuse_previous_results=True,
            previous_result_id=prior_result_id,
            additional_retrieval_required=False,
            active_filters=prior_filters,
        )


def _guess_needed_attributes(question: str) -> list[str]:
    """Infer attribute *names* from the question — not specific brand/source values.

    Concrete values (Primark, Zara, …) are discovered via KG metadata APIs when needed.
    """

    text = question.casefold()
    needed: list[str] = []
    # Generic attribute cues only (no source-specific brand lists).
    mapping = {
        "gender": ("men", "women", "male", "female", "unisex", "gender"),
        "price": ("price", "£", "$", "euro", "cost", "cheap", "currency"),
        "material": ("material", "composition", "fabric", "cotton", "polyester", "wool"),
        "brand": ("brand", "label"),
        "buyer": ("buyer", "customer"),
        "color": ("color", "colour"),
        "season": ("season", "spring", "summer", "autumn", "winter", "fall"),
        "category": ("category", "garment", "type"),
    }
    for attr, tokens in mapping.items():
        if any(token in text for token in tokens):
            needed.append(attr)
    return needed
