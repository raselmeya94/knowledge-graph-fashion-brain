"""Deterministic retrieve-mode planner (graph-agnostic).

Chooses text2cypher | analytical_tool | template from Intention + question.
Analysis is never assumed to be e-commerce — graphs come from Intention.preferred_graphs
/ GraphRouter, not from this module.
"""

from __future__ import annotations

import re

from fashion_brain.agents.contracts import Intention


def plan_retrieve(intention: Intention, question: str = "") -> Intention:
    """Fill retrieve_mode, response_type, metrics, operations when missing."""

    text = f"{question} {getattr(intention, 'thinking', '') or ''}".casefold()
    intent = str(intention.intent or "")
    # Product-spec sheet asks must never become assortment analysis (mode sheet path).
    if intent in {"specification", "product_details"} or bool(getattr(intention, "suggest", False)):
        intention.analysis = False
        intention.retrieve_mode = "template"
        intention.response_type = intention.response_type or (
            "specs" if intent in {"specification", "product_details"} else intention.response_type
        )
        intention.metrics = []
        intention.operations = ["retrieve"]
        return intention
    analysis = bool(getattr(intention, "analysis", False))
    group_by = intention.group_by
    filters = dict(intention.filters or {})
    count_entity = str(getattr(intention, "count_entity", None) or "").strip()
    facet_count = bool(count_entity) and count_entity.casefold() not in {"product", "products"}
    shape = str(getattr(intention, "query_shape", None) or "").strip() or None
    if not shape:
        try:
            from fashion_brain.planning.schema_grounding import detect_query_shape

            shape = detect_query_shape(question, schema_attrs=intention.available_attributes)
        except Exception:
            shape = None
        if shape:
            intention.query_shape = shape

    if shape == "shared_overlap" and intent not in {"shared_products", "shared"}:
        intention.intent = "shared_products"
        intent = "shared_products"
    if shape == "facet_listing" and not filters and count_entity:
        # Prefer listing products via the resolved facet entity.
        intention.metrics = list(dict.fromkeys([*(intention.metrics or []), "facet_listing"]))
    if shape == "entity_frequency":
        intention.metrics = list(dict.fromkeys([*(intention.metrics or []), "entity_frequency"]))
        if not intention.group_by and count_entity:
            intention.group_by = count_entity
            group_by = count_entity
    if shape == "category_rollup":
        intention.operations = list(dict.fromkeys([*(intention.operations or []), "category_rollup"]))
    if shape == "volatility":
        intention.operations = list(dict.fromkeys([*(intention.operations or []), "volatility"]))
    if shape == "state_transition":
        intention.operations = list(dict.fromkeys([*(intention.operations or []), "state_transition"]))

    if not intention.response_type:
        intention.response_type = _response_type(intent, analysis, text, group_by, facet_count)
    if not intention.metrics:
        intention.metrics = _metrics(intent, analysis, group_by, text, facet_count)
    if not intention.operations:
        intention.operations = _operations(intent, analysis, text, group_by)
    if not intention.retrieve_mode:
        intention.retrieve_mode = _retrieve_mode(intention, text)
    return intention


def _response_type(
    intent: str, analysis: bool, text: str, group_by: str | None, facet_count: bool = False
) -> str:
    if intent == "shared_products" or re.search(r"\b(across|between|compare|versus| vs )\b", text):
        return "comparison"
    if re.search(r"\b(trend|over time|history|historical|changed|change)\b", text):
        return "trend"
    if analysis:
        return "analysis_report"
    if facet_count or group_by or intent in {"breakdown", "product_count", "count"}:
        return "stats"
    if intent in {"product_details", "specification"}:
        return "specs"
    return "concise"


def _metrics(
    intent: str,
    analysis: bool,
    group_by: str | None,
    text: str,
    facet_count: bool = False,
) -> list[str]:
    metrics: list[str] = []
    # count_entity on Intention (any schema facet) → facet_count, not product_count.
    if facet_count:
        metrics.append("facet_count")
    elif analysis or group_by or intent in {"product_count", "breakdown", "count", "shared_products"}:
        metrics.append("product_count")
    if re.search(r"\b(average|avg|mean)\b.*\bprice\b|\bprice\b.*\b(average|avg|mean)\b", text):
        metrics.append("average_price")
    if re.search(r"\b(min|max|median)\b", text) and "price" in text:
        metrics.extend(["min_price", "max_price"])
    return list(dict.fromkeys(metrics))


def _operations(intent: str, analysis: bool, text: str, group_by: str | None) -> list[str]:
    ops: list[str] = []
    if analysis:
        ops.extend(["group", "aggregate", "profile"])
    elif group_by:
        ops.extend(["group", "aggregate"])
    elif intent in {"product_count", "count"}:
        ops.append("aggregate")
    if intent == "shared_products" or re.search(r"\b(compare|across|versus)\b", text):
        ops.append("compare")
    if re.search(r"\b(trend|history|over time)\b", text):
        ops.append("trend")
    return list(dict.fromkeys(ops)) or ["retrieve"]


def _retrieve_mode(intention: Intention, text: str) -> str:
    intent = str(intention.intent or "")
    shape = str(getattr(intention, "query_shape", None) or "")
    ops = {str(op) for op in (intention.operations or [])}
    if intent in {"system_meta", "live_sql", "news", "shared"}:
        return "template"
    if shape in {"category_rollup", "volatility", "state_transition"} or ops & {
        "category_rollup",
        "volatility",
        "state_transition",
    }:
        return "template"
    if shape == "entity_frequency" or "entity_frequency" in (intention.metrics or []):
        return "template"
    if re.search(r"\b(trend|history|historical|price change|availability change)\b", text):
        return "template"
    # Full (all facets) and focused (one group_by facet) share the analytical engine.
    if intention.analysis:
        return "analytical_tool"
    if intent == "shared_products":
        return "text2cypher"
    if intent in {"product_details", "specification"}:
        return "template"
    if intent in {"product_count", "count", "breakdown"} and not intention.analysis:
        return "template"
    if shape == "facet_listing":
        return "text2cypher"
    return "text2cypher"
