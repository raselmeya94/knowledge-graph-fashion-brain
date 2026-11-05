"""Catalog pipeline: context → intention → router → retrieve → optional spec."""

from __future__ import annotations

import uuid

from knowledge_graph.access import Principal
from knowledge_graph.logging_config import get_logger
from knowledge_graph.pipeline.service import Platform

from fashion_brain.agents.conversation_context import ConversationContextAgent
from fashion_brain.agents.product_spec import ProductSpecificationAgent
from fashion_brain.agents._impl import GraphRouterAgent, IntentionAgent, ThinkingBrain
from fashion_brain.agents.contracts import Retrieved
from fashion_brain.execution.merger import merge_retrieved
from fashion_brain.trace import trace_debug, trace_step

LOGGER = get_logger("fashion_brain.orchestration")


def draft_catalog_orchestrated(
    platform: Platform,
    question: str,
    principal: Principal,
    history: list | None = None,
    summary=None,
    graph_id: str | None = None,
    conversation_state: dict | None = None,
):
    """Run the multi-agent catalog path and return (retrieved, routes, intention, context, choice)."""

    from fashion_brain.ask import apply_buyer_spelling, _retrieve

    state = dict(conversation_state or {})
    trace_step("context", "ConversationContextAgent started", history_turns=len(history or []))
    LOGGER.info("orchestrator context agent started")
    context = ConversationContextAgent().analyze(
        question,
        history,
        conversation_state=state,
        available_attributes=state.get("available_attributes"),
    )
    trace_step(
        "context",
        "ConversationContextAgent finished",
        decision=context.decision,
        contextual=context.is_contextual,
        reuse=context.reuse_previous_results,
    )
    LOGGER.info("orchestrator context decision=%s", context.decision)

    if context.decision == "REUSE" and state.get("last_retrieved_text"):
        retrieved = Retrieved(
            text=str(state.get("last_retrieved_text") or ""),
            evidence=[],
            caveats=["Reused previous retrieval without a new knowledge-graph query."],
            unavailable=[],
            result_id=context.previous_result_id,
        )
        intention = IntentionAgent().understand(platform, question, history, summary)
        trace_step("retrieve", "Reused prior retrieval — skipped KG query")
        return retrieved, [intention.intent, "reuse"], intention, context, None

    LOGGER.info("orchestrator intention agent started question=%s", question)
    if summary is None:
        trace_step("thinking", "ThinkingBrain (orchestrator) started")
        summary = ThinkingBrain().read(platform, question, history)
        trace_step(
            "thinking",
            "ThinkingBrain (orchestrator) finished",
            intent=summary.actual_intent,
            filters=summary.filters,
            group_by=summary.group_by,
        )
    if context.is_contextual:
        summary.references_prior_chat = True

    trace_step("intention", "IntentionAgent started", answer_mode=platform.settings.answer_mode)
    intention = IntentionAgent().understand(platform, question, history, summary)
    note = apply_buyer_spelling(platform, intention)
    if note:
        trace_step("intention", "Buyer/brand spelling corrected", note=note)
    trace_step(
        "intention",
        "IntentionAgent finished",
        intent=intention.intent,
        filters=intention.filters,
        group_by=intention.group_by,
        analysis=intention.analysis,
        suggest=intention.suggest,
        limit=intention.limit,
        preferred_graphs=getattr(intention, "preferred_graphs", None),
    )
    LOGGER.info(
        "orchestrator intention finished intent=%s filters=%s preferred=%s attrs=%s",
        intention.intent,
        intention.filters,
        getattr(intention, "preferred_graphs", None),
        (getattr(intention, "available_attributes", None) or [])[:12],
    )
    if getattr(intention, "available_attributes", None):
        state["available_attributes"] = list(intention.available_attributes)
        trace_debug("Available attributes cached", count=len(intention.available_attributes))

    LOGGER.info("orchestrator graph router started")
    trace_step("router", "GraphRouterAgent started")
    choice = GraphRouterAgent().choose(intention, platform=platform, question=question)
    trace_step(
        "router",
        "GraphRouterAgent finished",
        primary=choice.graph,
        selected=choice.selected_graphs,
        global_kg=choice.use_global_kg,
        coverage=not choice.insufficient_coverage,
        reason=choice.reason,
        confidence=choice.confidence,
    )
    LOGGER.info(
        "orchestrator graph router finished graphs=%s global=%s coverage=%s",
        choice.selected_graphs,
        choice.use_global_kg,
        not choice.insufficient_coverage,
    )

    if choice.insufficient_coverage:
        retrieved = Retrieved(
            text="No registered knowledge graph can answer this question with the available sources.",
            evidence=[],
            caveats=[choice.reason or "insufficient_coverage"],
            unavailable=["knowledge_graph"],
            result_id=None,
        )
        trace_step("retrieve", "Insufficient coverage — no KG query")
        return retrieved, [intention.intent, "insufficient_coverage"], intention, context, choice

    graph_ids = list(choice.selected_graphs or [])
    if graph_id:
        graph_ids = [graph_id]
        trace_step("router", "Forced graph_id override", graph_id=graph_id)
    if choice.use_global_kg and "shared" not in graph_ids:
        graph_ids = [*graph_ids, "shared"]

    result_id = f"result_{uuid.uuid4().hex[:12]}"
    # System/registry overview is one metadata answer — do not fan out per product graph.
    if intention.intent == "system_meta":
        trace_step("retrieve", "System-meta retrieve started", graphs=graph_ids, result_id=result_id)
        retrieved = _retrieve(
            platform,
            principal,
            intention,
            question,
            graph_id=None,
            graph_ids=None,
        )
        retrieved.result_id = result_id
        trace_step(
            "retrieve",
            "System-meta retrieve finished",
            chars=len(retrieved.text or ""),
            evidence=len(retrieved.evidence or []),
        )
        routes = [intention.intent, "system_meta", *graph_ids]
        trace_step("pipeline", "Catalog orchestration complete", routes=routes)
        return retrieved, routes, intention, context, choice

    if len(graph_ids) <= 1:
        gid = graph_ids[0] if graph_ids else graph_id or choice.graph
        trace_step("retrieve", "KG retrieve started", graph_id=gid, result_id=result_id)
        retrieved = _retrieve(
            platform,
            principal,
            intention,
            question,
            graph_id=gid,
            graph_ids=None,
        )
        retrieved.result_id = result_id
        trace_step(
            "retrieve",
            "KG retrieve finished",
            graph_id=gid,
            chars=len(retrieved.text or ""),
            evidence=len(retrieved.evidence or []),
            caveats=len(retrieved.caveats or []),
        )
    else:
        product_ids = [gid for gid in graph_ids if gid not in {"shared", "master"}]
        if not product_ids:
            product_ids = graph_ids
        trace_step("retrieve", "Multi-graph retrieve started", graphs=product_ids, result_id=result_id)
        parts = []
        for gid in product_ids:
            trace_debug("Retrieving graph", graph_id=gid)
            part = _retrieve(
                platform,
                principal,
                intention,
                question,
                graph_id=gid,
                graph_ids=[gid, "shared"] if choice.use_global_kg else [gid],
            )
            parts.append(part)
            trace_debug(
                "Graph retrieve part done",
                graph_id=gid,
                chars=len(part.text or ""),
                evidence=len(part.evidence or []),
            )
        retrieved = merge_retrieved(parts, result_id=result_id)
        trace_step(
            "retrieve",
            "Multi-graph retrieve merged",
            graphs=product_ids,
            chars=len(retrieved.text or ""),
            evidence=len(retrieved.evidence or []),
        )

    if ProductSpecificationAgent().required(intention):
        LOGGER.info("orchestrator product specification agent started")
        trace_step("spec", "ProductSpecificationAgent enrich started")
        retrieved = ProductSpecificationAgent().enrich(retrieved, intention)
        trace_step("spec", "ProductSpecificationAgent enrich finished", chars=len(retrieved.text or ""))

    routes = [intention.intent, *graph_ids]
    trace_step("pipeline", "Catalog orchestration complete", routes=routes)
    return retrieved, routes, intention, context, choice
