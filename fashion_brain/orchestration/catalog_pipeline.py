"""Catalog pipeline: multi-graph LLM Cypher happy path.

Canonical product-catalog flow:
  UserQuestion → LLM understand → select product graphs → load schema card(s)
  → LLM Text2Cypher → Neo4j read → findings → LLM response.

Multi-graph only adds graph selection and combined schema cards. Templates in
``knowledge_graph.retrieval`` (including facet overlap) are fallback when
Text2Cypher is inapplicable, disabled, or fails.

``settings.simple_kg_rag`` keeps the thin schema+question→Cypher path; the
expanded extract/link path remains available when that flag is off.
"""

from __future__ import annotations

import re
import uuid

from knowledge_graph.access import Principal
from knowledge_graph.logging_config import get_logger
from knowledge_graph.pipeline.service import Platform

from fashion_brain.agents.conversation_context import ConversationContextAgent
from fashion_brain.agents.product_spec import ProductSpecificationAgent
from fashion_brain.agents._impl import GraphRouterAgent, IntentionAgent, ThinkingBrain
from fashion_brain.agents.contracts import Intention, Retrieved
from fashion_brain.agents.text2cypher import Text2CypherAgent
from fashion_brain.execution.merger import merge_retrieved
from fashion_brain.trace import trace_debug, trace_step

LOGGER = get_logger("fashion_brain.orchestration")


def _use_simple_kg_rag(platform: Platform) -> bool:
    return bool(getattr(platform.settings, "text2cypher_enabled", True)) and bool(
        getattr(platform.settings, "simple_kg_rag", True)
    )


def _apply_session_graph_memory(
    platform: Platform,
    intention: Intention,
    *,
    question: str,
    history: list | None,
    context,
    state: dict,
    graph_id: str | None,
) -> Intention:
    """Keep the same conversation on the same product KG when brand/buyer is ambiguous.

    Primark (and similar) can appear as brand in ecom and buyer/brand in plmex. If this
    session already answered from ecom_products, prefer that unless the user names another
    catalog. Does not invent filters — only preferred_graphs order.
    """

    sticky = str(state.get("last_graph_id") or graph_id or "").strip()
    if not sticky:
        return intention
    if not (history or context.is_contextual or state.get("last_result_id")):
        return intention
    try:
        from fashion_brain.ask import preferred_graphs_from_question
        from knowledge_graph.retrieval.ecom import registry_product_graphs

        active = set(registry_product_graphs(platform) or [])
    except Exception:
        return intention
    if sticky not in active:
        return intention
    named = list(preferred_graphs_from_question(question, platform) or [])
    # User explicitly named a different catalog → respect that.
    if named and sticky not in named:
        return intention
    preferred = list(getattr(intention, "preferred_graphs", None) or [])
    intention.preferred_graphs = [sticky, *[gid for gid in preferred if gid != sticky]]
    if named:
        for gid in named:
            if gid != sticky and gid not in intention.preferred_graphs:
                intention.preferred_graphs.append(gid)
    trace_step(
        "context",
        "Session graph memory applied",
        last_graph_id=sticky,
        preferred=intention.preferred_graphs,
        contextual=bool(context.is_contextual),
    )
    return intention


def _remember_graph(state: dict, graph_ids: list[str] | None, graph_id: str | None = None) -> None:
    """Persist which product graph this turn used (for the next message in the same chat)."""

    for gid in list(graph_ids or []) + ([graph_id] if graph_id else []):
        if gid and gid not in {"shared", "master", "kg_system"}:
            state["last_graph_id"] = str(gid)
            return


def _simple_intention(platform: Platform, question: str) -> Intention:
    """Cheap routing only — no LLM, no metadata value probing.

    Do NOT call build_field_catalog(question): that fans out dozens of
    /api/metadata/values HTTP calls and is a major latency source.
    Graph pick = registry token match, else default product graph.
    Cypher expert reads schema + question.
    """

    from fashion_brain.ask import plan_routes, preferred_graphs_from_question
    from fashion_brain.planning.kg_catalog import SYSTEM_GRAPH_ID
    from fashion_brain.planning.runtime_schema import resolve_product_graph_id

    route = plan_routes(question)[0]
    low = (question or "").casefold()
    # Lightweight system-meta detect (no catalog probe).
    if any(
        phrase in low
        for phrase in (
            "knowledge graph",
            "registered graph",
            "what graphs",
            "system meta",
            "what sources",
            "registry",
        )
    ):
        return Intention(
            "system_meta",
            {},
            None,
            False,
            "simple_kg_rag",
            preferred_graphs=[SYSTEM_GRAPH_ID],
        )
    if route in {"live_sql", "news"}:
        return Intention(route, {}, None, False, "simple_kg_rag", preferred_graphs=[])

    preferred = list(preferred_graphs_from_question(question, platform) or [])
    if route in {"shared", "shared_products"}:
        facet = None
        try:
            from knowledge_graph.retrieval.ecom import overlap_facet_from_question

            facet = overlap_facet_from_question(question, platform, preferred or None)
        except Exception:
            facet = None
        return Intention(
            "shared_products",
            {},
            facet,
            False,
            "simple_kg_rag",
            preferred_graphs=preferred,
        )

    if not preferred:
        gid = resolve_product_graph_id(platform)
        if gid:
            preferred = [gid]
    return Intention(
        route if route else "other",
        {},
        None,
        False,
        "simple_kg_rag",
        preferred_graphs=preferred,
    )


def _load_brands(platform: Platform, principal: Principal, gid: str) -> list[dict]:
    from fashion_brain.planning.schema_cache import get_brands

    return get_brands(platform, gid, principal)


def _product_graph_ids(graph_ids: list[str] | None, graph_id: str | None = None) -> list[str]:
    ids = [str(item) for item in (graph_ids or []) if item and item not in {"shared", "master"}]
    if not ids and graph_id and graph_id not in {"shared", "master"}:
        ids = [str(graph_id)]
    return ids


def _is_cross_graph_compare(question: str, intention: Intention, graph_ids: list[str] | None) -> bool:
    product_ids = _product_graph_ids(graph_ids)
    if len(product_ids) < 2:
        return False
    if intention.intent == "shared_products":
        return True
    text = f"{question} {getattr(intention, 'thinking', '') or ''}".casefold()
    return bool(
        re.search(
            r"\b(across|between|among|both\s+(?:graphs|sources)|cross[- ]source|shared)\b",
            text,
        )
    )


def _retrieve_simple_kg_rag(
    platform: Platform,
    principal: Principal,
    intention: Intention,
    question: str,
    *,
    graph_id: str | None,
    result_id: str | None,
    graph_ids: list[str] | None = None,
) -> Retrieved:
    """schema + brands + question → one Text2Cypher LLM → rows. No extract/link LLM."""

    from fashion_brain.ask import _retrieve
    from fashion_brain.stage_timing import timed

    selected = _product_graph_ids(graph_ids, graph_id)
    gid = selected[0] if selected else graph_id
    if not gid or not Text2CypherAgent().applicable(intention):
        return _retrieve(
            platform,
            principal,
            intention,
            question,
            graph_id=gid if len(selected) <= 1 else None,
            graph_ids=selected if len(selected) > 1 else None,
        )

    brands: list[dict] = []
    try:
        with timed("kg_brands", graph_id=gid):
            brands = _load_brands(platform, principal, gid)
    except Exception as exc:
        LOGGER.warning("brand list failed (continuing): %s", exc)

    multi = len(selected) > 1
    trace_step("text2cypher", "Simple KG-RAG Cypher started", graph_id=gid, graphs=selected if multi else None)
    with timed("text2cypher", graph_id=gid, simple=True, multi=multi):
        llm_hit = Text2CypherAgent().retrieve(
            platform,
            principal,
            question,
            intention,
            gid,
            graph_ids=selected if multi else None,
            resolved=None,
            extracted=None,
            brands=brands,
        )
    if llm_hit is not None:
        if result_id:
            llm_hit.result_id = result_id
        trace_step(
            "text2cypher",
            "Simple KG-RAG Cypher finished",
            chars=len(llm_hit.text or ""),
            evidence=len(llm_hit.evidence or []),
        )
        return llm_hit

    trace_step("text2cypher", "Simple KG-RAG missed — templated fallback", graph_id=gid, graphs=selected)
    retrieved = _retrieve(
        platform,
        principal,
        intention,
        question,
        graph_id=gid if not multi else None,
        graph_ids=selected if multi else None,
    )
    if result_id:
        retrieved.result_id = result_id
    return retrieved


def _retrieve_with_text2cypher(
    platform: Platform,
    principal: Principal,
    intention,
    question: str,
    *,
    graph_id: str | None,
    graph_ids: list[str] | None,
    result_id: str | None,
    effective_question: str | None = None,
    prior_query: dict | None = None,
    reuse_prior_query: bool = False,
) -> Retrieved:
    """LLM EntityExtract → optional soft link → LLM Text2Cypher.

    When ``reuse_prior_query`` and a prior Cypher exist, skip re-extracting past
    entities and adapt the stored query for the follow-up instead of rebuilding it.
    """

    ask_text = (effective_question or question or "").strip() or question
    gid = graph_id
    prior = dict(prior_query or {})
    prior_cypher = str(prior.get("cypher") or "").strip() or None
    prior_params = dict(prior.get("params") or {}) if prior.get("params") else None
    adapt = bool(reuse_prior_query and prior_cypher)

    # Deterministic temporal paths (price / availability events) — skip LLM Cypher guesswork.
    try:
        from fashion_brain.ask import _retrieve, availability_history_mode
        from fashion_brain.stage_timing import timed
        from knowledge_graph.retrieval.ecom import price_history_mode

        # Mode from the latest user question only — refined follow-up text must not
        # reclassify "recent products" as the prior brand-ranking ask.
        latest = (question or "").strip() or ask_text
        mode = price_history_mode(latest)
        avail_mode = availability_history_mode(latest)
        if mode or avail_mode:
            # Use module-level trace_step — never re-import it locally in this function.
            trace_step(
                "temporal_history",
                "Using templated temporal retrieve",
                graph_id=graph_id,
                price_mode=mode,
                availability_mode=avail_mode,
            )
            with timed(
                "kg_retrieve",
                graphs=[graph_id] if graph_id else graph_ids,
                price_history=mode,
                availability_history=avail_mode,
            ):
                retrieved = _retrieve(
                    platform,
                    principal,
                    intention,
                    latest,
                    graph_id=graph_id,
                    graph_ids=graph_ids,
                )
            if result_id:
                retrieved.result_id = result_id
            return retrieved
    except Exception:
        pass

    selected_products = _product_graph_ids(graph_ids, graph_id)
    cross_compare = _is_cross_graph_compare(ask_text, intention, selected_products)

    # Counts and grouped breakdowns already have schema-aware KG templates.
    # Skip EntityExtract + Text2Cypher LLM calls for these deterministic asks.
    # Cross-catalog / shared-across always stays on Text2Cypher (multi-graph).
    if (
        intention.intent in {"product_count", "count", "breakdown"}
        and not bool(getattr(intention, "analysis", False))
        and not bool(getattr(intention, "suggest", False))
        and not cross_compare
        and intention.intent != "shared_products"
    ):
        trace_step(
            "retrieve",
            "Using deterministic catalog count template",
            graph_id=gid,
            filters=intention.filters,
            group_by=intention.group_by,
            count_entity=getattr(intention, "count_entity", None),
        )
        with timed("kg_retrieve", graph_id=gid, deterministic_count=True):
            retrieved = _retrieve(
                platform,
                principal,
                intention,
                ask_text,
                graph_id=gid,
                graph_ids=None,
            )
        if result_id:
            retrieved.result_id = result_id
        return retrieved

    # Simple schema + question path; extraction/linking can be enabled when needed.
    if _use_simple_kg_rag(platform):
        return _retrieve_simple_kg_rag(
            platform,
            principal,
            intention,
            ask_text,
            graph_id=graph_id or (selected_products[0] if selected_products else None),
            result_id=result_id,
            graph_ids=selected_products if cross_compare else graph_ids,
        )

    from fashion_brain.ask import _retrieve, is_open_product_listing
    from fashion_brain.agents.entity_extract import EntityExtractAgent
    from fashion_brain.stage_timing import timed

    # Open "which products use X" asks: templated total + brand/category mix + examples.
    # Text2Cypher otherwise emits LIMIT 40–50 name dumps that drown the answer.
    # Always re-ground filters from the latest question so prior Primark|Zara cannot stick.
    latest_q = (question or "").strip() or ask_text
    if (
        not getattr(intention, "analysis", False)
        and is_open_product_listing(latest_q, limit=intention.limit)
        and intention.intent in {"product_count", "breakdown", "list", "count", "products"}
    ):
        from fashion_brain.ask import resolve_catalog_plan

        latest_plan = resolve_catalog_plan(latest_q, None, carry=False)
        latest_filters = dict(latest_plan.get("filters") or {})
        if latest_filters:
            intention.filters = latest_filters
            intention.group_by = None
            intention.intent = "product_count"
            trace_step(
                "open_listing",
                "Using templated count + mix + examples",
                graph_id=graph_id,
                filters=intention.filters,
            )
            with timed(
                "kg_retrieve",
                graphs=[graph_id] if graph_id else graph_ids,
                open_listing=True,
            ):
                retrieved = _retrieve(
                    platform,
                    principal,
                    intention,
                    latest_q,
                    graph_id=graph_id,
                    graph_ids=graph_ids,
                )
            if result_id:
                retrieved.result_id = result_id
            return retrieved

    gid = graph_id
    resolved = dict(prior.get("resolved") or {}) if adapt else None
    extracted = dict(prior.get("extracted") or {}) if adapt else None
    brands = None
    schema = None

    if gid and Text2CypherAgent().applicable(intention):
        try:
            from fashion_brain.execution.kg_client import kg_request, uses_kg_api
            from knowledge_graph.linking.entity_linker import (
                clarification_message,
                link_entities,
                resolved_payload,
            )

            brands = _load_brands(platform, principal, gid)
            try:
                from fashion_brain.planning.runtime_schema import get_runtime_schema

                schema = get_runtime_schema(platform, gid)
            except Exception:
                schema = None

            if adapt:
                # Past query already grounded entities — do not re-invent them.
                if intention.group_by and isinstance(extracted, dict):
                    extracted = dict(extracted)
                    extracted["group_by"] = intention.group_by
                trace_step(
                    "entity_extract",
                    "Skipped extract — adapting prior Cypher",
                    prior_params=list((prior_params or {}).keys()),
                    filters=intention.filters,
                    group_by=intention.group_by,
                )
            else:
                # --- LLM EntityExtract agent (standalone / new topic) ---
                with timed("entity_extract", graph_id=gid):
                    extracted_obj = EntityExtractAgent().extract(
                        platform, ask_text, brands=brands, schema=schema
                    )
                    # Never copy keyword garment leftovers into the LLM extract —
                    # EntityExtract + live brands own party grounding.
                    for key, value in (intention.filters or {}).items():
                        if not value or key in {"garment", "garment_type"}:
                            continue
                        if hasattr(extracted_obj, key) and not getattr(extracted_obj, key):
                            setattr(extracted_obj, key, str(value))
                    extracted = extracted_obj.to_dict()
                    if extracted_obj.count_entity:
                        intention.count_entity = extracted_obj.count_entity
                    if extracted_obj.group_by and not intention.group_by:
                        intention.group_by = extracted_obj.group_by
                    if getattr(extracted_obj, "country", None):
                        intention.filters = dict(intention.filters or {})
                        intention.filters["country"] = extracted_obj.country
                        intention.filters.pop("garment", None)
                    # Keep analysis subject (brand/buyer) on intention for multi-facet retrieve.
                    intention.filters = dict(intention.filters or {})
                    for facet in ("brand", "buyer", "material", "color", "category", "gender"):
                        value = getattr(extracted_obj, facet, None)
                        if value and not intention.filters.get(facet):
                            intention.filters[facet] = str(value)
                    # EntityExtract + live Brand nodes own party grounding.
                    from fashion_brain.ask import (
                        _compare_party_names,
                        ground_filters_from_live_brands,
                        is_composition_compare_question,
                        split_multi_party_filter,
                    )
                    from knowledge_graph.ontology import normalize_name

                    brand_names = [
                        str(b.get("name") or "") for b in (brands or []) if b.get("name")
                    ]
                    named_compare = _compare_party_names(ask_text)
                    composition_compare = is_composition_compare_question(ask_text)
                    prior_multi = split_multi_party_filter(
                        str((intention.filters or {}).get("brand") or "")
                    )
                    if named_compare:
                        # Never collapse Primark|Zara to a single extracted brand.
                        intention.filters["brand"] = " | ".join(named_compare)
                    elif len(prior_multi) >= 2:
                        intention.filters["brand"] = " | ".join(prior_multi)
                    elif extracted_obj.brand:
                        intention.filters["brand"] = str(extracted_obj.brand)
                        if not composition_compare:
                            intention.filters.pop("garment", None)
                            intention.filters.pop("garment_type", None)
                    if getattr(extracted_obj, "category", None):
                        intention.filters["category"] = str(extracted_obj.category)
                        if not composition_compare:
                            intention.filters.pop("garment", None)
                    if composition_compare:
                        intention.group_by = intention.group_by or "material"
                    intention.filters = ground_filters_from_live_brands(
                        ask_text, intention.filters, brand_names
                    )
                    if intention.filters.get("brand") or intention.filters.get("buyer"):
                        brand_key = normalize_name(
                            str(intention.filters.get("brand") or "").split("|")[0]
                        )
                        for junk in ("garment", "garment_type"):
                            if intention.filters.get(junk) and normalize_name(
                                str(intention.filters.get(junk))
                            ) == brand_key:
                                intention.filters.pop(junk, None)
                    # Full brand/party analysis stays multi-facet — do not demote to one breakdown.
                    if (
                        extracted_obj.intent == "breakdown"
                        and not bool(getattr(intention, "analysis", False))
                        and intention.intent
                        in {
                            "count",
                            "list",
                            "other",
                            "product_count",
                        }
                    ):
                        intention.intent = "breakdown"
                    trace_step(
                        "entity_extract",
                        "EntityExtractAgent finished",
                        extracted=extracted,
                    )

                # --- Optional soft linker ---
                if getattr(platform.settings, "entity_linker_enabled", True) and extracted_obj.as_link_dict():
                    with timed("entity_link", graph_id=gid):
                        if uses_kg_api(platform.settings):
                            link_payload = kg_request(
                                platform.settings,
                                "POST",
                                "/brain/link_entities",
                                clearance=getattr(principal, "clearance", None) or "internal",
                                json_body={"graph_id": gid, "entities": extracted_obj.as_link_dict()},
                            )
                            results_raw = (link_payload or {}).get("results") or []
                            from knowledge_graph.linking.entity_linker import LinkHit, LinkResult

                            link_results = []
                            for item in results_raw:
                                hits = [
                                    LinkHit(
                                        label=str(h.get("label") or ""),
                                        id=str(h.get("id") or ""),
                                        name=str(h.get("name") or ""),
                                        normalized_name=str(h.get("normalized_name") or ""),
                                        score=float(h.get("score") or 0),
                                        source=str(h.get("source") or "exact"),
                                    )
                                    for h in (item.get("hits") or [])
                                ]
                                link_results.append(
                                    LinkResult(
                                        field=str(item.get("field") or ""),
                                        term=str(item.get("term") or ""),
                                        status=str(item.get("status") or "no_match"),
                                        hits=hits,
                                    )
                                )
                        else:
                            link_results = link_entities(
                                platform, extracted_obj.as_link_dict(), graph_id=gid
                            )

                        clarify = clarification_message(
                            link_results, require_hard_match=False, schema=schema
                        )
                        resolved = {
                            k: v
                            for k, v in (resolved_payload(link_results) or {}).items()
                            if (v or {}).get("status") != "no_match"
                        }
                        trace_step(
                            "entity_link",
                            "Entity linker finished",
                            resolved_fields=list(resolved.keys()),
                            statuses={r.field: r.status for r in link_results},
                        )
                        if clarify:
                            retrieved = Retrieved(
                                text=clarify,
                                evidence=[],
                                caveats=["entity_link_no_match"],
                                unavailable=[],
                                synthetic_data=False,
                            )
                            if result_id:
                                retrieved.result_id = result_id
                            return retrieved
        except Exception as exc:
            LOGGER.warning("entity extract/link failed; continuing: %s", exc)
            try:
                trace_step("entity_extract", "Entity extract/link failed", error=str(exc)[:200])
            except Exception:
                LOGGER.debug("trace_step unavailable while reporting extract failure", exc_info=False)

    # Detailed analysis = many attribute-wise Cyphers (overview + top-N per facet),
    # not one shallow aggregate from Text2Cypher.
    if gid and bool(getattr(intention, "analysis", False)) and not intention.group_by:
        if intention.intent not in {"product_count", "analysis"}:
            intention.intent = "product_count"
        trace_step(
            "text2cypher",
            "Skipped single Cypher — multi-facet analysis profile",
            graph_id=gid,
            filters=intention.filters,
            limit=intention.limit,
        )
        with timed("kg_retrieve", graph_id=gid, analysis=True):
            retrieved = _retrieve(
                platform,
                principal,
                intention,
                ask_text,
                graph_id=gid,
                graph_ids=None,
            )
        if result_id:
            retrieved.result_id = result_id
        return retrieved

    if gid and Text2CypherAgent().applicable(intention):
        trace_step(
            "text2cypher",
            "Text2CypherAgent started",
            graph_id=gid,
            intent=intention.intent,
            adapt_prior=adapt,
        )
        multi_ids = selected_products if cross_compare and len(selected_products) > 1 else None
        with timed("text2cypher", graph_id=gid, adapt_prior=adapt, multi=bool(multi_ids)):
            llm_hit = Text2CypherAgent().retrieve(
                platform,
                principal,
                ask_text,
                intention,
                gid,
                graph_ids=multi_ids,
                resolved=resolved,
                extracted=extracted,
                brands=brands,
                prior_cypher=prior_cypher if adapt else None,
                prior_params=prior_params if adapt else None,
            )
        if llm_hit is not None:
            if result_id:
                llm_hit.result_id = result_id
            trace_step(
                "text2cypher",
                "Text2CypherAgent finished",
                chars=len(llm_hit.text or ""),
                evidence=len(llm_hit.evidence or []),
                adapted=adapt,
                graphs=multi_ids,
            )
            return llm_hit

    fallback_ids = selected_products if cross_compare and len(selected_products) > 1 else graph_ids
    retrieved = _retrieve(
        platform,
        principal,
        intention,
        question,
        graph_id=graph_id if not (cross_compare and len(selected_products) > 1) else None,
        graph_ids=fallback_ids,
    )
    if result_id:
        retrieved.result_id = result_id
    return retrieved


def draft_catalog_orchestrated(
    platform: Platform,
    question: str,
    principal: Principal,
    history: list | None = None,
    summary=None,
    graph_id: str | None = None,
    conversation_state: dict | None = None,
):
    """Run catalog QA: understand → select graphs → schema → Text2Cypher → Neo4j → findings.

    Templates are used only when Text2Cypher is skipped or fails.
    """

    from fashion_brain.ask import apply_buyer_spelling, _retrieve
    from fashion_brain.stage_timing import timed

    state = dict(conversation_state or {})
    trace_step("context", "ConversationContextAgent started", history_turns=len(history or []))
    with timed("context_agent"):
        context = ConversationContextAgent().analyze(
            question,
            history,
            conversation_state=state,
            available_attributes=state.get("available_attributes"),
            platform=platform,
        )
    trace_step(
        "context",
        "ConversationContextAgent finished",
        decision=context.decision,
        contextual=context.is_contextual,
        reuse=context.reuse_previous_results,
        refined=(context.refined_question or "")[:160],
    )

    if context.decision == "REUSE" and state.get("last_retrieved_text"):
        retrieved = Retrieved(
            text=str(state.get("last_retrieved_text") or ""),
            evidence=[],
            caveats=["Reused previous retrieval without a new knowledge-graph query."],
            unavailable=[],
            result_id=context.previous_result_id,
        )
        intention = IntentionAgent().understand(platform, question, history, summary)
        try:
            from fashion_brain.ask import catalog_metric_family

            state["last_metric_family"] = catalog_metric_family(question)
        except Exception:
            pass
        if conversation_state is not None:
            conversation_state.clear()
            conversation_state.update(state)
        trace_step("retrieve", "Reused prior retrieval — skipped KG query")
        return retrieved, [intention.intent, "reuse"], intention, context, None

    LOGGER.info("orchestrator LLM-agent path question=%s", question)
    # Standalone Text2Cypher asks: skip ThinkingBrain (EntityExtract plans filters).
    # Follow-ups still use ThinkingBrain for references_prior_chat.
    text2cypher_on = bool(getattr(platform.settings, "text2cypher_enabled", True))
    if summary is None and (history or not text2cypher_on):
        trace_step("thinking", "ThinkingBrain (orchestrator) started")
        summary = ThinkingBrain().read(
            platform, question, history, conversation_state=state
        )
        trace_step(
            "thinking",
            "ThinkingBrain (orchestrator) finished",
            intent=summary.actual_intent,
            filters=summary.filters,
            group_by=summary.group_by,
        )
    elif summary is None:
        trace_step("thinking", "ThinkingBrain skipped (standalone Text2Cypher path)")
    if summary is not None and context.is_contextual:
        summary.references_prior_chat = True
    # Same conversation id with prior turns → always tell intention/planner that memory applies
    # for catalog stickiness (even when filter carry is empty).
    if summary is not None and history and (context.is_contextual or state.get("last_graph_id")):
        summary.references_prior_chat = True

    with timed("intention_agent"):
        intention = IntentionAgent().understand(platform, question, history, summary)
    note = apply_buyer_spelling(platform, intention)
    if note:
        trace_step("intention", "Buyer/brand spelling corrected", note=note)
    try:
        from fashion_brain.planning.query_planner import plan_retrieve

        intention = plan_retrieve(intention, question)
        trace_step(
            "planner",
            "QueryPlanner finished",
            retrieve_mode=intention.retrieve_mode,
            response_type=intention.response_type,
            metrics=intention.metrics,
            operations=intention.operations,
        )
    except Exception as exc:
        LOGGER.warning("query planner skipped: %s", exc)
    # After planner: pin preferred_graphs to this conversation's last catalog when ambiguous.
    intention = _apply_session_graph_memory(
        platform,
        intention,
        question=question,
        history=history,
        context=context,
        state=state,
        graph_id=graph_id,
    )

    # Schema-aware constraint carry (country/origin, follow-up group_by, effective question).
    schema = None
    try:
        from fashion_brain.planning.runtime_schema import get_runtime_schema, resolve_product_graph_id

        pref = (getattr(intention, "preferred_graphs", None) or [None])[0] or graph_id
        schema = get_runtime_schema(platform, pref or resolve_product_graph_id(platform))
    except Exception:
        schema = None
    from fashion_brain.planning.active_constraints import apply_turn_constraints

    # Carry only when context both marks the turn contextual AND supplies filters to keep.
    # (is_contextual alone used to glue Primark|Zara onto unrelated cotton asks.)
    if context.active_filters:
        state["active_filters"] = dict(context.active_filters)
    filters, group_by, effective_question, state_patch = apply_turn_constraints(
        question,
        history,
        intention.filters,
        intention.group_by,
        schema=schema,
        conversation_state=state,
        is_contextual=bool(context.is_contextual),
        carry_filters=bool(context.is_contextual and context.active_filters),
        refined_question=context.refined_question,
    )
    intention.filters = dict(filters)
    temporal_ask = False
    avail_mode = None
    price_mode = None
    try:
        from fashion_brain.ask import availability_history_mode, temporal_count_entity
        from knowledge_graph.retrieval.ecom import price_history_mode

        avail_mode = availability_history_mode(question)
        price_mode = price_history_mode(question)
        temporal_ask = bool(price_mode or avail_mode or temporal_count_entity(question))
    except Exception:
        temporal_ask = False
    if temporal_ask:
        # Temporal templates key off product_count; never demote them to breakdown
        # just because a prior turn carried a group_by.
        if intention.intent in {"breakdown", "count", "list", "other", "product_count"}:
            intention.intent = "product_count"
        # Only keep a breakdown dimension when this turn itself is a brand/category ranking.
        if avail_mode == "events_by_brand" or price_mode == "events_by_brand":
            intention.group_by = "brand"
        elif price_mode == "trend_by_category":
            intention.group_by = "category"
        else:
            intention.group_by = None
    elif group_by:
        intention.group_by = group_by
        if context.is_contextual and intention.intent in {
            "product_count",
            "count",
            "list",
            "other",
        }:
            intention.intent = "breakdown"
    state.update(state_patch)
    trace_step(
        "intention",
        "IntentionAgent finished",
        intent=intention.intent,
        filters=intention.filters,
        group_by=intention.group_by,
        preferred_graphs=getattr(intention, "preferred_graphs", None),
        effective_question=effective_question[:160] if effective_question != question else None,
    )
    if getattr(intention, "available_attributes", None):
        state["available_attributes"] = list(intention.available_attributes)

    with timed("graph_router"):
        choice = GraphRouterAgent().choose(intention, platform=platform, question=question)
    trace_step(
        "router",
        "GraphRouterAgent finished",
        primary=choice.graph,
        selected=choice.selected_graphs,
        coverage=not choice.insufficient_coverage,
    )

    if choice.insufficient_coverage:
        retrieved = Retrieved(
            text="No registered knowledge graph can answer this question with the available sources.",
            evidence=[],
            caveats=[choice.reason or "insufficient_coverage"],
            unavailable=["knowledge_graph"],
            result_id=None,
        )
        return retrieved, [intention.intent, "insufficient_coverage"], intention, context, choice

    graph_ids = list(choice.selected_graphs or [])
    # Request-scoped graph pin, except multi-catalog shared/cross asks from the router.
    if graph_id and not (
        intention.intent == "shared_products" and len(choice.selected_graphs or []) > 1
    ):
        graph_ids = [graph_id]
    if choice.use_global_kg and "shared" not in graph_ids:
        graph_ids = [*graph_ids, "shared"]

    result_id = f"result_{uuid.uuid4().hex[:12]}"
    if intention.intent == "system_meta":
        with timed("kg_retrieve", graphs=graph_ids, intent="system_meta"):
            retrieved = _retrieve(
                platform, principal, intention, question, graph_id=None, graph_ids=None
            )
        retrieved.result_id = result_id
        return retrieved, [intention.intent, "system_meta", *graph_ids], intention, context, choice

    prior_query = {
        "cypher": state.get("last_cypher"),
        "params": state.get("last_cypher_params") or {},
        "resolved": state.get("last_resolved") or {},
        "extracted": state.get("last_extracted") or {},
        "graph_id": state.get("last_graph_id"),
    }
    reuse_prior = bool(context.is_contextual and prior_query.get("cypher"))

    product_ids = _product_graph_ids(graph_ids, graph_id or choice.graph)
    cross_compare = _is_cross_graph_compare(
        effective_question or question, intention, product_ids
    )

    # Graph-agnostic analysis tool path (any product KG — not ecom-only).
    if (
        getattr(intention, "retrieve_mode", None) == "analytical_tool"
        or (bool(getattr(intention, "analysis", False)) and not intention.group_by)
    ) and intention.intent not in {"system_meta", "live_sql", "news", "shared_products"}:
        from fashion_brain.analytics.engine import pick_analysis_graphs, run_product_analysis

        analysis_graphs = pick_analysis_graphs(
            platform,
            preferred=list(getattr(intention, "preferred_graphs", None) or []),
            graph_id=product_ids[0] if len(product_ids) == 1 else None,
            graph_ids=product_ids if len(product_ids) > 1 else None,
            filters=dict(intention.filters or {}),
        )
        if not analysis_graphs and product_ids:
            analysis_graphs = product_ids[:1]
        with timed("kg_retrieve", graphs=analysis_graphs, analysis=True, analytical_tool=True):
            retrieved = run_product_analysis(
                platform,
                principal,
                graph_ids=analysis_graphs,
                filters=dict(intention.filters or {}),
                limit=intention.limit,
                group_by=intention.group_by,
            )
        retrieved.result_id = result_id
        retrieved.response_type = intention.response_type or retrieved.response_type
        if ProductSpecificationAgent().required(intention):
            retrieved = ProductSpecificationAgent().enrich(retrieved, intention)
        state["active_filters"] = dict(intention.filters or {})
        state["group_by"] = intention.group_by
        state["last_result_id"] = result_id
        state["last_retrieved_text"] = retrieved.text
        _remember_graph(state, analysis_graphs)
        try:
            from fashion_brain.ask import catalog_metric_family

            state["last_metric_family"] = catalog_metric_family(effective_question or question)
        except Exception:
            pass
        if conversation_state is not None:
            conversation_state.clear()
            conversation_state.update(state)
        return retrieved, [intention.intent, "analytical_tool", *analysis_graphs], intention, context, choice

    if (
        len(product_ids) > 1
        and (
            cross_compare
            or intention.intent == "specification"
            or bool(getattr(intention, "suggest", False))
        )
    ):
        # Compare asks + dual-catalog specs (ecom retail + PLM development) in one retrieve.
        with timed("kg_retrieve", graphs=product_ids, multi_graph_cypher=True):
            retrieved = _retrieve_with_text2cypher(
                platform,
                principal,
                intention,
                question,
                graph_id=product_ids[0],
                graph_ids=product_ids,
                result_id=result_id,
                effective_question=effective_question,
                prior_query=prior_query,
                reuse_prior_query=False,
            )
    elif len(product_ids) <= 1:
        gid = product_ids[0] if product_ids else graph_id or choice.graph
        retrieved = _retrieve_with_text2cypher(
            platform,
            principal,
            intention,
            question,
            graph_id=gid,
            graph_ids=None,
            result_id=result_id,
            effective_question=effective_question,
            prior_query=prior_query,
            reuse_prior_query=reuse_prior
            and (not prior_query.get("graph_id") or prior_query.get("graph_id") == gid),
        )
    else:
        # Independent multi-catalog ask (no compare): per-graph Cypher + merge.
        parts = []
        with timed("kg_retrieve", graphs=product_ids, multi=True):
            for gid in product_ids:
                parts.append(
                    _retrieve_with_text2cypher(
                        platform,
                        principal,
                        intention,
                        question,
                        graph_id=gid,
                        graph_ids=[gid],
                        result_id=None,
                        effective_question=effective_question,
                        prior_query=prior_query,
                        reuse_prior_query=reuse_prior
                        and (not prior_query.get("graph_id") or prior_query.get("graph_id") == gid),
                    )
                )
        retrieved = merge_retrieved(parts, result_id=result_id)

    if ProductSpecificationAgent().required(intention):
        retrieved = ProductSpecificationAgent().enrich(retrieved, intention)

    # Persist constraints + working Cypher for the next turn.
    state["active_filters"] = dict(intention.filters or {})
    state["group_by"] = intention.group_by
    state["last_result_id"] = result_id
    state["last_retrieved_text"] = retrieved.text
    try:
        from fashion_brain.ask import catalog_metric_family

        state["last_metric_family"] = catalog_metric_family(
            effective_question or question
        )
    except Exception:
        pass
    if getattr(retrieved, "cypher", None):
        state["last_cypher"] = retrieved.cypher
        state["last_cypher_params"] = dict(getattr(retrieved, "cypher_params", None) or {})
        state["last_resolved"] = dict(getattr(retrieved, "resolved", None) or {})
        state["last_extracted"] = dict(getattr(retrieved, "extracted", None) or {})
    _remember_graph(
        state,
        list(product_ids or graph_ids or []),
        graph_id or (choice.graph if choice else None),
    )
    if conversation_state is not None:
        conversation_state.clear()
        conversation_state.update(state)

    routes = [intention.intent, *graph_ids]
    trace_step("pipeline", "Catalog orchestration complete", routes=routes)
    return retrieved, routes, intention, context, choice
