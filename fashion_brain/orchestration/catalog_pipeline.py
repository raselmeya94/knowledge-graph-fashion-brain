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
from fashion_brain.agents.contracts import (
    ClarifyResult,
    ContextDecision,
    Intention,
    ReadyResult,
    Retrieved,
    ValidatedIntent,
)
from fashion_brain.agents.query_understanding import (
    QueryUnderstandingAgent,
    apply_validated_intent,
    store_understanding,
)
from fashion_brain.agents.text2cypher import Text2CypherAgent
from fashion_brain.execution.merger import merge_retrieved
from fashion_brain.trace import trace_debug, trace_step

LOGGER = get_logger("fashion_brain.orchestration")


def _ontology_domains_for_question(platform: Platform, question: str) -> list[str]:
    try:
        from fashion_brain.planning.domain_schema_filter import classify_ontology_domains

        return [
            str(getattr(p, "ontology_id", "") or "")
            for p in classify_ontology_domains(platform, question, limit=3)
            if getattr(p, "ontology_id", None)
        ]
    except Exception:
        return []


def _try_query_cache(
    platform: Platform,
    question: str,
    *,
    ontology_domains: list[str] | None,
    graph_ids: list[str] | None,
    metric_family: str,
    intent: str,
) -> dict | None:
    try:
        ttl = float(getattr(platform.settings, "query_cache_ttl_seconds", 90) or 0)
        if ttl <= 0:
            return None
        from fashion_brain.retrieval.query_result_cache import get_cached_result

        return get_cached_result(
            question,
            platform=platform,
            ontology_domains=ontology_domains,
            graph_ids=graph_ids,
            metric_family=metric_family,
            intent=intent,
        )
    except Exception as exc:
        LOGGER.debug("query cache lookup skipped: %s", exc)
        return None


def _store_query_cache(
    platform: Platform,
    question: str,
    retrieved: Retrieved,
    *,
    ontology_domains: list[str] | None,
    graph_ids: list[str] | None,
    metric_family: str,
    intent: str,
) -> None:
    try:
        ttl = float(getattr(platform.settings, "query_cache_ttl_seconds", 90) or 0)
        if ttl <= 0 or not (retrieved.text or "").strip():
            return
        from fashion_brain.retrieval.query_result_cache import put_cached_result

        put_cached_result(
            question,
            text=retrieved.text,
            platform=platform,
            evidence=list(retrieved.evidence or []),
            caveats=list(retrieved.caveats or []),
            ontology_domains=ontology_domains,
            graph_ids=graph_ids,
            metric_family=metric_family,
            intent=intent,
        )
    except Exception as exc:
        LOGGER.debug("query cache store skipped: %s", exc)


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
    try:
        from fashion_brain.ask import preferred_graphs_from_question, question_has_catalog_deixis
        from knowledge_graph.retrieval.ecom import registry_product_graphs

        active = set(registry_product_graphs(platform) or [])
        deixis = question_has_catalog_deixis(question)
    except Exception:
        return intention
    # Prior catalog + "of them" / history / contextual turn → stay pinned.
    if not (
        history
        or context.is_contextual
        or state.get("last_result_id")
        or deixis
    ):
        return intention
    if sticky not in active:
        return intention
    named = list(preferred_graphs_from_question(question, platform) or [])
    # User explicitly named a different catalog → respect that.
    if named and sticky not in named:
        return intention
    # Cross-source asks keep multi-graph preferred lists.
    low = f" {str(question or '').casefold()} "
    wants_cross = any(
        hint in low
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
        )
    ) or len(named) >= 2
    if wants_cross:
        preferred = list(getattr(intention, "preferred_graphs", None) or [])
        intention.preferred_graphs = [sticky, *[gid for gid in preferred if gid != sticky]]
        for gid in named:
            if gid != sticky and gid not in intention.preferred_graphs:
                intention.preferred_graphs.append(gid)
    else:
        # Continuous turn on the same chat: stay on the sticky product KG only.
        # Color/garment value hits often exist on both ecom and plmex; that must not
        # fan out after this session already answered from one catalog (T2→T3).
        intention.preferred_graphs = [sticky]
    trace_step(
        "context",
        "Session graph memory applied",
        last_graph_id=sticky,
        preferred=intention.preferred_graphs,
        contextual=bool(context.is_contextual),
        pinned=not wants_cross,
    )
    return intention


def _prefer_ecom_for_price_history(
    platform: Platform,
    intention: Intention,
    *,
    question: str,
) -> Intention:
    """PriceChangeEvent / movers / trends live on ecom — not PLM.

    Session sticky often pins plmex after assortment asks; plmex has 0 price
    events, so trend questions would answer empty. Force ecom unless the user
    explicitly named another catalog.
    """

    try:
        from fashion_brain.ask import (
            _is_availability_history_question,
            _is_price_history_question,
            is_price_trend_analysis_ask,
        )
        from knowledge_graph.retrieval.ecom import price_history_mode, registry_product_graphs

        q = question or ""
        from fashion_brain.ask import is_availability_price_ops_brief_ask

        if not (
            is_price_trend_analysis_ask(q)
            or _is_price_history_question(q)
            or price_history_mode(q)
            or _is_availability_history_question(q)
            or is_availability_price_ops_brief_ask(q)
        ):
            return intention
    except Exception:
        return intention
    if _explicit_graphs_in_question(question, platform):
        return intention
    ecom = "ecom_products"
    try:
        for cand in registry_product_graphs(platform) or []:
            low = str(cand).casefold()
            if "ecom" in low or low in {"master_products", "master"}:
                ecom = str(cand)
                break
    except Exception:
        pass
    prefs = [str(g) for g in (getattr(intention, "preferred_graphs", None) or []) if g]
    intention.preferred_graphs = [ecom, *[g for g in prefs if g != ecom]]
    metrics = list(getattr(intention, "metrics", None) or [])
    if "price" not in {str(m).casefold() for m in metrics}:
        intention.metrics = [*metrics, "price"]
    return intention


def _prefer_plmex_for_assortment_filters(
    platform: Platform,
    intention: Intention,
    *,
    question: str,
    state: dict,
) -> Intention:
    """Prefer plmex for brand/buyer + garment/color/gender multi-filters.

    Ecom often has a thin Sweaters category (e.g. Primark sweater=42) while PLM
    holds the assortment oracle (560). Skip when the user named ecom or the
    session is already pinned, or when the ask is availability/stock.
    """

    session_sticky = str(state.get("last_graph_id") or "").strip()
    if session_sticky:
        return intention
    low = f" {str(question or '').casefold()} "
    if any(
        tok in low
        for tok in (
            " ecom ",
            " ecomm ",
            " ecommerce ",
            " e-commerce ",
            " e-comm ",
            "in stock",
            "instock",
        )
    ):
        return intention
    # Only skip when the user explicitly named a catalog (plm/ecom), not when
    # preferred_graphs_from_question soft-picks ecom because Primark is a retail brand.
    if _explicit_graphs_in_question(question, platform):
        return intention
    filters = dict(getattr(intention, "filters", None) or {})
    has_party = bool(filters.get("brand") or filters.get("buyer"))
    has_assortment = any(
        filters.get(k)
        for k in (
            "garment",
            "garment_type",
            "category",
            "color",
            "gender",
            "material",
        )
    )
    if not (has_party and has_assortment):
        return intention
    prefs = [str(g) for g in (getattr(intention, "preferred_graphs", None) or []) if g]
    plm = "plmex_products"
    try:
        from knowledge_graph.retrieval.ecom import registry_product_graphs

        for cand in registry_product_graphs(platform) or []:
            if "plm" in str(cand).casefold():
                plm = str(cand)
                break
    except Exception:
        pass
    intention.preferred_graphs = [plm, *[g for g in prefs if g != plm]]
    return intention


def _remember_graph(state: dict, graph_ids: list[str] | None, graph_id: str | None = None) -> None:
    """Persist which product graph this turn used (for the next message in the same chat)."""

    for gid in list(graph_ids or []) + ([graph_id] if graph_id else []):
        if gid and gid not in {"shared", "master", "kg_system"}:
            state["last_graph_id"] = str(gid)
            return


def _explicit_graphs_in_question(question: str, platform: Platform) -> list[str]:
    from fashion_brain.ask import (
        preferred_graphs_from_question,
        question_names_catalog_explicitly,
    )

    named = list(preferred_graphs_from_question(question, platform) or [])
    return named if named and question_names_catalog_explicitly(question, platform) else []


def _apply_explicit_graph_preferences(
    intention: Intention,
    question: str,
    platform: Platform,
) -> list[str]:
    explicit = _explicit_graphs_in_question(question, platform)
    if explicit:
        intention.preferred_graphs = explicit
    return explicit


def _resolve_meta_count_graph(
    platform: Platform,
    *,
    question: str,
    validated,
    graph_id: str | None,
    state: dict,
    filters: dict | None = None,
) -> str | None:
    """Pick the product KG for a simple metadata count.

    Explicit catalog tokens (plm/ecom/…) win. Follow-ups with deixis or a sticky
    ``last_graph_id`` stay pinned — weak facet tokens like ``brand`` must not
    steal the session onto another KG.
    """

    from fashion_brain.ask import question_has_catalog_deixis

    named = _explicit_graphs_in_question(question, platform)
    source = str(getattr(validated, "source", None) or "").strip() if validated else ""
    sticky = str(state.get("last_graph_id") or "").strip()
    if named:
        return named[0]
    if source:
        return source
    # Buyer / PLM-only facets → prefer plmex when the session is not pinned.
    low_q = (question or "").casefold()
    filt = {str(k).casefold(): v for k, v in dict(filters or {}).items() if v not in (None, "")}
    needs_plm = bool(filt.get("buyer")) or bool(
        re.search(r"\bbuyers?\b", low_q)
    ) or bool(filt.get("color") or filt.get("gender"))
    if needs_plm and not sticky:
        try:
            from knowledge_graph.retrieval.ecom import registry_product_graphs

            for cand in registry_product_graphs(platform) or []:
                if "plm" in str(cand).casefold():
                    return str(cand)
        except Exception:
            pass
        return "plmex_products"
    # "top 10 brands of them" after a plmex turn → sticky plmex, not default ecom.
    if sticky and (
        question_has_catalog_deixis(question)
        or state.get("last_result_id")
        or not named
    ):
        return sticky
    if graph_id:
        return str(graph_id)
    return sticky or None


def _apply_request_graph_pin(
    selected_graphs: list[str],
    *,
    graph_id: str | None,
    intent: str,
    question: str,
    platform: Platform,
) -> list[str]:
    """Keep the request graph unless the user explicitly names another source."""

    if not graph_id:
        return selected_graphs
    if intent == "shared_products" and len(selected_graphs) > 1:
        return selected_graphs

    named = _explicit_graphs_in_question(question, platform)
    if named and (len(named) > 1 or graph_id not in named):
        return selected_graphs
    return [graph_id]


def _canonical_facet_display_labels(
    platform: Platform,
    graph_id: str,
    facet: str,
    values: list[str],
) -> dict[str, str]:
    """Recover display names: ontology for brand/buyer, polished labels for all facets."""

    # Start as identity — display_values / Neo4j restore happens in the caller.
    labels = {str(value): str(value) for value in values}
    entity_type = {"brand": "Brand", "buyer": "Buyer"}.get(str(facet).casefold())
    registry = getattr(platform, "kg_registry", None)
    if entity_type is None or registry is None:
        return labels
    from knowledge_graph.ontology import normalize_name
    from knowledge_graph.retrieval.ecom import _display_label

    graph = registry.get_graph(graph_id)
    ontology = registry.load_ontology_object(graph.ontology_id)
    wanted = {normalize_name(value) for value in values}
    candidates: dict[str, set[str]] = {}
    for entity in getattr(ontology, "canonical_entities", None) or []:
        if str(getattr(entity, "type", "") or "").casefold() != entity_type.casefold():
            continue
        name = str(getattr(entity, "name", "") or "").strip()
        surfaces = [name, *(getattr(entity, "aliases", None) or [])]
        for surface in surfaces:
            key = normalize_name(str(surface or ""))
            if key in wanted:
                candidates.setdefault(key, set()).add(name)
    for value in values:
        matches = candidates.get(normalize_name(str(value)), set())
        if len(matches) == 1:
            # Prefer ontology surface (C&A / H&M / PRIMARK) then light polish.
            labels[str(value)] = _display_label(next(iter(matches)))
    return labels


def _merge_query_filters(
    *,
    validated=None,
    summary=None,
    intention=None,
    state: dict | None = None,
    include_session: bool = False,
) -> dict:
    """Merge filters from upstream agents.

    Session ``active_filters`` are included only when ``include_session`` is true
    (contextual carry). Blindly merging session turned catalog-total asks into
    the prior brand (e.g. Primark after an e-comm total question).
    """

    merged: dict = {}
    sources = []
    if include_session:
        sources.append(dict((state or {}).get("active_filters") or {}))
    sources.extend(
        (
            dict(getattr(summary, "filters", None) or {}),
            dict(getattr(intention, "filters", None) or {}),
            dict(getattr(validated, "filters", None) or {}),
        )
    )
    for source in sources:
        for key, value in source.items():
            if value in (None, ""):
                continue
            merged[str(key)] = value
    return merged


def _attach_selected_meta_reference(
    retrieved: Retrieved,
    *,
    graph_id: str | None,
    question: str = "",
    intention: Intention | None = None,
    validated: ValidatedIntent | None = None,
    platform: Platform | None = None,
) -> Retrieved:
    """Attach selected kg_stats filter shards as optional_metadata for ResponseAgent.

    Findings stay Neo/Cypher (or basic meta count). The LLM uses both: answer facts
    from findings, grounding colour from the relevant filter meta only — never a
    full-KG dump.
    """

    if retrieved is None or not graph_id:
        return retrieved
    evidence = list(getattr(retrieved, "evidence", None) or [])
    if any(str(getattr(item, "record_id", "") or "") == "optional_metadata" for item in evidence):
        return retrieved
    try:
        from knowledge_graph.models import EvidenceItem
        from knowledge_graph.retrieval.kg_stats import build_meta_reference_lines, stats_available
    except Exception:
        return retrieved

    generated = getattr(getattr(platform, "settings", None), "generated", None) if platform else None
    if not stats_available(str(graph_id), generated):
        return retrieved

    filters = dict(getattr(intention, "filters", None) or {})
    if validated is not None and getattr(validated, "filters", None):
        for key, value in dict(validated.filters or {}).items():
            filters.setdefault(key, value)
    count_entity = (
        getattr(intention, "count_entity", None)
        if intention is not None
        else None
    ) or (getattr(validated, "entity", None) if validated is not None else None)
    group_by = (
        getattr(intention, "group_by", None)
        if intention is not None
        else None
    ) or (getattr(validated, "group_by", None) if validated is not None else None)

    # Simple product counts: skip competing facet tops (Black=5433 beside white=2292).
    # Analysis / breakdown / ranking still get top_n for merchandising context.
    is_analysis = bool(getattr(intention, "analysis", False)) if intention is not None else False
    wants_breakdown = bool(group_by) or is_analysis
    if not wants_breakdown and (
        str(getattr(intention, "intent", "") or "").casefold()
        in {"product_count", "count", "stats"}
        or (filters and not group_by)
    ):
        # Still allow products_total only — no unique/top facet dump.
        try:
            from knowledge_graph.retrieval.kg_stats import product_total

            total = product_total(str(graph_id), generated)
            lines = [f"products_total={int(total)}"] if total is not None else []
        except Exception:
            lines = []
        if not lines:
            return retrieved
    else:
        lines = build_meta_reference_lines(
            str(graph_id),
            filters=filters,
            count_entity=str(count_entity) if count_entity else None,
            group_by=str(group_by) if group_by else None,
            question=question or "",
            generated=generated,
            max_filters=5,
            top_n=5,
        )
        if not lines:
            return retrieved

    # Prefer filter shards that were selected for this ask as source_path hint.
    filter_hint = "filters"
    if count_entity:
        filter_hint = str(count_entity).casefold().replace(" ", "_")
    elif filters:
        filter_hint = "+".join(sorted(str(k).casefold() for k in filters)[:3])

    evidence.append(
        EvidenceItem(
            source_id=str(graph_id),
            record_id="optional_metadata",
            title="kg filter meta reference",
            excerpt="; ".join(lines),
            source_path=f"kg_stats/{graph_id}/{filter_hint}",
            verification_status="observed",
            graph=str(graph_id),
            node_id=None,
            synthetic=False,
            requested_filters=dict(filters or {}),
            applied_filters=dict(filters or {}),
        )
    )
    retrieved.evidence = evidence
    return retrieved


def _reject_unfiltered_global_hit(requested_filters: dict, evidence: list) -> bool:
    """True when evidence answered a filtered ask with an unfiltered global stat."""

    if not requested_filters:
        return False
    try:
        from knowledge_graph.retrieval.kg_stats import filters_match, normalize_filter_dict
    except Exception:
        return False
    requested = normalize_filter_dict(requested_filters)
    if not requested:
        return False
    for item in evidence or []:
        applied = normalize_filter_dict(getattr(item, "applied_filters", None) or {})
        if filters_match(requested, applied):
            return False
        excerpt = str(getattr(item, "excerpt", "") or "")
        # Legacy global-total evidence shape.
        if getattr(item, "record_id", None) == "kg_stats" and excerpt.startswith("products=") and not applied:
            return True
    # No evidence applied the requested filters.
    return True


def _try_kg_stats_meta_answer(
    platform: Platform,
    *,
    question: str,
    validated,
    graph_id: str | None,
    state: dict,
    summary=None,
    intention=None,
) -> tuple[Retrieved, Intention, str] | None:
    """Answer *basic* 0–1 filter product/facet counts from kg_stats only.

    After QU, only simple quantity asks are answered from selected filter shards.
    Analysis / rankings / multi-filter / price / listings return None so the
    pipeline builds Neo4j Cypher, retrieves rows, then attaches the same selected
    filter meta as optional_metadata for the response LLM (findings + meta).

    Filter contract: merge QU + ThinkingBrain + Intention + session filters. Multi-filter
    or uncached dimension → None (``cache_miss``) so the caller falls through to Text2Cypher.
    Never returns a global total when any requested filter is present.
    """

    if validated is None and intention is None and summary is None:
        return None
    metrics = {str(m).casefold() for m in (getattr(validated, "metrics", None) or []) if m}
    if intention is not None:
        metrics |= {str(m).casefold() for m in (getattr(intention, "metrics", None) or []) if m}
    # Never collapse a price-change / trend / ranked product list into a product count.
    try:
        from fashion_brain.ask import (
            _is_price_history_question,
            is_price_trend_analysis_ask,
            is_ranked_product_listing_ask,
        )
        from knowledge_graph.retrieval.ecom import price_history_mode

        if (
            "price" in metrics
            or _is_price_history_question(question or "")
            or is_price_trend_analysis_ask(question or "")
            or is_ranked_product_listing_ask(question or "")
            or price_history_mode(question or "")
        ):
            return None
    except Exception:
        pass
    # Explicit top-N / limit asks are listings, not kg_stats totals.
    if getattr(intention, "limit", None) not in (None, 0) and not (
        "how many" in (question or "").casefold() or "count" in metrics
    ):
        try:
            from fashion_brain.ask import is_product_count_ask

            if not is_product_count_ask(question or ""):
                return None
        except Exception:
            return None
    # QU marks quantity asks with metrics=["count"]. Also accept product_count intention.
    intent_name = str(getattr(intention, "intent", "") or getattr(validated, "objective", "") or "").casefold()
    if (
        "count" not in metrics
        and "facet_count" not in metrics
        and intent_name not in {"product_count", "count"}
        and "how many" not in (question or "").casefold()
    ):
        return None
    depth = str(getattr(validated, "analysis_depth", "") or "unset").casefold() if validated else "unset"
    if depth == "full" or bool(getattr(intention, "analysis", False)):
        return None
    if list(getattr(validated, "comparison", None) or []):
        return None

    # Resolve distinct-facet filter key up front (kg → filter → unique count).
    # Clear mistaken group_by so "how many buyers" is not treated as a breakdown.
    inferred_facet: str | None = None
    try:
        from fashion_brain.ask import _facet_entity_count

        inferred_facet = _facet_entity_count(question or "")
        if inferred_facet and str(inferred_facet).casefold() not in {
            "product",
            "products",
            "item",
            "items",
        }:
            if intention is not None and getattr(intention, "group_by", None):
                intention.group_by = None
            if validated is not None and getattr(validated, "group_by", None):
                validated.group_by = None
            if intention is not None and not getattr(intention, "count_entity", None):
                intention.count_entity = str(inferred_facet)
    except Exception:
        inferred_facet = None

    # Breakdowns / focused group_by stay on the template or Cypher path.
    if getattr(validated, "group_by", None) or getattr(intention, "group_by", None):
        return None
    try:
        from fashion_brain.ask import is_open_product_listing

        if is_open_product_listing(question):
            return None
    except Exception:
        pass

    try:
        from knowledge_graph.retrieval.ecom import append_neo4j_count_answer, clean_product_filters
        from knowledge_graph.retrieval.kg_stats import (
            answer_simple_count,
            normalize_filter_dict,
            resolve_entity_to_filter,
            stats_available,
        )
    except Exception:
        return None

    filters = clean_product_filters(
        _merge_query_filters(validated=validated, summary=summary, intention=intention, state=state)
    )
    # Dynamic live-brand / facet grounding — never answer a named-party ask as global.
    try:
        from fashion_brain.ask import (
            enrich_ecom_question_filters,
            enrich_scope_from_question,
            live_brand_names,
        )

        grounded, _gb = enrich_scope_from_question(
            question,
            filters,
            brand_names=live_brand_names(platform, graph_id=graph_id),
            platform=platform,
        )
        filters = clean_product_filters(enrich_ecom_question_filters(question, grounded))
        if validated is not None and filters and not getattr(validated, "filters", None):
            validated.filters = dict(filters)
        # Do NOT mutate intention.filters here on cache-miss — that polluted
        # brand+garment asks with schema-prose category/family and forced Text2Cypher.
    except Exception:
        pass

    generated = getattr(getattr(platform, "settings", None), "generated", None)
    # Multi-attribute combination → metadata cache-miss → template/Cypher path.
    # Gate on pre-resolve filters so brand+sweater never becomes a false single-facet hit.
    pre_resolve = normalize_filter_dict(filters or {})
    if len(pre_resolve) >= 2:
        return None
    # Compare parties with a garment/facet also need Neo4j, not a single meta shard.
    if "|" in str((filters or {}).get("brand") or "") and any(
        (filters or {}).get(k)
        for k in ("garment", "garment_type", "category", "color", "gender", "material")
    ):
        return None

    gid = _resolve_meta_count_graph(
        platform,
        question=question,
        validated=validated,
        graph_id=graph_id,
        state=state,
        filters=filters,
    )
    if not gid and intention is not None:
        prefs = list(getattr(intention, "preferred_graphs", None) or [])
        gid = prefs[0] if prefs else None

    # Map garment=sweater → category=Sweaters/sweater via kg_stats (case/plural soft match).
    # Drop junk (buyer=dataset). Never answer a filtered ask as the global catalog total.
    # When several KGs match the same user token, prefer the strongest live meta hit
    # (plmex category=sweater 17287 beats ecom category=Sweaters 1).
    garment_keys = {
        "category",
        "garment",
        "garment_type",
        "family",
        "subfamily",
        "type",
        "product_type",
    }
    single_garment = (
        len(pre_resolve) == 1
        and next(iter(pre_resolve)).casefold().replace(" ", "_") in garment_keys
    )
    try:
        from knowledge_graph.retrieval.ecom import registry_product_graphs
        from knowledge_graph.retrieval.kg_stats import (
            _value_match_score,
            find_value_in_stats,
            has_dimension,
            resolve_filters_against_stats,
        )

        if single_garment:
            user_val = next(iter(pre_resolve.values()))
            best: tuple[float, int, str, dict] | None = None
            candidates = list(registry_product_graphs(platform) or [])
            # Prefer live registry, but always consider the sticky / requested graph_id.
            if gid and str(gid) not in candidates:
                candidates = [str(gid), *candidates]
            if not candidates and gid:
                candidates = [str(gid)]
            for cand in candidates:
                if not stats_available(str(cand), generated):
                    continue
                hit = find_value_in_stats(str(cand), str(user_val), generated=generated)
                if hit is None:
                    continue
                attr, canon, n = hit
                score = _value_match_score(str(user_val), canon)
                if best is None or (score, n) > (best[0], best[1]):
                    best = (score, int(n), str(cand), {attr: canon})
            if best is None:
                return None
            _, _, gid, filters = best
        else:
            # Prefer a graph that actually indexes the requested explicit facets.
            if filters and (not gid or any(
                k in {"color", "gender", "buyer", "material"}
                and not has_dimension(str(gid), k, generated)
                for k in filters
            )):
                best_gid = None
                best_n = -1
                for cand in registry_product_graphs(platform) or []:
                    if not stats_available(str(cand), generated):
                        continue
                    resolved = resolve_filters_against_stats(
                        str(cand), filters, generated=generated
                    )
                    if not resolved or len(resolved) != len(pre_resolve):
                        continue
                    # Prefer graphs that keep the same keys (no silent rebind) and
                    # have the dims indexed.
                    if any(
                        not has_dimension(str(cand), k, generated)
                        for k in resolved
                        if k in {"color", "gender", "buyer", "material", "brand"}
                    ):
                        # buyer/color missing on ecom — skip when those keys requested
                        if any(
                            k in pre_resolve and k not in resolved
                            for k in ("color", "gender", "buyer")
                        ):
                            continue
                    score = sum(
                        1 for k in resolved if has_dimension(str(cand), k, generated)
                    )
                    if score > best_n:
                        best_n = score
                        best_gid = str(cand)
                if best_gid:
                    gid = best_gid
            if not gid:
                return None
            if not stats_available(str(gid), generated):
                return None
            filters = resolve_filters_against_stats(str(gid), filters, generated=generated)
            if pre_resolve and not filters:
                return None
    except Exception:
        if not gid:
            return None
        try:
            from knowledge_graph.retrieval.kg_stats import resolve_filters_against_stats

            filters = resolve_filters_against_stats(str(gid), filters, generated=generated)
        except Exception:
            pass
        if pre_resolve and not filters:
            return None
    if not gid:
        return None
    if not stats_available(str(gid), generated):
        return None
    # Multi-attribute combination after resolve → still a cache-miss.
    if len(filters) >= 2:
        return None

    entity = str(
        getattr(validated, "entity", None)
        or getattr(intention, "count_entity", None)
        or inferred_facet
        or ""
    ).strip()
    count_entity = None
    ent_key = entity.casefold().replace(" ", "_")
    # Distinct-facet asks → direct kg/filter unique count (no full-meta scan).
    if ent_key and ent_key not in {"product", "products", "item", "items", ""}:
        if not filters:
            # Prefer resolving a bare entity token to a single products_by_* filter
            # (generic across brand/buyer/category/season/…). Only treat as distinct-facet
            # count when it cannot be resolved to a value.
            resolved = resolve_entity_to_filter(str(gid), entity or ent_key, generated)
            if resolved:
                filters = clean_product_filters(resolved)
            else:
                count_entity = ent_key

    # Probe kg_stats first — cache_miss must not call append (which could Cypher or globalize).
    # Direct path: graph_id → filter (count_entity) → unique value count.
    from fashion_brain.stage_timing import timed

    with timed("metadata_count_lookup", graph_id=str(gid), filter_count=len(filters)):
        meta = answer_simple_count(
            str(gid),
            filters=filters,
            count_entity=count_entity,
            generated=generated,
        )
    if meta is None:
        return None

    requested = normalize_filter_dict(meta.get("requested_filters") if meta.get("requested_filters") is not None else filters)
    applied = normalize_filter_dict(meta.get("applied_filters") or {})
    if requested and not applied:
        # Global total must never answer a filtered ask.
        return None

    lines: list[str] = []
    evidence: list = []
    caveats: list[str] = []
    # Reuse the same formatter as the retrieve path (meta-first, no Cypher for 0–1).
    # Optional facet meta is attached below via _attach_selected_meta_reference so
    # filtered category/garment counts also get related key=value context (not only brand).
    append_neo4j_count_answer(
        platform,
        Principal(subject="meta", clearance="internal"),
        str(gid),
        filters,
        None,
        None,
        lines,
        evidence,
        caveats,
        question=question,
        count_entity=count_entity,
        meta_result=meta,
        include_optional_metadata=False,
    )
    if not lines or not any(getattr(e, "record_id", None) == "kg_stats" for e in evidence):
        return None
    if _reject_unfiltered_global_hit(requested or filters, evidence):
        return None

    intention_out = Intention(
        intent="product_count",
        filters=filters,
        thinking="Answered from kg_stats metadata (skipped ThinkingBrain / Text2Cypher)",
        preferred_graphs=[str(gid)],
        count_entity=count_entity,
        retrieve_mode="template",
        metrics=list(getattr(validated, "metrics", None) or getattr(intention, "metrics", None) or [])
        or ["count"],
    )
    retrieved = Retrieved(
        text=lines[0],
        evidence=list(evidence),
        caveats=list(caveats),
        unavailable=[],
        retrieve_mode="template",
        response_type="stats" if count_entity or filters else "concise",
    )
    # Enrich thin count answers with selected filter shards (products_total, unique, top=).
    retrieved = _attach_selected_meta_reference(
        retrieved,
        graph_id=str(gid),
        question=question or "",
        intention=intention_out,
        validated=validated,
        platform=platform,
    )
    return retrieved, intention_out, str(gid)


def _normalize_cross_catalog_type_intention(
    intention: Intention,
    question: str,
    product_ids: list[str],
) -> None:
    """Undo agent remaps: brand=thinking, subfamily=sweater, group_by=subfamily.

    Mentions of garment_type/subfamily in the brief are *role guidance*, not
    exact filter values or a breakdown dimension.
    """

    if len(product_ids) < 2:
        return
    text = (question or "").casefold()
    if not re.search(r"\b(compare|versus|vs\.?|across|between)\b", text):
        return
    token = _cross_catalog_type_token(dict(intention.filters or {}), question)
    if not token:
        return
    from knowledge_graph.ontology import normalize_name

    want = normalize_name(token)
    filters = dict(intention.filters or {})
    for key in ("brand", "buyer"):
        if _is_grammar_party_safe(filters.get(key)):
            filters.pop(key, None)
    # Role facets must not hold the type token as an exact value.
    for key in ("subfamily", "family", "garment_type", "category", "plm_sub_category", "plm_sub_sub_category"):
        val = filters.get(key)
        if val in (None, ""):
            continue
        if normalize_name(str(val)) == want or want in normalize_name(str(val)).split():
            filters.pop(key, None)
    filters["garment"] = token
    intention.filters = filters
    # Separate counts per source — not a single-graph subfamily breakdown.
    if str(intention.group_by or "").casefold() in {
        "subfamily",
        "family",
        "garment_type",
        "category",
    }:
        intention.group_by = None
    metrics = {str(m).casefold() for m in (intention.metrics or []) if m}
    if "product_count" not in metrics and "count" not in metrics:
        intention.metrics = [*(intention.metrics or []), "product_count"]
    if intention.intent in {"analysis", "products", "breakdown", "other", ""}:
        intention.intent = "shared_products"
    prefs = list(intention.preferred_graphs or [])
    for gid in product_ids:
        if gid not in prefs:
            prefs.append(gid)
    intention.preferred_graphs = prefs


def _is_grammar_party_safe(value) -> bool:
    if value in (None, ""):
        return False
    try:
        from fashion_brain.ask import _is_grammar_party_name

        return _is_grammar_party_name(str(value))
    except Exception:
        low = str(value).casefold().strip()
        return low in {"thinking", "providing", "hidden", "analyze", "analysis"}


def _cross_catalog_type_token(filters: dict | None, question: str) -> str | None:
    """Garment/type token for cross-catalog compare (e.g. sweater)."""

    for key in ("garment", "garment_type", "category", "subfamily", "family"):
        raw = (filters or {}).get(key)
        if raw in (None, "") or "|" in str(raw):
            continue
        token = str(raw).strip()
        if token and 2 < len(token) <= 40:
            return token
    text = (question or "").casefold()
    m = re.search(
        r"\b(sweaters?|knitwear|cardigans?|jumpers?|hoodies?|dresses?|jeans|t[\s-]?shirts?)\b",
        text,
    )
    return m.group(1) if m else None


def _type_token_stems(token: str) -> list[str]:
    base = re.sub(r"[^a-z0-9]+", "", (token or "").casefold())
    if not base:
        return []
    stems = {base, base.rstrip("s") if len(base) > 4 else base}
    # Knit-top family for sweater / jumper / cardigan compares.
    if "sweat" in base or base in {"jumper", "jumpers", "cardigan", "cardigans", "knitwear"}:
        stems.update(
            {
                "sweat",
                "sweater",
                "jumper",
                "cardigan",
                "knitwear",
                "knit",
                "pullover",
            }
        )
    return [s for s in stems if len(s) >= 4]


def _type_label_closeness(token: str, label: str) -> float:
    """Score how close a facet label is to the asked type (prefix/suffix/compound).

    Higher = closer. Exact/plural ≈ 1.0; ``cardigan sweater`` / ``cardigans sweaters``
    for sweater ≈ 0.9+; retail synonyms (jumper/cardigan) mid; loose sweatshirt/knit lower.
    """

    from knowledge_graph.ontology import normalize_name

    want = normalize_name(token)
    lab = normalize_name(label)
    if not want or not lab:
        return 0.0
    want_stem = want.rstrip("s") if len(want) > 4 else want
    lab_stem = lab.rstrip("s") if len(lab) > 4 else lab
    if lab == want or lab_stem == want_stem:
        return 1.0
    parts = lab.split()
    part_stems = {(p.rstrip("s") if len(p) > 4 else p) for p in parts}
    compact_lab = lab.replace(" ", "")
    # Literal token as prefix/suffix/infix in the label (cardigans sweaters, sweater vest).
    if want_stem in part_stems or want in parts:
        if parts[-1].rstrip("s") == want_stem or parts[0].rstrip("s") == want_stem:
            return 0.94
        return 0.9
    if compact_lab.startswith(want_stem) or compact_lab.endswith(want_stem):
        return 0.88
    if len(want_stem) >= 5 and want_stem in compact_lab:
        # Avoid sweatshirt≈sweater false friend unless "sweater" itself appears.
        if want_stem == "sweater" and "sweater" not in lab and "sweaters" not in lab:
            pass  # fall through to synonym scoring
        else:
            return 0.82
    # Broader same-meaning retail synonyms (jumper/cardigan for sweater asks).
    synonym_stems = {
        s
        for s in _type_token_stems(token)
        if s not in {want_stem, "sweat", "knit"} and s != want
    }
    for stem in synonym_stems:
        if stem in part_stems or any(p.startswith(stem) or p.endswith(stem) for p in parts):
            return 0.7
        if stem in compact_lab:
            return 0.65
    # Loose family only (knitwear / sweatshirt) — last resort when exact is sparse.
    for stem in ("knitwear", "knit", "sweatshirt", "sweat"):
        if stem in lab or stem in compact_lab:
            return 0.55
    return 0.0


def _try_cross_catalog_type_compare(
    platform: Platform,
    *,
    question: str,
    intention: Intention,
    product_ids: list[str],
) -> Retrieved | None:
    """Separate per-source type counts from kg_stats — skip multi-graph Text2Cypher.

    A17-style asks: compare sweater / sweater-like across PLMEX vs ecom, warn that
    category is a false friend, and report garment_type / subfamily roles separately.
    When an exact primary is missing or sparse, surface closest prefix/suffix compounds
    (e.g. cardigan sweater) with a short clarifying sentence.
    """

    if len(product_ids) < 2:
        return None
    text = (question or "").casefold()
    if not re.search(r"\b(compare|versus|vs\.?|across|between)\b", text):
        return None
    metrics = {str(m).casefold() for m in (getattr(intention, "metrics", None) or []) if m}
    intent = str(getattr(intention, "intent", "") or "").casefold()
    wants_counts = bool(
        metrics & {"product_count", "count", "facet_count"}
        or intent in {"shared_products", "shared", "product_count", "count"}
        or re.search(r"\b(counts?|how many|separate)\b", text)
    )
    if not wants_counts:
        return None
    token = _cross_catalog_type_token(dict(intention.filters or {}), question)
    if not token:
        return None

    try:
        from knowledge_graph.models import EvidenceItem
        from knowledge_graph.retrieval.kg_stats import (
            answer_simple_count,
            dimension_bucket,
            resolve_filters_against_stats,
            stats_available,
        )
    except Exception:
        return None

    generated = getattr(getattr(platform, "settings", None), "generated", None)
    # Role facets per source — never merge category values across graphs.
    role_by_graph = {
        "plmex": ("category", "garment_type", "plm_sub_sub_category", "plm_sub_category"),
        "ecom": ("category", "subfamily", "family"),
    }

    lines: list[str] = [
        f"## Cross-catalog compare: `{token}`",
        "",
        "**Category is a false friend** across PLMEX and e-commerce — the same "
        "string does not mean the same thing. Counts below stay **per source**, "
        "using each graph's type roles (`garment_type` / `plm_sub_*` on PLMEX, "
        "`subfamily` / `family` on e-commerce) rather than merging category labels.",
        "",
    ]
    evidence: list[EvidenceItem] = []
    any_hit = False

    for gid in product_ids:
        if not stats_available(gid, generated):
            lines.append(f"### {gid}")
            lines.append("No kg_stats shard available for this source.")
            lines.append("")
            continue
        gkey = "plmex" if "plm" in gid.casefold() else "ecom"
        roles = role_by_graph.get(gkey, ("category", "subfamily", "garment_type"))
        resolved = resolve_filters_against_stats(
            gid, {"garment": token}, generated=generated
        ) or {}
        primary = (
            answer_simple_count(gid, filters=resolved, generated=generated)
            if resolved
            else None
        )
        primary_n = int(primary["n"]) if primary and primary.get("n") is not None else 0
        primary_attr = (
            str(primary.get("attribute") or next(iter(resolved), "type"))
            if primary
            else ""
        )
        primary_val = (
            str(primary.get("value") or next(iter(resolved.values()), token))
            if primary
            else ""
        )

        # Closest facet labels (prefix/suffix/compound), ranked by closeness then count.
        neighbors: list[tuple[float, int, str, str]] = []
        for attr in roles:
            try:
                values = dimension_bucket(gid, attr, generated=generated) or {}
            except Exception:
                values = {}
            if not isinstance(values, dict):
                continue
            for label, n in values.items():
                try:
                    count = int(n)
                except Exception:
                    continue
                if count <= 0:
                    continue
                score = _type_label_closeness(token, str(label))
                if score < 0.55:
                    continue
                # Skip the exact primary row.
                if (
                    primary
                    and primary_attr.casefold() == attr.casefold()
                    and primary_val.casefold() == str(label).casefold()
                ):
                    continue
                if score >= 0.99:
                    continue
                neighbors.append((score, count, attr, str(label)))
        neighbors.sort(key=lambda item: (-item[0], -item[1], item[2], item[3].casefold()))
        closest = neighbors[:8]
        compound = [row for row in closest if row[0] >= 0.85]
        related_big = [row for row in closest if row[1] >= max(10, primary_n * 5)]
        # Sparse exact: missing, tiny absolute, or clearly outnumbered by
        # closer compounds / same-family labels (ecom sweaters=1 vs jumpers/cardigans).
        sparse_primary = primary_n <= 0 or (
            primary_n <= 5 and bool(closest) and (bool(compound) or bool(related_big))
        )

        lines.append(f"### {gid}")
        if primary_n > 0:
            any_hit = True
            lines.append(
                f"- Exact/primary match on **{primary_attr}** = `{primary_val}` → "
                f"**{primary_n:,}** products"
            )
            evidence.append(
                EvidenceItem(
                    source_id=gid,
                    record_id=f"kg_stats:{primary_attr}",
                    title=f"{gid} {primary_attr}={primary_val}",
                    excerpt=f"n={primary_n}",
                    source_path="kg_stats",
                    verification_status="observed",
                    graph=gid,
                    node_id=None,
                    synthetic=False,
                )
            )
        else:
            lines.append(
                f"- No exact `{token}` value on this source's indexed type facets."
            )

        if closest:
            any_hit = True
            # Split: labels that literally carry the token vs broader synonyms.
            literal = [row for row in closest if row[0] >= 0.82]
            synonyms = [row for row in closest if 0.55 <= row[0] < 0.82]
            if sparse_primary:
                lit_bits: list[str] = []
                seen: set[str] = set()
                for _s, n, _a, lab in literal:
                    key = lab.casefold()
                    if key in seen:
                        continue
                    seen.add(key)
                    lit_bits.append(f"`{lab}` ({n:,})")
                    if len(lit_bits) >= 4:
                        break
                syn_bits: list[str] = []
                for _s, n, _a, lab in synonyms:
                    key = lab.casefold()
                    if key in seen:
                        continue
                    seen.add(key)
                    syn_bits.append(f"`{lab}` ({n:,})")
                    if len(syn_bits) >= 4:
                        break
                if primary_n <= 0:
                    lines.append(
                        f"- **Note:** Exact `{token}` was not found on this source."
                    )
                else:
                    lines.append(
                        f"- **Note:** Exact `{token}` is almost unused here "
                        f"(**{primary_n:,}** on `{primary_val}`). Prefer prefix/suffix "
                        f"compounds and same-meaning type roles below — do **not** treat "
                        f"{primary_n:,} as the full sweater-like assortment."
                    )
                if lit_bits:
                    lines.append(
                        f"- Closest labels that include `{token}` "
                        f"(prefix/suffix/compound): {', '.join(lit_bits)}."
                    )
                if syn_bits:
                    lines.append(
                        f"- Broader sweater-like type roles on this source "
                        f"(jumper/cardigan/knitwear — not merged with PLM): "
                        f"{', '.join(syn_bits)}."
                    )
            else:
                lines.append(
                    f"- Related **{gkey}** type-role values (not merged with the other source):"
                )
            # Literal compounds first, then synonyms — always show the ranked rows.
            ordered = literal + synonyms
            shown: set[tuple[str, str]] = set()
            for score, count, attr, label in ordered[:10]:
                key = (attr.casefold(), label.casefold())
                if key in shown:
                    continue
                shown.add(key)
                if score >= 0.82:
                    tag = "contains-token"
                elif score >= 0.65:
                    tag = "synonym"
                else:
                    tag = "related"
                lines.append(f"  - ({tag}) `{attr}` = `{label}` → **{count:,}**")
        lines.append("")

    if not any_hit:
        return None

    lines.append(
        "_Totals are not added across sources — category/subfamily vocabularies differ._"
    )
    return Retrieved(
        text="\n".join(lines),
        evidence=evidence,
        caveats=[
            "Answered from kg_stats per source (skipped multi-graph Text2Cypher).",
            "Category values were not merged across graphs.",
            "Sparse exact matches promote closest prefix/suffix compounds.",
        ],
        unavailable=[],
        synthetic_data=False,
        response_type="comparison",
        retrieve_mode="kg_stats",
    )


def _try_kg_stats_facet_ranking(
    platform: Platform,
    *,
    question: str,
    validated=None,
    graph_id: str | None,
    state: dict,
    intention=None,
) -> tuple[Retrieved, Intention, str] | None:
    """Answer 'top N brand/buyer names' from the selected filter shard (sticky KG).

    Follow-ups like 'list top 10 brand name of them' after a plmex brand count must
    stay on ``last_graph_id`` and never fall through to ecom product price listings.
    """

    try:
        from fashion_brain.ask import is_facet_value_ranking_ask, is_ranked_product_listing_ask
        from knowledge_graph.models import EvidenceItem
        from knowledge_graph.retrieval.kg_stats import (
            dimension_bucket,
            format_filter_reference,
            stats_available,
        )
    except Exception:
        return None
    if is_ranked_product_listing_ask(question or ""):
        return None
    session_sticky = str(state.get("last_graph_id") or "").strip() or None
    sticky = session_sticky or (str(graph_id or "").strip() or None)
    parsed = is_facet_value_ranking_ask(question or "", sticky)
    if parsed is None and intention is not None:
        # Intention may already hold group_by=brand + limit=10 from keyword plan.
        gb = str(getattr(intention, "group_by", None) or "").casefold()
        lim = getattr(intention, "limit", None)
        if gb and lim and not (getattr(intention, "filters", None) or {}):
            parsed = (gb, int(lim))
    if parsed is None:
        return None
    facet, limit = parsed
    gid = _resolve_meta_count_graph(
        platform,
        question=question,
        validated=validated,
        graph_id=graph_id,
        state=state,
    )
    if not gid and intention is not None:
        prefs = list(getattr(intention, "preferred_graphs", None) or [])
        gid = prefs[0] if prefs else sticky
    if not gid:
        gid = sticky
    # Unscoped "top N brands" defaults to plmex when the session is not pinned.
    # Do not treat the request default graph_id as sticky session memory.
    if not session_sticky and (
        not gid
        or (
            str(facet).casefold() == "brand"
            and "plm" not in str(gid).casefold()
        )
    ):
        try:
            from knowledge_graph.retrieval.ecom import registry_product_graphs
            from knowledge_graph.retrieval.kg_stats import dimension_bucket as _dim_bucket

            generated_probe = getattr(getattr(platform, "settings", None), "generated", None)
            for cand in registry_product_graphs(platform) or []:
                if "plm" not in str(cand).casefold():
                    continue
                if _dim_bucket(str(cand), facet, generated_probe):
                    gid = str(cand)
                    break
        except Exception:
            if not gid:
                gid = "plmex_products"
    if not gid:
        return None
    generated = getattr(getattr(platform, "settings", None), "generated", None)
    if not stats_available(str(gid), generated):
        return None
    from fashion_brain.stage_timing import timed

    with timed("metadata_facet_lookup", graph_id=str(gid), facet=facet):
        bucket = dimension_bucket(str(gid), facet, generated)
    if bucket is None:
        return None
    from knowledge_graph.retrieval.kg_stats import _EMPTY_KEYS

    ranked = sorted(
        (
            (str(name), int(count))
            for name, count in bucket.items()
            if str(name).strip().casefold() not in _EMPTY_KEYS
        ),
        key=lambda item: (-item[1], item[0].casefold()),
    )[: max(int(limit), 1)]
    if not ranked:
        return None
    display_labels = _canonical_facet_display_labels(
        platform,
        str(gid),
        facet,
        [name for name, _count in ranked],
    )
    try:
        from knowledge_graph.retrieval.ecom import _display_label
        from knowledge_graph.retrieval.kg_stats import resolve_facet_display
    except Exception:  # pragma: no cover
        _display_label = lambda x, **_kw: str(x)

        def resolve_facet_display(*_a, **_k):  # type: ignore
            return None

    def _shown(raw: str) -> str:
        # 1) Ontology brand/buyer surfaces  2) shard display_values  3) light polish
        preferred = display_labels.get(raw, raw)
        if preferred != raw:
            return preferred
        mapped = resolve_facet_display(
            raw,
            graph_id=str(gid),
            attribute=str(facet),
            generated=generated,
        )
        if mapped:
            return mapped
        return _display_label(
            raw,
            graph_id=str(gid),
            attribute=str(facet),
            generated=generated,
            platform=platform,
        )

    label = str(facet).replace("_", " ")
    lines = [
        f"Top {len(ranked)} {label}s by product count:",
        "| Rank | Name | Products |",
        "| ---: | --- | ---: |",
    ]
    for idx, (name, count) in enumerate(ranked, start=1):
        lines.append(f"| {idx} | {_shown(name)} | {count} |")
    # Prefer polished / ontology surfaces in evidence (not scrubbed kg_stats keys).
    top_bits = ", ".join(f"{_shown(name)}={count}" for name, count in ranked)
    reference = format_filter_reference(str(gid), facet, generated=generated, top_n=limit)
    if reference and "top:" in reference:
        head, _, _tail = reference.partition("top:")
        reference = f"{head}top: {top_bits}"
    elif not reference:
        reference = f"{facet}: top: {top_bits}"
    evidence = [
        EvidenceItem(
            source_id=str(gid),
            record_id="kg_stats",
            title=f"{gid} top {label}s",
            excerpt=reference or f"{facet} top {limit}",
            source_path=f"kg_stats/{gid}/{facet}.json",
            verification_status="observed",
            graph=str(gid),
            node_id=None,
            synthetic=False,
            requested_filters={},
            applied_filters={},
        )
    ]
    intention_out = Intention(
        intent="breakdown",
        filters={},
        group_by=facet,
        limit=int(limit),
        thinking="Answered top facet values from kg_stats filter shard (skipped Text2Cypher)",
        preferred_graphs=[str(gid)],
        retrieve_mode="template",
        metrics=["list", "facet_rank"],
        response_type="stats",
    )
    retrieved = Retrieved(
        text="\n".join(lines),
        evidence=evidence,
        caveats=[],
        unavailable=[],
        retrieve_mode="template",
        response_type="stats",
    )
    return retrieved, intention_out, str(gid)


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

    # No catalog named → keep both product graphs in play (search either; miss → not found).
    if not preferred:
        try:
            from knowledge_graph.retrieval.ecom import registry_product_graphs

            preferred = [str(g) for g in (registry_product_graphs(platform) or []) if g]
        except Exception:
            preferred = []
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
    # Example/listing asks ("show about 10 Matalan products") must not stop at a bare
    # count — fall through to the open-listing template that includes sample rows.
    from fashion_brain.ask import is_open_product_listing as _is_open_listing

    wants_examples = _is_open_listing(ask_text, limit=getattr(intention, "limit", None)) or bool(
        intention.limit
        and re.search(r"\b(example|examples|sample|samples|show|list)\b", ask_text.casefold())
    )
    if (
        intention.intent in {"product_count", "count", "breakdown"}
        and not bool(getattr(intention, "analysis", False))
        and not bool(getattr(intention, "suggest", False))
        and not cross_compare
        and intention.intent != "shared_products"
        and not wants_examples
    ):
        # Strip schema-prose junk + keep party + real facets only (frontend speed).
        try:
            from fashion_brain.ask import _is_schema_prose_filter
            from fashion_brain.planning.semantic_field_map import apply_semantic_field_map
            from knowledge_graph.ontology import normalize_name
            from knowledge_graph.retrieval.ecom import clean_product_filters

            cleaned, mapped_gb = apply_semantic_field_map(
                dict(intention.filters or {}),
                gid,
                group_by=getattr(intention, "group_by", None),
            )
            cleaned = clean_product_filters(cleaned)
            for key, value in list(cleaned.items()):
                if key in {"brand", "buyer"}:
                    continue
                if _is_schema_prose_filter(str(value)):
                    cleaned.pop(key, None)
            # When garment/type is set, drop redundant empty/prose category/family.
            if cleaned.get("garment") or cleaned.get("garment_type") or cleaned.get("subfamily"):
                for key in ("category", "family"):
                    val = cleaned.get(key)
                    if val is None or _is_schema_prose_filter(str(val)):
                        cleaned.pop(key, None)
                    elif normalize_name(str(val)) == normalize_name(
                        str(
                            cleaned.get("garment")
                            or cleaned.get("garment_type")
                            or cleaned.get("subfamily")
                            or ""
                        )
                    ):
                        cleaned.pop(key, None)
            intention.filters = cleaned
            if mapped_gb is not None:
                intention.group_by = mapped_gb
        except Exception:
            pass

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
        text = str(getattr(retrieved, "text", "") or "")
        # Metadata-backed counts (0–1 filter) must never fall through to Text2Cypher,
        # including explicit zero answers from kg_stats.
        meta_backed = False
        try:
            from knowledge_graph.retrieval.ecom import clean_product_filters
            from knowledge_graph.retrieval.kg_stats import (
                filters_match,
                normalize_filter_dict,
                stats_available,
            )

            cleaned = clean_product_filters(dict(intention.filters or {}))
            generated = getattr(getattr(platform, "settings", None), "generated", None)
            simple = len(cleaned) <= 1 and not (intention.group_by and cleaned)
            kg_evidence = [
                item
                for item in (getattr(retrieved, "evidence", None) or [])
                if getattr(item, "record_id", None) == "kg_stats"
            ]
            meta_backed = bool(
                simple
                and gid
                and stats_available(str(gid), generated)
                and kg_evidence
            )
            # Provenance guard: filtered ask answered by unfiltered global → reject.
            if meta_backed and cleaned:
                requested = normalize_filter_dict(cleaned)
                ok = any(
                    filters_match(
                        requested,
                        normalize_filter_dict(getattr(item, "applied_filters", None) or {}),
                    )
                    for item in kg_evidence
                )
                if not ok or _reject_unfiltered_global_hit(cleaned, kg_evidence):
                    meta_backed = False
        except Exception:
            meta_backed = False
        # Empty / zero-only template → Text2Cypher. Never discard a good brand+garment
        # template hit just because len(filters) >= 2 (that caused 400–500s UI waits).
        emptyish = (not text.strip()) or bool(
            re.search(
                r"(?im)^\s*(?:count|products?|n|total)\s*[:=]?\s*0\s*$|"
                r"\b0\s+products?\b|"
                r"no (?:matching )?products?\b",
                text,
            )
        )
        if (
            emptyish
            and not meta_backed
            and (
                intention.filters
                or intention.group_by
                or getattr(intention, "count_entity", None)
            )
        ):
            trace_step(
                "retrieve",
                "Deterministic count empty — falling through to Text2Cypher",
                graph_id=gid,
                filters=intention.filters,
                group_by=intention.group_by,
            )
        else:
            if result_id:
                retrieved.result_id = result_id
            if meta_backed:
                trace_step(
                    "retrieve",
                    "Answered from kg_stats metadata — skipped Text2Cypher",
                    graph_id=gid,
                    filters=intention.filters,
                )
            else:
                trace_step(
                    "retrieve",
                    "Deterministic template count kept — skipped Text2Cypher",
                    graph_id=gid,
                    filters=intention.filters,
                )
            return retrieved

    from fashion_brain.ask import _retrieve, is_open_product_listing
    from fashion_brain.agents.entity_extract import EntityExtractAgent
    from fashion_brain.stage_timing import timed

    latest_q = (question or "").strip() or ask_text
    # Explicit "about N … examples/show products" → product sample rows, not bare count.
    # Must run before simple_kg_rag so example asks are not collapsed to a count.
    if (
        wants_examples
        and not getattr(intention, "analysis", False)
        and intention.intent in {"product_count", "breakdown", "list", "count", "products"}
        and (intention.filters or {})
    ):
        sample_n = int(intention.limit or 10)
        # Use list/product_count ecom templates — intent "products" still maps to demo own_products.
        intention.intent = "list"
        intention.limit = max(1, min(sample_n, 25))
        intention.group_by = None
        trace_step(
            "open_listing",
            "Using product sample listing",
            graph_id=gid,
            filters=intention.filters,
            limit=intention.limit,
        )
        with timed("kg_retrieve", graph_id=gid, product_samples=True):
            retrieved = _retrieve(
                platform,
                principal,
                intention,
                latest_q,
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

    # Open "which products use X" asks: templated total + brand/category mix + examples.
    # Text2Cypher otherwise emits LIMIT 40–50 name dumps that drown the answer.
    # Always re-ground filters from the latest question so prior Primark|Zara cannot stick.
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


def _context_from_turn_plan(plan, question: str) -> ContextDecision:
    """Map TurnPlan into the ContextDecision shape the rest of the pipeline expects."""

    tt = str(getattr(plan, "turn_type", "") or "")
    intent_map = {
        "new_question": "new_independent_question",
        "follow_up": "continuation",
        "refinement": "modification",
        "analysis": "partial_context_reuse",
        "answers_clarification": "continuation",
        "answers_confirmation": "continuation",
        "repeat": "continuation",
        "meta": "new_independent_question",
    }
    intent_type = intent_map.get(tt, "new_independent_question")
    contextual = intent_type in {
        "continuation",
        "modification",
        "partial_context_reuse",
    }
    filters = dict(getattr(plan, "effective_filters", None) or {})
    return ContextDecision(
        decision="REUSE" if getattr(plan, "reuse_prior_answer", False) else "RETRIEVE",
        is_contextual=contextual and bool(filters or tt in {"follow_up", "refinement", "repeat"}),
        refined_question=str(getattr(plan, "rewritten_question", None) or question),
        reuse_previous_results=bool(getattr(plan, "reuse_prior_answer", False)),
        additional_retrieval_required=not bool(getattr(plan, "reuse_prior_answer", False)),
        active_filters=filters,
        intent_type=intent_type,
        inherited_filters=filters if contextual else {},
        inheritance_reason=f"turn_planner:{tt}",
    )


def _retrieve_availability_price_ops_brief(
    platform: Platform,
    principal: Principal,
    *,
    question: str,
    filters: dict,
    graph_id: str | None,
    state: dict,
    intention,
    context,
    conversation_state: dict | None,
):
    """Stock-status mix + price-event concentration by availability (A16 ops brief).

    Never falls through to full attribute-wise assortment analysis.
    """

    from contextlib import nullcontext

    try:
        from fashion_brain.ask import (
            is_availability_price_ops_brief_ask,
            live_brand_names,
            live_brands_mentioned,
            sanitize_party_filters,
        )
        from fashion_brain.stage_timing import timed
        from knowledge_graph.retrieval.ecom import (
            answer_availability_price_ops_brief,
            clean_product_filters,
            registry_product_graphs,
        )
    except Exception:
        return None

    q = str(question or "").strip()
    if not q or not is_availability_price_ops_brief_ask(q):
        return None

    clean = clean_product_filters(sanitize_party_filters(dict(filters or {})))
    if not (clean.get("brand") or clean.get("buyer")):
        try:
            names = live_brand_names(platform, "ecom_products") or live_brand_names(platform)
            hits = live_brands_mentioned(q, names)
            if hits:
                clean["brand"] = hits[0]
        except Exception:
            pass

    gid = str(graph_id or "").strip() or None
    if not gid or "plm" in gid.casefold():
        try:
            regs = list(registry_product_graphs(platform) or [])
            gid = next((g for g in regs if "ecom" in str(g).casefold()), None) or (
                regs[0] if regs else "ecom_products"
            )
        except Exception:
            gid = "ecom_products"

    try:
        with timed("kg_retrieve", graph_id=gid, availability_price_ops_brief=True):
            hit = answer_availability_price_ops_brief(platform, str(gid), q, clean)
    except Exception:
        LOGGER.debug("availability-price ops brief retrieve failed", exc_info=True)
        return None
    if hit is None:
        return None
    text, evidence, caveats = hit
    result_id = f"result_{uuid.uuid4().hex[:12]}"
    retrieved = Retrieved(
        text=text,
        evidence=list(evidence or []),
        caveats=list(caveats or []),
        unavailable=[],
        result_id=result_id,
        resolved={"filters": dict(clean), "graph_id": gid, "ops_brief": "availability_price"},
    )
    out_intention = intention if intention is not None else Intention(intent="product_count")
    out_intention.analysis = False
    out_intention.filters = dict(clean)
    out_intention.preferred_graphs = [str(gid)]
    if getattr(out_intention, "retrieve_mode", None) == "analytical_tool":
        out_intention.retrieve_mode = "template"
    state["active_filters"] = dict(clean)
    state["group_by"] = None
    state["last_result_id"] = result_id
    state["last_retrieved_text"] = retrieved.text
    state["last_graph_id"] = str(gid)
    try:
        from fashion_brain.ask import catalog_metric_family

        state["last_metric_family"] = catalog_metric_family(q)
    except Exception:
        state["last_metric_family"] = "availability_price"
    _remember_graph(state, [str(gid)], str(gid))
    if conversation_state is not None:
        conversation_state.clear()
        conversation_state.update(state)
    trace_step(
        "retrieve",
        "Availability × price ops brief — skipped assortment analysis",
        graph_id=gid,
        filters=clean,
    )
    return retrieved, ["availability_price_ops_brief", str(gid)], out_intention, context, None


def _retrieve_confirmed_price_trend(
    platform: Platform,
    principal: Principal,
    *,
    question: str,
    filters: dict,
    graph_id: str | None,
    state: dict,
    turn_plan,
    context,
    conversation_state: dict | None,
):
    """Run PriceChangeEvent trend retrieve after the user confirmed a price-trend ask.

    Bypasses ThinkingBrain / kg_stats product-count so 'yes' cannot become a Primark total.
    """

    from contextlib import nullcontext

    from fashion_brain.ask import _retrieve

    try:
        from fashion_brain.stage_timing import timed
    except Exception:  # pragma: no cover
        def timed(*_a, **_k):  # type: ignore
            return nullcontext()

    try:
        from fashion_brain.ask import (
            preferred_graphs_from_question,
            question_names_catalog_explicitly,
            sanitize_party_filters,
        )
        from knowledge_graph.retrieval.ecom import (
            clean_product_filters,
            price_history_mode,
            registry_product_graphs,
        )

        clean_filters = clean_product_filters(sanitize_party_filters(dict(filters or {})))
    except Exception:
        clean_filters = dict(filters or {})
        preferred_graphs_from_question = None  # type: ignore
        question_names_catalog_explicitly = None  # type: ignore
        price_history_mode = None  # type: ignore
        registry_product_graphs = None  # type: ignore

    # PriceChangeEvent lives on ecom. Never keep sticky/session PLM when this is a
    # price-history ask — unless the user explicitly named PLM.
    gid = (
        getattr(turn_plan, "preferred_graph", None)
        or graph_id
        or state.get("last_graph_id")
        or "ecom_products"
    )
    try:
        q = question or ""
        named_explicit = bool(
            question_names_catalog_explicitly and question_names_catalog_explicitly(q, platform)
        )
        prefs = list(preferred_graphs_from_question(q, platform) or []) if preferred_graphs_from_question else []
        price_mode = price_history_mode(q) if price_history_mode else None
        user_named_plm = named_explicit and any("plm" in str(g).casefold() for g in prefs)
        if price_mode and not user_named_plm:
            ecom = "ecom_products"
            if registry_product_graphs:
                for cand in registry_product_graphs(platform) or []:
                    low = str(cand).casefold()
                    if "ecom" in low or low in {"master_products", "master"}:
                        ecom = str(cand)
                        break
            if named_explicit and prefs:
                gid = str(prefs[0])
            else:
                gid = ecom
    except Exception:
        pass
    # Drop currency codes wrongly stored as buyer/brand (e.g. buyer=usd after "USD").
    for key in ("brand", "buyer"):
        if str(clean_filters.get(key) or "").casefold() in {
            "usd", "gbp", "eur", "both", "dollar", "dollars", "pound", "pounds",
        }:
            clean_filters.pop(key, None)
    # Keep the prior party when the user only answered a currency pick.
    if not (clean_filters.get("brand") or clean_filters.get("buyer")):
        prior = dict(state.get("active_filters") or {})
        for key in ("brand", "buyer"):
            val = prior.get(key)
            if val and str(val).casefold() not in {
                "usd", "gbp", "eur", "both", "dollar", "dollars", "pound", "pounds",
            }:
                clean_filters[key] = val
    intention = Intention(
        intent="product_count",  # temporal templates key off product_count
        filters=dict(clean_filters),
        thinking="Confirmed price trend analysis — PriceChangeEvent path",
        group_by=None,
        preferred_graphs=[str(gid)],
        retrieve_mode="template",
        metrics=["price"],
    )
    try:
        with timed("kg_retrieve", graph_id=gid, price_history="trend"):
            retrieved = _retrieve(
                platform,
                principal,
                intention,
                question,
                graph_id=str(gid),
                graph_ids=[str(gid)],
            )
    except Exception:
        LOGGER.debug("confirmed price-trend retrieve failed", exc_info=True)
        return None
    # Must look like a trend answer (summary / chart), not a bare count.
    text = str(getattr(retrieved, "text", "") or "")
    ev = list(getattr(retrieved, "evidence", None) or [])
    low_text = text.casefold()
    looks_trend = any(
        getattr(e, "record_id", None)
        in {
            "price_trend_summary",
            "price_trend_by_category",
            "price_trend_brand_compare",
            "price_trend_needs_scope",
            "price_trend_needs_currency",
            "price_trend_currency_scoped",
            "price_currency_switch_summary",
            "price_events_by_brand",
            "price_volatility",
            "price_change_event_count",
            "trending_products",
        }
        for e in ev
    ) or "```chart" in text or any(
        cue in low_text
        for cue in (
            "price change",
            "price decrease",
            "price increase",
            "price drop",
            "price drops",
            "same-currency",
            "same currency",
            "trend",
            "trending",
            "currency",
            "change %",
            "delta",
            "stock snaps",
            "availability",
        )
    )
    # Movers evidence uses product ids as record_id — accept when excerpts look like moves.
    if not looks_trend and ev and any(
        "→" in str(getattr(e, "excerpt", "") or "")
        or "price " in str(getattr(e, "excerpt", "") or "").casefold()
        for e in ev
    ):
        looks_trend = True
    if not looks_trend and (
        "products with brand" in low_text or "product analysis for" in low_text
    ):
        # Do not fall through to assortment — close politely on this graph.
        from fashion_brain.agents.contracts import Retrieved
        from knowledge_graph.retrieval.ecom import append_price_history_unavailable

        polite_lines: list[str] = []
        polite_ev: list = []
        polite_cv: list[str] = []
        append_price_history_unavailable(
            str(gid), polite_lines, polite_ev, polite_cv, question=question
        )
        retrieved = Retrieved(
            text="\n".join(polite_lines),
            evidence=polite_ev,
            caveats=polite_cv,
            unavailable=[],
            result_id=None,
        )
        looks_trend = True
    if not looks_trend:
        return None
    result_id = f"result_{uuid.uuid4().hex[:12]}"
    retrieved.result_id = result_id
    state["active_filters"] = dict(clean_filters)
    state["group_by"] = None
    state["last_result_id"] = result_id
    state["last_retrieved_text"] = retrieved.text
    state.pop("confirmation", None)
    state.pop("clarification", None)
    if turn_plan is not None:
        try:
            from fashion_brain.agents.turn_planner import next_state, scope_note

            turn_plan.effective_filters = dict(clean_filters)
            state.update(next_state(turn_plan, state))
            state["_response_scope_note"] = scope_note(turn_plan)
        except Exception:
            pass
    else:
        state["active_filters"] = dict(clean_filters)
        state["group_by"] = None
    _remember_graph(state, [str(gid)], str(gid))
    if conversation_state is not None:
        conversation_state.clear()
        conversation_state.update(state)
    route_tag = "turn_planner" if turn_plan is not None else "query_understanding"
    trace_step(
        "retrieve",
        "Confirmed price trend — PriceChangeEvent retrieve",
        graph_id=gid,
        filters=clean_filters,
    )
    return retrieved, ["price_trend", route_tag, str(gid)], intention, context, None


def _validated_from_turn_plan(plan, question: str) -> ValidatedIntent:
    metric = str(getattr(plan, "metric", "") or "").casefold()
    depth = str(getattr(plan, "analysis_depth", "") or "unset")
    metrics: list[str] = []
    if metric == "count":
        metrics = ["count"]
    elif metric and metric not in {"other", "list"}:
        metrics = [metric]
    objective = "analyze" if depth == "full" or metric == "analysis" else "execute"
    return ValidatedIntent(
        source=getattr(plan, "preferred_graph", None),
        filters=dict(getattr(plan, "effective_filters", None) or {}),
        objective=objective,
        metrics=metrics,
        group_by=getattr(plan, "group_by", None),
        analysis_depth=depth if depth in {"full", "focused", "unset"} else "unset",
        effective_question=str(getattr(plan, "rewritten_question", None) or question),
        entity=str(
            (getattr(plan, "effective_filters", None) or {}).get("brand")
            or (getattr(plan, "effective_filters", None) or {}).get("buyer")
            or ""
        )
        or None,
    )


def _meta_fast_path_eligible(
    question: str,
    state: dict,
    history: list | None = None,
) -> bool:
    """True when kg_stats may answer without turn-planner / Cypher LLM.

    Uses stats-driven detectors only — never question-specific hardcodes.
    Pending clarify/confirm still need the planner. Sticky ``last_graph_id`` alone
    must NOT block (WS session state is almost always non-empty).
    """

    if state.get("clarification") or state.get("confirmation"):
        return False
    try:
        from fashion_brain.ask import (
            _facet_entity_count,
            _is_analysis,
            is_facet_value_ranking_ask,
            is_price_trend_analysis_ask,
            is_product_count_ask,
            is_ranked_product_listing_ask,
            question_has_catalog_deixis,
        )
        from knowledge_graph.retrieval.ecom import price_history_mode

        if is_ranked_product_listing_ask(question):
            return False
        if is_price_trend_analysis_ask(question) or price_history_mode(question):
            return False
        if _is_analysis(question):
            return False
        # Modification of prior filtered scope needs planner filter carry —
        # except deixis facet rankings that only need sticky graph + shard.
        active = dict(state.get("active_filters") or {})
        from fashion_brain.ask import question_has_party_deixis, resolve_catalog_plan

        if active and not question_has_catalog_deixis(question):
            plan_filters = dict(
                resolve_catalog_plan(question, None, carry=False).get("filters") or {}
            )
            named_party = bool(plan_filters.get("brand") or plan_filters.get("buyer"))
            # "How many are red?" after sweaters must NOT fast-path alone — needs carry.
            if not question_has_party_deixis(question) and not named_party:
                if not is_facet_value_ranking_ask(question, state.get("last_graph_id")):
                    return False
        if is_facet_value_ranking_ask(question, state.get("last_graph_id")):
            return True
        if _facet_entity_count(question):
            return True
        if is_product_count_ask(question):
            return True
    except Exception:
        return False
    return False


def _finalize_meta_orchestrator_hit(
    *,
    retrieved: Retrieved,
    intention: Intention,
    gid: str,
    question: str,
    state: dict,
    conversation_state: dict | None,
    context,
    validated=None,
    route_label: str,
):
    """Shared bookkeeping for kg_stats early exits (fast path + post-planner)."""

    result_id = f"result_{uuid.uuid4().hex[:12]}"
    retrieved.result_id = result_id
    state["active_filters"] = dict(intention.filters or {})
    state["group_by"] = intention.group_by
    state["last_result_id"] = result_id
    state["last_retrieved_text"] = retrieved.text
    if validated is not None:
        state["validated_intent"] = validated.to_dict()
        store_understanding(
            state,
            validated=validated,
            summary=validated.effective_question or retrieved.text,
        )
    _remember_graph(state, [gid], gid)
    if conversation_state is not None:
        conversation_state.clear()
        conversation_state.update(state)
    trace_step(
        "retrieve",
        route_label,
        graph_id=gid,
        filters=intention.filters,
        group_by=intention.group_by,
        limit=intention.limit,
    )
    return retrieved, [intention.intent, "kg_stats", gid], intention, context, None


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

    source_question = question
    state = dict(conversation_state or {})
    turn_plan = None
    validated = None
    context = None

    # Initially off: do not study past chat history (standalone asks / clean eval turns).
    # Re-enable via fashion_brain.agents.conversation_context.STUDY_PAST_HISTORY = True.
    try:
        from fashion_brain.agents.conversation_context import STUDY_PAST_HISTORY
    except Exception:
        STUDY_PAST_HISTORY = False
    if not STUDY_PAST_HISTORY:
        history = []
        # Drop sticky session filters so prior turns cannot leak into this ask.
        # Keep pending clarification/confirmation so in-flight dialogs still resolve.
        if not (state.get("clarification") or {}).get("pending_slot") and not (
            state.get("confirmation") or {}
        ).get("question"):
            state["active_filters"] = {}

    # Metadata-sufficient asks: answer from kg_stats *before* turn-planner LLM.
    # Empirical: metadata_count_lookup ~1ms vs turn_planner LLM 5–37s.
    if _meta_fast_path_eligible(question, state, history):
        try:
            from fashion_brain.agents.contracts import ContextDecision
            from fashion_brain.ask import (
                enrich_ecom_question_filters,
                enrich_scope_from_question,
                ground_free_tokens_against_stats,
                is_catalog_dataset_total_ask,
                live_brand_names,
                preferred_graphs_from_question,
            )
            from knowledge_graph.retrieval.ecom import clean_product_filters
            from knowledge_graph.retrieval.kg_stats import normalize_filter_dict

            sticky = str(state.get("last_graph_id") or graph_id or "").strip() or None
            named = list(preferred_graphs_from_question(question, platform) or [])
            explicit_named = _explicit_graphs_in_question(question, platform)
            filters = {}
            try:
                with timed("metadata_scope_resolution", graph_id=sticky or graph_id):
                    grounded, _ = enrich_scope_from_question(
                        question,
                        {},
                        brand_names=live_brand_names(
                            platform, graph_id=sticky or graph_id
                        ),
                        platform=platform,
                    )
                    filters = clean_product_filters(
                        enrich_ecom_question_filters(question, grounded)
                    )
                    # Typos / near-misses ("swweaters") → live category/garment values.
                    filters = clean_product_filters(
                        ground_free_tokens_against_stats(
                            question,
                            filters,
                            preferred_graphs=explicit_named or named or (
                                [sticky] if sticky else []
                            ),
                            generated=getattr(
                                getattr(platform, "settings", None), "generated", None
                            ),
                        )
                    )
            except Exception:
                filters = {}
            gid_hint = _resolve_meta_count_graph(
                platform,
                question=question,
                validated=None,
                graph_id=explicit_named[0]
                if explicit_named
                else sticky or (named[0] if named else graph_id),
                state=state,
                filters=filters,
            )
            # Filtered count that still has no grounded filters → skip meta total;
            # fall through so LLM/planner can resolve typos against schema values.
            ungrounded_count = (
                not normalize_filter_dict(filters)
                and not is_catalog_dataset_total_ask(question)
            )
            # Multi-filter needs Cypher — do not fast-path.
            if (not ungrounded_count) and len(normalize_filter_dict(filters)) < 2:
                facet_hit = _try_kg_stats_facet_ranking(
                    platform,
                    question=question,
                    validated=None,
                    graph_id=gid_hint,
                    state=state,
                    intention=Intention(
                        intent="breakdown",
                        filters=filters,
                        preferred_graphs=[gid_hint] if gid_hint else named,
                    ),
                )
                if facet_hit is not None:
                    retrieved, intention, gid = facet_hit
                    context = ContextDecision(
                        decision="RETRIEVE",
                        intent_type="continuation"
                        if state.get("last_result_id")
                        else "new_independent_question",
                        is_contextual=bool(state.get("last_result_id")),
                        active_filters=dict(filters),
                        inheritance_reason="kg_stats meta fast path (skipped turn planner)",
                    )
                    return _finalize_meta_orchestrator_hit(
                        retrieved=retrieved,
                        intention=intention,
                        gid=gid,
                        question=question,
                        state=state,
                        conversation_state=conversation_state,
                        context=context,
                        route_label="kg_stats meta fast path — skipped turn planner / Text2Cypher",
                    )
                meta_hit = _try_kg_stats_meta_answer(
                    platform,
                    question=question,
                    validated=None,
                    graph_id=gid_hint,
                    state=state,
                    intention=Intention(
                        intent="product_count",
                        filters=filters,
                        preferred_graphs=[gid_hint] if gid_hint else named,
                        metrics=["count"],
                    ),
                )
                if meta_hit is not None:
                    retrieved, intention, gid = meta_hit
                    context = ContextDecision(
                        decision="RETRIEVE",
                        intent_type="continuation"
                        if state.get("last_result_id")
                        else "new_independent_question",
                        is_contextual=bool(state.get("last_result_id")),
                        active_filters=dict(filters),
                        inheritance_reason="kg_stats meta fast path (skipped turn planner)",
                    )
                    return _finalize_meta_orchestrator_hit(
                        retrieved=retrieved,
                        intention=intention,
                        gid=gid,
                        question=question,
                        state=state,
                        conversation_state=conversation_state,
                        context=context,
                        route_label="kg_stats meta fast path — skipped turn planner / Text2Cypher",
                    )
        except Exception:
            LOGGER.debug("meta fast path skipped", exc_info=True)

    # Turn planner (single LLM call). On llm_ok it owns scope/clarify; else legacy path.
    try:
        from fashion_brain.agents.turn_planner import (
            interruption_reply,
            next_state,
            plan_turn,
            resolve_scope,
            scope_note,
        )
        from fashion_brain.planning.kg_catalog import build_field_catalog

        with timed("turn_planner"):
            catalog = build_field_catalog(platform, question, probe_values=False)
            turn_plan = plan_turn(platform, question, history, state, catalog)
    except Exception as exc:
        LOGGER.debug("turn planner unavailable: %s", exc)
        turn_plan = None

    if turn_plan is not None and bool(getattr(turn_plan, "llm_ok", False)):
        resolve_scope(turn_plan, state)
        ask = interruption_reply(turn_plan)
        context = _context_from_turn_plan(turn_plan, question)
        trace_step(
            "turn_planner",
            "TurnPlan ready",
            turn_type=turn_plan.turn_type,
            carry=turn_plan.carry_filters,
            filters=turn_plan.effective_filters,
            clarify=bool(ask),
        )
        if ask:
            state.update(next_state(turn_plan, state))
            state["_response_scope_note"] = scope_note(turn_plan)
            if conversation_state is not None:
                conversation_state.clear()
                conversation_state.update(state)
            intention = Intention(
                intent="clarification",
                filters=dict(turn_plan.effective_filters),
                thinking=f"Turn planner interruption ({turn_plan.turn_type})",
                group_by=turn_plan.group_by,
                preferred_graphs=[turn_plan.preferred_graph]
                if turn_plan.preferred_graph
                else [],
            )
            retrieved = Retrieved(
                text=ask,
                evidence=[],
                caveats=[QueryUnderstandingAgent.AWAITING_CAVEAT],
                unavailable=[],
                result_id=None,
            )
            return retrieved, ["clarification", "turn_planner"], intention, context, None

        # Keep product-hotness / facet-ranking wording; never let an LLM avg-trend rewrite win.
        user_q = question
        try:
            from fashion_brain.ask import is_dominant_facet_ask, is_trending_product_ask

            rewritten = str(turn_plan.rewritten_question or "").strip()
            if is_dominant_facet_ask(user_q):
                question = rewritten if is_dominant_facet_ask(rewritten) else user_q
            elif is_trending_product_ask(user_q) and not is_trending_product_ask(rewritten):
                question = rewritten if "trending" in rewritten.casefold() else user_q
            else:
                question = rewritten or user_q
        except Exception:
            question = turn_plan.rewritten_question or question
        state["active_filters"] = dict(turn_plan.effective_filters)
        if turn_plan.group_by is not None or turn_plan.turn_type == "new_question":
            state["group_by"] = turn_plan.group_by
        # Confirmed / currency-clarified price-trend must not keep leftover group_by.
        if (
            turn_plan.turn_type in {"answers_confirmation", "answers_clarification"}
            and str(turn_plan.metric or "").casefold() == "price"
        ):
            turn_plan.group_by = None
            state["group_by"] = None
            state.pop("confirmation", None)
            if turn_plan.turn_type == "answers_clarification":
                state.pop("clarification", None)
        state["_response_scope_note"] = scope_note(turn_plan)
        validated = _validated_from_turn_plan(turn_plan, question)
        store_understanding(
            state,
            validated=validated,
            summary=validated.effective_question or validated.objective,
        )

        # Price-trend / movers → PriceChangeEvent path (skip count / full assortment).
        # Honor price_history_mode even when the LLM marks metric=analysis.
        # Never steal facet rankings ("which color is most…") into price-trend.
        facet_rank_ask = False
        try:
            from fashion_brain.ask import is_dominant_facet_ask

            facet_rank_ask = bool(is_dominant_facet_ask(user_q) or turn_plan.group_by)
        except Exception:
            facet_rank_ask = bool(turn_plan.group_by)
        price_metric = str(turn_plan.metric or "").casefold() == "price"
        price_mode_hit = False
        try:
            from knowledge_graph.retrieval.ecom import price_history_mode as _ph_mode

            price_mode_hit = bool(_ph_mode(question))
        except Exception:
            price_mode_hit = False
        if (price_metric or price_mode_hit) and not facet_rank_ask:
            if turn_plan.turn_type in {"answers_confirmation", "answers_clarification"}:
                turn_plan.group_by = None
                state["group_by"] = None
            price_hit = _retrieve_confirmed_price_trend(
                platform,
                principal,
                question=question,
                filters=dict(turn_plan.effective_filters or {}),
                graph_id=graph_id,
                state=state,
                turn_plan=turn_plan,
                context=context,
                conversation_state=conversation_state,
            )
            if price_hit is not None:
                return price_hit

        if (
            turn_plan.reuse_prior_answer
            and state.get("last_retrieved_text")
            and not (state.get("clarification") or {}).get("pending_slot")
        ):
            retrieved = Retrieved(
                text=str(state.get("last_retrieved_text") or ""),
                evidence=[],
                caveats=["Reused previous retrieval without a new knowledge-graph query."],
                unavailable=[],
                result_id=state.get("last_result_id"),
            )
            intention = Intention(
                intent="reuse",
                filters=dict(turn_plan.effective_filters),
                thinking="Turn planner reuse_prior_answer",
                group_by=turn_plan.group_by,
            )
            state.update(next_state(turn_plan, state))
            if conversation_state is not None:
                conversation_state.clear()
                conversation_state.update(state)
            trace_step("retrieve", "Turn planner reused prior retrieval")
            return retrieved, ["reuse", "turn_planner"], intention, context, None
        # Fall through to kg_stats / Thinking / Intention with validated + context set.
    else:
        # Legacy ConversationContext + QU path (unchanged).
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
        # Align session filters with context before QU / kg_stats (avoid sticky brand leak).
        if context.intent_type == "new_independent_question":
            state["active_filters"] = dict(context.active_filters or {})
        elif context.active_filters is not None and (
            context.is_contextual or context.intent_type == "partial_context_reuse"
        ):
            state["active_filters"] = dict(context.active_filters)

        pending_clarification = bool((state.get("clarification") or {}).get("pending_slot"))

        # Do not reuse prior findings while the user is answering a clarification.
        if (
            context.decision == "REUSE"
            and state.get("last_retrieved_text")
            and not pending_clarification
        ):
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

        # Progressive query understanding — one metadata-grounded clarify, or validated intent.
        with timed("query_understanding"):
            qu_result = QueryUnderstandingAgent().evaluate(
                platform,
                question,
                history,
                context=context,
                conversation_state=state,
                graph_id=graph_id,
            )
        if isinstance(qu_result, ClarifyResult):
            state["clarification"] = qu_result.state.to_dict()
            # Persist step-by-step analysis filters on the session for continuous turns.
            confirmed = dict(qu_result.state.confirmed or {})
            step_filters = dict(confirmed.get("filters") or {})
            if step_filters:
                state["active_filters"] = dict(step_filters)
            state["analysis_brief"] = {
                "entity": confirmed.get("entity"),
                "filters": step_filters,
                "filter_trail": list(confirmed.get("filter_trail") or []),
                "analysis_depth": confirmed.get("analysis_depth") or "unset",
                "focused_dimension": confirmed.get("focused_dimension"),
                "original_question": qu_result.state.original_question,
                "pending_slot": qu_result.slot,
                "turn_count": qu_result.state.turn_count,
            }
            store_understanding(
                state,
                clarification=qu_result.state,
                summary=str(confirmed.get("understanding_summary") or "")
                or f"Awaiting clarification ({qu_result.slot})",
                pending_slot=qu_result.slot,
            )
            if conversation_state is not None:
                conversation_state.clear()
                conversation_state.update(state)
            intention = Intention(
                intent="clarification",
                filters=step_filters,
                thinking=f"Awaiting clarification ({qu_result.slot})",
                group_by=confirmed.get("focused_dimension"),
            )
            retrieved = Retrieved(
                text=qu_result.question,
                evidence=[],
                caveats=[QueryUnderstandingAgent.AWAITING_CAVEAT],
                unavailable=[],
                result_id=None,
            )
            trace_step(
                "query_understanding",
                "Clarification required — skipped retrieval",
                slot=qu_result.slot,
                turn=qu_result.state.turn_count,
            )
            return retrieved, ["clarification", qu_result.slot], intention, context, None

        if isinstance(qu_result, ReadyResult):
            validated = qu_result.intent
            if qu_result.state and qu_result.state.pending_slot is None:
                state.pop("clarification", None)
            elif qu_result.state:
                state["clarification"] = qu_result.state.to_dict()
            else:
                state.pop("clarification", None)
            state["validated_intent"] = validated.to_dict()
            store_understanding(
                state,
                validated=validated,
                clarification=state.get("clarification"),
                summary=validated.effective_question or validated.objective,
            )
            trace_step(
                "query_understanding",
                "Intent ready",
                depth=validated.analysis_depth,
                filters=validated.filters,
                objective=validated.objective,
                metrics=validated.metrics,
            )
            # QU ReadyResult with metrics=["price"] → PriceChangeEvent (not assortment).
            qu_metrics = {
                str(m).casefold() for m in (getattr(validated, "metrics", None) or []) if m
            }
            price_ask = "price" in qu_metrics
            if not price_ask:
                try:
                    from fashion_brain.ask import (
                        _is_price_history_question,
                        is_price_trend_analysis_ask,
                    )

                    price_ask = bool(
                        is_price_trend_analysis_ask(question or "")
                        or _is_price_history_question(question or "")
                        or is_price_trend_analysis_ask(
                            str(getattr(validated, "effective_question", "") or "")
                        )
                    )
                except Exception:
                    price_ask = False
            if price_ask:
                price_hit = _retrieve_confirmed_price_trend(
                    platform,
                    principal,
                    question=str(
                        getattr(validated, "effective_question", None) or question
                    ),
                    filters=dict(getattr(validated, "filters", None) or {}),
                    graph_id=graph_id,
                    state=state,
                    turn_plan=turn_plan,
                    context=context,
                    conversation_state=conversation_state,
                )
                if price_hit is not None:
                    return price_hit

    # Top-N facet names from the sticky KG filter shard (e.g. plmex brands of them).
    facet_rank_hit = _try_kg_stats_facet_ranking(
        platform,
        question=question,
        validated=validated,
        graph_id=graph_id,
        state=state,
    )
    if facet_rank_hit is not None:
        retrieved, intention, gid = facet_rank_hit
        result_id = f"result_{uuid.uuid4().hex[:12]}"
        retrieved.result_id = result_id
        state["active_filters"] = dict(intention.filters or {})
        state["group_by"] = intention.group_by
        state["last_result_id"] = result_id
        state["last_retrieved_text"] = retrieved.text
        if validated is not None:
            state["validated_intent"] = validated.to_dict()
            store_understanding(
                state,
                validated=validated,
                summary=validated.effective_question or retrieved.text,
            )
        _remember_graph(state, [gid], gid)
        if conversation_state is not None:
            conversation_state.clear()
            conversation_state.update(state)
        trace_step(
            "retrieve",
            "Answered top facet values from kg_stats — skipped Text2Cypher",
            graph_id=gid,
            group_by=intention.group_by,
            limit=intention.limit,
        )
        return retrieved, [intention.intent, "kg_stats", gid], intention, context, None

    # Simple totals / 0–1 filter counts: answer from kg_stats for any source that
    # has persisted stats. Merge QU + ThinkingBrain filters so Scope is never dropped.
    # Multi-filter / uncached dimension → cache_miss → continue to Intention/Text2Cypher.
    if validated is not None:
        meta_hit = _try_kg_stats_meta_answer(
            platform,
            question=question,
            validated=validated,
            graph_id=graph_id,
            state=state,
            summary=summary,
        )
        if meta_hit is not None:
            retrieved, intention, gid = meta_hit
            result_id = f"result_{uuid.uuid4().hex[:12]}"
            retrieved.result_id = result_id
            state["active_filters"] = dict(intention.filters or {})
            state["group_by"] = intention.group_by
            state["last_result_id"] = result_id
            state["last_retrieved_text"] = retrieved.text
            state["validated_intent"] = validated.to_dict()
            store_understanding(
                state,
                validated=validated,
                summary=validated.effective_question or retrieved.text,
            )
            _remember_graph(state, [gid], gid)
            try:
                from fashion_brain.ask import catalog_metric_family

                state["last_metric_family"] = catalog_metric_family(question)
            except Exception:
                pass
            if turn_plan is not None and bool(getattr(turn_plan, "llm_ok", False)):
                from fashion_brain.agents.turn_planner import next_state, scope_note

                turn_plan.effective_filters = dict(intention.filters or turn_plan.effective_filters)
                state.update(next_state(turn_plan, state))
                state["_response_scope_note"] = scope_note(turn_plan)
            if conversation_state is not None:
                conversation_state.clear()
                conversation_state.update(state)
            trace_step(
                "retrieve",
                "Answered from kg_stats metadata — skipped ThinkingBrain / Text2Cypher",
                graph_id=gid,
                filters=intention.filters,
            )
            return retrieved, [intention.intent, "kg_stats", gid], intention, context, None

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
    # Filter carry follows ConversationContextAgent / TurnPlan only — not mere last_graph_id.
    # Catalog stickiness is handled by _apply_session_graph_memory separately.
    if summary is not None and context is not None and context.is_contextual and context.active_filters:
        summary.references_prior_chat = True
    elif summary is not None and context is not None and context.intent_type == "new_independent_question":
        summary.references_prior_chat = False
        try:
            from fashion_brain.ask import is_catalog_dataset_total_ask

            if is_catalog_dataset_total_ask(question):
                summary.filters.pop("brand", None)
                summary.filters.pop("buyer", None)
        except Exception:
            pass
    # Turn planner already resolved scope — do not let Thinking re-glue sticky brands.
    if (
        summary is not None
        and turn_plan is not None
        and bool(getattr(turn_plan, "llm_ok", False))
    ):
        summary.filters = dict(turn_plan.effective_filters)
        summary.group_by = turn_plan.group_by or summary.group_by
        summary.references_prior_chat = bool(
            context is not None and context.is_contextual and turn_plan.effective_filters
        )

    # QU already validated intent — do not let ThinkingBrain re-open clarification.
    if validated is not None and summary is not None and summary.clarification_required:
        summary.clarification_required = False

    ask_question = question
    if validated is not None and validated.effective_question:
        ask_question = validated.effective_question

    with timed("intention_agent"):
        intention = IntentionAgent().understand(
            platform,
            ask_question,
            history,
            summary,
            conversation_state=state,
        )
    if intention.intent == "clarification" and (
        getattr(intention, "clarify_text", "") or intention.thinking
    ):
        clarify_text = str(getattr(intention, "clarify_text", "") or intention.thinking or "").strip()
        options = list(getattr(intention, "clarify_options", None) or [])
        option_labels = [
            str(o.get("restated_question") if isinstance(o, dict) else o).strip()
            for o in options
            if (o.get("restated_question") if isinstance(o, dict) else o)
        ]
        state["clarification"] = {
            "pending_slot": "metadata",
            "question": clarify_text,
            "options": option_labels,
            "interpretations": options,
            "filters": dict(intention.filters or {}),
        }
        if conversation_state is not None:
            conversation_state.clear()
            conversation_state.update(state)
        retrieved = Retrieved(
            text=clarify_text,
            evidence=[],
            caveats=[QueryUnderstandingAgent.AWAITING_CAVEAT],
            unavailable=[],
            result_id=None,
        )
        trace_step(
            "intention",
            "Metadata resolver clarification — skipped retrieval",
            options=len(option_labels),
        )
        return retrieved, ["clarification", "metadata"], intention, context, None
    spec_ask = (
        str(getattr(intention, "intent", "") or "") in {"specification", "product_details"}
        or bool(getattr(intention, "suggest", False))
    )
    if validated is not None:
        intention = apply_validated_intent(intention, validated, summary=summary)
        # Re-evaluate after apply — IntentionAgent may already have locked specification.
        spec_ask = (
            str(getattr(intention, "intent", "") or "") in {"specification", "product_details"}
            or bool(getattr(intention, "suggest", False))
        )
        if validated.analysis_depth == "full" and not spec_ask:
            # Confirmed full analysis: validated filters win completely (brand only).
            intention.filters = dict(validated.filters or {})
            intention.group_by = validated.group_by
            intention.analysis = True
            state["active_filters"] = dict(intention.filters or {})
        elif validated.filters:
            # Prefer validated keys, but do not re-introduce dropped sticky facets.
            intention.filters = {**(intention.filters or {}), **validated.filters}
        state["validated_intent"] = validated.to_dict()
        if validated.effective_question and not spec_ask:
            # Downstream retrieve / planner should use the resolved analytical ask.
            question = validated.effective_question
    if turn_plan is not None and bool(getattr(turn_plan, "llm_ok", False)):
        if turn_plan.preferred_graph:
            prefs = list(intention.preferred_graphs or [])
            if turn_plan.preferred_graph not in prefs:
                intention.preferred_graphs = [turn_plan.preferred_graph, *prefs]
        if turn_plan.group_by and not intention.group_by and not spec_ask:
            intention.group_by = turn_plan.group_by
        if turn_plan.analysis_depth == "full" and not spec_ask:
            intention.analysis = True
        # Turn planner owns this turn's scope — do not let Intention re-glue sticky
        # sweater/color from earlier slices onto "top 10 … among them".
        planned = dict(turn_plan.effective_filters or {})
        try:
            from fashion_brain.ask import sanitize_party_filters
            from knowledge_graph.ontology import normalize_name

            planned = sanitize_party_filters(planned)
            qn = normalize_name(question or "")
            for key, value in list((intention.filters or {}).items()):
                if key in planned or value in (None, ""):
                    continue
                token = normalize_name(str(value))
                if token and token in qn:
                    planned[key] = value
        except Exception:
            planned = dict(turn_plan.effective_filters or {}) or dict(intention.filters or {})
        intention.filters = planned
    # Spec sheet path: never keep analysis flags / sentence-as-brand pollution.
    if spec_ask or str(getattr(intention, "intent", "") or "") in {
        "specification",
        "product_details",
    } or bool(getattr(intention, "suggest", False)):
        intention.analysis = False
        intention.retrieve_mode = "template"
        try:
            from fashion_brain.ask import sanitize_party_filters

            intention.filters = sanitize_party_filters(dict(intention.filters or {}))
        except Exception:
            pass
    # Ranked product listing: force limit + skip sticky facet re-merge below.
    ranked_listing = False
    try:
        from fashion_brain.ask import (
            is_ranked_product_listing_ask,
            ranked_product_listing_limit,
        )

        ranked_listing = is_ranked_product_listing_ask(question)
        if ranked_listing:
            intention.group_by = None
            lim = ranked_product_listing_limit(question)
            if lim and not intention.limit:
                intention.limit = lim
            intention.metrics = list(
                dict.fromkeys([*(intention.metrics or []), "list", "price"])
            )
    except Exception:
        ranked_listing = False
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
        question=source_question,
        history=history,
        context=context,
        state=state,
        graph_id=graph_id,
    )
    intention = _prefer_plmex_for_assortment_filters(
        platform,
        intention,
        question=source_question,
        state=state,
    )
    # After assortment/session pins: price/availability history must use ecom events.
    intention = _prefer_ecom_for_price_history(
        platform,
        intention,
        question=source_question,
    )
    explicit_graphs = _apply_explicit_graph_preferences(
        intention,
        source_question,
        platform,
    )
    if explicit_graphs:
        if list(intention.preferred_graphs or []) != explicit_graphs:
            trace_step(
                "router",
                "Explicit source mention overrides session/planner graph selection",
                selected=explicit_graphs,
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

    full_analysis = bool(
        validated is not None and str(getattr(validated, "analysis_depth", "") or "") == "full"
    )
    turn_plan_owns_scope = bool(
        turn_plan is not None and bool(getattr(turn_plan, "llm_ok", False))
    )
    # Carry only when context both marks the turn contextual AND supplies filters to keep.
    # (is_contextual alone used to glue Primark|Zara onto unrelated cotton asks.)
    # Full analysis after final_confirm must not re-absorb prior sweater/color scope.
    # Turn planner follow-ups already resolved filters — do not re-merge session sweater.
    if full_analysis or turn_plan_owns_scope or ranked_listing:
        state["active_filters"] = dict(intention.filters or {})
        carry_now = False
    else:
        if context.active_filters:
            state["active_filters"] = dict(context.active_filters)
        carry_now = bool(context.is_contextual and context.active_filters)
    filters, group_by, effective_question, state_patch = apply_turn_constraints(
        question,
        history,
        intention.filters,
        intention.group_by,
        schema=schema,
        conversation_state=state,
        is_contextual=bool(context.is_contextual) and not full_analysis,
        carry_filters=carry_now,
        refined_question=context.refined_question if not full_analysis else None,
    )
    if full_analysis:
        filters = dict(intention.filters or {})
        group_by = intention.group_by
        effective_question = (
            (validated.effective_question if validated is not None else None) or question
        )
        state_patch = {
            "active_filters": filters,
            "group_by": group_by,
            "last_question": question,
            "effective_question": effective_question,
        }
    intention.filters = dict(filters)
    temporal_ask = False
    avail_mode = None
    price_mode = None
    ops_brief_ask = False
    try:
        from fashion_brain.ask import (
            availability_history_mode,
            is_availability_price_ops_brief_ask,
            live_brand_names,
            live_brands_mentioned,
            temporal_count_entity,
        )
        from knowledge_graph.retrieval.ecom import price_history_mode

        avail_mode = availability_history_mode(question)
        price_mode = price_history_mode(question)
        ops_brief_ask = is_availability_price_ops_brief_ask(question)
        temporal_ask = bool(price_mode or avail_mode or temporal_count_entity(question))
    except Exception:
        temporal_ask = False
        ops_brief_ask = False
    if ops_brief_ask:
        # A16: availability × price ops brief — never full assortment analysis.
        intention.analysis = False
        intention.group_by = None
        if intention.intent in {"breakdown", "count", "list", "other", "analysis"}:
            intention.intent = "product_count"
        if not (intention.filters or {}).get("brand") and not (intention.filters or {}).get("buyer"):
            try:
                names = live_brand_names(platform, "ecom_products") or live_brand_names(platform)
                hits = live_brands_mentioned(question, names)
                if hits:
                    intention.filters = dict(intention.filters or {})
                    intention.filters["brand"] = hits[0]
            except Exception:
                pass
        intention.metrics = list(
            dict.fromkeys([*(intention.metrics or []), "availability", "price"])
        )
        if getattr(intention, "retrieve_mode", None) == "analytical_tool":
            intention.retrieve_mode = "template"
    elif bool(getattr(intention, "analysis", False)) and not (
        (intention.filters or {}).get("brand") or (intention.filters or {}).get("buyer")
    ):
        # Full merchant analysis named a live brand in prose but filters were dropped.
        try:
            names = live_brand_names(platform, "ecom_products") or live_brand_names(platform)
            hits = live_brands_mentioned(question, names)
            if len(hits) == 1:
                intention.filters = dict(intention.filters or {})
                intention.filters["brand"] = hits[0]
            elif len(hits) >= 2:
                intention.filters = dict(intention.filters or {})
                intention.filters["brand"] = " | ".join(hits[:4])
        except Exception:
            pass
    if temporal_ask and not ops_brief_ask:
        # Temporal templates key off product_count; never demote them to breakdown
        # just because a prior turn carried a group_by.
        if intention.intent in {"breakdown", "count", "list", "other", "product_count"}:
            intention.intent = "product_count"
        # Price/availability history must never become full assortment analysis.
        intention.analysis = False
        if "price" not in {str(m).casefold() for m in (intention.metrics or [])}:
            intention.metrics = [*(intention.metrics or []), "price" if price_mode else "availability"]
        # Only keep a breakdown dimension when this turn itself is a brand/category ranking.
        if avail_mode == "events_by_brand" or price_mode == "events_by_brand":
            intention.group_by = "brand"
        elif price_mode == "trend_by_category":
            intention.group_by = "category"
        else:
            intention.group_by = None
    elif group_by and not ops_brief_ask:
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

    # Drop grammar tokens mistaken as party filters ("brand name" → buyer=name).
    try:
        from fashion_brain.ask import _is_grammar_party_name

        cleaned_filters = {}
        for key, value in dict(intention.filters or {}).items():
            if str(key).casefold() in {"brand", "buyer"} and _is_grammar_party_name(str(value)):
                continue
            cleaned_filters[key] = value
        intention.filters = cleaned_filters
    except Exception:
        pass

    # Cross-catalog type compare (A17): agents often invent brand=thinking and remap
    # garment→subfamily=sweater because the brief mentions subfamily *roles*. Fix
    # filters, answer from kg_stats per source, and never await clarification.
    product_ids_for_cross = _product_graph_ids(
        list(getattr(intention, "preferred_graphs", None) or [])
        or list(graph_ids or [])
        or ([graph_id] if graph_id else []),
        graph_id,
    )
    if len(product_ids_for_cross) < 2:
        named = _explicit_graphs_in_question(source_question or question, platform)
        if len(named) >= 2:
            product_ids_for_cross = named
    _normalize_cross_catalog_type_intention(
        intention, source_question or question, product_ids_for_cross
    )
    cross_early = _try_cross_catalog_type_compare(
        platform,
        question=source_question or question,
        intention=intention,
        product_ids=product_ids_for_cross,
    )
    if cross_early is not None:
        result_id = f"result_{uuid.uuid4().hex[:12]}"
        cross_early.result_id = result_id
        state.pop("clarification", None)
        state["active_filters"] = dict(intention.filters or {})
        state["group_by"] = None
        state["last_result_id"] = result_id
        state["last_retrieved_text"] = cross_early.text
        _remember_graph(state, product_ids_for_cross, product_ids_for_cross[0])
        if conversation_state is not None:
            conversation_state.clear()
            conversation_state.update(state)
        trace_step(
            "retrieve",
            "Cross-catalog type compare from kg_stats before filter gate",
            graphs=product_ids_for_cross,
            filters=intention.filters,
        )
        return (
            cross_early,
            [intention.intent or "shared_products", "kg_stats", "cross_catalog", *product_ids_for_cross],
            intention,
            context,
            None,
        )

    # Strict metadata-grounded filter gate: never pass unresolved values into Cypher.
    # Empty filters (catalog totals) pass. Pending understanding_plan resumes here.
    try:
        from fashion_brain.planning.filter_validation import gate_intention_filters

        skip_filter_gate = (
            str(getattr(intention, "intent", "") or "").casefold() in {"clarification", "chitchat"}
            or temporal_ask
            or ranked_listing
        )
        if not skip_filter_gate:
            understanding_plan, intention = gate_intention_filters(
                platform,
                intention,
                question=source_question or question,
                state=state,
                verify_db=True,
            )
            state["understanding_plan"] = understanding_plan.to_dict()
            if not understanding_plan.execution_allowed:
                clarify_text = (
                    understanding_plan.clarification_text
                    or "I need a bit more detail before I can query the catalog."
                )
                state["clarification"] = {
                    "pending_slot": "filter_validation",
                    "question": clarify_text,
                    "options": [
                        c
                        for item in understanding_plan.clarification_items
                        for c in (item.candidates or [])
                    ][:8],
                    "filters": dict(intention.filters or {}),
                    "clarification_items": [
                        item.to_dict() for item in understanding_plan.clarification_items
                    ],
                }
                state["active_filters"] = dict(intention.filters or {})
                if conversation_state is not None:
                    conversation_state.clear()
                    conversation_state.update(state)
                retrieved = Retrieved(
                    text=clarify_text,
                    evidence=[],
                    caveats=[QueryUnderstandingAgent.AWAITING_CAVEAT],
                    unavailable=[],
                    result_id=None,
                    resolved={
                        "filters": dict(intention.filters or {}),
                        "understanding_plan": understanding_plan.to_dict(),
                    },
                )
                trace_step(
                    "filter_validation",
                    "Blocked retrieval — unresolved filters require clarification",
                    blocking=[
                        f"{f.field}:{f.status}"
                        for f in understanding_plan.filters
                        if f.status
                        in {"NOT_FOUND", "UNKNOWN", "AMBIGUOUS", "UNSUPPORTED"}
                    ],
                    kept=list(intention.filters or {}),
                )
                return (
                    retrieved,
                    ["clarification", "filter_validation"],
                    intention,
                    context,
                    None,
                )
            # Execution allowed — drop pending clarification residue for this path.
            if state.get("clarification", {}).get("pending_slot") == "filter_validation":
                state.pop("clarification", None)
    except Exception:
        LOGGER.warning("strict filter validation skipped", exc_info=True)

    # Top-N facet names after sticky graph memory (plmex "of them" follow-ups).
    facet_rank_hit = _try_kg_stats_facet_ranking(
        platform,
        question=question,
        validated=validated,
        graph_id=graph_id,
        state=state,
        intention=intention,
    )
    if facet_rank_hit is not None:
        retrieved, intention, gid = facet_rank_hit
        result_id = f"result_{uuid.uuid4().hex[:12]}"
        retrieved.result_id = result_id
        state["active_filters"] = dict(intention.filters or {})
        state["group_by"] = intention.group_by
        state["last_result_id"] = result_id
        state["last_retrieved_text"] = retrieved.text
        _remember_graph(state, [gid], gid)
        if conversation_state is not None:
            conversation_state.clear()
            conversation_state.update(state)
        trace_step(
            "retrieve",
            "Answered top facet values from kg_stats after Intention — skipped Text2Cypher",
            graph_id=gid,
            group_by=intention.group_by,
            limit=intention.limit,
        )
        return retrieved, [intention.intent, "kg_stats", gid], intention, context, None

    # Second meta chance: Intention/ThinkingBrain may have filled filters that QU omitted.
    # Still refuses multi-filter / uncached dims (cache_miss → Text2Cypher below).
    meta_hit = _try_kg_stats_meta_answer(
        platform,
        # Use the raw user ask — rewritten effective_question carries schema hints
        # that must not be parsed into facet filter values.
        question=question,
        validated=validated,
        graph_id=graph_id,
        state=state,
        summary=summary,
        intention=intention,
    )
    if meta_hit is not None:
        retrieved, meta_intention, gid = meta_hit
        # Keep Intention's richer preferred_graphs / metrics when meta only filled filters.
        if meta_intention.filters:
            intention.filters = dict(meta_intention.filters)
        result_id = f"result_{uuid.uuid4().hex[:12]}"
        retrieved.result_id = result_id
        state["active_filters"] = dict(intention.filters or {})
        state["group_by"] = intention.group_by
        state["last_result_id"] = result_id
        state["last_retrieved_text"] = retrieved.text
        _remember_graph(state, [gid], gid)
        if turn_plan is not None and bool(getattr(turn_plan, "llm_ok", False)):
            from fashion_brain.agents.turn_planner import next_state, scope_note

            turn_plan.effective_filters = dict(intention.filters or turn_plan.effective_filters)
            state.update(next_state(turn_plan, state))
            state["_response_scope_note"] = scope_note(turn_plan)
        if conversation_state is not None:
            conversation_state.clear()
            conversation_state.update(state)
        trace_step(
            "retrieve",
            "Answered from kg_stats metadata after Intention — skipped Text2Cypher",
            graph_id=gid,
            filters=intention.filters,
        )
        return retrieved, [intention.intent, "kg_stats", gid], intention, context, None

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

    graph_ids = _apply_request_graph_pin(
        list(choice.selected_graphs or []),
        graph_id=graph_id,
        intent=intention.intent,
        question=source_question,
        platform=platform,
    )
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

    # Short-TTL findings cache: skip Neo4j when same intent+domains+graphs already fetched.
    ask_for_cache = (effective_question or question or "").strip() or question
    ontology_domains = _ontology_domains_for_question(platform, ask_for_cache)
    metric_family = ""
    try:
        from fashion_brain.ask import catalog_metric_family

        metric_family = catalog_metric_family(ask_for_cache)
    except Exception:
        metric_family = ""
    product_ids_early = _product_graph_ids(graph_ids, graph_id or choice.graph)
    cached = _try_query_cache(
        platform,
        ask_for_cache,
        ontology_domains=ontology_domains,
        graph_ids=product_ids_early or graph_ids,
        metric_family=metric_family,
        intent=str(intention.intent or ""),
    )
    if cached is not None and context.decision != "REUSE":
        retrieved = Retrieved(
            text=str(cached.get("text") or ""),
            evidence=list(cached.get("evidence") or []),
            caveats=list(
                dict.fromkeys(
                    [
                        *(cached.get("caveats") or []),
                        "Served from short-TTL query cache (skipped Neo4j).",
                    ]
                )
            ),
            unavailable=[],
            result_id=result_id,
        )
        state["active_filters"] = dict(intention.filters or {})
        state["group_by"] = intention.group_by
        state["last_result_id"] = result_id
        state["last_retrieved_text"] = retrieved.text
        state["last_ontology_domains"] = list(ontology_domains)
        state["last_metric_family"] = metric_family
        _remember_graph(
            state,
            list(product_ids_early or graph_ids or []),
            graph_id or (choice.graph if choice else None),
        )
        if conversation_state is not None:
            conversation_state.clear()
            conversation_state.update(state)
        trace_step("retrieve", "Query cache hit — skipped Neo4j", domains=ontology_domains)
        return retrieved, [intention.intent, "query_cache", *graph_ids], intention, context, choice

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
    # Full analysis (all facets) and focused breakdown (group_by one facet) share this engine.
    # Never run assortment analysis for price-history / movers / availability-price ops briefs.
    try:
        from fashion_brain.ask import is_availability_price_ops_brief_ask
        from knowledge_graph.retrieval.ecom import price_history_mode as _ph_gate

        q_gate = effective_question or question
        if _ph_gate(q_gate) or is_availability_price_ops_brief_ask(q_gate):
            intention.analysis = False
            if getattr(intention, "retrieve_mode", None) == "analytical_tool":
                intention.retrieve_mode = "template"
    except Exception:
        pass

    # A16-style ops brief: stock-status mix + price events by availability (not full assortment).
    ops_brief_hit = _retrieve_availability_price_ops_brief(
        platform,
        principal,
        question=source_question or question,
        filters=dict(intention.filters or {}),
        graph_id=graph_id or (product_ids[0] if product_ids else None),
        state=state,
        intention=intention,
        context=context,
        conversation_state=conversation_state,
    )
    if ops_brief_hit is not None:
        return ops_brief_hit

    if (
        getattr(intention, "retrieve_mode", None) == "analytical_tool"
        or bool(getattr(intention, "analysis", False))
    ) and intention.intent not in {
        "system_meta",
        "live_sql",
        "news",
        "shared_products",
        "specification",
        "product_details",
    } and not bool(getattr(intention, "suggest", False)):
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
        # Align filters/group_by to the first analysis catalog so Scope + retrieve
        # use mapped columns (ecom buyer→brand, gender→section, …).
        if analysis_graphs:
            try:
                from fashion_brain.planning.semantic_field_map import apply_semantic_field_map
                from knowledge_graph.retrieval.ecom import (
                    clean_product_filters,
                    _align_party_filters_for_graph,
                )

                mapped, mapped_gb = apply_semantic_field_map(
                    dict(intention.filters or {}),
                    analysis_graphs[0],
                    group_by=getattr(intention, "group_by", None),
                )
                intention.filters = _align_party_filters_for_graph(
                    platform,
                    analysis_graphs[0],
                    clean_product_filters(mapped),
                )
                if mapped_gb is not None:
                    intention.group_by = mapped_gb
            except Exception:
                pass
        with timed("kg_retrieve", graphs=analysis_graphs, analysis=True, analytical_tool=True):
            retrieved = run_product_analysis(
                platform,
                principal,
                graph_ids=analysis_graphs,
                filters=dict(intention.filters or {}),
                limit=intention.limit,
                group_by=intention.group_by,
                question=effective_question or question or "",
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
            or intention.intent in {"shared_products", "shared", "specification"}
            or str(getattr(intention, "query_shape", "") or "") == "shared_overlap"
            or bool(getattr(intention, "suggest", False))
        )
    ):
        # Separate per-source type counts from kg_stats — avoid multi-graph Text2Cypher
        # (schema×2 + brands often overflows the model and self-heals for many minutes).
        cross_stats = _try_cross_catalog_type_compare(
            platform,
            question=effective_question or question,
            intention=intention,
            product_ids=product_ids,
        )
        if cross_stats is not None:
            cross_stats.result_id = result_id
            state["active_filters"] = dict(intention.filters or {})
            state["group_by"] = intention.group_by
            state["last_result_id"] = result_id
            state["last_retrieved_text"] = cross_stats.text
            _remember_graph(state, product_ids, product_ids[0])
            if conversation_state is not None:
                conversation_state.clear()
                conversation_state.update(state)
            trace_step(
                "retrieve",
                "Cross-catalog type compare from kg_stats — skipped Text2Cypher",
                graphs=product_ids,
                filters=intention.filters,
            )
            return (
                cross_stats,
                [intention.intent, "kg_stats", "cross_catalog", *product_ids],
                intention,
                context,
                choice,
            )
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
    state["last_ontology_domains"] = list(
        ontology_domains or state.get("last_ontology_domains") or []
    )
    # Price-trend answer that asked for brand/buyer/category OR currency → keep pending
    # clarification so the next "1" / "USD" / "both" turn builds the chart.
    try:
        for item in getattr(retrieved, "evidence", None) or []:
            rid = getattr(item, "record_id", None)
            if rid not in {"price_trend_needs_scope", "price_trend_needs_currency"}:
                continue
            import json as _json

            payload = _json.loads(str(getattr(item, "excerpt", "") or "") or "{}")
            interps = list(payload.get("interpretations") or [])
            # Currency picks carry a currency code on each interpretation.
            for it in interps:
                if not isinstance(it, dict):
                    continue
                cur = str(it.get("currency") or "").strip().upper()
                rq = str(it.get("restated_question") or "")
                if cur and cur not in rq.upper():
                    who = (
                        (it.get("filters") or {}).get("brand")
                        or (it.get("filters") or {}).get("buyer")
                        or payload.get("who")
                        or "this brand"
                    )
                    if cur == "BOTH":
                        it["restated_question"] = (
                            f"Summarize average price change trends for {who} in both currencies"
                        )
                    else:
                        it["restated_question"] = (
                            f"Summarize average price change trends for {who} in {cur}"
                        )
            opts = list(payload.get("options") or [])
            if not interps and not opts:
                break
            state["clarification"] = {
                "pending_slot": (
                    "price_trend_currency"
                    if rid == "price_trend_needs_currency"
                    else "price_trend_scope"
                ),
                "question": str(getattr(retrieved, "text", "") or "").split("\n")[0],
                "options": opts or [
                    str(i.get("restated_question") or "") for i in interps if isinstance(i, dict)
                ],
                "interpretations": interps,
                "filters": dict(intention.filters or {}),
                "pending_metric": "price",
                "pending_rewritten_question": "Summarize average price change trends",
            }
            if QueryUnderstandingAgent.AWAITING_CAVEAT not in (retrieved.caveats or []):
                retrieved.caveats = list(retrieved.caveats or []) + [
                    QueryUnderstandingAgent.AWAITING_CAVEAT
                ]
            break
    except Exception:
        LOGGER.debug("price_trend clarify state wire failed", exc_info=True)
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
    _store_query_cache(
        platform,
        ask_for_cache,
        retrieved,
        ontology_domains=ontology_domains,
        graph_ids=list(product_ids or graph_ids or []),
        metric_family=state.get("last_metric_family") or metric_family,
        intent=str(intention.intent or ""),
    )
    if turn_plan is not None and bool(getattr(turn_plan, "llm_ok", False)):
        from fashion_brain.agents.turn_planner import next_state, scope_note

        turn_plan.effective_filters = dict(intention.filters or turn_plan.effective_filters)
        state.update(next_state(turn_plan, state))
        state["_response_scope_note"] = scope_note(turn_plan)
    if conversation_state is not None:
        conversation_state.clear()
        conversation_state.update(state)

    # Cypher/template findings are primary; attach only the selected filter meta
    # shards so ResponseAgent can ground the final reply with both.
    gid_for_meta = None
    try:
        gid_for_meta = (
            (product_ids[0] if product_ids else None)
            or graph_id
            or (choice.graph if choice else None)
            or (list(getattr(intention, "preferred_graphs", None) or []) or [None])[0]
        )
    except Exception:
        gid_for_meta = graph_id
    retrieved = _attach_selected_meta_reference(
        retrieved,
        graph_id=str(gid_for_meta) if gid_for_meta else None,
        question=effective_question or question,
        intention=intention,
        validated=validated,
        platform=platform,
    )

    routes = [intention.intent, *graph_ids]
    trace_step("pipeline", "Catalog orchestration complete", routes=routes)
    return retrieved, routes, intention, context, choice
