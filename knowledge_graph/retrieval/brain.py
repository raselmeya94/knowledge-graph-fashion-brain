"""Fashion Brain retrieval. Runs inside the knowledge-graph service."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from knowledge_graph.access import Principal, allows
from knowledge_graph.adapters.products import SqlProductConnector
from knowledge_graph.errors import ProductBackendUnavailable
from knowledge_graph.logging_config import get_logger
from knowledge_graph.models import EvidenceItem
from knowledge_graph.ontology import normalize_name
from knowledge_graph.pipeline.service import Platform
from knowledge_graph.retrieval.news import search_document_graphs
from knowledge_graph.retrieval.products import CATALOG_FIELDS, field_text
from knowledge_graph.retrieval.queries import product_materials, products_sharing_material

LOGGER = get_logger("knowledge_graph.retrieval.brain")
DEMO_CAVEAT = "These figures describe the synthetic demo products, not a live PLMEX database."
CATALOG_CAVEAT = ""


def _analysis_fields_for_graph(platform, graph_id: str) -> list[str] | None:
    """Unique analysis columns for the selected product KG (from metadata schema)."""

    if not graph_id:
        return None
    try:
        from knowledge_graph.retrieval.metadata import metadata_overview

        overview = metadata_overview(platform, graph_id)
    except Exception:
        return None
    schema = (overview or {}).get("schema") or {}
    product = schema.get("Product") if isinstance(schema, dict) else None
    attrs = list((product or {}).get("attributes") or {})
    preferred = [
        "brand",
        "buyer",
        "category",
        "family",
        "subfamily",
        "garment_type",
        "material",
        "color",
        "gender",
        "season",
        "department",
        "availability",
        "section",
        "price",
        "gauge",
        "gsm",
    ]
    present = {str(item).casefold() for item in attrs}
    chosen = [key for key in preferred if key in present]
    return chosen or None


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
    "ai_event_tags",
    "ai_construction_tags",
    "ai_material_tags",
    "ai_finish_tags",
    "ai_special_tags",
)
# Fields shown in a buyer / product analysis (top values per type).
_ANALYSIS_FIELDS = (
    "garment_type",
    "category",
    "subcategory",
    "plm_main_category",
    "material",
    "color",
    "gender",
    "season",
    "brand",
    "department",
    "gauge",
    "gsm",
    "ai_event_tags",
    "ai_construction_tags",
    "ai_material_tags",
    "ai_finish_tags",
    "ai_special_tags",
)
_SEASON_WORDS = {
    "winter": "winter",
    "winer": "winter",
    "autumn": "winter",
    "fall": "winter",
    "aw": "winter",
    "spring": "spring",
    "summer": "summer",
}
_UNNAMED_GROUPS = frozenset({"", "unspecified", "null", "none", "n/a", "na", "-"})
_FILTER_NOISE = frozenset(
    {
        "normally",
        "usually",
        "typical",
        "typically",
        "making",
        "make",
        "makes",
        "made",
        "type",
        "types",
        "kind",
        "kinds",
        "which",
        "what",
        "product",
        "products",
        "does",
        "new",
        "sample",
        "samples",
        "suggest",
        "specification",
        "specifications",
    }
)
_FILTER_LABELS = {field.key: field.label.casefold() for field in CATALOG_FIELDS}
_FILTER_LABELS["garment"] = "category or garment type"
_GROUP_LABELS = {field.key: field.label for field in CATALOG_FIELDS}


class BrainRetrieveRequest(BaseModel):
    question: str = ""
    intent: str
    filters: dict[str, str] = Field(default_factory=dict)
    group_by: str | None = None
    unsupported: bool = False
    limit: int | None = None
    suggest: bool = False
    analysis: bool = False
    season_name: str | None = None
    upcoming: bool = False
    corrections: list[str] = Field(default_factory=list)
    # Optional scope: one individual KG, or several. Empty = all active registry graphs.
    graph_id: str | None = None
    graph_ids: list[str] | None = None


class BrainRetrieveResponse(BaseModel):
    text: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    unavailable: list[str] = Field(default_factory=list)
    synthetic_data: bool = False
    routes_hint: list[str] = Field(default_factory=list)


def retrieve(platform: Platform, principal: Principal, request: BrainRetrieveRequest) -> BrainRetrieveResponse:
    """Build grounded evidence text for one Fashion Brain intention.

    Individual source KGs hold facts; ``shared`` interconnect joins similar entities.
    Optional ``graph_id`` / ``graph_ids`` scopes to one or more individual KGs.
    """

    try:
        from fashion_brain.trace import trace_debug, trace_step
    except Exception:  # pragma: no cover - KG-only process without fashion_brain
        def trace_step(phase: str, message: str, **fields):  # type: ignore[misc]
            LOGGER.info("brain retrieve [%s] %s %s", phase, message, fields)

        def trace_debug(message: str, **fields):  # type: ignore[misc]
            LOGGER.debug("brain retrieve %s %s", message, fields)

    _apply_buyer_spelling(platform, request)
    evidence: list[EvidenceItem] = []
    caveats: list[str] = []
    unavailable: list[str] = []
    lines: list[str] = []
    intent = request.intent
    scoped = _scoped_graphs(platform, request)
    product_ids = _graph_ids_of_kind(scoped, "product")
    document_ids = _graph_ids_of_kind(scoped, "documents") + _graph_ids_of_kind(scoped, "custom")
    # Product graphs come from kg_registry (kind=product). No hard-coded source ids.
    if not product_ids and not (request.graph_id or request.graph_ids):
        from knowledge_graph.retrieval.ecom import registry_product_graphs

        product_ids = registry_product_graphs(platform)
    if not document_ids and not (request.graph_id or request.graph_ids):
        document_ids = ["news"]
    trace_step(
        "brain",
        "Retrieve path selected",
        intent=intent,
        graph_id=request.graph_id,
        graph_ids=request.graph_ids,
        product_ids=product_ids,
        filters=request.filters,
        group_by=request.group_by,
        analysis=request.analysis,
        suggest=request.suggest,
    )

    if intent == "live_sql":
        try:
            SqlProductConnector(platform.settings.sql_enabled, platform.settings.sql_dsn).connect()
        except ProductBackendUnavailable as exc:
            unavailable.append(exc.message)
        lines.append("Live PLMEX analytics are unavailable. No count was estimated.")
    elif intent == "system_meta":
        from knowledge_graph.retrieval.system_meta import append_system_meta_answer

        trace_step("brain", "System / registry metadata answer")
        append_system_meta_answer(
            platform,
            lines,
            evidence,
            caveats,
            question=request.question or "",
        )
    elif intent in {"product_count", "specification"}:
        from knowledge_graph.retrieval.ecom import (
            append_neo4j_analysis_answer,
            append_neo4j_count_answer,
            default_product_graph,
            neo4j_product_count,
        )

        target = default_product_graph(platform, request.graph_id)
        neo_count = neo4j_product_count(platform, target) if target else 0
        has_neo4j = bool(target and neo_count > 0)
        trace_debug("Neo4j product target", target=target, neo_count=neo_count, has_neo4j=has_neo4j)
        if request.graph_id and request.graph_id not in product_ids and target is None:
            lines.append(f"Graph '{request.graph_id}' is not a registered product graph; no product count was estimated.")
            trace_step("brain", "Unknown product graph", graph_id=request.graph_id)
        elif has_neo4j and intent == "product_count" and request.analysis and not request.suggest:
            # Field-wise ask ("material wise") with analysis flag → only that dimension.
            if request.group_by:
                analysis_fields = [str(request.group_by)]
            else:
                analysis_fields = _analysis_fields_for_graph(platform, target)
            trace_step(
                "brain",
                "Neo4j analysis answer",
                graph_id=target,
                fields=analysis_fields,
                filters=request.filters,
            )
            append_neo4j_analysis_answer(
                platform,
                principal,
                target,
                dict(request.filters or {}),
                request.limit,
                lines,
                evidence,
                caveats,
                analysis_fields=analysis_fields,
            )
            if not request.graph_id:
                request.graph_id = target
        elif has_neo4j and intent == "product_count" and not request.suggest and not request.analysis:
            trace_step(
                "brain",
                "Neo4j count/group answer",
                graph_id=target,
                group_by=request.group_by,
                filters=request.filters,
            )
            append_neo4j_count_answer(
                platform,
                principal,
                target,
                dict(request.filters or {}),
                request.group_by,
                request.limit,
                lines,
                evidence,
                caveats,
            )
            if not request.graph_id:
                request.graph_id = target
        else:
            trace_step("brain", "Catalog/Postgres answer path", intent=intent, suggest=request.suggest)
            _append_catalog_answer(
                platform,
                principal,
                {
                    "filters": request.filters,
                    "group_by": request.group_by,
                    "unsupported": request.unsupported,
                    "limit": request.limit,
                    "analysis": request.analysis,
                    "suggest": intent == "specification" or request.suggest,
                    "season_name": request.season_name,
                    "upcoming": request.upcoming,
                    "corrections": request.corrections,
                },
                lines,
                evidence,
                caveats,
                unavailable,
            )
    elif intent == "shared_products":
        # Shared Material nodes across product graphs (ecom / own_products).
        # Prefer verified RESOLVES_TO when present; else group by Material node id.
        groups = products_sharing_material(platform, principal)
        if not groups:
            from knowledge_graph.retrieval.ecom import registry_product_graphs

            product_graphs = registry_product_graphs(platform) or ["own_products"]
            any_visible = False
            any_products = False
            for graph_id in product_graphs:
                nodes = platform.graph.find_nodes(graph=graph_id, label="Product", include_deleted=False)
                if nodes:
                    any_products = True
                if any(allows(principal, node.properties.get("classification") or "internal") for node in nodes):
                    any_visible = True
                    break
            if any_products and not any_visible:
                unavailable.append("Product records are not visible at the current clearance.")
            lines.append("No material is shared by more than one visible product.")
        else:
            verified = [group for group in groups if group.get("verified")]
            observed = [group for group in groups if not group.get("verified")]
            if verified:
                lines.append("Materials shared by more than one product (verified canonical link):")
                for group in verified[:25]:
                    sample = ", ".join(group["products"][:8])
                    more = f" (+{len(group['products']) - 8} more)" if len(group["products"]) > 8 else ""
                    lines.append(
                        f"- {group['material']} ({group['canonical_id']}): {len(group['products'])} products — {sample}{more}."
                    )
            if observed:
                lines.append("Materials shared by more than one product (same Material node in the product graph):")
                for group in observed[:25]:
                    sample = ", ".join(group["products"][:8])
                    more = f" (+{len(group['products']) - 8} more)" if len(group["products"]) > 8 else ""
                    lines.append(
                        f"- {group['material']} ({group['canonical_id']}): {len(group['products'])} products — {sample}{more}."
                    )
            if len(groups) > 25:
                lines.append(f"…and {len(groups) - 25} more shared materials.")
            caveats.append("Product ids are from the active product knowledge graph(s).")
    elif intent == "products":
        rows = product_materials(platform, principal)
        if not rows:
            hidden_products = platform.graph.find_nodes(graph="own_products", label="Product", include_deleted=False)
            if hidden_products:
                unavailable.append("Demo product records are not visible at the current clearance.")
            lines.append("No visible demo product materials were found.")
        else:
            shown = rows[:20]
            lines.append(f"Top {len(shown)} of {len(rows)} product material rows from the own_products graph:")
            review_names = set()
            for row in shown:
                canonical = row["canonical_id"] or "no verified canonical link"
                if row["review"]:
                    review_names.add(row["material"].properties.get("name"))
                    canonical = "review required; not verified"
                lines.append(
                    f"- {row['product'].properties.get('prov_record_id')}: {row['material'].properties.get('name')} ({canonical})."
                )
                evidence.append(_evidence_from_edge(row["edge"], row["product"], "USES_MATERIAL"))
            if review_names:
                caveats.append(
                    "These source materials are used by products but are not verified canonical matches: "
                    + ", ".join(sorted(name for name in review_names if name))
                    + "."
                )
            caveats.append(DEMO_CAVEAT)
    elif intent == "news":
        rows = search_document_graphs(platform, principal, request.question, graph_ids=document_ids)
        graphs_hit = sorted({str(row.get("graph") or "") for row in rows if row.get("graph") and not row.get("hidden_only")})
        if not rows:
            label = ", ".join(document_ids) if document_ids else "document"
            lines.append(f"No matching fact in graph(s): {label}.")
        elif rows[0].get("hidden_only"):
            unavailable.append(rows[0]["line"])
            lines.append(rows[0]["line"])
        else:
            label = ", ".join(graphs_hit) if graphs_hit else "document graphs"
            lines.append(f"Facts from {label}:")
            for row in rows:
                lines.append(f"- {row['line']}")
                item = _evidence_from_edge(row["edge"], row["start"], row["edge"].type)
                if row.get("graph"):
                    item.graph = str(row["graph"])
                evidence.append(item)
            if len(graphs_hit) > 1:
                caveats.append("This answer merges facts from multiple individual knowledge graphs.")
            else:
                caveats.append("This answer is from document/news graphs, not our PLM data.")
    elif request.unsupported:
        lines.append(_outside_catalog_message())
    else:
        materials = [
            node.properties.get("name")
            for node in platform.graph.find_nodes(graph="shared", label="Material", include_deleted=False)
        ]
        ontology_path = "schemas/product-knowledge-ontology.json"
        try:
            ontology_path = platform.kg_registry.ontology_file("product").as_posix()
        except Exception:
            pass
        lines.append(
            "Canonical materials on the shared interconnect: "
            + ", ".join(sorted(str(name) for name in materials if name))
            + "."
        )
        evidence.append(
            EvidenceItem(
                source_id="ontology",
                record_id=ontology_path,
                title="Shared / product ontology",
                excerpt="Declared canonical materials on the shared interconnect.",
                source_path=ontology_path,
                verification_status="verified",
                graph="shared",
                synthetic=True,
            )
        )

    products = getattr(platform, "products", None)
    envelopes = getattr(products, "envelopes", None) or []
    synthetic = any(getattr(envelope, "synthetic", False) for envelope in envelopes)
    hint = [intent]
    if request.graph_id:
        hint.append(request.graph_id)
    elif request.graph_ids:
        hint.extend(request.graph_ids)
    text = " ".join(lines)
    trace_step(
        "brain",
        "Retrieve response ready",
        chars=len(text),
        evidence=len(evidence),
        caveats=len(caveats),
        unavailable=len(unavailable),
        routes_hint=hint,
    )
    return BrainRetrieveResponse(
        text=text,
        evidence=evidence,
        caveats=caveats,
        unavailable=unavailable,
        synthetic_data=synthetic,
        routes_hint=hint,
    )


def _scoped_graphs(platform: Platform, request: BrainRetrieveRequest) -> list[Any]:
    try:
        active = list(platform.kg_registry.list_graphs())
    except Exception:
        return []
    wanted: set[str] = set()
    if request.graph_ids:
        wanted.update(item for item in request.graph_ids if item)
    if request.graph_id:
        wanted.add(request.graph_id)
    if not wanted:
        return active
    return [item for item in active if item.id in wanted]


def _graph_ids_of_kind(graphs: list[Any], kind: str) -> list[str]:
    return [item.id for item in graphs if getattr(item, "kind", None) == kind]


def _reliable_filter(value: str) -> bool:
    tokens = normalize_name(value).split()
    if not tokens or len(tokens) > 4:
        return False
    return not (set(tokens) & _FILTER_NOISE)


def _apply_buyer_spelling(platform: Platform, request: BrainRetrieveRequest) -> None:
    buyer = request.filters.get("buyer")
    if not buyer:
        return
    names = _catalog_names(platform, "buyer")
    if not names:
        return
    exact = next((name for name in names if normalize_name(name) == normalize_name(buyer)), None)
    if exact:
        request.filters["buyer"] = exact
        return
    nearest = _nearest_name(buyer, names)
    if nearest is None:
        return
    request.filters["buyer"] = nearest
    note = f"{buyer} is not a buyer name. Using the closest product buyer, {nearest}."
    request.corrections.append(note)


def _catalog_names(platform: Platform, field: str) -> list[str]:
    products = getattr(platform, "products", None)
    envelopes = getattr(products, "envelopes", None) or []
    found: dict[str, str] = {}
    for envelope in envelopes:
        metadata = getattr(envelope, "metadata", {}) or {}
        product = metadata.get("product", {}) if isinstance(metadata, dict) else {}
        raw = field_text(product, field).strip()
        if raw:
            found.setdefault(normalize_name(raw), raw)
    return list(found.values())


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


def _nearest_name(typed: str, names: list[str]) -> str | None:
    target = normalize_name(typed)
    if len(target) < 4 or not names:
        return None
    best_name = None
    best_distance = 99
    second = 99
    for name in names:
        folded = normalize_name(name)
        if folded == target:
            return name
        distance = _edit_distance(target, folded)
        if distance < best_distance:
            second = best_distance
            best_distance = distance
            best_name = name
        elif distance < second:
            second = distance
    limit = 2 if len(target) >= 6 else 1
    if best_name is None or best_distance > limit or second == best_distance:
        return None
    return best_name


def _append_catalog_answer(platform, principal, plan, lines, evidence, caveats, unavailable) -> None:
    if plan.get("unsupported"):
        lines.append(_outside_catalog_message())
        return
    _normalize_named_season(plan)
    if plan.get("analysis"):
        _append_analysis(platform, principal, plan, lines, evidence, caveats, unavailable)
        return
    if plan.get("suggest"):
        _append_specification(platform, principal, plan, lines, evidence, caveats, unavailable)
        return
    visible = platform.products.count(principal)
    hidden = len(platform.products.envelopes) - visible
    filters = _query_filters(plan)
    group_by = plan.get("group_by")
    if not visible:
        lines.append("There are 0 products matching this request.")
    elif filters or group_by:
        matched, total = platform.products.search(principal, limit=5, **filters)
        phrase = _filter_phrase(filters)
        lines.append(f"There are {total} products" + (f" {phrase}." if phrase else "."))
        if group_by:
            groups = platform.products.count_by(principal, group_by, **filters)
            if plan.get("limit"):
                lines.extend(_ranking_lines(groups, _GROUP_LABELS.get(group_by, group_by), int(plan["limit"])))
            else:
                lines.append(f"{_GROUP_LABELS.get(group_by, group_by)} counts for that selection: {_group_text(groups)}.")
        for envelope in matched:
            evidence.append(_evidence_from_product(envelope))
    else:
        lines.append(f"There are {visible} products.")
    if CATALOG_CAVEAT:
        caveats.append(CATALOG_CAVEAT)
    if hidden:
        unavailable.append(f"{hidden} products are hidden by access control.")


def _normalize_named_season(plan: dict) -> None:
    filters = plan.setdefault("filters", {})
    mapped = _SEASON_WORDS.get(normalize_name(str(filters.get("season") or "")))
    if mapped:
        filters.pop("season", None)
        plan["season_name"] = plan.get("season_name") or mapped
    garment = filters.get("garment")
    if garment and not _reliable_filter(str(garment)):
        filters.pop("garment", None)


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
    garment = filters.get("garment")
    if garment and not _reliable_filter(str(garment)):
        filters.pop("garment", None)
    return filters


def _profile_matches(platform, principal, filters: dict) -> tuple[int, dict, Any]:
    products = platform.products
    code = products.prepare_filters(principal, filters).get("season_code") if hasattr(products, "prepare_filters") else None
    total, ranked = products.profile(principal, _SPEC_FIELDS, **filters)
    return total, ranked, code


def _usable_spec_value(value: str) -> bool:
    folded = value.strip().casefold()
    if folded in _UNNAMED_GROUPS:
        return False
    return re.fullmatch(r"0+(?:\.0+)?", folded.replace(" ", "")) is None


def _append_analysis(platform, principal, plan, lines, evidence, caveats, unavailable) -> None:
    """Type-wise top rankings for a buyer’s past products — not a suggested new spec."""

    filters = _query_filters(plan)
    filters.pop("upcoming", None)
    filters.pop("season_code", None)
    for note in plan.get("corrections") or []:
        lines.append(note)
    buyer = filters.get("buyer")
    top_n = int(plan.get("limit") or 5)
    total, ranked = platform.products.profile(principal, _ANALYSIS_FIELDS, **filters)
    if not total:
        who = buyer or "this selection"
        lines.append(f"There are no products for {who} to analyse.")
        if CATALOG_CAVEAT:
            caveats.append(CATALOG_CAVEAT)
        return
    who = buyer or "all buyers"
    lines.append(f"Product analysis for {who}: {total} products.")
    lines.append(f"Top {top_n} values by type (highest count first).")
    chart_fields = {"material", "color", "gender", "availability"}
    for field in _ANALYSIS_FIELDS:
        choices = [item for item in (ranked.get(field) or []) if _usable_spec_value(item[0])][:top_n]
        if not choices:
            continue
        label = _GROUP_LABELS.get(field, field.replace("_", " "))
        lines.append(f"### {label}")
        if field in chart_fields:
            chart_type = "pie" if field in {"gender", "material", "color"} else "bar"
            payload = {
                "type": chart_type,
                "title": f"{who} — {label}",
                "labels": [name for name, _count in choices],
                "values": [count for _name, count in choices],
            }
            lines.append("```chart")
            lines.append(json.dumps(payload, ensure_ascii=False))
            lines.append("```")
            top_name, top_count = choices[0]
            share = round(100.0 * top_count / total, 1) if total else 0.0
            lines.append(f"Leading value: {top_name} ({top_count}, {share}%).")
        else:
            lines.append("| Name | Count | Share % |")
            lines.append("| --- | ---: | ---: |")
            for name, count in choices:
                share = round(100.0 * count / total, 1) if total else 0.0
                lines.append(f"| {name} | {count} | {share} |")
    matched, _total = platform.products.search(principal, limit=5, **filters)
    for envelope in matched:
        evidence.append(_evidence_from_product(envelope))
    if CATALOG_CAVEAT:
        caveats.append(CATALOG_CAVEAT)


def _append_specification(platform, principal, plan, lines, evidence, caveats, unavailable) -> None:
    filters = _query_filters(plan)
    for note in plan.get("corrections") or []:
        lines.append(note)
    buyer = filters.get("buyer") or "our products"
    buyer_only = {key: value for key, value in filters.items() if key not in {"season_name", "season_code", "upcoming"}}
    buyer_total = platform.products.count(principal, **buyer_only)
    lines.append(f"Step 1. Buyer {buyer}: {buyer_total} products.")
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
            lines.append(
                f"No {season_name} style codes were stored, so the specification uses all {total} products for that buyer."
            )
            scope = f"all {total} products for that buyer"
    if not total:
        lines.append(f"There are no styles for {buyer}, so there is no specification to build.")
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
    lines.append(
        "Each count is how many of the filtered products use that value. The sheet is drawn from those products only."
    )
    if CATALOG_CAVEAT:
        caveats.append(CATALOG_CAVEAT)


def _ranking_lines(groups: list[tuple[str, int]], label: str, limit: int) -> list[str]:
    named = [(name, count) for name, count in groups if name.strip().casefold() not in _UNNAMED_GROUPS]
    missing = [(name, count) for name, count in groups if name.strip().casefold() in _UNNAMED_GROUPS]
    named.sort(key=lambda item: (-item[1], item[0].casefold()))
    shown = named[: max(limit, 1)]
    lines = [f"Top {len(shown)} {label.casefold()} by product count:"]
    lines.extend(f"{index}. {name}: {count}" for index, (name, count) in enumerate(shown, start=1))
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
    return (
        "No count was estimated. Ask about product counts or a breakdown by any stored field: "
        + names
        + "."
    )


def _filter_phrase(filters: dict[str, str]) -> str:
    parts = [f"{_FILTER_LABELS[key]} {value}" for key, value in filters.items() if value and key in _FILTER_LABELS]
    if not parts:
        return ""
    return "with " + " and ".join(parts)


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
    return EvidenceItem(
        source_id=str(edge.properties.get("prov_source_id") or ""),
        record_id=str(edge.properties.get("prov_record_id") or ""),
        title=node.properties.get("title") or node.properties.get("name"),
        excerpt=str(edge.properties.get("prov_evidence_text") or ""),
        field_ref=edge.properties.get("prov_field_ref"),
        section_ref=edge.properties.get("prov_section_ref"),
        source_path=edge.properties.get("prov_source_path"),
        url=edge.properties.get("prov_url"),
        content_hash=edge.properties.get("prov_content_hash"),
        verification_status=str(edge.properties.get("prov_verification_status") or "observed"),
        graph=edge.graph,
        node_id=node.id,
        relationship=relationship,
        synthetic=bool(edge.properties.get("prov_synthetic", True)),
    )
