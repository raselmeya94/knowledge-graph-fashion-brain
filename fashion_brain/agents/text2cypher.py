"""LLM Text-to-Cypher agent: schema + resolved entity ids → read Cypher → findings.

Templated helpers in ``knowledge_graph.retrieval.ecom`` remain as fallback when this
agent cannot produce a safe executable query.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from knowledge_graph.logging_config import get_logger
from knowledge_graph.models import EvidenceItem
from knowledge_graph.retrieval.cypher_safe import assert_read_only_cypher, rows_to_findings, run_read_cypher

from fashion_brain.agents.contracts import Intention, Retrieved
from fashion_brain.llm import build_model_provider

LOGGER = get_logger("fashion_brain.agents.text2cypher")

TEXT2CYPHER_SYSTEM = """\
You are Text2Cypher for Fashion Brain's Neo4j product knowledge graph.

Return ONE JSON object only (no markdown fence, no prose):
{"cypher":"MATCH ... RETURN ...","params":{"g":"<graph_id>"},"result_kind":"count|rows|breakdown","explanation":"one short sentence"}

Hard rules:
1. READ-ONLY: MATCH / OPTIONAL MATCH / WITH / WHERE / RETURN / ORDER BY / LIMIT / UNWIND only.
2. Always scope products with graph param: (p:Entity:Product {graph:$g}) and coalesce(p.lifecycle,'active') <> 'deleted'.
3. NEVER invent node names or guess spellings. Use ONLY ids / values from <resolved_entities> and <brands>.
4. Facet matching MUST use node id lists, e.g. (p)-[:USES_MATERIAL]->(m:Entity) WHERE m.id IN $material_ids
   Same for category (BELONGS_TO_CATEGORY / HAS_SUB_CATEGORY / HAS_FAMILY), color (HAS_COLOR), availability (HAS_AVAILABILITY).
5. Brand: use live brand name from <brands> with toLower(coalesce(p.brand,'')) = $brand OR HAS_BRAND id IN $brand_ids.
6. If a resolved field has status no_match, do not invent a substitute — return unsupported.
7. Prefer p.price_num for numeric price filters when filtering by price; fallback:
   toFloat(replace(replace(replace(replace(replace(coalesce(p.price,''),'€',''),'$',''),'£',''),',',''),' ',''))
8. Parameterize every filter. Always include "g" in params. Put resolved id lists in params (category_ids, material_ids, …).
9. Always RETURN specific properties or aggregates — never RETURN whole nodes. Always include LIMIT (default 5–15 for listings).
10. When a product can match multiple edges, use count(DISTINCT p).
11. Parent-category asks: DISTINCT parent names via PARENT_CATEGORY, LIMIT 10.
12. Do not compare facet .name to raw literals; use .id IN $ids (or normalized_name only if ids missing).
13. If extracted.count_entity is brand|material|color|category, count those facet nodes
    (e.g. MATCH (b:Entity:Brand {graph:$g}) WHERE coalesce(b.lifecycle,'active') <> 'deleted' RETURN count(b) AS total)
    — do NOT route to shared interconnect and do NOT invent zero brands.
14. If the question cannot be answered, return {"cypher":"","params":{},"result_kind":"unsupported","explanation":"..."}.
15. Copy entity labels EXACTLY from the schema card (e.g. GarmentCategory, never GamentCategory). Never invent or misspell labels.
16. Do not invent category/brand filters that are absent from <resolved_entities> / <extracted>.
"""


@dataclass
class CypherPlan:
    cypher: str
    params: dict[str, Any] = field(default_factory=dict)
    result_kind: str = "rows"
    explanation: str = ""
    source: str = "llm"


_SKIP_INTENTS = frozenset({"system_meta", "live_sql", "news", "shared", "shared_products"})


class Text2CypherAgent:
    """Schema-aware LLM that writes Neo4j Cypher for product catalog questions."""

    def applicable(self, intention: Intention) -> bool:
        if intention.unsupported:
            return False
        return intention.intent not in _SKIP_INTENTS

    def schema_card(self, platform, graph_id: str, *, brands: list[dict] | None = None) -> str:
        try:
            from fashion_brain.planning.runtime_schema import get_runtime_schema

            ctx = get_runtime_schema(platform, graph_id)
            card = ctx.schema_card()
        except Exception:
            card = f"Target graph_id: {graph_id}\nUse only labels/rels from <resolved_entities> and ontology."
        brand_block = ""
        if brands:
            brand_block = "Live brands:\n" + "\n".join(
                f"- id={b.get('id')} name={b.get('name')} normalized_name={b.get('normalized_name')}" for b in brands
            )
        return card + ("\n" + brand_block if brand_block else "")

    def generate(
        self,
        platform,
        question: str,
        intention: Intention,
        graph_id: str,
        *,
        resolved: dict[str, Any] | None = None,
        extracted: dict[str, Any] | None = None,
        brands: list[dict] | None = None,
    ) -> CypherPlan | None:
        if platform.settings.answer_mode != "llm":
            return None
        if not getattr(platform.settings, "text2cypher_enabled", True):
            return None
        provider = build_model_provider(platform.settings)
        if brands is None:
            try:
                from knowledge_graph.linking.entity_linker import list_live_brands

                brands = list_live_brands(platform, graph_id)
            except Exception:
                brands = []
        schema = self.schema_card(platform, graph_id, brands=brands)
        intent_bits = {
            "intent": intention.intent,
            "filters": intention.filters or {},
            "group_by": intention.group_by,
            "limit": intention.limit,
            "analysis": bool(getattr(intention, "analysis", False)),
            "suggest": bool(intention.suggest),
            "count_entity": getattr(intention, "count_entity", None)
            or (extracted or {}).get("count_entity"),
        }

        # Seed params from resolved ids so the model has concrete lists.
        seeded_params: dict[str, Any] = {"g": graph_id}
        for field, payload in (resolved or {}).items():
            if not isinstance(payload, dict):
                continue
            ids = payload.get("ids") or []
            if ids:
                seeded_params[f"{field}_ids"] = list(ids)
            if field == "brand" and payload.get("names"):
                seeded_params["brand"] = str(payload["names"][0]).casefold()
        if extracted:
            if extracted.get("price_max") is not None:
                seeded_params["price_max"] = extracted["price_max"]
            if extracted.get("price_min") is not None:
                seeded_params["price_min"] = extracted["price_min"]
            if extracted.get("currency"):
                seeded_params["currency"] = extracted["currency"]
            if extracted.get("limit") is not None:
                seeded_params["limit"] = extracted["limit"]

        prompt = (
            f"<schema>\n{schema}\n</schema>\n"
            f"<brands>\n{json.dumps(brands or [], ensure_ascii=False)}\n</brands>\n"
            f"<resolved_entities>\n{json.dumps(resolved or {}, ensure_ascii=False)}\n</resolved_entities>\n"
            f"<extracted>\n{json.dumps(extracted or {}, ensure_ascii=False)}\n</extracted>\n"
            f"<intention>\n{json.dumps(intent_bits, ensure_ascii=False)}\n</intention>\n"
            f"<seed_params>\n{json.dumps(seeded_params, ensure_ascii=False)}\n</seed_params>\n"
            f"<question>\n{question}\n</question>\n"
            "Write the Cypher JSON now. Reuse seed_params id lists."
        )
        try:
            from fashion_brain.stage_timing import timed

            with timed("text2cypher_llm", graph_id=graph_id):
                raw = provider.complete(
                    prompt,
                    timeout=_llm_timeout(platform.settings.request_timeout_seconds),
                    system=TEXT2CYPHER_SYSTEM,
                )
        except Exception as exc:
            LOGGER.warning("text2cypher LLM failed: %s", exc)
            return None
        plan = _parse_plan(raw, default_graph=graph_id)
        if plan is None or not plan.cypher.strip():
            return None
        # Merge seeded params under model params (model wins on conflict only if non-empty).
        merged = dict(seeded_params)
        merged.update(plan.params or {})
        plan.params = merged
        try:
            plan.cypher = assert_read_only_cypher(plan.cypher)
        except ValueError as exc:
            LOGGER.warning("text2cypher rejected unsafe Cypher: %s", exc)
            return None
        if not _labels_match_schema(platform, graph_id, plan.cypher):
            LOGGER.warning("text2cypher rejected Cypher with unknown labels: %s", plan.cypher[:200])
            return None
        plan.params.setdefault("g", graph_id)
        return plan

    def retrieve(
        self,
        platform,
        principal,
        question: str,
        intention: Intention,
        graph_id: str,
        *,
        resolved: dict[str, Any] | None = None,
        extracted: dict[str, Any] | None = None,
        brands: list[dict] | None = None,
    ) -> Retrieved | None:
        """Generate + execute Cypher. Return Retrieved, or None to fall back to templates."""

        from fashion_brain.trace import trace_debug, trace_step

        if not self.applicable(intention):
            return None
        if not getattr(platform.settings, "text2cypher_enabled", True):
            return None

        plan = self.generate(
            platform,
            question,
            intention,
            graph_id,
            resolved=resolved,
            extracted=extracted,
            brands=brands,
        )
        if plan is None:
            trace_step("text2cypher", "No LLM Cypher — templated fallback")
            return None

        trace_step(
            "text2cypher",
            "LLM Cypher ready",
            result_kind=plan.result_kind,
            params=list(plan.params.keys()),
            cypher=plan.cypher[:240],
        )
        try:
            rows = _execute(platform, principal, plan)
        except Exception as exc:
            LOGGER.warning("text2cypher execute failed: %s", exc)
            trace_step("text2cypher", "Execute failed — templated fallback", error=str(exc)[:200])
            return None

        text = rows_to_findings(rows, explanation=plan.explanation, cypher=plan.cypher)
        evidence = [
            EvidenceItem(
                source_id=graph_id,
                record_id="text2cypher",
                title="LLM Text-to-Cypher",
                excerpt=plan.cypher[:400],
                source_path="neo4j",
                verification_status="observed",
                graph=graph_id,
                node_id=None,
                synthetic=False,
            )
        ]
        trace_debug("text2cypher rows", count=len(rows), chars=len(text))
        caveats = ["Retrieved via LLM Text-to-Cypher (schema-grounded)."]
        if resolved:
            caveats.append("Entity-linked ids used for facet filters.")
        return Retrieved(
            text=text,
            evidence=evidence,
            caveats=caveats,
            unavailable=[],
            synthetic_data=False,
        )


def _execute(platform, principal, plan: CypherPlan) -> list[dict]:
    from fashion_brain.execution.kg_client import kg_request, uses_kg_api

    if uses_kg_api(platform.settings):
        clearance = getattr(principal, "clearance", None) or "internal"
        payload = kg_request(
            platform.settings,
            "POST",
            "/brain/cypher",
            clearance=clearance,
            json_body={"cypher": plan.cypher, "params": plan.params, "limit_rows": 200},
        )
        if isinstance(payload, dict) and payload.get("error"):
            raise RuntimeError(str(payload.get("error")))
        rows = (payload or {}).get("rows") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise RuntimeError("KG /brain/cypher returned no rows list.")
        return rows
    return run_read_cypher(platform, plan.cypher, plan.params)


def _llm_timeout(seconds: float | None) -> float | None:
    if seconds is None or seconds <= 0:
        return 45.0
    return min(45.0, float(seconds))


def _parse_plan(raw: str, *, default_graph: str) -> CypherPlan | None:
    from fashion_brain.ask import _extract_json_object

    data = _extract_json_object(raw or "")
    if not isinstance(data, dict):
        match = re.search(r"\bMATCH\b[\s\S]+?\bRETURN\b[\s\S]+", raw or "", re.IGNORECASE)
        if not match:
            return None
        return CypherPlan(cypher=match.group(0).strip(), params={"g": default_graph}, result_kind="rows")
    if str(data.get("result_kind") or "").casefold() == "unsupported":
        return None
    cypher = str(data.get("cypher") or "").strip()
    if not cypher:
        return None
    params = data.get("params") if isinstance(data.get("params"), dict) else {}
    clean_params: dict[str, Any] = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean_params[str(key)] = value
        elif isinstance(value, list) and all(isinstance(x, (str, int, float, bool)) for x in value):
            clean_params[str(key)] = value
    clean_params.setdefault("g", default_graph)
    return CypherPlan(
        cypher=cypher,
        params=clean_params,
        result_kind=str(data.get("result_kind") or "rows"),
        explanation=str(data.get("explanation") or "").strip(),
        source="llm",
    )


def _labels_match_schema(platform, graph_id: str, cypher: str) -> bool:
    """Reject Cypher that uses entity labels not present in RuntimeSchemaContext.

    Allows structural labels (Entity) and common Cypher noise; catches typos like
    ``GamentCategory`` instead of ``GarmentCategory``.
    """

    try:
        from fashion_brain.planning.runtime_schema import get_runtime_schema

        ctx = get_runtime_schema(platform, graph_id)
    except Exception:
        return True  # fail open if schema unavailable
    allowed = {
        "Entity",
        "SourceEntity",
        "Canonical",
        "Record",
        str(ctx.product_label or "Product"),
        *(ctx.entity_types or []),
        *(ctx.facet_types or []),
    }
    allowed_cf = {x.casefold() for x in allowed if x}
    # All :Token hits (node labels and relationship types).
    found = set(re.findall(r":([A-Za-z][A-Za-z0-9_]*)", cypher or ""))
    # Relationship types inside [:REL] or [:A|B] — not entity labels.
    rel_parts: set[str] = set()
    for blob in re.findall(r"\[:([^\]]+)\]", cypher or ""):
        for part in re.split(r"[|]", blob):
            token = part.strip().split("{")[0].strip()
            if token and re.match(r"^[A-Za-z][A-Za-z0-9_]*$", token):
                rel_parts.add(token)
    unknown = {
        lab
        for lab in found
        if lab.casefold() not in allowed_cf and lab not in rel_parts
    }
    if unknown:
        LOGGER.info("text2cypher unknown labels=%s allowed_sample=%s", sorted(unknown), sorted(allowed)[:12])
        return False
    return True


def _load_ontology(platform, graph_id: str) -> dict:
    try:
        from fashion_brain.execution.kg_client import kg_request, uses_kg_api

        ontology_id = None
        if uses_kg_api(platform.settings):
            reg = kg_request(platform.settings, "GET", "/api/graphs/registry", clearance="internal")
            for row in (reg or {}).get("registry") or (reg or {}).get("graphs") or []:
                gid = str(row.get("graph_id") or row.get("id") or "")
                if gid == graph_id:
                    ontology_id = row.get("ontology_id")
                    break
            if ontology_id:
                payload = kg_request(
                    platform.settings,
                    "GET",
                    f"/ontologies/{ontology_id}",
                    clearance="internal",
                )
                definition = (payload or {}).get("definition") if isinstance(payload, dict) else None
                if isinstance(definition, dict):
                    return definition
    except Exception:
        LOGGER.debug("text2cypher ontology via KG failed", exc_info=True)

    try:
        if hasattr(platform, "kg_registry"):
            for item in platform.kg_registry.list_graphs():
                if getattr(item, "id", None) == graph_id:
                    ontology_id = getattr(item, "ontology_id", None)
                    if ontology_id:
                        record = platform.kg_registry.load_ontology_object(ontology_id)
                        if hasattr(record, "model_dump"):
                            return record.model_dump()
                        if isinstance(record, dict):
                            return record
                    break
    except Exception:
        LOGGER.debug("text2cypher local ontology failed", exc_info=True)

    # No file fallback — agents must use RuntimeSchemaContext / registry only.
    return {}
