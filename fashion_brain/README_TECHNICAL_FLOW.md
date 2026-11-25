# Fashion Brain — Technical Flow (Question → Response)

This document describes **every major decision** from a user question to the final chat reply: entrypoints, routing, conversation context, query understanding, retrieval, and ResponseAgent.

Related: [`README_KG_RAG.md`](./README_KG_RAG.md) (Fashion Brain vs Knowledge Graph boundary).

---

## 1. System boundary

| Layer | Role |
|-------|------|
| **Fashion Brain** (`:8000`) | Chat orchestration, LLM agents, Text2Cypher prompts, answer prose |
| **Knowledge Graph** (`:8001` / Neo4j) | Schema, brands, Cypher execution, `kg_stats`, product data |

```
User UI / API / WebSocket
        │
        ▼
Fashion Brain  (agents, context, Cypher generation, prose)
        │
        ▼
KG API / Neo4j (facts only — no LLM answer writing)
        │
        ▼
Fashion Brain ResponseAgent → GroundedAnswer
```

---

## 2. Entry points

| Entry | Module | Notes |
|-------|--------|-------|
| HTTP `POST /ask` | `fashion_brain/api/app.py` | Hydrates **`session_state`** (same as WS when `conversation_id` is set) |
| WebSocket `/ws/ask` | same | Keeps **`session_state`** across turns (filters, `last_graph_id`, Cypher, clarification) |
| REST chat `/api/v1/chat` · `/v1/conversations/.../messages` | same | Same `session_state` + `answer_question` → catalog |
| CLI `python -m fashion_brain.cli serve` | `cli.py` | Serves the API |
| Core function | `fashion_brain.ask.answer_question` | Single entry for all answer modes |

### 2.1 Top-level `answer_question` decision

```
answer_question(question, history, graph_id?, conversation_state?)
    │
    └─ always → answer_from_catalog(... conversation_state or {})
         # Same catalog orchestrator for HTTP /ask, WS, eval, and conversation messages.
         # LangGraph ask_with_graph is not used on this entry (kept for explicit/dev callers).
```

**Important:** all product-chat entry points share the catalog orchestrator so sticky graph + filters and answer quality stay aligned.

---

## 3. Outer LangGraph router (when used)

File: `fashion_brain/flow.py` → `ask_with_graph`

```
START → router → { catalog | web_search | direct | sample_design | news } → END
```

| Route | When |
|-------|------|
| `sample_design` | Slash / garment sample design ask |
| `direct` | Short rewrite / follow-up that does not need KG (or forced) |
| `news` | News-style planned route |
| `catalog` | Product/KG domain (schema-grounded or planned PLM/ecom route) |
| `web_search` | Public facts when web tools enabled |

Catalog branch calls `answer_from_catalog` → same inner pipeline as below.

---

## 4. Catalog pipeline (main path)

Files:

- `ask.answer_from_catalog` → `ask.draft_catalog` → `orchestration/catalog_pipeline.draft_catalog_orchestrated`
- Then `ResponseAgent.compose` (except clarifications and finalized simple counts)

### 4.1 High-level sequence

```
1. ConversationContextAgent.analyze
2. Align session active_filters to context decision
3. Conservative kg_stats preflight for stateless basic counts / facet rankings
     → on a metadata hit, return before Turn Planner, Cypher, and response LLM
4. TurnPlanner (otherwise; one structured LLM call when available)
5. Branch:
     a. REUSE prior findings          → skip KG
     b. QueryUnderstandingAgent       → ClarifyResult | ReadyResult
     c. Post-understanding kg_stats   → skip Thinking / Cypher
6. ThinkingBrain (when history / follow-up / Text2Cypher off)
7. IntentionAgent + QueryPlanner
8. Session graph memory (sticky last_graph_id)
9. apply_turn_constraints (filter carry / rewrite)
10. Second kg_stats chance
11. GraphRouterAgent
12. Retrieve:
      analytical_tool | Text2Cypher | template fallback | multi-graph merge | query cache
13. Persist conversation_state
14. ResponseAgent.compose → GroundedAnswer (except finalized simple counts)
```

---

## 5. Conversation context (history vs current ask)

File: `fashion_brain/agents/conversation_context.py`  
Policy: current question wins; history is **not** permanent filters.

### 5.1 `intent_type` (exactly one)

| `intent_type` | Meaning | Filter carry |
|---------------|---------|--------------|
| `continuation` | Continues prior scope (“of them”, “and how many are red”) | **Yes** — keep relevant prior filters |
| `modification` | Same analysis, change a condition (“Zara instead”, “overall Primark or Zara”) | Partial — replace/remove conflicting facets |
| `new_independent_question` | Standalone ask | **No** — filters only from current question |
| `partial_context_reuse` | New task reusing a named party (“full analysis of Primark”) | Party only — drop garment/color/material/… |

### 5.2 `decision`

| Decision | Effect |
|----------|--------|
| `RETRIEVE` | Run QU + retrieval (normal) |
| `REFINE` | Prefer new retrieval with refined ask (falls back to RETRIEVE if no prior text) |
| `REUSE` | Return `last_retrieved_text` without Neo4j (blocked while clarification pending) |

### 5.3 Deterministic guards (after LLM/heuristic parse)

| Guard | Forces |
|-------|--------|
| `_guard_metric_shift` | Metric family change (price ↔ availability ↔ catalog) → RETRIEVE, drop carry |
| `_guard_standalone_filters` | New named party / open listing without deixis → `new_independent_question` |
| `_guard_broader_scope_reset` | Overall / full analysis → `partial_context_reuse`, no facet carry |
| `_guard_catalog_dataset_total` | “How many products in the e-comm/PLM data?” → `new_independent_question`, clear brand |

### 5.4 Variation matrix — history / connection

| Situation | Typical `intent_type` | Filters | Graph |
|-----------|----------------------|---------|-------|
| **Empty history**, empty session | `new_independent_question` | From current ask only | From question / router |
| **History exists**, first unrelated ask | `new_independent_question` | No inherit | May still remember `last_graph_id` only if sticky rules apply |
| **Connected follow-up** (“of them”, “those”) | `continuation` | Inherit brand/buyer (+ facets as needed) | Sticky `last_graph_id` |
| **Facet follow-up** (“breakdown by color”) | `continuation` | Inherit party; set `group_by` | Sticky |
| **New brand named** (“How many does Zara have?”) | `new_independent_question` | New brand only | May switch if ask names catalog |
| **Catalog/dataset total** (“products in the e-comm data”) | `new_independent_question` | **Clear prior brand** | Prefer named catalog (`ecom_products` / `plmex_products`) |
| **Full analysis of same brand** | `partial_context_reuse` | Brand only | Sticky unless catalog named |
| **Clarification answer** (pending slot) | Continues clarify thread | Confirmed filters accumulate | Unchanged |
| **Metric shift** (was price, now count) | Forced retrieve / no carry | Reset restrictive carry | Sticky graph may remain |

**Rule of thumb**

- **Filter stickiness** ≠ **graph stickiness**
- Graph can stay on `ecom_products` while filters are cleared for a new independent total.

---

## 6. Query Understanding (QU)

File: `fashion_brain/agents/query_understanding.py`

### 6.1 Inputs

- Current question
- `ContextDecision` (especially `intent_type` / `active_filters`)
- `conversation_state` (`active_filters`, `clarification`, …)

### 6.2 Filter seed rules

```
if dataset_total OR intent_type == new_independent_question:
    do not seed confirmed from session brand
elif context.active_filters present:
    seed confirmed from context.active_filters
elif deixis (no context object):
    seed from session
else:
    session filters cleared
```

### 6.3 Outputs

| Result | Meaning | Next step |
|--------|---------|-----------|
| `ClarifyResult` | Need entity / confirm / depth | Return clarify text; **skip** ResponseAgent rewrite; persist `clarification` |
| `ReadyResult` (`ValidatedIntent`) | Ready to execute | Continue pipeline |

### 6.4 `ValidatedIntent` fields that matter

| Field | Role |
|-------|------|
| `filters` | Authoritative scope when set |
| `metrics` | e.g. `["count"]` unlocks `kg_stats` fast path |
| `analysis_depth` | `full` / `focused` / `unset` |
| `source` | Named graph id (`ecom_products`, `plmex_products`, …) |
| `effective_question` | Resolved analytical wording for downstream |
| `group_by` | Breakdown dimension |

### 6.5 Clarification slots (progressive analysis)

Typical order for vague analysis:

1. `entity` — which brand/buyer  
2. `final_confirm` — confirm filters / full analysis  
3. (optional) focused dimension  

While `pending_slot` is set, REUSE of prior findings is disabled.

---

## 7. ThinkingBrain + Intention + Planner

### 7.1 ThinkingBrain (`agents/_impl.py`)

Runs when:

- History present, or Text2Cypher disabled, or orchestrator needs a summary for UI/planning

Produces `IntentSummary`:

- `actual_intent`, `filters`, `group_by`
- `references_prior_chat` — **must not** be set merely because `last_graph_id` exists
- Session glue (`_apply_session_scope`) only on **deixis** or **pending clarification**
- Dataset totals strip sticky brand/buyer

UI “Thinking” text is rendered from this summary.

### 7.2 IntentionAgent

Builds `Intention` (intent name, filters, preferred graphs, analysis flags).  
`carry` / `references_prior_chat` controls whether the catalog planner may merge history.

### 7.3 QueryPlanner (`planning/query_planner.py`)

Fills when missing:

| Field | Examples |
|-------|----------|
| `retrieve_mode` | `text2cypher` \| `analytical_tool` \| `template` |
| `response_type` | `stats` \| `analysis_report` \| `comparison` \| `trend` \| `specs` \| `concise` |
| `metrics` / `operations` / `query_shape` | count, entity_frequency, shared_overlap, … |

### 7.4 Session graph memory (`_apply_session_graph_memory`)

- Pins `preferred_graphs` to `last_graph_id` on continuous chats
- **Exception:** user names another catalog, or cross-source compare language
- Does **not** invent brand filters

### 7.5 Turn constraints (`planning/active_constraints.py`)

```
carry_now = context.is_contextual AND context.active_filters  (and not full_analysis)

if carry_now:
    merge prior active_filters ⊕ intention filters
    optionally rewrite question with REQUIRED filters
else:
    use intention filters only (no sticky brand leak)
```

---

## 8. Retrieval decision tree

### 8.1 Fast paths (skip Text2Cypher / Neo4j when possible)

```
metadata preflight (eligible stateless ask)?
  │
  ├─ 0–1 filter or facet ranking + kg_stats hit
  │     → reuse the validated stats result for formatting
  │       → no Turn Planner, Cypher, optional-metadata reload, or response LLM
  │
  └─ miss / ambiguous scope → continue to the existing planner

validated Ready?
  │
  ├─ metrics includes count AND 0–1 filters AND kg_stats hit
  │     → append_neo4j_count_answer / kg_stats     [EARLY META]
  │
  ├─ after Intention: same kg_stats chance again  [SECOND META]
  │
  ├─ short-TTL query cache hit
  │     → return cached findings
  │
  └─ else continue to GraphRouter + retrieve modes
```

**`kg_stats` contract**

- Answers simple totals / single-filter counts from `data/generated/kg_stats/{graph}.json`
- Multi-filter or uncached dimension → cache miss → Cypher / templates
- Must **not** merge stale session brand unless contextual carry is on
- Reuses the validated metadata result rather than reading the count shard again
- Direct count replies do not load optional facet metadata or invoke the response LLM
- Brand/buyer facet rankings restore display labels only on unique ontology name/alias matches; stored keys and counts remain unchanged

### 8.2 GraphRouterAgent

Chooses `selected_graphs` from intention + registry.  
Insufficient coverage → early empty findings.

### 8.3 Retrieve modes

| Mode | When | Backend |
|------|------|---------|
| `analytical_tool` | Full analysis / planner says so | `analytics.engine.run_product_analysis` |
| `text2cypher` | Default LLM KG path | Text2CypherAgent → Neo4j Cypher |
| `template` | Planner / fallback / simple counts | `knowledge_graph.retrieval.ecom` templates |
| Multi-graph compare | Cross ecom+PLM / shared materials | One multi-graph Cypher or per-graph merge |
| Prior Cypher adapt | Contextual + `last_cypher` present | Reuse/adapt previous query |

After retrieve, state persists:

- `active_filters`, `group_by`, `last_result_id`, `last_retrieved_text`
- `last_graph_id`, `last_cypher`, `last_cypher_params`, `last_metric_family`

---

## 9. ResponseAgent (findings → user prose)

File: `agents/_impl.py` → `ResponseAgent.compose`

Skipped when:

- Clarification awaiting (`Awaiting clarification`)
- A finalized simple count is already formatted from grounded retrieval / metadata
- `answer_mode != llm`
- Extremely large Text2Cypher tables (passthrough)

### 9.1 Prompt packaging

```
<question> … </question>
<findings> … grounded facts only … </findings>
<optional_metadata> … optional facet facts; agent decides depth … </optional_metadata>
<example_styles> … evidence excerpts … </example_styles>
<caveats> … </caveats>
<task> … adaptive hint by response_type … </task>
```

### 9.2 Writing rules (summary)

- Ground only in findings / optional metadata — never invent counts
- Adaptive format: short → prose; dense → table/chart
- No trailing “Would you like…” CTAs
- Count + optional metadata → short natural briefing (dynamic wording, not fixed templates)
- Stray numbers not in findings → fall back to retrieved text

Returns `GroundedAnswer`: `answer`, `evidence`, `caveats`, `routes`, `unavailable`.

---

## 10. End-to-end flows by variation

### A) Empty chat — “How many products does Primark have?”

```
Context: new_independent_question, filters={}
QU: Ready, filters={brand:primark}, metrics=[count]
kg_stats: hit on ecom (or sticky/named graph)
Direct metadata count: finalized answer text from kg_stats (no response LLM)
State: active_filters={brand:primark}, last_graph_id=ecom_products
```

### B) Connected follow-up — “And how many of them are red?”

```
Context: continuation, carry brand=primark
QU: Ready, filters={brand:primark, color:red} (or equivalent)
Retrieve: Cypher or template with both filters
State: active_filters updated
```

### C) New independent after Primark — “How many products are in the e-comm data?”

```
Context: new_independent_question (dataset-total guard)
State.active_filters cleared before QU
QU: Ready, filters={}, metrics=[count], source=ecom_products
kg_stats: global ecom total (~18k) — NOT Primark
```

### D) Full analysis — “Give me a full analysis of Primark”

```
Context: partial_context_reuse, active_filters={brand:primark}
QU: may clarify → final_confirm → analysis_depth=full
Retrieve: analytical_tool (assortment briefing)
ResponseAgent: analysis_report shape
```

### E) Clarification mid-flight

```
QU: ClarifyResult(slot=entity|final_confirm|…)
Return clarify question as answer text
Persist clarification + step filters
Next user message resumes QU._continue / LLM slot fill
```

### F) Cross-catalog compare

```
Intention prefers ecom + plmex (or shared)
Retrieve: multi-graph Text2Cypher or merge
ResponseAgent: comparison formatting, attribute sources separately
```

### G) REUSE (rare)

```
Context.decision=REUSE + last_retrieved_text present + not clarifying
Skip Neo4j; ResponseAgent may still rewrite unless clarification path
```

### H) LangGraph non-catalog (no session_state)

```
Router → web_search | direct | sample_design
No ConversationContext / kg_stats path
```

---

## 11. Session state schema (WebSocket)

Persisted on the conversation (subset):

| Key | Purpose |
|-----|---------|
| `last_graph_id` | Sticky product KG for next turn |
| `active_filters` | Last executed / contextual filters |
| `group_by` | Last breakdown dimension |
| `last_result_id` / `last_retrieved_text` | REUSE support |
| `last_cypher` / `last_cypher_params` | Contextual Cypher adapt |
| `last_metric_family` | Metric-shift guard |
| `clarification` | Progressive QU state |
| `validated_intent` | Last ReadyResult snapshot |
| `available_attributes` | Schema hints for follow-ups |
| `analysis_brief` | Mid-analysis UI / trail |

HTTP `/ask` without WS session memory is weaker on sticky filters; prefer WebSocket chat for multi-turn merchant dialogue.

---

## 12. Key modules (quick map)

| Concern | Path |
|---------|------|
| API / WS | `fashion_brain/api/app.py` |
| Answer entry | `fashion_brain/ask.py` (`answer_question`, `answer_from_catalog`) |
| LangGraph router | `fashion_brain/flow.py` |
| Catalog orchestrator | `fashion_brain/orchestration/catalog_pipeline.py` |
| Context resolution | `fashion_brain/agents/conversation_context.py` |
| Query understanding | `fashion_brain/agents/query_understanding.py` |
| Thinking / Intention / Response | `fashion_brain/agents/_impl.py` |
| Contracts | `fashion_brain/agents/contracts.py` |
| Turn filter carry | `fashion_brain/planning/active_constraints.py` |
| Retrieve mode planner | `fashion_brain/planning/query_planner.py` |
| Text2Cypher | `fashion_brain/agents/text2cypher.py` |
| Count / templates | `knowledge_graph/retrieval/ecom.py` |
| Precomputed stats | `knowledge_graph/retrieval/kg_stats.py` |
| Product analysis | `fashion_brain/analytics/engine.py` |

---

## 13. Decision checklist (debugging a wrong answer)

1. **Was session_state used?** (WS vs HTTP)  
2. **Context `intent_type` / `carry` / `active_filters`?**  
3. **Did dataset-total / new-independent clear brand?**  
4. **QU Ready vs Clarify? `metrics` / `analysis_depth` / `source`?**  
5. **Early `kg_stats` hit with wrong filters?**  
6. **Thinking `references_prior_chat` glued prior brand?**  
7. **Graph sticky vs named catalog?**  
8. **Retrieve mode: meta / analytical / Text2Cypher / template?**  
9. **ResponseAgent invented vs grounded? optional_metadata used?**  

Trace steps are emitted via `fashion_brain.trace` (`context`, `query_understanding`, `thinking`, `intention`, `retrieve`, `response`).

---

## 14. One-page flowchart

```
                    ┌─────────────────────┐
                    │  User question +    │
                    │  history + session  │
                    └─────────┬───────────┘
                              ▼
                    ┌─────────────────────┐
                    │ ConversationContext │
                    │ intent_type + carry │
                    └─────────┬───────────┘
                              │
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
        REUSE              CLARIFY            RETRIEVE
     prior text         return question      ValidatedIntent
           │                  │                  │
           │                  └────── UI ────────┤
           │                                     ▼
           │                          ┌─────────────────────┐
           │                          │ kg_stats? cache?    │
           │                          │ analytical / T2C /  │
           │                          │ template / multi-KG │
           │                          └─────────┬───────────┘
           │                                    ▼
           │                          ┌─────────────────────┐
           └──────────────────────────│ Retrieved findings  │
                                      └─────────┬───────────┘
                                                ▼
                                      ┌─────────────────────┐
                                      │ ResponseAgent prose │
                                      └─────────┬───────────┘
                                                ▼
                                      ┌─────────────────────┐
                                      │ GroundedAnswer +    │
                                      │ updated session     │
                                      └─────────────────────┘
```

---

*This reflects the Fashion Brain catalog orchestration as implemented in-repo. When behavior changes, update this file alongside `conversation_context.py` and `catalog_pipeline.py`.*
