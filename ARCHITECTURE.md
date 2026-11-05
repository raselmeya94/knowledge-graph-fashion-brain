# Fashion Brain Architecture Alignment

**Verdict:** The repository is on the right track. It already matches the core plan (separate KG + Fashion Brain services, per-source graphs, shared interconnect, JWT sessions, multi-agent draft pipeline). Gaps are structure and completeness — not a rewrite.

Full product definition: [`kg-plan.txt`](kg-plan.txt).

## System split

```text
frontend (:5173) → fashion_brain (:8000) → knowledge_graph (:8001)
                         │                        │
                    auth / chat / agents    sources / graphs / query
```

| Plan concept | Status | Primary code |
|---|---|---|
| Separate KG vs Fashion Brain APIs | Match | [`knowledge_graph/api/`](knowledge_graph/api/), [`fashion_brain/api/`](fashion_brain/api/) |
| Brands inside one E-Commerce KG | Match | `ecom_products` + `HAS_BRAND` |
| Source-specific graphs + Global/shared KG | Match | [`kg_registry/`](knowledge_graph/kg_registry/), [`resolution/shared.py`](knowledge_graph/resolution/shared.py), [`global_kg/`](knowledge_graph/global_kg/) |
| Validated queries (no free Cypher) | Match | [`graph/templates.py`](knowledge_graph/graph/templates.py) |
| Auth + conversation sessions | Strong | [`fashion_brain/auth.py`](fashion_brain/auth.py), [`store.py`](fashion_brain/store.py) |
| Dynamic multi-source (PLMEX + E-Com) | Dual registry | [`data/raw/source_registry.json`](data/raw/source_registry.json) |
| Five-agent orchestrator | Structured | [`fashion_brain/orchestration/`](fashion_brain/orchestration/), [`fashion_brain/agents/`](fashion_brain/agents/) |

## Target package layout

```text
knowledge_graph/
  sources/           # register, connect, schema, sync lifecycle
  adapters/          # own_products (PLMEX) + ecom
  kg_registry/       # graphs, ontologies, capabilities
  global_kg/         # shared/master interconnect facade
  api/               # /api/* routers (+ legacy mounts)
  pipeline/          # Platform rebuild
  retrieval/         # brain retrieve + metadata values
  construction/      # document/scraping pipeline

fashion_brain/
  orchestration/     # orchestrator + catalog pipeline + LangGraph router
  agents/            # router, intent, context, spec, response
  execution/         # kg_client + result_merger
  planning/          # keyword + model planners
  api/               # /v1/* + /api/v1/chat alias
```

## API aliases

Knowledge Graph (`:8001`) keeps legacy routes and adds plan-shaped aliases under `/api/*`:

- `GET/POST /api/sources`, `GET /api/sources/{id}`, `POST /api/sources/{id}/sync`, `GET /api/sources/{id}/schema`
- `GET /api/graphs`, `GET /api/graphs/registry`, `GET /api/graphs/{id}/schema`
- `POST /api/graphs/query`, `POST /api/graphs/cross-query`
- `GET /api/graphs/global`, `POST /api/graphs/rebuild`
- `GET /api/metadata`, `GET /api/metadata/values`

Fashion Brain (`:8000`) keeps `/v1/conversations/{id}/messages` and adds `POST /api/v1/chat`.

## Agent sequence

1. Auth + session  
2. Agent 3 Conversation Context → REUSE | REFINE | RETRIEVE  
3. Agent 1 Graph Router (multi-graph, brands as filters, insufficient coverage)  
4. Agent 2 Query Understanding  
5. Deterministic KG Query Execution + Result Merger  
6. Agent 4 Product Spec (conditional)  
7. Agent 5 Response Generation  

## Design principles (unchanged)

Dynamic sources, source independence, centralized e-commerce KG, global connectivity, dynamic metadata (no stuffing all values into prompts), intelligent routing, context reuse, controlled execution, conditional agents, traceability, scalability, security.

## API-driven rule (no hard-coded sources)

Fashion Brain **must not** hard-code source ids (`own_products`, `ecom_products`, …) or brand names (`Primark`, `Zara`, …) in agents.

| Concern | Source of truth |
|---|---|
| Which graphs exist | `GET /api/graphs` / `GET /api/graphs/registry` |
| Schema / relationships | `GET /api/sources/{id}/schema`, ontology registry |
| Attribute values (brands, …) | `GET /api/metadata/values` (paginated) |
| Sync / rebuild after admin changes | `POST /api/sources/{id}/sync`, `POST /api/graphs/rebuild` |
| Chat sessions / users | Fashion Brain SQLite at `data/generated/fashion_brain/app.db` (auth/history only — not product data) |

When an admin registers a new Postgres source and builds a graph via the KG API, Fashion Brain picks it up on the next router call through the registry — no code change.

## Why `data/generated/fashion_brain/app.db`?

That file is **only** Fashion Brain application state: users, JWT refresh tokens, conversations, messages, memory notes. It is not a knowledge graph. Product/graph data stays in Neo4j / KG snapshots under `data/generated/graph_snapshots/`. Keeping chat DB under `generated/` marks it as runtime (not source-of-truth schema).
