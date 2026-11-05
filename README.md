# Knowledge graph and Fashion Brain

These are two separate programs in one repository.

**Knowledge graph** builds and stores the graph. It loads catalog and snapshot data once, then serves retrieval over HTTP. Start it first:

```powershell
$env:PYTHONPATH = "."
python -m knowledge_graph.cli serve
```

Port 8001. Fashion Brain does not load the product graph.

**Fashion Brain** is the chat app. It calls the knowledge-graph API (`FASHIONGPT_KG_URL`, default `http://127.0.0.1:8001`) for evidence, then writes the answer with the LLM. Start it second:

```powershell
python -m fashion_brain.cli serve
```

Port 8000.

The construction pipeline is described in [knowledge_graph/README.md](knowledge_graph/README.md).

## Layout

Architecture map: [`ARCHITECTURE.md`](ARCHITECTURE.md) · Product plan: [`kg-plan.txt`](kg-plan.txt)

```text
knowledge_graph/          KG service (:8001)
  sources/                source lifecycle (register, schema, sync)
  adapters/               PLMEX (own_products) + e-commerce adapters
  kg_registry/            graphs, ontologies, capabilities
  global_kg/              shared/master interconnect facade
  api/                    /api/* routers (legacy routes still work)
  graph/                  memory / Neo4j + approved query templates
  resolution/             canonical links (RESOLVES_TO)
  pipeline/               product ingest / rebuild
  construction/           document, email, HTML, JSON, image tools
  retrieval/              brain retrieve + metadata values

fashion_brain/            chat service (:8000)
  orchestration/          agent orchestrator + LangGraph outer router
  agents/                 router, intent, context, spec, response
  execution/              kg_client + result merger
  planning/               keyword + model planners
  api/app.py              /v1/* + POST /api/v1/chat
  auth.py / store.py      JWT + SQLite conversations

data/                     all durable data (see data/README.md)
  raw/source_registry.json
  schemas/                product, ecom, news ontologies
  generated/              kg_registry, snapshots, fashion_brain/app.db
```

Raw product files under `data/raw/` are not modified during ingestion.

## Local setup

Python 3.11 or newer. From the repository root, with the conda `base` interpreter used for this project:

```powershell
pip install -r requirements.txt
$env:PYTHONPATH = "."
python -m knowledge_graph.cli validate
python -m fashion_brain.cli demo
```

The default graph is in memory. No Neo4j password is required. Answers use OpenRouter when `FASHIONGPT_MODEL_API_KEY` or `OPENROUTER_API_KEY` is set. Without a key, `/ask` returns the retrieved evidence text and `/health` reports `missing_api_key`. The default model is `~openai/gpt-sol-latest`. Change it with `FASHIONGPT_MODEL_NAME`. Copy `.env.example` to `.env` and put the key only in `.env`.

Optional local Neo4j:

```powershell
docker compose up -d
```

Set `FASHIONGPT_GRAPH_BACKEND=neo4j` and `FASHIONGPT_NEO4J_PASSWORD` in `.env`. Copy `.env.example` first. Paths in `.env` are relative to the `knowledge_graph` package (`raw_data`, `generated`, `schemas/ontology.json`).

## Chatbot

Start the knowledge graph first (slow once; holds the catalog and graphs), then Fashion Brain (fast start):

```powershell
# Terminal 1
python -m knowledge_graph.cli serve

# Terminal 2
python -m fashion_brain.cli serve
```

Interactive docs: http://127.0.0.1:8000/docs  
KG API docs: [knowledge_graph/API.md](knowledge_graph/API.md) · Swagger http://127.0.0.1:8001/docs · OpenAPI [knowledge_graph/openapi.json](knowledge_graph/openapi.json)

Postman (import both collections):

- Step-by-step: **[postman/POSTMAN_RUN.md](postman/POSTMAN_RUN.md)**
- [postman/knowledge_graph.postman_collection.json](postman/knowledge_graph.postman_collection.json) — graph API `:8001`
- [postman/fashion_brain.postman_collection.json](postman/fashion_brain.postman_collection.json) — chat API `:8000`

Fashion Brain retrieves evidence via `POST http://127.0.0.1:8001/brain/retrieve` (or plan `/api/graphs/cross-query`). UI is the separate [`frontend/`](frontend/) app.

```powershell
$headers = @{ "X-Clearance" = "internal" }
Invoke-RestMethod http://127.0.0.1:8001/products -Headers $headers
Invoke-RestMethod http://127.0.0.1:8000/ask -Method Post -Headers $headers -ContentType "application/json" -Body '{"question":"Which materials are used by our demo products?"}'
```

If `FASHIONGPT_API_TOKEN` is set, add `Authorization: Bearer <token>`. Products are classified `internal`, so a public clearance sees none.

Fashion Brain, port 8000 (backend for the frontend: user + admin):

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Chat process status |
| GET | `/` | API service info |
| GET | `/legacy/chat` | Optional legacy in-process chat demo |

The **UI is a separate project** in [`frontend/`](frontend/). Run it on its own port (default 5173) with `cd frontend; npm start`. It calls this API at `http://127.0.0.1:8000`.
| POST | `/ask` | Grounded answer (optional `conversation_id`) |
| WS | `/ws/ask` | Streaming ask |
| POST | `/v1/auth/register` | Create user or admin account |
| POST | `/v1/auth/login` | Login → access + refresh tokens |
| POST | `/v1/auth/refresh` | Refresh access token |
| POST | `/v1/auth/logout` | Revoke refresh token |
| GET/PATCH | `/v1/me` | Current user profile / clearance |
| GET/POST | `/v1/conversations` | List / create chats |
| GET/PATCH/DELETE | `/v1/conversations/{id}` | Get / rename-archive / delete |
| POST | `/v1/conversations/{id}/messages` | Ask inside a chat |
| POST | `/api/v1/chat` | Plan chat alias (`session_id` + `message`) |
| GET/POST/PATCH/DELETE | `/v1/memory` | Long-term user notes |
| GET | `/v1/admin/*` | Admin KG-making proxies (sources, pipeline, reviews, ingest) |

Admin routes require `role=admin`. They proxy to the private knowledge-graph service (`FASHIONGPT_KG_URL`). First admin can register without a bootstrap secret in development; set `FASHIONGPT_ADMIN_BOOTSTRAP_SECRET` for later admin signups.

Knowledge graph, port 8001:

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Graph process status |
| GET | `/sources` | Registered product source |
| GET | `/ingestion/status` | Ingestion state |
| POST | `/ingestion/run` | Development-only rebuild |
| GET | `/products` | Structured product search |
| POST | `/graphs/{graph}/query` | Approved templates (see `knowledge_graph/API.md`) |
| POST | `/graphs/shared/query` | Canonical materials, review queue, shared materials |
| POST | `/kg/pipeline` | Build relationships from a document, email, HTML page, JSON file, or image |

Graph calls take a template name and simple parameters. They do not accept Cypher. Approved templates: `list_products`, `product_materials`, `trace`, `review_queue`, `products_sharing_material`, `canonical_materials`.

Example questions:

- Which materials are used by our demo products?
- Which products share a material?
- How many synthetic demo products are in the catalog?

A live PLMEX count is refused. The SQL connector stays disabled.

## Adding a product row

1. Add the row to `knowledge_graph/raw_data/own_products/products.json` or `products.csv`.
2. Add a manifest record, then refresh hashes:

```powershell
$env:PYTHONPATH = "."
python knowledge_graph/scripts/refresh_manifest_hashes.py
```

3. Rebuild while the server is running:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/ingestion/run -Method Post -ContentType "application/json" -Body '{"sources":["own_products"]}'
```

Empty optional fields (fabric, supplier, manufacturer) do not create those edges. Required fields stay `product_id`, `product_name`, `category`, `season`, `material`, `classification`, and `disclaimer`.

## Tests

```powershell
$env:PYTHONPATH = "."
python -m pytest -q
```

Latest run: 80 passed, 1 skipped. The skipped test needs a local Neo4j and `FASHIONGPT_NEO4J_INTEGRATION=1`.

## Data flow

```mermaid
flowchart LR
  raw["raw_data/own_products"] --> adapter["product adapter"]
  adapter --> graph["own_products graph"]
  graph --> shared["shared canonical links"]
  shared --> brain["fashion_brain agent"]
  brain --> api["POST /ask"]
```
