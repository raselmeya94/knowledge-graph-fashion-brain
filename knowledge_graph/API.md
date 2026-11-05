# Knowledge Graph API

HTTP API for the `knowledge_graph` package.

| | |
| --- | --- |
| Base URL | `http://127.0.0.1:8001` |
| Start | `python -m knowledge_graph.cli serve` |
| Swagger UI | http://127.0.0.1:8001/docs |
| ReDoc | http://127.0.0.1:8001/redoc |
| OpenAPI | [`openapi.json`](openapi.json) |
| Postman | [`../postman/knowledge_graph.postman_collection.json`](../postman/knowledge_graph.postman_collection.json) · run guide [`../postman/POSTMAN_RUN.md`](../postman/POSTMAN_RUN.md) |

Regenerate OpenAPI after route changes:

```powershell
$env:PYTHONPATH = "."
python -c "import json; from pathlib import Path; from knowledge_graph.api import create_app; Path('knowledge_graph/openapi.json').write_text(json.dumps(create_app(platform=None).openapi(), indent=2), encoding='utf-8')"
```

---

## Authentication

| Header | When | Values |
| --- | --- | --- |
| `Authorization: Bearer <token>` | If `FASHIONGPT_API_TOKEN` is set, or env is not `development`/`test` | API token |
| `X-Clearance` | Optional (default `public`) | `public`, `internal`, `confidential` |

E-comm product nodes are usually `classification=internal` — send `X-Clearance: internal` for product counts.

Error body:

```json
{ "error": "<code>", "message": "<text>" }
```

---

## Plan-aligned `/api/*` aliases

These mirror the Fashion Brain KG management plan. Legacy routes without the `/api` prefix still work.

| Method | Path | Purpose |
| --- | --- | --- |
| GET/POST | `/api/sources` | List / register sources |
| GET | `/api/sources/{id}` | Source details |
| GET | `/api/sources/{id}/schema` | Schema + mapping |
| POST | `/api/sources/{id}/sync` | Unified sync job |
| GET | `/api/graphs` | List graphs |
| GET | `/api/graphs/registry` | Unified registry (capabilities, entities) |
| GET | `/api/graphs/{id}/schema` | Graph schema metadata |
| POST | `/api/graphs/query` | Single-graph template query |
| POST | `/api/graphs/cross-query` | Multi-graph brain retrieve |
| GET | `/api/graphs/global` | Global/shared interconnect info |
| POST | `/api/graphs/rebuild` | Rebuild graph or source |
| GET | `/api/metadata` | Dynamic metadata |
| GET | `/api/metadata/values` | Paginated attribute value discovery |

---

## Health and sources

### `GET /health`

```json
{
  "status": "ok",
  "service": "knowledge_graph",
  "graph_backend": "neo4j",
  "graph": "ok",
  "ollama": "ok"
}
```

### `GET /sources`

Registered sources from `data/raw/source_registry.json`.

### `GET /jobs` · `GET /jobs/{job_id}`

Background jobs. Optional query: `kind`, `limit`.

### `GET /ingestion/status`

Latest ingest job, reports, shared counts, degraded reason.

### `POST /ingestion/run`

Development only (`FASHIONGPT_ENV=development`). Starts a background rebuild.

```json
{ "sources": ["own_products"] }
```

Omit `sources` to rebuild all registered file sources.

---

## Graphs and ontologies

### `GET /graphs`

Active graphs + `shared` / `master` interconnect overlays, each with `node_count` / `edge_count`.

Product graphs use `"kind": "product"` (builtin: `ecom_products`).

### `GET /graphs/{graph_id}`

Graph metadata, counts, and ontology (when registered).

### `POST /graphs`

```json
{
  "id": "optional-slug",
  "name": "E-commerce products",
  "ontology_id": "ecom",
  "kind": "product",
  "description": "",
  "source_ids": ["ecom_products"]
}
```

`kind`: `product` | `documents` | `custom`.

### `POST /graphs/reload`

Hot-reload snapshots (background job).

### `POST /graphs/{graph_id}/sync` · `POST /graphs/{graph_id}/rebuild`

Background sync/rebuild for that graph.

### `DELETE /graphs/{graph_id}`

```json
{ "force": false }
```

### `DELETE /graphs/{graph_id}/edges/{edge_id}`

Soft-delete one edge.

### `PATCH /graphs/{graph_id}/edges/{edge_id}`

```json
{ "properties": { "note": "…" } }
```

### Ontologies

| Method | Path |
| --- | --- |
| GET | `/ontologies` |
| POST | `/ontologies` |
| GET | `/ontologies/{ontology_id}` |
| PUT | `/ontologies/{ontology_id}` |
| POST | `/ontologies/{ontology_id}/pairs` |
| DELETE | `/ontologies/{ontology_id}/pairs` |

**Create ontology** (`POST /ontologies`):

```json
{
  "name": "My ontology",
  "description": "",
  "clone_from": "ecom",
  "entity_types": [],
  "relationship_allowlist": {},
  "version": "1.0.0"
}
```

**Add/remove pair**:

```json
{ "type": "USES_MATERIAL", "start_type": "Product", "end_type": "Material" }
```

---

## Query

### `POST /brain/retrieve`

Grounded evidence text + evidence rows for a catalog/document intent.

```json
{
  "intent": "product_count",
  "question": "how many products",
  "filters": {},
  "group_by": null,
  "limit": null,
  "analysis": false,
  "suggest": false,
  "season_name": null,
  "upcoming": false,
  "corrections": [],
  "graph_id": "ecom_products",
  "graph_ids": null
}
```

| Field | Meaning |
| --- | --- |
| `intent` | `product_count`, `specification`, `shared_products`, `news`, `live_sql`, … |
| `filters` | e.g. `{ "brand": "matalan" }`, `{ "garment": "sweater" }` |
| `group_by` | e.g. `brand`, `category` |
| `analysis` | type-wise tops on a Neo4j product graph |
| `suggest` | specification path |
| `graph_id` / `graph_ids` | scope to one or more registry graphs |

**Response:**

```json
{
  "text": "…",
  "evidence": [
    {
      "source_id": "ecom_products",
      "record_id": "…",
      "title": "…",
      "excerpt": "…",
      "verification_status": "observed",
      "graph": "ecom_products",
      "node_id": "…",
      "synthetic": false
    }
  ],
  "caveats": [],
  "unavailable": [],
  "synthetic_data": false,
  "routes_hint": ["product_count", "ecom_products"]
}
```

Neo4j product graphs (`kind=product`) are counted via Product nodes. Empty PLM file envelopes are not required for e-comm answers.

### `POST /graphs/{graph_name}/query`

Approved templates only (no free Cypher).

```json
{
  "template": "list_products",
  "params": {},
  "limit": 20,
  "offset": 0
}
```

`limit` must be 1–100. Templates are defined in `knowledge_graph/graph/templates.py`.

### `GET /products`

File-catalog search. Query params: `q`, `material`, `season`, `category`, `limit`, `offset`.

For e-comm Neo4j data prefer `POST /brain/retrieve`.

---

## Construction

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/kg/tools` | List tools |
| GET | `/kg/ollama` | Ollama status |
| GET | `/kg/relationships` | Existing relationships |
| GET | `/kg/reviews` | Review queue |
| GET | `/kg/reviews/{review_id}` | One review |
| POST | `/kg/tools/{tool_name}` | Run tool |
| POST | `/extract` | Extract text/structure |
| POST | `/relationships/propose` | Propose edges |
| POST | `/kg/pipeline` | extract → propose → queue |
| POST | `/kg/pipeline/upload` | Pipeline with file upload |
| POST | `/kg/reviews/{review_id}` | Approve / edit / reject |
| DELETE | `/kg/reviews/{review_id}/relationships/{rel_id}` | Drop proposal |
| POST | `/kg/reviews/{review_id}/commit` | Write into graph |

**Pipeline body:**

```json
{
  "text": null,
  "filename": "article.pdf",
  "kind": "pdf",
  "use_model": true,
  "clean": false,
  "auto_commit": false,
  "graph_id": null,
  "ontology_id": null,
  "data_base64": null,
  "json_data": null
}
```

Kinds: `pdf`, `docx`, `doc`, `text`, `email`, `html`, `json`, `image`. Max upload ~8 MB.

---

## Scraping steps

| Method | Path |
| --- | --- |
| POST | `/scraping/extract` |
| POST | `/scraping/extract/upload` |
| POST | `/scraping/chunk` |
| POST | `/scraping/propose` |
| POST | `/scraping/queue` |
| POST | `/scraping/commit` |

Jobs are stored under `data/generated/scraping/files/`.

---

## CLI (same package, not HTTP)

| Command | Purpose |
| --- | --- |
| `python -m knowledge_graph.cli serve` | Run this API |
| `python -m knowledge_graph.cli ecom-ping` | Test Postgres `master_products` |
| `python -m knowledge_graph.cli ecom-sync --full` | Sync e-comm → Neo4j `ecom_products` |
| `python -m knowledge_graph.cli ingest` | File-source ingest |
| `python -m knowledge_graph.cli pipeline …` | Construction CLI |
| `python -m knowledge_graph.cli tools` | List tools |

---

## Typical errors

| HTTP | Meaning |
| --- | --- |
| 400 | Validation |
| 401 | Auth required / bad token |
| 403 | Forbidden (e.g. ingest outside development) |
| 404 | Missing graph, review, or edge |
| 504 | Request timeout |
