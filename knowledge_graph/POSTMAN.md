# Knowledge Graph — Postman pipeline (empty → full)

Complete step-by-step guide to rebuild the KG **only via HTTP / Postman**, using the **real** ontology, source, and query bodies from this project (not dummy placeholders).

| | |
| --- | --- |
| Base URL | `http://127.0.0.1:8001` |
| Collection | [`../postman/knowledge_graph.postman_collection.json`](../postman/knowledge_graph.postman_collection.json) |
| Request bodies | [`../postman/bodies/`](../postman/bodies/) |
| Swagger | http://127.0.0.1:8001/docs |
| Route reference | [`API.md`](API.md) |

**Assumption after wipe:** Neo4j is empty, `data/raw` is empty, no `kg_registry.json`, no schemas on disk. The API boots empty; you create everything with POST.

Do **not** start with sync/query — those fail until ontologies, graphs, and sources exist.

---

## 0. One-time setup

### 0.1 Start KG

```powershell
cd d:\Rasel-P\R&D-KG
$env:PYTHONPATH = "."
python -m knowledge_graph.cli serve
```

### 0.2 Postman collection variables

| Variable | Value |
| --- | --- |
| `baseUrl` | `http://127.0.0.1:8001` |
| `clearance` | `internal` |
| `apiToken` | empty unless `FASHIONGPT_API_TOKEN` is set in `.env` |
| `graphId` | `ecom_products` (after Step 3) |
| `sourceId` | `ecom_products` (after Step 4) |
| `ontologyId` | `ecom` (after Step 2) |
| `jobId` | filled from sync response |

Every request should send:

```http
X-Clearance: internal
```

If `apiToken` is set:

```http
Authorization: Bearer {{apiToken}}
```

### 0.3 E-comm Postgres (required for e-com sync)

In repo-root `.env` (real values from your machine):

```env
FASHIONGPT_GRAPH_BACKEND=neo4j
FASHIONGPT_NEO4J_URI=bolt://localhost:7687
FASHIONGPT_NEO4J_USER=neo4j
FASHIONGPT_NEO4J_PASSWORD=<your-password>
FASHIONGPT_ECOM_DSN=postgresql://USER:PASSWORD@HOST:5432/DB
# or POSTGRES_HOST / POSTGRES_USER / POSTGRES_PASSWORD / POSTGRES_DB
```

### 0.4 E-com column mapping file (not an API)

Sync reads the mapping from disk. Before Step 5, copy:

```text
postman/bodies/ecom_master_products_mapping.json
  → data/schemas/ecom_master_products_mapping.json
```

### 0.5 How to use body files in Postman

For each POST below: **Body → raw → JSON**, then paste the file contents from `postman/bodies/…`, or use Postman’s file body if you prefer.

---

## Pipeline overview

```text
1 Health / empty checks
2 POST ontologies      (ecom → product → news → system)
3 POST graphs          (ecom_products → own_products → news → kg_system)
4 POST sources         (ecom_products → own_products → news)
5 POST sync            (e-com first; PLM after CSV is on disk)
6 GET metadata / values
7 POST query / brain retrieve / cross-query
8 Optional: document construction → review → commit
```

---

## Step 1 — Health and empty state

| # | Method | URL | Body | Expect |
| --- | --- | --- | --- | --- |
| 1.1 | `GET` | `/health` | — | `status: ok`, `service: knowledge_graph`, `graph_backend: neo4j` |
| 1.2 | `GET` | `/api/sources` | — | `{ "sources": [] }` |
| 1.3 | `GET` | `/ontologies` | — | `{ "ontologies": [] }` |
| 1.4 | `GET` | `/api/graphs` | — | only `shared` / `master` overlays (0 nodes) |
| 1.5 | `GET` | `/api/graphs/registry` | — | `{ "registry": [] }` (no product graphs yet) |
| 1.6 | `GET` | `/jobs?limit=5` | — | job list (may be empty) |

Collection folder: **Health**, then **Plan API aliases** GETs / **Registry → GET ontologies**.

---

## Step 2 — Register ontologies (real schemas)

Order matters only if you later `clone_from`; here we POST full definitions.

| # | Method | URL | Body file |
| --- | --- | --- | --- |
| 2.1 | `POST` | `/ontologies` | [`POST_ontologies_ecom.json`](../postman/bodies/POST_ontologies_ecom.json) |
| 2.2 | `POST` | `/ontologies` | [`POST_ontologies_product.json`](../postman/bodies/POST_ontologies_product.json) |
| 2.3 | `POST` | `/ontologies` | [`POST_ontologies_news.json`](../postman/bodies/POST_ontologies_news.json) |
| 2.4 | `POST` | `/ontologies` | [`POST_ontologies_system.json`](../postman/bodies/POST_ontologies_system.json) |

**What these contain (real project data):**

- **ecom** — Product, Brand, Material, GarmentCategory, Color, Availability, …; brands Zara / Primark / Matalan / C&A / Walmart as canonical entities; relationships `HAS_BRAND`, `USES_MATERIAL`, `HAS_COLOR`, `PRICED_IN`, …
- **product** — full PLMEX ontology (Product, Fabric, Season, Supplier, Company, AI tags, …)
- **news** — Document / Company / Brand / Topic / Country / … with `MENTIONS`, `EXPORTS_TO`, …
- **system** — KnowledgeGraph / Source / Attribute / SharedInterconnect / Capability

**Verify**

| # | Method | URL | Expect |
| --- | --- | --- | --- |
| 2.5 | `GET` | `/ontologies` | 4 active ontologies: `ecom`, `product`, `news`, `system` |
| 2.6 | `GET` | `/ontologies/ecom` | definition includes `Brand`, `HAS_BRAND`, canonical `brand:primark` |

On disk after this step: `data/schemas/ontologies/{ecom,product,news,system}.json` and `data/generated/kg_registry.json` (created on first save).

### Optional ontology edits

| Method | URL | Body example |
| --- | --- | --- |
| `POST` | `/ontologies/{id}/pairs` | `{ "type": "USES_MATERIAL", "start_type": "Product", "end_type": "Material" }` |
| `DELETE` | `/ontologies/{id}/pairs` | same shape |
| `PUT` | `/ontologies/{id}` | partial replace (`name`, `entity_types`, `relationship_allowlist`, …) |

---

## Step 3 — Create graphs

`kind=product` is only allowed for ids `ecom_products` or `own_products`.

| # | Method | URL | Body file |
| --- | --- | --- | --- |
| 3.1 | `POST` | `/graphs` | [`POST_graphs_ecom_products.json`](../postman/bodies/POST_graphs_ecom_products.json) |
| 3.2 | `POST` | `/graphs` | [`POST_graphs_own_products.json`](../postman/bodies/POST_graphs_own_products.json) |
| 3.3 | `POST` | `/graphs` | [`POST_graphs_news.json`](../postman/bodies/POST_graphs_news.json) |
| 3.4 | `POST` | `/graphs` | [`POST_graphs_kg_system.json`](../postman/bodies/POST_graphs_kg_system.json) |

**Verify**

| # | Method | URL | Expect |
| --- | --- | --- | --- |
| 3.5 | `GET` | `/api/graphs` | `ecom_products`, `own_products`, `news`, `kg_system` (+ shared/master), all `node_count: 0` |
| 3.6 | `GET` | `/api/graphs/registry` | same graphs with capabilities / entities |
| 3.7 | `GET` | `/api/graphs/ecom_products/schema` | schema metadata for e-com |
| 3.8 | `GET` | `/api/graphs/global` | shared interconnect info |

Set collection vars: `graphId=ecom_products`, `ontologyId=ecom`.

---

## Step 4 — Register sources

| # | Method | URL | Body file |
| --- | --- | --- | --- |
| 4.1 | `POST` | `/api/sources` | [`POST_api_sources_ecom_products.json`](../postman/bodies/POST_api_sources_ecom_products.json) |
| 4.2 | `POST` | `/api/sources` | [`POST_api_sources_own_products.json`](../postman/bodies/POST_api_sources_own_products.json) |
| 4.3 | `POST` | `/api/sources` | [`POST_api_sources_news.json`](../postman/bodies/POST_api_sources_news.json) |

**Verify**

| # | Method | URL | Expect |
| --- | --- | --- | --- |
| 4.4 | `GET` | `/api/sources` | `ecom_products`, `own_products`, `news` |
| 4.5 | `GET` | `/api/sources/ecom_products` | connector `authorized_sql`, enabled true |
| 4.6 | `GET` | `/api/sources/ecom_products/schema` | schema / mapping notes |
| 4.7 | `GET` | `/api/sources/own_products/schema` | PLM schema notes |

On disk: `data/raw/source_registry.json`.

Set `sourceId=ecom_products`.

---

## Step 5 — Sync data into Neo4j

### 5A — E-commerce (Postgres `master_products` → Neo4j)

Prerequisites: Step 0.3 DSN, Step 0.4 mapping file, Steps 2–4 done.

| # | Method | URL | Body |
| --- | --- | --- | --- |
| 5.1 | `POST` | `/api/sources/ecom_products/sync` | _(empty)_ |
| 5.2 | `GET` | `/jobs/{{jobId}}` | poll until `status` is finished / failed |

Save `job.id` from 5.1 into `jobId`.

**Alternate (same job family):**

| Method | URL | Notes |
| --- | --- | --- |
| `POST` | `/graphs/ecom_products/sync` | legacy graph sync |
| `POST` | `/api/graphs/rebuild` | body `{ "graph_id": "ecom_products" }` or `{ "source_id": "ecom_products" }` |

**Verify after success**

| # | Method | URL | Expect |
| --- | --- | --- | --- |
| 5.3 | `GET` | `/api/graphs` | `ecom_products.node_count` > 0 (past runs were thousands of nodes) |
| 5.4 | `GET` | `/ingestion/status` | last reports / shared counts |
| 5.5 | `GET` | `/api/metadata?graph_id=ecom_products` | attributes list |
| 5.6 | `GET` | `/api/metadata/values?graph_id=ecom_products&attribute=brand&limit=25` | real brands e.g. primark, zara, matalan, … |

CLI equivalent (if you prefer terminal later): `python -m knowledge_graph.cli ecom-ping` then `ecom-sync --full`.

### 5B — PLM / own_products (file CSV)

1. Place catalog under:

```text
data/raw/own_products/products_data.csv
data/raw/own_products/manifest.json
```

(`manifest.json` must list the CSV with a content hash — use a previous manifest or regenerate after restore.)

2. Sync:

| # | Method | URL | Body |
| --- | --- | --- | --- |
| 5.7 | `POST` | `/api/sources/own_products/sync` | _(empty)_ |
| 5.8 | `GET` | `/jobs/{{jobId}}` | poll |

**Or (dev only):**

| Method | URL | Body |
| --- | --- | --- |
| `POST` | `/ingestion/run` | `{ "sources": ["own_products"] }` |

Requires `FASHIONGPT_ENV=development`.

### 5C — Hot reload (optional)

| Method | URL |
| --- | --- |
| `POST` | `/graphs/reload` |

Then poll `/jobs/{{jobId}}`.

---

## Step 6 — Metadata discovery (Agent 2 inputs)

| # | Method | URL | Expect |
| --- | --- | --- | --- |
| 6.1 | `GET` | `/api/metadata` | all graphs metadata |
| 6.2 | `GET` | `/api/metadata?graph_id=ecom_products` | e-com fields |
| 6.3 | `GET` | `/api/metadata/values?attribute=brand&graph_id=ecom_products&limit=50` | brand samples |
| 6.4 | `GET` | `/api/metadata/values?attribute=material&graph_id=ecom_products&q=cot&limit=20` | filtered materials |
| 6.5 | `GET` | `/api/metadata/values?attribute=category&graph_id=own_products&limit=25` | PLM categories (after 5B) |

Never dump full distinct lists into LLM prompts — use these paginated APIs.

---

## Step 7 — Query (real filters from past usage)

### 7.1 Single-graph template query

| Method | URL | Body file |
| --- | --- | --- |
| `POST` | `/api/graphs/query` | [`POST_api_graphs_query_ecom.json`](../postman/bodies/POST_api_graphs_query_ecom.json) |

Legacy equivalent: `POST /graphs/ecom_products/query` with the same `template` / `params` / `limit`.

Approved templates for product graphs: `list_products`, `product_materials`, `products_sharing_material`, `trace`.

### 7.2 Brain retrieve (Fashion Brain evidence pack)

| Method | URL | Body file |
| --- | --- | --- |
| `POST` | `/brain/retrieve` | [`POST_brain_retrieve_ecom_brand.json`](../postman/bodies/POST_brain_retrieve_ecom_brand.json) |

Real filter used historically: `"brand": "primark"` (also try `zara`, `matalan`, `walmart`, `c-and-a`).

Other useful bodies (edit in Postman):

```json
{
  "question": "Which products use cotton?",
  "intent": "product_count",
  "filters": { "material": "cotton" },
  "graph_id": "ecom_products",
  "analysis": false,
  "suggest": false
}
```

```json
{
  "question": "analysis PRIMARK buyer past products",
  "intent": "product_count",
  "filters": { "buyer": "PRIMARK" },
  "analysis": true,
  "limit": 5,
  "graph_id": "own_products"
}
```

### 7.3 Cross-query / Global KG

| Method | URL | Body file |
| --- | --- | --- |
| `POST` | `/api/graphs/cross-query` | [`POST_api_graphs_cross_query.json`](../postman/bodies/POST_api_graphs_cross_query.json) |

### 7.4 Shared interconnect templates

| Method | URL | Body |
| --- | --- | --- |
| `POST` | `/graphs/shared/query` | `{ "template": "canonical_materials", "params": {}, "limit": 50, "offset": 0 }` |
| `POST` | `/graphs/shared/query` | `{ "template": "products_sharing_material", "params": {}, "limit": 20, "offset": 0 }` |

### 7.5 File catalog search (PLM envelopes only)

| Method | URL |
| --- | --- |
| `GET` | `/products?material=cotton&limit=5&offset=0` |

Prefer `/brain/retrieve` for Neo4j e-com.

---

## Step 8 — Document / news construction (optional)

Body: [`POST_kg_pipeline_news_text.json`](../postman/bodies/POST_kg_pipeline_news_text.json)

| # | Method | URL | Notes |
| --- | --- | --- | --- |
| 8.1 | `GET` | `/kg/tools` | list tools |
| 8.2 | `GET` | `/kg/ollama` | Ollama status |
| 8.3 | `POST` | `/kg/pipeline` | body file above → returns `review_id` |
| 8.4 | `GET` | `/kg/reviews` | queue |
| 8.5 | `GET` | `/kg/reviews/{{reviewId}}` | inspect proposals |
| 8.6 | `POST` | `/kg/reviews/{{reviewId}}` | approve / edit / reject (see collection) |
| 8.7 | `POST` | `/kg/reviews/{{reviewId}}/commit` | write into `news` graph |
| 8.8 | `GET` | `/kg/relationships` | committed relationships |

**PDF / upload path**

| Method | URL |
| --- | --- |
| `POST` | `/kg/pipeline/upload` | multipart file + `kind=pdf`, `graph_id=news`, `ontology_id=news`, `use_model=true` |
| `POST` | `/extract` | extract only |
| `POST` | `/relationships/propose` | propose only |

**Scraping wizard** (collection folder **Scraping**):  
`/scraping/extract` → `/chunk` → `/propose` → `/queue` → `/commit`  
Use `ontology_id=news`, `graph_id=news` after Steps 2–3.

**Verify news**

| Method | URL | Body |
| --- | --- | --- |
| `POST` | `/brain/retrieve` | `{ "question": "Who exports denim to the EU?", "intent": "news", "filters": {}, "graph_id": "news" }` |

---

## Endpoint checklist (all KG HTTP surfaces)

### Plan `/api/*`

| Method | Path | When in this guide |
| --- | --- | --- |
| GET/POST | `/api/sources` | Steps 1, 4 |
| GET | `/api/sources/{id}` | 4.5 |
| GET | `/api/sources/{id}/schema` | 4.6–4.7 |
| POST | `/api/sources/{id}/sync` | Step 5 |
| GET | `/api/graphs` | 1.4, 3.5, 5.3 |
| GET | `/api/graphs/registry` | 1.5, 3.6 |
| GET | `/api/graphs/{id}/schema` | 3.7 |
| GET | `/api/graphs/global` | 3.8 |
| POST | `/api/graphs/query` | 7.1 |
| POST | `/api/graphs/cross-query` | 7.3 |
| POST | `/api/graphs/rebuild` | 5A alternate |
| GET | `/api/metadata` | Step 6 |
| GET | `/api/metadata/values` | Step 6 |

### Registry / graphs (legacy mounts, same server)

| Method | Path |
| --- | --- |
| GET/POST | `/ontologies` |
| GET/PUT | `/ontologies/{id}` |
| POST/DELETE | `/ontologies/{id}/pairs` |
| GET/POST | `/graphs` |
| GET | `/graphs/{id}` |
| POST | `/graphs/reload` |
| POST | `/graphs/{id}/sync` · `/graphs/{id}/rebuild` |
| DELETE | `/graphs/{id}` |
| DELETE/PATCH | `/graphs/{id}/edges/{edge_id}` |
| POST | `/graphs/{name}/query` |
| POST | `/graphs/shared/query` |

### Catalog / jobs / brain

| Method | Path |
| --- | --- |
| GET | `/health` |
| GET | `/sources` |
| GET | `/jobs` · `/jobs/{id}` |
| GET | `/ingestion/status` |
| POST | `/ingestion/run` |
| GET | `/products` |
| POST | `/brain/retrieve` |

### Construction / scraping

| Method | Path |
| --- | --- |
| GET | `/kg/tools` · `/kg/ollama` · `/kg/relationships` · `/kg/reviews` · `/kg/reviews/{id}` |
| POST | `/kg/tools/{tool}` · `/extract` · `/relationships/propose` · `/kg/pipeline` · `/kg/pipeline/upload` |
| POST | `/kg/reviews/{id}` · `/kg/reviews/{id}/commit` |
| DELETE | `/kg/reviews/{id}/relationships/{rel_id}` |
| POST | `/scraping/extract` · `/extract/upload` · `/chunk` · `/propose` · `/queue` · `/commit` |

---

## Body file index

| File | Used for |
| --- | --- |
| `POST_ontologies_ecom.json` | POST `/ontologies` |
| `POST_ontologies_product.json` | POST `/ontologies` |
| `POST_ontologies_news.json` | POST `/ontologies` |
| `POST_ontologies_system.json` | POST `/ontologies` |
| `POST_graphs_ecom_products.json` | POST `/graphs` |
| `POST_graphs_own_products.json` | POST `/graphs` |
| `POST_graphs_news.json` | POST `/graphs` |
| `POST_graphs_kg_system.json` | POST `/graphs` |
| `POST_api_sources_ecom_products.json` | POST `/api/sources` |
| `POST_api_sources_own_products.json` | POST `/api/sources` |
| `POST_api_sources_news.json` | POST `/api/sources` |
| `POST_api_graphs_query_ecom.json` | POST `/api/graphs/query` |
| `POST_api_graphs_cross_query.json` | POST `/api/graphs/cross-query` |
| `POST_brain_retrieve_ecom_brand.json` | POST `/brain/retrieve` |
| `POST_kg_pipeline_news_text.json` | POST `/kg/pipeline` |
| `ecom_master_products_mapping.json` | copy → `data/schemas/` before e-com sync |
| `ontology_*.json` | raw ontology definitions (source for the POST_ wrappers) |

---

## Recommended Postman run order (folders)

**Prefer collection folder `0 Ecom pipeline (empty → live)`** — numbered 01–18 with the real ecom ontology body embedded (Product/Brand/Material + HAS_BRAND/USES_MATERIAL + Zara/Primark/Matalan/C&A/Walmart).

Variables after import: `ontologyId=ecom`, `graphId=ecom_products`, `sourceId=ecom_products`, `brand=primark`.

Then optionally add product/news/system via body files in Steps 2–4 above.

Legacy Path A folders still work after ecom exists.

---

## Common failures

| Symptom | Cause | Fix |
| --- | --- | --- |
| Sync 404 source | Source not registered | Step 4 |
| Graph create rejects `kind=product` | id not `ecom_products` / `own_products` | Use those ids |
| E-com sync errors on DSN | Postgres not configured | Step 0.3 + `ecom-ping` |
| Empty brands after sync | Mapping file missing | Step 0.4 |
| `clone_from: news` fails | News ontology not created | Run Step 2.3 first |
| Product rows missing | Clearance `public` | Set `X-Clearance: internal` |
| PLM sync no data | CSV / manifest missing | Restore under `data/raw/own_products/` |

---

## Related docs

- [`API.md`](API.md) — full route contracts  
- [`README.md`](README.md) — package overview  
- [`../postman/POSTMAN_RUN.md`](../postman/POSTMAN_RUN.md) — short Path A/B (Fashion Brain)  
- [`../kg-plan.txt`](../kg-plan.txt) — architecture plan (Graph Router uses `/api/graphs/registry`)
