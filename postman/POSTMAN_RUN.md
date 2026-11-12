# Postman — step-wise run guide

Import these two collections (do **not** mix ports):

| Collection | File | Port | Role |
|---|---|---|---|
| **knowledge_graph** | [`knowledge_graph.postman_collection.json`](knowledge_graph.postman_collection.json) | `8001` | Sources, graphs, registry, sync, retrieve |
| **fashion_brain** | [`fashion_brain.postman_collection.json`](fashion_brain.postman_collection.json) | `8000` | Auth, chat, memory; admin proxies to KG |

Swagger (optional): http://127.0.0.1:8001/docs · http://127.0.0.1:8000/docs

> **Empty KG / e-com first:** re-import [`knowledge_graph.postman_collection.json`](knowledge_graph.postman_collection.json) and run folder **`0 UI wizard (empty → live)`** (inspect → ontology → mapping → DB → source → graph → build → status).  
> Plan: [`../knowledge_graph/UI_CONTROL_PLAN.md`](../knowledge_graph/UI_CONTROL_PLAN.md) · guide: [`../knowledge_graph/POSTMAN.md`](../knowledge_graph/POSTMAN.md).

---

## 0. One-time setup

### 0.1 Environment

```powershell
cd d:\Rasel-P\R&D-KG
copy .env.example .env
# Edit .env: Neo4j password (if used), OpenRouter key (optional), Postgres for e-com (optional)
pip install -r requirements.txt
$env:PYTHONPATH = "."
```

### 0.2 Start services (two terminals)

```powershell
# Terminal 1 — Knowledge Graph (start first; warm load can take a while)
$env:PYTHONPATH = "."
python -m knowledge_graph.cli serve

# Terminal 2 — Fashion Brain
$env:PYTHONPATH = "."
python -m fashion_brain.cli serve
```

Optional frontend: `cd frontend; npm start` → http://127.0.0.1:5173

### 0.3 Postman variables

**knowledge_graph** collection:

| Variable | Default | Notes |
|---|---|---|
| `baseUrl` | `http://127.0.0.1:8001` | |
| `clearance` | `internal` | Product rows are usually internal |
| `apiToken` | _(empty)_ | Set only if `FASHIONGPT_API_TOKEN` is set in `.env` |
| `graphId` | `ecom_products` or `own_products` | Change after you list graphs |

**fashion_brain** collection:

| Variable | Default | Notes |
|---|---|---|
| `baseUrl` | `http://127.0.0.1:8000` | |
| `clearance` | `internal` | Sent as `X-Clearance` |
| `accessToken` / `refreshToken` | _(auto)_ | Filled by **POST login** test script |
| `conversationId` | _(auto)_ | Filled by create conversation / chat |

---

## Path A — Knowledge Graph only (no chat)

Run folders **in this order**:

### Step 1 — Health
1. **Health → GET health**  
   Expect `service: knowledge_graph`, `status: ok` (or `degraded` if Neo4j is down and you use memory).

### Step 2 — See what is registered (API-driven)
2. **Plan API aliases → GET /api/sources** — PLMEX + e-com (and any future sources).  
3. **Plan API aliases → GET /api/graphs/registry** — graph ids, capabilities, entities (this feeds Fashion Brain router).  
4. **Plan API aliases → GET /api/graphs/global** — shared/master interconnect.  
5. **Plan API aliases → GET /api/metadata** (or **GET /api/metadata/values** with `attribute=brand`) — discover values; do not paste huge lists into prompts.

### Step 3 — Sync / rebuild data
Pick one:

| Goal | Request |
|---|---|
| File PLM / own_products | **Catalog → POST ingestion run** body `{"sources":["own_products"]}` then poll **GET jobs/{id}** or **GET ingestion status** |
| E-com Postgres | **Plan API aliases → POST /api/sources/ecom_products/sync** (needs `FASHIONGPT_ECOM_DSN` / `POSTGRES_*`) |
| Named graph rebuild | **Plan API aliases → POST /api/graphs/rebuild** body `{"graph_id":"own_products"}` or `{"source_id":"ecom_products"}` |

Jobs are **async**. Use **GET /jobs/{job_id}** until `status` is finished.

### Step 4 — Query
6. **Plan API aliases → POST /api/graphs/query** — single graph template query.  
7. **Plan API aliases → POST /api/graphs/cross-query** — multi-graph / Global KG retrieve pack.  
8. Or legacy **Brain → POST /brain/retrieve** (same evidence pack Fashion Brain uses).

### Step 5 — Optional construction / scraping
Use **Construction** / **Scraping** folders only when ingesting documents (email, HTML, PDF). Not required for product chat.

---

## Path B — Fashion Brain chat (uses KG)

KG must already be running (Path A Step 1+ recommended so registry/graphs have data).

### Step 1 — Health
1. **Health → GET health** on `:8000`  
   Expect `kg: ok` (Fashion Brain can reach `:8001`).

### Step 2 — Auth
2. **1 Auth → POST register admin** (first time only; skip if user exists).  
3. **1 Auth → POST login** — test script saves `accessToken` + `refreshToken`.  
4. **1 Auth → GET me** — confirm role/clearance.

### Step 3 — Chat (either style)

**Preferred (plan alias)**  
5. **Plan chat alias → POST /api/v1/chat**  
   - First call: omit `session_id` or leave empty to create a session.  
   - Response sets `session_id`; save it into `conversationId` (or use the test script).  
   - Later calls: send `"session_id": "{{conversationId}}"`.

**Classic UI path**  
5. **2 Conversations → POST create conversation** (saves `conversationId`).  
6. **2 Conversations → POST conversation message**.

### Step 4 — Admin KG from Fashion Brain (optional)
Admin JWT required. Proxies to `:8001`:

7. **3 Admin → GET /v1/admin/sources**  
8. **3 Admin → GET /v1/admin/graphs**  
9. **3 Admin → POST /v1/admin/graphs/{id}/sync** or **ingestion/run**

New graphs registered on the KG API appear here automatically; Fashion Brain does not hard-code source names.

---

## Quick smoke checklist

| # | Collection | Request | Pass if |
|---|---|---|---|
| 1 | KG | GET `/health` | 200 |
| 2 | KG | GET `/api/graphs/registry` | ≥1 graph with `capabilities` |
| 3 | KG | GET `/api/sources` | `own_products` and/or `ecom_products` |
| 4 | FB | GET `/health` | `kg` not `unreachable` |
| 5 | FB | POST login | `accessToken` set |
| 6 | FB | POST `/api/v1/chat` | `status: completed` + non-empty `response` |

---

## Common failures

| Symptom | Fix |
|---|---|
| FB health `kg: unreachable` | Start KG on 8001; check `FASHIONGPT_KG_URL` |
| Empty product answers / public clearance | Set collection `clearance` = `internal` |
| 401 on KG | Clear `apiToken` in Postman if `.env` has empty `FASHIONGPT_API_TOKEN`, or match them |
| E-com sync fails | Set Postgres DSN in `.env`; confirm `master_products` exists |
| Chat without LLM key | Answers may be template/evidence-only; set `FASHIONGPT_MODEL_API_KEY` for full LLM replies |
| Job “started” forever | Poll `GET /jobs/{job_id}`; check KG terminal logs |

---

## Adding a new source later (API-only)

1. KG: **POST /api/sources** (register).  
2. KG: **GET /api/sources/{id}/schema** (inspect).  
3. KG: **POST /api/sources/{id}/sync** (build graph).  
4. KG: **GET /api/graphs/registry** (confirm new `graph_id` + capabilities).  
5. FB: ask again via **POST /api/v1/chat** — router reads registry; no code change.
