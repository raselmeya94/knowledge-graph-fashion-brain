# Knowledge Graph — UI Control Plan (API-only)

Goal: a **Fashion Brain / admin UI** (or Postman) can create, inspect, build, and query a knowledge graph **only through HTTP**.  
No copying files, no editing code, no hidden disk steps between clicks.

One-time outside UI (machine setup only): Neo4j up, `.env` secrets, `python -m knowledge_graph.cli serve`.  
Everything after that is API.

---

## 1. Product principles

| Principle | Meaning |
| --- | --- |
| **Inspect before create** | UI always GETs current state first (schemas, sources, graphs). |
| **Explicit register** | Ontology, mapping, DB connection, source are separate POSTs. |
| **Build is one action** | `POST …/build` starts a job from already-registered pieces. |
| **Status is pollable** | UI polls job/build status for products, nodes, edges, errors. |
| **Idempotent** | Re-POST existing ids returns clear conflict / reuse, never silent corruption. |
| **Secrets stay in `.env`** | UI never stores DB passwords in request history if avoidable; connections reference a named config or server `.env`. |

---

## 2. UI screen map → API

```text
┌─────────────────────────────────────────────────────────────┐
│  Dashboard                                                  │
│  GET /health                                                │
│  GET /api/bootstrap  (schemas? sources? graphs? jobs?)      │
└─────────────────────────────────────────────────────────────┘
         │ empty / incomplete
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Schema studio                                              │
│  GET  /ontologies                                           │
│  POST /ontologies                                           │
│  GET  /ontologies/{id}                                      │
│  PUT  /ontologies/{id}                                      │
│  POST /api/sources/{source_id}/mapping                      │
│  GET  /api/sources/{source_id}/schema                       │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Connections                                                │
│  POST /api/db/connections          (save named DB config)   │
│  GET  /api/db/connections                                   │
│  POST /api/db/connections/{id}/test                         │
│  GET  /api/db/connection?table=…   (inspect tables/cols)    │
│  POST /api/sources                 (bind source ↔ DB)       │
│  GET  /api/sources / {id}                                   │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Graphs                                                     │
│  GET  /api/graphs / registry                                │
│  POST /graphs                      (register empty KG)      │
│  POST /api/graphs/{graph_id}/build (or /api/kg/build)       │
│  GET  /jobs/{job_id}               (building_status)        │
│  GET  /api/graphs/{graph_id}       (node/edge counts)       │
└─────────────────────────────────────────────────────────────┘
         │ ready
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Explore / ask                                              │
│  GET  /api/metadata · /api/metadata/values                  │
│  POST /api/graphs/query · /api/graphs/cross-query           │
│  POST /brain/retrieve                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Official empty → live sequence (UI wizard)

Every step is one button / one request. Stop on failure; show response to the user.

| # | UI label | Method | Endpoint | Purpose |
| --- | --- | --- | --- | --- |
| **1** | Health check | `GET` | `/health` | Service + Neo4j (+ Ollama) alive |
| **2** | Schemas present? | `GET` | `/ontologies` | List ontologies (empty = none yet) |
| **3** | Sources present? | `GET` | `/api/sources` | List registered sources |
| **4** | KGs present? | `GET` | `/api/graphs` + `/api/graphs/registry` | Graphs + node/edge counts |
| **5** | Create ontology | `POST` | `/ontologies` | Push full ontology JSON from UI |
| **6** | Verify ontology | `GET` | `/ontologies/{ontology_id}` | Confirm types / allowlist |
| **7** | Upload mapping | `POST` | `/api/sources/{source_id}/mapping` | Column → graph mapping (**API, not file copy**) |
| **8** | Register / test DB | `POST` | `/api/db/connections` then `…/test` or inspect | Database name, host; then table metadata |
| **9** | Inspect table | `GET` | `/api/db/connection?table=master_products` | Columns, row counts, brands |
| **10** | Register source | `POST` | `/api/sources` | Source id, connector, enabled |
| **11** | Register graph | `POST` | `/graphs` | Graph id, `ontology_id`, `source_ids`, kind |
| **12** | Build KG | `POST` | `/api/graphs/{graph_id}/build` | Start build from source + schema + params |
| **13** | Building status | `GET` | `/jobs/{job_id}` (+ optional `/api/graphs/{id}/build/status`) | Products / nodes / edges / errors / % |
| **14** | Graph ready? | `GET` | `/api/graphs/{graph_id}` | Final `node_count`, `edge_count` |
| **15** | Discover values | `GET` | `/api/metadata/values?attribute=brand&graph_id=…` | Facets for UI filters |
| **16** | Query / ask | `POST` | `/api/graphs/query` · `/brain/retrieve` | Product answers |

### Wizard rules

1. Steps **1–4** are read-only. If ontology/source/graph already exist, UI skips create or offers “use existing”.
2. Steps **5–11** are registration only — **no Neo4j product load yet**.
3. Step **12** is the only long job; UI must poll **13** until `status` is `done` / `failed`.
4. Never call sync/build before mapping + ontology + source + graph exist → API returns **422** with a clear missing-piece list.

---

## 4. Resource model (what the UI stores as IDs)

```text
ontology_id     e.g. ecom
source_id       e.g. ecom_products
graph_id        e.g. ecom_products
connection_id   e.g. ecom_postgres   (named DB config)
job_id          returned by build/sync
```

**Bindings**

- Graph → `ontology_id` + `source_ids[]`
- Source → connector (`authorized_sql`) + optional `connection_id` + mapping
- Build → `graph_id` (+ `source_id`, `brand`, `limit`, `full`)

---

## 5. Endpoint catalog for full UI control

### 5.1 Already exist (keep)

| Method | Path | UI use |
| --- | --- | --- |
| `GET` | `/health` | Step 1 |
| `GET` | `/ontologies` | Step 2 |
| `POST` | `/ontologies` | Step 5 |
| `GET` | `/ontologies/{id}` | Step 6 |
| `PUT` | `/ontologies/{id}` | Edit schema in UI |
| `GET` | `/api/sources` | Step 3 |
| `POST` | `/api/sources` | Step 10 |
| `GET` | `/api/sources/{id}` | Detail |
| `GET` | `/api/sources/{id}/schema` | Mapping + ontology view |
| `POST` | `/api/sources/{id}/sync` | Legacy build trigger (keep) |
| `GET` | `/api/graphs` | Step 4 / 14 |
| `GET` | `/api/graphs/registry` | Router / capabilities |
| `POST` | `/graphs` | Step 11 |
| `GET` | `/graphs/{id}` | Counts + ontology |
| `GET` | `/jobs` · `/jobs/{id}` | Step 13 |
| `GET` | `/api/db/connection` | Step 9 inspect |
| `POST` | `/api/kg/build` | Optional shortcut (not primary wizard) |
| `GET` | `/api/metadata` · `/values` | Step 15 |
| `POST` | `/api/graphs/query` · `cross-query` · `/brain/retrieve` | Step 16 |

### 5.2 Add for perfect UI control (gaps)

| Method | Path | Why |
| --- | --- | --- |
| `GET` | `/api/bootstrap` | One call for UI home: health summary + ontologies count + sources + graphs + active jobs |
| `POST` | `/api/sources/{source_id}/mapping` | Upload mapping JSON from UI (removes file copy) |
| `GET` | `/api/sources/{source_id}/mapping` | Read mapping back into UI editor |
| `PUT` | `/api/sources/{source_id}/mapping` | Replace mapping |
| `POST` | `/api/db/connections` | Save named connection metadata (host/db/user; password via env ref or one-time set) |
| `GET` | `/api/db/connections` | List connections for dropdown |
| `POST` | `/api/db/connections/{id}/test` | Test without full inspect |
| `POST` | `/api/graphs/{graph_id}/build` | Canonical build: `{ source_id, ontology_id?, brand?, limit?, full? }` |
| `GET` | `/api/graphs/{graph_id}/build/status` | Friendly building_status: products, nodes, edges, errors, phase, job_id |
| `GET` | `/api/setup/checklist` | `{ ontology: bool, mapping: bool, source: bool, graph: bool, db: bool, ready_to_build: bool, missing: [] }` |

Optional later:

| Method | Path | Why |
| --- | --- | --- |
| `DELETE` | `/api/sources/{id}` | UI remove source |
| `POST` | `/api/graphs/{id}/reset` | Clear graph data (dangerous; confirm in UI) |
| `GET` | `/api/graphs/{id}/stats` | Brand/material histograms for dashboards |

---

## 6. Build + building_status contract

### Start build

`POST /api/graphs/ecom_products/build`

```json
{
  "source_id": "ecom_products",
  "brand": null,
  "limit": null,
  "full": false
}
```

Response:

```json
{
  "status": "started",
  "job": { "id": "abc123", "kind": "build", "status": "queued" }
}
```

### Poll status (UI progress bar)

`GET /jobs/abc123` **and/or** `GET /api/graphs/ecom_products/build/status`

Target status payload for UI:

```json
{
  "graph_id": "ecom_products",
  "job_id": "abc123",
  "phase": "syncing",
  "status": "running",
  "progress_percent": 42,
  "products": 1200,
  "node_writes": 5400,
  "edge_writes": 8900,
  "errors": 0,
  "brands": { "primark": 400, "zara": 800 },
  "message": "ecom sync progress products=1200",
  "finished_at": null
}
```

When done:

- `status: "done"` → UI enables Explore
- `status: "failed"` → show `errors` list / last error message  
  (sync already tracks per-row errors; surface them here)

Job progress should be updated **during** sync (not only at end), using existing `ecom_sync_state` / job `progress` field.

---

## 7. Checklist gate (prevent broken builds)

`GET /api/setup/checklist?graph_id=ecom_products&source_id=ecom_products`

```json
{
  "ready_to_build": false,
  "missing": ["mapping", "source"],
  "checks": {
    "health": true,
    "ontology": true,
    "mapping": false,
    "db": true,
    "source": false,
    "graph": true
  }
}
```

UI **disables** “Build” until `ready_to_build: true`.

---

## 8. Postman / UI wizard folder order (rewrite target)

```text
0 UI wizard (empty → live)
  01 GET /health
  02 GET /ontologies
  03 GET /api/sources
  04 GET /api/graphs
  05 GET /api/setup/checklist
  06 POST /ontologies
  07 GET /ontologies/ecom
  08 POST /api/sources/ecom_products/mapping
  09 POST /api/db/connections   (or use .env + GET inspect)
  10 GET /api/db/connection?table=master_products
  11 POST /api/sources
  12 POST /graphs
  13 GET /api/setup/checklist   (expect ready_to_build=true)
  14 POST /api/graphs/ecom_products/build
  15 GET /jobs/{jobId}          (poll)
  16 GET /api/graphs/ecom_products/build/status
  17 GET /api/graphs
  18 GET /api/metadata/values?attribute=brand
  19 POST /api/graphs/query
  20 POST /brain/retrieve
```

Shortcut `POST /api/kg/build` stays for demos only — **not** the official UI path.

---

## 9. Implementation phases

| Phase | Deliverable | UI unlock |
| --- | --- | --- |
| **P0** | Document this plan + reorder Postman to inspect → register → build | Clear UX order |
| **P1** | `POST/GET/PUT …/mapping` | No file copy |
| **P2** | `GET /api/setup/checklist` + `GET /api/bootstrap` | Home screen + Build gate |
| **P3** | `POST /api/graphs/{id}/build` + rich `build/status` with live counts | Progress UI |
| **P4** | `POST/GET /api/db/connections` (+ test) | Connection manager screen |
| **P5** | Live job `progress` during sync | Smooth progress bar |

---

## 10. What stays out of the UI (by design)

| Item | Where |
| --- | --- |
| Neo4j password / Postgres password | `.env` (or secret store); UI only sees host/db/user |
| Starting the Python process | Ops / local terminal / service manager |
| Docker Neo4j | Ops |

---

## 11. Success criteria

Someone with only Postman/UI can:

1. See empty system (health + empty lists).
2. Push ontology + mapping + source + graph via API.
3. Confirm DB table metadata via API.
4. Start build and watch products/nodes/edges increase.
5. Query brands and products when build completes.
6. Never touch `data/schemas/` on disk by hand.

---

## 12. Decision summary

| Topic | Decision |
| --- | --- |
| Official path | **Visible wizard steps** (section 3), not one opaque mega-call |
| Mapping | **API upload** required before build |
| DB | Inspect via API; secrets in `.env` / named connection |
| Build | `POST /api/graphs/{id}/build` + poll job / build status |
| Shortcut | `/api/kg/build` optional only |

## 13. Implementation status

| Phase | Status |
| --- | --- |
| P1 Mapping API | **Done** — `GET/POST/PUT /api/sources/{id}/mapping` |
| P2 Checklist + bootstrap | **Done** — `GET /api/setup/checklist`, `GET /api/bootstrap` |
| P3 Graph build + status | **Done** — `POST /api/graphs/{id}/build`, `GET …/build/status` |
| P4 DB connections | **Done** — `GET/POST /api/db/connections`, `POST …/test` |
| P5 Live job % during sync | Partial — status reads `ecom_sync_state` + job; soft progress estimate |
| Postman folder 0 | **Done** — `0 UI wizard (empty → live)` |

Restart the KG API after pull so new routes load.
