# Knowledge Graph — run on a new device

KG API is independent of Fashion Brain chat. Clone this repo, start Neo4j (or use memory), serve on `:8001`.

## What you need

| Requirement | Notes |
| --- | --- |
| Python 3.11+ | conda/venv OK |
| Docker (recommended) | Neo4j via `docker-compose.yml` |
| Optional Postgres | Only for `ecom_products` sync (`master_products`) |
| Optional Ollama | Only for document LLM construction |

Fashion Brain (`:8000`) and `frontend/` are **not** required to run KG.

## Setup

```powershell
cd <repo>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# Edit .env: set FASHIONGPT_NEO4J_PASSWORD to match docker-compose (default fashiongpt-local-dev)
```

Neo4j:

```powershell
docker compose up -d
# password in compose: neo4j / fashiongpt-local-dev
```

Or skip Docker and set in `.env`:

```env
FASHIONGPT_GRAPH_BACKEND=memory
```

## Start

Always from **repo root** (so `knowledge_graph` and `data/` resolve):

```powershell
$env:PYTHONPATH = "."
python -m knowledge_graph.cli serve
```

Check: http://127.0.0.1:8001/health · Swagger http://127.0.0.1:8001/docs

## Empty → e-com data (Postman)

1. Import `postman/knowledge_graph.postman_collection.json`
2. Run folder **`0 Ecom pipeline (empty → live)`**
3. Bodies live in `postman/bodies/` (ontology + graph + source already filled)
4. Before sync: ensure `data/schemas/ecom_master_products_mapping.json` exists (committed)
5. Set Postgres in `.env`, then `POST /api/sources/ecom_products/sync`

Full guide: [POSTMAN.md](POSTMAN.md)

## What is / is not committed

| Path | Commit? |
| --- | --- |
| `knowledge_graph/` | yes — code |
| `data/schemas/` | yes — mapping + seed ontologies |
| `postman/` | yes — KG collection + bodies |
| `data/generated/` | **no** — runtime (snapshots, registry cache, sync state) |
| `.env` | **no** — secrets |
| Neo4j / Postgres data | **no** — local services |

On a new machine the graph starts **empty** until you POST ontology/graph/source and sync (or restore your own Neo4j dump).

## Independence notes

- `knowledge_graph` may try to import `fashion_brain.trace` for debug logs; if missing, it falls back to normal logging.
- Package imports are under `knowledge_graph.*` only for serve/sync/query.
- `PYTHONPATH=.` at repo root is required unless you `pip install -e .`.
