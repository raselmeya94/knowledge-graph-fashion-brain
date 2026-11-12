# Knowledge Graph service — API data retriever

Port **8001**. This service stores and serves product/knowledge graphs. It does **not** run LLM, Ollama, or RAG orchestration.

## In scope

- Registry, sources, sync, graph build/status
- Neo4j read/write for graph data
- `GET /brain/runtime_schema`, `GET /brain/brands`
- `POST /brain/cypher` (read-only execution)
- `POST /brain/link_entities` (exact + full-text lookup)
- `POST /brain/retrieve` (templated retrieve)
- Health without Ollama

## Linking vs embeddings (important)

| In KG? | What | Why |
|--------|------|-----|
| **Yes** | **Linking** = `POST /brain/link_entities` | Pure **Neo4j lookup**: match a string to Brand/Material/… node ids via exact + full-text index. This is data retrieval, like a search API. |
| **No** | **Embeddings** (sentence-transformers) | Removed (`linking/embeddings.py` deleted). |

Fashion Brain decides *which* question words to send; KG only resolves them against the graph.

## Out of scope (removed from API surface)

- `/kg/*` construction / Ollama / vision extract → **HTTP 410**
- Sentence-transformer embedding warm / vector link merge
- Chat / Text2Cypher / LangGraph → **Fashion Brain (`:8000`)**

Legacy `knowledge_graph/construction/` code may still exist on disk for reference; it is **not registered** on the KG FastAPI app.
