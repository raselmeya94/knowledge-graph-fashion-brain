# Fashion Brain ↔ Knowledge Graph boundary

## Rule

| Layer | Owns | Does **not** own |
|-------|------|------------------|
| **`knowledge_graph`** (API `:8001`) | Neo4j, ontology, build/sync, registry, `/brain/cypher`, `/brain/brands`, `/brain/runtime_schema`, `/brain/link_entities`, full-text lookup | LLM agents, LangGraph, RAG routing, answer prose |
| **`fashion_brain`** (API `:8000`) | Question→answer flow, LLM agents, Text2Cypher prompts, Graph RAG state machine, question candidate picking | Graph storage / ingestion |

KG is already an **API-complete data system**. Fashion Brain is the **chat / RAG client** that calls those APIs.

```
User question
    → Fashion Brain (orchestrate, LLM, RAG)
         → HTTP KG API (schema, brands, link, cypher)
              → Neo4j
    → Fashion Brain formats answer
```

## Why this was confused before

Settings such as `graph_rag_mode` / `text2cypher_enabled` live in shared `.env` / `knowledge_graph.config.Settings` for one env file — **but only Fashion Brain should interpret them**. The KG service must not grow LangGraph/RAG pipelines inside it.

Question→candidate n-grams live in `fashion_brain/linking/`.  
Neo4j lookup stays in `knowledge_graph/linking/` (`link_one`, `link_by_label`, `/brain/link_entities`).

## Current Fashion Brain Q&A (high level)

1. Context / follow-up  
2. Pick graph (registry)  
3. Call KG for schema + brands (+ optional link)  
4. LLM Text2Cypher in Fashion Brain  
5. `POST /brain/cypher` on KG  
6. Answer in Fashion Brain  

Restart Fashion Brain after `.env` changes: `python -m fashion_brain.cli serve`
