# Knowledge graph layout

**Code** stays in `knowledge_graph/`. **Data** lives in the sibling root `data/`.

```text
R&D-KG/
├── knowledge_graph/          # Python only
│   ├── graph/                # CODE: memory/Neo4j store + templates (not snapshots)
│   ├── construction/
│   ├── kg_registry/
│   ├── pipeline/
│   ├── retrieval/
│   └── …
│
└── data/                     # ALL data (see data/README.md)
    ├── schemas/              # Ontologies
    │   └── ontologies/       # Custom confirmed ontologies (wiki, …)
    ├── raw/                  # PLM / source files
    ├── config/
    └── generated/
        └── graph_snapshots/  # Saved graph instances (not the graph/ package)
```

| Name | What it is |
| --- | --- |
| `knowledge_graph/graph/` | Source code for the graph engine |
| `data/generated/graph_snapshots/<source>/` | Individual KG snapshots (news, own_products, wiki, …) |
| `data/generated/graph_snapshots/master/` | Persisted **shared interconnect** (canonicals + `RESOLVES_TO`) |

**Model:** one individual KG per similar corpus; `shared`/`master` interconnect when similar entities link; chat fans out across individuals and joins via shared when links exist. Fashion Brain never loads snapshots — start KG first so memory is warm (`POST /graphs/reload` after commits).

Relative `.env` paths resolve against `data/`.
