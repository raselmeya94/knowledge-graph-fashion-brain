# Knowledge graph

`knowledge_graph` stores two kinds of facts.

- The product catalog in `data/raw/own_products` is ingested into the `own_products` graph and linked to canonical materials, fabrics, categories, seasons, and suppliers.
- Documents, email, HTML, JSON, and images go through the construction pipeline. Proposed relationships stay in a review queue until they are approved and committed to the `documents` graph.

Fashion Brain is a separate chat process. It does not load graph snapshots; it calls this API for evidence (`POST /brain/retrieve`). Start the knowledge-graph API first (it loads the catalog and snapshots once):

```powershell
$env:PYTHONPATH = "."
python -m knowledge_graph.cli serve
```

That listens on http://127.0.0.1:8001.

**New device / clone → run:** see **[FRESH_DEVICE.md](FRESH_DEVICE.md)**  

**Postman rebuild (empty → full, real bodies):** see **[POSTMAN.md](POSTMAN.md)**  
Request JSON lives in [`../postman/bodies/`](../postman/bodies/). Collection: [`../postman/knowledge_graph.postman_collection.json`](../postman/knowledge_graph.postman_collection.json).

**API documentation:** see **[API.md](API.md)** (auth, routes, request/response examples).  
Interactive Swagger: http://127.0.0.1:8001/docs · OpenAPI JSON: [`openapi.json`](openapi.json).

Then start Fashion Brain on port 8000 with `FASHIONGPT_KG_URL=http://127.0.0.1:8001`.

```powershell
python -m knowledge_graph.cli tools
python -m knowledge_graph.cli pipeline --kind email --file linen.eml
python -m knowledge_graph.cli approve <review-id>
python -m knowledge_graph.cli commit <review-id>
```

`pipeline` prints a review id. `approve` marks that review approved. `commit` writes it into the graph. Add `--model` when Ollama should propose extra relationships, or `--commit` to approve and save in one step.

Fashion Brain is the separate chat process on port 8000: `python -m fashion_brain.cli serve`. Rebuild the product catalog with `python -m knowledge_graph.cli ingest`.

Paths in `.env` are relative to the repo **`data/`** root (`raw`, `generated`, `schemas/…`). Copy `.env.example` to `.env` at the repository root.

## Layout

See **[LAYOUT.md](LAYOUT.md)** and **[`../data/README.md`](../data/README.md)**. Short version:

```text
knowledge_graph/   # code only (includes graph/ package = engine code)
data/              # all data
  schemas/         # ontologies (+ schemas/ontologies/<custom>.json)
  raw/             # PLM files + source_registry.json
  generated/       # runtime: graph_snapshots/, envelopes, reviews, scrape jobs
```

`knowledge_graph/graph/` is **code**. Snapshot files are under `data/generated/graph_snapshots/` — not the same thing.

## Product catalog

Registered source: `own_products`. The connector reads JSON and CSV from disk. The SQL connector stays disabled, so this package does not connect to a live PLMEX database.

Required product fields: `product_id`, `product_name`, `category`, `season`, `material`, `classification`, and `disclaimer`. Empty optional fields (fabric, supplier, manufacturer) do not create those edges.

Add a row:

1. Edit `data/raw/own_products/products.json` or `products.csv`.
2. Refresh the manifest hash from the repository root:

```powershell
$env:PYTHONPATH = "."
python knowledge_graph/scripts/refresh_manifest_hashes.py
```

3. Rebuild while the API is running:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/ingestion/run -Method Post -ContentType "application/json" -Body '{"sources":["own_products"]}'
```

`POST /ingestion/run` is available when `FASHIONGPT_ENV=development`.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/sources` | Registered sources |
| GET | `/ingestion/status` | Ingestion state |
| POST | `/ingestion/run` | Development rebuild |
| GET | `/products` | Product search |
| POST | `/brain/retrieve` | Evidence pack for Fashion Brain |
| POST | `/graphs/own_products/query` | Approved product templates |
| POST | `/graphs/shared/query` | Canonical entities and shared materials |

Graph calls take a template name and simple parameters. They do not accept Cypher. Templates: `list_products`, `product_materials`, `trace`, `review_queue`, `products_sharing_material`, `canonical_materials`.

Product relationship types include `USES_MATERIAL`, `USES_FABRIC`, `BELONGS_TO_CATEGORY`, `IN_SEASON`, `SUPPLIED_BY`, `MANUFACTURED_BY`, `ORDERED_BY`, and `BRANDED_AS`.

Default local backend is **Neo4j** (`FASHIONGPT_GRAPH_BACKEND=neo4j`). Start `docker compose up -d` from the repository root. Snapshots under `data/generated/graph_snapshots/` reload into Neo4j; chat queries use parameterized Cypher inside the repository (templates only — no free-form Cypher). Set `FASHIONGPT_GRAPH_BACKEND=memory` for an in-process store with no Docker.

## Scraping construction (step APIs)

Buyer / scraped documents use a paragraph-aware path (separate from the one-shot `/kg/pipeline`):

```text
POST /scraping/extract[/upload]   pdf | doc | docx | json | text → file_id + full text
POST /scraping/chunk              paragraph chunks with same file_id
POST /scraping/propose            per-paragraph relations (novelty=existing|new)
POST /scraping/queue              review for admin approve/reject
POST /kg/reviews/{id}             approve selected edges
POST /scraping/commit             existing or new ontology + graph → master snapshot
```

Propose uses the selected ontology’s relationship names as **hints**. New `UPPER_SNAKE` types are allowed and tagged `novelty: "new"` / `needs_ontology_approval`. Commit with `"ontology": {"mode": "new", "clone_from": "news", ...}` creates ontology + graph and refreshes master so `trace` works on the new graph.

Jobs are stored under `generated/scraping/files/{file_id}.json`.

## Ontologies and graphs

The API manages named ontologies and graphs (not only the two built-in files).

Built-ins (current e-comm-first registry may list only product graph until others are registered):

| Graph id | Ontology id | Role |
| --- | --- | --- |
| `ecom_products` | `ecom` | E-comm `master_products` synced into Neo4j |
| `own_products` | `product` | Optional PLM catalog (file ingest) |
| `news` | `news` | PDF / Word / plain-text news construction |
| `documents` | `product` | Email, HTML, JSON, image construction |

Admin lifecycle (all HTTP):

```text
POST /ontologies              create or clone an ontology
POST /graphs                  create a graph bound to that ontology
POST /extract                 format-wise raw text
POST /relationships/propose   propose edges for a graph_id
POST /kg/pipeline[/upload]    extract → propose → queue review (pass graph_id)
POST /kg/reviews/{id}         approve / edit / reject
DELETE /kg/reviews/{id}/relationships/{rel_id}
POST /kg/reviews/{id}/commit  write into the selected graph (+ extend ontology if needed)
DELETE /graphs/{id}/edges/{edge_id}
POST /graphs/{id}/sync|rebuild
```

Omit `graph_id` to keep the old defaults: PDF/doc/text → `news`, other kinds → `documents`. Registry state lives in `generated/kg_registry.json`. Custom ontology files live under `generated/ontologies/`.

## Construction pipeline

A caller submits one source. The pipeline picks an extractor, proposes relationships, and stores a review. Approved relationships are the only ones written to the selected graph.

```text
source
  -> extract text, HTML, JSON, or an image description
  -> extract metadata
  -> read relationships already in the selected graph
  -> propose existing and new relationships
  -> queue a review (tagged with graph_id + ontology_id)
  -> on approval, ingest into that graph
```

| Source | Tool | What it produces |
| --- | --- | --- |
| Email (`.eml` or text with `From` and `Subject`) | `extract_text` | Body text. `From` and `To` become `SENT_BY` and `SENT_TO`. |
| PDF, `.docx`, `.doc`, `.txt`, `.md` | `extract_text` | Plain text. PDF needs `pypdf`. A legacy `.doc` keeps readable text and drops formatting. |
| HTML | `extract_html` | Title, visible text, and links. The page is labeled product, newsletter, or article. The HTML you submit is what gets read. |
| JSON object or list | `parse_json` | Structured fields. `author`, `publisher`, `topic`, `material`, and a `relationships` array become proposals. |
| PNG, JPEG, WEBP, GIF | `understand_image` | A text description from the Ollama vision model. Relationship identification then runs on that text. |

Files larger than 8 MB are rejected. Set `kind` when the filename does not make the type obvious: `pdf`, `docx`, `doc`, `text`, `email`, `html`, `json`, or `image`.

### Tools

Each tool can run on its own through `POST /kg/tools/{name}`.

| Tool | Role |
| --- | --- |
| `extract_text` | PDF, Word, email, and plain text. `clean: true` asks Ollama to tidy the text. |
| `extract_html` | Title, text, links, and page kind. |
| `parse_json` | JSON object or list, plus a text view of the fields. |
| `understand_image` | Image description through Ollama. |
| `extract_metadata` | Title, kind, size, addresses, links, and dates. |
| `list_relationships` | Relationships already stored in the graph. |
| `identify_relationships` | Mark existing relationships and propose new allowlisted ones. |
| `structure_data` | One payload of entities and relationships. |
| `queue_review` | Save that payload for review. |
| `update_review` | Approve, reject, or edit proposals. |
| `ingest_graph` | Write approved relationships into the graph. |
| `ollama` | Local model call. `action` is `health`, `complete`, or `describe_image`. |

Ollama is used for image descriptions, optional text cleanup, and extra relationship proposals. Header, phrase, and JSON-field extraction still runs when Ollama is down. Check it with `GET /kg/ollama`.

```text
FASHIONGPT_OLLAMA_BASE_URL=http://127.0.0.1:11434
FASHIONGPT_OLLAMA_MODEL=llama3.1:8b
FASHIONGPT_OLLAMA_VISION_MODEL=
FASHIONGPT_OLLAMA_TIMEOUT_SECONDS=60
```

### Relationship review

Proposals are stored in `generated/construction/reviews.json`. A relationship is `existing` when the same type and endpoint names are already in the graph, and `new` otherwise. `review_state` starts as `pending`.

Document relationship types allowed by `schemas/ontology.json`:

| Type | From | To |
| --- | --- | --- |
| `MENTIONS` | Document | Person, Organization, Topic, Location, Event, Product, Material, Company |
| `AUTHORED_BY` | Document | Person |
| `PUBLISHED_BY` | Document | Organization |
| `ABOUT` | Document | Topic |
| `SENT_BY`, `SENT_TO` | Document | Person or Organization |
| `OCCURS_AT` | Event | Location |
| `RELATED_TO` | Document, Person, Organization, Topic, or Material | The same type, plus Person to Organization |

A proposal whose type or endpoint pair is not on that allowlist is queued with `needs_ontology_approval: true` and a warning. Evidence still has to name both endpoints. Approving and committing that edge writes it into the graph and adds the triple to the ontology file for future extractions. `auto_commit` skips those edges so ontology growth stays a human decision.

### Construction endpoints

These routes use the same authentication as the rest of the API. Send `X-Clearance: internal`. If `FASHIONGPT_API_TOKEN` is set, also send `Authorization: Bearer <token>`.

| Method | Path | Purpose |
| --- | --- | --- |
| GET/POST | `/ontologies` | List or create ontologies |
| GET/PUT | `/ontologies/{id}` | Read or replace an ontology |
| POST/DELETE | `/ontologies/{id}/pairs` | Add or remove one allowlist pair |
| GET/POST | `/graphs` | List or create graphs |
| GET | `/graphs/{id}` | Graph metadata and counts |
| POST | `/graphs/{id}/sync` | Sync / snapshot (product graphs rebuild sources) |
| POST | `/graphs/{id}/rebuild` | Rebuild from sources or restore snapshot |
| DELETE | `/graphs/{id}` | Soft-delete a graph (`force` for built-ins) |
| DELETE/PATCH | `/graphs/{id}/edges/{edge_id}` | Soft-delete or patch a committed edge |
| POST | `/extract` | Format-wise raw text extraction |
| POST | `/relationships/propose` | Propose relationships for a `graph_id` |
| GET | `/kg/tools` | Tool names and descriptions |
| POST | `/kg/tools/{name}` | Run one tool |
| GET | `/kg/ollama` | Ollama reachability and installed model names |
| GET | `/kg/relationships` | Stored relationships (`?graph=`) |
| POST | `/kg/pipeline` | Run the pipeline from JSON (`graph_id` optional) |
| POST | `/kg/pipeline/upload` | Run the pipeline from a file (`graph_id` form field) |
| GET | `/kg/reviews` | Review queue (`?graph_id=`) |
| GET | `/kg/reviews/{id}` | One review |
| POST | `/kg/reviews/{id}` | Approve, reject, or edit |
| DELETE | `/kg/reviews/{id}/relationships/{rel_id}` | Reject or drop one proposal |
| POST | `/kg/reviews/{id}/commit` | Save approved relationships into the review's graph |

`POST /kg/pipeline` accepts `text`, `filename`, `kind`, `json_data`, `data_base64`, and optional `graph_id`. `use_model` defaults to `true`. `auto_commit` defaults to `false`, so the pipeline stops at the review queue. `clean` asks Ollama to tidy extracted text.

You do not have to use the combined pipeline. Call the stages yourself:

```text
POST /extract
POST /relationships/propose
POST /kg/tools/structure_data
POST /kg/tools/queue_review
POST /kg/reviews/{id}            # approve / edit
POST /kg/reviews/{id}/commit
```

Or any single tool via `POST /kg/tools/{name}`.

Email text:

```powershell
$headers = @{ "X-Clearance" = "internal" }
$body = @{
  kind = "email"
  filename = "linen.eml"
  use_model = $false
  text = @"
From: Ada Lovelace <ada@example.com>
To: Grace Hopper <grace@example.com>
Subject: Linen shipment

Published by Northwind Mills.
"@
} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8001/kg/pipeline -Method Post -Headers $headers -ContentType "application/json" -Body $body
```

That proposal contains `SENT_BY` Ada Lovelace, `SENT_TO` Grace Hopper, and `PUBLISHED_BY` Northwind Mills. Approve and commit it:

```powershell
$id = "<review id from the pipeline response>"
Invoke-RestMethod "http://127.0.0.1:8001/kg/reviews/$id" -Method Post -Headers $headers -ContentType "application/json" -Body '{"decision":"approved"}'
Invoke-RestMethod "http://127.0.0.1:8001/kg/reviews/$id/commit" -Method Post -Headers $headers
```

To change one relationship before approval, send `edits` with that relationship's `id`, for example a new `end_name` or `end_type`. `decision` applies to every pending relationship unless `relationship_ids` limits it.

Upload a file:

```powershell
curl.exe -X POST "http://127.0.0.1:8001/kg/pipeline/upload" -H "X-Clearance: internal" -F "file=@note.docx" -F "kind=docx" -F "use_model=false"
```

JSON with explicit relationships:

```json
{
  "kind": "json",
  "use_model": false,
  "json_data": {
    "title": "Lookbook",
    "author": "Ada Lovelace",
    "topic": "Linen",
    "relationships": [
      {"type": "ABOUT", "source": "document", "target": "Linen", "target_type": "Topic"}
    ]
  }
}
```

A committed document is a `Document` node in the `documents` graph. People, organizations, topics, and the other endpoints are `SourceEntity` nodes. The memory backend also rewrites `generated/graph_snapshot.json` after a successful commit.

## Tests

From the repository root:

```powershell
$env:PYTHONPATH = "."
python -m pytest -q tests/test_construction_pipeline.py
```

## Postman

Import `postman/KnowledgeGraph.postman_collection.json`. It contains only the `/kg` construction routes and calls port 8001. Product search, catalog ingestion, graph templates, and chat stay in `postman/FashionGPT.postman_collection.json` on port 8000.

`base_url` defaults to `http://127.0.0.1:8001`. Leave `api_token` empty when `FASHIONGPT_API_TOKEN` is unset. Run **Run email pipeline** first. That request stores `review_id` and `relationship_id` for **Get review**, **Edit review**, **Approve review**, and **Commit review**.
