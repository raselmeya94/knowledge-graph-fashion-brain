# Run the product knowledge graph and Fashion Brain

Use this from the repository root, `D:\Rasel-P\R&D-KG`, in PowerShell.

The conda environment is `fashion-kg`. Graph code lives in `knowledge_graph`. The chatbot lives in `fashion_brain`.

## 1. Environment

Create the environment once:

```powershell
conda create -y -n fashion-kg python=3.11
conda activate fashion-kg
pip install -r requirements.txt
```

Every later command in this file assumes:

```powershell
conda activate fashion-kg
cd D:\Rasel-P\R&D-KG
$env:PYTHONPATH = "."
```

## 2. Raw CSV

Put the product file here:

```text
knowledge_graph/raw_data/own_products/
```

`knowledge_graph/raw_data/source_registry.json` already registers the source `own_products`. Ingestion reads only the files named in `knowledge_graph/raw_data/own_products/manifest.json`. A CSV that is sitting in the folder but is not listed in the manifest is left untouched.

`products_data.csv` is already registered. The manifest record id is `*`, so every row in that file becomes a product. `Code` is the product id. There are 124,682 codes, and none are repeated.

These are the column names, in file order:

`Created by`, `Code`, `Composition`, `Category`, `PLM Main Category`, `PLM Sub Category`, `PLM Sub Sub Category`, `Brand`, `Unit`, `Gauge`, `Gender`, `Sample Collection Type`, `Shipment Quantity`, `Shipment Date`, `Garment Type`, `Design For`, `Knit Machine Type`, `Knitting Time`, `Design Feature`, `Merchandiser`, `Department`, `Shopping Department`, `Season Year`, `Designer`, `Style Reference`, `Customer Style Reference`, `Style Feature`, `color`, `Weight`, `GSM`, `Cuttable Width`, `Sample Making Date`, `Current Stock`, `Current Location`, `Purchased By`, `Created At`, `Updated At`, `Buyer`.

| Column | Graph |
| --- | --- |
| `Code` | product id `plmex:product:{Code}` |
| `Style Reference` | product name. If the cell is empty, the name is the code |
| `Composition` | one `USES_MATERIAL` edge per percent, for example `50%Viscose 28%Polyester 22%Nylon` becomes Viscose, Polyester, and Nylon. The evidence text keeps the percent |
| `Category` | `BELONGS_TO_CATEGORY` |
| `Season Year` | `IN_SEASON` |
| `Buyer` | `ORDERED_BY` a company |
| `Brand` | `BRANDED_AS` a company |
| every other column | kept on the product record. An empty cell does not create an edge |

Rows are classified `internal`. This file is not marked synthetic.

After the CSV changes, refresh the file hash. This is the only command that writes into `raw_data`. Ingestion does not.

```powershell
python knowledge_graph/scripts/refresh_manifest_hashes.py
python -m knowledge_graph.cli validate
```

`validate` prints `Product data, registry, and ontology validated.` when the rows, hashes, and ontology agree.

## 3. Build the knowledge graph

```powershell
python -m knowledge_graph.cli ingest
```

What that command does:

1. Reads `source_registry.json` and the product manifest.
2. Loads each listed CSV or JSON row. Empty optional fields stay empty and do not create an edge.
3. Checks the content hash. A mismatch stops that row and does not change the raw file.
4. Writes one envelope per product under `knowledge_graph/generated/envelopes/own_products/`.
5. Builds the `own_products` graph from `Code`, `Composition`, `Category`, `Season Year`, `Buyer`, and `Brand`. The other columns stay on the product record.
6. Links a source material to a canonical material in `knowledge_graph/schemas/ontology.json` only on an exact id, exact name, or a curated alias. A shared word such as "cotton" inside "Cotton Blend" stays in the review queue.
7. Writes `knowledge_graph/generated/ingestion_state.json`, `graph_snapshot.json`, and `reports/latest.json`.

Run `ingest` again after a row changes. Unchanged hashes are skipped. A product removed from the manifest is marked deleted in the graph and is no longer returned in answers.

The default graph is in memory. Neo4j is optional and is not required for the steps below.

## 4. Neo4j + knowledge-graph API

Graph data lives in **Neo4j** (`FASHIONGPT_GRAPH_BACKEND=neo4j`). Snapshots under `data/generated/graph_snapshots/` are the reload source; serve/reload MERGEs them into Neo4j and chat fetches via Cypher (parameterized — no free-form Cypher from users).

```powershell
docker compose up -d
# wait until healthy, Browser http://127.0.0.1:7474  (neo4j / fashiongpt-local-dev)
python -m knowledge_graph.cli serve
```

Default API: http://127.0.0.1:8001. After scrape/commit, KG refreshes shared/master; admin commit also calls `POST /graphs/reload`. Fashion Brain never loads the graph — only `/brain/retrieve`.

Optional chat scope: pass `graph_id` on conversation/ask, or leave empty for multi-source.

To fall back to in-process memory (no Docker): set `FASHIONGPT_GRAPH_BACKEND=memory`.

## 5. Run Fashion Brain

Copy `.env.example` to `.env` in the repository root. Put the OpenRouter key only in `.env`:

```text
FASHIONGPT_MODEL_API_KEY=your-openrouter-key
FASHIONGPT_ANSWER_MODE=llm
FASHIONGPT_MODEL_PROVIDER=openrouter
FASHIONGPT_MODEL_NAME=~openai/gpt-sol-latest
```

Start the chatbot:

```powershell
python -m fashion_brain.cli serve
```

Leave that window open. The API is http://127.0.0.1:8000. Interactive docs are http://127.0.0.1:8000/docs. There is no page at `/`.

Check the model before asking questions:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

`model` is `configured` when the key is present. `missing_api_key` means `/ask` still returns the retrieved product text, without an OpenRouter rewrite.

## 5. Questions and answers

Products are classified `internal`. Send `X-Clearance: internal`.

In a second PowerShell window, with the server still running:

```powershell
$headers = @{ "X-Clearance" = "internal" }

Invoke-RestMethod http://127.0.0.1:8000/ask -Method Post -Headers $headers -ContentType "application/json" -Body '{"question":"Which materials are used by our demo products?"}'

Invoke-RestMethod http://127.0.0.1:8000/ask -Method Post -Headers $headers -ContentType "application/json" -Body '{"question":"Which products share a material?"}'

Invoke-RestMethod http://127.0.0.1:8000/ask -Method Post -Headers $headers -ContentType "application/json" -Body '{"question":"How many synthetic demo products are in the catalog?"}'
```

The same questions from the command line, without the server:

```powershell
python -m fashion_brain.cli ask "Which materials are used by our demo products?" --clearance internal
```

What an answer contains:

- `answer` is the OpenRouter wording when the key is set. The model is given only the retrieved product evidence.
- `evidence` lists the product id, field, and content hash the sentence came from.
- `unavailable` is set when the clearance cannot see the products, or when a live PLMEX count was requested. The chatbot does not guess that count.
- A public clearance sees no internal products:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/products?material=linen" -Headers @{ "X-Clearance" = "public" }
```

After adding rows, refresh hashes, then rebuild without stopping the server:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/ingestion/run -Method Post -ContentType "application/json" -Body '{"sources":["own_products"]}'
```

That trigger works only when `FASHIONGPT_ENV` is `development`.
