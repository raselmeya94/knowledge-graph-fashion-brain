# Plan: Connect Postgres (PLMEX + e-comm) to the knowledge graph

## Short answer

| Store | Role | Continuously updating? |
| --- | --- | --- |
| **Postgres (PLMEX)** | System of record for design / sample / PLM products | Yes — source of truth |
| **Postgres (e-comm)** | System of record for live listings, price, stock, orders | Yes — source of truth |
| **Neo4j** | Knowledge graph: entities, relationships, shared materials, cross-source links | Yes — *derived* store, rebuilt or synced from Postgres |
| **Catalog index / envelopes** (`data/generated/`) | Fast PLM-style filters for Fashion Brain (`garment`, `color`, buyer, …) | Refreshed on sync / ingest |

**Do not replace Postgres with Neo4j.**  
Postgres keeps operational data. Neo4j (plus the catalog index) holds the *graph view* Fashion Brain queries.

**Target for products: live Postgres, not local CSV/JSON.**  
Local files under `data/raw/…` are only the *current* demo path. With live access, `own_products` (PLMEX) and e-comm stop reading disk as primary; adapters `SELECT` from Postgres (or call GET APIs), then write the same envelopes / catalog index / Neo4j projection.

```text
┌─────────────────┐     ┌──────────────────┐
│  PLMEX Postgres │     │ E-comm Postgres  │
│  LIVE SoR       │     │ LIVE SoR         │
│  (not CSV)      │     │ (not CSV)        │
└────────┬────────┘     └────────┬─────────┘
         │  read-only DSN / GET   │
         ▼                        ▼
   ┌─────────────────────────────────────┐
   │  Knowledge Graph service (:8001)    │
   │  SQL adapters (primary)             │
   │  → envelopes → catalog_index        │
   │  → Neo4j (relationships / shared)   │
   │  CSV/JSON = offline fallback only   │
   └──────────────────┬──────────────────┘
                      │ POST /brain/retrieve
                      ▼
              Fashion Brain (:8000)
```

---

## Today vs live target

| | **Today (demo)** | **Live target** |
| --- | --- | --- |
| PLM source | `own_products` file adapter — CSV/JSON in `data/raw/…` | Same source id (or `plmex_products`) — **Postgres read** via DSN / GET |
| E-comm | Not connected | Second Postgres source |
| SQL flags | `FASHIONGPT_SQL_ENABLED=false`; connector refuses DSN | Enabled; read-only DSNs; real `discover()` from DB |
| Fashion Brain | Calls KG only | Still calls KG only — never opens Postgres itself |
| Local CSV | Primary | Optional dump / air-gapped fallback |

So: you do **not** keep shipping products as local files once live is on. Files are a bootstrap/offline path; production ingest is always from the live DB views.

---

## Target architecture (two databases)

Register **two product sources** in `source_registry.json` (names illustrative):

| Source id | System | Postgres content (examples) | Graph id |
| --- | --- | --- | --- |
| `plmex_products` | PLMEX | style code, category, garment type, composition, buyer, season, color, samples | `plmex_products` |
| `ecom_products` | Online store | SKU, listing title, price, stock, channel, PLM style link, sales | `ecom_products` |

Then use the existing **shared / interconnect** layer (`shared` / `master`) so the same style can resolve across PLM and e-comm (e.g. `plmex:product:01201900001` ↔ `ecom:sku:SKU-123` via `RESOLVES_TO` or a curated `LISTED_AS` edge).

### Responsibilities

| Layer | Owns |
| --- | --- |
| Postgres PLMEX | Authoring, samples, BOM, buyer, season |
| Postgres e-comm | Sellable SKUs, price, stock, channel metrics |
| Adapters in `knowledge_graph/adapters/` | Map each DB row → normalized product envelope |
| Neo4j | Product–material–buyer–brand–season edges; cross-source links |
| Catalog index | Fast “how many red sweaters?” style filters |
| Fashion Brain | NL question → `/brain/retrieve` → grounded answer |

---

## Connection pattern (recommended)

### Prefer sync into KG, not chat-time SQL

Fashion Brain questions should hit **KG retrieval** (catalog + Neo4j templates), not ad-hoc SQL from the chat process.

Reasons:

- Clearance / classification stay in one place.
- Ontology + material resolution stay consistent.
- Chat latency does not depend on PLM/e-comm DB load.
- You can still answer “live PLMEX count” later via a **narrow, approved** live query path (`live_sql` intent already exists).

### Two sync modes

**A. Batch / CDC sync (default for catalog + graph)**

1. Read-only Postgres credentials (separate DSNs for PLMEX and e-comm).
2. Adapter `discover()` pages `SELECT` (or listens to CDC / `updated_at` watermark).
3. Normalize → content hash → envelope (same as file ingest).
4. Rebuild or upsert catalog index + Neo4j nodes/edges for that source.
5. Schedule: every N minutes, or webhook after PLM publish.

**B. Live read path (optional, narrow)**

- Only for questions that *must* be real-time (stock now, live order count).
- Implement behind `SqlProductConnector` with allowlisted queries (never free-form SQL from the LLM).
- Return caveats: “live e-comm stock” vs “imported PLM snapshot.”

Start with **A** for both systems; add **B** only where freshness matters.

---

## How each Postgres connects

### 1. PLMEX Postgres → live `own_products` (or rename `plmex_products`)

Replace the file `discover()` path with SQL (same normalize / hash / envelope pipeline):

```text
PLMEX Postgres view kg_plmex_products
        │
        ▼
OwnProductsAdapter.discover()   # SELECT … WHERE updated_at > watermark
        │
        ▼
normalize_product() → envelope → catalog_index + Neo4j
```

- Map DB columns to the existing product shape (`code`, composition, category, garment type, color, buyer, season, …) — same fields Fashion Brain already filters.
- Source identity prefix stays `plmex`.
- **Stop treating CSV as primary.** Keep file adapter only if `FASHIONGPT_SQL_ENABLED=false` or DSN unreachable (dev / offline).
- Watermark sync (or CDC) so continuous PLM writes show up in KG without re-copying files.

### 2. E-comm Postgres → `ecom_products`

- Different schema: SKU, price, stock, channel, `style_reference` / PLM code foreign key.
- New adapter (do not force e-comm rows through PLM-only columns).
- Graph edges examples: `LISTED_AS`, `HAS_STOCK`, `SOLD_ON_CHANNEL`, plus link to PLM product when style code matches.
- Catalog filters for e-comm: sku, channel, in_stock, price band — extend `CATALOG_FIELDS` carefully so PLM and e-comm fields do not collide.

### 3. Cross-source resolution

| Link | Rule |
| --- | --- |
| Exact PLM `Code` = e-comm `style_code` | Auto `RESOLVES_TO` / `LISTED_AS` |
| Fuzzy name only | Review queue (same as materials today) |
| One PLM style → many SKUs (color/size) | Parent product + child SKU nodes |

Fashion Brain can then answer:

- “How many PRIMARK sweaters in red?” → PLM catalog  
- “Which of those are in stock online?” → join via shared link + e-comm stock field  
- “Materials shared across PLM and e-comm styles?” → Neo4j shared interconnect  

---

## Neo4j vs Postgres — decision rules

| Question type | Use |
| --- | --- |
| Exact row attributes, counts, filters (color, buyer, gauge) | Catalog index / product repo (fed from Postgres sync) |
| Relationships (uses material, same buyer, shared material across products) | Neo4j |
| Cross system (PLM style ↔ e-comm SKU) | Neo4j interconnect + both catalogs |
| Operational write (create order, update stock) | **Postgres only** — KG never writes back by default |
| Live “right now” stock | Optional allowlisted live SQL on e-comm Postgres |

Neo4j is the **graph projection**, not the warehouse.

---

## Two data worlds (do not mix)

| World | What you have | Role |
| --- | --- | --- |
| **A. PLM practice** | Local `own_products` CSV/JSON | Stand-in to learn PLMEX-shaped fields. Keep as-is for now. |
| **B. Live e-comm** | Postgres already running (schema below) | First real live source into the KG. |

**What next = connect world B first.** Do not wait for live PLMEX Postgres.

---

## Confirmed e-comm schema (Fashion Brain scrape DB)

Already in Postgres:

| Table | Use for KG |
| --- | --- |
| **`master_products`** | **Primary ingest surface** — one row per `(brand, master_id)`; shared cols both brands fill |
| `zara_products`, `primark_products`, `matalan_products`, `c_and_a_products`, `walmart_products` | Brand extras only when needed (ids, sizes, badges, Walmart rating, …) |
| `*_colorways` | Child color rows — optional second-pass for color filters / images |
| `product_change_events` | Later: “what changed” questions; not needed for v1 catalog |
| `scrape_jobs` | Ops only; not product graph |

Brands in `master_products.brand`: `zara`, `primark`, (+ matalan / c-and-a / walmart as they land on master).

---

## Field map: `master_products` → KG product envelope

Use **master only** for v1. Map into a new source (e.g. `ecom_products`), not into PLM `own_products`.

| Postgres (`master_products`) | KG / Fashion Brain |
| --- | --- |
| `(brand, master_id)` | `product_id` = `{brand}:{master_id}` |
| `name` | `product_name` / title |
| `brand` | `brand` (+ filter) |
| `category` / `family` / `subfamily` / `section` | `category` / subcategory-like fields |
| `category_path` | keep as path text / tags |
| `composition` + `composition_detail` | `material` / composition (+ Neo4j material edges) |
| `construction` | construction / tags |
| `colors` (JSONB) + colorways later | `color` (flatten primary or explode one row per color in v2) |
| `availability`, `in_stock_count` | stock / availability filters |
| `price`, `currency`, `price_min`/`price_max` | price fields (e-comm catalog) |
| `product_url`, `image`, `images` | evidence / UI links |
| `content_hash`, `updated_at` | ingest watermark / skip unchanged |
| `scraped_at`, `catalog_id`, `scrape_job_id` | provenance on envelope |

Do **not** force PLM-only fields (`buyer`, `gauge`, `garment_type`) onto e-comm rows — leave empty or map only when present (`type` / `kind` on brand tables if useful).

---

## Canonical pipeline (yes — you are right)

```text
Postgres (raw)
    → schema / normalize
    → relationships
    → store in Neo4j
    → user query
    → Fashion Brain response
```

That is the correct live path. Chat never opens Postgres.

### End-to-end (e-comm `master_products`)

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  A. DATA PLANE (sync job — runs on schedule / watermark)                │
│                                                                         │
│  [1] Postgres                                                           │
│      master_products (brand, master_id, name, composition, …)           │
│           │  read-only DSN                                              │
│           │  only rows where updated_at / content_hash changed          │
│           ▼                                                             │
│  [2] Schema / normalize                                                 │
│      ecom_master_products_mapping.json                                  │
│      ecom_master.normalize_master_row(row)                              │
│      → product_id = ecom:{brand}:{master_id}                            │
│           ▼                                                             │
│  [3] Relationships                                                      │
│      ecom-products-ontology.json (allowlist)                            │
│      ecom_master.build_relationships(product)                           │
│      HAS_BRAND, BELONGS_TO_CATEGORY, USES_MATERIAL, HAS_COLOR, …        │
│           ▼                                                             │
│  [4] Store Neo4j (Docker volume — outside project folder)               │
│      MERGE Product + Brand/Material/Category/Color nodes                │
│      MERGE relationship edges                                           │
│      indexes on brand, category, material, color, availability          │
│      (no growing envelopes / catalog_index inside the repo)             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  B. QUERY PLANE (every user question)                                   │
│                                                                         │
│  [5] User query  →  Fashion Brain (:8000)                               │
│           │  intent + filters (brand, color, …)                         │
│           ▼                                                             │
│  [6] KG retrieve  →  POST :8001/brain/retrieve                          │
│           │  approved Neo4j templates / Cypher — not free SQL           │
│           ▼                                                             │
│  [7] Findings (counts, samples, relationship facts)                     │
│           ▼                                                             │
│  [8] Response agent (LLM)  →  grounded answer to user                   │
└─────────────────────────────────────────────────────────────────────────┘
```

| Step | What | Who runs | Grows where |
| --- | --- | --- | --- |
| 1 Postgres | Raw scrape / PLM | Scraper / PLMEX | Postgres disk |
| 2 Schema | Map columns → product fields | Sync job | Code/schemas only (fixed) |
| 3 Relationships | Build edges from ontology | Sync job | Code only (fixed) |
| 4 Neo4j store | Upsert nodes + edges | Sync job | Neo4j volume |
| 5–8 Ask | User → retrieve → answer | Fashion Brain + KG API | No project file growth |

**Rule:** sync writes Neo4j; ask reads Neo4j. Schema/ontology JSON in the repo stays small and almost never grows.

---

## Zero growth inside the project folder

Today demo mode writes many files under `data/generated/` (envelopes + `catalog_index.jsonl`). That **will** grow. If you want **no size increase inside the repo/project**:

| Store | Keep growing? | Where |
| --- | --- | --- |
| Postgres (e-comm / PLM) | Yes | Outside project (DB server) |
| Neo4j | Yes | Docker volume / DB disk — **not** under `D:\Rasel-P\R&D-KG` |
| Envelopes / catalog_index.jsonl | **No** | Skip, or put on external path / object storage |
| Git repo | **No** | Already gitignores `data/generated/` |

**Target pipeline (no project file growth):**

```text
Postgres (raw)  ──sync──►  Neo4j (relationships + product props + facet indexes)
                              │
                              ▼
                         Fashion Brain  (query Neo4j / SQL views only)
```

Practical rules:

1. **Do not write per-product JSON under `data/generated/envelopes/`** for e-comm (or PLM once live).
2. **Do not maintain a growing `catalog_index.jsonl` in the project** — use Neo4j indexes, or a small Postgres **facet table** / materialized view (`DISTINCT brand`, colors, …), or Redis.
3. Point `FASHIONGPT_GENERATED_DIR` off-repo if you still need temp files (e.g. `D:\kg-runtime\generated`).
4. Neo4j data dir = Docker volume, never a folder inside the git project.
5. Sync job: read Postgres → upsert Neo4j only (and optional external facet store).

Tradeoff: you lose easy “open one envelope file” debugging; use Neo4j Browser / SQL instead.

---

## E-comm Neo4j relationship schema (v1 from `master_products`)

Entity labels (reuse ontology where possible):

| Node label | Key / example | From column |
| --- | --- | --- |
| `Product` | `ecom:{brand}:{master_id}` | `brand` + `master_id` |
| `Brand` | `zara`, `primark`, … | `brand` |
| `Material` | parsed from composition | `composition` / `composition_detail` |
| `GarmentCategory` | category / family / subfamily | `category`, `family`, `subfamily` |
| `Color` | color name | `colors` JSONB or colorways |
| `Availability` | `in_stock` / text | `availability` |
| `Currency` / price props | on product or `HAS_PRICE` | `price`, `currency` |

Relationships (v1 — write these first):

```text
(:Product)-[:BRANDED_AS]->(:Brand)
(:Product)-[:BELONGS_TO_CATEGORY]->(:GarmentCategory)      # category
(:Product)-[:HAS_SUB_CATEGORY]->(:GarmentCategory)       # family / subfamily
(:Product)-[:USES_MATERIAL {percentage?}]->(:Material)   # from composition_detail
(:Product)-[:HAS_COLOR]->(:Color)                        # each color in colors[]
(:Product)-[:HAS_AVAILABILITY]->(:Availability)          # or property on Product
(:Product)-[:HAS_SECTION]->(:Section)                    # section (optional)
```

Later (v2):

```text
(:Product)-[:HAS_COLORWAY]->(:Colorway)-[:HAS_COLOR]->(:Color)   # *_colorways
(:Product)-[:SAME_AS|RESOLVES_TO]->(:Product)                    # link to PLM own_products
```

Product node properties (not everything — keep Neo4j lean):  
`name`, `reference`, `price`, `currency`, `in_stock_count`, `product_url`, `image`, `source=ecom`, `graph=ecom_products`.

Heavy JSON (`images`, `prices_seen`, full dumps) stays on the **envelope / catalog**, not as Neo4j clutter.

---

## What next (ordered)

### Step 0 — Access (today)

1. Read-only Postgres role for KG (SELECT on `master_products`, later `*_colorways`).
2. Put DSN only in `.env` (never commit):

```text
FASHIONGPT_ECOM_DSN=postgresql://kg_readonly:...@host:5432/fashion_brain
FASHIONGPT_SQL_ENABLED=true
```

3. Sanity check: `SELECT brand, count(*) FROM master_products GROUP BY brand;`

### Step 1 — Register source + adapter (next build)

1. Add `ecom_products` to `source_registry.json` (`connector: authorized_sql`).
2. New adapter: page `master_products` ordered by `updated_at`, normalize → envelope, hash = `content_hash` or canonical JSON hash.
3. Write envelopes under `data/generated/envelopes/ecom_products/`.
4. Extend catalog fields for e-comm: `brand`, `availability`, `price`, `category`, `composition`, `color` (from JSONB).
5. Keep `own_products` file path untouched (PLM practice).

### Step 2 — Sync + ask

1. CLI: `python -m knowledge_graph.cli sync --sources ecom_products` (or ingest hook).
2. Rebuild / append **e-comm** catalog index (separate from PLM index, or tagged `source_type`).
3. Fashion Brain questions: “how many Primark products?”, “Zara in stock”, “composition cotton” — routed to e-comm source.
4. Caveat text: “synced from e-comm scrape DB”, not “imported PLM CSV”.

### Step 3 — Colorways + Neo4j (after v1 works)

1. Join `*_colorways` or explode `master_products.colors` for color filters.
2. Neo4j: Product –[:HAS_BRAND]→ Brand, –[:USES_MATERIAL]→ from composition, –[:IN_CATEGORY]→.
3. Optional: link e-comm Primark name/reference ↔ local PLM Primark styles in `shared` (fuzzy → review queue).

### Step 4 — Brand dump tables + change events (later)

- Pull Walmart `rating` / `gender` only when users ask.
- `product_change_events` for “price dropped” style answers.
- Still no chat-time free SQL.

### Step 5 — Live PLMEX (when you get real PLM DB)

- Separate DSN; replace CSV for `own_products` only then.
- Cross-link PLM ↔ e-comm via reference / style codes.

---

## Recommended first milestone (1 slice)

**Done when:**

- KG can `SELECT` from live `master_products`.
- Envelopes exist for at least one brand (e.g. Primark or Zara).
- Fashion Brain answers: *“How many Primark products are in the e-comm catalog?”* with a count matching Postgres (± sync lag).
- Local PLM CSV still works for PLM-style questions.

Skip brand-specific tables and colorways until that works.

---

## Security & ops

- KG uses **read-only** roles; separate DSN per system.
- Never put production DSNs in git; use secrets manager / env.
- Clearance on envelopes (`internal` / `public`) — e-comm scrape data is usually `internal`.
- Prefer sync from replica; do not load scrape writers.
- Fashion Brain (:8000) never opens the e-comm DSN — only KG (:8001).

---

## What not to do

- Do not point Fashion Brain at Postgres for every chat turn.
- Do not ingest all five `*_products` dump tables first — start with **`master_products` only**.
- Do not merge e-comm rows into `own_products` / PLM envelopes.
- Do not treat Neo4j as system of record for price or stock.
- Do not enable free-form SQL from the LLM.

---

## Success criteria

1. `master_products` sync → e-comm catalog counts match SQL.  
2. Filters work: brand, category, composition, availability.  
3. PLM local CSV still answers PLM-style questions separately.  
4. Later: colorways + Neo4j materials + optional PLM↔e-comm links.  
5. Live SQL (if any) allowlisted and caveated.
