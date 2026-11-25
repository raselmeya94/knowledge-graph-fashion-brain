# Knowledge Graph — Relationships & User Columns

Per-KG map of **user-facing columns** (chat / filters / `kg_stats`), the **Neo4j relationships** they connect through, and a short description of each edge.

| Graph id | Ontology (live) | Role |
| --- | --- | --- |
| `ecom_products` | `data/schemas/ontologies/ecom.json` | Scraped retail catalog (Zara, Primark, Matalan, C&A, …) |
| `plmex_products` | `data/schemas/ontologies/plmex.json` | PLM / sample / buyer catalog |

**Model:** almost every user filter is `(Product)-[REL]->(FacetEntity)`. Fashion Brain retrieve maps filter keys → relationship types via `_REL_FIELDS` in `knowledge_graph/retrieval/ecom.py`. Counts also use product properties and `kg_stats` shards under `data/generated/kg_stats/<graph>/`.

```
User ask: "Primark sweaters"
        │
        ▼
  filters: brand=primark, category=sweater
        │
        ▼
  (p:Product)-[:HAS_BRAND]->(Brand {name})
  (p:Product)-[:BELONGS_TO_CATEGORY]->(GarmentCategory {name})
```

Sources of truth:

- Ontologies: `data/schemas/ontologies/{ecom,plmex}.json`
- Retrieve map: `_REL_FIELDS` in `retrieval/ecom.py`
- Indexed filter values: `data/generated/kg_stats/<graph>/<filter>.json`

---

## 1. User columns → relationships (chat / meta filters)

### 1.1 `ecom_products` — columns users ask about

| User column / filter | Typical ask | Neo4j relationship(s) | Target entity | Notes |
| --- | --- | --- | --- | --- |
| `brand` | How many Primark products? | `HAS_BRAND` | `Brand` | Also mirrored on `Product.brand` |
| `category` | How many jackets? | `BELONGS_TO_CATEGORY` | `GarmentCategory` | Soft-match values (sweater ↔ Sweaters) |
| `family` | Knitwear family | `HAS_FAMILY` | `Family` / `GarmentCategory` | |
| `subfamily` | Jumpers | `HAS_SUBFAMILY`, `HAS_SUB_CATEGORY` | `Subfamily` / `GarmentCategory` | |
| `section` | Women’s section | `HAS_SECTION` | `Section` | Often implies target group |
| `construction` | Construction mix | `HAS_CONSTRUCTION` | `Construction` | |
| `fit` | Slim fit | `HAS_FIT` | `Fit` | |
| `availability` | In stock | `HAS_AVAILABILITY` | `Availability` | Snapshots via `HAS_AVAILABILITY_SNAPSHOT` |
| `currency` | Price in EUR | `PRICED_IN` | `Currency` | Price on edge / snapshots |
| `country_of_origin` | Made in … | `MADE_IN` | `Country` | |
| `catalog_id` | Catalog slice | `IN_CATALOG` | `Catalog` | Scrape catalog id |
| `domain` | Site domain | `SCRAPED_FROM` | `Domain` | |
| `enriched_from` | Source URL | *(property / listing)* | — | Often URL text, not a core facet edge |
| `material` *(ask)* | Cotton products | `USES_MATERIAL`, `PRIMARY_MATERIAL` | `Material` | Composition % on edge |
| `color` *(ask)* | Red items | `HAS_COLOR`, `PRIMARY_COLOR` | `Color` | |
| `price` / trends | Price moving? | `HAS_PRICE_CHANGE`, `HAS_PRICE_SNAPSHOT`, `PRICED_IN` | `PriceChangeEvent` / `PriceSnapshot` / `Currency` | Temporal path |

`kg_stats` currently indexes:  
`availability`, `brand`, `catalog_id`, `category`, `construction`, `country_of_origin`, `currency`, `domain`, `enriched_from`, `family`, `fit`, `section`, `subfamily`.

### 1.2 `plmex_products` — columns users ask about

| User column / filter | Typical ask | Neo4j relationship(s) | Target entity | Notes |
| --- | --- | --- | --- | --- |
| `brand` | Brand count | `HAS_BRAND` | `Brand` | Many PLM brands |
| `buyer` | Buyer Matalan | `HAS_BUYER` | `Buyer` | PLM party (not ecom) |
| `category` | Sweater count | `BELONGS_TO_CATEGORY` | `GarmentCategory` | Strong PLM category (e.g. `sweater`) |
| `garment_type` | Garment type | `HAS_GARMENT_TYPE` | `GarmentType` | |
| `gender` | Women / men | `HAS_GENDER` | `Gender` | |
| `color` | White products | `HAS_COLOR`, `PRIMARY_COLOR` | `Color` | |
| `material` | Cotton | `USES_MATERIAL`, `PRIMARY_MATERIAL` | `Material` | |
| `department` | Department | `BELONGS_TO_DEPARTMENT` | `Department` | |
| `shopping_department` | Shopping dept | `BELONGS_TO_SHOPPING_DEPARTMENT` | `ShoppingDepartment` | |
| `plm_main_category` | PLM main | `HAS_PLM_MAIN_CATEGORY` | `PlmMainCategory` | Hierarchy |
| `plm_sub_category` | PLM sub | `HAS_PLM_SUB_CATEGORY` | `PlmSubCategory` | |
| `plm_sub_sub_category` | PLM sub-sub | `HAS_PLM_SUB_SUB_CATEGORY` | `PlmSubSubCategory` | |
| `design_for` / `design_purpose` | Designed for | `DESIGNED_FOR` | `DesignPurpose` | |
| `design_feature` | Design feature | `HAS_DESIGN_FEATURE` | `DesignFeature` | |
| `style_feature` | Style feature | `HAS_STYLE_FEATURE` | `StyleFeature` | |
| `gauge` | Gauge | `HAS_GAUGE` | `Gauge` | |
| `knit_machine_type` / `machine_type` | Knit machine | `USES_KNIT_MACHINE` | `MachineType` | |
| `sample_type` / `sample_collection_type` | Sample type | `HAS_SAMPLE_TYPE` | `SampleType` | |
| `season_year` | Season year | `IN_SEASON_YEAR` | `SeasonYear` | |
| `created_by` / `designer` / `merchandiser` / `purchased_by` | People | `CREATED_BY`, `DESIGNED_BY`, `MANAGED_BY`, `PURCHASED_BY` | `Person` / `Buyer` | |
| `location` / `current_location` | Where stored | `STORED_AT` | `Location` | |
| `inventory_record` | Inventory | `HAS_INVENTORY_RECORD` | `InventoryRecord` | |
| `shipment` | Shipment | `HAS_SHIPMENT` | `Shipment` | |
| `ai_event_tags` / `event_tag` | Event tags | `HAS_EVENT_TAG` | `EventTag` | AI / tagging |
| `ai_construction_tags` / `construction_tag` | Construction tags | `HAS_CONSTRUCTION_TAG` | `ConstructionTag` | |
| `ai_material_tags` | AI material tags | `HAS_AI_MATERIAL_TAG` | `AiMaterialTag` | |
| `ai_finish_tags` / `finish_tag` | Finish tags | `HAS_FINISH_TAG` | `FinishTag` | |
| `ai_special_tags` / `special_tag` | Special tags | `HAS_SPECIAL_TAG` | `SpecialTag` | |
| `gsm`, `weight`, `cuttable_width`, `unit`, `code`, … | Spec numbers | *(product properties)* | — | Property filters, not always edges |

`kg_stats` indexes the full PLM filter list under `data/generated/kg_stats/plmex_products/`.

### 1.3 Retrieve map (`_REL_FIELDS`) — filter key → edge type

Used by Cypher filter builders so chat filters hit the same edges as ontology sync:

| Filter key | Relationship type(s) |
| --- | --- |
| `category` | `BELONGS_TO_CATEGORY` |
| `family` | `HAS_FAMILY` |
| `subfamily` | `HAS_SUBFAMILY`, `HAS_SUB_CATEGORY` |
| `color` | `HAS_COLOR`, `PRIMARY_COLOR` |
| `material` | `USES_MATERIAL`, `PRIMARY_MATERIAL` |
| `availability` | `HAS_AVAILABILITY` |
| `brand` | `HAS_BRAND` |
| `buyer` | `HAS_BUYER` |
| `garment_type` | `HAS_GARMENT_TYPE` |
| `gender` | `HAS_GENDER` |
| `gauge` | `HAS_GAUGE` |
| `sample_collection_type` | `HAS_SAMPLE_TYPE` |
| `plm_main_category` | `HAS_PLM_MAIN_CATEGORY` |
| `plm_sub_category` | `HAS_PLM_SUB_CATEGORY` |
| `plm_sub_sub_category` | `HAS_PLM_SUB_SUB_CATEGORY` |
| `design_for` | `DESIGNED_FOR` |
| `knit_machine_type` | `USES_KNIT_MACHINE` |
| `design_feature` | `HAS_DESIGN_FEATURE` |
| `style_feature` | `HAS_STYLE_FEATURE` |
| `department` | `BELONGS_TO_DEPARTMENT` |
| `fit` | `HAS_FIT` |
| `construction` | `HAS_CONSTRUCTION` |
| `section` | `HAS_SECTION` |
| `ai_event_tags` | `HAS_EVENT_TAG` |
| `ai_construction_tags` | `HAS_CONSTRUCTION_TAG` |
| `ai_material_tags` | `HAS_AI_MATERIAL_TAG` |
| `ai_finish_tags` | `HAS_FINISH_TAG` |
| `ai_special_tags` | `HAS_SPECIAL_TAG` |
| `season_year` | `IN_SEASON_YEAR` |

---

## 2. `ecom_products` — every relationship

Format: **REL** — `From → To` — description.

### Identity & scrape provenance

| Relationship | Connection | Description |
| --- | --- | --- |
| `HAS_BRAND` | `Product → Brand` | Retail brand of the listing (exactly one). |
| `SOLD_BY` | `Brand → Domain` | Brand’s selling domain. |
| `SCRAPED_FROM` | `Product → Domain` | Site domain the product was scraped from. |
| `IN_CATALOG` | `Product → Catalog` | Scrape catalog / category-page run. |
| `SCRAPED_IN_JOB` | `Product\|PriceSnapshot\|AvailabilitySnapshot → ScrapeJob` | Which scrape job produced the observation. |
| `HAS_LISTING` | `Product → ProductListing` | Catalog / SEO product ids for the style. |
| `LISTING_IN_CATALOG` | `ProductListing → Catalog` | Listing membership in a catalog. |
| `HAS_REFERENCE` | `Product → Reference` | Style / display reference codes. |

### Assortment hierarchy

| Relationship | Connection | Description |
| --- | --- | --- |
| `HAS_SECTION` | `Product → Section` | Merch section (e.g. women, men). |
| `FOR_TARGET_GROUP` | `Product\|Section → TargetGroup` | Intended shopper group (women/men/kids/…). |
| `BELONGS_TO_CATEGORY` | `Product → GarmentCategory` | Primary garment category. |
| `HAS_FAMILY` | `Product → Family\|GarmentCategory` | Family within the assortment tree. |
| `HAS_SUBFAMILY` | `Product → Subfamily` | Finer family / subfamily label. |
| `HAS_SUB_CATEGORY` | `Product → GarmentCategory\|Subfamily` | Sub-category link (legacy / alternate). |
| `PARENT_CATEGORY` | `GarmentCategory → GarmentCategory`, `Subfamily → Family`, `Family → GarmentCategory` | Parent/child category hierarchy. |
| `CATEGORY_IN_SECTION` | `GarmentCategory → Section` | Category co-occurrence in a section. |

### Materials, make & care

| Relationship | Connection | Description |
| --- | --- | --- |
| `USES_MATERIAL` | `Product → Material` | Composition fibre (often with `%` on edge). |
| `PRIMARY_MATERIAL` | `Product → Material` | Highest-% fibre. |
| `IN_MATERIAL_GROUP` | `Material → MaterialGroup` | natural / synthetic / semi-synthetic. |
| `HAS_CERTIFICATION` | `Product\|Material → Certification` | Organic, recycled, GOTS, etc. |
| `HAS_CONSTRUCTION` | `Product → Construction` | Construction / knit-woven detail. |
| `HAS_FIT` | `Product → Fit` | Fit label (slim, regular, …). |
| `HAS_CARE` | `Product → CareInstruction` | Care copy / action. |
| `MADE_IN` | `Product → Country` | Country of origin. |

### Style attributes (often LLM-extracted)

| Relationship | Connection | Description |
| --- | --- | --- |
| `HAS_SILHOUETTE` | `Product → Silhouette` | Silhouette shape. |
| `HAS_NECKLINE` | `Product → Neckline` | Neckline type. |
| `HAS_SLEEVE_LENGTH` | `Product → SleeveLength` | Sleeve length. |
| `HAS_LENGTH` | `Product → GarmentLength` | Garment length (mini/midi/…). |
| `HAS_CLOSURE` | `Product → Closure` | Closure type. |
| `HAS_PATTERN` | `Product → Pattern` | Print / pattern. |
| `SUITED_FOR_OCCASION` | `Product → Occasion` | Occasion tag. |
| `SUITED_FOR_SEASON` | `Product → Season` | Season suitability. |
| `HAS_STYLE_TAG` | `Product → StyleTag` | Style / trend tags. |

### Color & size / stock

| Relationship | Connection | Description |
| --- | --- | --- |
| `HAS_COLOR` | `Product → Color` | Colorways. |
| `PRIMARY_COLOR` | `Product → Color` | Primary colorway. |
| `IN_COLOR_FAMILY` | `Color → ColorFamily` | Neutral / warm / cool grouping. |
| `HAS_SIZE_AVAILABILITY` | `Product → SizeAvailability` | Per-size stock observation (when scraper provides sizes). |
| `FOR_SIZE` | `SizeAvailability → Size` | Size label. |
| `OF_COLOR` | `SizeAvailability → Color` | Size row tied to a color. |
| `HAS_AVAILABILITY` | `Product → Availability` | Current aggregate availability status. |
| `HAS_AVAILABILITY_SNAPSHOT` | `Product → AvailabilitySnapshot` | Historical stock snapshot per scrape. |
| `SNAPSHOT_STATUS` | `AvailabilitySnapshot → Availability` | Status node for a snapshot. |

### Price & FX

| Relationship | Connection | Description |
| --- | --- | --- |
| `PRICED_IN` | `Product → Currency` | Current price / currency (amounts often on edge). |
| `IN_PRICE_BAND` | `Product → PriceBand` | Budget / mid / premium band. |
| `PRICE_BAND_IN_CURRENCY` | `PriceBand → Currency` | Band currency. |
| `HAS_PRICE_SNAPSHOT` | `Product → PriceSnapshot` | Historical observed price. |
| `SNAPSHOT_IN_CURRENCY` | `PriceSnapshot → Currency` | Snapshot currency. |
| `HAS_PRICE_CHANGE` | `Product → PriceChangeEvent` | Old→new price event (trend / drops / rises). |
| `FX_FROM` / `FX_TO` | `FxRate → Currency` | FX conversion endpoints. |

### Media, versions & similarity

| Relationship | Connection | Description |
| --- | --- | --- |
| `HAS_IMAGE` | `Product → Image` | Product images (position on edge). |
| `HAS_MAIN_IMAGE` | `Product → Image` | Hero image. |
| `HAS_VERSION` | `Product → ProductVersion` | Content/colorway version over time. |
| `NEXT_VERSION` | `ProductVersion → ProductVersion` | Version chain. |
| `VARIANT_OF` | `Product → Product` | Same reference, different color/listing. |
| `SIMILAR_TO` | `Product → Product` | Attribute similarity. |
| `SAME_STYLE_AS` | `Product → Product` | Embedding / style neighbours. |
| `CROSS_BRAND_MATCH` | `Product → Product` | Cross-brand attribute match. |
| `SAME_REFERENCE_AS` | `Product → Product` | Shared reference codes. |
| `SUBSTITUTE_FOR` | `Product → Product` | Cheaper similar alternative. |
| `PRICED_HIGHER_THAN` | `Product → Product` | Same family, higher price (FX-aware). |
| `COMPLEMENTS` | `Product → Product` | Outfit pairing (top↔bottom, …). |

### Useful ecom edge properties

| On relationship | Properties (examples) |
| --- | --- |
| `USES_MATERIAL` | `percentage`, `rank`, `is_primary`, `part` |
| `HAS_COLOR` | `position`, `is_primary` |
| `PRICED_IN` | `price`, `original_price`, `discount_pct`, `on_sale` |
| `HAS_PRICE_CHANGE` | `old_price`, `new_price`, `delta`, `delta_pct`, `direction` |
| `HAS_AVAILABILITY` | `in_stock_count`, `out_stock_count`, `stock_ratio` |
| Common provenance | `source`, `confidence`, `extracted_at`, `extractor_version` |

---

## 3. `plmex_products` — every relationship

### Party & materials

| Relationship | Connection | Description |
| --- | --- | --- |
| `HAS_BRAND` | `Product → Brand` | Style brand. |
| `HAS_BUYER` | `Product → Buyer` | Buying customer / account (PLM). |
| `USES_MATERIAL` | `Product → Material` | Bill of materials / fibres. |
| `PRIMARY_MATERIAL` | `Product → Material` | Dominant fibre. |

### Category & PLM hierarchy

| Relationship | Connection | Description |
| --- | --- | --- |
| `BELONGS_TO_CATEGORY` | `Product → GarmentCategory` | Main garment category (e.g. sweater). |
| `HAS_GARMENT_TYPE` | `Product → GarmentType` | Garment type facet. |
| `HAS_PLM_MAIN_CATEGORY` | `Product → PlmMainCategory` | PLM main category. |
| `HAS_PLM_SUB_CATEGORY` | `Product → PlmSubCategory` | PLM sub category. |
| `HAS_PLM_SUB_SUB_CATEGORY` | `Product → PlmSubSubCategory` | PLM sub-sub category. |
| `PARENT_CATEGORY` | `PlmSubCategory → PlmMainCategory`, `PlmSubSubCategory → PlmSubCategory`, `GarmentCategory → PlmMainCategory` | PLM tree links. |

### Product attributes

| Relationship | Connection | Description |
| --- | --- | --- |
| `HAS_GENDER` | `Product → Gender` | Women / men / girl / boy / …. |
| `HAS_COLOR` | `Product → Color` | Color. |
| `PRIMARY_COLOR` | `Product → Color` | Primary color. |
| `HAS_GAUGE` | `Product → Gauge` | Knit gauge. |
| `HAS_SAMPLE_TYPE` | `Product → SampleType` | Sample collection type. |
| `DESIGNED_FOR` | `Product → DesignPurpose` | Design purpose / end use. |
| `USES_KNIT_MACHINE` | `Product → MachineType` | Knit machine type. |
| `HAS_DESIGN_FEATURE` | `Product → DesignFeature` | Design features. |
| `HAS_STYLE_FEATURE` | `Product → StyleFeature` | Style features. |
| `BELONGS_TO_DEPARTMENT` | `Product → Department` | Ops department. |
| `BELONGS_TO_SHOPPING_DEPARTMENT` | `Product → ShoppingDepartment` | Shopping department. |
| `IN_SEASON_YEAR` | `Product → SeasonYear` | Season / year. |
| `HAS_UNIT` | `Product → Unit` | Unit of measure. |

### People & references

| Relationship | Connection | Description |
| --- | --- | --- |
| `CREATED_BY` | `Product → Person` | Creator. |
| `DESIGNED_BY` | `Product → Person` | Designer. |
| `MANAGED_BY` | `Product → Person` | Merchandiser / manager. |
| `PURCHASED_BY` | `Product → Person\|Buyer` | Purchaser. |
| `HAS_STYLE_REFERENCE` | `Product → StyleReference` | Internal style reference. |
| `HAS_CUSTOMER_STYLE_REFERENCE` | `Product → CustomerStyleReference` | Customer style reference. |

### Inventory & logistics

| Relationship | Connection | Description |
| --- | --- | --- |
| `STORED_AT` | `Product\|InventoryRecord → Location` | Storage / current location. |
| `HAS_INVENTORY_RECORD` | `Product → InventoryRecord` | Inventory rows. |
| `HAS_SHIPMENT` | `Product → Shipment` | Shipment events. |

### AI / tagging

| Relationship | Connection | Description |
| --- | --- | --- |
| `HAS_EVENT_TAG` | `Product → EventTag` | Event tags. |
| `HAS_CONSTRUCTION_TAG` | `Product → ConstructionTag` | Construction tags. |
| `HAS_AI_MATERIAL_TAG` | `Product → AiMaterialTag` | AI material tags. |
| `HAS_FINISH_TAG` | `Product → FinishTag` | Finish tags. |
| `HAS_SPECIAL_TAG` | `Product → SpecialTag` | Special tags. |
| `RELATED_TO_TAG` | `*Tag → *Tag` | Tag co-occurrence / relatedness. |

### Similarity

| Relationship | Connection | Description |
| --- | --- | --- |
| `SAME_STYLE_AS` | `Product → Product` | Style neighbours. |
| `VARIANT_OF` | `Product → Product` | Variants of a style. |
| `SIMILAR_TO` | `Product → Product` | General similarity. |

---

## 4. How Fashion Brain uses this

| Layer | What it uses |
| --- | --- |
| Chat filters | User words → filter keys (`brand`, `category`, …) → soft-matched to `kg_stats` values → Cypher via `_REL_FIELDS` |
| Basic counts | Prefer `kg_stats/<graph>/<filter>.json` (meta); multi-filter → Neo4j |
| Price / availability history | `HAS_PRICE_CHANGE`, `HAS_AVAILABILITY_SNAPSHOT` |
| Spec / analysis | Multiple facet edges + product properties (`gsm`, composition, …) |

**Pattern for grounded Cypher (conceptual):**

```cypher
MATCH (p:Entity:Product {graph: $g})
WHERE coalesce(p.lifecycle,'active') <> 'deleted'
  AND (p)-[:HAS_BRAND]->(:Entity {name: $brand})
  AND (p)-[:BELONGS_TO_CATEGORY]->(:Entity)
      // name match / soft match applied in app layer
RETURN count(p) AS c
```

---

## 5. Maintenance

| Change | Update |
| --- | --- |
| New user filter column | Ontology allowlist + sync mapping + `_REL_FIELDS` + rebuild `kg_stats` |
| New relationship | `data/schemas/ontologies/{ecom\|plmex}.json` → rebuild graph |
| New indexed values only | Re-run kg_stats generation for that graph |

Related docs: [`API.md`](./API.md) · [`README.md`](./README.md) · Fashion Brain flow [`../fashion_brain/README_TECHNICAL_FLOW.md`](../fashion_brain/README_TECHNICAL_FLOW.md)
