# Schemas (committed)

| Path | Purpose |
| --- | --- |
| `ecom_master_products_mapping.json` | Column mapping for Postgres `master_products` → KG fields |
| `ontologies/<id>.json` | Ontologies created via `POST /ontologies` (e.g. `ecom.json`) |

On a fresh clone, either keep the seed `ontologies/ecom.json` or recreate with Postman (`postman/bodies/POST_ontologies_ecom.json`).
