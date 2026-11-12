# Postman request bodies (real project data)

Paste these into Postman **Body → raw → JSON** for the matching endpoint.

Full step order: [`../../knowledge_graph/POSTMAN.md`](../../knowledge_graph/POSTMAN.md).

| File | Endpoint |
| --- | --- |
| `POST_ontologies_*.json` | `POST /ontologies` |
| `POST_graphs_*.json` | `POST /graphs` |
| `POST_api_sources_*.json` | `POST /api/sources` |
| `POST_api_kg_build.json` | `POST /api/kg/build` (optional demo shortcut) |
| `POST_api_graphs_build.json` | `POST /api/graphs/{id}/build` |
| `POST_api_db_connections.json` | `POST /api/db/connections` |
| `ecom_master_products_mapping.json` | also `PUT /api/sources/ecom_products/mapping` |
| `POST_api_graphs_query_ecom.json` | `POST /api/graphs/query` (`list_products`) |
| `POST_api_graphs_get_product.json` | `POST /api/graphs/query` (`get_product`, uses `{{productId}}`) |
| `POST_api_graphs_cross_query.json` | `POST /api/graphs/cross-query` |
| `POST_brain_retrieve_ecom_brand.json` | `POST /brain/retrieve` |
| `POST_kg_pipeline_news_text.json` | `POST /kg/pipeline` |
| `ecom_master_products_mapping.json` | copy to `data/schemas/` before e-com sync (also auto-restored by `/api/kg/build`) |
