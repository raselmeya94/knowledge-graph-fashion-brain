"""Patch knowledge_graph Postman collection for ecom-first empty→live pipeline."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COLLECTION = ROOT / "knowledge_graph.postman_collection.json"
BODIES = ROOT / "bodies"


def raw_body(obj: dict | list | str) -> dict:
    if isinstance(obj, str):
        text = obj
    else:
        text = json.dumps(obj, indent=2)
    return {
        "mode": "raw",
        "raw": text,
        "options": {"raw": {"language": "json"}},
    }


def req(name: str, method: str, url: str, *, body: dict | list | str | None = None, description: str = "", event: list | None = None) -> dict:
    item: dict = {
        "name": name,
        "request": {
            "method": method,
            "header": [{"key": "Content-Type", "value": "application/json"}] if body is not None and method != "GET" else [],
            "url": url,
            "description": description,
        },
    }
    if body is not None:
        item["request"]["body"] = raw_body(body)
    if event:
        item["event"] = event
    return item


def save_job_id_script() -> list:
    return [
        {
            "listen": "test",
            "script": {
                "type": "text/javascript",
                "exec": [
                    "try {",
                    "  const j = pm.response.json();",
                    "  const id = j.job && j.job.id;",
                    "  if (id) pm.collectionVariables.set('jobId', id);",
                    "} catch (e) {}",
                ],
            },
        }
    ]


def main() -> None:
    col = json.loads(COLLECTION.read_text(encoding="utf-8"))
    ontology = json.loads((BODIES / "POST_ontologies_ecom.json").read_text(encoding="utf-8"))
    graph = json.loads((BODIES / "POST_graphs_ecom_products.json").read_text(encoding="utf-8"))
    source = json.loads((BODIES / "POST_api_sources_ecom_products.json").read_text(encoding="utf-8"))

    # Collection vars — ecom defaults (not buyer-scraping / own_products)
    var_map = {
        "ontologyId": "ecom",
        "graphId": "ecom_products",
        "sourceId": "ecom_products",
        "brand": "primark",
        "material": "cotton",
        "nodeId": "ecom:product:example",
        "productId": "",
    }
    existing = {v["key"]: v for v in col.get("variable", [])}
    for key, value in var_map.items():
        if key in existing:
            existing[key]["value"] = value
        else:
            col.setdefault("variable", []).append({"key": key, "value": value})

    mapping = json.loads((BODIES / "ecom_master_products_mapping.json").read_text(encoding="utf-8"))
    db_conn = json.loads((BODIES / "POST_api_db_connections.json").read_text(encoding="utf-8"))
    graph_build = json.loads((BODIES / "POST_api_graphs_build.json").read_text(encoding="utf-8"))
    query_body = json.loads((BODIES / "POST_api_graphs_query_ecom.json").read_text(encoding="utf-8"))
    retrieve_brand = json.loads((BODIES / "POST_brain_retrieve_ecom_brand.json").read_text(encoding="utf-8"))

    col["info"]["description"] = (
        "Knowledge graph API on port 8001.\n\n"
        "**UI wizard (folder 0) — all API, no file copy:**\n"
        "1. Health + inspect (ontologies / sources / graphs / checklist)\n"
        "2. POST ontology → GET ontology\n"
        "3. PUT mapping → POST db connection → GET db inspect → POST source → POST graph\n"
        "4. Checklist ready → POST build → poll job + build/status\n"
        "5. Metadata / query / brain\n\n"
        "Guide: `knowledge_graph/UI_CONTROL_PLAN.md` · `knowledge_graph/POSTMAN.md`.\n\n"
        "Auth: Bearer `{{apiToken}}` only if set. Always `X-Clearance: {{clearance}}` (internal)."
    )

    ecom_folder = {
        "name": "0 UI wizard (empty → live)",
        "description": (
            "Full UI-control path. Inspect first, then register ontology/mapping/DB/source/graph, "
            "then build and poll status. No manual file copies. Postgres password stays in .env."
        ),
        "item": [
            req("01 GET /health", "GET", "{{baseUrl}}/health", description="Expect status ok."),
            req("02 GET /ontologies (schemas present?)", "GET", "{{baseUrl}}/ontologies"),
            req("03 GET /api/sources (sources present?)", "GET", "{{baseUrl}}/api/sources"),
            req("04 GET /api/graphs (KGs present?)", "GET", "{{baseUrl}}/api/graphs"),
            req(
                "05 GET /api/setup/checklist",
                "GET",
                "{{baseUrl}}/api/setup/checklist?graph_id=ecom_products&source_id=ecom_products",
                description="ready_to_build + missing[]. Expect false until register steps done.",
            ),
            req("06 GET /api/bootstrap", "GET", "{{baseUrl}}/api/bootstrap", description="UI home: counts + lists + checklist."),
            req(
                "07 POST /ontologies — ecom",
                "POST",
                "{{baseUrl}}/ontologies",
                body=ontology,
                description="Push full ecom ontology from Postman body.",
                event=[
                    {
                        "listen": "test",
                        "script": {
                            "type": "text/javascript",
                            "exec": [
                                "pm.collectionVariables.set('ontologyId', 'ecom');",
                                "pm.test('ontology ok or already exists', function () {",
                                "  pm.expect([200, 400, 422]).to.include(pm.response.code);",
                                "});",
                            ],
                        },
                    }
                ],
            ),
            req("08 GET /ontologies/ecom", "GET", "{{baseUrl}}/ontologies/ecom"),
            req(
                "09 PUT /api/sources/ecom_products/mapping",
                "PUT",
                "{{baseUrl}}/api/sources/ecom_products/mapping",
                body=mapping,
                description="Upload column mapping via API (idempotent replace). No disk copy.",
            ),
            req(
                "10 POST /api/db/connections",
                "POST",
                "{{baseUrl}}/api/db/connections",
                body=db_conn,
                description="Named DB connection. Password via password_env / .env — never returned.",
            ),
            req(
                "11 POST /api/db/connections/ecom_postgres/test",
                "POST",
                "{{baseUrl}}/api/db/connections/ecom_postgres/test?table=master_products",
                body={},
                description="Test named connection + table metadata.",
            ),
            req(
                "12 GET /api/db/connection",
                "GET",
                "{{baseUrl}}/api/db/connection?table=master_products&schema=public",
                description="Inspect default .env DSN table/columns/brands.",
            ),
            req(
                "13 POST /api/sources — ecom_products",
                "POST",
                "{{baseUrl}}/api/sources",
                body=source,
                event=[
                    {
                        "listen": "test",
                        "script": {
                            "type": "text/javascript",
                            "exec": [
                                "pm.collectionVariables.set('sourceId', 'ecom_products');",
                                "pm.test('source ok or already exists', function () {",
                                "  pm.expect([200, 400, 422]).to.include(pm.response.code);",
                                "});",
                            ],
                        },
                    }
                ],
            ),
            req(
                "14 POST /graphs — ecom_products",
                "POST",
                "{{baseUrl}}/graphs",
                body=graph,
                event=[
                    {
                        "listen": "test",
                        "script": {
                            "type": "text/javascript",
                            "exec": [
                                "pm.collectionVariables.set('graphId', 'ecom_products');",
                                "pm.test('graph ok or already exists', function () {",
                                "  pm.expect([200, 400, 422]).to.include(pm.response.code);",
                                "});",
                            ],
                        },
                    }
                ],
            ),
            req(
                "15 GET /api/setup/checklist (must be ready)",
                "GET",
                "{{baseUrl}}/api/setup/checklist?graph_id=ecom_products&source_id=ecom_products",
                description="Expect ready_to_build=true before build.",
            ),
            req(
                "16 POST /api/graphs/ecom_products/build",
                "POST",
                "{{baseUrl}}/api/graphs/ecom_products/build",
                body=graph_build,
                description="Start build from source_id + schema. Poll job + build/status.",
                event=save_job_id_script(),
            ),
            req("17 GET /jobs/{jobId}", "GET", "{{baseUrl}}/jobs/{{jobId}}", description="Poll until done/failed."),
            req(
                "18 GET /api/graphs/ecom_products/build/status",
                "GET",
                "{{baseUrl}}/api/graphs/ecom_products/build/status?job_id={{jobId}}",
                description="products / node_count / edge_count / errors / progress_percent.",
            ),
            req("19 GET /api/graphs", "GET", "{{baseUrl}}/api/graphs", description="node_count > 0 after build."),
            req(
                "20 GET /api/metadata/values?attribute=brand",
                "GET",
                "{{baseUrl}}/api/metadata/values?attribute=brand&graph_id=ecom_products&limit=25",
            ),
            req(
                "21 POST /api/graphs/query",
                "POST",
                "{{baseUrl}}/api/graphs/query",
                body=query_body,
            ),
            req(
                "22 POST /brain/retrieve",
                "POST",
                "{{baseUrl}}/brain/retrieve",
                body=retrieve_brand,
            ),
        ],
    }
    # Drop previous wizard / ecom folder if re-run
    items = [
        f
        for f in col.get("item", [])
        if not str(f.get("name", "")).startswith("0 Ecom")
        and not str(f.get("name", "")).startswith("0 UI wizard")
    ]
    col["item"] = [ecom_folder, *items]

    # Fix Registry folder requests that still use buyer-scraping / wrong pairs
    for folder in col["item"]:
        if folder.get("name") != "Registry":
            continue
        for item in folder.get("item", []):
            name = item.get("name", "")
            if name == "POST ontologies clone news":
                item["name"] = "POST ontologies — ecom (full body)"
                item["request"]["body"] = raw_body(ontology)
                item["request"]["description"] = "Same body as folder 0 step 04. Prefer folder 0 for ordered runs."
            elif name == "POST ontology pair":
                item["request"]["body"] = raw_body(
                    {"type": "HAS_BRAND", "start_type": "Product", "end_type": "Brand"}
                )
                item["request"]["description"] = "Ecom allowlist pair (not news SOURCES_FROM)."
            elif name == "POST graph":
                item["name"] = "POST graph — ecom_products"
                item["request"]["body"] = raw_body(graph)

    # Fix Plan API register source + retrieve filters; prepend new endpoints
    for folder in col["item"]:
        if folder.get("name") != "Plan API aliases":
            continue
        folder["description"] = (
            "Plan API aliases for admin UI. Prefer folder 0 UI wizard for empty→live. "
            "Includes bootstrap, checklist, mapping, db connections, graph build/status."
        )
        plan_items = folder.get("item", [])
        drop_names = {
            "0 GET /api/db/connection",
            "0b POST /api/kg/build",
            "0c GET /api/bootstrap",
            "0d GET /api/setup/checklist",
            "0e PUT /api/sources/ecom_products/mapping",
            "0f POST /api/graphs/ecom_products/build",
            "0g GET /api/graphs/ecom_products/build/status",
        }
        plan_items = [item for item in plan_items if item.get("name") not in drop_names]
        plan_items = [
            req("0c GET /api/bootstrap", "GET", "{{baseUrl}}/api/bootstrap"),
            req(
                "0d GET /api/setup/checklist",
                "GET",
                "{{baseUrl}}/api/setup/checklist?graph_id=ecom_products&source_id=ecom_products",
            ),
            req(
                "0 GET /api/db/connection",
                "GET",
                "{{baseUrl}}/api/db/connection?table=master_products&schema=public",
            ),
            req(
                "0e PUT /api/sources/ecom_products/mapping",
                "PUT",
                "{{baseUrl}}/api/sources/ecom_products/mapping",
                body=mapping,
            ),
            req(
                "0f POST /api/graphs/ecom_products/build",
                "POST",
                "{{baseUrl}}/api/graphs/ecom_products/build",
                body=graph_build,
                event=save_job_id_script(),
            ),
            req(
                "0g GET /api/graphs/ecom_products/build/status",
                "GET",
                "{{baseUrl}}/api/graphs/ecom_products/build/status?job_id={{jobId}}",
            ),
            *plan_items,
        ]
        folder["item"] = plan_items
        for item in folder.get("item", []):
            name = item.get("name", "")
            if "POST /api/sources (register)" in name:
                item["name"] = "15 POST /api/sources — ecom_products"
                item["request"]["body"] = raw_body(source)
            elif name.startswith("14 POST /api/graphs/cross-query"):
                item["request"]["body"] = raw_body(
                    {
                        "intent": "product_count",
                        "question": "Cotton products in e-commerce",
                        "filters": {"material": "cotton"},
                        "graph_ids": ["ecom_products"],
                        "use_global_kg": True,
                    }
                )
    for folder in col["item"]:
        if folder.get("name") != "Brain retrieve (for Fashion Brain)":
            continue
        for item in folder.get("item", []):
            if "product count" in item.get("name", "").lower():
                item["request"]["body"] = raw_body(
                    {
                        "question": "Which e-com products use cotton?",
                        "intent": "product_count",
                        "filters": {"material": "cotton"},
                        "graph_id": "ecom_products",
                        "graph_ids": ["ecom_products"],
                        "group_by": None,
                        "unsupported": False,
                        "limit": None,
                        "suggest": False,
                        "analysis": False,
                        "season_name": None,
                        "upcoming": False,
                        "corrections": [],
                    }
                )
            elif "buyer analysis" in item.get("name", "").lower():
                item["name"] = "POST brain retrieve — brand primark"
                item["request"]["body"] = raw_body(
                    {
                        "question": "How many Primark products?",
                        "intent": "product_count",
                        "filters": {"brand": "primark"},
                        "graph_id": "ecom_products",
                        "graph_ids": ["ecom_products"],
                        "analysis": False,
                        "suggest": False,
                        "limit": 10,
                        "corrections": [],
                    }
                )

    # Graph queries: add ecom list_products at top of that folder
    for folder in col["item"]:
        if folder.get("name") != "Graph queries":
            continue
        ecom_query = req(
            "POST ecom_products — list_products",
            "POST",
            "{{baseUrl}}/graphs/ecom_products/query",
            body={"template": "list_products", "params": {}, "limit": 20, "offset": 0},
        )
        folder["item"] = [ecom_query, *folder.get("item", [])]
        break

    COLLECTION.write_text(json.dumps(col, indent=2) + "\n", encoding="utf-8")
    print("patched", COLLECTION)
    print("folder 0 steps:", len(ecom_folder["item"]))


if __name__ == "__main__":
    main()
