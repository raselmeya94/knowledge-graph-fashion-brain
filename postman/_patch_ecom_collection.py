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

    col["info"]["description"] = (
        "Knowledge graph API on port 8001.\n\n"
        "**E-com empty→live order (folder 0):**\n"
        "1. GET health / empty checks\n"
        "2. POST /ontologies (ecom — full schema + Zara/Primark/Matalan/C&A/Walmart)\n"
        "3. POST /graphs (ecom_products, kind=product)\n"
        "4. POST /api/sources (ecom_products, authorized_sql)\n"
        "5. POST sync → poll job → metadata/values → query/retrieve\n\n"
        "Bodies also in `postman/bodies/`. Guide: `knowledge_graph/POSTMAN.md`.\n\n"
        "Auth: Bearer `{{apiToken}}` only if set. Always `X-Clearance: {{clearance}}` (internal)."
    )

    ecom_folder = {
        "name": "0 Ecom pipeline (empty → live)",
        "description": (
            "Run top→bottom after a wipe. Uses the real ecom ontology "
            "(Product/Brand/Material/… + HAS_BRAND/USES_MATERIAL/…). "
            "Copy postman/bodies/ecom_master_products_mapping.json → data/schemas/ before sync. "
            "Postgres DSN must be set in .env."
        ),
        "item": [
            req("01 GET health", "GET", "{{baseUrl}}/health", description="Expect status ok."),
            req("02 GET /ontologies (should be empty or only prior POSTs)", "GET", "{{baseUrl}}/ontologies"),
            req("03 GET /api/sources (should be empty)", "GET", "{{baseUrl}}/api/sources"),
            req(
                "04 POST /ontologies — ecom (full body)",
                "POST",
                "{{baseUrl}}/ontologies",
                body=ontology,
                description="Creates ontology id=ecom with entity_types + relationship_allowlist + brand canonicals.",
                event=[
                    {
                        "listen": "test",
                        "script": {
                            "type": "text/javascript",
                            "exec": [
                                "pm.collectionVariables.set('ontologyId', 'ecom');",
                                "pm.test('ontology created or already exists', function () {",
                                "  pm.expect([200, 400]).to.include(pm.response.code);",
                                "});",
                            ],
                        },
                    }
                ],
            ),
            req(
                "05 GET /ontologies/ecom",
                "GET",
                "{{baseUrl}}/ontologies/ecom",
                description="Must return definition with Brand + HAS_BRAND, not 500.",
            ),
            req(
                "06 POST /graphs — ecom_products",
                "POST",
                "{{baseUrl}}/graphs",
                body=graph,
                description="kind=product only allowed for ecom_products / own_products.",
                event=[
                    {
                        "listen": "test",
                        "script": {
                            "type": "text/javascript",
                            "exec": ["pm.collectionVariables.set('graphId', 'ecom_products');"],
                        },
                    }
                ],
            ),
            req("07 GET /api/graphs/registry", "GET", "{{baseUrl}}/api/graphs/registry"),
            req(
                "08 POST /api/sources — ecom_products",
                "POST",
                "{{baseUrl}}/api/sources",
                body=source,
                event=[
                    {
                        "listen": "test",
                        "script": {
                            "type": "text/javascript",
                            "exec": ["pm.collectionVariables.set('sourceId', 'ecom_products');"],
                        },
                    }
                ],
            ),
            req("09 GET /api/sources/ecom_products", "GET", "{{baseUrl}}/api/sources/ecom_products"),
            req(
                "10 POST /api/sources/ecom_products/sync",
                "POST",
                "{{baseUrl}}/api/sources/ecom_products/sync",
                body={},
                description="Needs FASHIONGPT_ECOM_DSN / POSTGRES_* and mapping file under data/schemas/.",
                event=save_job_id_script(),
            ),
            req("11 GET /jobs/{jobId}", "GET", "{{baseUrl}}/jobs/{{jobId}}", description="Poll until finished."),
            req("12 GET /api/graphs", "GET", "{{baseUrl}}/api/graphs", description="ecom_products node_count > 0 after sync."),
            req(
                "13 GET /api/metadata/values?attribute=brand",
                "GET",
                "{{baseUrl}}/api/metadata/values?attribute=brand&graph_id=ecom_products&limit=25",
                description="Expect primark, zara, matalan, c-and-a, walmart samples when data synced.",
            ),
            req(
                "14 GET /api/metadata/values?attribute=material",
                "GET",
                "{{baseUrl}}/api/metadata/values?attribute=material&graph_id=ecom_products&limit=25",
            ),
            req(
                "15 POST /api/graphs/query — list_products",
                "POST",
                "{{baseUrl}}/api/graphs/query",
                body={
                    "graph_id": "ecom_products",
                    "template": "list_products",
                    "params": {},
                    "limit": 10,
                    "offset": 0,
                },
            ),
            req(
                "16 POST /brain/retrieve — brand primark",
                "POST",
                "{{baseUrl}}/brain/retrieve",
                body={
                    "question": "How many Primark products are in the e-commerce catalog?",
                    "intent": "product_count",
                    "filters": {"brand": "{{brand}}"},
                    "group_by": None,
                    "unsupported": False,
                    "limit": None,
                    "suggest": False,
                    "analysis": False,
                    "season_name": None,
                    "upcoming": False,
                    "corrections": [],
                    "graph_id": "ecom_products",
                    "graph_ids": ["ecom_products"],
                    "use_global_kg": False,
                },
                description="Uses ontology Brand canonicals (primark/zara/matalan/c-and-a/walmart).",
            ),
            req(
                "17 POST /brain/retrieve — material cotton",
                "POST",
                "{{baseUrl}}/brain/retrieve",
                body={
                    "question": "Which e-com products use cotton?",
                    "intent": "product_count",
                    "filters": {"material": "{{material}}"},
                    "graph_id": "ecom_products",
                    "graph_ids": ["ecom_products"],
                    "analysis": False,
                    "suggest": False,
                    "corrections": [],
                },
            ),
            req(
                "18 POST ontology pair — HAS_BRAND (idempotent check)",
                "POST",
                "{{baseUrl}}/ontologies/ecom/pairs",
                body={"type": "HAS_BRAND", "start_type": "Product", "end_type": "Brand"},
                description="Relationship already in allowlist; useful to verify pairs API.",
            ),
        ],
    }

    # Drop previous "0 Ecom…" folder if re-run
    items = [f for f in col.get("item", []) if not str(f.get("name", "")).startswith("0 Ecom")]
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

    # Fix Plan API register source + retrieve filters
    for folder in col["item"]:
        if folder.get("name") != "Plan API aliases":
            continue
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
