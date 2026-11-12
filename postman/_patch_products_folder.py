"""Add Products (list → detail) folder to knowledge_graph Postman collection."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COLLECTION = ROOT / "knowledge_graph.postman_collection.json"
BODIES = ROOT / "bodies"


def raw_body(obj: dict | list | str) -> dict:
    text = obj if isinstance(obj, str) else json.dumps(obj, indent=2)
    return {"mode": "raw", "raw": text, "options": {"raw": {"language": "json"}}}


def req(
    name: str,
    method: str,
    url: str,
    *,
    body: dict | list | str | None = None,
    description: str = "",
    event: list | None = None,
) -> dict:
    item: dict = {
        "name": name,
        "request": {
            "method": method,
            "header": [{"key": "Content-Type", "value": "application/json"}]
            if body is not None and method != "GET"
            else [],
            "url": url,
            "description": description,
        },
    }
    if body is not None:
        item["request"]["body"] = raw_body(body)
    if event:
        item["event"] = event
    return item


def save_product_id_from_products_get() -> list:
    return [
        {
            "listen": "test",
            "script": {
                "type": "text/javascript",
                "exec": [
                    "pm.test('products returned', function () {",
                    "  pm.response.to.have.status(200);",
                    "  const j = pm.response.json();",
                    "  pm.expect(j.items || []).to.be.an('array');",
                    "  if ((j.items || []).length) {",
                    "    const id = j.items[0].product_id || (j.items[0].product && j.items[0].product.product_id);",
                    "    if (id) pm.collectionVariables.set('productId', id);",
                    "  }",
                    "});",
                ],
            },
        }
    ]


def save_product_id_from_list_products() -> list:
    return [
        {
            "listen": "test",
            "script": {
                "type": "text/javascript",
                "exec": [
                    "pm.test('list_products returned', function () {",
                    "  pm.response.to.have.status(200);",
                    "  const j = pm.response.json();",
                    "  const items = j.items || [];",
                    "  pm.expect(items).to.be.an('array');",
                    "  if (items.length && items[0].id) {",
                    "    pm.collectionVariables.set('productId', items[0].id);",
                    "  }",
                    "});",
                ],
            },
        }
    ]


def main() -> None:
    col = json.loads(COLLECTION.read_text(encoding="utf-8"))
    get_product = json.loads((BODIES / "POST_api_graphs_get_product.json").read_text(encoding="utf-8"))
    # Ensure {{productId}} placeholder
    get_product.setdefault("params", {})["node_id"] = "{{productId}}"

    # Collection variable
    existing = {v["key"]: v for v in col.get("variable", [])}
    if "productId" not in existing:
        col.setdefault("variable", []).append(
            {"key": "productId", "value": "ecom:c-and-a:c-and-a:2053536"}
        )
    else:
        if not existing["productId"].get("value"):
            existing["productId"]["value"] = "ecom:c-and-a:c-and-a:2053536"

    products_folder = {
        "name": "Products (list → detail)",
        "description": (
            "1) GET /products?material=cotton — Neo4j product cards (saves productId).\n"
            "2) POST /api/graphs/query get_product — full properties + relationships.\n"
            "Also: list_products template, then same get_product detail."
        ),
        "item": [
            req(
                "01 GET /products?material=cotton",
                "GET",
                "{{baseUrl}}/products?material=cotton&limit=5&offset=0",
                description=(
                    "Product cards from Neo4j when file catalog is empty. "
                    "Saves collection var productId from items[0].product_id."
                ),
                event=save_product_id_from_products_get(),
            ),
            req(
                "02 GET /products?brand=primark",
                "GET",
                "{{baseUrl}}/products?brand={{brand}}&limit=5&offset=0",
                description="Filter by brand. Also updates productId when items exist.",
                event=save_product_id_from_products_get(),
            ),
            req(
                "03 POST /api/graphs/query — get_product (full detail)",
                "POST",
                "{{baseUrl}}/api/graphs/query",
                body=get_product,
                description=(
                    "Full product detail: properties + relationships. "
                    "Uses {{productId}} from step 01/02 (or set manually)."
                ),
                event=[
                    {
                        "listen": "test",
                        "script": {
                            "type": "text/javascript",
                            "exec": [
                                "pm.test('get_product has properties', function () {",
                                "  pm.response.to.have.status(200);",
                                "  const j = pm.response.json();",
                                "  pm.expect(j.properties || j.id).to.ok;",
                                "});",
                            ],
                        },
                    }
                ],
            ),
            req(
                "04 POST /api/graphs/query — list_products",
                "POST",
                "{{baseUrl}}/api/graphs/query",
                body={
                    "graph_id": "ecom_products",
                    "template": "list_products",
                    "params": {},
                    "limit": 10,
                    "offset": 0,
                },
                description="Template list (summary cards). Saves productId from items[0].id.",
                event=save_product_id_from_list_products(),
            ),
            req(
                "05 POST /api/graphs/query — get_product after list",
                "POST",
                "{{baseUrl}}/api/graphs/query",
                body=get_product,
                description="Same detail call using productId from step 04.",
            ),
            req(
                "06 POST /graphs/ecom_products/query — get_product (legacy path)",
                "POST",
                "{{baseUrl}}/graphs/ecom_products/query",
                body={
                    "template": "get_product",
                    "params": {"node_id": "{{productId}}"},
                    "limit": 1,
                    "offset": 0,
                },
                description="Legacy graph-scoped query path (same template).",
            ),
        ],
    }

    # Replace existing products folder if present
    items = [
        f
        for f in col.get("item", [])
        if not str(f.get("name", "")).startswith("Products (list")
    ]
    # Insert after UI wizard (folder 0) if present
    insert_at = 0
    for i, f in enumerate(items):
        if str(f.get("name", "")).startswith("0 UI wizard") or str(f.get("name", "")).startswith("0 Ecom"):
            insert_at = i + 1
            break
    items.insert(insert_at, products_folder)
    col["item"] = items

    # Deduplicate Graph queries get_product entry
    for folder in col["item"]:
        if folder.get("name") != "Graph queries":
            continue
        existing_names = {item.get("name") for item in folder.get("item", [])}
        extras = []
        name = "POST ecom_products — get_product"
        if name not in existing_names:
            extras.append(
                req(
                    name,
                    "POST",
                    "{{baseUrl}}/graphs/ecom_products/query",
                    body={
                        "template": "get_product",
                        "params": {"node_id": "{{productId}}"},
                        "limit": 1,
                        "offset": 0,
                    },
                    description="Full product detail. Set productId from Products folder step 01.",
                )
            )
        if extras:
            folder["item"] = [*extras, *folder.get("item", [])]
        break

    # Plan API: add products shortcuts near top if missing
    for folder in col["item"]:
        if folder.get("name") != "Plan API aliases":
            continue
        names = {item.get("name") for item in folder.get("item", [])}
        add = []
        if "P1 GET /products?material=cotton" not in names:
            add.append(
                req(
                    "P1 GET /products?material=cotton",
                    "GET",
                    "{{baseUrl}}/products?material=cotton&limit=5&offset=0",
                    event=save_product_id_from_products_get(),
                )
            )
        if "P2 POST /api/graphs/query — get_product" not in names:
            add.append(
                req(
                    "P2 POST /api/graphs/query — get_product",
                    "POST",
                    "{{baseUrl}}/api/graphs/query",
                    body=get_product,
                )
            )
        if add:
            folder["item"] = [*add, *folder.get("item", [])]
        break

    COLLECTION.write_text(json.dumps(col, indent=2) + "\n", encoding="utf-8")
    (BODIES / "POST_api_graphs_get_product.json").write_text(
        json.dumps(get_product, indent=2) + "\n", encoding="utf-8"
    )
    print("updated", COLLECTION)
    print("products folder steps:", len(products_folder["item"]))


if __name__ == "__main__":
    main()
