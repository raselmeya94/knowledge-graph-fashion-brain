"""Build KG product + Neo4j relationship proposals from a master_products row.

No Postgres connection here — pass a row dict (from SQL or tests). Sync/adapter
wiring comes next; this module is the schema→graph contract for e-comm v1.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from knowledge_graph.ontology import normalize_name

_MAPPING_PATH = Path(__file__).resolve().parents[2] / "data" / "schemas" / "ecom_master_products_mapping.json"

_COMPOSITION_PART = re.compile(
    r"(?P<pct>\d+(?:\.\d+)?)\s*%\s*(?P<name>[A-Za-z][A-Za-z0-9 \-/]+)|(?P<only>[A-Za-z][A-Za-z0-9 \-/]{1,40})"
)


def load_mapping() -> dict:
    return json.loads(_MAPPING_PATH.read_text(encoding="utf-8"))


def product_id(brand: str, master_id: str) -> str:
    return f"ecom:{brand}:{master_id}"


def normalize_master_row(row: dict[str, Any]) -> dict[str, Any]:
    """Flatten a master_products row into a catalog-style product dict."""

    brand = str(row.get("brand") or "").strip()
    master_id = str(row.get("master_id") or "").strip()
    if not brand or not master_id:
        raise ValueError("master_products row requires brand and master_id")

    colors = _as_list(row.get("colors"))
    composition_detail = _as_list(row.get("composition_detail"))
    return {
        "product_id": product_id(brand, master_id),
        "source": "ecom",
        "source_table": "master_products",
        "master_id": master_id,
        "brand": brand,
        "product_name": _text(row.get("name")),
        "reference": _text(row.get("reference")),
        "display_reference": _text(row.get("display_reference")),
        "description": _text(row.get("description")),
        "composition": _text(row.get("composition")),
        "composition_detail": composition_detail,
        "construction": _text(row.get("construction")),
        "care": _text(row.get("care")),
        "fit": _text(row.get("fit")),
        "country_of_origin": _text(row.get("country_of_origin")),
        "section": _text(row.get("section")),
        "category": _text(row.get("category")),
        "family": _text(row.get("family")),
        "subfamily": _text(row.get("subfamily")),
        "category_path": _text(row.get("category_path")),
        "availability": _text(row.get("availability")),
        "in_stock_count": row.get("in_stock_count"),
        "out_stock_count": row.get("out_stock_count"),
        "currency": _text(row.get("currency")),
        "price": _text(row.get("price")),
        "price_num": parse_price_num(row.get("price")),
        "original_price": _text(row.get("original_price")),
        "original_price_num": parse_price_num(row.get("original_price")),
        "price_min": _text(row.get("price_min")),
        "price_max": _text(row.get("price_max")),
        "image": _text(row.get("image")),
        "images": _as_list(row.get("images")),
        "colors": colors,
        "color": colors[0] if colors else "",
        "color_count": row.get("color_count") if row.get("color_count") is not None else len(colors),
        "product_url": _text(row.get("product_url")),
        "product_url_with_variant": _text(row.get("product_url_with_variant")),
        "source_url": _text(row.get("source_url")),
        "domain": _text(row.get("domain")),
        "catalog_id": _text(row.get("catalog_id")),
        "scraped_at": _text(row.get("scraped_at")),
        "content_hash": _text(row.get("content_hash")),
        "pdp_enriched": bool(row.get("pdp_enriched")),
        "material": _text(row.get("composition")),
    }


def build_relationships(product: dict[str, Any]) -> list[dict[str, Any]]:
    """Return Neo4j relationship proposals for one normalized e-comm product."""

    pid = product["product_id"]
    rels: list[dict[str, Any]] = []

    def edge(rel_type: str, end_type: str, name: str, *, field_ref: str, properties: dict | None = None) -> None:
        value = _present(name)
        if not value:
            return
        rels.append(
            {
                "rel_type": rel_type,
                "start_id": pid,
                "start_type": "Product",
                "end_type": end_type,
                "end_name": value,
                "end_id": f"{end_type.lower()}:{normalize_name(value).replace(' ', '-')}",
                "field_ref": field_ref,
                "properties": properties or {},
            }
        )

    edge("HAS_BRAND", "Brand", product.get("brand", ""), field_ref="brand")
    edge("BELONGS_TO_CATEGORY", "GarmentCategory", product.get("category", ""), field_ref="category")
    edge("HAS_FAMILY", "GarmentCategory", product.get("family", ""), field_ref="family")
    edge("HAS_SUB_CATEGORY", "GarmentCategory", product.get("subfamily", ""), field_ref="subfamily")
    edge("HAS_SECTION", "Section", product.get("section", ""), field_ref="section")
    edge("HAS_AVAILABILITY", "Availability", product.get("availability", ""), field_ref="availability")
    edge("PRICED_IN", "Currency", product.get("currency", ""), field_ref="currency")
    edge("IN_CATALOG", "Catalog", product.get("catalog_id", ""), field_ref="catalog_id")
    edge("SCRAPED_FROM", "Domain", product.get("domain", ""), field_ref="domain")
    edge("HAS_CONSTRUCTION", "Construction", product.get("construction", ""), field_ref="construction")

    for color in product.get("colors") or []:
        edge("HAS_COLOR", "Color", str(color), field_ref="colors")

    for material, percentage in _materials(product):
        props = {"percentage": percentage} if percentage is not None else {}
        edge("USES_MATERIAL", "Material", material, field_ref="composition", properties=props)

    # Category hierarchy: subfamily → family → category
    family = _present(product.get("family"))
    category = _present(product.get("category"))
    subfamily = _present(product.get("subfamily"))
    if subfamily and family:
        rels.append(_parent_category(subfamily, family, "subfamily→family"))
    if family and category:
        rels.append(_parent_category(family, category, "family→category"))

    return rels


def graph_sketch(product: dict[str, Any]) -> dict[str, Any]:
    """Compact view: product node props + relationship list (for docs/tests)."""

    mapping = load_mapping()
    props = {
        key: product.get(key)
        for key in mapping["neo4j_product_properties"]
        if product.get(key) not in (None, "", [])
    }
    props["name"] = product.get("product_name") or props.get("name")
    props["source"] = "ecom"
    # Ensure numeric price props even when mapping filter skipped 0.0
    if product.get("price_num") is not None:
        props["price_num"] = product.get("price_num")
    elif product.get("price"):
        parsed = parse_price_num(product.get("price"))
        if parsed is not None:
            props["price_num"] = parsed
    if product.get("original_price_num") is not None:
        props["original_price_num"] = product.get("original_price_num")
    elif product.get("original_price"):
        parsed = parse_price_num(product.get("original_price"))
        if parsed is not None:
            props["original_price_num"] = parsed
    return {
        "product_id": product["product_id"],
        "labels": ["Product", "Entity"],
        "properties": props,
        "relationships": build_relationships(product),
    }


def _parent_category(child: str, parent: str, note: str) -> dict[str, Any]:
    return {
        "rel_type": "PARENT_CATEGORY",
        "start_type": "GarmentCategory",
        "start_name": child,
        "start_id": f"garmentcategory:{normalize_name(child).replace(' ', '-')}",
        "end_type": "GarmentCategory",
        "end_name": parent,
        "end_id": f"garmentcategory:{normalize_name(parent).replace(' ', '-')}",
        "field_ref": note,
        "properties": {},
    }


def _materials(product: dict[str, Any]) -> list[tuple[str, float | None]]:
    detail = product.get("composition_detail") or []
    found: list[tuple[str, float | None]] = []
    for item in detail:
        if isinstance(item, dict):
            name = _present(item.get("name") or item.get("material") or item.get("fiber"))
            pct = item.get("percentage")
            if pct is None:
                pct = item.get("percent")
            if name:
                found.append((name, _number(pct)))
        elif isinstance(item, str) and _present(item):
            found.append((item.strip(), None))
    if found:
        return found
    text = _present(product.get("composition"))
    if not text:
        return []
    for match in _COMPOSITION_PART.finditer(text):
        if match.group("name"):
            found.append((match.group("name").strip(), _number(match.group("pct"))))
        elif match.group("only"):
            token = match.group("only").strip()
            if token.lower() not in {"and", "with", "of"}:
                found.append((token, None))
    return found


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            return [text]
    return [value]


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _present(value: Any) -> str:
    text = _text(value)
    if not text or text.lower() in {"null", "none", "n/a", "na", "-"}:
        return ""
    return text


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_price_num(value: Any) -> float | None:
    """Parse display prices like €219.00 / $14.00 / £7.00 into a float."""

    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value)
    for ch in ("€", "$", "£", ",", " "):
        text = text.replace(ch, "")
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None
