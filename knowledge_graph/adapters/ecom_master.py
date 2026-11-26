"""Build KG product + Neo4j relationship proposals from a master_products row.

No Postgres connection here — pass a row dict (from SQL or tests). Sync/adapter
wiring comes next; this module is the schema→graph contract for e-comm v1.
"""

from __future__ import annotations

import json
from typing import Any

from knowledge_graph import etl_normalizer as norm
from knowledge_graph.config import DATA_ROOT
from knowledge_graph.ontology import normalize_name

_MAPPING_PATH = DATA_ROOT / "schemas" / "ecom_master_products_mapping.json"


def load_mapping() -> dict:
    if not _MAPPING_PATH.is_file():
        raise FileNotFoundError(
            f"E-com mapping missing: {_MAPPING_PATH}. "
            "POST /api/sources/ecom_products/mapping with the mapping JSON first."
        )
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
    price_num = parse_price_num(row.get("price"))
    original_price_num = parse_price_num(row.get("original_price"))
    ref_norm = norm.coalesce_reference(row.get("reference"), row.get("display_reference"))
    target_group = norm.target_group_from_section(row.get("section"), row.get("category_path"))
    product = {
        "product_id": product_id(brand, master_id),
        "source": "ecom",
        "source_table": "master_products",
        "master_id": master_id,
        "brand": brand,
        "product_name": _text(row.get("name")),
        "name": _text(row.get("name")),
        "reference": _text(row.get("reference")),
        "display_reference": _text(row.get("display_reference")),
        "ref_norm": ref_norm,
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
        "price_num": price_num,
        "original_price": _text(row.get("original_price")),
        "original_price_num": original_price_num,
        "discount_pct": norm.pct_discount(original_price_num, price_num),
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
        "_target_group": target_group,
        "_price_band": norm.price_band(price_num, row.get("currency")),
        "_ref_norm": ref_norm,
    }
    product["search_text"] = norm.synthesize_ecom_search_text(product)
    return product


def build_relationships(product: dict[str, Any]) -> list[dict[str, Any]]:
    """Return Neo4j relationship proposals for one normalized e-comm product."""

    pid = product["product_id"]
    rels: list[dict[str, Any]] = []

    def edge(
        rel_type: str,
        end_type: str,
        name: str,
        *,
        field_ref: str,
        properties: dict | None = None,
        end_id: str | None = None,
    ) -> None:
        value = _present(name)
        if not value and not end_id:
            return
        resolved = end_id
        if not resolved:
            if end_type == "Brand":
                resolved = norm.hub_id("brand", value)
            elif end_type == "Material":
                resolved = norm.hub_id("material", value)
            elif end_type == "Color":
                resolved = norm.hub_id("color", value)
            elif end_type == "Construction":
                resolved = norm.hub_id("construction", value)
            elif end_type == "Fit":
                resolved = norm.hub_id("fit", value)
            elif end_type == "CareInstruction":
                resolved = norm.hub_id("care", value)
            elif end_type == "TargetGroup":
                resolved = norm.hub_id("targetgroup", value)
            elif end_type == "PriceBand":
                resolved = norm.hub_id("priceband", value)
            elif end_type == "Reference":
                resolved = norm.reference_id(value)
            else:
                resolved = f"{end_type.lower()}:{normalize_name(value).replace(' ', '-')}"
        rels.append(
            {
                "rel_type": rel_type,
                "start_id": pid,
                "start_type": "Product",
                "end_type": end_type,
                "end_name": value or resolved,
                "end_id": resolved,
                "field_ref": field_ref,
                "properties": properties or {},
            }
        )

    edge("HAS_BRAND", "Brand", product.get("brand", ""), field_ref="brand")
    edge("BELONGS_TO_CATEGORY", "GarmentCategory", product.get("category", ""), field_ref="category")
    edge("HAS_FAMILY", "Family", product.get("family", ""), field_ref="family")
    edge("HAS_SUBFAMILY", "Subfamily", product.get("subfamily", ""), field_ref="subfamily")
    edge("HAS_SECTION", "Section", product.get("section", ""), field_ref="section")
    edge("HAS_AVAILABILITY", "Availability", product.get("availability", ""), field_ref="availability")
    edge("PRICED_IN", "Currency", product.get("currency", ""), field_ref="currency")
    edge("IN_CATALOG", "Catalog", product.get("catalog_id", ""), field_ref="catalog_id")
    edge("SCRAPED_FROM", "Domain", product.get("domain", ""), field_ref="domain")
    edge("HAS_CONSTRUCTION", "Construction", product.get("construction", ""), field_ref="construction")
    edge("HAS_FIT", "Fit", product.get("fit", ""), field_ref="fit")
    edge("HAS_CARE", "CareInstruction", product.get("care", ""), field_ref="care")
    edge("MADE_IN", "Country", product.get("country_of_origin", ""), field_ref="country_of_origin")
    ref = product.get("_ref_norm") or product.get("ref_norm")
    if ref:
        edge(
            "HAS_REFERENCE",
            "Reference",
            ref,
            field_ref="reference",
            properties={
                "raw_reference": _present(product.get("reference")) or _present(product.get("display_reference")),
                "ref_kind": "ecom_ref",
            },
            end_id=norm.reference_id(str(ref)),
        )
    edge("HAS_MAIN_IMAGE", "Image", product.get("image", ""), field_ref="image")

    tg = _present(product.get("_target_group"))
    if tg:
        edge("FOR_TARGET_GROUP", "TargetGroup", tg, field_ref="section", properties={"provenance": "rule"})
    band = _present(product.get("_price_band"))
    if band:
        edge("IN_PRICE_BAND", "PriceBand", band, field_ref="price")

    colors, patterns = norm.parse_colors(product.get("colors"))
    for index, color in enumerate(colors):
        edge(
            "HAS_COLOR",
            "Color",
            color["name"],
            field_ref="colors",
            properties={"position": index, "is_primary": index == 0},
            end_id=color["id"],
        )
    if colors:
        edge("PRIMARY_COLOR", "Color", colors[0]["name"], field_ref="colors", end_id=colors[0]["id"])
    for pattern in patterns:
        edge("HAS_PATTERN", "Pattern", pattern["name"], field_ref="colors", end_id=pattern["id"])

    materials = norm.parse_composition_detail(
        product.get("composition_detail"), product.get("composition")
    )
    for material in materials:
        props = {"percentage": material["percentage"]} if material["percentage"] is not None else {}
        edge(
            "USES_MATERIAL",
            "Material",
            material["name"],
            field_ref="composition",
            properties=props,
            end_id=material["id"],
        )
    if materials:
        primary = max(
            materials,
            key=lambda item: (
                item["percentage"] is not None,
                item["percentage"] if item["percentage"] is not None else -1.0,
            ),
        )
        primary_props = {"percentage": primary["percentage"]} if primary["percentage"] is not None else {}
        edge(
            "PRIMARY_MATERIAL",
            "Material",
            primary["name"],
            field_ref="composition",
            properties=primary_props,
            end_id=primary["id"],
        )

    # Hierarchy: Subfamily → Family → GarmentCategory (ontology PARENT_CATEGORY pairs)
    family = _present(product.get("family"))
    category = _present(product.get("category"))
    subfamily = _present(product.get("subfamily"))
    if subfamily and family:
        rels.append(_parent_link("Subfamily", subfamily, "Family", family, "subfamily→family"))
    if family and category:
        rels.append(_parent_link("Family", family, "GarmentCategory", category, "family→category"))

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
    if product.get("search_text"):
        props["search_text"] = product["search_text"]
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
    if product.get("discount_pct") is not None:
        props["discount_pct"] = product.get("discount_pct")
    if product.get("ref_norm"):
        props["ref_norm"] = product.get("ref_norm")
    return {
        "product_id": product["product_id"],
        "labels": ["Product", "Entity"],
        "properties": props,
        "relationships": build_relationships(product),
    }


def _parent_link(
    start_type: str,
    child: str,
    end_type: str,
    parent: str,
    note: str,
) -> dict[str, Any]:
    return {
        "rel_type": "PARENT_CATEGORY",
        "start_type": start_type,
        "start_name": child,
        "start_id": f"{start_type.lower()}:{normalize_name(child).replace(' ', '-')}",
        "end_type": end_type,
        "end_name": parent,
        "end_id": f"{end_type.lower()}:{normalize_name(parent).replace(' ', '-')}",
        "field_ref": note,
        "properties": {},
    }


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


def parse_price_num(value: Any) -> float | None:
    """Parse display prices like €219.00 / $14.00 / £7.00 into a float."""

    return norm.parse_decimal(value)
