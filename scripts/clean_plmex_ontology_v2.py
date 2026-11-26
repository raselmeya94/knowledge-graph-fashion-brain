"""One-shot cleaner: plmex ontology v1.1 → Fashion Brain v2.0."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "data" / "schemas" / "ontologies" / "plmex.json"

COLOR_ALIASES = {
    "GERY": "GREY",
    "LIGHT GRAY": "GREY",
    "GRAY": "GREY",
    "MULTI COLOUR": "MULTICOLOUR",
    "MULTI-COLOUR": "MULTICOLOUR",
}
PATTERN_TOKENS = {"STRIPES", "STRIPED", "FLORAL", "CHECK", "PLAID"}
BRAND_DROP = {
    "UK SHOWROOM SAMPLE",
    "SHOPPING SAMPLE(UK)",
    "BD DESIGN TEAM",
    "OLDER GIRL",
    "INFANT",
}
FIBER_ALIASES = {"spandex": "elastane", "polyamide": "nylon", "rayon": "viscose"}


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")


def clean_fiber(name: str) -> str:
    text = re.sub(r"\s+", " ", name.strip().lower())
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "", text).strip()
    text = re.sub(r"\s+", " ", text).strip(" -/")
    return FIBER_ALIASES.get(text, text)


def main() -> None:
    ontology = json.loads(PATH.read_text(encoding="utf-8"))
    ontology["version"] = "2.0.0"
    ontology["description"] = (
        "PLMEX ontology v2 for Fashion Brain. Product is central (code PK). "
        "Shared hubs (Brand, Material, Color, Reference, TargetGroup, Construction, Occasion) "
        "use fashion_brain IDs. category maps to ProductionMethod (not GarmentCategory). "
        "Materials are pure fiber names; percentages live on USES_MATERIAL edges."
    )
    ontology["entity_types"] = [
        "Product",
        "Brand",
        "Buyer",
        "RetailGroup",
        "Material",
        "MaterialGroup",
        "ProductionMethod",
        "PlmMainCategory",
        "PlmSubCategory",
        "PlmSubSubCategory",
        "GarmentType",
        "CanonicalCategory",
        "Gender",
        "TargetGroup",
        "Color",
        "ColorFamily",
        "Pattern",
        "Gauge",
        "SampleType",
        "DesignPurpose",
        "MachineType",
        "DesignFeature",
        "StyleFeature",
        "Department",
        "ShoppingDepartment",
        "SeasonYear",
        "Person",
        "Reference",
        "Location",
        "InventoryRecord",
        "Shipment",
        "Unit",
        "Occasion",
        "Construction",
        "EventTag",
        "ConstructionTag",
        "AiMaterialTag",
        "FinishTag",
        "SpecialTag",
    ]

    allow = ontology.get("relationship_allowlist") or {}
    allow.pop("HAS_STYLE_REFERENCE", None)
    allow.pop("HAS_CUSTOMER_STYLE_REFERENCE", None)
    allow.pop("BELONGS_TO_CATEGORY", None)
    if "PARENT_CATEGORY" in allow:
        allow["PARENT_CATEGORY"] = [
            pair
            for pair in allow["PARENT_CATEGORY"]
            if pair != ["GarmentCategory", "PlmMainCategory"]
        ]
    allow["MADE_WITH_METHOD"] = [["Product", "ProductionMethod"]]
    allow["FOR_TARGET_GROUP"] = [["Product", "TargetGroup"]]
    allow["HAS_REFERENCE"] = [["Product", "Reference"]]
    allow["SUITED_FOR_OCCASION"] = [["Product", "Occasion"]]
    allow["HAS_CONSTRUCTION"] = [["Product", "Construction"]]
    allow["HAS_PATTERN"] = [["Product", "Pattern"]]
    allow["IN_DEPARTMENT"] = [["Person", "Department"]]
    allow["IN_GROUP"] = [["Buyer", "RetailGroup"], ["Brand", "RetailGroup"]]
    allow["IN_MATERIAL_GROUP"] = [["Material", "MaterialGroup"]]
    allow["IN_COLOR_FAMILY"] = [["Color", "ColorFamily"]]
    allow["MAPS_TO_CATEGORY"] = [
        ["GarmentType", "CanonicalCategory"],
        ["PlmSubSubCategory", "CanonicalCategory"],
        ["PlmSubCategory", "CanonicalCategory"],
        ["Product", "CanonicalCategory"],
    ]
    allow["REFERS_TO_MATERIAL"] = [["AiMaterialTag", "Material"]]
    allow["RESOLVES_TO"] = [["Product", "Product"]]
    allow["SIMILAR_TO"] = [["Product", "Product"]]
    allow["STORED_AT"] = [["InventoryRecord", "Location"]]
    ontology["relationship_allowlist"] = allow

    kept: list[dict] = []
    seen: set[str] = set()

    def add(entity: dict) -> None:
        eid = entity["id"]
        if eid in seen:
            for existing in kept:
                if existing["id"] != eid:
                    continue
                for alias in entity.get("aliases") or []:
                    if alias not in existing["aliases"]:
                        existing["aliases"].append(alias)
            return
        seen.add(eid)
        kept.append(entity)

    for entity in ontology.get("canonical_entities") or []:
        etype = entity.get("type")
        name = entity.get("name") or ""
        if etype == "Material":
            fiber = clean_fiber(name)
            if not fiber:
                continue
            add(
                {
                    "id": f"fashion:material:{slug(fiber)}",
                    "type": "Material",
                    "name": fiber,
                    "aliases": [name] if name.upper() != fiber.upper() else [fiber.upper()],
                    "external_ids": [],
                }
            )
            continue
        if etype == "Color":
            up = name.upper().strip()
            if up in PATTERN_TOKENS:
                add(
                    {
                        "id": f"fashion:pattern:{slug(up)}",
                        "type": "Pattern",
                        "name": up.lower(),
                        "aliases": [up],
                        "external_ids": [],
                    }
                )
                continue
            canon = COLOR_ALIASES.get(up, up)
            parts = [COLOR_ALIASES.get(part.strip(), part.strip()) for part in canon.split(" AND ")]
            for part in parts:
                add(
                    {
                        "id": f"fashion:color:{slug(part)}",
                        "type": "Color",
                        "name": "Multicolour" if part == "MULTICOLOUR" else part.title(),
                        "aliases": [part] if part == up else [up, part],
                        "external_ids": [],
                    }
                )
            continue
        if etype == "Gauge":
            raw = name.strip().upper()
            if raw == "CROCHET":
                add(
                    {
                        "id": "fashion:construction:crochet",
                        "type": "Construction",
                        "name": "crochet",
                        "aliases": ["CROCHET"],
                        "external_ids": [],
                    }
                )
                continue
            for tok in [t.strip() for t in re.split(r"[,/]+", raw) if t.strip()]:
                tok = re.sub(r"^(\d+(?:\.\d+)?)G$", r"\1GG", tok)
                add(
                    {
                        "id": f"fashion:gauge:{slug(tok)}",
                        "type": "Gauge",
                        "name": tok,
                        "aliases": [tok],
                        "external_ids": [],
                    }
                )
            continue
        if etype == "SeasonYear" and name == "2205":
            add(
                {
                    "id": "seasonyear:2025",
                    "type": "SeasonYear",
                    "name": "2025",
                    "aliases": ["2025", "2205"],
                    "external_ids": [],
                }
            )
            continue
        if etype == "Brand" and name.upper() in BRAND_DROP:
            continue
        if etype == "Brand":
            add(
                {
                    "id": f"brand:{slug(name)}",
                    "type": "Brand",
                    "name": name,
                    "aliases": entity.get("aliases") or [name],
                    "external_ids": [],
                }
            )
            continue
        if etype == "GarmentCategory":
            add(
                {
                    "id": f"plmex:method:{slug(name)}",
                    "type": "ProductionMethod",
                    "name": name,
                    "aliases": entity.get("aliases") or [name],
                    "external_ids": [],
                }
            )
            continue
        eid = entity.get("id") or f"plmex:{str(etype).lower()}:{slug(name)}"
        clone = dict(entity)
        clone["id"] = eid
        add(clone)

    for fiber in [
        "cotton",
        "polyester",
        "recycled polyester",
        "viscose",
        "acrylic",
        "nylon",
        "elastane",
        "wool",
        "linen",
        "silk",
        "lyocell",
        "modal",
    ]:
        add(
            {
                "id": f"fashion:material:{slug(fiber)}",
                "type": "Material",
                "name": fiber,
                "aliases": [fiber.upper()],
                "external_ids": [],
            }
        )

    ontology["canonical_entities"] = kept
    PATH.write_text(json.dumps(ontology, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    materials = [e["name"] for e in kept if e["type"] == "Material"]
    print(f"wrote {PATH} entities={len(kept)} materials={materials}")


if __name__ == "__main__":
    main()
