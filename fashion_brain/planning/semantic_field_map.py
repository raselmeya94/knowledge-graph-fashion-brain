"""Graph-aware semantic field aliases (ecom ↔ plmex).

Does **not** create Neo4j relationships. Rewrites planner filter *keys* onto the
facet that actually holds the value on the active product graph.

Confirmed starters (see data/generated/eval/SEMANTIC_MAPPING_CROSSCHECK.md):
  section ↔ gender
  subfamily ↔ garment_type
  colors / colour → color (value side; key already normalized)
  composition → material (filter keys; group_by uses material-like fallbacks)
  construction ↔ composition ↔ material (same meaning, different column names)
  brand ↔ buyer (retailer aliases; Matalan prefers buyer on PLM)

Never remaps ``category`` *filter values* across graphs (false friend).
For *group_by* / analysis, ``category`` is a hierarchy *role* — when that column
is empty on the active graph, try role-similar columns (family/section/… on ecom,
plm_main_category/garment_type/… on PLM).

Same for construction/material: PLM ``construction`` is often empty while
``composition`` / ``material`` hold the blend; ecom often stores body composition
under ``construction``. Fall through sibling columns until a usable mix appears.
"""

from __future__ import annotations

from typing import Any

from knowledge_graph.ontology import normalize_name

# Retailers where ecom brand aligns primarily with PLM *buyer* (not PLM brand).
_PLM_PREFER_BUYER: frozenset[str] = frozenset(
    {
        "matalan",
    }
)

# User said "by category/type/family" — hierarchy role, not a single column name.
_HIERARCHY_GROUP_KEYS: frozenset[str] = frozenset(
    {
        "category",
        "garment",
        "garment_type",
        "family",
        "subfamily",
        "section",
        "plm_main_category",
        "plm_sub_category",
        "plm_sub_sub_category",
        "product_type",
        "product_group",
    }
)

# User said construction / composition / material — same fibre/blend meaning.
_MATERIAL_GROUP_KEYS: frozenset[str] = frozenset(
    {
        "construction",
        "composition",
        "material",
        "materials",
        "fabric",
        "fabrics",
        "ai_construction_tags",
    }
)

# Normalize synonym group_by keys onto the canonical facet name.
_MATERIAL_GROUP_ALIASES: dict[str, str] = {
    "materials": "material",
    "fabric": "material",
    "fabrics": "material",
}

def _is_ecom(graph_id: str | None) -> bool:
    g = str(graph_id or "").strip().casefold()
    return bool(g) and "plm" not in g and ("ecom" in g or g in {"master_products", "master"})


def _is_plmex(graph_id: str | None) -> bool:
    g = str(graph_id or "").strip().casefold()
    return bool(g) and "plm" in g


def _party_token(value: str) -> str:
    return normalize_name(value).replace(" ", "").replace("-", "").replace("&", "and")


def apply_semantic_field_map(
    filters: dict[str, Any] | None,
    graph_id: str | None,
    *,
    group_by: str | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Rewrite filters (and optional group_by) for the active product graph.

    Idempotent. Unknown / empty graph_id → return filters unchanged.
    """

    raw = {
        str(k).strip(): v
        for k, v in (filters or {}).items()
        if k not in (None, "") and v not in (None, "")
    }
    out = dict(raw)
    gb = str(group_by).strip() if group_by not in (None, "") else None

    if _is_ecom(graph_id):
        out, gb = _map_for_ecom(out, gb)
    elif _is_plmex(graph_id):
        out, gb = _map_for_plmex(out, gb)

    # Universal key aliases (both graphs).
    if "colour" in out and "color" not in out:
        out["color"] = out.pop("colour")
    elif "colour" in out:
        out.pop("colour", None)
    if "colors" in out and "color" not in out:
        out["color"] = out.pop("colors")
    elif "colors" in out:
        out.pop("colors", None)
    if "composition" in out and "material" not in out:
        out["material"] = out.pop("composition")
    elif "composition" in out:
        out.pop("composition", None)

    return out, gb


def _map_for_ecom(out: dict[str, Any], gb: str | None) -> tuple[dict[str, Any], str | None]:
    # Audience: Fashion Brain v2 writes FOR_TARGET_GROUP hubs; keep gender AND mirror to section
    # so both TargetGroup and legacy Section filters match.
    if "target_group" in out:
        tg = out.pop("target_group")
        out.setdefault("gender", tg)
        out.setdefault("section", tg)
    if "gender" in out and "section" not in out:
        out["section"] = out["gender"]

    # Garment noun: garment_type → subfamily (ecom controlled vocab).
    if "garment_type" in out and "subfamily" not in out and "garment" not in out:
        out["subfamily"] = out.pop("garment_type")
    elif "garment_type" in out and ("subfamily" in out or "garment" in out):
        out.pop("garment_type", None)

    # Buyer unsupported on ecom → brand.
    if "buyer" in out and "brand" not in out:
        out["brand"] = out.pop("buyer")
    elif "buyer" in out:
        # Same party already as brand, or brand wins on ecom.
        out.pop("buyer", None)

    if gb == "gender":
        gb = "section"
    elif gb == "garment_type":
        gb = "subfamily"
    elif gb == "buyer":
        gb = "brand"

    return out, gb


def _map_for_plmex(out: dict[str, Any], gb: str | None) -> tuple[dict[str, Any], str | None]:
    # Audience: section / target_group → gender (HAS_GENDER + FOR_TARGET_GROUP).
    if "target_group" in out:
        tg = out.pop("target_group")
        out.setdefault("gender", tg)
    if "section" in out and "gender" not in out:
        out["gender"] = out.pop("section")
    elif "section" in out:
        out.pop("section", None)

    # Garment noun: subfamily → garment_type (PLM controlled vocab).
    if "subfamily" in out and "garment_type" not in out and "garment" not in out:
        out["garment_type"] = out.pop("subfamily")
    elif "subfamily" in out and ("garment_type" in out or "garment" in out):
        out.pop("subfamily", None)

    # family alone is not remapped to plm_sub_* (role-only; value sets differ).
    # Retail party: brand → buyer for known retailers (esp. Matalan).
    brand = out.get("brand")
    if brand and "buyer" not in out:
        compact = normalize_name(str(brand))
        prefer_buyer = compact in _PLM_PREFER_BUYER or _party_token(str(brand)) in {
            _party_token(x) for x in _PLM_PREFER_BUYER
        }
        if prefer_buyer:
            out["buyer"] = out.pop("brand")
        # primark/zara: leave brand (both facets populated on PLM).

    if gb == "section":
        gb = "gender"
    elif gb == "subfamily":
        gb = "garment_type"
    elif gb == "family":
        # Soft: prefer plm_sub_category for breakdowns when user said family.
        gb = "plm_sub_category"

    return out, gb


def remap_group_by(group_by: str | None, graph_id: str | None) -> str | None:
    """Standalone group_by remap (when filters are unchanged)."""

    _, gb = apply_semantic_field_map({}, graph_id, group_by=group_by)
    return gb


def category_like_fallbacks(graph_id: str | None) -> list[str]:
    """Role-similar hierarchy columns when the asked category-like facet is empty.

    Order = prefer first. Value sets differ across graphs — this is role fallback
    for *breakdowns*, not a value merge.
    """

    if _is_ecom(graph_id):
        return ["category", "family", "section", "subfamily"]
    if _is_plmex(graph_id):
        return [
            "category",
            "plm_main_category",
            "plm_sub_category",
            "garment_type",
            "plm_sub_sub_category",
        ]
    return ["category", "family", "garment_type", "subfamily"]


def material_like_fallbacks(graph_id: str | None) -> list[str]:
    """Sibling fibre/blend columns when construction/composition/material is empty.

    Order = prefer first. Column names differ across graphs; meaning is the same.
    """

    if _is_ecom(graph_id):
        # Ecom PDPs often put body composition into ``construction``.
        return ["construction", "composition", "material"]
    if _is_plmex(graph_id):
        # PLM ``construction`` is sparse; blend strings live on ``composition``.
        return ["construction", "composition", "material"]
    return ["construction", "composition", "material"]


def resolve_group_by_candidates(
    group_by: str | None,
    graph_id: str | None,
) -> list[str]:
    """Primary remapped group_by plus hierarchy/material role fallbacks.

    Example: user/LLM said ``group_by=category`` on ecom → try category, then
    family, section, subfamily until one facet has real product mix.

    Example: ``group_by=construction`` on PLM → try construction, then
    composition, then material (same blend meaning).
    """

    raw = str(group_by or "").strip()
    if not raw:
        return []
    primary = remap_group_by(raw, graph_id) or raw
    folded = primary.strip().casefold().replace(" ", "_").replace("-", "_")
    if not folded:
        return []
    folded = _MATERIAL_GROUP_ALIASES.get(folded, folded)
    out: list[str] = [folded]
    role_key = raw.strip().casefold().replace(" ", "_").replace("-", "_")
    role_key = _MATERIAL_GROUP_ALIASES.get(role_key, role_key)
    if folded in _HIERARCHY_GROUP_KEYS or role_key in _HIERARCHY_GROUP_KEYS:
        for alt in category_like_fallbacks(graph_id):
            if alt not in out:
                out.append(alt)
    if folded in _MATERIAL_GROUP_KEYS or role_key in _MATERIAL_GROUP_KEYS:
        for alt in material_like_fallbacks(graph_id):
            if alt not in out:
                out.append(alt)
    return out
