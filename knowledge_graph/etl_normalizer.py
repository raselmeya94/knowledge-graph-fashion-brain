"""Deterministic Fashion Brain normalizers for ecom + plmex ETL.

Strip percentages out of Material names, canonicalize colors/gauges/refs,
and emit stable hub IDs so both sources can share Brand / Material / Color /
Reference / TargetGroup nodes in Neo4j.
"""

from __future__ import annotations

import re
from typing import Any

from knowledge_graph.ids import slug
from knowledge_graph.ontology import normalize_name

# Fiber aliases → canonical fiber name (no percentages).
MATERIAL_ALIASES: dict[str, str] = {
    "spandex": "elastane",
    "lycra": "elastane",
    "polyamide": "nylon",
    "pa": "nylon",
    "rayon": "viscose",
    "viscose rayon": "viscose",
    "recycled poly": "recycled polyester",
    "rpet": "recycled polyester",
    "polyester recycled": "recycled polyester",
    "recycled-polyester": "recycled polyester",
}

COLOR_ALIASES: dict[str, str] = {
    "gery": "grey",
    "gray": "grey",
    "light gray": "light grey",
    "dk grey": "dark grey",
    "dk gray": "dark grey",
    "multi colour": "multicolour",
    "multi-colour": "multicolour",
    "multi color": "multicolour",
    "multi-color": "multicolour",
    "multi": "multicolour",
    "multicolor": "multicolour",
    "off white": "off-white",
    "ecru": "off-white",
}

PATTERN_TOKENS = {
    "stripes",
    "striped",
    "stripe",
    "floral",
    "check",
    "checked",
    "plaid",
    "leopard",
    "polka dot",
    "polka-dot",
    "print",
    "printed",
}

BRAND_PLACEHOLDERS = {
    "uk showroom sample",
    "shopping sample(uk)",
    "shopping sample (uk)",
    "bd design team",
    "older girl",
    "infant",
}

BRAND_TO_TARGET_GROUP = {
    "older girl": "girls",
    "infant": "baby",
}

GENDER_TO_TARGET_GROUP = {
    "women": "women",
    "woman": "women",
    "men": "men",
    "man": "men",
    "girl": "girls",
    "girls": "girls",
    "boy": "boys",
    "boys": "boys",
    "baby girl": "baby",
    "baby boy": "baby",
    "maternal & infants": "baby",
    "maternal and infants": "baby",
    "unisex": "unisex",
    "pets": "pets",
    "others": "others",
    "kids": "kids",
    "children": "kids",
}

SECTION_TARGET_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"woman|women|ladies", re.I), "women"),
    (re.compile(r"\bman\b|\bmen\b", re.I), "men"),
    (re.compile(r"girl|girls", re.I), "girls"),
    (re.compile(r"boy|boys", re.I), "boys"),
    (re.compile(r"baby|newborn|infant", re.I), "baby"),
    (re.compile(r"kids|children", re.I), "kids"),
    (re.compile(r"unisex", re.I), "unisex"),
    (re.compile(r"pets?", re.I), "pets"),
]

_COMPOSITION = re.compile(
    r"(?P<pct>\d+(?:\.\d+)?)\s*%\s*(?P<name>[A-Za-z][A-Za-z0-9 \-/]+?)"
    r"(?=\s*\d+\s*%|$|,|;|/)"
    r"|"
    r"(?P<name2>[A-Za-z][A-Za-z0-9 \-/]+?)\s+(?P<pct2>\d+(?:\.\d+)?)\s*%?"
)

_COMPOSITION_FALLBACK = re.compile(
    r"(?P<pct>\d+(?:\.\d+)?)\s*%\s*(?P<name>[A-Za-z][A-Za-z0-9 \-/]+)"
    r"|"
    r"(?P<only>[A-Za-z][A-Za-z0-9 \-/]{1,40})"
)

_GAUGE_G = re.compile(r"^(\d+(?:\.\d+)?)G$", re.I)
_COLOR_SPLIT = re.compile(r"\s*(?:,|/|&|\band\b)\s*", re.I)
_EMPTY = {"", "null", "none", "n/a", "na", "-", "#n/a"}


def present(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        text = value.isoformat()
    else:
        text = str(value).strip()
    if not text or text.lower() in _EMPTY:
        return ""
    return text


def parse_decimal(value: Any) -> float | None:
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


def pct_discount(original: float | None, price: float | None) -> float | None:
    if original is None or price is None or original <= 0:
        return None
    return round((original - price) / original * 100.0, 2)


def price_band(price: float | None, currency: str | None = None) -> str | None:
    if price is None:
        return None
    # GBP/EUR-style retail buckets from the Fashion Brain audit.
    if price < 15:
        band = "budget"
    elif price < 35:
        band = "value"
    elif price < 75:
        band = "mid"
    else:
        band = "premium"
    cur = (currency or "gbp").strip().lower() or "gbp"
    return f"{cur}:{band}"


def normalize_reference(raw: Any) -> str | None:
    text = present(raw)
    if not text:
        return None
    norm = re.sub(r"[^A-Z0-9]", "", text.upper())
    return norm if len(norm) >= 4 else None


def coalesce_reference(*values: Any) -> str | None:
    for value in values:
        norm = normalize_reference(value)
        if norm:
            return norm
    return None


def clean_fiber_name(raw: str) -> str:
    text = re.sub(r"\s+", " ", present(raw)).strip().lower()
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "", text).strip()
    text = re.sub(r"\s+", " ", text).strip(" -/")
    if not text:
        return ""
    return MATERIAL_ALIASES.get(text, text)


def parse_composition(raw: Any) -> list[dict[str, Any]]:
    """Return [{id, name, percentage}] with fiber-only Material ids."""

    text = present(raw)
    if not text:
        return []

    matches = list(_COMPOSITION.finditer(text))
    pairs: list[tuple[str, float | None]] = []
    if matches:
        for match in matches:
            if match.group("name"):
                pairs.append((match.group("name"), float(match.group("pct"))))
            elif match.group("name2"):
                pairs.append((match.group("name2"), float(match.group("pct2"))))
    else:
        for match in _COMPOSITION_FALLBACK.finditer(text):
            if match.group("name"):
                pairs.append((match.group("name"), float(match.group("pct"))))
            elif match.group("only"):
                token = match.group("only").strip()
                if token.lower() not in {"and", "with", "of"}:
                    pairs.append((token, None))

    aggregated: dict[str, float | None] = {}
    for fiber, pct in pairs:
        canon = clean_fiber_name(fiber)
        if not canon:
            continue
        if pct is None:
            aggregated.setdefault(canon, None)
        else:
            prev = aggregated.get(canon)
            aggregated[canon] = (prev or 0.0) + float(pct)

    return [
        {
            "id": hub_id("material", name),
            "name": name,
            "percentage": _clamp_percentage(pct),
        }
        for name, pct in aggregated.items()
    ]


def _clamp_percentage(pct: float | None) -> float | None:
    """Keep edge percentages in a query-safe 0..100 range (drop absurd aggregates)."""

    if pct is None:
        return None
    try:
        value = float(pct)
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    if value > 100:
        return 100.0
    return value


def parse_composition_detail(detail: Any, fallback_text: Any = None) -> list[dict[str, Any]]:
    items: list[Any]
    if detail is None:
        items = []
    elif isinstance(detail, list):
        items = detail
    elif isinstance(detail, str):
        text = detail.strip()
        if not text:
            items = []
        else:
            try:
                import json

                parsed = json.loads(text)
                items = parsed if isinstance(parsed, list) else [parsed]
            except Exception:
                items = [text]
    else:
        items = [detail]

    found: dict[str, float | None] = {}
    for item in items:
        if isinstance(item, dict):
            name = present(item.get("name") or item.get("material") or item.get("fiber"))
            pct = item.get("percentage")
            if pct is None:
                pct = item.get("percent")
            canon = clean_fiber_name(name)
            if not canon:
                continue
            if pct is None or pct == "":
                found.setdefault(canon, None)
            else:
                try:
                    num = float(pct)
                except (TypeError, ValueError):
                    found.setdefault(canon, None)
                else:
                    prev = found.get(canon)
                    found[canon] = (prev or 0.0) + num
        elif present(item):
            for row in parse_composition(item):
                name = row["name"]
                pct = row["percentage"]
                if pct is None:
                    found.setdefault(name, None)
                else:
                    prev = found.get(name)
                    found[name] = (prev or 0.0) + float(pct)

    if found:
        return [
            {
                "id": hub_id("material", name),
                "name": name,
                "percentage": _clamp_percentage(pct),
            }
            for name, pct in found.items()
        ]
    return parse_composition(fallback_text)


def parse_colors(raw: Any) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split multi-colors, fix typos, route pattern tokens to Pattern."""

    if isinstance(raw, list):
        tokens = [present(x) for x in raw if present(x)]
    else:
        text = present(raw)
        if not text:
            return [], []
        # Prefer JSON list when colors arrive as jsonb string.
        if text.startswith("["):
            try:
                import json

                parsed = json.loads(text)
                if isinstance(parsed, list):
                    tokens = []
                    for item in parsed:
                        if isinstance(item, dict):
                            tokens.append(present(item.get("name") or item.get("color")))
                        else:
                            tokens.append(present(item))
                    tokens = [t for t in tokens if t]
                else:
                    tokens = _COLOR_SPLIT.split(text)
            except Exception:
                tokens = _COLOR_SPLIT.split(text)
        else:
            tokens = _COLOR_SPLIT.split(text)

    colors: list[dict[str, str]] = []
    patterns: list[dict[str, str]] = []
    seen_c: set[str] = set()
    seen_p: set[str] = set()
    for tok in tokens:
        t = re.sub(r"\s+", " ", tok).strip().lower()
        if not t:
            continue
        if t in PATTERN_TOKENS:
            key = slug(t)
            if key not in seen_p:
                seen_p.add(key)
                patterns.append({"id": hub_id("pattern", t), "name": t})
            continue
        canon = COLOR_ALIASES.get(t, t)
        key = slug(canon)
        if key not in seen_c:
            seen_c.add(key)
            colors.append({"id": hub_id("color", canon), "name": canon})
    return colors, patterns


def parse_gauges(raw: Any) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split gauges, normalize 7G→7GG, route CROCHET to Construction."""

    text = present(raw)
    if not text:
        return [], []
    gauges: list[dict[str, str]] = []
    constructions: list[dict[str, str]] = []
    seen_g: set[str] = set()
    seen_c: set[str] = set()
    for tok in re.split(r"[,/;]+", text):
        t = tok.strip().upper()
        if not t:
            continue
        if t == "CROCHET":
            if "crochet" not in seen_c:
                seen_c.add("crochet")
                constructions.append({"id": hub_id("construction", "crochet"), "name": "crochet"})
            continue
        t = _GAUGE_G.sub(r"\1GG", t)
        key = slug(t)
        if key not in seen_g:
            seen_g.add(key)
            gauges.append({"id": hub_id("gauge", t), "name": t})
    return gauges, constructions


def target_group_from_gender(gender: Any) -> str | None:
    key = present(gender).lower()
    if not key:
        return None
    return GENDER_TO_TARGET_GROUP.get(key)


def target_group_from_section(*texts: Any) -> str | None:
    blob = " ".join(present(t) for t in texts if present(t))
    if not blob:
        return None
    for pattern, group in SECTION_TARGET_RULES:
        if pattern.search(blob):
            return group
    return None


def clean_brand(raw: Any) -> tuple[str | None, str | None]:
    """Return (canonical_brand_or_None, target_group_from_placeholder_or_None)."""

    name = present(raw)
    if not name:
        return None, None
    low = name.lower()
    if low in BRAND_PLACEHOLDERS:
        return None, BRAND_TO_TARGET_GROUP.get(low)
    return name, None


def clean_season_year(raw: Any) -> str | None:
    text = present(raw)
    if not text:
        return None
    fixes = {"2205": "2025"}
    text = fixes.get(text, text)
    if re.fullmatch(r"\d{4}", text):
        year = int(text)
        if 2010 <= year <= 2030:
            return text
        return None
    return text


def hub_id(kind: str, name: str) -> str:
    """Deterministic cross-source hub ids."""

    key = slug(normalize_name(name) or name)
    kind = kind.strip().lower()
    if kind in {
        "material",
        "color",
        "gauge",
        "construction",
        "occasion",
        "care",
        "fit",
        "pattern",
        "feature",
        "stylefeature",
        "finish",
        "special",
        "aimaterial",
    }:
        return f"fashion:{kind}:{key}"
    if kind == "brand":
        return f"brand:{key}"
    if kind == "targetgroup":
        return f"targetgroup:{key}"
    if kind == "reference":
        return f"ref:{key.upper() if key.isalnum() else key}"
    if kind == "retailgroup":
        return f"group:{key}"
    if kind == "materialgroup":
        return f"materialgroup:{key}"
    if kind == "colorfamily":
        return f"colorfamily:{key}"
    if kind == "priceband":
        return f"priceband:{key}"
    if kind == "seasonyear":
        return f"seasonyear:{key}"
    return f"{kind}:{key}"


def reference_id(ref_norm: str) -> str:
    return f"ref:{ref_norm}"


def inventory_id(code: str) -> str:
    return f"plmex:inv:{present(code)}"


def shipment_id(code: str, shipment_date: Any) -> str:
    date = present(shipment_date) or "unknown"
    return f"plmex:ship:{present(code)}:{slug(date)}"


def synthesize_plmex_display_name(product: dict[str, Any]) -> str:
    parts = [
        present(product.get("gender")),
        present(product.get("color")),
        present(product.get("garment_type") or product.get("plm_sub_sub_category")),
    ]
    name = " ".join(p for p in parts if p)
    composition = present(product.get("composition"))
    if composition:
        name = f"{name} ({composition})" if name else composition
    return name or present(product.get("code")) or "plmex product"


def synthesize_plmex_search_text(product: dict[str, Any]) -> str:
    keys = [
        "brand",
        "buyer",
        "gender",
        "plm_main_category",
        "plm_sub_category",
        "plm_sub_sub_category",
        "garment_type",
        "category",
        "gauge",
        "composition",
        "color",
        "design_feature",
        "style_feature",
        "ai_construction_tags",
        "ai_material_tags",
        "ai_event_tags",
        "ai_finish_tags",
        "ai_special_tags",
        "style_reference",
        "customer_style_reference",
    ]
    return " ".join(present(product.get(k)) for k in keys if present(product.get(k)))


def synthesize_ecom_search_text(product: dict[str, Any]) -> str:
    keys = [
        "brand",
        "section",
        "name",
        "product_name",
        "category_path",
        "composition",
        "construction",
        "fit",
        "description",
        "care",
    ]
    return " ".join(present(product.get(k)) for k in keys if present(product.get(k)))
