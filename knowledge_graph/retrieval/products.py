"""Product repository. JSON/CSV envelopes are authoritative for the synthetic demo."""

from __future__ import annotations

import re

from knowledge_graph.access import Principal, allows
from knowledge_graph.errors import ProductBackendUnavailable
from knowledge_graph.models import RecordEnvelope
from knowledge_graph.ontology import normalize_name


class CatalogField:
    """One stored product column the catalog can filter or group by."""

    def __init__(self, key: str, label: str, aliases: tuple[str, ...], *, attr: str | None, column: str | None):
        self.key = key
        self.label = label
        self.aliases = aliases
        self.attr = attr
        self.column = column


def _field(key: str, label: str, aliases: tuple[str, ...], *, attr: str | None = None, column: str | None = None) -> CatalogField:
    return CatalogField(key, label, aliases, attr=key if attr is None else attr, column=column)


# Every column kept on the product record or in source_columns. Fabric and composition
# are asked as material because that is where the catalog stores the composition text.
CATALOG_FIELDS: tuple[CatalogField, ...] = (
    _field("product_id", "Code", ("code", "product id", "product code"), column="Code"),
    _field("product_name", "Product name", ("product name",)),
    _field("style_reference", "Style reference", ("style", "style reference"), column="Style Reference"),
    _field("category", "Category", ("category", "categories"), column="Category"),
    _field("subcategory", "PLM sub category", ("subcategory", "sub category", "plm sub category"), column="PLM Sub Category"),
    _field("plm_main_category", "PLM main category", ("main category", "plm main category"), column="PLM Main Category"),
    _field("plm_sub_sub_category", "PLM sub sub category", ("sub sub category", "plm sub sub category"), column="PLM Sub Sub Category"),
    _field("garment_type", "Garment type", ("garment", "garments", "garment type"), column="Garment Type"),
    _field("material", "Material", ("material", "materials", "composition", "fabric", "fabrics"), column="Composition"),
    _field("color", "Color", ("color", "colors", "colour", "colours"), column="color"),
    _field("gender", "Gender", ("gender", "genders"), column="Gender"),
    _field("brand", "Brand", ("brand", "brands"), column="Brand"),
    _field("buyer", "Buyer", ("buyer", "buyers", "customer", "customers"), column="Buyer"),
    _field("season", "Season", ("season", "seasons", "season year", "year", "years"), column="Season Year"),
    _field("department", "Department", ("department", "departments"), column="Department"),
    _field("designer", "Designer", ("designer", "designers"), column="Designer"),
    _field("gauge", "Gauge", ("gauge",), column="Gauge"),
    _field("created_by", "Created by", ("created by", "creator"), column="Created by"),
    _field("unit", "Unit", ("unit", "units"), attr=None, column="Unit"),
    _field("sample_collection_type", "Sample collection type", ("sample collection", "sample collection type"), attr=None, column="Sample Collection Type"),
    _field("shipment_quantity", "Shipment quantity", ("shipment quantity", "shipment qty"), attr=None, column="Shipment Quantity"),
    _field("shipment_date", "Shipment date", ("shipment date",), attr=None, column="Shipment Date"),
    _field("design_for", "Design for", ("design for",), attr=None, column="Design For"),
    _field("knit_machine_type", "Knit machine type", ("knit machine", "knit machine type", "machine type"), attr=None, column="Knit Machine Type"),
    _field("knitting_time", "Knitting time", ("knitting time",), attr=None, column="Knitting Time"),
    _field("design_feature", "Design feature", ("design feature",), attr=None, column="Design Feature"),
    _field("merchandiser", "Merchandiser", ("merchandiser", "merchandisers"), attr=None, column="Merchandiser"),
    _field("shopping_department", "Shopping department", ("shopping department",), attr=None, column="Shopping Department"),
    _field("customer_style_reference", "Customer style reference", ("customer style", "customer style reference"), attr=None, column="Customer Style Reference"),
    _field("style_feature", "Style feature", ("style feature",), attr=None, column="Style Feature"),
    _field("weight", "Weight", ("weight",), attr=None, column="Weight"),
    _field("gsm", "GSM", ("gsm",), attr=None, column="GSM"),
    _field("cuttable_width", "Cuttable width", ("cuttable width", "width"), attr=None, column="Cuttable Width"),
    _field("sample_making_date", "Sample making date", ("sample making date",), attr=None, column="Sample Making Date"),
    _field("current_stock", "Current stock", ("current stock", "stock"), attr=None, column="Current Stock"),
    _field("current_location", "Current location", ("current location", "location"), attr=None, column="Current Location"),
    _field("purchased_by", "Purchased by", ("purchased by",), attr=None, column="Purchased By"),
    _field("created_at", "Created at", ("created at",), attr=None, column="Created At"),
    _field("updated_at", "Updated at", ("updated at",), attr="source_updated_at", column="Updated At"),
    _field(
        "ai_event_tags",
        "AI event tags",
        ("ai event tags", "event tags", "event tag"),
        attr="ai_event_tags",
        column="AI Event Tags",
    ),
    _field(
        "ai_construction_tags",
        "AI construction tags",
        ("ai construction tags", "construction tags", "construction tag"),
        attr="ai_construction_tags",
        column="AI Construction Tags",
    ),
    _field(
        "ai_material_tags",
        "AI material tags",
        ("ai material tags", "material tags", "material tag"),
        attr="ai_material_tags",
        column="AI Material Tags",
    ),
    _field(
        "ai_finish_tags",
        "AI finish tags",
        ("ai finish tags", "finish tags", "finish tag"),
        attr="ai_finish_tags",
        column="AI Finish Tags",
    ),
    _field(
        "ai_special_tags",
        "AI special tags",
        ("ai special tags", "special tags", "special tag"),
        attr="ai_special_tags",
        column="AI Special Tags",
    ),
)
FIELD_BY_KEY = {field.key: field for field in CATALOG_FIELDS}


def field_text(product: dict, key: str) -> str:
    """Read one catalog column from the product attribute or the original source column."""

    spec = FIELD_BY_KEY.get(key)
    if spec is None:
        return str(product.get(key) or "")
    if spec.attr:
        value = product.get(spec.attr)
        if isinstance(value, list):
            return ", ".join(str(item) for item in value if str(item).strip())
        if value not in (None, ""):
            return str(value)
    if spec.column:
        return str((product.get("source_columns") or {}).get(spec.column) or "")
    return ""


_TYPE_FIELDS = ("category", "subcategory", "garment_type", "plm_main_category", "plm_sub_sub_category")
_COLOR_CHARACTERS = {
    "red": "红",
    "blue": "蓝",
    "green": "绿",
    "black": "黑",
    "white": "白",
    "yellow": "黄",
    "pink": "粉",
    "purple": "紫",
    "grey": "灰",
    "gray": "灰",
    "brown": "棕",
    "orange": "橙",
}


class FileProductRepository:
    def __init__(self, envelopes: list[RecordEnvelope]):
        self.envelopes = [
            envelope
            for envelope in envelopes
            if envelope.source_type == "own_products" and envelope.lifecycle == "active"
        ]

    def _match(self, envelope: RecordEnvelope, **filters: str | None) -> bool:
        product = envelope.metadata.get("product", {})
        text = normalize_name(" ".join(field_text(product, field.key) for field in CATALOG_FIELDS))
        words = set(text.split())
        query = filters.get("q")
        if query:
            tokens = [token for token in normalize_name(query).split() if token]
            if tokens and not all(token in words for token in tokens):
                return False
        color = filters.get("color")
        if color and not all(_color_hit(field_text(product, "color"), token) for token in normalize_name(color).split()):
            return False
        garment = filters.get("garment")
        if garment and not all(_type_hit(product, token) for token in normalize_name(garment).split()):
            return False
        if not _season_hit(product, filters.get("season_name"), filters.get("season_code")):
            return False
        for field in CATALOG_FIELDS:
            if field.key == "color":
                continue
            expected = filters.get(field.key)
            if not expected:
                continue
            # Sweater is often Category while Jumper is Garment Type — treat type
            # columns as one family so model plans that pick garment_type still hit.
            if field.key in _TYPE_FIELDS:
                if not all(_type_hit(product, token) for token in normalize_name(expected).split()):
                    return False
            elif not _words_hit(field_text(product, field.key), expected):
                return False
        return True

    def prepare_filters(self, principal: Principal, filters: dict) -> dict:
        """Resolve upcoming season to the latest style-reference code, such as AW26."""

        prepared = {key: value for key, value in filters.items() if value not in (None, "")}
        if prepared.get("upcoming") and not prepared.get("season_code"):
            if prepared.get("season_name"):
                prepared["season_code"] = self._latest_season_code(principal, prepared)
            else:
                code = self._latest_any_season_code(principal, prepared)
                if code:
                    prepared["season_code"] = code
                    prepared["season_name"] = "winter" if code.upper().startswith(("AW", "FW")) else "summer"
        prepared.pop("upcoming", None)
        return prepared

    def search(self, principal: Principal, *, limit: int = 20, offset: int = 0, **filters: str | None):
        filters = self.prepare_filters(principal, filters)
        matched = [
            envelope
            for envelope in self.envelopes
            if allows(principal, envelope.classification) and self._match(envelope, **filters)
        ]
        matched.sort(key=lambda envelope: envelope.record_id)
        return matched[offset : offset + limit], len(matched)

    def count(self, principal: Principal, **filters: str | None) -> int:
        _page, total = self.search(principal, limit=1, offset=0, **filters)
        return total

    def count_by(self, principal: Principal, field: str, **filters: str | None) -> list[tuple[str, int]]:
        """Count visible matches grouped by one product field."""

        filters = self.prepare_filters(principal, filters)
        totals: dict[str, list] = {}
        for envelope in self.envelopes:
            if not allows(principal, envelope.classification) or not self._match(envelope, **filters):
                continue
            raw = field_text(envelope.metadata.get("product", {}), field).strip()
            key = raw.casefold() if raw else ""
            bucket = totals.get(key)
            if bucket is None:
                totals[key] = [raw or "unspecified", 1]
            else:
                bucket[1] += 1
        return sorted((label, count) for label, count in totals.values())

    def profile(self, principal: Principal, fields: tuple[str, ...] | list[str], **filters: str | None) -> tuple[int, dict[str, list[tuple[str, int]]]]:
        """Count one pass of matching products, with the most common value of each field first."""

        filters = self.prepare_filters(principal, filters)
        totals = {field: {} for field in fields}
        matched = 0
        for envelope in self.envelopes:
            if not allows(principal, envelope.classification) or not self._match(envelope, **filters):
                continue
            matched += 1
            product = envelope.metadata.get("product", {})
            for field in fields:
                raw = field_text(product, field).strip()
                if not raw or raw.casefold() in {"", "unspecified", "null", "none", "n/a", "na", "-"}:
                    continue
                key = raw.casefold()
                bucket = totals[field].get(key)
                if bucket is None:
                    totals[field][key] = [raw, 1]
                else:
                    bucket[1] += 1
        ranked = {
            field: sorted(((label, count) for label, count in buckets.values()), key=lambda item: (-item[1], item[0].casefold()))
            for field, buckets in totals.items()
        }
        return matched, ranked

    def _latest_season_code(self, principal: Principal, filters: dict) -> str | None:
        name = str(filters.get("season_name") or "")
        narrowed = {key: value for key, value in filters.items() if key not in {"season_name", "season_code", "upcoming"}}
        counts: dict[str, int] = {}
        for envelope in self.envelopes:
            if not allows(principal, envelope.classification) or not self._match(envelope, **narrowed):
                continue
            raw = field_text(envelope.metadata.get("product", {}), "style_reference")
            for code in season_codes(raw, name):
                counts[code] = counts.get(code, 0) + 1
        if not counts:
            return None
        return max(counts, key=lambda code: (int(code[2:]), counts[code]))

    def _latest_any_season_code(self, principal: Principal, filters: dict) -> str | None:
        """Latest AW/SS code for the buyer when the user said upcoming season without winter/summer."""

        narrowed = {key: value for key, value in filters.items() if key not in {"season_name", "season_code", "upcoming"}}
        counts: dict[str, int] = {}
        for envelope in self.envelopes:
            if not allows(principal, envelope.classification) or not self._match(envelope, **narrowed):
                continue
            raw = field_text(envelope.metadata.get("product", {}), "style_reference")
            for name in ("winter", "summer"):
                for code in season_codes(raw, name):
                    counts[code] = counts.get(code, 0) + 1
        if not counts:
            return None
        return max(counts, key=lambda code: (int(code[2:]), counts[code]))

    def season_code_counts(self, principal: Principal, season_name: str, **filters: str | None) -> list[tuple[str, int]]:
        """Count plausible season codes, such as AW26, inside an already filtered buyer set."""

        narrowed = {key: value for key, value in filters.items() if key not in {"season_name", "season_code", "upcoming"} and value not in (None, "")}
        counts: dict[str, int] = {}
        for envelope in self.envelopes:
            if not allows(principal, envelope.classification) or not self._match(envelope, **narrowed):
                continue
            raw = field_text(envelope.metadata.get("product", {}), "style_reference")
            seen = season_codes(raw, season_name)
            for code in seen[:1]:
                counts[code] = counts.get(code, 0) + 1
        return sorted(counts.items(), key=lambda item: (-int(item[0][2:]), -item[1], item[0]))


def season_codes(raw: str, name: str) -> list[str]:
    """Plausible fashion-season codes. AW80 is not a season year; AW26 is."""

    pattern = r"(AW|FW)(\d{2})" if name == "winter" else r"(SS)(\d{2})"
    found = []
    for prefix, year in re.findall(pattern, raw.upper()):
        if not 18 <= int(year) <= 35:
            continue
        code = f"{prefix}{year}"
        if code not in found:
            found.append(code)
    return found


def _season_hit(product: dict, name: str | None, code: str | None) -> bool:
    if not name and not code:
        return True
    raw = field_text(product, "style_reference")
    if code:
        return code in raw.upper()
    if name in {"winter", "spring", "summer"}:
        return bool(season_codes(raw, name))
    return True


def _words_hit(raw: str, expected: str) -> bool:
    words = set(normalize_name(raw).split())
    tokens = [token for token in normalize_name(expected).split() if token]
    return bool(tokens) and all(_term_hit(words, token) for token in tokens)


def _term_hit(words: set[str], term: str) -> bool:
    singular = term[:-1] if term.endswith("s") and not term.endswith("ss") and len(term) > 3 else term
    return term in words or singular in words


def _color_hit(raw_color: str, term: str) -> bool:
    if _term_hit(set(normalize_name(raw_color).split()), term):
        return True
    singular = term[:-1] if term.endswith("s") and not term.endswith("ss") and len(term) > 3 else term
    character = _COLOR_CHARACTERS.get(singular)
    return bool(character and character in raw_color)


def _type_hit(product: dict, term: str) -> bool:
    words: set[str] = set()
    for field in _TYPE_FIELDS:
        words.update(normalize_name(str(product.get(field, ""))).split())
    return _term_hit(words, term)


class SqlProductRepository:
    """Interface for a future authorized connector. This build never opens a live DSN."""

    def __init__(self, enabled: bool, dsn: str):
        self.enabled = enabled
        self.dsn = dsn

    def count(self, principal: Principal, **filters: str | None) -> int:
        if not self.enabled or not self.dsn:
            raise ProductBackendUnavailable(
                "Authorized SQL/API product connector is not configured. "
                "Live PLMEX analytics will not be estimated."
            )
        raise ProductBackendUnavailable(
            "SQL connector settings are present, but this build refuses to open a live database connection."
        )
