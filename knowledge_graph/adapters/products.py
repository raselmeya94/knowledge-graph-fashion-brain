"""PLMEX-owned synthetic product adapter. JSON and CSV now; SQL/API stays unconfigured."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from knowledge_graph.adapters.base import DiscoveredRecord, DiscoveryResult, SourceAdapter, register_adapter, synthetic_label_errors
from knowledge_graph.errors import ProductBackendUnavailable
from knowledge_graph.hashing import canonical_json_hash, sha256_bytes
from knowledge_graph.ids import register_source_identity
from knowledge_graph.models import ExtractedEntity, ExtractedRel, RecordEnvelope
from knowledge_graph.ontology import Ontology, validate_relationship
from knowledge_graph.registry import load_manifest

register_source_identity("own_products", "plmex", "product", "Product")

# Headers from data/raw/own_products/products_data.csv (includes AI * Tags).
RAW_COLUMNS = [
    "Created by",
    "Code",
    "Composition",
    "Category",
    "PLM Main Category",
    "PLM Sub Category",
    "PLM Sub Sub Category",
    "Brand",
    "Unit",
    "Gauge",
    "Gender",
    "Sample Collection Type",
    "Shipment Quantity",
    "Shipment Date",
    "Garment Type",
    "Design For",
    "Knit Machine Type",
    "Knitting Time",
    "Design Feature",
    "Merchandiser",
    "Department",
    "Shopping Department",
    "Season Year",
    "Designer",
    "Style Reference",
    "Customer Style Reference",
    "Style Feature",
    "color",
    "Weight",
    "GSM",
    "Cuttable Width",
    "Sample Making Date",
    "Current Stock",
    "Current Location",
    "Purchased By",
    "Created At",
    "Updated At",
    "Buyer",
    "AI Event Tags",
    "AI Construction Tags",
    "AI Material Tags",
    "AI Finish Tags",
    "AI Special Tags",
]
FILE_RECORD_ID = "*"
_COMPOSITION_PART = re.compile(
    r"(\d+(?:\.\d+)?)\s*%\s*(.+?)(?=\s*(?:[/+,&、]\s*)?\d+(?:\.\d+)?\s*%|$)",
    re.IGNORECASE,
)
_BLANK = {"", "#n/a", "n/a", "na", "null", "none", "-", "--"}
# CSV AI tag cells: "polo shirt, menswear, regular fit" or pipe-separated.
_AI_TAG_COLUMNS = (
    ("AI Event Tags", "ai_event_tags", "EventTag", "HAS_EVENT_TAG"),
    ("AI Construction Tags", "ai_construction_tags", "ConstructionTag", "HAS_CONSTRUCTION_TAG"),
    ("AI Material Tags", "ai_material_tags", "AiMaterialTag", "HAS_AI_MATERIAL_TAG"),
    ("AI Finish Tags", "ai_finish_tags", "FinishTag", "HAS_FINISH_TAG"),
    ("AI Special Tags", "ai_special_tags", "SpecialTag", "HAS_SPECIAL_TAG"),
)


def _split_multi(value: object) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [part.strip() for part in str(value).split("|") if part.strip()]


def _split_tags(value: object) -> list[str]:
    """Split AI tag cells on commas or pipes; keep multi-word tags intact."""

    if value is None or value == "":
        return []
    if isinstance(value, list):
        parts = [str(item).strip() for item in value]
    else:
        parts = re.split(r"[,|]", str(value))
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        tag = _present(part)
        if not tag:
            continue
        key = tag.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(tag)
    return out


def _cell(raw: dict, *names: str) -> str:
    for name in names:
        value = raw.get(name)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _present(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.casefold() in _BLANK else text


def _number(value: object) -> int | float | None:
    text = _present(value).replace(",", "")
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return int(number) if number.is_integer() else number


def parse_composition(text: str) -> list[tuple[str, str]]:
    """Split a composition cell into (percent, material name) pairs."""

    parts = []
    for percent, name in _COMPOSITION_PART.findall(text or ""):
        cleaned = re.sub(r"\s+", " ", name).strip(" ,;/+-&、")
        if cleaned:
            parts.append((percent, cleaned))
    if parts:
        return parts
    cleaned = _present(text)
    return [("", cleaned)] if cleaned else []


def _locations(text: str) -> list[tuple[str, int | float | None]]:
    found = []
    for line in re.split(r"[\r\n]+", text or ""):
        match = re.search(r"\((\d+(?:\.\d+)?)\)\s*$", line.strip())
        quantity = _number(match.group(1)) if match else None
        name = _present(line[: match.start()] if match else line).strip(" ›")
        if name:
            found.append((name, quantity))
    return found


def normalize_product(raw: dict) -> dict:
    code = _cell(raw, "Code", "product_id")
    style = _cell(raw, "Style Reference", "product_name")
    composition = _cell(raw, "Composition", "composition", "material")
    synthetic = raw.get("synthetic")
    if isinstance(synthetic, str):
        synthetic = synthetic.strip().lower() in {"true", "yes", "1"}
    elif synthetic is None:
        synthetic = False
    classification = _cell(raw, "classification") or "internal"
    columns = {name: _cell(raw, name) for name in RAW_COLUMNS}
    if code and not columns["Code"]:
        columns["Code"] = code
    if composition and not columns["Composition"]:
        columns["Composition"] = composition
    return {
        "product_id": code,
        "product_name": style or code,
        "category": _cell(raw, "Category", "category"),
        "subcategory": _cell(raw, "PLM Sub Category", "subcategory"),
        "season": _cell(raw, "Season Year", "season"),
        "material": composition,
        "material_code": None,
        "fabric": None,
        "composition": composition,
        "composition_parts": parse_composition(composition),
        "buyer": _cell(raw, "Buyer"),
        "brand": _cell(raw, "Brand"),
        "garment_type": _cell(raw, "Garment Type"),
        "color": _cell(raw, "color"),
        "gender": _cell(raw, "Gender"),
        "gauge": _cell(raw, "Gauge"),
        "designer": _cell(raw, "Designer"),
        "created_by": _cell(raw, "Created by"),
        "department": _cell(raw, "Department"),
        "plm_main_category": _cell(raw, "PLM Main Category"),
        "plm_sub_sub_category": _cell(raw, "PLM Sub Sub Category"),
        "style_reference": style,
        "customer_style_reference": _cell(raw, "Customer Style Reference"),
        "design_feature": _cell(raw, "Design Feature"),
        "style_feature": _cell(raw, "Style Feature"),
        "design_for": _cell(raw, "Design For"),
        "knit_machine": _cell(raw, "Knit Machine Type"),
        "sample_type": _cell(raw, "Sample Collection Type"),
        "merchandiser": _cell(raw, "Merchandiser"),
        "unit": _cell(raw, "Unit"),
        "knitting_time": _cell(raw, "Knitting Time"),
        "weight": _cell(raw, "Weight"),
        "gsm": _cell(raw, "GSM"),
        "cuttable_width": _cell(raw, "Cuttable Width"),
        "shipment_quantity": _cell(raw, "Shipment Quantity"),
        "shipment_date": _cell(raw, "Shipment Date"),
        "sample_making_date": _cell(raw, "Sample Making Date"),
        "current_stock": _cell(raw, "Current Stock"),
        "current_location": _cell(raw, "Current Location"),
        "created_at": _cell(raw, "Created At"),
        "source_updated_at": _cell(raw, "Updated At", "source_updated_at") or None,
        "classification": classification,
        "source_permissions": _cell(raw, "source_permissions") or "plm-internal",
        "synthetic": bool(synthetic),
        "disclaimer": _cell(raw, "disclaimer"),
        "source_columns": columns,
        **{
            attr: _split_tags(columns.get(column) or _cell(raw, column))
            for column, attr, _entity, _rel in _AI_TAG_COLUMNS
        },
    }


def load_products_from_file(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data["products"] if isinstance(data, dict) else data
        return [normalize_product(row) for row in rows]
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [normalize_product(row) for row in csv.DictReader(handle)]


def product_hash(product: dict) -> str:
    return canonical_json_hash(product)


class SqlProductConnector:
    """Authorized SQL/API connector placeholder. It never opens a live connection in this build."""

    def __init__(self, enabled: bool, dsn: str):
        self.enabled = enabled
        self.dsn = dsn

    def connect(self):
        if not self.enabled or not self.dsn:
            raise ProductBackendUnavailable(
                "Authorized SQL/API product connector is not configured. "
                "Live PLMEX counts will not be estimated from synthetic data."
            )
        raise ProductBackendUnavailable(
            "SQL connector flags are set, but this build refuses to open a live database connection."
        )


class OwnProductsAdapter(SourceAdapter):
    source_type = "own_products"

    def __init__(self, source, source_root):
        super().__init__(source, source_root)
        self.ontology: Ontology | None = None

    def bind_runtime(self, ontology, extraction_mode: str, model_provider, timeout: float) -> None:
        self.ontology = ontology

    def discover(self) -> DiscoveryResult:
        result = DiscoveryResult()
        manifest = load_manifest(self.source_root / "manifest.json")
        result.manifest_ids = [entry.record_id for entry in manifest.records]
        cache: dict[str, list[dict]] = {}
        listed: set[str] = set()
        for entry in manifest.records:
            path = self.source_root / entry.path
            listed.add(entry.path.replace("\\", "/"))
            if not path.exists():
                result.errors.append(f"Missing product file {entry.path}.")
                continue
            key = str(path.resolve())
            if key not in cache:
                cache[key] = load_products_from_file(path)
            if entry.record_id == FILE_RECORD_ID:
                file_hash = sha256_bytes(path.read_bytes())
                if entry.hash_method == "file" and file_hash != entry.content_hash:
                    result.errors.append(f"Content hash does not match the manifest for {entry.path}. The raw file was not modified.")
                    continue
                for product in cache[key]:
                    if not product["product_id"]:
                        result.errors.append(f"A row in {entry.path} is missing Code.")
                        continue
                    result.manifest_ids.append(product["product_id"])
                    result.records.append(
                        DiscoveredRecord(
                            source_id=self.source.source_id,
                            record_id=product["product_id"],
                            path=path,
                            relative_path=entry.path.replace("\\", "/"),
                            format=entry.format,
                            hash_method="canonical_product",
                            manifest_hash=product_hash(product),
                            source_timestamp=entry.source_timestamp or product.get("source_updated_at"),
                            synthetic_declared=entry.synthetic,
                            raw_bytes=b"",
                            parsed={"product": product},
                        )
                    )
                continue
            product = next((item for item in cache[key] if item["product_id"] == entry.record_id), None)
            if product is None:
                result.errors.append(f"Product {entry.record_id} was not found in {entry.path}.")
                continue
            result.records.append(
                DiscoveredRecord(
                    source_id=self.source.source_id,
                    record_id=entry.record_id,
                    path=path,
                    relative_path=entry.path.replace("\\", "/"),
                    format=entry.format,
                    hash_method=entry.hash_method,
                    manifest_hash=entry.content_hash,
                    source_timestamp=entry.source_timestamp or product.get("source_updated_at"),
                    synthetic_declared=entry.synthetic,
                    raw_bytes=path.read_bytes(),
                    parsed={"product": product},
                )
            )
        for path in self.source_root.iterdir():
            if path.name == "manifest.json" or not path.is_file():
                continue
            if path.name not in listed:
                result.warnings.append(f"Unregistered file was left untouched: {path.name}.")
        return result

    def validate(self, record: DiscoveredRecord) -> list[str]:
        errors: list[str] = []
        product = record.parsed["product"]
        if record.hash_method != "canonical_product":
            errors.append("Product records must use canonical_product hashes.")
        if product_hash(product) != record.manifest_hash:
            errors.append("Content hash does not match the manifest. The raw file was not modified.")
        if not product.get("product_id"):
            errors.append("Missing Code.")
        if not product.get("product_name"):
            errors.append("Missing product name.")
        if product.get("classification") not in {"public", "internal", "confidential"}:
            errors.append("classification must be public, internal, or confidential.")
        if product["product_id"] != record.record_id:
            errors.append("Code does not match the manifest record_id.")
        if product["synthetic"] not in {True, False}:
            errors.append("synthetic flag is required.")
        errors.extend(synthetic_label_errors(product["product_name"], bool(product["synthetic"]), product["disclaimer"]))
        return errors

    def normalize(self, record: DiscoveredRecord, ingested_at: str) -> RecordEnvelope:
        product = record.parsed["product"]
        content = (
            f"{product['product_name']}. Category {product['category']}. "
            f"Material {product['material']}. Composition {product['composition']}."
        )
        return RecordEnvelope(
            source_id=record.source_id,
            source_type=self.source_type,
            record_id=record.record_id,
            title=product["product_name"],
            content=content,
            content_hash=product_hash(product),
            synthetic=bool(product["synthetic"]),
            classification=product["classification"],
            source_path=record.relative_path,
            source_timestamp=product.get("source_updated_at"),
            ingested_at=ingested_at,
            metadata={"product": product, "format": record.format, "source_permissions": product["source_permissions"]},
        )

    def extract(self, envelope: RecordEnvelope) -> tuple[RecordEnvelope, list[str]]:
        if self.ontology is None:
            raise RuntimeError("Product adapter runtime was not bound.")
        product = envelope.metadata["product"]
        errors: list[str] = []
        entities = [
            ExtractedEntity(local_id="record", entity_type="Product", name=product["product_name"], field_ref="product_name")
        ]
        relationships: list[ExtractedRel] = []

        seen: set[str] = set()

        def ensure(entity_type: str, name: str | None, field_ref: str, properties: dict | None = None) -> str:
            cleaned = _present(name)
            if not cleaned:
                return ""
            local_id = f"{entity_type}:{cleaned}"
            if local_id not in seen:
                seen.add(local_id)
                entities.append(
                    ExtractedEntity(
                        local_id=local_id,
                        entity_type=entity_type,
                        name=cleaned,
                        evidence_text=cleaned,
                        field_ref=field_ref,
                        properties=dict(properties or {}),
                    )
                )
            return local_id

        def link(
            rel_type: str,
            start_local: str,
            end_local: str,
            field_ref: str,
            evidence: str,
            start_type: str = "Product",
            end_type: str = "",
            properties: dict | None = None,
        ) -> None:
            if not start_local or not end_local or start_local == end_local:
                return
            start_labels = ["Entity", "Record", "Product"] if start_type == "Product" else ["Entity", "SourceEntity", start_type]
            try:
                validate_relationship(
                    rel_type,
                    start_labels,
                    ["Entity", "SourceEntity", end_type],
                    self.ontology,
                )
            except Exception as exc:
                errors.append(str(exc))
                return
            relationships.append(
                ExtractedRel(
                    type=rel_type,
                    start_local_id=start_local,
                    end_local_id=end_local,
                    evidence_text=evidence,
                    field_ref=field_ref,
                    verification_status="observed",
                    properties=dict(properties or {}),
                )
            )

        def add(rel_type: str, entity_type: str, name: str | None, field_ref: str, properties: dict | None = None) -> None:
            local_id = ensure(entity_type, name, field_ref)
            if local_id:
                link(rel_type, "record", local_id, field_ref, _present(name) or "", end_type=entity_type, properties=properties)

        for percent, name in product.get("composition_parts") or []:
            amount = _number(percent)
            evidence = f"{percent}%{name}" if percent else name
            before = len(relationships)
            add(
                "USES_MATERIAL",
                "Material",
                name,
                "Composition",
                {"percentage": amount} if amount is not None else None,
            )
            if len(relationships) == before + 1:
                relationships[-1].evidence_text = evidence
        add("BELONGS_TO_CATEGORY", "GarmentCategory", product["category"], "Category")
        add("HAS_SUB_CATEGORY", "GarmentCategory", product.get("subcategory"), "PLM Sub Category")
        add("HAS_SUB_CATEGORY", "GarmentCategory", product.get("plm_sub_sub_category"), "PLM Sub Sub Category")
        add("HAS_GARMENT_TYPE", "GarmentCategory", product.get("garment_type"), "Garment Type")
        main = ensure("GarmentCategory", product.get("plm_main_category"), "PLM Main Category")
        sub = ensure("GarmentCategory", product.get("subcategory"), "PLM Sub Category")
        sub_sub = ensure("GarmentCategory", product.get("plm_sub_sub_category"), "PLM Sub Sub Category")
        link("PARENT_CATEGORY", sub, main, "PLM Main Category", _present(product.get("plm_main_category")), "GarmentCategory", "GarmentCategory")
        link("PARENT_CATEGORY", sub_sub, sub, "PLM Sub Category", _present(product.get("subcategory")), "GarmentCategory", "GarmentCategory")
        add("IN_SEASON", "Season", product["season"], "Season Year")
        add("ORDERED_BY", "Company", product.get("buyer"), "Buyer")
        add("BRANDED_AS", "Company", product.get("brand"), "Brand")
        add("HAS_COLOR", "Color", product.get("color"), "color")
        add("HAS_DESIGN_FEATURE", "DesignFeature", product.get("design_feature"), "Design Feature")
        add("HAS_STYLE_FEATURE", "StyleFeature", product.get("style_feature"), "Style Feature")
        add("DESIGNED_FOR", "DesignPurpose", product.get("design_for"), "Design For")
        add("USES_KNIT_MACHINE", "MachineType", product.get("knit_machine"), "Knit Machine Type")
        add("HAS_GAUGE", "Gauge", product.get("gauge"), "Gauge")
        add("HAS_SAMPLE_TYPE", "SampleType", product.get("sample_type"), "Sample Collection Type")
        add("BELONGS_TO_DEPARTMENT", "Department", product.get("department"), "Department")
        add("CREATED_BY", "Person", product.get("created_by"), "Created by")
        add("DESIGNED_BY", "Person", product.get("designer"), "Designer")
        add("MANAGED_BY", "Person", product.get("merchandiser"), "Merchandiser")
        add("HAS_STYLE_REFERENCE", "StyleReference", product.get("style_reference"), "Style Reference")
        add("HAS_CUSTOMER_STYLE_REFERENCE", "CustomerStyleReference", product.get("customer_style_reference"), "Customer Style Reference")
        for column, attr, entity_type, rel_type in _AI_TAG_COLUMNS:
            for index, tag in enumerate(product.get(attr) or []):
                local_id = ensure(entity_type, tag, column)
                if local_id:
                    link(
                        rel_type,
                        "record",
                        local_id,
                        f"{column}:{index}",
                        tag,
                        end_type=entity_type,
                    )
        for index, (place, quantity) in enumerate(_locations(product.get("current_location") or "")):
            local_id = ensure("Location", place, "Current Location")
            link(
                "STORED_AT",
                "record",
                local_id,
                f"Current Location:{index}",
                place,
                end_type="Location",
                properties={"quantity": quantity} if quantity is not None else None,
            )
        stock = _number(product.get("current_stock"))
        if stock is not None:
            local_id = ensure(
                "InventoryRecord",
                f"{product['product_id']} stock",
                "Current Stock",
                {"quantity": stock},
            )
            link("HAS_INVENTORY_RECORD", "record", local_id, "Current Stock", str(stock), end_type="InventoryRecord")
        shipped = _number(product.get("shipment_quantity"))
        shipped_on = _present(product.get("shipment_date"))
        if shipped is not None or shipped_on:
            shipment_props = {}
            if shipped is not None:
                shipment_props["quantity"] = shipped
            if shipped_on:
                shipment_props["shipment_date"] = shipped_on
            local_id = ensure("Shipment", f"{product['product_id']} shipment", "Shipment Quantity", shipment_props)
            link("HAS_SHIPMENT", "record", local_id, "Shipment Quantity", shipped_on or str(shipped or ""), end_type="Shipment")
        node_properties = {
            "code": product.get("product_id") or None,
            "composition": _present(product.get("composition")) or None,
            "gender": _present(product.get("gender")) or None,
            "unit": _present(product.get("unit")) or None,
            "knitting_time": _present(product.get("knitting_time")) or None,
            "weight": _number(product.get("weight")),
            "gsm": _number(product.get("gsm")),
            "cuttable_width": _number(product.get("cuttable_width")),
            "shipment_quantity": shipped,
            "shipment_date": shipped_on or None,
            "sample_making_date": _present(product.get("sample_making_date")) or None,
            "current_stock": stock,
            "created_at": _present(product.get("created_at")) or None,
            "updated_at": product.get("source_updated_at"),
            "season_year": _number(product.get("season")) if str(product.get("season") or "").isdigit() else None,
        }
        envelope.metadata["node_properties"] = {key: value for key, value in node_properties.items() if value is not None}
        envelope.entities = entities
        envelope.relationships = relationships
        return envelope, errors


register_adapter("own_products", OwnProductsAdapter)
