"""Map product_change_events rows → PriceChangeEvent / PriceSnapshot / AvailabilitySnapshot.

Schema contract (confirmed against fashion_brain_live):
  before_json/after_json hold scalars: price (number), currency, availability,
  in_stock_count, out_stock_count, price_min/max, prices_seen.
  master_id is brand-prefixed (e.g. primark:991…). Product id stays ecom:{brand}:{master_id}.
"""

from __future__ import annotations

from typing import Any

from knowledge_graph.adapters.ecom_master import parse_price_num, product_id
from knowledge_graph.ontology import normalize_name


_PRICE_FIELDS = frozenset({"price", "original_price", "price_min", "price_max", "currency", "prices_seen"})
_AVAIL_FIELDS = frozenset({"availability", "in_stock_count", "out_stock_count"})


def _iso(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        text = value.isoformat()
        return text.replace("+00:00", "Z") if text.endswith("+00:00") else text
    return str(value).strip()


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _field_set(fields_changed: Any) -> set[str]:
    return {str(x).strip() for x in _as_list(fields_changed) if str(x).strip()}


def _price_num(blob: dict[str, Any]) -> float | None:
    if "price" not in blob:
        return None
    return parse_price_num(blob.get("price"))


def _currency(blob: dict[str, Any]) -> str:
    raw = blob.get("currency")
    return str(raw).strip().upper() if raw not in (None, "") else ""


def _touched_price(change_type: str, fields: set[str], before: dict, after: dict) -> bool:
    if change_type == "price":
        return True
    if fields & _PRICE_FIELDS:
        return True
    return _price_num(before) is not None or _price_num(after) is not None


def _touched_availability(change_type: str, fields: set[str], before: dict, after: dict) -> bool:
    if change_type == "availability":
        return True
    if fields & _AVAIL_FIELDS:
        return True
    return any(k in before or k in after for k in _AVAIL_FIELDS)


def build_event_graph(row: dict[str, Any]) -> dict[str, Any] | None:
    """Return nodes/edges to upsert for one change-event row, or None if nothing graph-worthy."""

    brand = str(row.get("brand") or "").strip()
    master_id = str(row.get("master_id") or "").strip()
    if not brand or not master_id:
        return None

    event_id = row.get("id")
    if event_id is None:
        return None

    change_type = str(row.get("change_type") or "").strip().lower()
    fields = _field_set(row.get("fields_changed"))
    before = _as_dict(row.get("before_json"))
    after = _as_dict(row.get("after_json"))
    detected_at = _iso(row.get("detected_at") or row.get("scraped_at"))
    scraped_at = _iso(row.get("scraped_at") or row.get("detected_at"))
    scrape_job_id = str(row.get("scrape_job_id") or "").strip()
    pid = product_id(brand, master_id)

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    product_patch: dict[str, Any] = {}

    if _touched_price(change_type, fields, before, after):
        old_num = _price_num(before)
        new_num = _price_num(after)
        old_cur = _currency(before)
        new_cur = _currency(after) or old_cur
        currency_changed = bool(old_cur and new_cur and old_cur != new_cur)

        if old_num is not None and new_num is not None:
            if currency_changed:
                direction = "currency_change"
                delta = None
                delta_pct = None
            else:
                delta = round(new_num - old_num, 6)
                delta_pct = round((delta / old_num) * 100.0, 4) if old_num != 0 else None
                if delta > 0:
                    direction = "up"
                elif delta < 0:
                    direction = "down"
                else:
                    direction = "unchanged"

            chg_id = f"price_chg:{event_id}"
            nodes.append(
                {
                    "id": chg_id,
                    "labels": ["Entity", "PriceChangeEvent"],
                    "properties": {
                        "id": chg_id,
                        "name": f"price {direction} {old_num}->{new_num}",
                        "pg_event_id": int(event_id) if str(event_id).isdigit() else event_id,
                        "brand": brand,
                        "master_id": master_id,
                        "old_price": before.get("price"),
                        "old_price_num": old_num,
                        "new_price": after.get("price"),
                        "new_price_num": new_num,
                        "delta": delta,
                        "delta_pct": delta_pct,
                        "direction": direction,
                        "currency": new_cur or old_cur,
                        "old_currency": old_cur or None,
                        "new_currency": new_cur or None,
                        "currency_changed": currency_changed,
                        "detected_at": detected_at,
                        "scraped_at": scraped_at,
                        "change_type": change_type,
                        "fields_changed": sorted(fields),
                        "scrape_job_id": scrape_job_id or None,
                        "source": "product_change_events",
                    },
                }
            )
            edges.append({"rel_type": "HAS_PRICE_CHANGE", "start_id": pid, "end_id": chg_id, "field_ref": "product_change_events"})
            if detected_at:
                product_patch["last_price_change_at"] = detected_at
                product_patch["last_price_direction"] = direction

        if new_num is not None:
            snap_id = f"price_snap:{event_id}:after"
            nodes.append(
                {
                    "id": snap_id,
                    "labels": ["Entity", "PriceSnapshot"],
                    "properties": {
                        "id": snap_id,
                        "name": f"price {new_num}",
                        "pg_event_id": int(event_id) if str(event_id).isdigit() else event_id,
                        "brand": brand,
                        "master_id": master_id,
                        "price": after.get("price"),
                        "price_num": new_num,
                        "original_price": after.get("original_price"),
                        "original_price_num": parse_price_num(after.get("original_price")),
                        "currency": new_cur or None,
                        "observed_at": detected_at or scraped_at,
                        "source": "product_change_events",
                        "scrape_job_id": scrape_job_id or None,
                        "side": "after",
                    },
                }
            )
            edges.append({"rel_type": "HAS_PRICE_SNAPSHOT", "start_id": pid, "end_id": snap_id, "field_ref": "product_change_events"})
            if new_cur:
                cur_id = f"currency:{normalize_name(new_cur).replace(' ', '-')}"
                nodes.append(
                    {
                        "id": cur_id,
                        "labels": ["Entity", "SourceEntity", "Currency"],
                        "properties": {
                            "id": cur_id,
                            "name": new_cur.lower(),
                            "normalized_name": normalize_name(new_cur),
                            "entity_type": "Currency",
                            "code": new_cur,
                        },
                    }
                )
                edges.append({"rel_type": "SNAPSHOT_IN_CURRENCY", "start_id": snap_id, "end_id": cur_id, "field_ref": "currency"})

    if _touched_availability(change_type, fields, before, after):
        status = str(after.get("availability") or before.get("availability") or "").strip()
        if status or after.get("in_stock_count") is not None or before.get("in_stock_count") is not None:
            snap_id = f"avail_snap:{event_id}"
            nodes.append(
                {
                    "id": snap_id,
                    "labels": ["Entity", "AvailabilitySnapshot"],
                    "properties": {
                        "id": snap_id,
                        "name": status or "availability",
                        "pg_event_id": int(event_id) if str(event_id).isdigit() else event_id,
                        "brand": brand,
                        "master_id": master_id,
                        "status": status or None,
                        "in_stock_count": after.get("in_stock_count", before.get("in_stock_count")),
                        "out_stock_count": after.get("out_stock_count", before.get("out_stock_count")),
                        "observed_at": detected_at or scraped_at,
                        "source": "product_change_events",
                        "scrape_job_id": scrape_job_id or None,
                        "change_type": change_type,
                    },
                }
            )
            edges.append(
                {
                    "rel_type": "HAS_AVAILABILITY_SNAPSHOT",
                    "start_id": pid,
                    "end_id": snap_id,
                    "field_ref": "product_change_events",
                }
            )
            if status:
                avail_id = f"availability:{normalize_name(status).replace(' ', '-')}"
                nn = normalize_name(status)
                nodes.append(
                    {
                        "id": avail_id,
                        "labels": ["Entity", "SourceEntity", "Availability"],
                        "properties": {
                            "id": avail_id,
                            "name": nn or status,
                            "normalized_name": nn,
                            "entity_type": "Availability",
                            "status": status,
                        },
                    }
                )
                edges.append({"rel_type": "SNAPSHOT_STATUS", "start_id": snap_id, "end_id": avail_id, "field_ref": "availability"})

    if not nodes:
        return None

    return {
        "product_id": pid,
        "brand": brand,
        "master_id": master_id,
        "pg_event_id": event_id,
        "nodes": nodes,
        "edges": edges,
        "product_patch": product_patch,
    }
