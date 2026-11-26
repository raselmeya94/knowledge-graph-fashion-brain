"""Strict metadata-grounded filter validation + clarification loop.

Code owns validation statuses and the execution gate. The LLM may propose filter
phrases; it must not invent catalog values or silently alter user filters.

Statuses:
  VALID       — exact / case-insensitive / registered-alias match, or DB confirms rows
  NOT_FOUND   — DB verification confirms zero matching rows for this value
  UNKNOWN     — metadata incomplete and DB could not confirm
  AMBIGUOUS   — multiple plausible matches; never auto-picked
  UNSUPPORTED — field not available on the selected source schema
"""

from __future__ import annotations

import logging
import re
from typing import Any

from knowledge_graph.ontology import normalize_name

from fashion_brain.agents.contracts import (
    ClarificationItem,
    UnderstandingPlan,
    ValidatedFilter,
)
from fashion_brain.planning.metadata_resolver import (
    CORE_METADATA_FIELDS,
    FREEFORM_FILTER_KEYS,
    MATCH_MIN,
    STRICT_METADATA_FIELDS,
    _sim,
    build_meta_index,
)

LOGGER = logging.getLogger(__name__)

_BLOCKING = frozenset({"NOT_FOUND", "UNKNOWN", "AMBIGUOUS", "UNSUPPORTED"})

# Registered value aliases (normalized key → preferred display / match target).
# Keep small and data-driven; do not grow into a question-specific rule book.
_VALUE_ALIASES: dict[str, tuple[str, ...]] = {
    "colour": ("color",),
    "colors": ("color",),
    "colours": ("color",),
    "navy blue": ("navy", "blue"),
    "dark blue": ("navy", "blue"),
    "womens": ("women", "woman"),
    "women's": ("women", "woman"),
    "mens": ("men", "man"),
    "men's": ("men", "man"),
    "in stock": ("instock", "available"),
    "instock": ("in stock", "available"),
    "out of stock": ("outofstock",),
    # Country of origin: ISO ↔ English (filters accept either form).
    "cn": ("china",),
    "china": ("cn",),
    "bd": ("bangladesh",),
    "bangladesh": ("bd",),
    "tr": ("turkey", "turkiye"),
    "turkey": ("tr", "turkiye"),
    "turkiye": ("tr", "turkey"),
    "kh": ("cambodia",),
    "cambodia": ("kh",),
    "ma": ("morocco",),
    "morocco": ("ma",),
    "pk": ("pakistan",),
    "pakistan": ("pk",),
    "vn": ("vietnam",),
    "vietnam": ("vn",),
    "in": ("india",),
    "india": ("in",),
    "gb": ("united kingdom", "uk"),
    "uk": ("united kingdom", "gb"),
    "united kingdom": ("gb", "uk"),
}

_REMOVE_RE = re.compile(
    r"(?i)\b(remove|drop|ignore|skip|without|clear|cancel)\b.*\b"
    r"(?P<field>brand|buyer|color|colour|material|category|garment|gender|season|"
    r"availability|family|section|department)\b"
    r"|\b(remove|drop|ignore|skip|without)\s+(?:the\s+)?(?P<value>[\w'’ -]{2,40})\b"
)


def _field_key(name: str) -> str:
    key = str(name or "").strip().casefold().replace(" ", "_")
    if key == "colour":
        return "color"
    return key


def _alias_norms(value: str) -> list[str]:
    """Normalized forms for ANY catalog value (case/plural/registered aliases)."""

    norm = normalize_name(value)
    out = [norm] if norm else []
    for alias in _VALUE_ALIASES.get(norm, ()):
        an = normalize_name(alias)
        if an and an not in out:
            out.append(an)
    # Universal singular/plural variants for every column value.
    try:
        from knowledge_graph.retrieval.kg_stats import _plural_forms

        for seed in list(out):
            for form in _plural_forms(seed):
                if form and form not in out:
                    out.append(form)
    except Exception:
        pass
    return out


def _index_values_for_field(index: dict[str, list[tuple[str, str]]], field: str) -> list[tuple[str, str]]:
    key = _field_key(field)
    vals = list(index.get(key) or [])
    if key == "color":
        vals = list(dict.fromkeys([*vals, *(index.get("colour") or [])]))
    elif key == "garment":
        vals = list(
            dict.fromkeys(
                [
                    *vals,
                    *(index.get("category") or []),
                    *(index.get("garment_type") or []),
                    *(index.get("subfamily") or []),
                ]
            )
        )
    elif key == "category":
        vals = list(dict.fromkeys([*vals, *(index.get("garment") or [])]))
    elif key == "section":
        # Audience values may still be indexed under gender.
        vals = list(dict.fromkeys([*vals, *(index.get("gender") or [])]))
    elif key == "gender":
        vals = list(dict.fromkeys([*vals, *(index.get("section") or [])]))
    elif key == "subfamily":
        vals = list(dict.fromkeys([*vals, *(index.get("garment_type") or [])]))
    elif key == "garment_type":
        vals = list(dict.fromkeys([*vals, *(index.get("subfamily") or [])]))
    elif key == "buyer":
        vals = list(dict.fromkeys([*vals, *(index.get("brand") or [])]))
    elif key == "brand":
        vals = list(dict.fromkeys([*vals, *(index.get("buyer") or [])]))
    return vals


def _field_supported(meta, field: str, source: str | None) -> bool | None:
    """True/False when schema coverage known; None when coverage is thin."""

    key = _field_key(field)
    if key in FREEFORM_FILTER_KEYS:
        return True
    graphs = list(getattr(meta, "field_graphs", {}).get(key) or [])
    if key == "color":
        graphs = list(dict.fromkeys([*graphs, *(getattr(meta, "field_graphs", {}).get("colour") or [])]))
    if not graphs and key not in (getattr(meta, "values", {}) or {}):
        # Thin overview — do not claim UNSUPPORTED without evidence.
        return None
    if source:
        src = str(source).strip()
        if graphs and src not in graphs:
            # Field exists on other KGs but not this source.
            cols = (getattr(meta, "columns", {}) or {}).get(src) or []
            colset = {str(c).casefold() for c in cols}
            if colset and key not in colset and (key != "color" or "colour" not in colset):
                return False
    return True if graphs or key in (getattr(meta, "values", {}) or {}) else None


def _token_in_label(label: str, token: str) -> bool:
    """True when ``token`` appears as a whole word inside ``label`` (white ∈ arctic white).

    Plural-safe: sweater ∈ Cardigans Sweaters / sweaters.
    """

    hay = normalize_name(label)
    needle = normalize_name(token)
    if not hay or not needle:
        return False
    parts = hay.split()
    if needle == hay or needle in parts or f" {needle} " in f" {hay} ":
        return True
    try:
        from knowledge_graph.retrieval.kg_stats import _plural_forms

        forms = _plural_forms(needle)
    except Exception:
        forms = {needle, needle.rstrip("s") if len(needle) > 3 else needle}
    part_forms: set[str] = set()
    for part in parts:
        try:
            from knowledge_graph.retrieval.kg_stats import _plural_forms

            part_forms |= _plural_forms(part)
        except Exception:
            part_forms.add(part)
            if len(part) > 3:
                part_forms.add(part.rstrip("s"))
    return bool(forms & part_forms) or bool(forms & {hay})


def _contains_family_match(requested: str, labels: list[str]) -> bool:
    """Clear base token (white/cotton/sweater) whose candidates all contain that token.

    Cypher already soft-CONTAINS these values — do not block on compound catalog labels.
    """

    needle = normalize_name(requested)
    if not needle or len(needle) < 3 or len(needle.split()) > 2:
        return False
    if not labels:
        return False
    return all(_token_in_label(lab, needle) for lab in labels)


def _match_value(
    requested: str,
    values: list[tuple[str, str]],
) -> tuple[str | None, float, list[str], str]:
    """Return (normalized_raw, score, candidates, evidence).

    Auto-accepts unique case/plural soft hits (sweater ↔ Sweaters). Typo near-misses
    stay AMBIGUOUS so the user can confirm. Base tokens whose candidates all contain
    the token (white → arctic white / beige and white) stay VALID for CONTAINS retrieve.
    """

    if not requested:
        return None, 0.0, [], "empty_value"
    norms = _alias_norms(requested)
    # Exact / alias against live values.
    for raw, nrm in values:
        if nrm in norms or normalize_name(raw) in norms:
            return raw, 1.0, [], "metadata_exact_match"
    # Case-insensitive raw equality.
    req_cf = requested.strip().casefold()
    for raw, nrm in values:
        if raw.casefold() == req_cf:
            return raw, 1.0, [], "metadata_case_insensitive_match"

    # Soft plural / stem (sweater ↔ sweaters / Sweaters) — unique family → VALID.
    try:
        from knowledge_graph.retrieval.kg_stats import _plural_forms, _value_match_score
    except Exception:
        _plural_forms = None  # type: ignore[assignment]
        _value_match_score = None  # type: ignore[assignment]

    soft_labels: list[str] = []
    if _value_match_score is not None and _plural_forms is not None:
        soft_hits: list[tuple[float, str]] = []
        want_forms = set()
        for n in norms:
            want_forms |= _plural_forms(n)
        for raw, nrm in values:
            score = float(_value_match_score(requested, raw))
            if score < 0.96 and not (want_forms & _plural_forms(nrm)):
                continue
            if score < 0.96:
                score = 0.96
            soft_hits.append((score, raw))
        if soft_hits:
            soft_hits.sort(key=lambda item: (-item[0], len(item[1]), item[1].casefold()))
            # Collapse Sweater/Sweaters into one family via plural forms.
            families: dict[frozenset[str], str] = {}
            for score, raw in soft_hits:
                fam = frozenset(_plural_forms(normalize_name(raw)))
                families.setdefault(fam, raw)
            if len(families) == 1:
                return soft_hits[0][1], soft_hits[0][0], [], "metadata_plural_soft_match"
            soft_labels = [raw for _, raw in soft_hits[:8]]
            if _contains_family_match(requested, soft_labels):
                return (
                    requested.strip(),
                    0.92,
                    soft_labels[:5],
                    "metadata_contains_family",
                )

    # Near matches → candidates only (never auto-VALID for typos).
    scored: list[tuple[float, str]] = []
    for raw, nrm in values:
        best = max((_sim(n, nrm) for n in norms), default=0.0)
        best = max(best, _sim(normalize_name(requested), nrm))
        if best >= MATCH_MIN:
            scored.append((best, raw))
    scored.sort(key=lambda item: (-item[0], item[1].casefold()))
    candidates = list(dict.fromkeys([raw for _, raw in scored[:5]]))
    if candidates and _contains_family_match(requested, candidates):
        return requested.strip(), 0.9, candidates, "metadata_contains_family"
    if candidates:
        return None, scored[0][0] if scored else 0.0, candidates, "metadata_ambiguous_or_near"
    return None, 0.0, [], "metadata_value_not_found"


def _db_verify_value(platform, graph_id: str | None, field: str, value: str) -> tuple[str, str]:
    """Return (status, evidence) using a deterministic count when Neo4j is available.

    When ``graph_id`` is unset, probe every product graph. Soft-grounds the value
    against kg_stats (sweater → catalog Sweaters/sweater) before counting.
    """

    if not platform or not value:
        return "UNKNOWN", "db_unavailable"
    try:
        from knowledge_graph.retrieval.ecom import cypher_count_products, registry_product_graphs
        from knowledge_graph.retrieval.kg_stats import find_value_in_stats

        key = _field_key(field)
        # Prefer an explicit product graph; otherwise search all registered product KGs.
        if graph_id:
            candidates = [str(graph_id)]
        else:
            candidates = [str(g) for g in (registry_product_graphs(platform) or []) if g]
            if not candidates:
                candidates = ["ecom_products", "plmex_products"]
        seen: set[str] = set()
        last_zero = ""
        for g in candidates:
            if g in seen:
                continue
            seen.add(g)
            try_value = value
            # Soft-ground onto live catalog casing/plural for this graph.
            try:
                hit = find_value_in_stats(g, value, prefer_attrs=(key, "garment_type", "category", "subfamily"))
                if hit and (hit[0] == key or key in {"garment", "category", "garment_type", "subfamily"}):
                    try_value = hit[1]
            except Exception:
                pass
            try:
                n = cypher_count_products(platform, g, {key: try_value})
            except Exception:
                continue
            if n is None:
                continue
            if int(n) > 0:
                return "VALID", f"db_count:{g}:{int(n)}"
            last_zero = g
        if seen:
            return "NOT_FOUND", f"db_count_zero:{last_zero or next(iter(seen))}"
    except Exception as exc:
        LOGGER.debug("db verify skipped: %s", exc)
    return "UNKNOWN", "db_unavailable"


def validate_filters_strict(
    platform,
    filters: dict[str, Any] | None,
    *,
    source: str | None = None,
    intent: str = "",
    group_by: str | None = None,
    limit: int | None = None,
    effective_question: str = "",
    verify_db: bool = True,
    index: dict[str, list[tuple[str, str]]] | None = None,
    meta=None,
) -> UnderstandingPlan:
    """Validate every explicit filter; block execution when any required status is unresolved."""

    raw = {
        str(k).strip(): str(v).strip()
        for k, v in (filters or {}).items()
        if k not in (None, "") and v not in (None, "")
    }
    fields = list(dict.fromkeys([*raw.keys(), *CORE_METADATA_FIELDS]))
    if meta is None:
        try:
            meta = build_meta_index(platform, fields) if platform is not None else None
        except Exception:
            meta = None
    value_index = index if index is not None else (getattr(meta, "values", None) or {})

    validated: list[ValidatedFilter] = []
    for field, value in raw.items():
        key = _field_key(field)
        original = value
        # Multi-value "A | B" — validate each part; all must be VALID.
        parts = [p.strip() for p in re.split(r"\s*\|\s*", value) if p.strip()] or [value]
        if key in FREEFORM_FILTER_KEYS:
            validated.append(
                ValidatedFilter(
                    original_text=original,
                    field=key,
                    requested_value=value,
                    normalized_value=value,
                    status="VALID",
                    confidence=1.0,
                    evidence="freeform_passthrough",
                    source_graph=source,
                )
            )
            continue

        supported = _field_supported(meta, key, source) if meta is not None else None
        if supported is False:
            validated.append(
                ValidatedFilter(
                    original_text=original,
                    field=key,
                    requested_value=value,
                    normalized_value=None,
                    status="UNSUPPORTED",
                    confidence=1.0,
                    evidence="field_absent_on_source",
                    source_graph=source,
                )
            )
            continue

        values = _index_values_for_field(value_index, key)
        part_statuses: list[ValidatedFilter] = []
        for part in parts:
            matched, score, candidates, evidence = _match_value(part, values)
            if matched:
                part_statuses.append(
                    ValidatedFilter(
                        original_text=part,
                        field=key,
                        requested_value=part,
                        normalized_value=matched,
                        status="VALID",
                        confidence=score,
                        evidence=evidence,
                        candidates=[],
                        source_graph=source,
                    )
                )
                continue
            if candidates:
                part_statuses.append(
                    ValidatedFilter(
                        original_text=part,
                        field=key,
                        requested_value=part,
                        normalized_value=None,
                        status="AMBIGUOUS",
                        confidence=score,
                        evidence=evidence,
                        candidates=candidates,
                        source_graph=source,
                    )
                )
                continue
            # No metadata hit.
            if not values:
                # Incomplete metadata — try DB before UNKNOWN.
                if verify_db and platform is not None:
                    status, ev = _db_verify_value(platform, source, key, part)
                    part_statuses.append(
                        ValidatedFilter(
                            original_text=part,
                            field=key,
                            requested_value=part,
                            normalized_value=part if status == "VALID" else None,
                            status=status,
                            confidence=1.0 if status == "VALID" else 0.0,
                            evidence=ev,
                            source_graph=source,
                        )
                    )
                else:
                    part_statuses.append(
                        ValidatedFilter(
                            original_text=part,
                            field=key,
                            requested_value=part,
                            normalized_value=None,
                            status="UNKNOWN",
                            confidence=0.0,
                            evidence="metadata_incomplete",
                            source_graph=source,
                        )
                    )
                continue
            # Metadata present but value absent — verify DB (never assume NOT_FOUND from meta alone).
            if verify_db and platform is not None:
                status, ev = _db_verify_value(platform, source, key, part)
                if status == "VALID":
                    part_statuses.append(
                        ValidatedFilter(
                            original_text=part,
                            field=key,
                            requested_value=part,
                            normalized_value=part,
                            status="VALID",
                            confidence=1.0,
                            evidence=ev,
                            source_graph=source,
                        )
                    )
                elif status == "NOT_FOUND":
                    part_statuses.append(
                        ValidatedFilter(
                            original_text=part,
                            field=key,
                            requested_value=part,
                            normalized_value=None,
                            status="NOT_FOUND",
                            confidence=1.0,
                            evidence=ev,
                            source_graph=source,
                        )
                    )
                else:
                    part_statuses.append(
                        ValidatedFilter(
                            original_text=part,
                            field=key,
                            requested_value=part,
                            normalized_value=None,
                            status="UNKNOWN",
                            confidence=0.0,
                            evidence=ev or "metadata_value_not_found",
                            source_graph=source,
                        )
                    )
            else:
                # Strict categorical: missing from a populated index without DB → UNKNOWN
                # (do not claim NOT_FOUND without DB proof).
                status = "UNKNOWN"
                if key in STRICT_METADATA_FIELDS:
                    evidence = "metadata_value_not_found_no_db"
                else:
                    evidence = "metadata_value_not_found"
                part_statuses.append(
                    ValidatedFilter(
                        original_text=part,
                        field=key,
                        requested_value=part,
                        normalized_value=None,
                        status=status,
                        confidence=0.0,
                        evidence=evidence,
                        source_graph=source,
                    )
                )

        if len(part_statuses) == 1:
            validated.append(part_statuses[0])
            validated[-1].original_text = original
            validated[-1].requested_value = value
        else:
            # Collapse multi-part: VALID only if every part VALID.
            if part_statuses and all(p.status == "VALID" for p in part_statuses):
                joined = " | ".join(p.normalized_value or p.requested_value for p in part_statuses)
                validated.append(
                    ValidatedFilter(
                        original_text=original,
                        field=key,
                        requested_value=value,
                        normalized_value=joined,
                        status="VALID",
                        confidence=min(p.confidence for p in part_statuses),
                        evidence="metadata_exact_match_multi",
                        source_graph=source,
                    )
                )
            else:
                # Prefer first blocking part for clarification.
                bad = next((p for p in part_statuses if p.status in _BLOCKING), part_statuses[0])
                validated.append(
                    ValidatedFilter(
                        original_text=original,
                        field=key,
                        requested_value=value,
                        normalized_value=None,
                        status=bad.status,
                        confidence=bad.confidence,
                        evidence=bad.evidence,
                        candidates=list(bad.candidates),
                        source_graph=source,
                    )
                )

    # Optional: validate group_by as a field name (not a value).
    if group_by:
        gb = _field_key(group_by)
        if gb not in FREEFORM_FILTER_KEYS:
            supported = _field_supported(meta, gb, source) if meta is not None else None
            if supported is False:
                validated.append(
                    ValidatedFilter(
                        original_text=str(group_by),
                        field="group_by",
                        requested_value=str(group_by),
                        normalized_value=None,
                        status="UNSUPPORTED",
                        confidence=1.0,
                        evidence="group_by_field_absent_on_source",
                        source_graph=source,
                    )
                )

    blocking = [f for f in validated if f.status in _BLOCKING and f.field != "group_by"]
    # group_by UNSUPPORTED also blocks.
    blocking.extend([f for f in validated if f.field == "group_by" and f.status in _BLOCKING])

    items = [_clarification_item(f) for f in blocking]
    text = _clarification_message(items, valid=[f for f in validated if f.status == "VALID"])
    allowed = not blocking
    return UnderstandingPlan(
        intent=intent or "",
        source=source,
        filters=validated,
        group_by=group_by,
        limit=limit,
        clarification_required=not allowed,
        clarification_items=items,
        clarification_text=text if not allowed else "",
        execution_allowed=allowed,
        effective_question=effective_question or "",
    )


def _clarification_item(filt: ValidatedFilter) -> ClarificationItem:
    field = filt.field
    req = filt.requested_value
    cands = list(filt.candidates or [])[:5]
    if filt.status == "AMBIGUOUS" and cands:
        opts = ", ".join(cands[:3])
        q = (
            f'I found several possible matches for {field} "{req}" ({opts}). '
            f"Which one should I use, or should I remove this filter?"
        )
    elif filt.status == "NOT_FOUND":
        hint = f" Did you mean {cands[0]}?" if cands else ""
        q = (
            f'I could not find {field} "{req}" in the catalog.{hint} '
            f"Correct the value, or tell me to remove this filter."
        )
    elif filt.status == "UNSUPPORTED":
        q = (
            f'The selected source does not support the "{field}" filter. '
            f"Choose another source, or remove this filter."
        )
    else:  # UNKNOWN
        hint = f" Suggestions: {', '.join(cands[:3])}." if cands else ""
        q = (
            f'I could not verify {field} "{req}" against catalog metadata.{hint} '
            f"Confirm the exact value, replace it, or remove this filter."
        )
    return ClarificationItem(
        field=field,
        requested_value=req,
        status=filt.status,
        question=q,
        candidates=cands,
    )


def _clarification_message(
    items: list[ClarificationItem],
    *,
    valid: list[ValidatedFilter] | None = None,
) -> str:
    if not items:
        return ""
    lines: list[str] = []
    if valid:
        kept = ", ".join(
            f"{f.field}={f.normalized_value or f.requested_value}" for f in valid[:6]
        )
        if kept:
            lines.append(f"Keeping: {kept}.")
    if len(items) == 1:
        lines.append(items[0].question)
    else:
        lines.append("I need to clarify a few filters before I query:")
        for i, item in enumerate(items, 1):
            lines.append(f"{i}. {item.question}")
    lines.append(
        'Reply with a correction (e.g. "use Blue"), a number for a suggestion, '
        'or "remove <field>".'
    )
    return "\n".join(lines)


def execution_filters(plan: UnderstandingPlan) -> dict[str, str]:
    """Filters safe to pass into Cypher / kg_stats — VALID only."""

    return plan.valid_filters_dict()


def _seed_index_from_plan(plan: UnderstandingPlan) -> dict[str, list[tuple[str, str]]]:
    """Micro-index so revalidation does not flip known-good filters to UNKNOWN offline."""

    seed: dict[str, list[tuple[str, str]]] = {}

    def add(field: str, raw: str) -> None:
        key = _field_key(field)
        value = str(raw or "").strip()
        if not key or not value:
            return
        bucket = seed.setdefault(key, [])
        norm = normalize_name(value)
        if any(n == norm for _, n in bucket):
            return
        bucket.append((value, norm))

    for filt in plan.filters:
        if filt.status == "VALID":
            add(filt.field, filt.normalized_value or filt.requested_value)
        for cand in filt.candidates or []:
            add(filt.field, cand)
    for item in plan.clarification_items:
        for cand in item.candidates or []:
            add(item.field, cand)
    return seed


def apply_clarification_response(
    platform,
    plan: UnderstandingPlan,
    user_text: str,
    *,
    verify_db: bool = True,
    index: dict[str, list[tuple[str, str]]] | None = None,
    meta=None,
) -> UnderstandingPlan:
    """Update only unresolved filters from the user reply, then revalidate."""

    text = (user_text or "").strip()
    if not text or not plan.clarification_required:
        return plan

    pending_fields = {c.field for c in plan.clarification_items}
    # Start from previously VALID filters only.
    base = plan.valid_filters_dict()
    # Re-include unresolved requested values so user can correct them in place.
    unresolved_map = {
        f.field: f.requested_value
        for f in plan.filters
        if f.status in _BLOCKING and f.field != "group_by"
    }
    removed: set[str] = set()

    # Remove / cancel a filter.
    rem = _REMOVE_RE.search(text)
    if rem:
        field = (rem.group("field") or "").casefold()
        value = (rem.group("value") or "").strip()
        if field:
            key = _field_key(field)
            base.pop(key, None)
            unresolved_map.pop(key, None)
            removed.add(key)
        elif value:
            vn = normalize_name(value)
            for key, raw in list(unresolved_map.items()):
                if normalize_name(raw) == vn or vn in normalize_name(raw):
                    unresolved_map.pop(key, None)
                    base.pop(key, None)
                    removed.add(key)
            # Also "remove color" style caught as value=color
            if _field_key(value) in pending_fields or _field_key(value) in unresolved_map:
                key = _field_key(value)
                unresolved_map.pop(key, None)
                base.pop(key, None)
                removed.add(key)

    # Numbered candidate: "1" / "2" against first clarification item with candidates.
    if re.fullmatch(r"\d{1,2}", text):
        idx = int(text) - 1
        for item in plan.clarification_items:
            if item.candidates and 0 <= idx < len(item.candidates):
                unresolved_map[item.field] = item.candidates[idx]
                break

    # "use Blue" / "Blue instead" / bare candidate match.
    use_match = re.search(
        r"(?i)\b(?:use|prefer|change\s+to|replace\s+with|mean(?:t)?)\s+"
        r"(?P<val>[\w'’ -]{2,40})\s*$",
        text,
    )
    bare = text
    replacement = (use_match.group("val") if use_match else "").strip() or None
    if replacement is None and len(plan.clarification_items) == 1:
        item = plan.clarification_items[0]
        # Bare reply matching a candidate or free correction for the single open field.
        for cand in item.candidates:
            if normalize_name(bare) == normalize_name(cand) or bare.casefold() == cand.casefold():
                replacement = cand
                break
        if replacement is None and bare and not rem and not re.fullmatch(r"\d{1,2}", bare):
            # Treat as a corrected value for the single pending field.
            if not re.search(r"(?i)\b(remove|drop|ignore|skip|without|cancel)\b", bare):
                replacement = bare
        if replacement is not None:
            unresolved_map[item.field] = replacement

    if replacement and use_match:
        # Apply to the first pending field, or field named in the sentence.
        target = None
        for item in plan.clarification_items:
            if item.field in normalize_name(text) or _field_key(item.field) in text.casefold():
                target = item.field
                break
        target = target or (plan.clarification_items[0].field if plan.clarification_items else None)
        if target:
            unresolved_map[target] = replacement

    # Merge: valid base + remaining unresolved (still to revalidate).
    merged = {**base, **{k: v for k, v in unresolved_map.items() if k not in removed}}

    seed = _seed_index_from_plan(plan)
    for key, value in merged.items():
        bucket = seed.setdefault(_field_key(key), [])
        norm = normalize_name(value)
        if norm and not any(n == norm for _, n in bucket):
            bucket.append((value, norm))
    if index:
        for key, pairs in index.items():
            bucket = seed.setdefault(_field_key(key), [])
            for raw, nrm in pairs:
                if not any(n == nrm for _, n in bucket):
                    bucket.append((raw, nrm))

    return validate_filters_strict(
        platform,
        merged,
        source=plan.source,
        intent=plan.intent,
        group_by=plan.group_by,
        limit=plan.limit,
        effective_question=plan.effective_question,
        verify_db=verify_db,
        index=seed,
        meta=meta,
    )


def gate_intention_filters(
    platform,
    intention,
    *,
    question: str = "",
    state: dict | None = None,
    verify_db: bool = True,
) -> tuple[UnderstandingPlan, Any]:
    """Validate intention.filters; on success replace with VALID-only values.

    Returns (plan, intention). When execution is blocked, intention.filters still
    contain only previously VALID values (unresolved keys removed — not guessed).
    """

    source = None
    prefs = list(getattr(intention, "preferred_graphs", None) or [])
    explicit_source = False
    try:
        from fashion_brain.ask import question_names_catalog_explicitly

        explicit_source = bool(question_names_catalog_explicitly(question or "", platform))
    except Exception:
        explicit_source = False

    sticky = str((state or {}).get("last_graph_id") or "").strip() or None
    if explicit_source and prefs:
        source = str(prefs[0])
    elif sticky and not prefs:
        source = sticky
    elif sticky and prefs and sticky in prefs:
        source = sticky
    elif explicit_source and sticky:
        source = sticky
    else:
        # User did not name a catalog — do not pin validation to one graph.
        # Soft DB verify will probe ecom + plmex; miss everywhere → NOT_FOUND.
        source = None
        if not prefs:
            try:
                from knowledge_graph.retrieval.ecom import registry_product_graphs

                prefs = [str(g) for g in (registry_product_graphs(platform) or []) if g]
                if prefs:
                    intention.preferred_graphs = prefs
            except Exception:
                pass

    # Graph-aware semantic aliases before validation (section↔gender, brand↔buyer, …).
    # When source is unset, apply universal aliases only (composition→material, …).
    try:
        from fashion_brain.planning.semantic_field_map import apply_semantic_field_map

        mapped, mapped_gb = apply_semantic_field_map(
            dict(getattr(intention, "filters", None) or {}),
            source,
            group_by=getattr(intention, "group_by", None),
        )
        intention.filters = mapped
        if mapped_gb is not None or getattr(intention, "group_by", None):
            intention.group_by = mapped_gb
    except Exception:
        LOGGER.debug("semantic field map skipped", exc_info=True)

    # Resume: if a prior plan is awaiting clarification, apply this user turn first.
    prior = None
    if state:
        prior = UnderstandingPlan.from_dict(state.get("understanding_plan"))
    if prior and prior.clarification_required and (question or "").strip():
        plan = apply_clarification_response(
            platform, prior, question, verify_db=verify_db
        )
    else:
        plan = validate_filters_strict(
            platform,
            dict(getattr(intention, "filters", None) or {}),
            source=source,
            intent=str(getattr(intention, "intent", "") or ""),
            group_by=getattr(intention, "group_by", None),
            limit=getattr(intention, "limit", None),
            effective_question=question or "",
            verify_db=verify_db,
        )

    # Always strip non-VALID filters before any retrieve/Cypher path.
    intention.filters = execution_filters(plan)
    if plan.group_by and any(
        f.field == "group_by" and f.status in _BLOCKING for f in plan.filters
    ):
        intention.group_by = None
    elif plan.group_by:
        intention.group_by = plan.group_by
    return plan, intention
