"""Read-only Cypher validation and execution for LLM Text-to-Cypher."""

from __future__ import annotations

import re
from typing import Any

# Block write/admin. Allow only fulltext queryNodes under CALL db.index.fulltext.
_WRITE = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|CALL\s+dbms|FOREACH|ALTER|GRANT|DENY)\b",
    re.IGNORECASE,
)
_FORBIDDEN_DB_CALL = re.compile(r"\bCALL\s+db\.(?!index\.fulltext\.queryNodes\b)", re.IGNORECASE)
_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"//.*?$", re.MULTILINE)


def sanitize_cypher(cypher: str) -> str:
    text = _COMMENT.sub(" ", str(cypher or ""))
    text = _LINE_COMMENT.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


# Facet vars comparing .name to a string literal (reject — use id / normalized_name).
_FACET_NAME_LITERAL = re.compile(
    r"\b([a-zA-Z_][\w]*)\.name\s*(?:=~|=|<>|CONTAINS|STARTS\s+WITH|ENDS\s+WITH)\s*(?:'[^']*'|\"[^\"]*\"|\$\w+)",
    re.IGNORECASE,
)
# Product display name in RETURN is fine; ban comparisons in WHERE-like positions for known facet aliases.
_FACET_ALIASES = frozenset(
    {"m", "mat", "material", "c", "cat", "category", "col", "color", "colour", "a", "avail", "availability", "b", "brand", "x"}
)


def assert_read_only_cypher(cypher: str) -> str:
    """Return cleaned Cypher or raise ValueError if it is not a safe read query."""

    cleaned = sanitize_cypher(cypher)
    if not cleaned:
        raise ValueError("Empty Cypher.")
    if ";" in cleaned.rstrip(";"):
        raise ValueError("Multiple Cypher statements are not allowed.")
    if _WRITE.search(cleaned):
        raise ValueError("Write/admin Cypher clauses are not allowed.")
    if _FORBIDDEN_DB_CALL.search(cleaned):
        raise ValueError("Only CALL db.index.fulltext.queryNodes is allowed among db.* procedures.")
    if not re.search(r"\b(MATCH|OPTIONAL\s+MATCH|CALL|WITH|UNWIND)\b", cleaned, re.IGNORECASE):
        raise ValueError("Cypher must start from MATCH/CALL/WITH/UNWIND.")
    if not re.search(r"\bRETURN\b", cleaned, re.IGNORECASE):
        raise ValueError("Cypher must include RETURN.")
    cleaned = _reject_or_rewrite_facet_name_literals(cleaned)
    # Cap runaway result size when the model omits LIMIT on row listings.
    if not re.search(r"\bLIMIT\s+\d+", cleaned, re.IGNORECASE) and re.search(
        r"\bRETURN\b(?!.*\bcount\s*\()", cleaned, re.IGNORECASE | re.DOTALL
    ):
        cleaned = f"{cleaned.rstrip(';')} LIMIT 50"
    return cleaned


def _reject_or_rewrite_facet_name_literals(cypher: str) -> str:
    """Rewrite facet `.name` comparisons to `.normalized_name` with toLower param guidance.

    Product `p.name` RETURN/ORDER is allowed. Facet aliases comparing `.name` to a
    literal or param are rewritten to `toLower(coalesce(var.normalized_name,''))`.
    """

    def repl(match: re.Match[str]) -> str:
        var = match.group(1)
        if var.casefold() in {"p", "product"}:
            # Product title compare is rare but allowed (listing filters by title).
            return match.group(0)
        if var.casefold() not in _FACET_ALIASES and not var.casefold().startswith(("cat", "mat", "col", "brand", "avail")):
            # Unknown alias — still rewrite to normalized_name to avoid case bugs.
            pass
        op_match = re.search(
            r"\.name\s*(=|<>|CONTAINS|STARTS\s+WITH|ENDS\s+WITH)\s*(.+)$",
            match.group(0),
            re.IGNORECASE,
        )
        if not op_match:
            return match.group(0)
        op = op_match.group(1)
        rhs = op_match.group(2).strip()
        # Prefer equality on normalized_name; CONTAINS stays CONTAINS.
        left = f"toLower(coalesce({var}.normalized_name,''))"
        if rhs.startswith("$"):
            right = f"toLower(toString({rhs}))"
        elif (rhs.startswith("'") and rhs.endswith("'")) or (rhs.startswith('"') and rhs.endswith('"')):
            lit = rhs[1:-1]
            right = json_escape_lower(lit)
        else:
            right = f"toLower(toString({rhs}))"
        return f"{left} {op} {right}"

    return _FACET_NAME_LITERAL.sub(repl, cypher)


def json_escape_lower(value: str) -> str:
    text = str(value).casefold().replace("\\", "\\\\").replace("'", "\\'")
    return f"'{text}'"


# Relationship type in [:REL] or [:REL|OTHER]
_REL_TYPES = re.compile(r"\[:?\s*([A-Z][A-Z0-9_]*)\s*(?:\|[A-Z0-9_|]*)?\s*(?:[*\d.]*)?\s*(?:\{[^}]*\})?\s*\]")
_LABELED_DIR = re.compile(
    r"\((?:[a-zA-Z_][\w]*)?(?::`?([A-Za-z_][\w]*)`?)*\)\s*"
    r"-\[\s*:?\s*([A-Z][A-Z0-9_]*)[^\]]*\]->\s*"
    r"\((?:[a-zA-Z_][\w]*)?(?::`?([A-Za-z_][\w]*)`?)*\)"
)
_LABELED_REV = re.compile(
    r"\((?:[a-zA-Z_][\w]*)?(?::`?([A-Za-z_][\w]*)`?)*\)\s*"
    r"<-\[\s*:?\s*([A-Z][A-Z0-9_]*)[^\]]*\]-\s*"
    r"\((?:[a-zA-Z_][\w]*)?(?::`?([A-Za-z_][\w]*)`?)*\)"
)
_ALWAYS_ALLOWED_RELS = frozenset(
    {
        "RESOLVES_TO",
        "SIMILAR_TO",
        "IN_GRAPH",
        "IN_DOMAIN",
        "HAS_ALIAS",
        "IN_GROUP",
        "IN_MATERIAL_GROUP",
        "IN_COLOR_FAMILY",
        "MAPS_TO_CATEGORY",
        "FOR_TARGET_GROUP",
        "MADE_WITH_METHOD",
        "SUITED_FOR_OCCASION",
        "IN_PRICE_BAND",
        "HAS_REFERENCE",
        "HAS_PATTERN",
        "REFERS_TO_MATERIAL",
        "USES_MATERIAL",
        "PRIMARY_MATERIAL",
        "HAS_BRAND",
        "HAS_COLOR",
        "PRIMARY_COLOR",
        "HAS_BUYER",
        "HAS_SECTION",
        "HAS_GENDER",
    }
)


def extract_relationship_types(cypher: str) -> list[str]:
    cleaned = sanitize_cypher(cypher)
    return list(dict.fromkeys(_REL_TYPES.findall(cleaned)))


def assert_allowlisted_cypher(
    cypher: str,
    allowlist: dict[str, list[list[str]]] | None,
    *,
    strict_direction: bool = True,
) -> str:
    """Validate ``[:REL]`` types (and simple labeled directions) against allowlist.

    Empty allowlist → no-op (fail-open). Structural rels always allowed.
    """

    cleaned = sanitize_cypher(cypher)
    if not cleaned:
        raise ValueError("Empty Cypher.")
    if not allowlist:
        return cleaned

    normalized: dict[str, list[list[str]]] = {}
    for rel, pairs in allowlist.items():
        key = str(rel).upper()
        bucket: list[list[str]] = []
        for pair in pairs or []:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                bucket.append([str(pair[0]), str(pair[1])])
        normalized[key] = bucket

    for rel in extract_relationship_types(cleaned):
        upper = rel.upper()
        if upper in _ALWAYS_ALLOWED_RELS:
            continue
        if upper not in normalized:
            known = ", ".join(sorted(normalized.keys())[:30]) or "(none)"
            raise ValueError(
                f"Relationship type {upper} is not in the ontology allowlist. "
                f"Valid types include: {known}."
            )

    if strict_direction:
        for match in _LABELED_DIR.finditer(cleaned):
            src, rel, tgt = match.group(1), match.group(2).upper(), match.group(3)
            if not src or not tgt or rel in _ALWAYS_ALLOWED_RELS:
                continue
            pairs = normalized.get(rel) or []
            if pairs and not _pair_compatible(src, tgt, pairs):
                valid = ", ".join(f"[{a}, {b}]" for a, b in pairs)
                raise ValueError(
                    f"Direction ({src})-[:{rel}]->({tgt}) is not allowlisted. "
                    f"Valid [Source, Target] pairs for {rel}: {valid}."
                )
        for match in _LABELED_REV.finditer(cleaned):
            tgt, rel, src = match.group(1), match.group(2).upper(), match.group(3)
            if not src or not tgt or rel in _ALWAYS_ALLOWED_RELS:
                continue
            pairs = normalized.get(rel) or []
            if pairs and not _pair_compatible(src, tgt, pairs):
                valid = ", ".join(f"[{a}, {b}]" for a, b in pairs)
                raise ValueError(
                    f"Direction ({src})-[:{rel}]->({tgt}) is not allowlisted. "
                    f"Valid [Source, Target] pairs for {rel}: {valid}."
                )
    return cleaned


def _pair_compatible(src: str, tgt: str, pairs: list[list[str]]) -> bool:
    s, t = src.casefold(), tgt.casefold()
    for a, b in pairs:
        if str(a).casefold() == s and str(b).casefold() == t:
            return True
    return False


def run_read_cypher(
    platform,
    cypher: str,
    params: dict[str, Any] | None = None,
    *,
    limit_rows: int = 200,
    allowlist: dict[str, list[list[str]]] | None = None,
) -> list[dict]:
    """Execute a validated read Cypher against Neo4j; return list of row dicts."""

    import time

    cleaned = assert_read_only_cypher(cypher)
    if allowlist is not None:
        cleaned = assert_allowlisted_cypher(cleaned, allowlist)
    session_factory = getattr(platform.graph, "_session", None)
    if not callable(session_factory):
        raise RuntimeError("Neo4j session is not available.")
    bind = dict(params or {})
    started = time.perf_counter()
    try:
        from knowledge_graph.retrieval.cypher_log import log_cypher_hit

        log_cypher_hit(cleaned, bind, source="text2cypher")
    except Exception:
        pass
    with session_factory() as session:
        rows = [dict(record) for record in session.run(cleaned, bind)]
    rows = rows[: max(1, int(limit_rows))]
    ms = (time.perf_counter() - started) * 1000.0
    try:
        from fashion_brain.stage_timing import record_cypher, record_stage

        record_stage("cypher_exec", ms, kind="text2cypher", rows=len(rows))
        record_cypher(cleaned, params=bind, rows=len(rows), ms=ms)
    except Exception:
        pass
    return rows


_COLON_ID = re.compile(r"^[a-z0-9_]+:[a-z0-9_]+:[a-z0-9_\-./]+$", re.IGNORECASE)
_SLUG_ONLY = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)+$")


def humanize_findings_labels(value: Any, *, row: dict | None = None, key: str | None = None) -> str:
    """Display-only: prefer human names over raw colon/slug graph ids.

    Does not invent facts; numbers and empty values pass through unchanged.
    Evidence record_ids stay untouched — this is for findings prose/tables only.
    """

    if value is None:
        return "—"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    text = str(value).strip()
    if not text:
        return "—"
    # Prefer a parallel name/title column when the cell is an internal id.
    if row and key and str(key).casefold() in {"id", "node_id", "canonical_id", "product_id", "record_id"}:
        for alt in ("name", "title", "canonical_name", "product_name", "label"):
            alt_val = row.get(alt)
            if alt_val not in (None, ""):
                return str(alt_val).replace("|", "/").replace("\n", " ")
    if _COLON_ID.match(text):
        # prefix:type:slug → title-case last segment
        last = text.rsplit(":", 1)[-1]
        return _title_slug(last)
    if _SLUG_ONLY.match(text) and " " not in text:
        return _title_slug(text)
    return text.replace("|", "/").replace("\n", " ")


def _title_slug(slug: str) -> str:
    parts = re.split(r"[-_]+", str(slug or "").strip())
    return " ".join(p[:1].upper() + p[1:] for p in parts if p)


def rows_to_findings(rows: list[dict], *, explanation: str = "", cypher: str = "") -> str:
    """Turn Neo4j rows into grounded findings text for ResponseAgent."""

    lines: list[str] = []
    if explanation:
        lines.append(explanation.strip())
    if not rows:
        lines.append("The Cypher query returned 0 rows.")
        if cypher:
            lines.append(f"(query: `{_short_cypher(cypher)}`)")
        return "\n".join(lines)

    # Single count cell
    if len(rows) == 1 and len(rows[0]) == 1:
        key, value = next(iter(rows[0].items()))
        if str(key).casefold() in {"c", "count", "n", "total"} or isinstance(value, (int, float)):
            lines.append(f"Result: {value}")
            return "\n".join(lines)

    keys = list(rows[0].keys())
    # Two-column breakdown (label, count)
    if len(keys) == 2 and any(str(k).casefold() in {"n", "c", "count", "total", "product_count"} for k in keys):
        label_key = next(
            k for k in keys if str(k).casefold() not in {"n", "c", "count", "total", "product_count"}
        )
        count_key = next(k for k in keys if k != label_key)
        lines.append(f"Breakdown ({len(rows)} groups):")
        lines.append(f"| {label_key} | {count_key} |")
        lines.append("| --- | --- |")
        for row in rows[:40]:
            label = humanize_findings_labels(row.get(label_key), row=row, key=str(label_key))
            lines.append(f"| {label} | {row.get(count_key)} |")
        return "\n".join(lines)

    # Deduplicate identical listing rows (e.g. repeated parent category names).
    unique: list[dict] = []
    seen: set[tuple] = set()
    for row in rows:
        marker = tuple((k, str(row.get(k))) for k in keys)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(row)

    # Product-name dumps: keep a short sample so ResponseAgent does not paste 40 SKUs.
    key_fold = {str(k).casefold() for k in keys}
    looks_like_product_list = bool(
        key_fold & {"name", "product", "product_name", "title"}
    ) and not (key_fold & {"n", "c", "count", "total", "product_count"})
    row_cap = 5 if looks_like_product_list else 40
    shown = unique[:row_cap]
    lines.append(f"Rows ({len(shown)} of {len(unique)}):")
    if looks_like_product_list and len(unique) > row_cap:
        lines.append(
            "Sample only — prefer total count and brand/category mix over listing every product."
        )
    lines.append("| " + " | ".join(str(k) for k in keys) + " |")
    lines.append("| " + " | ".join("---" for _ in keys) + " |")
    for row in shown:
        cells = []
        for key in keys:
            cells.append(humanize_findings_labels(row.get(key), row=row, key=str(key)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _short_cypher(cypher: str, limit: int = 180) -> str:
    text = sanitize_cypher(cypher)
    return text if len(text) <= limit else text[: limit - 1] + "…"
