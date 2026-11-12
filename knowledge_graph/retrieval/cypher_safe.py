"""Read-only Cypher validation and execution for LLM Text-to-Cypher."""

from __future__ import annotations

import re
from typing import Any

_WRITE = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|CALL\s+dbms|CALL\s+db\.|FOREACH|ALTER|GRANT|DENY)\b",
    re.IGNORECASE,
)
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


def run_read_cypher(platform, cypher: str, params: dict[str, Any] | None = None, *, limit_rows: int = 200) -> list[dict]:
    """Execute a validated read Cypher against Neo4j; return list of row dicts."""

    import time

    cleaned = assert_read_only_cypher(cypher)
    session_factory = getattr(platform.graph, "_session", None)
    if not callable(session_factory):
        raise RuntimeError("Neo4j session is not available.")
    bind = dict(params or {})
    started = time.perf_counter()
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
            lines.append(f"| {row.get(label_key)} | {row.get(count_key)} |")
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
            value = row.get(key)
            if value is None:
                cells.append("—")
            else:
                cells.append(str(value).replace("|", "/").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _short_cypher(cypher: str, limit: int = 180) -> str:
    text = sanitize_cypher(cypher)
    return text if len(text) <= limit else text[: limit - 1] + "…"
