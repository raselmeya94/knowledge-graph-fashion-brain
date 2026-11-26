"""Country-of-origin aliases: ISO codes ↔ English names for catalog facets.

E-commerce origin fields mix ``CN`` / ``China`` / ``china``. Display and filter
matching collapse these to one canonical English label. Multi-origin strings
(``Cambodia Bangladesh`` vs ``Bangladesh Cambodia``) are order-normalized.
"""

from __future__ import annotations

import re

# ISO 3166-1 alpha-2 → English short name (catalog-facing).
# Keep focused on codes seen in ecom country_of_origin (+ common apparel origins).
ISO2_TO_NAME: dict[str, str] = {
    "bd": "Bangladesh",
    "be": "Belgium",
    "bg": "Bulgaria",
    "cn": "China",
    "eg": "Egypt",
    "es": "Spain",
    "et": "Ethiopia",
    "gb": "United Kingdom",
    "id": "Indonesia",
    "in": "India",
    "it": "Italy",
    "jp": "Japan",
    "kh": "Cambodia",
    "lk": "Sri Lanka",
    "ma": "Morocco",
    "mg": "Madagascar",
    "mm": "Myanmar",
    "pk": "Pakistan",
    "pt": "Portugal",
    "tn": "Tunisia",
    "tr": "Turkey",
    "uk": "United Kingdom",
    "us": "United States",
    "vn": "Vietnam",
}

# English / alternate names → same canonical English label.
_NAME_ALIASES: dict[str, str] = {
    "bangladesh": "Bangladesh",
    "belgium": "Belgium",
    "bulgaria": "Bulgaria",
    "cambodia": "Cambodia",
    "china": "China",
    "egypt": "Egypt",
    "england": "United Kingdom",
    "ethiopia": "Ethiopia",
    "great britain": "United Kingdom",
    "india": "India",
    "indonesia": "Indonesia",
    "italy": "Italy",
    "japan": "Japan",
    "madagascar": "Madagascar",
    "morocco": "Morocco",
    "myanmar": "Myanmar",
    "burma": "Myanmar",
    "pakistan": "Pakistan",
    "portugal": "Portugal",
    "spain": "Spain",
    "sri lanka": "Sri Lanka",
    "srilanka": "Sri Lanka",
    "tunisia": "Tunisia",
    "turkey": "Turkey",
    "turkiye": "Turkey",
    "türkiye": "Turkey",
    "united kingdom": "United Kingdom",
    "uk": "United Kingdom",
    "u k": "United Kingdom",
    "united states": "United States",
    "usa": "United States",
    "u s a": "United States",
    "vietnam": "Vietnam",
    "viet nam": "Vietnam",
}

# Non-country junk sometimes scraped into country_of_origin.
_NON_COUNTRY = frozenset(
    {
        "empty",
        "unknown",
        "n a",
        "na",
        "none",
        "null",
        "breathable designed to support",
    }
)


def _norm_key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().casefold().replace("_", " "))


def canonical_country_name(value: str) -> str | None:
    """Single-country token → canonical English name, or None if not a known country."""

    key = _norm_key(value)
    if not key or key in _NON_COUNTRY:
        return None
    # Strip trailing punctuation.
    key = key.strip(" .;,/|")
    if not key or key in _NON_COUNTRY:
        return None
    if len(key) == 2 and key.isalpha():
        return ISO2_TO_NAME.get(key)
    if key in _NAME_ALIASES:
        return _NAME_ALIASES[key]
    # Exact phrase only for single-token canonicalization — never substring
    # ("cambodia bangladesh" must not collapse to Bangladesh alone).
    return None


def _split_origin_tokens(value: str) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    # Separators seen in ecom: ";", ",", "/", "|", " and ", whitespace runs between codes.
    parts = re.split(r"\s*(?:;|,|/|\||\band\b)\s*|\s{2,}", text, flags=re.IGNORECASE)
    out: list[str] = []
    for part in parts:
        part = part.strip(" .")
        if not part:
            continue
        words = part.split()
        # Space-separated ISO pairs: "KH BD".
        if len(words) >= 2 and all(len(w) == 2 and w.isalpha() for w in words):
            out.extend(words)
            continue
        # Space-separated country names: "Cambodia Bangladesh" / "Sri Lanka India".
        if len(words) >= 2:
            split_names = _split_known_country_words(words)
            if split_names and len(split_names) >= 2:
                out.extend(split_names)
                continue
        out.append(part)
    return out


def _split_known_country_words(words: list[str]) -> list[str] | None:
    """Greedy left-to-right match of known country phrases over whitespace tokens."""

    known = sorted(
        {*_NAME_ALIASES.keys(), *ISO2_TO_NAME.keys()},
        key=len,
        reverse=True,
    )
    i = 0
    found: list[str] = []
    while i < len(words):
        matched = None
        for alias in known:
            alias_words = alias.split()
            n = len(alias_words)
            if n == 0 or i + n > len(words):
                continue
            window = " ".join(words[i : i + n]).casefold()
            if window == alias:
                matched = " ".join(words[i : i + n])
                i += n
                break
        if not matched:
            return None
        found.append(matched)
    return found if found else None


def country_display_label(value: str) -> str | None:
    """User-facing country label; multi-origin → sorted ``A; B``.

    Returns None when the value is not country-like (caller keeps original).
    """

    raw = str(value or "").strip()
    if not raw:
        return None
    if _norm_key(raw) in _NON_COUNTRY:
        return None
    tokens = _split_origin_tokens(raw) or [raw]
    names: list[str] = []
    for tok in tokens:
        name = canonical_country_name(tok)
        if name:
            names.append(name)
        else:
            # Keep unknown token as lightly cleaned title-case fragment.
            cleaned = re.sub(r"\s+", " ", tok).strip()
            if cleaned and _norm_key(cleaned) not in _NON_COUNTRY:
                names.append(cleaned.title() if cleaned.isupper() or cleaned.islower() else cleaned)
    if not names:
        return None
    # Dedupe preserving sorted order for stable multi-origin labels.
    uniq = sorted(dict.fromkeys(names), key=lambda n: n.casefold())
    return "; ".join(uniq)


def country_merge_key(value: str) -> str | None:
    """Stable merge key for analysis tables (CN and China → same key)."""

    label = country_display_label(value)
    if not label:
        return None
    return _norm_key(label)


def known_country_alias_pairs_from_values(values: dict[str, int] | list[str]) -> list[dict]:
    """Report code↔name collisions present in a values map (for audits / tests)."""

    if isinstance(values, dict):
        keys = list(values.keys())
        counts = values
    else:
        keys = [str(v) for v in values]
        counts = {k: 1 for k in keys}

    by_canon: dict[str, list[tuple[str, int]]] = {}
    for key in keys:
        if _norm_key(key) in {"empty"}:
            continue
        # Only pure single-country forms (ISO or one known name) — skip multi-origin strings.
        if re.search(r"[;,/|]", key):
            continue
        if len(key.split()) >= 2 and not canonical_country_name(key):
            continue
        canon = country_display_label(key)
        if not canon or ";" in canon:
            continue
        by_canon.setdefault(canon, []).append((key, int(counts.get(key) or 0)))

    collisions: list[dict] = []
    for canon, items in sorted(by_canon.items(), key=lambda kv: kv[0].casefold()):
        forms = sorted({k for k, _ in items}, key=lambda s: s.casefold())
        if len(forms) < 2:
            continue
        total = sum(c for _, c in items)
        collisions.append(
            {
                "canonical": canon,
                "forms": forms,
                "counts": {k: int(counts.get(k) or 0) for k in forms},
                "merged_count": total,
            }
        )
    return collisions
