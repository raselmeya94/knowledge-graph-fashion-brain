"""Split a news PDF into articles and keep trade facts on allowlisted edges."""

from __future__ import annotations

import re

from knowledge_graph.ontology import Ontology

_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_MONTH = (
    r"January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
_GEOS = {
    "european union": ("EU", "Region"),
    "eu": ("EU", "Region"),
    "united states": ("United States", "Country"),
    "united kingdom": ("United Kingdom", "Country"),
    "usa": ("United States", "Country"),
    "uk": ("United Kingdom", "Country"),
    "us": ("United States", "Country"),
    "bangladesh": ("Bangladesh", "Country"),
    "pakistan": ("Pakistan", "Country"),
    "turkey": ("Turkey", "Country"),
    "china": ("China", "Country"),
    "india": ("India", "Country"),
    "vietnam": ("Vietnam", "Country"),
    "cambodia": ("Cambodia", "Country"),
    "indonesia": ("Indonesia", "Country"),
    "myanmar": ("Myanmar", "Country"),
    "sri lanka": ("Sri Lanka", "Country"),
    "morocco": ("Morocco", "Country"),
    "tunisia": ("Tunisia", "Country"),
    "egypt": ("Egypt", "Country"),
    "mexico": ("Mexico", "Country"),
    "germany": ("Germany", "Country"),
    "france": ("France", "Country"),
    "italy": ("Italy", "Country"),
    "spain": ("Spain", "Country"),
    "japan": ("Japan", "Country"),
    "south korea": ("South Korea", "Country"),
}
_ALIASES = {
    "eu": ("eu", "european union"),
    "european union": ("eu", "european union"),
    "united states": ("united states", "usa", "us"),
    "united kingdom": ("united kingdom", "uk"),
}
_PRODUCTS = ("denim", "jeans", "apparel", "garment", "garments", "textile", "textiles", "cotton", "fabric", "yarn", "knitwear", "clothing")
_GARMENT = ("denim", "garment", "apparel", "textile", "cotton", "fabric", "yarn", "knit", "clothing", "fashion")
_OUTSIDE = ("lng", "liquefied natural gas", "cargo", "creative industr", "artisan", "soft power")
_ORDINALS = {"largest": 1, "top": 1, "first": 1, "second": 2, "third": 3, "fourth": 4}
_RANK_WORDS = {1: ("1", "first", "largest", "top"), 2: ("2", "second"), 3: ("3", "third"), 4: ("4", "fourth")}
_MONEY = re.compile(r"[€$£]\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|m|bn))?", re.IGNORECASE)
_PERIOD = re.compile(rf"\b({_MONTH})\s*[-–]\s*({_MONTH})\s+(20\d{{2}})\b", re.IGNORECASE)
_PLACEHOLDER = {"location", "unknown", "entity", "period"}
_QUANTITIES = ("value", "share_pct", "previous_share", "yoy_pct", "amount", "pct")


def segment_articles(text: str) -> list[dict]:
    """Separate a headline list from the stories that follow it."""

    blocks = _blocks(text)
    titles: list[str] = []
    body: list[str] = []
    for block in blocks:
        if not body and _is_title(block):
            titles.append(block)
        else:
            body.append(block)
    if len(titles) < 2:
        blob = " ".join(blocks) if blocks else _flatten(text)
        return [{"title": titles[0] if titles else blob[:180], "text": blob, "domain": domain_of(blob)}]
    groups = {index: [] for index in range(len(titles))}
    current = 0
    for block in body:
        scores = [_overlap(block, title) for title in titles]
        best = max(range(len(scores)), key=lambda index: scores[index])
        if scores[best] >= 2:
            current = best
        groups[current].append(block)
    articles = []
    for index, title in enumerate(titles):
        blob = " ".join([title, *groups[index]]).strip()
        articles.append({"title": title, "text": blob, "domain": domain_of(blob)})
    return articles


def domain_of(text: str) -> str:
    low = text.casefold()
    garment = sum(low.count(word) for word in _GARMENT)
    outside = sum(low.count(word) for word in _OUTSIDE)
    if garment and garment >= outside:
        return "garment"
    return "out_of_domain"


def extract_trade_facts(title: str, text: str) -> list[dict]:
    """One EXPORTS_TO or RANKED_IN edge per exporter, with the figures on the edge."""

    flat = _flatten(text)
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", flat) if part.strip()]
    product = next((word for word in _PRODUCTS if re.search(rf"\b{word}\b", flat, re.IGNORECASE)), "")
    period = _period(flat)
    found: list[dict] = []
    context: dict[str, tuple[str, str]] = {}
    for index, sentence in enumerate(sentences):
        if re.search(r"\b(supplier|exports?\s+to)\b", sentence, re.IGNORECASE) and _MONEY.search(sentence):
            geos = _geos(sentence)
            if len(geos) >= 2:
                exporter, destination = geos[0], geos[1]
                context["destination"] = destination
                value = _compact_money(_MONEY.search(sentence).group(0))
                share = _capture(r"capturing\s+(\d+(?:\.\d+)?)%", sentence)
                evidence = sentence
                found.append(
                    _edge(
                        "EXPORTS_TO",
                        exporter[0],
                        exporter[1],
                        destination[0],
                        destination[1],
                        evidence,
                        _props(product=product, value=value, share_pct=_num(share), period=period),
                    )
                )
                rank = _ordinal(sentence)
                if rank:
                    found.append(
                        _edge(
                            "RANKED_IN",
                            exporter[0],
                            "Country",
                            destination[0],
                            "Market",
                            evidence,
                            _props(rank=rank, product=product, period=period),
                        )
                    )
        for match in re.finditer(
            rf"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)\s+ranked\s+({'|'.join(_ORDINALS)})\b",
            sentence,
        ):
            exporter = _lookup_geo(match.group(1))
            money = _MONEY.search(sentence[match.start() : match.start() + 80])
            destination = context.get("destination")
            if exporter and money and destination:
                evidence = _window(sentences, index, (exporter[0], destination[0]))
                found.append(
                    _edge(
                        "EXPORTS_TO",
                        exporter[0],
                        exporter[1],
                        destination[0],
                        destination[1],
                        evidence,
                        _props(product=product, value=_compact_money(money.group(0)), period=period),
                    )
                )
                found.append(
                    _edge(
                        "RANKED_IN",
                        exporter[0],
                        "Country",
                        destination[0],
                        "Market",
                        evidence,
                        _props(rank=_ORDINALS[match.group(2).casefold()], product=product, period=period),
                    )
                )
        for match in re.finditer(
            r"followed by\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)\s+with\s+([€$£]\s*[\d,.]+(?:\s*(?:million|billion))?)",
            sentence,
            re.IGNORECASE,
        ):
            exporter = _lookup_geo(match.group(1))
            destination = context.get("destination")
            if exporter and destination:
                evidence = _window(sentences, index, (exporter[0], destination[0]))
                rank = 1 + sum(1 for item in found if item["type"] == "RANKED_IN")
                found.append(
                    _edge(
                        "EXPORTS_TO",
                        exporter[0],
                        exporter[1],
                        destination[0],
                        destination[1],
                        evidence,
                        _props(product=product, value=_compact_money(match.group(2)), period=period),
                    )
                )
                found.append(
                    _edge(
                        "RANKED_IN",
                        exporter[0],
                        "Country",
                        destination[0],
                        "Market",
                        evidence,
                        _props(rank=rank, product=product, period=period),
                    )
                )
    _attach_change(found, sentences)
    cite = re.search(r"according to\s+([A-Z][\w&.'-]+)", flat)
    if cite:
        found.append(
            _edge(
                "CITES_DATA_FROM",
                title,
                "Document",
                cite.group(1),
                "Organization",
                cite.group(0),
                {},
            )
        )
    return found


def normalize_news_relationship(item: dict) -> dict:
    if item.get("type") == "RANKED":
        item["type"] = "RANKED_IN"
    for side in ("start", "end"):
        name = str(item.get(f"{side}_name") or "")
        geo = _GEOS.get(name.casefold())
        if not geo:
            continue
        if item.get("type") in {"RANKED_IN", "HAS_MARKET_SHARE"} and side == "end":
            item[f"{side}_name"] = geo[0]
            item[f"{side}_type"] = "Market"
        else:
            item[f"{side}_name"], item[f"{side}_type"] = geo
    return item


def news_relationship_label(item: dict) -> str:
    return f"{item.get('type') or ''} {item.get('start_name') or ''} -> {item.get('end_name') or ''}"


def ontology_gap_warning(item: dict, ontology: Ontology) -> str | None:
    """Warn when an evidenced edge uses a type or pair not yet on the allowlist."""

    rel_type = str(item.get("type") or "")
    start_type = str(item.get("start_type") or "")
    end_type = str(item.get("end_type") or "")
    allowed = ontology.relationship_allowlist.get(rel_type, [])
    if [start_type, end_type] in allowed:
        return None
    label = news_relationship_label(item)
    return (
        f"Needs review {label}: {start_type} -> {end_type} is not allowlisted yet; "
        "approve and commit to add it to the ontology for future use."
    )


def reject_news_relationship(item: dict, ontology: Ontology) -> str | None:
    """Return a hard reject when a news edge fails entity or evidence checks.

    Allowlist gaps are not hard rejects. Those stay as review proposals so a person
    can approve them and extend the ontology.
    """

    rel_type = str(item.get("type") or "")
    start_type = str(item.get("start_type") or "")
    end_type = str(item.get("end_type") or "")
    start_name = str(item.get("start_name") or "")
    end_name = str(item.get("end_name") or "")
    label = news_relationship_label(item)
    if start_type not in ontology.entity_types or end_type not in ontology.entity_types:
        return f"Rejected {label}: {start_type} -> {end_type} is not a news entity pair."
    if start_name.casefold() in _PLACEHOLDER or end_name.casefold() in _PLACEHOLDER or _looks_like_period(end_name):
        return f"Rejected {label}: a period or placeholder cannot be a node."
    evidence = str(item.get("evidence") or "")
    if start_type != "Document" and not _name_in(start_name, evidence):
        return f"Rejected {label}: evidence does not name {start_name}."
    if end_type != "Document" and not _name_in(end_name, evidence):
        return f"Rejected {label}: evidence does not name {end_name}."
    properties = item.get("properties") or {}
    for key in _QUANTITIES:
        value = properties.get(key)
        if value in (None, ""):
            continue
        for number in re.findall(r"\d+(?:\.\d+)?", str(value).lstrip("-")):
            if not _number_in(number, evidence):
                return f"Rejected {label}: {key} {value} is not in the evidence."
    rank = properties.get("rank")
    if rank not in (None, ""):
        words = _RANK_WORDS.get(int(rank), (str(rank),))
        followed = re.search(rf"followed by\s+{re.escape(start_name)}\b", evidence, re.IGNORECASE)
        if not followed and not any(word in evidence.casefold() for word in words):
            return f"Rejected {label}: rank {rank} is not in the evidence."
    return None


def _attach_change(found: list[dict], sentences: list[str]) -> None:
    lead = next((item for item in found if item["type"] == "EXPORTS_TO"), None)
    if lead is None:
        return
    for index, sentence in enumerate(sentences):
        change = re.search(
            r"\b(fell|declined|dropped|decreased|rose|increased|grew)\s+(\d+(?:\.\d+)?)%\s+year-on-year",
            sentence,
            re.IGNORECASE,
        )
        previous = re.search(r"market share from\s+(\d+(?:\.\d+)?)%", sentence, re.IGNORECASE)
        if change and _name_in(lead["start_name"], sentence):
            sign = -1 if change.group(1).casefold() in {"fell", "declined", "dropped", "decreased"} else 1
            lead["properties"]["yoy_pct"] = sign * float(change.group(2))
            if sentence not in lead["evidence"]:
                lead["evidence"] = f"{lead['evidence']} {sentence}"
        if previous and lead["properties"].get("share_pct") not in (None, ""):
            evidence = lead["evidence"] if sentence in lead["evidence"] else f"{lead['evidence']} {sentence}"
            found.append(
                _edge(
                    "HAS_MARKET_SHARE",
                    lead["start_name"],
                    "Country",
                    lead["end_name"],
                    "Market",
                    evidence,
                    _props(
                        product=lead["properties"].get("product"),
                        share_pct=lead["properties"].get("share_pct"),
                        previous_share=_num(previous.group(1)),
                        period=lead["properties"].get("period"),
                    ),
                )
            )


def _edge(rel_type: str, start: str, start_type: str, end: str, end_type: str, evidence: str, properties: dict) -> dict:
    return {
        "type": rel_type,
        "start_name": start,
        "start_type": start_type,
        "end_name": end,
        "end_type": end_type,
        "evidence": evidence,
        "method": "deterministic",
        "properties": properties,
    }


def _props(**values) -> dict:
    return {key: value for key, value in values.items() if value not in (None, "")}


def _blocks(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) <= 1:
        return [_flatten(text)] if text.strip() else []
    blocks: list[str] = []
    current = lines[0]
    for line in lines[1:]:
        if _continues(current, line):
            joiner = "" if current.endswith("-") else " "
            current = current[:-1] + line if current.endswith("-") else current + joiner + line
        else:
            blocks.append(current.strip())
            current = line
    blocks.append(current.strip())
    return [block for block in blocks if block]


def _continues(previous: str, line: str) -> bool:
    if previous.endswith("-"):
        return True
    if previous[-1:] in ".!?":
        return False
    if previous.endswith(",") or line[:1].isdigit() or line[:1].islower():
        return True
    return len(previous) > 90 and not _is_title(previous)


def _is_title(value: str) -> bool:
    return bool(value) and len(value) <= 160 and not re.search(r"\d", value) and value[-1] not in ".!?"


def _overlap(block: str, title: str) -> int:
    title_tokens = [token for token in re.findall(r"[a-z0-9]+", title.casefold()) if len(token) >= 4 and token not in _GEOS]
    block_tokens = set(re.findall(r"[a-z0-9]+", block.casefold()))
    return sum(1 for token in title_tokens if token in block_tokens or any(item.startswith(token[:5]) for item in block_tokens))


def _flatten(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _period(text: str) -> str:
    match = _PERIOD.search(text)
    if not match:
        return ""
    start = _month_number(match.group(1))
    end = _month_number(match.group(2))
    if not start or not end:
        return ""
    return f"{match.group(3)}-{start:02d}..{end:02d}"


def _month_number(value: str) -> int:
    name = value.casefold()
    if name in _MONTHS:
        return _MONTHS[name]
    for month, number in _MONTHS.items():
        if month.startswith(name):
            return number
    return 0


def _geos(text: str) -> list[tuple[str, str]]:
    found = []
    occupied: list[tuple[int, int]] = []
    for name in sorted(_GEOS, key=len, reverse=True):
        for match in re.finditer(rf"\b{re.escape(name)}\b", text, re.IGNORECASE):
            if any(start <= match.start() < end for start, end in occupied):
                continue
            occupied.append(match.span())
            found.append((match.start(), _GEOS[name]))
    return [item for _pos, item in sorted(found)]


def _lookup_geo(value: str) -> tuple[str, str] | None:
    return _GEOS.get(value.casefold()) or _GEOS.get(value.split()[-1].casefold())


def _ordinal(sentence: str) -> int:
    match = re.search(r"\b(largest|top|first|second|third|fourth)\b", sentence, re.IGNORECASE)
    return _ORDINALS[match.group(1).casefold()] if match else 0


def _capture(pattern: str, text: str) -> str:
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1) if match else ""


def _num(value: str):
    if not value:
        return ""
    return float(value) if "." in value else int(value)


def _compact_money(value: str) -> str:
    match = re.match(r"\s*([€$£])\s*(\d[\d,]*(?:\.\d+)?)(?:\s*(million|billion|m|bn))?", value.strip(), re.IGNORECASE)
    if not match:
        return re.sub(r"\s+", "", value)
    suffix = {"million": "M", "billion": "B", "m": "M", "bn": "B"}.get((match.group(3) or "").casefold(), "")
    return f"{match.group(1)}{match.group(2).replace(',', '')}{suffix}"


def _window(sentences: list[str], index: int, required: tuple[str, ...]) -> str:
    evidence = sentences[index]
    start = index
    while start > 0 and not all(_name_in(item, evidence) or _number_in(item, evidence) for item in required):
        start -= 1
        evidence = f"{sentences[start]} {evidence}"
        if index - start >= 2:
            break
    return evidence


def _name_in(name: str, text: str) -> bool:
    low = text.casefold()
    options = _ALIASES.get(name.casefold(), (name.casefold(),))
    return any(option in low for option in options)


def _number_in(number: str, text: str) -> bool:
    if number in text:
        return True
    if "." in number:
        whole, fraction = number.split(".", 1)
        return re.search(rf"{re.escape(whole)}\.{re.escape(fraction)}0*\b", text) is not None
    return False


def _looks_like_period(name: str) -> bool:
    return _PERIOD.fullmatch(name.strip()) is not None
