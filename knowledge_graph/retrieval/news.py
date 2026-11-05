"""Search document/custom graphs for a question. Product facts stay on product graphs."""

from __future__ import annotations

import re

from knowledge_graph.access import Principal, allows

_CUES = {
    "EXPORTS_TO": ("export", "exports", "exported", "exporter"),
    "IMPORTS_FROM": ("import", "imports", "imported", "importer"),
    "RANKED_IN": ("rank", "ranked", "ranking", "largest"),
    "HAS_MARKET_SHARE": ("market share",),
    "CITES_DATA_FROM": ("eurostat", "according to", "data source"),
}
_FACT_KEYS = ("product", "value", "share_pct", "previous_share", "yoy_pct", "period", "rank", "amount")
_ALIAS = {"european union": "eu", "eu": "eu"}


def news_question(text: str) -> bool:
    """Trade and news questions. Catalog counts and material lists stay on the product graph."""

    low = text.casefold()
    if any(phrase in low for phrase in ("how many", "catalog", "catalogue", "demo product", "our product")):
        return False
    if "share" in low and "material" in low:
        return False
    if any(cue in low for cues in _CUES.values() for cue in cues):
        return True
    return "denim" in low and any(token in _tokens(low) for token in ("eu", "bangladesh", "pakistan", "turkey"))


def search_news(platform, principal: Principal, question: str, *, limit: int = 12) -> list[dict]:
    """Back-compat: search the news graph only."""

    return search_document_graphs(platform, principal, question, graph_ids=["news"], limit=limit)


def search_document_graphs(
    platform,
    principal: Principal,
    question: str,
    *,
    graph_ids: list[str] | None = None,
    limit: int = 12,
) -> list[dict]:
    """Edges from document/custom graphs whose names, product, or relation match the question."""

    targets = [item for item in (graph_ids or ["news"]) if item]
    if not targets:
        return []
    text = question.casefold()
    tokens = _tokens(text)
    wanted = {rel for rel, cues in _CUES.items() if any(cue in text for cue in cues)}
    ranked: list[tuple[int, dict]] = []
    hidden = False
    for graph_id in targets:
        for edge in platform.graph.find_edges(graph=graph_id, include_deleted=False):
            if edge.type == "RESOLVES_TO":
                continue
            start = platform.graph.get_node(edge.start_id, include_deleted=False)
            end = platform.graph.get_node(edge.end_id, include_deleted=False)
            if start is None or end is None:
                continue
            classification = edge.properties.get("prov_classification") or start.properties.get("prov_classification")
            if classification and not allows(principal, str(classification)):
                hidden = True
                continue
            start_name = str(start.properties.get("name") or "")
            end_name = str(end.properties.get("name") or "")
            product = str(edge.properties.get("product") or "")
            name_hit = _named(start, text, tokens) or _named(end, text, tokens)
            product_hit = bool(product) and product.casefold() in text
            type_hit = edge.type in wanted
            # Loose match: any question token in names when no cue matched.
            loose = False
            if not name_hit and not product_hit and not type_hit and tokens:
                blob = f"{start_name} {end_name} {product} {edge.type}".casefold()
                loose = any(token in blob for token in tokens if len(token) > 2)
            if not name_hit and not product_hit and not type_hit and not loose:
                continue
            score = int(name_hit) + int(product_hit) + (2 if type_hit else 0) + (1 if loose else 0)
            ranked.append(
                (
                    score,
                    {
                        "edge": edge,
                        "start": start,
                        "end": end,
                        "start_name": start_name,
                        "end_name": end_name,
                        "line": _line(edge, start_name, end_name, graph_id=graph_id),
                        "graph": graph_id,
                        "hidden_only": False,
                    },
                )
            )
    ranked.sort(key=lambda item: (-item[0], item[1]["line"]))
    if ranked:
        return [row for _score, row in ranked[:limit]]
    if hidden:
        return [{"hidden_only": True, "line": "Document records are not visible at the current clearance.", "graph": targets[0]}]
    return []


def _named(node, text: str, tokens: set[str]) -> bool:
    if node is None or "Document" in node.labels or node.properties.get("entity_type") == "Document":
        return False
    name = str(node.properties.get("name") or "").casefold()
    if not name:
        return False
    if name in _ALIAS:
        return _ALIAS[name] in tokens or name in text
    return name in text


def _line(edge, start_name: str, end_name: str, *, graph_id: str = "news") -> str:
    details = []
    for key in _FACT_KEYS:
        value = edge.properties.get(key)
        if value not in (None, ""):
            details.append(f"{key} {value}")
    suffix = f" ({', '.join(details)})" if details else ""
    prefix = f"[{graph_id}] " if graph_id and graph_id != "news" else ""
    return f"{prefix}{start_name} {edge.type} {end_name}{suffix}."


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text))
