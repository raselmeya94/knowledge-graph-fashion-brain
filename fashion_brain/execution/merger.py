"""Result merger for multi-graph retrieval payloads."""

from __future__ import annotations

from fashion_brain.agents.contracts import Retrieved
from knowledge_graph.models import EvidenceItem


def merge_retrieved(parts: list[Retrieved], *, result_id: str | None = None) -> Retrieved:
    """Combine multi-KG evidence while preserving source provenance.

    Products from different graphs are never collapsed into one entity.
    """

    if not parts:
        return Retrieved(text="", evidence=[], caveats=[], unavailable=[], result_id=result_id)
    if len(parts) == 1:
        only = parts[0]
        return Retrieved(
            text=only.text,
            evidence=list(only.evidence),
            caveats=list(only.caveats),
            unavailable=list(only.unavailable),
            synthetic_data=only.synthetic_data,
            result_id=result_id or only.result_id,
        )

    texts: list[str] = []
    evidence: list[EvidenceItem] = []
    caveats: list[str] = []
    unavailable: list[str] = []
    synthetic = False
    seen_evidence: set[str] = set()

    nonempty_parts = [part for part in parts if part.text.strip() and not _looks_empty_block(part.text)]
    skipped_empty = len(parts) - len(nonempty_parts)
    # Keep at least one part even if all look empty (honest empty answer).
    use_parts = nonempty_parts or parts

    for index, part in enumerate(use_parts, start=1):
        if part.text.strip():
            texts.append(f"### {_source_label(part, index)}\n{part.text.strip()}")
        for item in part.evidence:
            key = f"{item.source_id}:{item.record_id}:{item.title}"
            if key in seen_evidence:
                continue
            seen_evidence.add(key)
            evidence.append(item)
        for caveat in part.caveats:
            if caveat not in caveats:
                caveats.append(caveat)
        for item in part.unavailable:
            if item not in unavailable:
                unavailable.append(item)
        synthetic = synthetic or bool(part.synthetic_data)

    caveats.append(
        "Results from multiple product catalogs are listed separately; products are not merged by name similarity."
    )
    if skipped_empty:
        caveats.append(
            f"{skipped_empty} catalog source(s) returned no rows for this ask and were omitted from the tables."
        )
    return Retrieved(
        text="\n\n".join(texts),
        evidence=evidence,
        caveats=caveats,
        unavailable=unavailable,
        synthetic_data=synthetic,
        result_id=result_id,
    )


def _source_label(part: Retrieved, index: int) -> str:
    from fashion_brain.agents.answer_formation import source_heading_label

    graph_ids = {
        str(item.graph or item.source_id).strip()
        for item in part.evidence
        if str(item.graph or item.source_id).strip()
    }
    if len(graph_ids) == 1:
        return source_heading_label(graph_ids.pop())
    # Prefer resolved graph from part.resolved when evidence is empty (zero hits).
    resolved = getattr(part, "resolved", None) or {}
    for key in ("graph_id", "graph", "source_id"):
        gid = str(resolved.get(key) or "").strip()
        if gid:
            return source_heading_label(gid)
    # Avoid "Source 1/2" leaking into user answers — use a neutral catalog label.
    return f"Catalog {index}"


def _looks_empty_block(text: str) -> bool:
    """True for zero-count / dash-only secondary catalog tables."""

    import re

    body = (text or "").strip().casefold()
    if not body:
        return True
    if "0 rows" in body or "no rows" in body:
        return True
    # Avg-price style tables with only em-dashes / zeros and no positive product counts.
    if re.search(r"\|\s*(—|-|n/?a)\s*\|", body) and not re.search(r"\|\s*[1-9]\d*", body):
        return True
    if re.search(r"\|\s*0\s*\|", body) and not re.search(r"\|\s*[1-9]\d*", body):
        return True
    return False
