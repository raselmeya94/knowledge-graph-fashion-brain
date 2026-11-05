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

    for index, part in enumerate(parts, start=1):
        if part.text.strip():
            texts.append(f"[Graph result {index}]\n{part.text.strip()}")
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

    caveats.append("Results from multiple knowledge graphs are listed separately; products are not merged by name similarity.")
    return Retrieved(
        text="\n\n".join(texts),
        evidence=evidence,
        caveats=caveats,
        unavailable=unavailable,
        synthetic_data=synthetic,
        result_id=result_id,
    )
