"""Agent 4: product specification generation (conditional)."""

from __future__ import annotations

from fashion_brain.agents.contracts import Intention, Retrieved


class ProductSpecificationAgent:
    """Runs only when the user asks for a product specification / suggested style."""

    def required(self, intention: Intention) -> bool:
        return bool(intention.suggest) or intention.intent == "specification"

    def enrich(self, retrieved: Retrieved, intention: Intention) -> Retrieved:
        """Mark specification provenance; KG retrieve already embeds the sheet when suggest=True."""

        if not self.required(intention):
            return retrieved
        caveats = list(retrieved.caveats or [])
        note = "Specification fields marked retrieved come from KG product data; any AI-suggested gaps are listed separately when present."
        if note not in caveats:
            caveats.append(note)
        return Retrieved(
            text=retrieved.text,
            evidence=list(retrieved.evidence),
            caveats=caveats,
            unavailable=list(retrieved.unavailable),
            synthetic_data=retrieved.synthetic_data,
            result_id=retrieved.result_id,
        )
