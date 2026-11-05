"""Record-level access checks used by retrieval, not only by the API layer."""

from __future__ import annotations

from pydantic import BaseModel, Field

CLEARANCE_RANK = {"public": 0, "internal": 1, "confidential": 2}


class Principal(BaseModel):
    subject: str = "anonymous"
    clearance: str = "public"
    roles: list[str] = Field(default_factory=list)

    def rank(self) -> int:
        if self.clearance not in CLEARANCE_RANK:
            raise ValueError(f"Unknown clearance '{self.clearance}'.")
        return CLEARANCE_RANK[self.clearance]


def allows(principal: Principal, classification: str | None) -> bool:
    """Fail closed when a record has no classification or an unknown level."""

    if not classification or classification not in CLEARANCE_RANK:
        return False
    return principal.rank() >= CLEARANCE_RANK[classification]
