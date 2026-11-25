"""Shared dataclasses for Fashion Brain agents."""

from __future__ import annotations

from dataclasses import dataclass, field

from knowledge_graph.models import EvidenceItem


@dataclass
class Intention:
    intent: str
    filters: dict = field(default_factory=dict)
    group_by: str | None = None
    unsupported: bool = False
    thinking: str = ""
    limit: int | None = None
    suggest: bool = False
    analysis: bool = False
    season_name: str | None = None
    upcoming: bool = False
    corrections: list[str] = field(default_factory=list)
    preferred_graphs: list[str] = field(default_factory=list)
    available_attributes: list[str] = field(default_factory=list)
    # Distinct facet-node count ("how many brands") — not a product group_by.
    count_entity: str | None = None
    # Structured query plan (optional; filled by QueryPlanner / Intention).
    metrics: list[str] = field(default_factory=list)
    operations: list[str] = field(default_factory=list)
    response_type: str | None = None  # concise|stats|comparison|trend|analysis_report|…
    retrieve_mode: str | None = None  # text2cypher|analytical_tool|template
    time_range: str | None = None
    # Planner shape: facet_listing|entity_frequency|category_rollup|volatility|
    # state_transition|shared_overlap — filled from live schema + linguistic shape.
    query_shape: str | None = None
    # Metadata-resolver clarification (options are full-question interpretations).
    clarify_text: str = ""
    clarify_options: list = field(default_factory=list)


@dataclass
class GraphChoice:
    graph: str | None
    template: str
    selected_graphs: list[str] = field(default_factory=list)
    use_global_kg: bool = False
    reason: str = ""
    confidence: float = 0.0
    insufficient_coverage: bool = False


@dataclass
class Retrieved:
    text: str
    evidence: list[EvidenceItem]
    caveats: list[str]
    unavailable: list[str]
    synthetic_data: bool = False
    result_id: str | None = None
    # Last working graph query — reused/adapted on contextual follow-ups.
    cypher: str | None = None
    cypher_params: dict = field(default_factory=dict)
    resolved: dict = field(default_factory=dict)
    extracted: dict = field(default_factory=dict)
    response_type: str | None = None
    retrieve_mode: str | None = None


@dataclass
class IntentSummary:
    """Structured reading of a question before any catalog query runs."""

    actual_intent: str
    text: str
    filters: dict = field(default_factory=dict)
    group_by: str | None = None
    clarification_required: bool = False
    references_prior_chat: bool = False
    resolved_terms: list[str] = field(default_factory=list)


@dataclass
class ContextDecision:
    """Agent 3 output: whether to reuse, refine, or retrieve."""

    decision: str  # REUSE | REFINE | RETRIEVE
    is_contextual: bool = False
    refined_question: str = ""
    reuse_previous_results: bool = False
    previous_result_id: str | None = None
    additional_retrieval_required: bool = True
    active_filters: dict = field(default_factory=dict)
    # Conversation-context resolution (current ask vs history).
    intent_type: str = "new_independent_question"
    # continuation | modification | new_independent_question | partial_context_reuse
    inherited_filters: dict = field(default_factory=dict)
    removed_filters: dict = field(default_factory=dict)
    inheritance_reason: str = ""
    needs_clarification: bool = False


@dataclass
class ClarificationState:
    """Progressive clarification session (persisted on conversation_state['clarification'])."""

    pending_slot: str | None = None  # entity|scope_or_direction|analysis_depth|focused_dimension|final_confirm
    confirmed: dict = field(default_factory=dict)  # filters, filter_trail, entity, analysis_depth, …
    offered_options: list[str] = field(default_factory=list)
    turn_count: int = 0
    original_question: str = ""

    def to_dict(self) -> dict:
        return {
            "pending_slot": self.pending_slot,
            "confirmed": dict(self.confirmed or {}),
            "offered_options": list(self.offered_options or []),
            "turn_count": int(self.turn_count or 0),
            "original_question": self.original_question or "",
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> ClarificationState | None:
        if not data or not isinstance(data, dict):
            return None
        return cls(
            pending_slot=data.get("pending_slot"),
            confirmed=dict(data.get("confirmed") or {}),
            offered_options=list(data.get("offered_options") or []),
            turn_count=int(data.get("turn_count") or 0),
            original_question=str(data.get("original_question") or ""),
        )


@dataclass
class ValidatedIntent:
    """Structured handoff from Query Understanding when the ask is ready to execute."""

    source: str | None = None
    entity: str | None = None
    filters: dict = field(default_factory=dict)
    objective: str = ""
    metrics: list[str] = field(default_factory=list)
    group_by: str | None = None
    comparison: list[str] = field(default_factory=list)
    output: str | None = None
    analysis_depth: str = "unset"  # full|focused|unset
    available_dimensions: list[str] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)
    effective_question: str = ""

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "entity": self.entity,
            "filters": dict(self.filters or {}),
            "objective": self.objective or "",
            "metrics": list(self.metrics or []),
            "group_by": self.group_by,
            "comparison": list(self.comparison or []),
            "output": self.output,
            "analysis_depth": self.analysis_depth or "unset",
            "available_dimensions": list(self.available_dimensions or []),
            "ambiguities": list(self.ambiguities or []),
            "effective_question": self.effective_question or "",
        }


# Strict metadata-grounded filter validation (execution gate).
FilterValidationStatus = str  # VALID|NOT_FOUND|UNKNOWN|AMBIGUOUS|UNSUPPORTED


@dataclass
class ValidatedFilter:
    """One explicit user filter with deterministic validation provenance."""

    original_text: str
    field: str
    requested_value: str
    normalized_value: str | None = None
    status: FilterValidationStatus = "UNKNOWN"
    confidence: float = 0.0
    evidence: str = ""
    candidates: list[str] = field(default_factory=list)
    source_graph: str | None = None

    def to_dict(self) -> dict:
        return {
            "original_text": self.original_text,
            "field": self.field,
            "requested_value": self.requested_value,
            "normalized_value": self.normalized_value,
            "status": self.status,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "candidates": list(self.candidates or []),
            "source_graph": self.source_graph,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> ValidatedFilter | None:
        if not data or not isinstance(data, dict):
            return None
        return cls(
            original_text=str(data.get("original_text") or ""),
            field=str(data.get("field") or ""),
            requested_value=str(data.get("requested_value") or ""),
            normalized_value=(
                None
                if data.get("normalized_value") in (None, "")
                else str(data.get("normalized_value"))
            ),
            status=str(data.get("status") or "UNKNOWN"),
            confidence=float(data.get("confidence") or 0.0),
            evidence=str(data.get("evidence") or ""),
            candidates=list(data.get("candidates") or []),
            source_graph=(
                None if data.get("source_graph") in (None, "") else str(data.get("source_graph"))
            ),
        )


@dataclass
class ClarificationItem:
    """One unresolved filter that must be confirmed before execution."""

    field: str
    requested_value: str
    status: FilterValidationStatus
    question: str
    candidates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "requested_value": self.requested_value,
            "status": self.status,
            "question": self.question,
            "candidates": list(self.candidates or []),
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> ClarificationItem | None:
        if not data or not isinstance(data, dict):
            return None
        return cls(
            field=str(data.get("field") or ""),
            requested_value=str(data.get("requested_value") or ""),
            status=str(data.get("status") or "UNKNOWN"),
            question=str(data.get("question") or ""),
            candidates=list(data.get("candidates") or []),
        )


@dataclass
class UnderstandingPlan:
    """Metadata-grounded understanding with an execution gate.

    LLM may propose filters; code alone decides VALID vs clarify vs block.
    """

    intent: str = ""
    source: str | None = None
    filters: list[ValidatedFilter] = field(default_factory=list)
    group_by: str | None = None
    limit: int | None = None
    clarification_required: bool = False
    clarification_items: list[ClarificationItem] = field(default_factory=list)
    clarification_text: str = ""
    execution_allowed: bool = False
    effective_question: str = ""

    def valid_filters_dict(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for item in self.filters:
            if item.status != "VALID":
                continue
            value = item.normalized_value or item.requested_value
            if value:
                out[item.field] = value
        return out

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "source": self.source,
            "filters": [f.to_dict() for f in self.filters],
            "group_by": self.group_by,
            "limit": self.limit,
            "clarification_required": self.clarification_required,
            "clarification_items": [c.to_dict() for c in self.clarification_items],
            "clarification_text": self.clarification_text,
            "execution_allowed": self.execution_allowed,
            "effective_question": self.effective_question,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> UnderstandingPlan | None:
        if not data or not isinstance(data, dict):
            return None
        filters = [
            vf
            for raw in (data.get("filters") or [])
            if (vf := ValidatedFilter.from_dict(raw)) is not None
        ]
        items = [
            ci
            for raw in (data.get("clarification_items") or [])
            if (ci := ClarificationItem.from_dict(raw)) is not None
        ]
        return cls(
            intent=str(data.get("intent") or ""),
            source=(None if data.get("source") in (None, "") else str(data.get("source"))),
            filters=filters,
            group_by=(None if data.get("group_by") in (None, "") else str(data.get("group_by"))),
            limit=(int(data["limit"]) if data.get("limit") not in (None, "") else None),
            clarification_required=bool(data.get("clarification_required")),
            clarification_items=items,
            clarification_text=str(data.get("clarification_text") or ""),
            execution_allowed=bool(data.get("execution_allowed")),
            effective_question=str(data.get("effective_question") or ""),
        )


@dataclass
class ClarifyResult:
    """Ask one metadata-grounded follow-up; do not retrieve yet."""

    question: str
    state: ClarificationState
    slot: str


@dataclass
class ReadyResult:
    """Intent is clear enough to hand off to Intention / retrieval."""

    intent: ValidatedIntent
    state: ClarificationState | None = None
