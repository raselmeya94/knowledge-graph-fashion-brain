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
