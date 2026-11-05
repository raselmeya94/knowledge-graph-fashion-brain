"""Fashion Brain specialized agents (plan Agents 1–5 + satellites)."""

from fashion_brain.llm import build_model_provider

from fashion_brain.agents.contracts import (
    ContextDecision,
    GraphChoice,
    Intention,
    IntentSummary,
    Retrieved,
)
from fashion_brain.agents.conversation_context import ConversationContextAgent
from fashion_brain.agents.product_spec import ProductSpecificationAgent
from fashion_brain.agents._impl import (
    IntentionAgent,
    GraphRouterAgent,
    RESPONSE_SYSTEM,
    ResponseAgent,
    SAMPLE_DESIGN_SYSTEM,
    SampleDesignAgent,
    THINKING_SYSTEM,
    ThinkingBrain,
    _specification_text,
    requests_sample_design,
    response_messages,
    sample_design_messages,
    sample_design_sheet,
    sample_image_prompt,
    ungrounded_numbers,
)

__all__ = [
    "ContextDecision",
    "ConversationContextAgent",
    "GraphChoice",
    "GraphRouterAgent",
    "Intention",
    "IntentionAgent",
    "IntentSummary",
    "ProductSpecificationAgent",
    "RESPONSE_SYSTEM",
    "ResponseAgent",
    "Retrieved",
    "SAMPLE_DESIGN_SYSTEM",
    "SampleDesignAgent",
    "THINKING_SYSTEM",
    "ThinkingBrain",
    "_specification_text",
    "build_model_provider",
    "requests_sample_design",
    "response_messages",
    "sample_design_messages",
    "sample_design_sheet",
    "sample_image_prompt",
    "ungrounded_numbers",
]
