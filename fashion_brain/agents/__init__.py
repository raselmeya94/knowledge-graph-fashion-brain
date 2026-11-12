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
from fashion_brain.agents.product_spec import (
    ProductSpecificationAgent,
    SPEC_ANCHOR_FIELDS,
    SPEC_CORE_FIELDS,
    SPEC_FIELD_ORDER,
    SPEC_IMAGE_FIELDS,
    SPEC_SUPPORT_FIELDS,
    SAMPLE_IMAGE_SYSTEM,
    SPECIFICATION_RESPONSE_SYSTEM,
    findings_are_product_specification,
)
from fashion_brain.agents.slash_commands import parse_slash_command
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
from fashion_brain.agents.entity_extract import EntityExtractAgent, ExtractedEntities
from fashion_brain.agents.text2cypher import Text2CypherAgent

__all__ = [
    "ContextDecision",
    "ConversationContextAgent",
    "EntityExtractAgent",
    "ExtractedEntities",
    "GraphChoice",
    "GraphRouterAgent",
    "Intention",
    "IntentionAgent",
    "IntentSummary",
    "ProductSpecificationAgent",
    "RESPONSE_SYSTEM",
    "SPEC_ANCHOR_FIELDS",
    "SPEC_CORE_FIELDS",
    "SPEC_FIELD_ORDER",
    "SPEC_IMAGE_FIELDS",
    "SPEC_SUPPORT_FIELDS",
    "SAMPLE_IMAGE_SYSTEM",
    "SPECIFICATION_RESPONSE_SYSTEM",
    "ResponseAgent",
    "Retrieved",
    "SAMPLE_DESIGN_SYSTEM",
    "SampleDesignAgent",
    "THINKING_SYSTEM",
    "Text2CypherAgent",
    "ThinkingBrain",
    "_specification_text",
    "build_model_provider",
    "findings_are_product_specification",
    "parse_slash_command",
    "requests_sample_design",
    "response_messages",
    "sample_design_messages",
    "sample_design_sheet",
    "sample_image_prompt",
    "ungrounded_numbers",
]
