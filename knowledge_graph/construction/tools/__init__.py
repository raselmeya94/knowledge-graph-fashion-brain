"""Individual construction tools. Each one can run without the others."""

from __future__ import annotations

from knowledge_graph.construction.tools.extract_html import ExtractHtmlTool
from knowledge_graph.construction.tools.extract_text import ExtractTextTool
from knowledge_graph.construction.tools.ingest import IngestGraphTool
from knowledge_graph.construction.tools.metadata import ExtractMetadataTool
from knowledge_graph.construction.tools.parse_json import ParseJsonTool
from knowledge_graph.construction.tools.relationships import IdentifyRelationshipsTool, ListRelationshipsTool
from knowledge_graph.construction.tools.review import (
    QueueReviewTool,
    RemoveReviewRelationshipTool,
    UpdateReviewTool,
)
from knowledge_graph.construction.tools.structure import StructureDataTool
from knowledge_graph.construction.tools.understand_image import UnderstandImageTool


def tool_registry() -> dict:
    # No OllamaTool — LLM removed from KG.
    tools = [
        ExtractTextTool(),
        ExtractHtmlTool(),
        ParseJsonTool(),
        UnderstandImageTool(),
        ExtractMetadataTool(),
        ListRelationshipsTool(),
        IdentifyRelationshipsTool(),
        StructureDataTool(),
        QueueReviewTool(),
        UpdateReviewTool(),
        RemoveReviewRelationshipTool(),
        IngestGraphTool(),
    ]
    return {tool.name: tool for tool in tools}
