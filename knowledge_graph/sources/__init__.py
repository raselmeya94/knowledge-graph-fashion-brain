"""Source lifecycle: register, schema discovery, and unified sync."""

from knowledge_graph.sources.schema import source_schema
from knowledge_graph.sources.sync import sync_source

__all__ = ["source_schema", "sync_source"]
