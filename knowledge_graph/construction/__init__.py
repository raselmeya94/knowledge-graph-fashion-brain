"""Document construction pipeline: extract a source, propose relationships, review, then ingest."""

from knowledge_graph.construction.identity import register_documents_source

register_documents_source()

__all__ = ["register_documents_source"]
