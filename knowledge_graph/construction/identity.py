"""Graph identity for documents built by the construction pipeline."""

from __future__ import annotations

from knowledge_graph.graph.build import build_envelopes, register_builder
from knowledge_graph.ids import register_source_identity


NEWS_KINDS = frozenset({"pdf", "doc", "docx", "text"})


def build_documents_graph(envelopes, repo, ontology) -> list[str]:
    return build_envelopes(envelopes, repo, ontology, "documents")


def build_news_graph(envelopes, repo, ontology) -> list[str]:
    return build_envelopes(envelopes, repo, ontology, "news")


def register_documents_source() -> None:
    register_source_identity("documents", "kg", "document", "Document")
    register_builder("documents", build_documents_graph)
    register_source_identity("news", "news", "document", "Document")
    register_builder("news", build_news_graph)
