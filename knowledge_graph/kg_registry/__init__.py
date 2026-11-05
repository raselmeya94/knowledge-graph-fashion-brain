"""Named ontology and graph registry for the flexible multi-graph API."""

from knowledge_graph.kg_registry.models import GraphCreate, GraphRecord, OntologyCreate, OntologyRecord, OntologyReplace
from knowledge_graph.kg_registry.store import KgRegistryStore, default_graph_for_kind, ensure_graph_identity

__all__ = [
    "GraphCreate",
    "GraphRecord",
    "KgRegistryStore",
    "OntologyCreate",
    "OntologyRecord",
    "OntologyReplace",
    "default_graph_for_kind",
    "ensure_graph_identity",
]
