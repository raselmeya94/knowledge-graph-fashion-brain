"""Entity linking: user terms → real Neo4j facet node ids."""

from knowledge_graph.linking.entity_linker import (
    EntityLinker,
    LinkHit,
    LinkResult,
    link_entities,
    list_live_brands,
    list_live_values,
)

__all__ = [
    "EntityLinker",
    "LinkHit",
    "LinkResult",
    "link_entities",
    "list_live_brands",
    "list_live_values",
]
