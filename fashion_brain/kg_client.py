"""Compatibility shim — prefer fashion_brain.execution.kg_client."""

from fashion_brain.execution.kg_client import *  # noqa: F403
from fashion_brain.execution.kg_client import (
    kg_base_url,
    kg_request,
    list_graphs,
    list_registry,
    metadata_values,
    ping_kg,
    reload_kg_graphs,
    retrieve_remote,
    uses_kg_api,
)

__all__ = [
    "kg_base_url",
    "kg_request",
    "list_graphs",
    "list_registry",
    "metadata_values",
    "ping_kg",
    "reload_kg_graphs",
    "retrieve_remote",
    "uses_kg_api",
]
