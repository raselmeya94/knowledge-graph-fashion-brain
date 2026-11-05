"""Load only the data each Fashion Brain question needs."""

from __future__ import annotations

from fashion_brain.ask import plan_routes
from knowledge_graph.retrieval.news import news_question


def ensure_ask_data(platform, question: str) -> None:
    """Catalog questions use the product index. Graphs load only when the answer needs them."""

    route = plan_routes(question)[0]
    if news_question(question) or route == "news":
        platform.load_for_query(["news"])
        return
    if route in {"shared_products", "shared"}:
        # Warm whatever product/document graphs the registry currently lists — no hard-coded ids.
        graph_ids = [item.id for item in platform.kg_registry.list_graphs() if item.status == "active"]
        if "master" not in graph_ids:
            graph_ids.append("master")
        platform.load_for_query(graph_ids or None)
        return
    # Product counts/specs read Neo4j ecom_products; skip empty PLM catalog envelopes.
    if route in {"product_count", "specification", "products"}:
        return
    if not getattr(platform, "products", None) or not platform.products.envelopes:
        platform.load_catalog()
