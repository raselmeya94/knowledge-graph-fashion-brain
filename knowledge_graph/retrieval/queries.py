"""Product graph queries used by the knowledge graph and the chatbot."""

from __future__ import annotations

from knowledge_graph.access import Principal, allows
from knowledge_graph.pipeline.service import Platform


def _product_graph_ids(platform: Platform) -> list[str]:
    """Active product graphs from the registry, with legacy own_products fallback."""

    try:
        from knowledge_graph.retrieval.ecom import registry_product_graphs

        ids = registry_product_graphs(platform)
    except Exception:
        ids = []
    if not ids:
        ids = ["own_products"]
    elif "own_products" not in ids:
        # Keep legacy graph if it still has nodes.
        try:
            if platform.graph.find_nodes(graph="own_products", label="Product", include_deleted=False):
                ids = [*ids, "own_products"]
        except Exception:
            pass
    return ids


def product_materials(platform: Platform, principal: Principal, graph_ids: list[str] | None = None) -> list[dict]:
    rows = []
    for graph_id in graph_ids or _product_graph_ids(platform):
        products = platform.graph.find_nodes(graph=graph_id, label="Product", include_deleted=False)
        for product in products:
            classification = product.properties.get("classification") or product.properties.get("prov_classification") or "internal"
            if not allows(principal, classification):
                continue
            for edge in platform.graph.find_edges(start_id=product.id, rel_type="USES_MATERIAL", include_deleted=False):
                material = platform.graph.get_node(edge.end_id, include_deleted=False)
                if material is None:
                    continue
                resolutions = platform.graph.find_edges(start_id=material.id, rel_type="RESOLVES_TO", include_deleted=False)
                verified = next(
                    (item for item in resolutions if item.properties.get("prov_verification_status") == "verified"),
                    None,
                )
                review = [item for item in resolutions if item.properties.get("prov_verification_status") == "review"]
                rows.append(
                    {
                        "product": product,
                        "edge": edge,
                        "material": material,
                        "graph_id": graph_id,
                        "canonical_id": verified.end_id if verified else None,
                        "material_key": verified.end_id if verified else material.id,
                        "review": bool(review) and verified is None,
                    }
                )
    return rows


def products_sharing_material(platform: Platform, principal: Principal) -> list[dict]:
    """Materials used by more than one visible product.

    Prefer verified RESOLVES_TO canonical ids when present. Otherwise group by the
    shared Material node id (ecom compositions already reuse the same Material node).
    """

    groups: dict[str, dict] = {}
    for row in product_materials(platform, principal):
        # Skip only when an explicit review queue blocks a verified merge.
        if row["review"]:
            continue
        key = row["material_key"]
        if not key:
            continue
        group = groups.setdefault(
            key,
            {
                "canonical_id": row["canonical_id"] or key,
                "material": row["material"].properties.get("name") or key,
                "products": [],
                "verified": bool(row["canonical_id"]),
                "graph_id": row.get("graph_id"),
            },
        )
        product_id = (
            row["product"].properties.get("prov_record_id")
            or row["product"].properties.get("product_id")
            or row["product"].id
        )
        if product_id not in group["products"]:
            group["products"].append(str(product_id))
    shared = [group for group in groups.values() if len(group["products"]) > 1]
    shared.sort(key=lambda item: (-len(item["products"]), str(item.get("material") or "")))
    return shared
