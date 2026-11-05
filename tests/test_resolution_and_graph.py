import pytest

from knowledge_graph.errors import RelationshipValidationError
from knowledge_graph.graph.memory import MemoryGraphRepository
from knowledge_graph.ids import primary_id, source_entity_id
from knowledge_graph.models import GraphEdge, GraphNode, ReviewDecision
from knowledge_graph.ontology import load_ontology
from knowledge_graph.resolution.matcher import match_entity
from tests.conftest import ROOT


@pytest.fixture()
def ontology():
    from knowledge_graph.config import DATA_ROOT

    return load_ontology(DATA_ROOT / "schemas" / "product-knowledge-ontology.json")


def test_exact_alias_verifies_and_similar_name_stays_in_review(ontology):
    verified = match_entity(
        name="organic cotton",
        entity_type="Material",
        external_id=None,
        ontology=ontology,
        decisions=[],
    )
    assert verified[0].status == "verified"
    assert verified[0].method == "curated_alias"
    assert verified[0].canonical_id == "fashion:material:cotton"
    assert verified[0].similarity is None

    review = match_entity(
        name="Cotton Blend",
        entity_type="Material",
        external_id=None,
        ontology=ontology,
        decisions=[],
    )
    assert len(review) == 1
    assert review[0].status == "review"
    assert review[0].canonical_id == "fashion:material:cotton"
    assert "prov_confidence" not in review[0].model_dump()

    supplier = match_entity(
        name="Northwind Textile Group",
        entity_type="Supplier",
        external_id=None,
        ontology=ontology,
        decisions=[],
    )
    assert supplier[0].status == "review"
    assert supplier[0].canonical_id == "fashion:supplier:northwind-mills"

    other_type = match_entity(
        name="Northwind Mills",
        entity_type="Company",
        external_id="SUP-DEMO-NORTHWIND",
        ontology=ontology,
        decisions=[],
    )
    assert other_type[0].status == "unresolved"


def test_stable_id_and_human_review_override(ontology):
    exact = match_entity(
        name="Organic Cotton",
        entity_type="Material",
        external_id="MAT-COTTON",
        ontology=ontology,
        decisions=[],
    )
    assert exact[0].status == "verified"
    assert exact[0].method == "exact_stable_id"

    approved = match_entity(
        name="Cotton Blend",
        entity_type="Material",
        external_id=None,
        ontology=ontology,
        decisions=[
            ReviewDecision(
                start_name="Cotton Blend",
                start_type="Material",
                canonical_id="fashion:material:cotton",
                status="approved",
                note="Reviewed for the demo.",
            )
        ],
    )
    assert approved[0].status == "verified"
    assert approved[0].method == "human_review"

    rejected = match_entity(
        name="Cotton Blend",
        entity_type="Material",
        external_id=None,
        ontology=ontology,
        decisions=[
            ReviewDecision(
                start_name="cotton blend",
                start_type="Material",
                canonical_id="fashion:material:cotton",
                status="rejected",
            )
        ],
    )
    assert rejected[0].status == "rejected"


def test_similarity_never_auto_verifies(ontology):
    class Scorer:
        def score(self, left: str, right: str) -> float:
            return 0.99 if right == "Linen" else 0.0

    matches = match_entity(
        name="Zephyr",
        entity_type="Material",
        external_id=None,
        ontology=ontology,
        decisions=[],
        scorer=Scorer(),
    )
    assert matches[0].status == "review"
    assert matches[0].method == "similarity_review"
    assert matches[0].canonical_id == "fashion:material:linen"


def test_relationship_allowlist_and_unique_ids(ontology):
    repo = MemoryGraphRepository(ontology)
    repo.ensure_constraints()
    repo.ensure_constraints()
    product = GraphNode(id="plmex:product:P1", labels=["Entity", "Record", "Product"], graph="own_products", properties={})
    material = GraphNode(
        id="plmex:material:cotton",
        labels=["Entity", "SourceEntity", "Material"],
        graph="own_products",
        properties={},
    )
    repo.upsert_node(product)
    repo.upsert_node(material)
    repo.upsert_node(GraphNode(id=product.id, labels=product.labels, graph=product.graph, properties={"name": "updated"}))
    assert len(repo.find_nodes(include_deleted=True)) == 2
    with pytest.raises(RelationshipValidationError):
        repo.upsert_edge(
            GraphEdge(
                id="bad",
                type="MENTIONS",
                start_id=product.id,
                end_id=material.id,
                graph="own_products",
                properties={},
            )
        )
    repo.upsert_edge(
        GraphEdge(
            id="ok",
            type="USES_MATERIAL",
            start_id=product.id,
            end_id=material.id,
            graph="own_products",
            properties={"lifecycle": "active"},
        )
    )
    assert primary_id("own_products", "PLMEX-DEMO-001") == "plmex:product:PLMEX-DEMO-001"
    assert source_entity_id("own_products", "Material", "organic cotton") == "plmex:material:organic-cotton"
    assert source_entity_id("own_products", "Company", "威海倍佳贸易有限公司") == "plmex:company:威海倍佳贸易有限公司"
    with pytest.raises(ValueError):
        source_entity_id("own_products", "Company", "---")


def test_new_source_registration_does_not_require_engine_edits():
    from knowledge_graph.adapters.base import create_adapter, register_adapter
    from knowledge_graph.graph.build import get_builder, register_builder
    from knowledge_graph.ids import register_source_identity
    from knowledge_graph.models import SourceConfig

    class FutureAdapter:
        source_type = "future_feed"

        def __init__(self, source, root):
            self.source = source

    register_source_identity("future_feed", "future", "record", "Technology")
    register_adapter("future_feed", FutureAdapter)
    register_builder("future_feed", lambda envelopes, repo, ontology: [])
    source = SourceConfig(
        source_id="future_feed",
        source_type="future_feed",
        connector="filesystem",
        supported_formats=["json"],
        ingestion_mode="batch",
        update_strategy="content_hash",
        permissions_model="public",
        enabled=True,
    )
    assert create_adapter(source, ROOT).source_type == "future_feed"
    assert get_builder("future_feed")([], None, None) == []
