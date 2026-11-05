"""Optional checks against a local Neo4j started with docker compose."""

import os

import pytest

from knowledge_graph.config import DATA_ROOT, Settings
from knowledge_graph.ontology import load_ontology
from knowledge_graph.pipeline.service import Platform
from tests.conftest import RAW


@pytest.mark.integration
def test_live_neo4j_round_trip(tmp_path):
    if os.environ.get("FASHIONGPT_NEO4J_INTEGRATION") != "1":
        pytest.skip("Set FASHIONGPT_NEO4J_INTEGRATION=1 with a local Neo4j to run this test.")
    settings = Settings(
        env="test",
        raw_data_dir=RAW,
        generated_dir=tmp_path / "generated",
        graph_backend="neo4j",
        neo4j_uri=os.environ.get("FASHIONGPT_NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.environ.get("FASHIONGPT_NEO4J_USER", "neo4j"),
        neo4j_password=os.environ.get("FASHIONGPT_NEO4J_PASSWORD", "fashiongpt-local-dev"),
        api_token="test-token",
        model_provider="none",
    )
    platform = Platform(settings)
    assert load_ontology(DATA_ROOT / "schemas" / "product-knowledge-ontology.json")
    reports = platform.rebuild()
    assert all(not report.errors for report in reports)
    assert platform.graph.ping() == "ok"
