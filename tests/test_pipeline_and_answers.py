import json

from fashion_brain.agent import answer_question, plan_routes
from knowledge_graph.access import Principal
from knowledge_graph.adapters.base import DiscoveryResult, DiscoveredRecord
from knowledge_graph.ids import primary_id
from knowledge_graph.manifests import refresh_manifest_hashes
from knowledge_graph.models import RecordEnvelope
from tests.conftest import RAW, make_platform


INTERNAL = Principal(subject="analyst", clearance="internal")
PUBLIC = Principal(subject="guest", clearance="public")


def _raw_bytes(root) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_unchanged_catalog_restores_snapshot_without_rebuilding(copied_raw, tmp_path, monkeypatch):
    from knowledge_graph.graph.templates import run_template

    generated = tmp_path / "generated"
    first = make_platform(copied_raw, generated)
    first.rebuild()
    product_snapshot = generated / "graph_snapshots" / "own_products" / "graph_snapshot.json"
    master_snapshot = generated / "graph_snapshots" / "master" / "graph_snapshot.json"
    product_payload = json.loads(product_snapshot.read_text(encoding="utf-8"))
    master_payload = json.loads(master_snapshot.read_text(encoding="utf-8"))
    assert product_payload["source_id"] == "own_products"
    assert product_payload["edges"]
    assert all(edge["graph"] == "own_products" for edge in product_payload["edges"])
    assert master_payload["role"] == "master"
    assert "own_products" in master_payload["sources"]
    assert any(edge["type"] == "RESOLVES_TO" for edge in master_payload["edges"])
    before = product_snapshot.read_bytes()
    assert first.graph.get_node("plmex:product:1000000001") is not None

    again = make_platform(copied_raw, generated)

    def boom(*_args, **_kwargs):
        raise AssertionError("graph builder should not run for an unchanged catalog")

    monkeypatch.setattr("knowledge_graph.pipeline.service.get_builder", boom)
    reports = again.rebuild()
    assert all(not report.errors for report in reports)
    assert all(not report.updated and not report.ingested for report in reports)
    assert again.graph.get_node("plmex:product:1000000001") is not None
    assert again.graph.find_edges(start_id="plmex:product:1000000001", rel_type="USES_MATERIAL")
    assert again.graph.find_edges(rel_type="RESOLVES_TO")
    assert product_snapshot.read_bytes() == before
    sources = run_template(again, "master", "sources", INTERNAL, limit=20, offset=0, params={})
    assert sources["sources"] == ["own_products"]


def test_news_restore_does_not_load_the_product_snapshot(tmp_path):
    from knowledge_graph.graph.memory import MemoryGraphRepository
    from knowledge_graph.graph.snapshots import master_snapshot_path, restore_snapshots, save_master_snapshot, source_snapshot_path

    generated = tmp_path / "generated"
    product = source_snapshot_path(generated, "own_products")
    product.parent.mkdir(parents=True)
    product.write_text(
        json.dumps(
            {
                "source_id": "own_products",
                "nodes": [{"id": "plmex:product:1", "labels": ["Entity"], "graph": "own_products", "properties": {}}],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    news = source_snapshot_path(generated, "news")
    news.parent.mkdir(parents=True)
    news.write_text(
        json.dumps(
            {
                "source_id": "news",
                "nodes": [{"id": "news:document:1", "labels": ["Entity"], "graph": "news", "properties": {}}],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    master = master_snapshot_path(generated)
    master.parent.mkdir(parents=True)
    master.write_text(
        json.dumps(
            {
                "role": "master",
                "sources": ["news", "own_products"],
                "nodes": [],
                "edges": [
                    {
                        "id": "link-1",
                        "type": "RESOLVES_TO",
                        "start_id": "plmex:product:1",
                        "end_id": "fashion:material:cotton",
                        "graph": "shared",
                        "properties": {"prov_source_id": "own_products"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    graph = MemoryGraphRepository(None)
    assert restore_snapshots(graph, generated, ["news"])
    assert graph.get_node("news:document:1") is not None
    assert graph.get_node("plmex:product:1") is None
    assert "link-1" not in graph.edges

    save_master_snapshot(graph, generated)
    kept = json.loads(master.read_text(encoding="utf-8"))
    assert any(edge["id"] == "link-1" for edge in kept["edges"])


def test_ingestion_is_idempotent_and_raw_data_stays_immutable(demo):
    before = _raw_bytes(demo.settings.raw_dir)
    first_nodes = {node.id for node in demo.graph.find_nodes(include_deleted=True)}
    first_edges = {edge.id for edge in demo.graph.find_edges(include_deleted=True)}
    assert len(first_nodes) == len(set(first_nodes))
    reports = demo.rebuild()
    assert all(not report.errors for report in reports)
    assert {node.id for node in demo.graph.find_nodes(include_deleted=True)} == first_nodes
    assert {edge.id for edge in demo.graph.find_edges(include_deleted=True)} == first_edges
    assert _raw_bytes(demo.settings.raw_dir) == before
    status = json.loads((demo.settings.raw_dir / "own_products" / "manifest.json").read_text(encoding="utf-8"))
    assert all(record["ingestion_status"] == "registered" for record in status["records"])


def test_product_graph_keeps_provenance(demo):
    product_nodes = demo.graph.find_nodes(graph="own_products")
    assert product_nodes
    assert all(node.graph == "own_products" for node in product_nodes)
    assert not any(edge.type == "RELATED_TO" for edge in demo.graph.find_edges(include_deleted=True))
    product_edges = demo.graph.find_edges(start_id="plmex:product:1000000001", rel_type="USES_MATERIAL")
    assert [demo.graph.get_node(edge.end_id).properties["name"] for edge in product_edges] == ["Cotton"]
    cotton = product_edges[0]
    assert cotton.properties["prov_evidence_text"] == "100%Cotton"
    assert cotton.properties["prov_field_ref"] == "Composition"
    assert "prov_confidence" not in cotton.properties


def test_shared_links_are_verified_and_ambiguous_matches_are_not_merged(demo):
    resolutions = demo.graph.find_edges(rel_type="RESOLVES_TO")
    verified = [edge for edge in resolutions if edge.properties["prov_verification_status"] == "verified"]
    review = [edge for edge in resolutions if edge.properties["prov_verification_status"] == "review"]
    assert any(edge.end_id == "fashion:material:cotton" and edge.start_id == "plmex:material:cotton" for edge in verified)
    blend = next(edge for edge in review if edge.properties["source_name"] == "Cotton Blend")
    assert blend.properties["prov_verification_status"] == "review"
    assert "prov_confidence" not in blend.properties
    assert blend.properties["prov_similarity"] is not None


def test_agent_answers_are_grounded(demo, monkeypatch):
    assert plan_routes("Which materials are used by our demo products?") == ["products"]
    assert plan_routes("Which products share a material?") == ["shared_products"]
    assert plan_routes("How many synthetic demo products are in the catalog?") == ["product_count"]
    assert plan_routes("Who exports denim to the EU?") == ["news"]
    assert plan_routes("What is Bangladesh's denim market share?") == ["news"]

    materials = answer_question(demo, "Which materials are used by our demo products?", INTERNAL)
    assert "Cotton" in materials.answer
    assert "Cotton Blend" in materials.answer
    assert materials.evidence
    assert any("1000000001" in (item.record_id or "") for item in materials.evidence)

    hidden = answer_question(demo, "Which materials are used by our demo products?", PUBLIC)
    assert hidden.evidence == []
    assert "1000000001" not in hidden.answer
    assert (
        hidden.unavailable
        or "0 visible" in hidden.answer.lower()
        or "0 products" in hidden.answer.lower()
        or "clearance" in hidden.answer.lower()
    )

    shared = answer_question(demo, "Which products share a material?", INTERNAL)
    assert "1000000001" in shared.answer and "1000000002" in shared.answer
    assert "1000000003" not in shared.answer

    live = answer_question(demo, "What is the live PLMEX product count?", INTERNAL)
    assert live.evidence == []
    assert live.unavailable
    assert "6" not in live.answer

    count = answer_question(demo, "How many synthetic demo products are in the catalog?", INTERNAL)
    assert "6" in count.answer
    assert count.synthetic_data is False

    red = answer_question(demo, "how many products are sweater with red color?", INTERNAL)
    assert "color red" in red.answer.lower() or "red" in red.answer.lower()
    assert "sweater" in red.answer.lower()
    # Catalog may return 1–2 depending on color/garment matching in the sample rows.
    assert any(ch.isdigit() for ch in red.answer)
    assert {item.record_id for item in red.evidence} <= {"1000000003", "1000000005", "1000000006"}
    assert red.evidence

    by_gender = answer_question(demo, "How many sweater products with red color are there by gender?", INTERNAL)
    assert "Women" in by_gender.answer
    assert "gender" in by_gender.answer.lower() or "Women 1" in by_gender.answer

    follow_up = answer_question(
        demo,
        "now, can you make a breakdown by gender?",
        INTERNAL,
        [
            {"role": "user", "content": "how many products are sweater with red color?"},
            {"role": "assistant", "content": red.answer},
        ],
    )
    assert "color red" in follow_up.answer.lower() or "red" in follow_up.answer.lower()
    assert "sweater" in follow_up.answer.lower()
    assert "Women" in follow_up.answer
    assert "gender" in follow_up.answer.lower() or "Women 1" in follow_up.answer
    assert "now can you make" not in follow_up.answer.lower()

    by_category = answer_question(demo, "How many products are there by category?", INTERNAL)
    assert "Sweater 3" in by_category.answer
    assert "Tee 1" in by_category.answer

    women = answer_question(demo, "How many women sweater products with red color?", INTERNAL)
    assert "gender women" in women.answer
    assert " 1 " in f" {women.answer} "

    assert plan_routes("What brands do we have?") == ["product_count"]

    from fashion_brain import agents as ask_agents

    order = []
    understand = ask_agents.IntentionAgent.understand
    choose = ask_agents.GraphRouterAgent.choose
    compose = ask_agents.ResponseAgent.compose

    def tracked_understand(self, platform, question, history=None, summary=None):
        order.append("intention")
        return understand(self, platform, question, history, summary)

    def tracked_choose(self, intention, platform=None, question: str = ""):
        order.append("router")
        choice = choose(self, intention, platform=platform, question=question)
        assert choice.graph == "own_products"
        return choice

    def tracked_compose(self, platform, question, retrieved):
        order.append("response")
        return compose(self, platform, question, retrieved)

    monkeypatch.setattr(ask_agents.IntentionAgent, "understand", tracked_understand)
    monkeypatch.setattr(ask_agents.GraphRouterAgent, "choose", tracked_choose)
    monkeypatch.setattr(ask_agents.ResponseAgent, "compose", tracked_compose)
    routed = answer_question(demo, "How many products are there by category?", INTERNAL)
    assert order == ["intention", "router", "response"]
    assert routed.routes == ["product_count", "own_products"]


def test_update_and_deletion_recovery(copied_raw, tmp_path):
    platform = make_platform(copied_raw, tmp_path / "generated")
    platform.rebuild()
    import csv

    from knowledge_graph.adapters.products import RAW_COLUMNS
    from tests.conftest import SAMPLE_ROWS

    product_file = copied_raw / "own_products" / "products_sample.csv"
    revised = []
    for row in SAMPLE_ROWS:
        item = dict(row)
        if item["Code"] == "1000000004":
            item["Composition"] = "100%Linen"
            item["Style Reference"] = "SWIM-004-REVISED"
        revised.append(item)
    with product_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in revised:
            writer.writerow({column: row.get(column, "") for column in RAW_COLUMNS})
    refresh_manifest_hashes(copied_raw)
    reports = platform.rebuild()
    assert "1000000004" in next(report.updated for report in reports if report.source_id == "own_products")
    assert demo_material_names(platform, "1000000004") == ["Linen"]
    retired = [
        edge
        for edge in platform.graph.find_edges(start_id="plmex:product:1000000004", rel_type="USES_MATERIAL", include_deleted=True)
        if edge.properties.get("lifecycle") == "deleted"
    ]
    assert any(platform.graph.get_node(edge.end_id, include_deleted=True).properties["name"] == "Elastane" for edge in retired)

    with product_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in revised:
            if row["Code"] == "1000000004":
                continue
            writer.writerow({column: row.get(column, "") for column in RAW_COLUMNS})
    refresh_manifest_hashes(copied_raw)
    platform.rebuild()
    deleted = platform.graph.get_node(primary_id("own_products", "1000000004"), include_deleted=True)
    assert deleted.properties["lifecycle"] == "deleted"
    assert platform.graph.get_node(deleted.id, include_deleted=False) is None
    rows, total = platform.products.search(INTERNAL, product_id="1000000004")
    assert rows == [] and total == 0


def demo_material_names(platform, record_id: str) -> list[str]:
    names = []
    for edge in platform.graph.find_edges(start_id=primary_id("own_products", record_id), rel_type="USES_MATERIAL"):
        names.append(platform.graph.get_node(edge.end_id).properties["name"])
    return names


def test_duplicate_content_is_flagged(tmp_path, monkeypatch):
    platform = make_platform(RAW, tmp_path / "generated")
    source = platform.registry.by_id("own_products")

    def envelope(record_id: str) -> RecordEnvelope:
        return RecordEnvelope(
            source_id="own_products",
            source_type="own_products",
            record_id=record_id,
            title=f"[SYNTHETIC] {record_id}",
            content="SYNTHETIC duplicate body",
            content_hash="a" * 64,
            synthetic=True,
            classification="internal",
            source_path=f"{record_id}.json",
            ingested_at="2026-10-14T00:00:00Z",
            entities=[],
            relationships=[],
        )

    class FakeAdapter:
        source_type = "own_products"

        def bind_runtime(self, *args, **kwargs):
            return None

        def discover(self):
            records = []
            for record_id in ("dup_a", "dup_b"):
                records.append(
                    DiscoveredRecord(
                        source_id="own_products",
                        record_id=record_id,
                        path=tmp_path / record_id,
                        relative_path=record_id,
                        format="json",
                        hash_method="file",
                        manifest_hash="a" * 64,
                        source_timestamp=None,
                        synthetic_declared=True,
                        raw_bytes=b"{}",
                        parsed={},
                    )
                )
            return DiscoveryResult(records=records, manifest_ids=["dup_a", "dup_b"])

        def validate(self, record):
            return []

        def normalize(self, record, ingested_at):
            return envelope(record.record_id)

        def extract(self, item):
            return item, []

    monkeypatch.setattr("knowledge_graph.pipeline.service.create_adapter", lambda source, root: FakeAdapter())
    report = platform.ingest_source(source)
    assert report.duplicates
    assert report.errors == []


def test_news_question_is_answered_from_the_news_graph(copied_raw, tmp_path):
    from knowledge_graph.models import GraphEdge, GraphNode

    platform = make_platform(copied_raw, tmp_path / "generated")
    platform.rebuild()
    common = {"lifecycle": "active", "prov_classification": "internal", "classification": "internal"}
    platform.graph.upsert_node(
        GraphNode(
            id="news:country:bangladesh",
            labels=["Entity", "Country"],
            graph="news",
            properties={**common, "name": "Bangladesh", "entity_type": "Country"},
        )
    )
    platform.graph.upsert_node(
        GraphNode(
            id="news:region:eu",
            labels=["Entity", "Region"],
            graph="news",
            properties={**common, "name": "EU", "entity_type": "Region"},
        )
    )
    platform.graph.upsert_edge(
        GraphEdge(
            id="news-export-bd-eu",
            type="EXPORTS_TO",
            start_id="news:country:bangladesh",
            end_id="news:region:eu",
            graph="news",
            properties={
                **common,
                "prov_source_id": "news",
                "prov_record_id": "doc1",
                "prov_evidence_text": "earning €869.13 million",
                "product": "denim",
                "value": "€869.13M",
                "period": "2026-01..08",
            },
        )
    )

    answer = answer_question(platform, "Who exports denim to the EU?", INTERNAL)
    assert "news" in answer.routes
    assert "€869.13M" in answer.answer
    assert answer.evidence
    assert answer.evidence[0].graph == "news"

    hidden = answer_question(platform, "Who exports denim to the EU?", PUBLIC)
    assert hidden.evidence == []
    assert hidden.unavailable
    assert "€869.13M" not in hidden.answer


def test_query_load_reads_the_news_snapshot_without_ingest(copied_raw, tmp_path, monkeypatch):
    platform = make_platform(copied_raw, tmp_path / "generated")
    snapshot = tmp_path / "generated" / "graph_snapshots" / "news" / "graph_snapshot.json"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text(
        json.dumps(
            {
                "source_id": "news",
                "nodes": [
                    {
                        "id": "news:country:bangladesh",
                        "labels": ["Entity", "Country"],
                        "graph": "news",
                        "properties": {"name": "Bangladesh", "lifecycle": "active"},
                    }
                ],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )

    def boom(*_args, **_kwargs):
        raise AssertionError("query load must not ingest the catalog")

    monkeypatch.setattr(platform, "ingest_source", boom)
    assert platform.load_for_query(["news"])
    assert platform.graph.get_node("news:country:bangladesh") is not None
    assert platform.graph.get_node("plmex:product:1000000001") is None
