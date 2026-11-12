from fastapi.testclient import TestClient

from fashion_brain.api.app import _set_answer_title, _title_from_answer, create_app
from knowledge_graph.api import create_app as create_graph_app
from knowledge_graph.errors import GraphBackendUnavailable
from knowledge_graph.graph.neo4j_repo import Neo4jGraphRepository, connect_neo4j
from knowledge_graph.ontology import load_ontology
from knowledge_graph.config import DATA_ROOT
from tests.conftest import ROOT, make_platform


def _client(platform):
    app = create_app(platform, ingest_on_startup=False)
    return TestClient(app)


def _graph_client(platform):
    return TestClient(create_graph_app(platform))


def test_conversation_title_comes_from_answer_prose():
    assert _title_from_answer(
        "## **Primark leads the comparison.**\n\n| Brand | Count |\n| --- | ---: |"
    ) == "Primark leads the comparison."
    assert _title_from_answer("| Brand | Count |\n| --- | ---: |\n") == ""
    assert len(_title_from_answer("A very long answer title that should be shortened at a word boundary " * 2)) <= 64

    class Store:
        title = "New chat"

        def get_conversation(self, _conversation_id, _user_id):
            return {"title": self.title}

        def touch_conversation(self, _conversation_id, title=None):
            if title:
                self.title = title

    store = Store()
    _set_answer_title(store, "chat", "user", "Primark leads the comparison.")
    _set_answer_title(store, "chat", "user", "A later answer should not replace this title.")
    assert store.title == "Primark leads the comparison."


def test_conversation_message_persists_answer_based_title(demo):
    client = _client(demo)
    headers = {"Authorization": f"Bearer {demo.settings.api_token}", "X-Clearance": "internal"}
    created = client.post("/v1/conversations", headers=headers, json={"title": "New chat"})
    assert created.status_code == 200
    conversation_id = created.json()["id"]

    response = client.post(
        f"/v1/conversations/{conversation_id}/messages",
        headers=headers,
        json={"question": "Which products share a material?"},
    )
    assert response.status_code == 200
    expected_title = _title_from_answer(response.json()["answer"]["answer"])
    assert expected_title
    saved = client.get(f"/v1/conversations/{conversation_id}", headers=headers)
    assert saved.json()["title"] == expected_title


def test_sample_image_is_served_from_the_generated_folder(demo):
    folder = demo.settings.generated / "samples"
    folder.mkdir(parents=True, exist_ok=True)
    name = "a" * 32 + ".png"
    (folder / name).write_bytes(b"\x89PNG")
    client = _client(demo)
    found = client.get(f"/samples/{name}")
    assert found.status_code == 200
    assert found.content.startswith(b"\x89PNG")
    assert client.get("/samples/../chat.html").status_code == 404


def test_kg_builder_proxy_is_admin_only_and_allowlisted(demo, monkeypatch):
    calls = []

    def fake_kg_request(settings, method, path, **kwargs):
        calls.append((method, path, kwargs))
        return {"status": "ok"}

    monkeypatch.setattr("fashion_brain.api.app.kg_request", fake_kg_request)
    client = _client(demo)
    headers = {"Authorization": "Bearer test-token", "X-Clearance": "internal"}

    unauthorized = client.get("/v1/admin/kg-api/health")
    assert unauthorized.status_code == 401

    response = client.get(
        "/v1/admin/kg-api/api/setup/checklist?graph_id=products&source_id=products",
        headers=headers,
    )
    assert response.status_code == 200
    assert calls[0][0:2] == ("GET", "/api/setup/checklist")
    assert calls[0][2]["params"] == {"graph_id": "products", "source_id": "products"}
    assert calls[0][2]["clearance"] == "internal"

    unsupported = client.get("/v1/admin/kg-api/api/graphs/registry", headers=headers)
    assert unsupported.status_code == 404
    assert len(calls) == 1


def test_api_auth_validation_and_grounded_ask(demo):
    client = _client(demo)
    graph = _graph_client(demo)
    page = client.get("/")
    assert page.status_code == 200
    # Root may be JSON service info or legacy HTML chat.
    content_type = page.headers.get("content-type", "")
    if "text/html" in content_type:
        assert "/ws/ask" in page.text
    else:
        payload = page.json()
        assert payload.get("service") in {"fashion_brain", "Fashion Brain", None} or "docs" in page.text.lower()

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["graph"] == "ok"
    assert health.json()["model"] == "disabled"

    denied = graph.get("/sources")
    assert denied.status_code == 401

    headers = {"Authorization": "Bearer test-token", "X-Clearance": "internal"}
    sources = graph.get("/sources", headers=headers)
    assert sources.status_code == 200
    source_ids = {item["source_id"] for item in sources.json()["sources"]}
    assert "own_products" in source_ids

    bad_query = graph.post(
        "/graphs/own_products/query",
        headers=headers,
        json={"template": "MATCH_ALL", "params": {"cypher": "MATCH (n) DETACH DELETE n"}},
    )
    assert bad_query.status_code == 422

    materials = graph.post(
        "/graphs/own_products/query",
        headers=headers,
        json={"template": "product_materials", "params": {"record_id": "1000000001"}},
    )
    assert materials.status_code == 200
    assert materials.json()["items"][0]["material"] == "Cotton"

    isolated = graph.post(
        "/graphs/own_products/query",
        headers=headers,
        json={"template": "list_articles"},
    )
    assert isolated.status_code == 422

    shared = graph.post(
        "/graphs/shared/query",
        headers=headers,
        json={"template": "products_sharing_material"},
    )
    assert shared.status_code == 200
    cotton = next(item for item in shared.json()["items"] if item["canonical_id"] == "fashion:material:cotton")
    assert "1000000001" in cotton["products"]
    assert "1000000002" in cotton["products"]
    assert "1000000003" not in cotton["products"]

    review = graph.post("/graphs/shared/query", headers=headers, json={"template": "review_queue"})
    assert any(item["source_name"] == "Cotton Blend" for item in review.json()["items"])

    public_headers = {"Authorization": "Bearer test-token", "X-Clearance": "public"}
    products = graph.get("/products", headers=public_headers)
    assert products.json()["total"] == 0
    internal_products = graph.get("/products", headers=headers, params={"material": "elastane"})
    assert internal_products.json()["total"] == 1
    assert internal_products.json()["items"][0]["product_id"] == "1000000004"

    ask = client.post(
        "/ask",
        headers=headers,
        json={"question": "Which products share a material?"},
    )
    assert ask.status_code == 200
    body = ask.json()
    assert "1000000001" in body["answer"]
    assert "1000000002" in body["answer"]

    invalid = client.post("/ask", headers=headers, json={"question": "no"})
    assert invalid.status_code == 422

    with client.websocket_connect("/ws/ask", headers=headers) as socket:
        socket.send_json({"question": "Which products share a material?"})
        phases = []
        while True:
            event = socket.receive_json()
            phases.append(event)
            if event.get("type") == "done":
                break
        kinds = [event["type"] for event in phases if event["type"] != "done"]
        assert kinds[0] == "thinking"
        assert kinds.index("thinking") < kinds.index("reasoning") < kinds.index("answer")
        streamed = "".join(event["text"] for event in phases if event["type"] == "answer")
        assert streamed == phases[-1]["answer"]["answer"]
        assert "1000000001" in streamed
        assert "1000000002" in streamed
        socket.send_json({"question": "no"})
        rejected = socket.receive_json()
        assert rejected["error"] == "validation_failed"
        socket.send_bytes(b'{"question":"no"}')
        binary = socket.receive_json()
        assert binary["error"] == "validation_failed"
        assert "3 to 2000" in binary["message"]

    with client.websocket_connect("/ws/ask", headers=headers) as socket:
        socket.send_json({"question": "Which products share a material?"})
        assert socket.receive_json()["type"] == "thinking"

    with client.websocket_connect("/ws/ask") as socket:
        denied_socket = socket.receive_json()
        assert denied_socket["error"] == "unauthorized"

    trace = graph.post(
        "/graphs/own_products/query",
        headers=headers,
        json={"template": "trace", "params": {"node_id": "plmex:product:1000000001"}},
    )
    assert trace.status_code == 200
    assert trace.json()["edges"]

    hidden_trace = graph.post(
        "/graphs/own_products/query",
        headers=public_headers,
        json={"template": "trace", "params": {"node_id": "plmex:product:1000000001"}},
    )
    assert hidden_trace.status_code == 403


def test_ingestion_endpoint_is_development_only(demo, tmp_path):
    demo.settings.env = "staging"
    try:
        client = _graph_client(demo)
        headers = {"Authorization": "Bearer test-token"}
        response = client.post("/ingestion/run", headers=headers, json={})
        assert response.status_code == 403
    finally:
        demo.settings.env = "test"

    staging = make_platform(demo.settings.raw_dir, tmp_path / "generated", env="staging", api_token="")
    open_client = _graph_client(staging)
    assert open_client.get("/health").status_code == 200
    assert open_client.get("/sources").status_code == 401


def test_neo4j_password_missing_does_not_open_a_connection(tmp_path):
    from tests.conftest import RAW

    platform = make_platform(RAW, tmp_path / "generated", graph_backend="neo4j", neo4j_password="")
    assert platform.degraded_reason
    assert "password" in platform.degraded_reason.lower()
    client = _graph_client(platform)
    health = client.get("/health")
    assert health.json()["status"] == "degraded"
    headers = {"Authorization": "Bearer test-token", "X-Clearance": "internal"}
    query = client.post("/graphs/shared/query", headers=headers, json={"template": "canonical_materials"})
    assert query.status_code == 503


def test_neo4j_queries_are_parameterized():
    ontology = load_ontology(DATA_ROOT / "schemas" / "product-knowledge-ontology.json")

    class Session:
        def __init__(self):
            self.calls = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, query, parameters=None):
            self.calls.append((query, parameters or {}))
            return []

    class Driver:
        def __init__(self):
            self.session_obj = Session()

        def session(self, **kwargs):
            return self.session_obj

    repo = Neo4jGraphRepository(Driver(), ontology)
    malicious = "MATCH (n) DETACH DELETE n"
    from knowledge_graph.models import GraphNode

    repo.upsert_node(
        GraphNode(
            id="fashion:material:cotton",
            labels=["Entity", "Canonical", "Material"],
            graph="shared",
            properties={"name": malicious, "lifecycle": "active"},
        )
    )
    query, params = repo.driver.session_obj.calls[-1]
    assert "$props" in query
    assert params["props"]["name"] == malicious
    assert "DETACH DELETE" not in query
    statements = repo.ensure_constraints()
    assert any("entity_id" in statement for statement in statements)


def test_shared_material_retrieval_uses_one_parameterized_neo4j_query():
    from types import SimpleNamespace

    from knowledge_graph.access import Principal
    from knowledge_graph.retrieval.queries import products_sharing_material
    from knowledge_graph.ontology import empty_ontology

    ontology = empty_ontology()

    class Session:
        def __init__(self):
            self.calls = []
            self.rows = [
                {
                    "canonical_id": "fashion:material:cotton",
                    "material_key": "fashion:material:cotton",
                    "material": "Cotton",
                    "graph_ids": ["catalog"],
                    "product_count": 2,
                    "product_refs": [
                        {"graph_id": "catalog", "product_id": "style-1"},
                        {"graph_id": "catalog", "product_id": "style-2"},
                    ],
                },
            ]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, query, parameters=None):
            self.calls.append((query, parameters or {}))
            return self.rows

    class Driver:
        def __init__(self):
            self.session_obj = Session()

        def session(self, **kwargs):
            return self.session_obj

    platform = SimpleNamespace(
        graph=Neo4jGraphRepository(Driver(), ontology),
        kg_registry=SimpleNamespace(
            list_graphs=lambda: [SimpleNamespace(id="catalog", kind="product", status="active")]
        ),
    )
    groups = products_sharing_material(platform, Principal(subject="analyst", clearance="internal"))

    assert len(platform.graph.driver.session_obj.calls) == 1
    query, parameters = platform.graph.driver.session_obj.calls[0]
    assert "p.graph IN $graph_ids" in query
    assert "collect(DISTINCT p.graph)" in query
    assert "product_refs[..8]" in query
    assert parameters == {
        "graph_ids": ["catalog"],
        "allowed_classifications": ["public", "internal"],
        "across_graphs": False,
    }
    assert groups == [
        {
            "canonical_id": "fashion:material:cotton",
            "material": "Cotton",
            "products": ["style-1", "style-2"],
            "product_refs": [
                {"graph_id": "catalog", "product_id": "style-1"},
                {"graph_id": "catalog", "product_id": "style-2"},
            ],
            "product_count": 2,
            "verified": True,
            "graph_id": "catalog",
            "graph_ids": ["catalog"],
        }
    ]


def test_shared_material_groups_can_require_cross_graph_matches():
    from types import SimpleNamespace

    from knowledge_graph.access import Principal
    from knowledge_graph.models import GraphEdge, GraphNode
    from knowledge_graph.retrieval.queries import products_sharing_material

    rows = []

    def add_row(graph_id, product_id, material_id, name, canonical_id):
        product = GraphNode(
            id=f"{graph_id}:{product_id}",
            labels=["Entity", "Product"],
            graph=graph_id,
            properties={"prov_record_id": product_id, "classification": "internal"},
        )
        material = GraphNode(
            id=material_id,
            labels=["Entity", "SourceEntity", "Material"],
            graph=graph_id,
            properties={"name": name},
        )
        edge = GraphEdge(
            id=f"uses:{graph_id}:{product_id}",
            type="USES_MATERIAL",
            start_id=product.id,
            end_id=material.id,
            graph=graph_id,
            properties={},
        )
        rows.append(
            {
                "product": product,
                "material": material,
                "edge": edge,
                "graph_id": graph_id,
                "canonical_id": canonical_id,
                "material_key": canonical_id,
                "review": False,
            }
        )

    add_row("ecom_products", "ecom-1", "ecom:cotton", "Cotton", "fashion:material:cotton")
    add_row("plmex_products", "plmex-1", "plmex:cotton", "Cotton", "fashion:material:cotton")
    add_row("ecom_products", "linen-1", "ecom:linen", "Linen", "fashion:material:linen")
    add_row("ecom_products", "linen-2", "ecom:linen", "Linen", "fashion:material:linen")

    class Graph:
        def product_material_rows(self, graph_ids):
            return [row for row in rows if row["graph_id"] in graph_ids]

    platform = SimpleNamespace(graph=Graph())
    principal = Principal(subject="analyst", clearance="internal")

    cross_graph = products_sharing_material(
        platform,
        principal,
        graph_ids=["ecom_products", "plmex_products"],
        across_graphs=True,
    )
    assert [group["material"] for group in cross_graph] == ["Cotton"]
    assert cross_graph[0]["graph_ids"] == ["ecom_products", "plmex_products"]
    assert cross_graph[0]["product_count"] == 2
    assert cross_graph[0]["product_refs"] == [
        {"graph_id": "ecom_products", "product_id": "ecom-1"},
        {"graph_id": "plmex_products", "product_id": "plmex-1"},
    ]


def test_connect_neo4j_reports_unavailable(monkeypatch):
    from knowledge_graph.config import Settings

    settings = Settings(graph_backend="neo4j", neo4j_password="local-only", neo4j_uri="bolt://127.0.0.1:1")

    def explode(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr("neo4j.GraphDatabase.driver", explode)
    try:
        connect_neo4j(settings, load_ontology(DATA_ROOT / "schemas" / "product-knowledge-ontology.json"))
        raised = False
    except GraphBackendUnavailable:
        raised = True
    assert raised
