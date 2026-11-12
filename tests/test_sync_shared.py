from types import SimpleNamespace

from knowledge_graph.pipeline import ecom_sync, mapping_sql_sync


def test_mapped_sql_sync_builds_shared_graph_before_master_snapshot(monkeypatch, tmp_path):
    calls = []

    class Platform:
        settings = SimpleNamespace(generated=tmp_path)
        graph = object()

        def build_shared(self):
            calls.append("shared")
            return {"verified": 1, "review": 0, "rejected": 0}

    mapping = {"graph_id": "mapped_graph", "source_table": "public.products"}
    monkeypatch.setattr(mapping_sql_sync, "mapping_exists", lambda _source_id: True)
    monkeypatch.setattr(mapping_sql_sync, "get_mapping", lambda _source_id: {"mapping": mapping})
    monkeypatch.setattr(mapping_sql_sync, "_apply_ontology", lambda *_args: None)
    monkeypatch.setattr(mapping_sql_sync, "_register_identity", lambda *_args: None)
    monkeypatch.setattr(mapping_sql_sync, "_resolve_dsn", lambda *_args: "test-dsn")
    monkeypatch.setattr(mapping_sql_sync, "_select_columns", lambda _mapping: ["id"])
    monkeypatch.setattr(
        mapping_sql_sync,
        "_iter_rows",
        lambda *_args, **_kwargs: iter([{"id": "p1"}]),
    )
    monkeypatch.setattr(
        mapping_sql_sync,
        "_row_to_graph",
        lambda *_args, **_kwargs: ({"product_id": "p1"}, []),
    )
    monkeypatch.setattr(mapping_sql_sync, "_upsert", lambda *_args, **_kwargs: {"nodes": 1, "edges": 0})

    from knowledge_graph.graph import snapshots
    from knowledge_graph.retrieval import system_meta

    monkeypatch.setattr(snapshots, "save_source_snapshots", lambda *_args: calls.append("source") or [])
    monkeypatch.setattr(snapshots, "save_master_snapshot", lambda *_args: calls.append("master"))
    monkeypatch.setattr(system_meta, "build_system_meta", lambda *_args, **_kwargs: calls.append("system_meta"))

    report = mapping_sql_sync.sync_mapped_sql_to_neo4j(Platform(), "mapped_source")

    assert report["ok"] is True
    assert report["shared"]["verified"] == 1
    assert calls.index("shared") < calls.index("master")


def test_ecom_generated_refresh_builds_shared_graph_before_master_snapshot(monkeypatch, tmp_path):
    calls = []
    platform = SimpleNamespace(
        settings=SimpleNamespace(generated=tmp_path),
        graph=object(),
        build_shared=lambda: calls.append("shared") or {"verified": 1},
    )
    monkeypatch.setattr(ecom_sync, "refresh_ecom_stats_only", lambda _platform: {"stats_products": 1})

    from knowledge_graph.graph import snapshots
    from knowledge_graph.retrieval import schema_hooks, system_meta

    monkeypatch.setattr(snapshots, "save_source_snapshots", lambda *_args: calls.append("source") or [])
    monkeypatch.setattr(snapshots, "save_master_snapshot", lambda *_args: calls.append("master"))
    monkeypatch.setattr(system_meta, "build_system_meta", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(schema_hooks, "on_schema_changed", lambda *_args, **_kwargs: {})

    refreshed = ecom_sync.refresh_generated_after_ecom_sync(platform)

    assert refreshed["shared"] == {"verified": 1}
    assert calls.index("shared") < calls.index("master")
