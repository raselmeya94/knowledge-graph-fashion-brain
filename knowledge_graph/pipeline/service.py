"""Ingestion, graph builds, and the in-memory platform used by the CLI and API."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from knowledge_graph.adapters import products as _products  # noqa: F401  (register adapter)
from knowledge_graph.adapters.base import SourceAdapter, create_adapter
from knowledge_graph.config import Settings
from knowledge_graph.errors import GraphBackendUnavailable
from knowledge_graph.graph.build import get_builder
from knowledge_graph.graph.memory import MemoryGraphRepository
from knowledge_graph.graph.snapshots import restore_snapshots, save_master_snapshot, save_source_snapshots
from knowledge_graph.graph.neo4j_repo import UnavailableGraphRepository, connect_neo4j
from knowledge_graph.ids import primary_id
from knowledge_graph.logging_config import get_logger
from knowledge_graph.models import IngestionReport, RecordEnvelope, RecordState, ReviewDecision, SourceConfig
from knowledge_graph.kg_registry import KgRegistryStore
from knowledge_graph.ontology import Ontology
from knowledge_graph.pipeline.state import StateStore
from knowledge_graph.registry import load_registry
from knowledge_graph.resolution.matcher import TokenJaccardScorer
from knowledge_graph.resolution.shared import build_shared_graph
from knowledge_graph.retrieval.products import FileProductRepository

LOGGER = get_logger("knowledge_graph.pipeline")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def open_graph(settings: Settings, ontology: Ontology):
    if settings.graph_backend == "memory":
        graph = MemoryGraphRepository(ontology)
        graph.ensure_constraints()
        return graph
    if settings.graph_backend == "neo4j":
        try:
            return connect_neo4j(settings, ontology)
        except GraphBackendUnavailable as exc:
            return UnavailableGraphRepository(str(exc))
    return UnavailableGraphRepository(f"Unknown graph backend '{settings.graph_backend}'.")


class Platform:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.kg_registry = KgRegistryStore(settings)
        self.kg_registry.register_all_identities()
        # Boot without baked-in schemas: ontologies/graphs arrive via POST APIs.
        self.ontology = self.kg_registry.union_ontology()
        try:
            self.news_ontology = self.kg_registry.load_ontology_object("news")
        except Exception:
            self.news_ontology = self.ontology
        source_registry_path = settings.raw_dir / "source_registry.json"
        if source_registry_path.exists():
            self.registry = load_registry(source_registry_path)
        else:
            from knowledge_graph.models import SourceRegistry

            self.registry = SourceRegistry(version=1, sources=[])
        self.graph = open_graph(settings, self.ontology)
        self.state = StateStore(settings.generated / "ingestion_state.json")
        self.envelopes: dict[tuple[str, str], RecordEnvelope] = {}
        self.products = FileProductRepository([])
        self.last_reports: list[IngestionReport] = []
        self.shared_counts: dict[str, int] = {}
        self.degraded_reason: str | None = None
        try:
            self.graph.ping()
        except GraphBackendUnavailable as exc:
            self.degraded_reason = exc.message
        # Warm facet embeddings once in a background thread (non-blocking boot).
        if getattr(settings, "entity_linker_embeddings", False) and not self.degraded_reason:
            import threading

            def _warm():
                try:
                    from knowledge_graph.linking.embeddings import warm_facet_index

                    warm = warm_facet_index(self, "ecom_products")
                    LOGGER.info("entity linker embeddings warm=%s", warm)
                except Exception as exc:
                    LOGGER.warning("entity linker embedding warm skipped: %s", exc)

            threading.Thread(target=_warm, name="entity-linker-warm", daemon=True).start()

    def ontology_for(self, graph_id: str) -> Ontology:
        graph = self.kg_registry.get_graph(graph_id)
        return self.kg_registry.load_ontology_object(graph.ontology_id)

    def ontology_path_for(self, graph_id: str) -> Path:
        graph = self.kg_registry.get_graph(graph_id)
        return self.kg_registry.ontology_file(graph.ontology_id)

    def refresh_union_ontology(self) -> None:
        """Reload ontology files into the live graph validator after allowlist edits."""

        union = self.kg_registry.union_ontology()
        self.ontology = union
        try:
            self.news_ontology = self.kg_registry.load_ontology_object("news")
        except Exception:
            self.news_ontology = union
        if hasattr(self.graph, "ontology"):
            self.graph.ontology = union
        if hasattr(self.graph, "_allowed_rel_types"):
            self.graph._allowed_rel_types = set(union.relationship_allowlist) | {"RESOLVES_TO"}
        if hasattr(self.graph, "_allowed_labels"):
            self.graph._allowed_labels = set(union.entity_types) | {
                "Entity",
                "SourceEntity",
                "Canonical",
                "Record",
            }

    def envelope_path(self, source_id: str, record_id: str) -> Path:
        return self.settings.generated / "envelopes" / source_id / f"{record_id}.json"

    def save_envelope(self, envelope: RecordEnvelope) -> None:
        path = self.envelope_path(envelope.source_id, envelope.record_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(envelope.model_dump_json(indent=2), encoding="utf-8")
        self.envelopes[(envelope.source_id, envelope.record_id)] = envelope

    def load_envelope(self, source_id: str, record_id: str) -> RecordEnvelope | None:
        path = self.envelope_path(source_id, record_id)
        if not path.exists():
            return None
        return RecordEnvelope.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def review_decisions(self) -> list[ReviewDecision]:
        path = self.settings.reviews_file
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return [ReviewDecision.model_validate(item) for item in data.get("decisions", [])]

    def _bind(self, adapter: SourceAdapter) -> None:
        binder = getattr(adapter, "bind_runtime", None)
        if binder:
            binder(self.ontology, "deterministic", None, self.settings.request_timeout_seconds)

    def ingest_source(self, source: SourceConfig, *, now: str | None = None) -> IngestionReport:
        timestamp = now or utc_now()
        report = IngestionReport(source_id=source.source_id)
        if not source.enabled:
            report.warnings.append("Source is disabled.")
            return report
        try:
            adapter = create_adapter(source, self.settings.raw_dir / source.source_id)
        except Exception as exc:
            report.errors.append({"record_id": "*", "message": str(exc)})
            return report
        self._bind(adapter)
        discovered = adapter.discover()
        report.warnings.extend(discovered.warnings)
        for message in discovered.errors:
            report.errors.append({"record_id": "*", "message": message})
        hashes: dict[str, str] = {}
        for record in discovered.records:
            try:
                errors = adapter.validate(record)
                if errors:
                    self._fail_record(source, record.record_id, "; ".join(errors), report)
                    continue
                previous = self.state.get(source.source_id, record.record_id)
                if (
                    previous
                    and previous.status in {"ingested", "unchanged"}
                    and previous.content_hash == record.manifest_hash
                    and previous.graph_status == "built"
                ):
                    existing = self.load_envelope(source.source_id, record.record_id)
                    if existing and existing.lifecycle == "active":
                        self.envelopes[(source.source_id, record.record_id)] = existing
                        previous.status = "unchanged"
                        self.state.put(previous)
                        report.unchanged.append(record.record_id)
                        hashes[record.manifest_hash] = record.record_id
                        continue
                envelope = adapter.normalize(record, timestamp)
                envelope, extract_errors = adapter.extract(envelope)
                if extract_errors:
                    self._fail_record(source, record.record_id, "; ".join(extract_errors), report)
                    continue
                if envelope.content_hash in hashes:
                    report.duplicates.append(
                        {"record_id": record.record_id, "duplicate_of": hashes[envelope.content_hash]}
                    )
                    envelope.warnings.append(f"Duplicate content of {hashes[envelope.content_hash]}.")
                hashes[envelope.content_hash] = record.record_id
                status = "updated" if previous and previous.status in {"ingested", "unchanged", "deleted"} else "ingested"
                envelope.lifecycle = "active"
                self.save_envelope(envelope)
                self.state.put(
                    RecordState(
                        source_id=source.source_id,
                        record_id=record.record_id,
                        content_hash=envelope.content_hash,
                        status=status,
                        graph_status="pending",
                        ingested_at=timestamp,
                        title=envelope.title,
                    )
                )
                if status == "updated":
                    report.updated.append(record.record_id)
                else:
                    report.ingested.append(record.record_id)
            except Exception as exc:
                self._fail_record(source, record.record_id, str(exc), report)
        listed_ids = set(discovered.manifest_ids)
        for previous in self.state.for_source(source.source_id):
            if not listed_ids or previous.record_id in listed_ids or previous.status == "deleted":
                continue
            self._mark_deleted(source, previous.record_id)
            report.deleted.append(previous.record_id)
        self.state.save()
        LOGGER.info(
            "ingested source %s ingested=%s updated=%s unchanged=%s deleted=%s errors=%s",
            source.source_id,
            len(report.ingested),
            len(report.updated),
            len(report.unchanged),
            len(report.deleted),
            len(report.errors),
        )
        return report

    def _fail_record(self, source: SourceConfig, record_id: str, message: str, report: IngestionReport) -> None:
        report.errors.append({"record_id": record_id, "message": message})
        if self.degraded_reason is None:
            try:
                self.graph.retire_record(source.source_id, record_id)
            except GraphBackendUnavailable:
                pass
        previous = self.state.get(source.source_id, record_id)
        self.state.put(
            RecordState(
                source_id=source.source_id,
                record_id=record_id,
                content_hash=previous.content_hash if previous else "",
                status="error",
                graph_status="error",
                ingested_at=previous.ingested_at if previous else None,
                title=previous.title if previous else "",
                error=message,
            )
        )
        self.envelopes.pop((source.source_id, record_id), None)

    def _mark_deleted(self, source: SourceConfig, record_id: str) -> None:
        if self.degraded_reason is None:
            self.graph.retire_record(source.source_id, record_id)
        existing = self.load_envelope(source.source_id, record_id)
        if existing:
            existing.lifecycle = "deleted"
            self.save_envelope(existing)
        previous = self.state.get(source.source_id, record_id)
        self.state.put(
            RecordState(
                source_id=source.source_id,
                record_id=record_id,
                content_hash=previous.content_hash if previous else "",
                status="deleted",
                graph_status="deleted",
                ingested_at=previous.ingested_at if previous else None,
                title=previous.title if previous else "",
            )
        )

    def build_source(self, source: SourceConfig) -> list[str]:
        if self.degraded_reason:
            return [self.degraded_reason]
        active = [
            envelope
            for (source_id, _record_id), envelope in sorted(self.envelopes.items())
            if source_id == source.source_id and envelope.lifecycle == "active"
        ]
        pending: list[RecordEnvelope] = []
        for envelope in active:
            node = self.graph.get_node(primary_id(envelope.source_type, envelope.record_id), include_deleted=False)
            state = self.state.get(envelope.source_id, envelope.record_id)
            if node is not None and state and state.graph_status == "built" and state.content_hash == envelope.content_hash:
                continue
            if node is not None:
                self.graph.retire_record(envelope.source_id, envelope.record_id)
            pending.append(envelope)
        errors: list[str] = []
        if pending:
            ontology = self.ontology
            try:
                ontology = self.ontology_for(source.source_id)
            except Exception:
                if source.source_type == "ecom_products":
                    try:
                        ontology = self.kg_registry.load_ontology_object("ecom")
                    except Exception:
                        pass
            errors = get_builder(source.source_type)(pending, self.graph, ontology)
        self.graph.refresh_orphans(source.source_id)
        failed = {item.split(":", 1)[0] for item in errors}
        for envelope in pending:
            state = self.state.get(envelope.source_id, envelope.record_id)
            if state is None:
                continue
            if envelope.record_id in failed:
                state.graph_status = "error"
                state.error = "; ".join(item for item in errors if item.startswith(envelope.record_id))
            else:
                state.graph_status = "built"
                state.status = "ingested" if state.status == "updated" else state.status
                state.error = None
            self.state.put(state)
        pending_ids = {item.record_id for item in pending}
        for envelope in active:
            state = self.state.get(envelope.source_id, envelope.record_id)
            if state and state.graph_status == "pending" and envelope.record_id not in pending_ids:
                state.graph_status = "built"
                self.state.put(state)
        self.state.save()
        return errors

    def _active_envelopes(self, source_id: str) -> list[RecordEnvelope]:
        return [
            envelope
            for (stored_source_id, _record_id), envelope in self.envelopes.items()
            if stored_source_id == source_id and envelope.lifecycle == "active"
        ]

    def _states_already_built(self, source_id: str, active: list[RecordEnvelope]) -> bool:
        if not active:
            return False
        for envelope in active:
            state = self.state.get(source_id, envelope.record_id)
            if (
                state is None
                or state.status not in {"ingested", "unchanged"}
                or state.graph_status != "built"
                or state.content_hash != envelope.content_hash
            ):
                return False
        return True

    def _memory_has_records(self, source_type: str, active: list[RecordEnvelope]) -> bool:
        if not hasattr(self.graph, "nodes"):
            return False
        for envelope in active:
            if self.graph.get_node(primary_id(source_type, envelope.record_id), include_deleted=False) is None:
                return False
        return True

    def _restore_memory_snapshot(self, source_ids: list[str] | None = None) -> bool:
        return restore_snapshots(self.graph, self.settings.generated, source_ids)

    def load_for_query(self, source_ids: list[str] | None = None) -> bool:
        """Open saved snapshots for questions. This does not ingest or rebuild."""

        loaded = self._restore_memory_snapshot(source_ids)
        if loaded:
            LOGGER.info("query graph ready sources=%s", source_ids or "all")
        return loaded

    def reload_query_data(self) -> dict[str, int | str]:
        """Re-read registry, all source snapshots, master/shared interconnect, and catalog.

        Fashion Brain must not load graph data; call this only inside the KG process
        after commit/create so chat stays warm without restart.
        """

        self.kg_registry = KgRegistryStore(self.settings)
        self.kg_registry.register_all_identities()
        self.refresh_union_ontology()
        # Memory backend: replace in-process graph. Neo4j: merge snapshots only — do not wipe live DB.
        if self.settings.graph_backend != "neo4j" and hasattr(self.graph, "load_snapshot"):
            self.graph.load_snapshot(
                {"nodes": [], "edges": [], "schema": list(getattr(self.graph, "applied_schema", []) or [])}
            )
        loaded = self.load_for_query()
        catalog_n = self.load_catalog()
        shared_n = len(self.graph.find_nodes(graph="shared", include_deleted=False)) if hasattr(self.graph, "find_nodes") else 0
        source_n = len(self.kg_registry.list_graphs())
        counts = getattr(self.graph, "counts", None)
        if callable(counts):
            tallies = counts()
            node_n = int(tallies.get("nodes", 0))
            edge_n = int(tallies.get("edges", 0))
        else:
            node_n = len(getattr(self.graph, "nodes", {}) or {})
            edge_n = len(getattr(self.graph, "edges", {}) or {})
        stats: dict[str, int | str] = {
            "loaded": int(bool(loaded)),
            "backend": getattr(self.settings, "graph_backend", "memory"),
            "registry_graphs": source_n,
            "nodes": node_n,
            "edges": edge_n,
            "shared_nodes": shared_n,
            "catalog_rows": catalog_n,
        }
        LOGGER.info(
            "query data reloaded graphs=%s nodes=%s edges=%s shared=%s catalog=%s",
            source_n,
            node_n,
            edge_n,
            shared_n,
            catalog_n,
        )
        if getattr(self.settings, "entity_linker_embeddings", False):
            try:
                from knowledge_graph.linking.embeddings import reset_facet_index, warm_facet_index

                reset_facet_index()
                warm_facet_index(self, "ecom_products")
            except Exception as exc:
                LOGGER.warning("entity linker re-warm after reload failed: %s", exc)
        return stats

    def load_saved_envelopes(self, source_id: str) -> int:
        """Read envelopes already written by an ingest. The catalog file is not opened."""

        folder = self.settings.generated / "envelopes" / source_id
        if not folder.is_dir():
            return 0
        count = 0
        for path in folder.glob("*.json"):
            envelope = RecordEnvelope.model_validate(json.loads(path.read_text(encoding="utf-8")))
            self.envelopes[(envelope.source_id, envelope.record_id)] = envelope
            count += 1
        self.refresh_retrieval()
        LOGGER.info("loaded saved envelopes source=%s records=%s", source_id, count)
        return count

    def load_catalog(self) -> int:
        """Open the product list used for counts and filters. The graph snapshot is not read."""

        from knowledge_graph.retrieval.catalog_index import load_catalog_rows

        rows = load_catalog_rows(self.settings.generated)
        self.envelopes = {(row.source_id, row.record_id): row for row in rows}
        self.refresh_retrieval()
        return len(rows)

    def build_shared(self) -> dict[str, int]:
        if self.degraded_reason:
            return {}
        scorer = None
        if self.settings.resolution_similarity == "lexical":
            scorer = TokenJaccardScorer(set(self.ontology.review_stopwords), self.ontology.review_min_token_length)
        self.shared_counts = build_shared_graph(self.graph, self.ontology, self.review_decisions(), scorer=scorer)
        return self.shared_counts

    def refresh_retrieval(self) -> None:
        active = [envelope for envelope in self.envelopes.values() if envelope.lifecycle == "active"]
        self.products = FileProductRepository(active)

    def rebuild(self, source_ids: list[str] | None = None) -> list[IngestionReport]:
        selected = []
        for source in self.registry.sources:
            if not source.enabled:
                continue
            if source_ids is not None and source.source_id not in source_ids:
                continue
            # SQL e-com sync uses sources.sync / ecom_sync — skip in file rebuild.
            if source.connector == "authorized_sql" and source_ids is None:
                continue
            if source.connector == "authorized_sql" and source.source_id in (source_ids or []):
                # Explicit request: still skip file adapter path; caller should use sync_source.
                continue
            selected.append(source)
        reports = [self.ingest_source(source) for source in selected]
        graph_errors: list[str] = []
        if self.degraded_reason:
            for report in reports:
                report.warnings.append(self.degraded_reason)
        else:
            self._restore_memory_snapshot()
            built: list[str] = []
            for source in selected:
                active = self._active_envelopes(source.source_id)
                if self._states_already_built(source.source_id, active) and self._memory_has_records(
                    source.source_type, active
                ):
                    LOGGER.info("graph already built for %s, skipping rebuild", source.source_id)
                    continue
                LOGGER.info("building graph for %s records=%s", source.source_id, len(active))
                graph_errors.extend(self.build_source(source))
                built.append(source.source_id)
            if built:
                self.build_shared()
                save_source_snapshots(self.graph, self.settings.generated, built)
                save_master_snapshot(self.graph, self.settings.generated)
            else:
                LOGGER.info("graph build skipped; PLM data is unchanged")
        self.refresh_retrieval()
        for report, source in zip(reports, selected):
            for message in graph_errors:
                if message.startswith(source.source_id) or ":" in message:
                    record_id = message.split(":", 1)[0]
                    if any(record_id == item for item in [*report.ingested, *report.updated, *report.unchanged]):
                        report.errors.append({"record_id": record_id, "message": message})
        self.last_reports = reports
        (self.settings.generated / "reports").mkdir(parents=True, exist_ok=True)
        (self.settings.generated / "reports" / "latest.json").write_text(
            json.dumps([report.model_dump() for report in reports], indent=2),
            encoding="utf-8",
        )
        return reports
