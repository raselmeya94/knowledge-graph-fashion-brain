"""In-process facet embeddings for synonym linking (loaded once per process)."""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Any

LOGGER = logging.getLogger("knowledge_graph.linking.embeddings")

_LOCK = threading.Lock()
_MODEL = None
_MODEL_NAME: str | None = None
_INDEX: "FacetEmbeddingIndex | None" = None


@dataclass
class FacetRow:
    label: str
    id: str
    name: str
    normalized_name: str


class FacetEmbeddingIndex:
    """Cosine top-k over facet display names. Built once, reused per request."""

    def __init__(self, rows: list[FacetRow], vectors, model_name: str):
        import numpy as np

        self.rows = rows
        self.model_name = model_name
        mat = np.asarray(vectors, dtype=np.float32)
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = mat / norms

    def search(self, query: str, *, label: str | None = None, top_k: int = 5) -> list[tuple[FacetRow, float]]:
        import numpy as np

        text = str(query or "").strip()
        if not text or self.matrix.size == 0:
            return []
        model = _get_model(self.model_name)
        if model is None:
            return []
        vec = np.asarray(model.encode([text], normalize_embeddings=True), dtype=np.float32)[0]
        scores = self.matrix @ vec
        order = np.argsort(-scores)
        out: list[tuple[FacetRow, float]] = []
        for idx in order:
            row = self.rows[int(idx)]
            if label and row.label != label:
                continue
            out.append((row, float(scores[int(idx)])))
            if len(out) >= top_k:
                break
        return out


def _get_model(model_name: str):
    global _MODEL, _MODEL_NAME
    with _LOCK:
        if _MODEL is not None and _MODEL_NAME == model_name:
            return _MODEL
        try:
            from sentence_transformers import SentenceTransformer

            LOGGER.info("Loading entity-linker embedding model %s", model_name)
            _MODEL = SentenceTransformer(model_name)
            _MODEL_NAME = model_name
            return _MODEL
        except Exception as exc:
            LOGGER.warning("Entity-linker embeddings unavailable: %s", exc)
            _MODEL = None
            _MODEL_NAME = None
            return None


def load_facet_rows(platform, graph_id: str, facet_types: list[str] | None = None) -> list[FacetRow]:
    session_factory = getattr(platform.graph, "_session", None)
    if not callable(session_factory):
        return []
    if not facet_types:
        try:
            from knowledge_graph.retrieval.runtime_schema import build_runtime_schema

            facet_types = list(build_runtime_schema(platform, graph_id).facet_types)
        except Exception:
            facet_types = []
    if not facet_types:
        return []
    # Dynamic label predicate: n:Label1 OR n:Label2 …
    ors = " OR ".join(f"n:{re.sub(r'[^A-Za-z0-9_]', '', lab)}" for lab in facet_types)
    query = (
        "MATCH (n:Entity) "
        f"WHERE n.graph = $g AND coalesce(n.lifecycle,'active') <> 'deleted' AND ({ors}) "
        "RETURN labels(n) AS labels, n.id AS id, coalesce(n.name,'') AS name, "
        "coalesce(n.normalized_name,'') AS nn"
    )
    rows: list[FacetRow] = []
    facet_set = set(facet_types)
    with session_factory() as session:
        for record in session.run(query, {"g": graph_id}):
            labels = set(record["labels"] or [])
            label = next((x for x in facet_types if x in labels), None)
            if not label or label not in facet_set:
                continue
            rows.append(
                FacetRow(
                    label=label,
                    id=str(record["id"]),
                    name=str(record["name"] or ""),
                    normalized_name=str(record["nn"] or ""),
                )
            )
    return rows


def build_facet_index(platform, graph_id: str, *, model_name: str) -> FacetEmbeddingIndex | None:
    model = _get_model(model_name)
    if model is None:
        return None
    rows = load_facet_rows(platform, graph_id)
    if not rows:
        return FacetEmbeddingIndex([], [], model_name)
    texts = [f"{r.name} {r.normalized_name}".strip() for r in rows]
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return FacetEmbeddingIndex(rows, vectors, model_name)


def get_or_build_facet_index(platform, graph_id: str | None = None) -> FacetEmbeddingIndex | None:
    global _INDEX
    from knowledge_graph.retrieval.runtime_schema import default_product_graph_id

    gid = graph_id or default_product_graph_id(platform)
    if not gid:
        return None
    settings = getattr(platform, "settings", None)
    if settings is not None and not getattr(settings, "entity_linker_embeddings", True):
        return None
    model_name = getattr(settings, "entity_linker_model", None) or (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    with _LOCK:
        if _INDEX is not None and _INDEX.model_name == model_name and _INDEX.rows:
            return _INDEX
        try:
            _INDEX = build_facet_index(platform, gid, model_name=model_name)
        except Exception as exc:
            LOGGER.warning("Failed to build facet embedding index: %s", exc)
            _INDEX = None
        return _INDEX


def warm_facet_index(platform, graph_id: str | None = None) -> dict[str, Any]:
    """Load model + build index once (KG startup / reload)."""

    index = get_or_build_facet_index(platform, graph_id)
    if index is None:
        return {"status": "unavailable", "rows": 0}
    return {"status": "ok", "rows": len(index.rows), "model": index.model_name}


def reset_facet_index() -> None:
    global _INDEX
    with _LOCK:
        _INDEX = None
