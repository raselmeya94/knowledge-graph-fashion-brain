"""HTTP client for the knowledge-graph service."""

from __future__ import annotations

from typing import Any

import httpx

from knowledge_graph.errors import FashionGPTError, GraphBackendUnavailable
from knowledge_graph.retrieval.brain import BrainRetrieveRequest, BrainRetrieveResponse


def kg_base_url(settings) -> str:
    return str(getattr(settings, "kg_url", "") or "").rstrip("/")


def uses_kg_api(settings) -> bool:
    return bool(kg_base_url(settings))


def _headers(settings, clearance: str | None = None, *, content_type: str | None = "application/json") -> dict[str, str]:
    headers: dict[str, str] = {}
    if content_type:
        headers["Content-Type"] = content_type
    if clearance:
        headers["X-Clearance"] = clearance
    token = getattr(settings, "api_token", "") or ""
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _timeout(settings) -> float:
    timeout = getattr(settings, "request_timeout_seconds", None)
    return timeout if timeout and timeout > 0 else 120.0


def _raise_kg(response: httpx.Response, action: str) -> None:
    if response.status_code < 400:
        return
    detail = response.text
    try:
        payload = response.json()
        detail = payload.get("message") or payload.get("error") or detail
    except Exception:
        pass
    raise GraphBackendUnavailable(f"Knowledge graph {action} failed: {detail}")


def kg_request(
    settings,
    method: str,
    path: str,
    *,
    clearance: str | None = "internal",
    json_body: Any = None,
    data: Any = None,
    files: Any = None,
    params: dict[str, Any] | None = None,
) -> Any:
    """Call a KG HTTP path and return JSON (or raw text for empty bodies)."""

    base = kg_base_url(settings)
    if not base:
        raise FashionGPTError("Knowledge graph URL is not configured.")
    content_type = None if files is not None else ("application/json" if json_body is not None or method.upper() != "GET" else None)
    headers = _headers(settings, clearance, content_type=content_type)
    url = f"{base}{path}"
    try:
        response = httpx.request(
            method.upper(),
            url,
            headers=headers,
            json=json_body,
            data=data,
            files=files,
            params=params,
            timeout=_timeout(settings),
        )
    except httpx.TimeoutException as exc:
        raise GraphBackendUnavailable(
            f"Knowledge graph timed out after {_timeout(settings):.0f}s on {method.upper()} {path}. "
            "Long jobs (PLM data ingest) should start in the background — restart KG serve if this persists."
        ) from exc
    except httpx.HTTPError as exc:
        raise GraphBackendUnavailable(
            "Fashion Brain could not reach the knowledge graph service. Start it with: python -m knowledge_graph.cli serve"
        ) from exc
    _raise_kg(response, path)
    if not response.content:
        return {"status": "ok"}
    try:
        return response.json()
    except Exception:
        return {"raw": response.text}


def ping_kg(settings, *, timeout: float = 3.0) -> str:
    """Return ok, unreachable, or an error snippet from the KG health endpoint."""

    base = kg_base_url(settings)
    if not base:
        return "disabled"
    try:
        response = httpx.get(f"{base}/health", timeout=timeout)
        if response.status_code >= 400:
            return "unreachable"
        payload = response.json()
        return str(payload.get("status") or "ok")
    except Exception:
        return "unreachable"


def retrieve_remote(settings, clearance: str, request: BrainRetrieveRequest) -> BrainRetrieveResponse:
    """POST /brain/retrieve on the knowledge-graph service."""

    payload = kg_request(
        settings,
        "POST",
        "/brain/retrieve",
        clearance=clearance or "public",
        json_body=request.model_dump(),
    )
    return BrainRetrieveResponse.model_validate(payload)


def list_graphs(settings, clearance: str = "internal") -> dict[str, Any]:
    """List active graphs from KG. No hard-coded stub when the service is configured."""

    if not kg_base_url(settings):
        return {"graphs": []}
    try:
        return kg_request(settings, "GET", "/graphs", clearance=clearance)
    except GraphBackendUnavailable:
        return {"graphs": [], "error": "unreachable"}


def list_registry(settings, clearance: str = "internal") -> dict[str, Any]:
    """Unified registry (capabilities, entities, source_ids) for Graph Router."""

    if not kg_base_url(settings):
        return {"registry": []}
    try:
        return kg_request(settings, "GET", "/api/graphs/registry", clearance=clearance)
    except GraphBackendUnavailable:
        return {"registry": [], "error": "unreachable"}


def metadata_values(
    settings,
    *,
    attribute: str,
    graph_id: str | None = None,
    q: str | None = None,
    limit: int = 50,
    clearance: str = "internal",
) -> dict[str, Any]:
    """Paginated value discovery from KG metadata API."""

    if not kg_base_url(settings):
        return {"values": [], "total": 0}
    params: dict[str, Any] = {"attribute": attribute, "limit": limit}
    if graph_id:
        params["graph_id"] = graph_id
    if q:
        params["q"] = q
    try:
        return kg_request(settings, "GET", "/api/metadata/values", clearance=clearance, params=params)
    except GraphBackendUnavailable:
        return {"values": [], "total": 0, "error": "unreachable"}


def reload_kg_graphs(settings, clearance: str = "internal") -> dict[str, Any]:
    """Ask the KG process to hot-reload snapshots (FB does not load graph data)."""

    if not kg_base_url(settings):
        return {"status": "disabled"}
    try:
        return kg_request(settings, "POST", "/graphs/reload", clearance=clearance)
    except GraphBackendUnavailable as exc:
        return {"status": "unreachable", "message": str(exc)}
