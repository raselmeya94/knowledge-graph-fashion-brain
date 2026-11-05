"""Background jobs for long KG work (ingest, rebuild, reload)."""

from __future__ import annotations

import asyncio
import uuid
from collections import deque
from typing import Any, Callable

import anyio

from knowledge_graph.errors import NotFoundError, ValidationFailed
from knowledge_graph.logging_config import get_logger
from knowledge_graph.pipeline.service import utc_now

LOGGER = get_logger("knowledge_graph.jobs")

TERMINAL = frozenset({"done", "failed", "cancelled"})
ACTIVE = frozenset({"queued", "running"})


class JobStore:
    """In-process job registry. Survives for the life of the KG serve process."""

    def __init__(self, *, max_jobs: int = 100) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._order: deque[str] = deque()
        self._max_jobs = max_jobs
        self._lock = asyncio.Lock()
        self._exclusive: dict[str, str] = {}  # exclusive_key -> job_id
        self._tasks: set[asyncio.Task[Any]] = set()
        self._stopping = False

    def get(self, job_id: str) -> dict[str, Any]:
        job = self._jobs.get(job_id)
        if job is None:
            raise NotFoundError(f"Job '{job_id}' was not found.")
        return dict(job)

    def list(self, *, limit: int = 30, kind: str | None = None) -> list[dict[str, Any]]:
        if limit < 1 or limit > 200:
            raise ValidationFailed("limit must be between 1 and 200.")
        rows: list[dict[str, Any]] = []
        for job_id in reversed(self._order):
            job = self._jobs.get(job_id)
            if job is None:
                continue
            if kind and job.get("kind") != kind:
                continue
            rows.append(dict(job))
            if len(rows) >= limit:
                break
        return rows

    def latest_for_kind(self, kind: str) -> dict[str, Any] | None:
        for job_id in reversed(self._order):
            job = self._jobs.get(job_id)
            if job and job.get("kind") == kind:
                return dict(job)
        return None

    def _trim(self) -> None:
        while len(self._order) > self._max_jobs:
            old_id = self._order.popleft()
            old = self._jobs.pop(old_id, None)
            if not old:
                continue
            key = old.get("exclusive_key")
            if key and self._exclusive.get(key) == old_id:
                self._exclusive.pop(key, None)

    def _active_exclusive(self, exclusive_key: str | None) -> dict[str, Any] | None:
        if not exclusive_key:
            return None
        existing_id = self._exclusive.get(exclusive_key)
        if not existing_id:
            return None
        existing = self._jobs.get(existing_id)
        if existing and existing.get("status") in ACTIVE:
            return dict(existing)
        return None

    def _create_locked(
        self,
        *,
        kind: str,
        params: dict[str, Any] | None,
        exclusive_key: str | None,
        message: str,
    ) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id,
            "kind": kind,
            "status": "queued",
            "params": dict(params or {}),
            "exclusive_key": exclusive_key,
            "message": message,
            "progress": None,
            "result": None,
            "error": None,
            "created_at": utc_now(),
            "started_at": None,
            "finished_at": None,
        }
        self._jobs[job_id] = job
        self._order.append(job_id)
        if exclusive_key:
            self._exclusive[exclusive_key] = job_id
        self._trim()
        return dict(job)

    def _patch(self, job_id: str, **fields: Any) -> dict[str, Any]:
        job = self._jobs.get(job_id)
        if job is None:
            raise NotFoundError(f"Job '{job_id}' was not found.")
        job.update(fields)
        return dict(job)

    async def start(
        self,
        *,
        kind: str,
        runner: Callable[[], Any],
        params: dict[str, Any] | None = None,
        exclusive_key: str | None = None,
        message: str = "",
    ) -> tuple[dict[str, Any], bool]:
        """Queue a sync runner on a worker thread.

        Returns ``(job, created)``. If an exclusive job is already active, returns
        that job with ``created=False`` (no second worker).
        """

        async with self._lock:
            if self._stopping:
                raise ValidationFailed("Knowledge graph is shutting down; not starting new jobs.")
            existing = self._active_exclusive(exclusive_key)
            if existing is not None:
                return existing, False
            job = self._create_locked(
                kind=kind,
                params=params,
                exclusive_key=exclusive_key,
                message=message or f"{kind} queued",
            )
            job_id = job["id"]

        async def _execute() -> None:
            self._patch(
                job_id,
                status="running",
                started_at=utc_now(),
                message=message or f"{kind} running",
            )
            LOGGER.info("job started id=%s kind=%s", job_id, kind)
            try:
                result = await anyio.to_thread.run_sync(runner)
                if isinstance(result, dict) and "message" in result:
                    done_message = str(result.get("message") or f"{kind} finished")
                else:
                    done_message = f"{kind} finished"
                self._patch(
                    job_id,
                    status="done",
                    finished_at=utc_now(),
                    result=result,
                    message=done_message,
                    progress=100,
                )
                LOGGER.info("job done id=%s kind=%s", job_id, kind)
            except asyncio.CancelledError:
                self._patch(
                    job_id,
                    status="cancelled",
                    finished_at=utc_now(),
                    message=f"{kind} cancelled",
                )
                LOGGER.info("job cancelled id=%s kind=%s", job_id, kind)
                raise
            except Exception as exc:
                self._patch(
                    job_id,
                    status="failed",
                    finished_at=utc_now(),
                    error=str(exc),
                    message=f"{kind} failed",
                )
                LOGGER.exception("job failed id=%s kind=%s", job_id, kind)
            finally:
                async with self._lock:
                    if exclusive_key and self._exclusive.get(exclusive_key) == job_id:
                        self._exclusive.pop(exclusive_key, None)

        task = asyncio.create_task(_execute(), name=f"kg-job-{kind}-{job_id}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job, True

    async def shutdown(self, *, wait_seconds: float = 2.0) -> None:
        """Cancel in-flight job tasks so uvicorn can exit on one Ctrl+C."""

        self._stopping = True
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if not tasks:
            return
        LOGGER.info("cancelling %s background job task(s)", len(tasks))
        await asyncio.wait(tasks, timeout=wait_seconds)
        for job in self._jobs.values():
            if job.get("status") in ACTIVE:
                job["status"] = "cancelled"
                job["finished_at"] = utc_now()
                job["message"] = f"{job.get('kind') or 'job'} cancelled on shutdown"
        self._exclusive.clear()
