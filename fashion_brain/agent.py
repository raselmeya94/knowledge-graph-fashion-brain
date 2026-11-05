"""Compatibility shim. Planning and answers live in ``fashion_brain.ask``."""

from __future__ import annotations

from fashion_brain.ask import *  # noqa: F403
from fashion_brain.ask import (  # noqa: F401 — explicit for type checkers / monkeypatch targets
    DEMO_CAVEAT,
    CATALOG_CAVEAT,
    _FIELD_ALIASES,
    _PLAN_GROUPS,
    _append_catalog_answer,
    _grounded,
    _plan_timeout,
    _plan_with_model,
    _reliable_filter,
    _use_agent_graph,
    _user_questions,
    answer_from_catalog,
    answer_question,
    apply_buyer_spelling,
    catalog_focus,
    draft_catalog,
    plan_catalog_question,
    plan_routes,
    resolve_catalog_plan,
)
