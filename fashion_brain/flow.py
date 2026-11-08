"""LangGraph ask flow. The model chooses PLM data, web search, or its own answer."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from knowledge_graph.access import Principal
from knowledge_graph.errors import ModelUnavailableError
from knowledge_graph.logging_config import get_logger
from knowledge_graph.models import EvidenceItem, GroundedAnswer
from knowledge_graph.pipeline.service import Platform

from fashion_brain.llm import build_model_provider

LOGGER = get_logger("fashion_brain.flow")

Route = Literal["catalog", "web_search", "direct", "sample_design", "news"]
DIRECT_CAVEAT = "This answer is the model's own reply. It is not from our product records."
WEB_CAVEAT = "This answer can use web search. It is not a count from our products."

ROUTER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "plm",
            "description": (
                "Look up our PLM data: product counts, lists, breakdowns, buyer analysis, "
                "and product specifications or suggested new styles drawn from stored products. "
                "Any stored product field can be filtered or grouped, including gender, color, material, "
                "brand, buyer, season, category, garment type, gauge, gsm, department, designer, "
                "merchandiser, machine type, and other product fields."
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Research public information and write summaries or reports that are not about our products.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query."}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer_directly",
            "description": "Answer from the model when the question needs neither our products nor a web search.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]
ROUTER_SYSTEM = (
    "Choose exactly one tool. "
    "Use plm when the user asks about our stored PLM products: counts, lists, breakdowns, buyer analysis, "
    "product specifications, suggested new styles, or upcoming-season specs built from those products. "
    "Also use plm for system/registry questions: how many sources or knowledge graphs exist, graph descriptions, "
    "attributes and unique values per source, or what is in the shared interconnect. "
    "Use search_web for research, summaries, reports, industry background, countries, and other public information. "
    "Use answer_directly for general knowledge that does not need our products or fresh web pages. "
    "A mention of garments or fashion is a products question only when the user asks about our stored products. "
    "Earlier messages are this conversation. If the latest message rewrites, shortens, translates, or reformats the previous reply, choose answer_directly. "
    "Do not choose answer_directly for product specifications or upcoming-season requests — those use plm."
)

_PLM_ROUTES = frozenset(
    {"specification", "product_count", "shared_products", "products", "live_sql", "system_meta"}
)
DIRECT_SYSTEM = (
    "Answer from your own knowledge. This reply is not from our product records. Say when you are unsure. "
    "Earlier messages are this conversation. When the user asks to rewrite, shorten, or reformat the previous reply, "
    "change that reply and keep its facts. One sentence means exactly one sentence. "
    "Do not ask for text that is already in the conversation. Do not mention clearance."
)
_REWRITE = re.compile(
    r"\b(single sentence|one sentence|one line|in a sentence|shorter|shorten|rephrase|rewrite|reword|translate|bullet points|condense|simplify)\b",
    re.IGNORECASE,
)
WEB_SEARCH_TOOL = [{"type": "openrouter:web_search", "parameters": {"max_results": 5}}]


class AskState(TypedDict, total=False):
    question: str
    history: list[dict[str, str]]
    route: Route
    tool_query: str
    answer: GroundedAnswer


def normalize_history(history: list | None) -> list[dict[str, str]]:
    """Keep the latest user and assistant turns that belong in the next prompt."""

    cleaned: list[dict[str, str]] = []
    for turn in history or []:
        if isinstance(turn, dict):
            role = turn.get("role")
            content = turn.get("content")
        else:
            role = getattr(turn, "role", None)
            content = getattr(turn, "content", None)
        if role not in {"user", "assistant"}:
            continue
        text = str(content or "").strip()
        if not text:
            continue
        cleaned.append({"role": role, "content": text[:12000]})
    return cleaned[-8:]


def chat_messages(system: str, question: str, history: list | None = None) -> list[dict[str, str]]:
    """System prompt, earlier turns, then the latest user question."""

    messages = [{"role": "system", "content": system}]
    messages.extend(normalize_history(history))
    messages.append({"role": "user", "content": question})
    return messages


def continues_previous(question: str, history: list | None) -> bool:
    """True when this message asks to change the reply already in the chat."""

    return bool(normalize_history(history) and _REWRITE.search(question))


def follow_up_prompt(question: str, history: list | None = None) -> str:
    prior = normalize_history(history)
    if not prior:
        return question
    lines = []
    for turn in prior:
        speaker = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{speaker}:\n{turn['content']}")
    lines.append(f"User:\n{question}")
    return "\n\n".join(lines)


def ask_with_graph(
    platform: Platform,
    question: str,
    principal: Principal,
    history: list | None = None,
    graph_id: str | None = None,
) -> GroundedAnswer:
    from fashion_brain.trace import current_trace, end_trace, start_trace, trace_step

    owns_trace = current_trace() is None
    if owns_trace:
        start_trace(question, source="graph/ask")
    try:
        trace_step("intake", "LangGraph ask started", graph_id=graph_id)
        graph = _compile(platform, principal, graph_id=graph_id)
        try:
            state = graph.invoke({"question": question, "history": normalize_history(history)})
        except Exception as exc:
            LOGGER.info("agent graph failed error=%s; catalog pipeline used", exc.__class__.__name__)
            trace_step("route", "Agent graph failed → catalog", error=exc.__class__.__name__)
            from fashion_brain.agent import answer_from_catalog

            answer = answer_from_catalog(platform, question, principal, history, graph_id=graph_id)
            answer.unavailable.append(f"Agent router did not run ({exc.__class__.__name__}). The PLM data answer is returned.")
            if owns_trace:
                end_trace(status="fallback", routes=answer.routes)
            return answer
        answer = state["answer"]
        if owns_trace:
            end_trace(status="ok", routes=getattr(answer, "routes", None))
        return answer
    except Exception:
        if owns_trace:
            end_trace(status="error")
        raise


def _compile(platform: Platform, principal: Principal, graph_id: str | None = None):
    def route_question(state: AskState) -> dict:
        from fashion_brain.agents import requests_sample_design
        from fashion_brain.trace import trace_step

        if requests_sample_design(state["question"], state.get("history")):
            LOGGER.info("agent router selected route=sample_design")
            trace_step("route", "Selected sample_design")
            return {"route": "sample_design", "tool_query": state["question"]}
        if continues_previous(state["question"], state.get("history")):
            LOGGER.info("agent router selected route=direct reason=follow_up")
            trace_step("route", "Selected direct (follow-up)")
            return {"route": "direct", "tool_query": state["question"]}
        from fashion_brain.agent import plan_routes

        planned = plan_routes(state["question"])
        if planned == ["news"]:
            LOGGER.info("agent router selected route=news")
            trace_step("route", "Selected news")
            return {"route": "news", "tool_query": state["question"]}
        if planned and planned[0] in _PLM_ROUTES:
            LOGGER.info("agent router selected route=catalog reason=planned_%s", planned[0])
            trace_step("route", "Selected catalog (planned PLM)", planned=planned[0])
            return {"route": "catalog", "tool_query": state["question"]}
        provider = build_model_provider(platform.settings)
        try:
            body = provider.chat(
                chat_messages(ROUTER_SYSTEM, state["question"], state.get("history")),
                timeout=platform.settings.request_timeout_seconds,
                tools=ROUTER_TOOLS,
                tool_choice="required",
            )
        except ModelUnavailableError as exc:
            LOGGER.info("agent router unavailable error=%s; catalog selected", exc.message)
            trace_step("route", "LLM router unavailable → catalog", error=exc.message)
            return {"route": "catalog", "tool_query": state["question"]}
        route, query = _route_from_body(body, state["question"])
        LOGGER.info("agent router selected route=%s", route)
        trace_step("route", f"Selected {route}", query=query if route == "web_search" else None)
        return {"route": route, "tool_query": query}

    def use_catalog(state: AskState) -> dict:
        from fashion_brain.agent import answer_from_catalog

        answer = answer_from_catalog(platform, state["question"], principal, state.get("history"), graph_id=graph_id)
        if _catalog_declined(answer):
            LOGGER.info("catalog has no answer; web search selected")
            web = search_web(state)["answer"]
            web.routes = ["catalog", *[route for route in web.routes if route != "catalog"]]
            web.unavailable.insert(0, "Our products do not cover this question.")
            return {"answer": web}
        return {"answer": answer.model_copy(update={"routes": ["catalog", *answer.routes]})}

    def search_web(state: AskState) -> dict:
        provider = build_model_provider(platform.settings)
        query = state.get("tool_query") or state["question"]
        tools = WEB_SEARCH_TOOL if provider.name == "openrouter" else None
        try:
            body = provider.chat(
                chat_messages(
                    (
                        "Answer in the form the user asked for, including a summary report when they request one. "
                        "Use web search for public facts and research. "
                        "If you already know the answer, answer without searching. "
                        "If the user is only rewriting the previous reply, rewrite it and do not search."
                    ),
                    state["question"] if state.get("history") else query,
                    state.get("history"),
                ),
                timeout=platform.settings.request_timeout_seconds,
                tools=tools,
            )
        except ModelUnavailableError as exc:
            answer = _direct_answer(platform, state["question"], state.get("history"))
            answer.routes = ["web_search", "direct"]
            answer.unavailable.append(f"Web search did not run ({exc.message}). The model's own answer is returned.")
            return {"answer": answer}
        message = _message(body)
        content = str(message.get("content") or "").strip()
        if not content:
            answer = _direct_answer(platform, state["question"], state.get("history"))
            answer.routes = ["web_search", "direct"]
            answer.unavailable.append("Web search returned no text. The model's own answer is returned.")
            return {"answer": answer}
        return {
            "answer": GroundedAnswer(
                question=state["question"],
                routes=["web_search"],
                answer=content,
                evidence=_web_evidence(message),
                caveats=[WEB_CAVEAT],
                synthetic_data=False,
            )
        }

    def answer_directly(state: AskState) -> dict:
        return {"answer": _direct_answer(platform, state["question"], state.get("history"))}

    def design_sample(state: AskState) -> dict:
        return {"answer": _sample_design_answer(platform, state["question"], state.get("history"))}

    def use_news(state: AskState) -> dict:
        from fashion_brain.agent import answer_from_catalog

        return {
            "answer": answer_from_catalog(
                platform, state["question"], principal, state.get("history"), graph_id=graph_id
            )
        }

    graph = StateGraph(AskState)
    graph.add_node("router", route_question)
    graph.add_node("catalog", use_catalog)
    graph.add_node("web_search", search_web)
    graph.add_node("direct", answer_directly)
    graph.add_node("sample_design", design_sample)
    graph.add_node("news", use_news)
    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        _select_route,
        {
            "catalog": "catalog",
            "web_search": "web_search",
            "direct": "direct",
            "sample_design": "sample_design",
            "news": "news",
        },
    )
    graph.add_edge("catalog", END)
    graph.add_edge("web_search", END)
    graph.add_edge("direct", END)
    graph.add_edge("sample_design", END)
    graph.add_edge("news", END)
    return graph.compile()


def _catalog_declined(answer: GroundedAnswer) -> bool:
    return answer.answer.startswith("No count was estimated")


def _select_route(state: AskState) -> Route:
    route = state.get("route", "catalog")
    if route not in {"catalog", "web_search", "direct", "sample_design", "news"}:
        return "catalog"
    return route


def _route_from_body(body: dict, question: str) -> tuple[Route, str]:
    calls = _message(body).get("tool_calls") or []
    if not calls:
        return "catalog", question
    function = calls[0].get("function") or {}
    name = function.get("name")
    try:
        arguments = json.loads(function.get("arguments") or "{}")
    except json.JSONDecodeError:
        arguments = {}
    query = str(arguments.get("query") or question) if isinstance(arguments, dict) else question
    if name == "search_web":
        return "web_search", query
    if name == "answer_directly":
        return "direct", question
    return "catalog", question


def _message(body: dict) -> dict:
    try:
        message = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return {}
    return message if isinstance(message, dict) else {}


def _direct_answer(platform: Platform, question: str, history: list | None = None) -> GroundedAnswer:
    provider = build_model_provider(platform.settings)
    try:
        text = provider.complete(
            follow_up_prompt(question, history),
            timeout=platform.settings.request_timeout_seconds,
            system=DIRECT_SYSTEM,
        )
    except ModelUnavailableError as exc:
        return GroundedAnswer(
            question=question,
            routes=["direct"],
            answer="No answer was written. The model is unavailable.",
            unavailable=[exc.message],
            caveats=[DIRECT_CAVEAT],
            synthetic_data=False,
        )
    return GroundedAnswer(
        question=question,
        routes=["direct"],
        answer=text,
        caveats=[DIRECT_CAVEAT],
        synthetic_data=False,
    )


def iter_ask_events(
    platform: Platform,
    question: str,
    principal: Principal,
    history: list | None = None,
    graph_id: str | None = None,
):
    """Yield websocket deltas: thinking, reasoning, answer, then one done event."""

    from fashion_brain.agent import _use_agent_graph
    from fashion_brain.trace import end_trace, start_trace, trace_step

    trace = start_trace(question, source="ws/ask")
    history = normalize_history(history)
    trace_step(
        "intake",
        "Question accepted",
        history_turns=len(history or []),
        graph_id=graph_id,
        clearance=getattr(principal, "clearance", None),
        agent_graph=_use_agent_graph(platform),
    )
    try:
        yield from _iter_ask_events_traced(platform, question, principal, history, graph_id=graph_id)
        end_trace(status="ok")
    except Exception as exc:
        trace.warn("Ask pipeline failed", error=f"{exc.__class__.__name__}: {exc}")
        end_trace(status="error")
        raise


def _iter_ask_events_traced(
    platform: Platform,
    question: str,
    principal: Principal,
    history: list | None = None,
    graph_id: str | None = None,
):
    from fashion_brain.agent import _use_agent_graph
    from fashion_brain.trace import trace_step

    history = normalize_history(history)
    if not _use_agent_graph(platform):
        trace_step("route", "Agent graph off → static catalog path")
        yield from _iter_static_catalog(platform, question, principal, history, graph_id=graph_id)
        return

    from fashion_brain.agents import requests_sample_design

    if requests_sample_design(question, history):
        trace_step("route", "Selected sample_design")
        yield {"type": "thinking", "text": "Intent: design a sample from the product specification.\n"}
        yield {"type": "thinking", "text": "Selected sample design. "}
        yield from _iter_sample_design(platform, question, history)
        return

    if continues_previous(question, history):
        trace_step("route", "Selected direct (follow-up rewrite)")
        yield {"type": "thinking", "text": "Intent: rewrite the previous reply in the requested form.\n"}
        yield {"type": "thinking", "text": "Selected direct. "}
        yield from _iter_direct_stream(platform, question, history)
        return

    from fashion_brain.agent import plan_routes

    planned = plan_routes(question)
    trace_step("route", "Keyword route plan", planned=planned)
    if planned == ["news"]:
        trace_step("route", "Selected news")
        yield {"type": "thinking", "text": "Intent: answer from the news graph.\n"}
        yield {"type": "thinking", "text": "Searching the news graph. "}
        yield from _iter_news_stream(platform, question, principal, history, graph_id=graph_id)
        return

    from fashion_brain.agents import ThinkingBrain

    trace_step("thinking", "ThinkingBrain started")
    summary = ThinkingBrain().read(platform, question, history)
    trace_step(
        "thinking",
        "ThinkingBrain finished",
        intent=summary.actual_intent,
        filters=summary.filters,
        group_by=summary.group_by,
        prior_chat=summary.references_prior_chat,
    )
    yield {"type": "thinking", "text": summary.text + "\n"}

    if planned and planned[0] in _PLM_ROUTES:
        trace_step("route", "Selected catalog (planned PLM route)", planned=planned[0])
        yield {"type": "thinking", "text": "Choosing how to answer. "}
        yield {"type": "thinking", "text": "Selected catalog. "}
        yield from _iter_catalog_stream(platform, question, principal, history, summary, graph_id=graph_id)
        return

    provider = build_model_provider(platform.settings)
    yield {"type": "thinking", "text": "Choosing how to answer. "}
    trace_step("route", "LLM router started")
    try:
        body = provider.chat(
            chat_messages(ROUTER_SYSTEM, question, history),
            timeout=platform.settings.request_timeout_seconds,
            tools=ROUTER_TOOLS,
            tool_choice="required",
        )
        route, query = _route_from_body(body, question)
    except ModelUnavailableError as exc:
        LOGGER.info("agent router unavailable error=%s; catalog selected", exc.message)
        route, query = "catalog", question
        trace_step("route", "LLM router unavailable → catalog", error=exc.message)
    trace_step("route", f"Selected {route}", query=query if route == "web_search" else None)
    yield {"type": "thinking", "text": f"Selected {route.replace('_', ' ')}. "}
    if route == "web_search":
        yield from _iter_web_stream(platform, question, query, history)
    elif route == "direct":
        yield from _iter_direct_stream(platform, question, history)
    elif route == "sample_design":
        yield from _iter_sample_design(platform, question, history)
    else:
        yield from _iter_catalog_stream(platform, question, principal, history, summary, graph_id=graph_id)


def _iter_news_stream(
    platform: Platform,
    question: str,
    principal: Principal,
    history: list | None = None,
    graph_id: str | None = None,
):
    from fashion_brain.agent import answer_from_catalog

    answer = answer_from_catalog(platform, question, principal, history, graph_id=graph_id)
    yield from _deltas("reasoning", _evidence_reasoning(answer.evidence, fallback="No news fact matched this question."))
    yield from _deltas("answer", answer.answer)
    yield from _iter_source_events(answer.evidence)
    yield {"type": "done", "answer": answer.model_dump()}


def _iter_static_catalog(
    platform: Platform,
    question: str,
    principal: Principal,
    history: list | None = None,
    graph_id: str | None = None,
):
    from fashion_brain.agent import answer_from_catalog
    from fashion_brain.agents import ThinkingBrain

    summary = ThinkingBrain().read(platform, question, history)
    yield {"type": "thinking", "text": summary.text + "\n"}
    yield {"type": "thinking", "text": "Reading our product data. "}
    answer = answer_from_catalog(platform, question, principal, history, graph_id=graph_id)
    yield from _deltas("reasoning", _catalog_reasoning_text(answer.answer, answer.evidence, answer.caveats))
    yield from _deltas("answer", answer.answer)
    yield from _iter_source_events(answer.evidence)
    yield {"type": "done", "answer": answer.model_dump()}


def _iter_catalog_stream(
    platform: Platform,
    question: str,
    principal: Principal,
    history: list | None = None,
    summary=None,
    graph_id: str | None = None,
):
    from fashion_brain.agent import _grounded, catalog_focus, draft_catalog
    from fashion_brain.agents import ResponseAgent, response_messages
    from fashion_brain.trace import trace_step

    yield {"type": "thinking", "text": "Searching our product data. "}
    trace_step("catalog", "Draft catalog / orchestration started", graph_id=graph_id)
    retrieved, routes, intention = draft_catalog(platform, question, principal, history, summary, graph_id=graph_id)
    focus = catalog_focus(intention)
    trace_step(
        "catalog",
        "Draft catalog finished",
        routes=routes,
        focus=focus,
        finding_chars=len(retrieved.text or ""),
        evidence=len(retrieved.evidence or []),
    )
    if focus:
        yield {"type": "thinking", "text": focus + " "}
    if retrieved.text.startswith("No count was estimated"):
        trace_step("catalog", "No catalog answer → falling back to web")
        yield {"type": "thinking", "text": "Our products do not cover this question. "}
        for event in _iter_web_stream(platform, question, question, []):
            if event.get("type") == "done":
                answer = GroundedAnswer.model_validate(event["answer"])
                answer.routes = ["catalog", *[item for item in answer.routes if item != "catalog"]]
                answer.unavailable.insert(0, "Our products do not cover this question.")
                event = {"type": "done", "answer": answer.model_dump()}
            yield event
        return

    # Reasoning = grounded findings (counts/breakdown), not raw sample product excerpts.
    reasoning = _catalog_reasoning_text(retrieved.text, retrieved.evidence, retrieved.caveats)
    trace_step("reasoning", "Streaming grounded findings", chars=len(reasoning))
    yield from _deltas("reasoning", reasoning)
    # Analysis / chart findings must reach the UI intact (chart fence + table).
    # The response LLM often shrinks these to two sentences and drops ```chart blocks.
    from fashion_brain.agents._impl import _findings_have_chart, _wants_full_breakdown

    if _wants_full_breakdown(question, retrieved.text) and (
        _findings_have_chart(retrieved.text) or "|" in (retrieved.text or "")
    ):
        text = retrieved.text.strip()
        trace_step("response", "Analysis/chart findings returned directly", chars=len(text))
        yield from _deltas("answer", text)
    else:
        parts: list[str] = []
        try:
            trace_step("response", "Response model stream started")
            for event in _iter_model(
                build_model_provider(platform.settings),
                response_messages(question, retrieved),
                timeout=platform.settings.request_timeout_seconds,
            ):
                # Model chain-of-thought is formatting noise; catalog reasoning already streamed above.
                if event["type"] == "reasoning":
                    continue
                if event["type"] == "answer":
                    parts.append(event["text"])
                yield event
            text = "".join(parts).strip()
            trace_step("response", "Response model stream finished", chars=len(text))
        except ModelUnavailableError:
            text = ""
            trace_step("response", "Response model unavailable → template compose")
        if not text:
            text = ResponseAgent().compose(platform, question, retrieved)
            trace_step("response", "Template response composed", chars=len(text))
            yield from _deltas("answer", text)
        elif _findings_have_chart(retrieved.text) and not _findings_have_chart(text):
            text = retrieved.text.strip()
            trace_step("response", "Restored chart findings after LLM drop", chars=len(text))
            yield from _deltas("answer", text)
    answer = _grounded(question, ["catalog", *routes], text, retrieved, platform)
    trace_step("sources", "Streaming evidence/references", count=len(answer.evidence or []))
    yield from _iter_source_events(answer.evidence)
    trace_step("done", "Catalog stream complete", routes=answer.routes, answer_chars=len(answer.answer or ""))
    yield {"type": "done", "answer": answer.model_dump()}


def _iter_web_stream(platform: Platform, question: str, query: str, history: list | None = None):
    provider = build_model_provider(platform.settings)
    yield {"type": "reasoning", "text": "Checking public sources. "}
    sink: dict = {}
    parts: list[str] = []
    tools = WEB_SEARCH_TOOL if provider.name == "openrouter" else None
    try:
        for event in _iter_model(
            provider,
            chat_messages(
                (
                    "Answer in the form the user asked for, including a summary report when they request one. "
                    "Use web search for public facts and research. "
                    "If you already know the answer, answer without searching. "
                    "If the user is only rewriting the previous reply, rewrite it and do not search."
                ),
                question if history else query,
                history,
            ),
            timeout=platform.settings.request_timeout_seconds,
            tools=tools,
            sink=sink,
        ):
            if event["type"] == "answer":
                parts.append(event["text"])
            yield event
    except ModelUnavailableError as exc:
        answer = _direct_answer(platform, question, history)
        answer.routes = ["web_search", "direct"]
        answer.unavailable.append(f"Web search did not run ({exc.message}). The model's own answer is returned.")
        yield from _deltas("answer", answer.answer)
        yield from _iter_source_events(answer.evidence)
        yield {"type": "done", "answer": answer.model_dump()}
        return
    text = "".join(parts).strip()
    if not text:
        answer = _direct_answer(platform, question, history)
        answer.routes = ["web_search", "direct"]
        answer.unavailable.append("Web search returned no text. The model's own answer is returned.")
        yield from _deltas("answer", answer.answer)
        yield from _iter_source_events(answer.evidence)
        yield {"type": "done", "answer": answer.model_dump()}
        return
    message = _message(sink.get("body") or {})
    answer = GroundedAnswer(
        question=question,
        routes=["web_search"],
        answer=text,
        evidence=_web_evidence(message),
        caveats=[WEB_CAVEAT],
        synthetic_data=False,
    )
    yield from _iter_source_events(answer.evidence)
    yield {"type": "done", "answer": answer.model_dump()}


_IMAGE_EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg", "image/webp": "webp"}


def _store_sample_image(directory: Path, data: bytes, media: str) -> str:
    extension = _IMAGE_EXTENSIONS.get(media.split(";")[0].strip().lower(), "png")
    name = f"{uuid.uuid4().hex}.{extension}"
    folder = directory / "samples"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_bytes(data)
    return f"/samples/{name}"


def _sample_image_markdown(platform: Platform, specification: str) -> str:
    """Draw one studio image from the specification. Return markdown, or a short failure line."""

    if platform.settings.answer_mode != "llm" or not specification.strip():
        return ""
    provider = build_model_provider(platform.settings)
    generate = getattr(provider, "generate_image", None)
    if not callable(generate):
        return ""
    from fashion_brain.agents import sample_image_prompt

    try:
        data, media = generate(
            sample_image_prompt(specification),
            model=platform.settings.image_model,
            timeout=platform.settings.request_timeout_seconds,
        )
    except ModelUnavailableError as exc:
        LOGGER.info("sample image unavailable error=%s", exc.message)
        return "\n\nThe sample image was not generated."
    if not data:
        return "\n\nThe sample image was not generated."
    return "\n\n![Sample design](" + _store_sample_image(platform.settings.generated, data, media) + ")"


def _sample_design_answer(platform: Platform, question: str, history: list | None) -> GroundedAnswer:
    from fashion_brain.agents import SampleDesignAgent, _specification_text

    text = SampleDesignAgent().design(platform, question, history)
    text = (text + _sample_image_markdown(platform, _specification_text(history))).strip()
    return GroundedAnswer(
        question=question,
        routes=["sample_design"],
        answer=text,
        caveats=["This sample design uses the product specification from this conversation."],
        synthetic_data=False,
    )


def _iter_sample_design(platform: Platform, question: str, history: list | None = None):
    from fashion_brain.agents import SampleDesignAgent, sample_design_messages, sample_design_sheet, _specification_text

    specification = _specification_text(history)
    sheet = sample_design_sheet(specification)
    yield {"type": "reasoning", "text": "Reading the product specification and preparing the sample design. "}
    provider = build_model_provider(platform.settings)
    parts: list[str] = []
    try:
        for event in _iter_model(
            provider,
            sample_design_messages(question, specification),
            timeout=platform.settings.request_timeout_seconds,
        ):
            if event["type"] == "answer":
                parts.append(event["text"])
            else:
                yield event
    except ModelUnavailableError:
        parts = []
    text = "".join(parts).strip()
    if not text or re.search(r"can.?t physically|cannot physically|unable to manufacture|can.?t manufacture", text, re.IGNORECASE):
        text = sheet or SampleDesignAgent().design(platform, question, history)
    yield from _deltas("answer", text)
    yield {"type": "thinking", "text": "Drawing the sample. "}
    image = _sample_image_markdown(platform, specification)
    if image:
        text = (text + image).strip()
        yield from _deltas("answer", image)
    answer = GroundedAnswer(
        question=question,
        routes=["sample_design"],
        answer=text,
        caveats=["This sample design uses the product specification from this conversation."],
        synthetic_data=False,
    )
    yield {"type": "done", "answer": answer.model_dump()}


def _iter_direct_stream(platform: Platform, question: str, history: list | None = None):
    provider = build_model_provider(platform.settings)
    yield {"type": "reasoning", "text": "Using the previous reply. " if normalize_history(history) else "Using the model's own knowledge. "}
    parts: list[str] = []
    try:
        for event in _iter_model(
            provider,
            chat_messages(DIRECT_SYSTEM, question, history),
            timeout=platform.settings.request_timeout_seconds,
        ):
            if event["type"] == "answer":
                parts.append(event["text"])
            yield event
    except ModelUnavailableError as exc:
        answer = _direct_answer(platform, question, history)
        if not answer.unavailable:
            answer.unavailable.append(exc.message)
        yield from _deltas("answer", answer.answer)
        yield {"type": "done", "answer": answer.model_dump()}
        return
    text = "".join(parts).strip() or "No answer was written."
    answer = GroundedAnswer(
        question=question,
        routes=["direct"],
        answer=text,
        caveats=[DIRECT_CAVEAT],
        synthetic_data=False,
    )
    yield {"type": "done", "answer": answer.model_dump()}


def _iter_model(provider, messages, *, timeout: float | None, tools=None, tool_choice=None, sink: dict | None = None, remap: dict | None = None):
    if hasattr(provider, "iter_chat"):
        for kind, text in provider.iter_chat(messages, timeout=timeout, tools=tools, tool_choice=tool_choice, sink=sink):
            kind = (remap or {}).get(kind, kind)
            if text:
                yield {"type": kind, "text": text}
        return
    body = provider.chat(messages, timeout=timeout, tools=tools, tool_choice=tool_choice)
    if sink is not None:
        sink["body"] = body
    message = _message(body)
    for kind, key in (("thinking", "thinking"), ("reasoning", "reasoning"), ("answer", "content")):
        text = message.get(key) or ""
        mapped = (remap or {}).get(kind, kind)
        if text:
            yield {"type": mapped, "text": str(text)}


def _catalog_reasoning_text(findings: str, evidence: list | None = None, caveats: list | None = None) -> str:
    """Explain how the catalog answer was grounded — counts/breakdown, not raw SKU lines.

    Chart/table markdown belongs in the answer bubble (FBMarkdown), not the plain-text
    reasoning panel — dumping fences here made plots look "broken" in the UI.
    """

    parts: list[str] = []
    text = (findings or "").strip()
    has_viz = "```chart" in text or "```fbchart" in text or ("|" in text and "---" in text)
    if has_viz:
        headline = next((line.strip() for line in text.splitlines() if line.strip()), "")
        if headline and not headline.startswith("```") and not headline.startswith("|"):
            parts.append(headline)
        parts.append("Breakdown chart and table are rendered in the answer below.")
    elif text:
        parts.append(text[:800])
    graphs = sorted(
        {
            str(getattr(item, "graph", None) or getattr(item, "source_id", "") or "").strip()
            for item in (evidence or [])
            if getattr(item, "graph", None) or getattr(item, "source_id", None)
        }
    )
    graphs = [name for name in graphs if name]
    if graphs:
        parts.append(f"Source graph: {', '.join(graphs)}.")
    if evidence:
        parts.append(f"Checked {len(evidence)} sample product row(s) as supporting evidence.")
    for note in caveats or []:
        note = str(note or "").strip()
        if note and note not in text and note not in " ".join(parts):
            parts.append(note)
    return " ".join(parts)[:1600] if parts else "No grounded product findings were returned."


def _evidence_reasoning(evidence: list | None, *, fallback: str) -> str:
    """News-style reasoning: titled facts, not a blob of concatenated excerpts."""

    lines: list[str] = []
    for item in evidence or []:
        title = (getattr(item, "title", None) or getattr(item, "record_id", None) or "").strip()
        excerpt = (getattr(item, "excerpt", None) or "").strip()
        if title and excerpt:
            lines.append(f"{title}: {excerpt}")
        elif excerpt:
            lines.append(excerpt)
        elif title:
            lines.append(title)
        if len(lines) >= 5:
            break
    body = " | ".join(lines).strip()
    return (body[:1200] if body else fallback)


def _deltas(kind: str, text: str, step: int = 64):
    for index in range(0, len(text), step):
        yield {"type": kind, "text": text[index : index + step]}


def _iter_source_events(evidence: list | None):
    """Stream sources/references after the answer so the UI can paint them live."""

    for item in evidence or []:
        if hasattr(item, "model_dump"):
            payload = item.model_dump()
        elif isinstance(item, dict):
            payload = dict(item)
        else:
            continue
        yield {"type": "source", "item": payload}


def _web_evidence(message: dict) -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []
    for item in message.get("annotations") or []:
        if not isinstance(item, dict):
            continue
        citation = item.get("url_citation") if item.get("type") == "url_citation" else item
        if not isinstance(citation, dict):
            continue
        url = str(citation.get("url") or "").strip()
        if not url:
            continue
        title = str(citation.get("title") or url)
        excerpt = str(citation.get("content") or citation.get("text") or title)
        evidence.append(
            EvidenceItem(
                source_id="web",
                record_id=url,
                title=title,
                excerpt=excerpt,
                url=url,
                verification_status="observed",
                synthetic=False,
            )
        )
    return evidence
