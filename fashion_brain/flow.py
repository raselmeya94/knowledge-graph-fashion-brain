"""LangGraph ask flow. The model chooses catalog KG data, web search, or its own answer."""

from __future__ import annotations

import json
import math
import re
import time
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

from fashion_brain.llm import build_image_provider, build_model_provider, resolve_image_model

LOGGER = get_logger("fashion_brain.flow")

Route = Literal["catalog", "web_search", "direct", "sample_design", "news"]
DIRECT_CAVEAT = "This answer is the model's own reply. It is not from the product catalog."
WEB_CAVEAT = "This answer can use web search. It is not a count from the product catalog."

ROUTER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "catalog",
            "description": (
                "Look up the registered product catalog / e-commerce knowledge graph: product counts, "
                "lists, breakdowns, brand analysis, price and availability history, and specifications "
                "drawn from stored catalog products. Filter or group by live product fields such as "
                "brand, category, color, material, availability, section, and price."
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Research public information and write summaries or reports that are not catalog lookups.",
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
            "description": "Answer from the model when the question needs neither the product catalog nor a web search.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]
ROUTER_SYSTEM = (
    "Choose exactly one tool after interpreting the latest request and relevant conversation context. "
    "Use catalog only when the user wants facts from the registered product knowledge graph "
    "(currently the e-commerce catalog): counts, product lists, field filters, comparisons, rankings, "
    "breakdowns, brand analysis, price/availability history, product specifications from stored styles, "
    "registered sources/graphs, or the shared interconnect. "
    "This catalog is scraped/synced retail product data — not an internal PLM 'own products' system. "
    "Respect the distinction between a catalog data ask and a general question that merely mentions "
    "fashion, garments, materials, or brands. "
    "Use search_web for public research, current or emerging industry trends, news, market information, "
    "country facts, and reports that require recent or source-backed information. "
    "Use answer_directly for stable general knowledge, explanations, writing, brainstorming, advice, and "
    "other requests that need neither catalog records nor current public information. "
    "For ambiguous wording, infer the user's intended source from the full question and conversation; do not "
    "route every fashion-related question to catalog. For example, 'what is GSM?' is general knowledge, while "
    "'how many Primark products are in stock?' is catalog; 'what are the latest fashion trends?' needs web research, "
    "while 'average price change trends for Primark' needs catalog temporal records. "
    "Earlier messages are this conversation. If the latest message only rewrites, shortens, translates, or "
    "reformats the previous reply, choose answer_directly and preserve its facts. "
    "Product specifications grounded in stored catalog styles use catalog."
)

# These routes may bypass the semantic router only when the question explicitly asks for catalog data.
_PLM_ROUTES = frozenset(
    {"product_count", "product_details", "products", "shared_products", "specification"}
)
DIRECT_SYSTEM = (
    "Answer from your own knowledge. This reply is not from the product catalog. Say when you are unsure. "
    "Earlier messages are this conversation. When the user asks to rewrite, shorten, or reformat the previous reply, "
    "change that reply and keep its facts. One sentence means exactly one sentence. "
    "Do not ask for text that is already in the conversation. Do not mention clearance."
)
VISUALIZE_SYSTEM = (
    "Transform only the immediately previous assistant reply into a clear visualization. "
    "Do not search, query, refresh, or infer any catalog or web data. Preserve every plotted "
    "label, value, sign, and unit from the previous reply. If it contains a markdown table, "
    "create one ```chart JSON``` block using its labels and numeric values. "
    "If the user names pie, bar, or line, set JSON type to that exact word — never override it. "
    "When type is pie and some values are negative, put absolute magnitudes in values "
    "(pie slices cannot be negative) and keep the signed figures in the labels. "
    "Otherwise prefer a bar chart for category comparisons and a line chart only for values "
    "ordered over time. Do not repeat the original table. If the previous reply has no usable "
    "numeric data, say so plainly."
)
_REWRITE = re.compile(
    r"\b(single sentence|one sentence|one line|in a sentence|shorter|shorten|rephrase|rewrite|reword|translate|bullet points|condense|simplify|visuali[sz](?:e|ation|ed|ing)|"
    r"chart|plot|graph|make it visual|show (?:it|that|this) visually)\b",
    re.IGNORECASE,
)
_VISUALIZE = re.compile(
    r"\b(visuali[sz](?:e|ation|ed|ing)|chart|plot|graph|make it visual|show (?:it|that|this) visually)\b",
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
            answer.unavailable.append(
                f"Agent router did not run ({exc.__class__.__name__}). The catalog answer is returned."
            )
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
        if planned and planned[0] in _PLM_ROUTES and _is_explicit_catalog_question(question, planned[0]):
            LOGGER.info("agent router selected route=catalog reason=planned_%s", planned[0])
            trace_step("route", "Selected catalog (planned route)", planned=planned[0])
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
    # "catalog" is current; accept legacy "plm" tool name from older prompts/models.
    if name in {"catalog", "plm", None}:
        return "catalog", question
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
            timeout=_model_timeout(platform),
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
    conversation_state: dict | None = None,
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
        yield from _iter_ask_events_traced(
            platform,
            question,
            principal,
            history,
            graph_id=graph_id,
            conversation_state=conversation_state,
        )
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
    conversation_state: dict | None = None,
):
    from fashion_brain.agent import _use_agent_graph
    from fashion_brain.trace import trace_step

    history = normalize_history(history)

    # Slash triggers first — never let "/advanced_garment_design" become garment_type=design.
    from fashion_brain.agents.slash_commands import parse_slash_command

    slash = parse_slash_command(question)
    if slash is not None:
        yield from _iter_slash_command(
            platform,
            slash,
            principal,
            history,
            graph_id=graph_id,
            conversation_state=conversation_state,
        )
        return

    if _VISUALIZE.search(question) and continues_previous(question, history):
        trace_step("route", "Selected direct (visualization follow-up)")
        yield {"type": "status", "text": "Visualizing the previous results…"}
        yield {"type": "thinking", "text": "Intent: visualize the previous reply without new retrieval.\n"}
        yield from _iter_direct_stream(platform, question, history)
        return

    if not _use_agent_graph(platform):
        trace_step("route", "Agent graph off → static catalog path")
        yield from _iter_static_catalog(
            platform,
            question,
            principal,
            history,
            graph_id=graph_id,
            conversation_state=conversation_state,
        )
        return

    from fashion_brain.agents import requests_sample_design

    if requests_sample_design(question, history):
        trace_step("route", "Selected sample_design")
        yield {"type": "status", "text": "Preparing the sample design…"}
        yield {"type": "thinking", "text": "Intent: design a sample from the product specification.\n"}
        yield {"type": "thinking", "text": "Selected sample design. "}
        yield from _iter_sample_design(platform, question, history)
        return

    if continues_previous(question, history):
        trace_step("route", "Selected direct (follow-up rewrite)")
        yield {"type": "status", "text": "Reworking the previous reply…"}
        yield {"type": "thinking", "text": "Intent: rewrite the previous reply in the requested form.\n"}
        yield {"type": "thinking", "text": "Selected direct. "}
        yield from _iter_direct_stream(platform, question, history)
        return

    from fashion_brain.agent import plan_routes

    planned = plan_routes(question)
    trace_step("route", "Keyword route plan", planned=planned)
    if planned == ["news"]:
        trace_step("route", "Selected news")
        yield {"type": "status", "text": "Searching news sources…"}
        yield {"type": "thinking", "text": "Intent: answer from the news graph.\n"}
        yield {"type": "thinking", "text": "Searching the news graph. "}
        yield from _iter_news_stream(platform, question, principal, history, graph_id=graph_id)
        return

    from fashion_brain.agents import ThinkingBrain

    yield {
        "type": "thinking",
        "text": "Understanding your request against the live catalog. ",
    }
    trace_step("thinking", "ThinkingBrain started")
    summary = yield from _run_with_progress(
        lambda: ThinkingBrain().read(
            platform, question, history, conversation_state=conversation_state
        ),
        progress="Checking your request against live product fields and values…",
    )
    trace_step(
        "thinking",
        "ThinkingBrain finished",
        intent=summary.actual_intent,
        filters=summary.filters,
        group_by=summary.group_by,
        prior_chat=summary.references_prior_chat,
    )
    yield {"type": "thinking", "text": summary.text + "\n"}

    if planned and planned[0] in _PLM_ROUTES and _is_explicit_catalog_question(question, planned[0]):
        trace_step("route", "Selected catalog (planned route)", planned=planned[0])
        yield {"type": "status", "text": "Searching matching product records…"}
        yield {"type": "thinking", "text": "Looking up the product catalog. "}
        yield from _iter_catalog_stream(
            platform,
            question,
            principal,
            history,
            summary,
            graph_id=graph_id,
            conversation_state=conversation_state,
        )
        return

    provider = build_model_provider(platform.settings)
    yield {"type": "status", "text": "Choosing the best source for your question…"}
    trace_step("route", "LLM router started")
    try:
        body = yield from _run_with_progress(
            lambda: provider.chat(
                chat_messages(ROUTER_SYSTEM, question, history),
                timeout=_model_timeout(platform),
                tools=ROUTER_TOOLS,
                tool_choice="required",
            ),
            progress="Still selecting the best source for your question…",
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
        yield from _iter_catalog_stream(
            platform,
            question,
            principal,
            history,
            summary,
            graph_id=graph_id,
            conversation_state=conversation_state,
        )


def _iter_news_stream(
    platform: Platform,
    question: str,
    principal: Principal,
    history: list | None = None,
    graph_id: str | None = None,
):
    from fashion_brain.agent import answer_from_catalog

    answer = yield from _run_with_progress(
        lambda: answer_from_catalog(platform, question, principal, history, graph_id=graph_id),
        progress="Still checking the available news sources…",
    )
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
    conversation_state: dict | None = None,
):
    from fashion_brain.agent import answer_from_catalog
    from fashion_brain.agents import ThinkingBrain

    yield {
        "type": "thinking",
        "text": "Understanding your request against the live catalog. ",
    }
    summary = yield from _run_with_progress(
        lambda: ThinkingBrain().read(
            platform, question, history, conversation_state=conversation_state
        ),
        progress="Checking your request against live product fields and values…",
    )
    yield {"type": "thinking", "text": summary.text + "\n"}
    answer = yield from _run_with_progress(
        lambda: answer_from_catalog(
            platform,
            question,
            principal,
            history,
            graph_id=graph_id,
            conversation_state=conversation_state,
        ),
        progress="Searching matching product records…",
    )
    yield from _deltas(
        "reasoning",
        _catalog_reasoning_text(
            answer.answer, answer.evidence, answer.caveats, question=question
        ),
    )
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
    conversation_state: dict | None = None,
):
    from fashion_brain.agent import _grounded, catalog_focus, draft_catalog
    from fashion_brain.agents import response_messages
    from fashion_brain.agents._impl import _findings_have_table
    from fashion_brain.trace import trace_step

    yield {"type": "status", "text": "Searching matching product records…"}
    # Start response-plan panel while catalog retrieve runs (thinking + plan in parallel in UI).
    provisional = _provisional_reasoning_plan(question, summary)
    if provisional:
        yield from _deltas("reasoning", provisional)
    trace_step("catalog", "Draft catalog / orchestration started", graph_id=graph_id)
    state = conversation_state if conversation_state is not None else {}
    retrieved, routes, intention = yield from _run_with_progress(
        lambda: draft_catalog(
            platform,
            question,
            principal,
            history,
            summary,
            graph_id=graph_id,
            conversation_state=state,
        ),
        progress="Still searching the catalog for matching products…",
    )
    focus = catalog_focus(intention)
    trace_step(
        "catalog",
        "Draft catalog finished",
        routes=routes,
        focus=focus,
        finding_chars=len(retrieved.text or ""),
        evidence=len(retrieved.evidence or []),
    )
    # Analysis plan: spell out multi-facet viz intent in Thinking (not only "Type-wise top…").
    if bool(getattr(intention, "analysis", False)):
        who = (intention.filters or {}).get("brand") or (intention.filters or {}).get("buyer") or "selection"
        try:
            from knowledge_graph.retrieval.ecom import is_analysis_facet_attr

            attrs = [
                str(a)
                for a in (getattr(intention, "available_attributes", None) or [])
                if is_analysis_facet_attr(str(a))
            ][:10]
        except Exception:
            attrs = list(getattr(intention, "available_attributes", None) or [])[:10]
        attr_txt = ", ".join(attrs) if attrs else "live product attributes"
        yield {
            "type": "thinking",
            "text": (
                f"Analysis plan for {who}: overview metrics plus ranked product attributes "
                f"({attr_txt}); response will choose the best presentation. "
            ),
        }
    if focus:
        yield {"type": "thinking", "text": focus + " "}
    yield {"type": "status", "text": "Preparing an answer from the retrieved product data…"}
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

    # Final reasoning plan from findings (provisional Ask/Fetch already streamed above).
    reasoning = _catalog_reasoning_text(
        retrieved.text,
        retrieved.evidence,
        retrieved.caveats,
        question=question,
        intention=intention,
    )
    trace_step("reasoning", "Streaming response plan memo", chars=len(reasoning))
    yield from _deltas("reasoning", "\n\n" + reasoning if provisional else reasoning)
    # Keep structured retrieval output intact instead of paying for a second model call.
    parts: list[str] = []
    text = _canonical_composition_compare_answer(retrieved.text)
    direct_retrieval = (
        intention.intent in {"product_count", "product_details", "count", "breakdown"}
        and not bool(getattr(intention, "analysis", False))
        and not bool(getattr(intention, "suggest", False))
    )
    if not text and (direct_retrieval or _findings_have_table(retrieved.text) or _findings_have_chart(retrieved.text)):
        text = retrieved.text
    streamed_answer = False
    if text:
        text = _finalize_analysis_answer(text, retrieved.text)
        trace_step("response", "Preserved retrieved structured answer", chars=len(text))
        yield from _deltas("answer", text)
        streamed_answer = True
    else:
        try:
            trace_step("response", "Response model stream started")
            for event in _iter_model(
                build_model_provider(platform.settings),
                response_messages(question, retrieved, reasoning_plan=reasoning),
                timeout=_model_timeout(platform),
            ):
                # Keep the user-facing rationale deterministic; never stream private model thoughts.
                if event["type"] in {"reasoning", "thinking"}:
                    continue
                if event["type"] == "answer":
                    parts.append(event["text"])
                    streamed_answer = True
                yield event
            text = "".join(parts).strip()
            trace_step("response", "Response model stream finished", chars=len(text))
        except ModelUnavailableError:
            text = ""
            trace_step("response", "Response model unavailable → template compose")
        if not text:
            text = retrieved.text
            trace_step("response", "Using retrieved findings after response model timeout", chars=len(text))
        text = _finalize_analysis_answer(text, retrieved.text)
        # Replace any streamed draft that still had raw <!--fb:viz:--> markers / missing charts.
        if streamed_answer:
            yield {"type": "answer", "text": text, "replace": True}
        else:
            yield from _deltas("answer", text)
            streamed_answer = True
    if not streamed_answer:
        text = _finalize_analysis_answer(text, retrieved.text)
    answer = _grounded(question, ["catalog", *routes], text, retrieved, platform)
    trace_step("sources", "Streaming evidence/references", count=len(answer.evidence or []))
    yield from _iter_source_events(answer.evidence)
    trace_step("done", "Catalog stream complete", routes=answer.routes, answer_chars=len(answer.answer or ""))
    yield {"type": "done", "answer": answer.model_dump()}


def _iter_web_stream(platform: Platform, question: str, query: str, history: list | None = None):
    provider = build_model_provider(platform.settings)
    yield {"type": "status", "text": "Checking public sources and drafting your answer…"}
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
            timeout=_model_timeout(platform),
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


def _iter_slash_command(
    platform: Platform,
    slash,
    principal: Principal,
    history: list | None,
    *,
    graph_id: str | None = None,
    conversation_state: dict | None = None,
):
    """Handle /trending, /specification, /garment_design, /advanced_garment_design."""

    from fashion_brain.agents.product_spec import (
        NoSpecificationError,
        extract_last_specification,
        plan_design,
        parse_design_command,
        spec_from_text,
    )
    from fashion_brain.agents.slash_commands import slash_needs_clarify, slash_rewrite_question
    from fashion_brain.trace import trace_step

    # Design triggers: LLM reads any pasted/history text into a spec, then draws.
    # Never catalog-rewrite (that invented garment_type=design).
    if slash.name in {"garment_design", "advanced_garment_design"}:
        yield {"type": "status", "text": f"/{slash.name}"}
        yield {
            "type": "thinking",
            "text": f"Intent: /{slash.name} — reading the product specification for sample views.\n",
        }
        if not slash.body and extract_last_specification(history, platform=platform) is None:
            clarify = slash_needs_clarify(slash, has_specification=False) or "Need a product specification first."
            trace_step("route", f"Slash /{slash.name} needs clarification")
            yield from _deltas("answer", clarify)
            answer = GroundedAnswer(
                question=f"/{slash.name}",
                routes=["slash", slash.name],
                answer=clarify,
                caveats=["No product specification available for garment images."],
                synthetic_data=False,
            )
            yield {"type": "done", "answer": answer.model_dump()}
            return
        command = parse_design_command(
            f"/{slash.name}" + (f" {slash.body}" if slash.body else "")
        )
        if command is None:
            command = parse_design_command(
                "/advanced_garment_design" if slash.name.startswith("advanced") else "/garment_design"
            )
        pasted_spec = spec_from_text(slash.body, platform=platform) if slash.body else None
        try:
            plan = plan_design(command, history, spec=pasted_spec, platform=platform)
        except NoSpecificationError:
            clarify = (
                "I need a product specification to draw from — paste one, "
                "use one from this chat, or run /specification first."
            )
            yield {"type": "thinking", "text": f"Intent: /{slash.name} — no usable specification yet.\n"}
            yield from _deltas("answer", clarify)
            answer = GroundedAnswer(
                question=f"/{slash.name}",
                routes=["slash", slash.name],
                answer=clarify,
                caveats=["No product specification available for garment images."],
                synthetic_data=False,
            )
            yield {"type": "done", "answer": answer.model_dump()}
            return
        source = "your input" if pasted_spec else "this chat"
        trace_step(
            "route",
            f"Slash /{slash.name} → garment image design",
            shots=len(plan.shots),
            spec_source=source,
        )
        yield {"type": "thinking", "text": f"Intent: /{slash.name} — drawing from {source}.\n"}
        yield from _iter_garment_design_shots(platform, plan)
        return

    clarify = slash_needs_clarify(slash, has_specification=False)
    if clarify:
        trace_step("route", f"Slash /{slash.name} needs clarification")
        yield {"type": "status", "text": f"/{slash.name}"}
        yield {"type": "thinking", "text": f"Intent: /{slash.name} — need a bit more detail.\n"}
        yield from _deltas("answer", clarify)
        answer = GroundedAnswer(
            question=f"/{slash.name}",
            routes=["slash", slash.name],
            answer=clarify,
            caveats=["Slash command needs more detail before running."],
            synthetic_data=False,
        )
        yield {"type": "done", "answer": answer.model_dump()}
        return

    rewritten = slash_rewrite_question(slash)
    if not rewritten:
        clarify = slash_needs_clarify(slash, has_specification=False) or "Please add more detail after the command."
        yield from _deltas("answer", clarify)
        answer = GroundedAnswer(
            question=f"/{slash.name}",
            routes=["slash", slash.name],
            answer=clarify,
            caveats=["Slash command needs more detail before running."],
            synthetic_data=False,
        )
        yield {"type": "done", "answer": answer.model_dump()}
        return

    trace_step("route", f"Slash /{slash.name} rewritten", rewritten=rewritten[:160])
    yield {"type": "thinking", "text": f"Intent: /{slash.name} → {rewritten}\n"}
    yield from _iter_ask_events_traced(
        platform,
        rewritten,
        principal,
        history,
        graph_id=graph_id,
        conversation_state=conversation_state,
    )


def _sample_board_value(item) -> str:
    if item is None:
        return ""
    if item.count is not None and item.total:
        return f"{item.value} ({item.count} of {item.total})"
    return str(item.value or "")


def _advanced_sample_board_json(plan, shot_rows: list[dict]) -> str:
    """One collage payload for the chat UI (header, hero, views, spec table, footer)."""

    spec = plan.spec
    buyer = (spec.text("buyer") or spec.text("brand") or "Fashion Brain").strip()
    season = (spec.text("season") or "").strip()
    total = spec.total
    if total and season:
        subtitle = f"Based on {total:,} {season.title()} Products ({buyer.title()})"
    elif total:
        subtitle = f"Based on {total:,} Products ({buyer.title()})"
    else:
        subtitle = f"From the product specification ({buyer.title()})"

    row_keys = (
        ("buyer", "Buyer"),
        ("season", "Season"),
        ("garment_type", "Garment"),
        ("category", "Category"),
        ("plm_sub_sub_category", "Sub Category"),
        ("material", "Material"),
        ("composition", "Composition"),
        ("color", "Colour"),
        ("gauge", "Gauge"),
        ("gender", "Gender"),
        ("gsm", "GSM"),
        ("knit_machine_type", "Knit Machine"),
    )
    rows = []
    for key, label in row_keys:
        item = spec.top(key)
        if item and item.value:
            rows.append([label, _sample_board_value(item)])

    chips: list[str] = []
    if season:
        chips.append(f"{season.title()} Collection")
    for key in ("ai_material_tags", "ai_construction_tags", "garment_type", "category"):
        for item in (spec.values.get(key) or [])[:2]:
            label = str(item.value or "").strip()
            if label and label.lower() not in {c.lower() for c in chips}:
                chips.append(label.title() if label.islower() else label)
            if len(chips) >= 4:
                break
        if len(chips) >= 4:
            break

    garment = spec.text("garment_type") or spec.text("category") or "garment"
    tagline = f"Stylish. Comfortable. Perfect for {season.title()}." if season else f"Suggested common style — {garment}."
    payload = {
        "title": "Garment Sample – Suggested Common Style",
        "subtitle": subtitle,
        "buyer": buyer.upper(),
        "shots": shot_rows,
        "rows": rows,
        "chips": chips[:4],
        "tagline": tagline,
    }
    return "```sampleboard\n" + json.dumps(payload, ensure_ascii=False) + "\n```"


def _iter_garment_design_shots(platform: Platform, plan):
    """Generate one or many sample images from a DesignPlan and stream markdown."""

    from fashion_brain.agents.product_spec import DesignPlan

    assert isinstance(plan, DesignPlan)
    mode_label = "advanced multi-view board" if plan.mode == "advanced" else "sample image"
    yield {"type": "status", "text": f"Generating {mode_label}…"}
    yield {
        "type": "thinking",
        "text": f"Intent: /{'advanced_garment_design' if plan.mode == 'advanced' else 'garment_design'} from the last product specification.\n",
    }

    if plan.mode == "advanced":
        shot_rows: list[dict] = []
        for index, shot in enumerate(plan.shots, start=1):
            # Space OpenRouter image calls so in-flight credit holds can clear between tiles.
            if index > 1:
                yield {
                    "type": "status",
                    "text": f"Pacing before {shot.title} ({index}/{len(plan.shots)})…",
                }
                time.sleep(6.0)
            yield {"type": "status", "text": f"Drawing {shot.title} ({index}/{len(plan.shots)})…"}
            markdown, err = _generate_shot_markdown(platform, shot)
            url = ""
            error = ""
            if markdown:
                match = re.search(r"!\[[^\]]*\]\(([^)]+)\)", markdown)
                url = match.group(1) if match else ""
            elif err and "402" in err:
                error = (
                    "The sample image was not generated — image provider credits / in-flight limit "
                    "(HTTP 402). Wait for other requests to finish, or add credits, then retry."
                )
            else:
                error = "The sample image was not generated."
            shot_rows.append(
                {"key": shot.key, "title": shot.title, "url": url, "error": error}
            )
            # Progressive collage: re-send a growing board so the UI updates tile by tile.
            board = _advanced_sample_board_json(plan, shot_rows)
            yield {"type": "answer", "text": board, "replace": True}

        text = _advanced_sample_board_json(plan, shot_rows).strip()
        answer = GroundedAnswer(
            question="/advanced_garment_design",
            routes=["slash", "garment_design", plan.mode],
            answer=text,
            caveats=["Sample images are drawn from the product specification in this conversation."],
            synthetic_data=False,
        )
        yield {"type": "done", "answer": answer.model_dump()}
        return

    parts: list[str] = ["Garment sample from the product specification:\n"]
    yield from _deltas("answer", "".join(parts))
    for index, shot in enumerate(plan.shots, start=1):
        yield {"type": "status", "text": f"Drawing {shot.title} ({index}/{len(plan.shots)})…"}
        markdown, err = _generate_shot_markdown(platform, shot)
        if markdown:
            block = f"\n\n**{shot.title}**\n{markdown}"
        elif err and "402" in err:
            block = (
                f"\n\n**{shot.title}**\n"
                "The sample image was not generated — image provider credits / in-flight limit "
                "(HTTP 402). Wait for other requests to finish, or add credits, then retry."
            )
        else:
            block = f"\n\n**{shot.title}**\nThe sample image was not generated."
        parts.append(block)
        yield from _deltas("answer", block)

    text = "".join(parts).strip()
    answer = GroundedAnswer(
        question="/garment_design",
        routes=["slash", "garment_design", plan.mode],
        answer=text,
        caveats=["Sample images are drawn from the product specification in this conversation."],
        synthetic_data=False,
    )
    yield {"type": "done", "answer": answer.model_dump()}


def _generate_shot_markdown(platform: Platform, shot) -> tuple[str, str]:
    """Return (markdown, error). Retries on provider 402 (credits held by in-flight jobs)."""

    if platform.settings.answer_mode != "llm":
        return "", ""
    try:
        provider = build_image_provider(platform.settings)
    except ModelUnavailableError as exc:
        LOGGER.info("sample image unavailable error=%s", exc.message)
        return "", exc.message
    generate = getattr(provider, "generate_image", None)
    if not callable(generate):
        return "", "image provider has no generate_image"
    last_err = ""
    # Up to 5 tries: OpenRouter often frees in-flight credit a few seconds after prior tiles finish.
    for attempt in range(5):
        try:
            try:
                data, media = generate(
                    shot.prompt,
                    model=resolve_image_model(platform.settings),
                    timeout=_model_timeout(platform),
                    aspect_ratio=getattr(shot, "aspect_ratio", None),
                    seed=getattr(shot, "seed", None),
                )
            except TypeError:
                # Test doubles may not accept aspect_ratio/seed.
                data, media = generate(
                    shot.prompt,
                    model=resolve_image_model(platform.settings),
                    timeout=_model_timeout(platform),
                )
        except ModelUnavailableError as exc:
            last_err = exc.message
            LOGGER.info("sample image unavailable error=%s attempt=%s", exc.message, attempt + 1)
            if "402" in exc.message and attempt < 4:
                time.sleep(8.0 * (attempt + 1))
                continue
            return "", last_err
        if not data:
            return "", last_err or "empty image"
        return (
            "![%s](%s)" % (shot.title, _store_sample_image(platform.settings.generated, data, media)),
            "",
        )
    return "", last_err


def _sample_image_markdown(platform: Platform, specification: str) -> str:
    """Draw one studio image via OpenRouter only. Return markdown, or a short failure line."""

    if platform.settings.answer_mode != "llm" or not specification.strip():
        return ""
    from fashion_brain.agents import sample_image_prompt

    try:
        provider = build_image_provider(platform.settings)
    except ModelUnavailableError as exc:
        LOGGER.info("sample image unavailable error=%s", exc.message)
        return "\n\nThe sample image was not generated."
    generate = getattr(provider, "generate_image", None)
    if not callable(generate):
        return ""
    try:
        data, media = generate(
            sample_image_prompt(specification),
            model=resolve_image_model(platform.settings),
            timeout=_model_timeout(platform),
        )
    except ModelUnavailableError as exc:
        LOGGER.info("sample image unavailable error=%s", exc.message)
        return "\n\nThe sample image was not generated."
    if not data:
        return "\n\nThe sample image was not generated."
    return "\n\n![Sample design](" + _store_sample_image(platform.settings.generated, data, media) + ")"


def _sample_design_answer(platform: Platform, question: str, history: list | None) -> GroundedAnswer:
    from fashion_brain.agents import SampleDesignAgent, _specification_text

    specification = _specification_text(history, question)
    text = SampleDesignAgent().design(platform, question, history)
    text = (text + _sample_image_markdown(platform, specification)).strip()
    return GroundedAnswer(
        question=question,
        routes=["sample_design"],
        answer=text,
        caveats=["This sample design uses the product specification from this conversation."],
        synthetic_data=False,
    )


def _iter_sample_design(platform: Platform, question: str, history: list | None = None):
    from fashion_brain.agents import SampleDesignAgent, sample_design_messages, sample_design_sheet, _specification_text

    specification = _specification_text(history, question)
    sheet = sample_design_sheet(specification)
    yield {"type": "status", "text": "Preparing the sample from the product specification…"}
    yield {"type": "reasoning", "text": "Reading the product specification and preparing the sample design. "}
    provider = build_model_provider(platform.settings)
    parts: list[str] = []
    try:
        for event in _iter_model(
            provider,
            sample_design_messages(question, specification),
            timeout=_model_timeout(platform),
        ):
            if event["type"] == "answer":
                parts.append(event["text"])
                yield event
    except ModelUnavailableError:
        parts = []
    text = "".join(parts).strip()
    # Only emit a full answer blob when the model did not stream one (avoids duplicate sheets).
    if not text or re.search(r"can.?t physically|cannot physically|unable to manufacture|can.?t manufacture", text, re.IGNORECASE):
        text = sheet or SampleDesignAgent().design(platform, question, history)
        yield from _deltas("answer", text)
    image = yield from _run_with_progress(
        lambda: _sample_image_markdown(platform, specification),
        progress="Generating the sample image…",
    )
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
    if _VISUALIZE.search(question):
        transformed = _visualize_previous_table(history, question)
        if transformed:
            yield {"type": "status", "text": "Visualizing the previous results without a new catalog lookup…"}
            yield {"type": "reasoning", "text": "Using the data already shown in the previous reply. "}
            yield from _deltas("answer", transformed)
            answer = GroundedAnswer(
                question=question,
                routes=["direct"],
                answer=transformed,
                caveats=["Visualization is based only on the previous reply; no new data was retrieved."],
                synthetic_data=False,
            )
            yield {"type": "done", "answer": answer.model_dump()}
            return

    provider = build_model_provider(platform.settings)
    yield {"type": "status", "text": "Drafting your answer…"}
    yield {"type": "reasoning", "text": "Using the previous reply. " if normalize_history(history) else "Using the model's own knowledge. "}
    parts: list[str] = []
    try:
        system = DIRECT_SYSTEM
        if _VISUALIZE.search(question):
            wanted = _requested_chart_type(question)
            system = (
                VISUALIZE_SYSTEM
                + f" The user requested chart type '{wanted}'. "
                f'The ```chart JSON must use "type":"{wanted}".'
            )
        for event in _iter_model(
            provider,
            chat_messages(system, question, history),
            timeout=_model_timeout(platform),
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
    text = "".join(parts).strip()
    if not text:
        text = (
            "I couldn't create a chart because the previous reply doesn't contain a supported numeric table or chart."
            if _VISUALIZE.search(question)
            else "No answer was written."
        )
    answer = GroundedAnswer(
        question=question,
        routes=["direct"],
        answer=text,
        caveats=[DIRECT_CAVEAT],
        synthetic_data=False,
    )
    yield {"type": "done", "answer": answer.model_dump()}


def _visualize_previous_table(history: list | None, question: str = "") -> str:
    """Build a chart from the latest assistant reply without another model or KG call."""

    previous = next(
        (
            turn["content"]
            for turn in reversed(normalize_history(history))
            if turn["role"] == "assistant"
        ),
        "",
    )
    if not previous:
        return ""

    chart_match = re.search(r"```(?:chart|fbchart)\s*(\{.*?\})\s*```", previous, re.DOTALL | re.IGNORECASE)
    if chart_match:
        try:
            chart = json.loads(chart_match.group(1))
        except json.JSONDecodeError:
            chart = None
        if isinstance(chart, dict):
            labels = chart.get("labels")
            values = chart.get("values")
            if (
                isinstance(labels, list)
                and isinstance(values, list)
                and len(labels) >= 2
                and len(labels) == len(values)
                and all(isinstance(label, str) and label.strip() for label in labels)
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    for value in values
                )
            ):
                return _chart_answer(
                    labels,
                    values,
                    chart_type=_requested_chart_type(question),
                    title=str(chart.get("title") or "Previous results"),
                )

    lines = previous.splitlines()
    for index, line in enumerate(lines):
        if "|" not in line:
            continue
        headers = [cell.strip().strip("*`") for cell in line.strip().strip("|").split("|")]
        normalized = [re.sub(r"[^a-z0-9%]+", " ", cell.casefold()).strip() for cell in headers]
        label_index = next(
            (
                i
                for i, header in enumerate(normalized)
                if header in {
                    "category",
                    "categories",
                    "brand",
                    "brands",
                    "material",
                    "materials",
                    "product",
                    "products",
                    "name",
                    "item",
                }
                or header.endswith(" category")
                or header.startswith("category ")
            ),
            None,
        )
        if label_index is None:
            continue
        value_index = next(
            (
                i
                for i, header in enumerate(normalized)
                if i != label_index
                and any(
                    word in header
                    for word in (
                        "change",
                        "percent",
                        "pct",
                        "avg",
                        "average",
                        "share",
                        "count",
                        "products",
                        "events",
                        "price",
                    )
                )
            ),
            None,
        )
        if value_index is None:
            continue

        labels: list[str] = []
        values: list[float] = []
        has_percent = False
        for row in lines[index + 1 :]:
            if "|" not in row:
                if labels:
                    break
                continue
            cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
            if len(cells) <= max(label_index, value_index):
                if labels:
                    break
                continue
            if all(re.fullmatch(r":?-{2,}:?", cell.replace(" ", "")) for cell in cells if cell):
                continue
            label = cells[label_index].strip("*` ")
            raw_value = cells[value_index].replace(",", "").strip()
            match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s*(%)?", raw_value)
            if not label or not match:
                if labels:
                    break
                continue
            labels.append(label)
            values.append(float(match.group(1)))
            has_percent = has_percent or bool(match.group(2))
            if len(labels) >= 30:
                break

        if len(labels) < 2:
            continue
        dimension = headers[label_index].strip() or "Item"
        metric = headers[value_index].strip() or "Value"
        title = f"{metric}{' (%)' if has_percent and '%' not in metric else ''} by {dimension}"
        return _chart_answer(
            labels,
            values,
            chart_type=_requested_chart_type(question),
            title=title,
        )
    return ""


def _requested_chart_type(question: str) -> str:
    lowered = question.casefold()
    if re.search(r"\bpie[\s-]?charts?\b|\bas a pie\b|\bto pie\b|\bpie\b", lowered):
        return "pie"
    if re.search(r"\bline[\s-]?charts?\b|\bas a line\b|\bline\b", lowered):
        return "line"
    if re.search(r"\bbar[\s-]?charts?\b|\bas a bar\b|\bbar\b", lowered):
        return "bar"
    return "bar"


def _chart_answer(
    labels: list[str],
    values: list[int | float],
    *,
    chart_type: str,
    title: str,
) -> str:
    plot_labels = list(labels)
    plot_values: list[int | float] = list(values)
    if chart_type == "pie":
        metric_title = title.split(" by ", 1)[0]
        dimension = title.split(" by ", 1)[1] if " by " in title else "Item"
        magnitude = any(value < 0 for value in values)
        title = (
            f"Share of absolute {metric_title} by {dimension}"
            if magnitude
            else f"Share of {metric_title} by {dimension}"
        )
        # Pie slices need non-negative weights; keep signed originals in the labels.
        if magnitude:
            plot_labels = [
                f"{label} ({value:+.2f})" if isinstance(value, float) else f"{label} ({value:+d})"
                for label, value in zip(labels, values)
            ]
            plot_values = [abs(float(value)) for value in values]
    chart = {
        "type": chart_type,
        "title": title,
        "labels": plot_labels,
        "values": plot_values,
    }
    return "Visualization of the previous results:\n\n```chart\n" + json.dumps(
        chart, ensure_ascii=False
    ) + "\n```"


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


def _run_with_progress(operation, *, progress: str | list[str], interval: float = 2.0):
    """Run a blocking step while sending one honest progress update if it takes time."""

    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
    from contextvars import copy_context

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fashion-brain-planning")
    future = executor.submit(copy_context().run, operation)
    messages = [progress] if isinstance(progress, str) else list(progress)
    announced = False
    try:
        while True:
            try:
                return future.result(timeout=interval)
            except FutureTimeoutError:
                if messages and not announced:
                    yield {"type": "status", "text": messages[0]}
                    announced = True
    finally:
        executor.shutdown(wait=False)


def _model_timeout(platform, *, ceiling: float = 20.0) -> float:
    """Bound one model stage so a slow local or remote model cannot stall a chat turn."""

    configured = getattr(platform.settings, "request_timeout_seconds", None)
    if configured is None or configured <= 0:
        configured = getattr(platform.settings, "ollama_timeout_seconds", None)
    if configured is None or configured <= 0:
        return ceiling
    return min(float(configured), ceiling)


def _is_explicit_catalog_question(question: str, route: str) -> bool:
    if route in {"product_details", "specification"}:
        return True
    text = (question or "").casefold()
    asks_catalog_action = bool(
        re.search(
            r"\b(how many|count|number of|distribution|breakdown|analy[sz]e|compare|rank|"
            r"show|list|find|which|what)\b",
            text,
        )
    )
    names_catalog_data = bool(
        re.search(
            r"\b(products?|catalog|assortment|styles?|brands?|buyers?|categories?|"
            r"garments?|materials?|colors?|availability|in[\s-]?stock|prices?|specifications?)\b",
            text,
        )
    )
    if asks_catalog_action and names_catalog_data:
        return True
    if asks_catalog_action:
        from fashion_brain.ask import garment_type_from_question

        return bool(garment_type_from_question(question))
    return False


def _provisional_reasoning_plan(question: str, summary) -> str:
    """Short plan shown while retrieval runs — no request echo / checklist."""

    intent = str(getattr(summary, "actual_intent", "") or "").strip() or "catalog lookup"
    intent = intent[0].lower() + intent[1:] if intent else "catalog lookup"
    parts = [f"Looking up: {intent}."]
    filters = getattr(summary, "filters", None) or {}
    group_by = getattr(summary, "group_by", None)
    scope = [f"{key.replace('_', ' ')} = {value}" for key, value in list(filters.items())[:6] if value]
    if group_by:
        scope.append(f"grouped by {str(group_by).replace('_', ' ')}")
    if scope:
        parts.append("Using " + "; ".join(scope) + ".")
    matches = list(getattr(summary, "resolved_terms", None) or [])
    if matches:
        parts.append("Live match: " + "; ".join(matches[:4]) + ".")
    return " ".join(parts)


def _canonical_composition_compare_answer(findings: str) -> str:
    """Render source-generated brand tables and chart without LLM markdown rewriting."""

    lines = (findings or "").splitlines()
    title = next(
        (line.strip() for line in lines if line.startswith("Composition compare —")),
        "",
    )
    if not title:
        return ""
    parties = re.findall(r"([^(),]+?)\s+(\d+)(?=,|\))", title.partition("(")[2])
    if not parties:
        return ""

    product_type = re.search(r"similar (.+?) across", title)
    type_name = product_type.group(1).strip() if product_type else "products"
    count_text = "; ".join(
        f"{name.strip()} has {count} matching {type_name} products" for name, count in parties
    )
    blocks = [count_text + "."]
    section: list[str] = []
    in_chart = False

    def flush_section() -> None:
        if section:
            blocks.append("\n".join(section).strip())
            section.clear()

    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if line.startswith("**") and "matching" in line:
            flush_section()
            match = re.match(r"\*\*(.+?)\*\*\s+—\s+(.+)", line)
            if match:
                section.append(f"### {match.group(1).strip()} — {match.group(2).strip()}")
            index += 1
            continue
        if line.startswith("```") and line[3:].strip().casefold() in {"chart", "fbchart"}:
            flush_section()
            in_chart = True
            section.append("```chart")
            index += 1
            continue
        if in_chart:
            section.append(lines[index])
            if line == "```":
                in_chart = False
                flush_section()
            index += 1
            continue
        if line.startswith("Response must include"):
            index += 1
            continue
        if line.startswith("Compare takeaway:"):
            flush_section()
            blocks.append(line.removeprefix("Compare takeaway:").strip())
            index += 1
            continue
        if line.startswith("|"):
            section.append(line)
        index += 1
    flush_section()
    return "\n\n".join(block for block in blocks if block).strip()


def _catalog_reasoning_text(
    findings: str,
    evidence: list | None = None,
    caveats: list | None = None,
    *,
    question: str = "",
    intention=None,
) -> str:
    """User-facing rationale based on the request, retrieval result, and answer shape.

    This is a concise explanation of decisions and evidence, not private chain-of-thought.
    """

    text = (findings or "").strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    is_analysis = "product analysis for" in text.casefold() or "attribute-wise" in text.casefold()
    if is_analysis:
        return _analysis_reasoning_text(
            lines,
            evidence=evidence,
            caveats=caveats,
            question=question,
            intention=intention,
        )

    has_table = "|" in text and "---" in text
    has_chart = "```chart" in text or "```fbchart" in text
    has_trend_line = '"type": "line"' in text or '"type":"line"' in text
    has_drivers = "probable drivers" in text.casefold()
    composition_compare = "composition compare" in text.casefold() or (
        "material mix" in text.casefold() and "matching" in text.casefold()
    )
    shared_catalogs = bool(
        getattr(intention, "intent", None) == "shared_products"
        or "values present in every selected product catalog" in text.casefold()
    )
    headline = next(
        (
            ln.replace("**", "")
            for ln in lines
            if ln
            and not ln.startswith(("```", "|", "#", "-", "*"))
            and "example" not in ln.casefold()
        ),
        "",
    )
    table_titles = [
        ln.rstrip(":")
        for ln in lines
        if ln.endswith(":")
        and any(
            tok in ln.casefold()
            for tok in ("brand", "category", "by ", "breakdown", "example", "weekly", "period")
        )
    ][:4]
    data_bits: list[str] = []
    if headline:
        data_bits.append(headline)
    if has_trend_line:
        data_bits.append("The findings include a chart of recorded price changes over time.")
    if has_drivers:
        data_bits.append("The findings include indicators that may help explain the recorded changes.")
    if composition_compare:
        brand_heads = []
        for ln in lines:
            if "matching" not in ln.casefold() or "—" not in ln:
                continue
            head = ln.split("—", 1)[0].replace("*", "").strip()
            if head:
                brand_heads.append(head)
        brand_heads = brand_heads[:4]
        data_bits.append(
            "The findings contain separate material-mix comparisons"
            + (f" ({', '.join(brand_heads)})" if brand_heads else "")
            + "."
        )
    if table_titles:
        data_bits.append("The returned breakdowns cover: " + "; ".join(table_titles) + ".")
    elif has_table and not composition_compare:
        data_bits.append("The findings include a breakdown table.")
    for note in caveats or []:
        note = str(note or "").strip()
        if not note:
            continue
        low = note.casefold()
        if any(tok in low for tok in ("knowledge graph", "neo4j", "graph_id", "ecom_products", "cypher")):
            continue
        data_bits.append(note)
        break

    filters = getattr(intention, "filters", None) or {}
    group_by = getattr(intention, "group_by", None)
    scope = [] if shared_catalogs else [
        f"{key.replace('_', ' ')} = {value}" for key, value in filters.items() if value
    ]
    if group_by and not shared_catalogs:
        scope.append(f"grouped by {str(group_by).replace('_', ' ')}")
    result = " ".join(data_bits) if data_bits else "No grounded product findings were returned."
    if not text:
        result = "The retrieval returned no product findings."
    blocks = []
    if shared_catalogs:
        blocks.append("Compared shared values across the selected catalogs.")
    elif scope:
        blocks.append("Checked: " + "; ".join(scope) + ".")
    blocks.append(result)
    # Skip meta "answer approach" for plain single-count replies — the answer itself is enough.
    simple_count = (
        not shared_catalogs
        and not has_table
        and not has_chart
        and not scope
        and bool(re.search(r"\bthere are\b.+\bproducts?\b", result.casefold()))
    )
    if not simple_count:
        plan = (
            "List only values shared by every selected catalog, with a separate product count for each."
            if shared_catalogs
            else _response_plan_for_findings(
                question=question,
                findings=text,
                has_table=has_table,
                has_chart=has_chart,
                analysis=False,
            )
        )
        blocks.append(plan)
    return "\n\n".join(blocks)[:1600]


def _response_plan_for_findings(
    *,
    question: str,
    findings: str,
    has_table: bool,
    has_chart: bool,
    analysis: bool,
) -> str:
    """Decide prose / table / chart guidance for the final reply."""

    q = (question or "").casefold()
    f = (findings or "").casefold()
    if "no grounded" in f or "0 products" in f or "0 rows" in f:
        return "Nothing matched — suggest one broader question; skip a chart."
    if analysis:
        return "Summarize metrics, keep useful tables/charts, and close with the main takeaways."
    if "example products" in f or "showing up to" in f:
        return "Lead with the total, keep any brand/category mix compact, and show only a few examples if present."
    trend_ask = any(
        tok in q for tok in ("trend", "trends", "trending", "average price", "price change", "over time")
    ) or "price change trend" in f or "probable drivers" in f
    if trend_ask and (has_chart or "probable drivers" in f):
        return "Lead with the recorded trend, keep the chart and evidence-based drivers, and avoid invented causes."
    composition_compare = (
        "composition compare" in f
        or ("composition" in q and any(tok in q for tok in ("compare", "across", "vs", "versus", "similar")))
        or ("material mix" in f and "matching" in f)
    )
    if composition_compare:
        return "Compare brands side by side, keep each material mix distinct, and close with a data-backed takeaway."
    if has_chart:
        return "Keep the chart from the findings and add a brief explanation without duplicating it as a table."
    if has_table and any(tok in q for tok in ("most", "rank", "top", "compare", "vs", "versus", "by ")):
        return "Introduce the result briefly and keep the ranking/breakdown table."
    if has_table:
        return "Explain briefly and keep the breakdown table."
    if any(tok in q for tok in ("how many", "count", "total")):
        return "Answer with the count directly."
    return "Answer directly; use a small table only when several values need comparison."


def _analysis_reasoning_text(
    lines: list[str],
    *,
    evidence: list | None = None,
    caveats: list | None = None,
    question: str = "",
    intention=None,
) -> str:
    """Explain the analysis scope, leading findings, and answer presentation."""

    ask = " ".join(str(question or "").split())
    if len(ask) > 140:
        ask = ask[:137] + "…"
    blocks: list[str] = []
    filters = getattr(intention, "filters", None) or {}
    scope = [f"{key.replace('_', ' ')} = {value}" for key, value in filters.items() if value]
    if scope:
        blocks.append("I'm analyzing this scope: " + "; ".join(scope) + ".")
    elif ask:
        blocks.append(f"I'm treating this as an analysis of: {ask}.")
    overview = next((ln for ln in lines if ln.lower().startswith("product analysis for")), "")
    if overview:
        blocks.append("The retrieved overview says: " + overview.replace("**", ""))

    metrics: list[str] = []
    in_metrics = False
    for ln in lines:
        if ln.startswith("|") and "metric" in ln.casefold() and "value" in ln.casefold():
            in_metrics = True
            continue
        if in_metrics:
            if not ln.startswith("|"):
                break
            if "---" in ln:
                continue
            cells = [c.strip() for c in ln.strip("|").split("|")]
            if len(cells) >= 2 and cells[0].casefold() not in {"metric", "name"}:
                metrics.append(f"{cells[0]}={cells[1]}")
    if metrics:
        blocks.append("Key metrics: " + "; ".join(metrics[:8]) + ".")

    section: str | None = None
    field_notes: list[str] = []
    in_chart = False
    for ln in lines:
        if ln.startswith("```"):
            in_chart = not in_chart
            continue
        if in_chart:
            continue
        if ln.startswith("<!--fb:viz:"):
            section = None
            continue
        if ln.startswith("### "):
            section = ln[4:].strip()
            continue
        if not section:
            continue
        if ln.lower().startswith("leading value:"):
            lead = ln.split(":", 1)[1].strip().rstrip(".")
            field_notes.append(f"{section}→{lead}")
            section = None
            continue
        if ln.startswith("|") and "---" not in ln:
            cells = [c.strip() for c in ln.strip("|").split("|")]
            if len(cells) >= 2 and cells[0] and cells[0].casefold() not in {"metric", "name"}:
                field_notes.append(f"{section}→{cells[0]} ({cells[1]})")
                section = None
    if field_notes:
        blocks.append("Leading values in the returned breakdowns: " + "; ".join(field_notes[:10]) + ".")
    for note in caveats or []:
        note = str(note or "").strip()
        if not note:
            continue
        low = note.casefold()
        if any(tok in low for tok in ("knowledge graph", "neo4j", "graph_id", "ecom_products", "cypher")):
            continue
        blocks.append("Data note: " + note)
        break
    plan = _response_plan_for_findings(
        question=question,
        findings="\n".join(lines),
        has_table=any(ln.startswith("|") for ln in lines),
        has_chart=any("```chart" in ln or "```fbchart" in ln for ln in lines),
        analysis=True,
    )
    blocks.append(plan)
    return "\n\n".join(blocks)[:2200] if blocks else "No grounded product findings were returned."


_FB_VIZ_MARK = re.compile(r"<!--\s*fb:viz:[^>]*-->", re.IGNORECASE)


def _strip_fb_viz_markers(text: str) -> str:
    """Remove internal <!--fb:viz:…--> markers that must never reach the chat UI."""

    cleaned = _FB_VIZ_MARK.sub("", text or "")
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _finalize_analysis_answer(answer: str, findings: str) -> str:
    """Strip internal viz markers and restore missing ```chart blocks from findings."""

    return _append_missing_analysis_viz(_strip_fb_viz_markers(answer), findings)


def _append_missing_analysis_viz(answer: str, findings: str) -> str:
    """Keep LLM prose; append only missing field charts — never re-dump findings."""

    prose = _strip_fb_viz_markers(answer)
    raw = (findings or "").strip()
    if not raw:
        return prose
    # Model already embedded visualizations — do not append a second copy.
    if "```chart" in prose or "```fbchart" in prose:
        return prose
    if _findings_have_chart(raw):
        viz = _extract_analysis_viz_blocks(raw)
        if viz:
            return (prose + "\n\n" + viz).strip() if prose else viz
    # Table-only field viz, and the answer has no markdown table yet.
    if "<!--fb:viz:" in raw and "| ---" in raw and "|" not in prose:
        viz = _extract_analysis_viz_blocks(raw)
        if viz:
            return (prose + "\n\n" + viz).strip() if prose else viz
    return prose


def _extract_analysis_viz_blocks(findings: str) -> str:
    """Pull only marked field visualizations (<!--fb:viz:…-->) for merge — charts/tables."""

    lines = (findings or "").splitlines()
    out: list[str] = []
    capturing = False
    buf: list[str] = []
    in_fence = False
    for ln in lines:
        if ln.startswith("<!--fb:viz:"):
            if buf:
                out.extend(buf)
                out.append("")
            buf = []
            capturing = True
            in_fence = False
            continue
        if not capturing:
            continue
        if ln.startswith("### "):
            # Keep a light heading so the chart has context in the answer.
            buf.append(ln)
            continue
        if ln.casefold().startswith("leading value:"):
            continue
        if ln.startswith("```"):
            in_fence = not in_fence
            buf.append(ln)
            continue
        buf.append(ln)
        if not in_fence and ln.startswith("|") and "---" not in ln:
            continue
    if buf:
        out.extend(buf)
    return _strip_fb_viz_markers("\n".join(out))


def _findings_have_chart(text: str) -> bool:
    return "```chart" in (text or "") or "```fbchart" in (text or "")


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
