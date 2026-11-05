"""LangGraph agent that routes a question to a model, web search, or an external API."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Literal

import httpx
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from typing_extensions import TypedDict

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

Route = Literal["direct", "web_search", "external"]

ROUTER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "answer_directly",
            "description": "Answer from the model when the question does not need fresh web facts or another system.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the public web for current events, recent facts, or anything the model should look up.",
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
            "name": "external_resource",
            "description": "Ask the Fashion Brain catalog API, or another configured HTTP resource, about products and materials.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Question to send to the external resource."}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]

TOOL_ROUTES = {
    "answer_directly": "direct",
    "web_search": "web_search",
    "external_resource": "external",
}


class AgentState(TypedDict, total=False):
    question: str
    route: Route
    tool_query: str
    notes: str
    response: str


def _model_name() -> str:
    raw = os.getenv("FASHIONGPT_MODEL_NAME", "openai/gpt-4o-mini").strip()
    return raw[1:] if raw.startswith("~") else raw


def _api_key() -> str:
    return os.getenv("OPENROUTER_API_KEY") or os.getenv("FASHIONGPT_MODEL_API_KEY") or ""


def build_client() -> OpenAI:
    key = _api_key()
    if not key:
        raise SystemExit("Set OPENROUTER_API_KEY or FASHIONGPT_MODEL_API_KEY in .env before running this agent.")
    return OpenAI(
        base_url=os.getenv("FASHIONGPT_MODEL_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=key,
        default_headers={
            "HTTP-Referer": os.getenv("FASHIONGPT_MODEL_APP_URL", "http://127.0.0.1:8000"),
            "X-OpenRouter-Title": os.getenv("FASHIONGPT_MODEL_APP_TITLE", "Fashion Brain"),
        },
    )


def _complete(client: OpenAI, messages: list[dict], **extra) -> str:
    completion = client.chat.completions.create(
        model=_model_name(),
        messages=messages,
        temperature=0,
        **extra,
    )
    content = completion.choices[0].message.content
    return (content or "").strip()


def _fallback_route(question: str) -> tuple[Route, str]:
    text = question.lower()
    if any(word in text for word in ("latest", "today", "news", "current", "web search", "look up")):
        return "web_search", question
    if any(word in text for word in ("catalog", "product", "material", "buyer", "garment", "fashion brain")):
        return "external", question
    return "direct", question


def route_question(state: AgentState) -> dict:
    """Pick one tool: a direct model answer, web search, or an external resource."""

    client = build_client()
    completion = client.chat.completions.create(
        model=_model_name(),
        temperature=0,
        tool_choice="required",
        tools=ROUTER_TOOLS,
        messages=[
            {
                "role": "system",
                "content": (
                    "Choose exactly one tool. Use web_search for current or public facts. "
                    "Use external_resource for this product catalog. Use answer_directly otherwise."
                ),
            },
            {"role": "user", "content": state["question"]},
        ],
    )
    tool_calls = completion.choices[0].message.tool_calls or []
    if not tool_calls:
        route, query = _fallback_route(state["question"])
        return {"route": route, "tool_query": query}

    call = tool_calls[0]
    route = TOOL_ROUTES.get(call.function.name, "direct")
    try:
        arguments = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError:
        arguments = {}
    query = str(arguments.get("query") or state["question"])
    return {"route": route, "tool_query": query}


def answer_directly(state: AgentState) -> dict:
    client = build_client()
    notes = _complete(
        client,
        [
            {"role": "system", "content": "Answer the user directly. Say when you are unsure."},
            {"role": "user", "content": state["question"]},
        ],
    )
    return {"notes": notes}


def search_web(state: AgentState) -> dict:
    client = build_client()
    notes = _complete(
        client,
        [
            {
                "role": "system",
                "content": "Use web search results. Summarize the answer and name the sources you used.",
            },
            {"role": "user", "content": state.get("tool_query") or state["question"]},
        ],
        extra_body={"plugins": [{"id": "web", "max_results": 5}]},
    )
    return {"notes": notes}


def call_external_resource(state: AgentState) -> dict:
    url = os.getenv("LLM_TEST_EXTERNAL_URL", "http://127.0.0.1:8000/ask")
    timeout = float(os.getenv("FASHIONGPT_REQUEST_TIMEOUT_SECONDS", "30"))
    headers = {
        "Content-Type": "application/json",
        "X-Clearance": os.getenv("LLM_TEST_CLEARANCE", "internal"),
    }
    token = os.getenv("FASHIONGPT_API_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    question = state.get("tool_query") or state["question"]
    try:
        response = httpx.post(url, json={"question": question}, headers=headers, timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return {"notes": f"External resource at {url} failed: {exc.__class__.__name__}: {exc}"}
    try:
        payload = response.json()
    except json.JSONDecodeError:
        return {"notes": response.text}
    if isinstance(payload, dict) and payload.get("answer"):
        return {"notes": str(payload["answer"])}
    return {"notes": json.dumps(payload, ensure_ascii=False)}


def generate_response(state: AgentState) -> dict:
    notes = state.get("notes", "").strip()
    if state.get("route") != "external":
        return {"response": notes}
    client = build_client()
    response = _complete(
        client,
        [
            {
                "role": "system",
                "content": (
                    "Write the user-facing answer from the external resource notes. "
                    "Do not add facts that are not in the notes. If the notes describe a failure, say that plainly."
                ),
            },
            {"role": "user", "content": f"Question:\n{state['question']}\n\nNotes:\n{notes}"},
        ],
    )
    return {"response": response or notes}


def _select_route(state: AgentState) -> Route:
    route = state.get("route", "direct")
    if route not in {"direct", "web_search", "external"}:
        return "direct"
    return route


def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("router", route_question)
    graph.add_node("direct", answer_directly)
    graph.add_node("web_search", search_web)
    graph.add_node("external", call_external_resource)
    graph.add_node("respond", generate_response)
    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        _select_route,
        {"direct": "direct", "web_search": "web_search", "external": "external"},
    )
    graph.add_edge("direct", "respond")
    graph.add_edge("web_search", "respond")
    graph.add_edge("external", "respond")
    graph.add_edge("respond", END)
    return graph.compile()


def ask(question: str) -> AgentState:
    agent = build_agent()
    return agent.invoke({"question": question})


def main() -> None:
    if len(sys.argv) > 1:
        result = ask(" ".join(sys.argv[1:]))
        print(f"route: {result.get('route')}")
        print(result.get("response", ""))
        return

    print("Fashion Brain test agent. Empty line exits.")
    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question or question.lower() in {"quit", "exit"}:
            return
        result = ask(question)
        print(f"route: {result.get('route')}")
        print(result.get("response", ""))
        print()


if __name__ == "__main__":
    main()
