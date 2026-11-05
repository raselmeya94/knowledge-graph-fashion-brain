"""Fashion Brain FastAPI: user chat BFF + admin KG-making proxies."""

from __future__ import annotations

import json
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

import anyio
from fastapi import Depends, FastAPI, File, Form, Header, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError

from fashion_brain.agent import answer_question
from fashion_brain.auth import AuthService, require_admin
from fashion_brain.flow import iter_ask_events
from fashion_brain.kg_client import kg_request, list_graphs, ping_kg, reload_kg_graphs, uses_kg_api
from fashion_brain.llm import resolve_model_api_key
from fashion_brain.store import FashionBrainStore
from knowledge_graph.access import CLEARANCE_RANK, Principal
from knowledge_graph.config import load_settings
from knowledge_graph.errors import FashionGPTError, NotFoundError, PermissionDenied, RequestTimeout, Unauthorized, ValidationFailed
from knowledge_graph.logging_config import configure_logging, get_logger
from knowledge_graph.models import GroundedAnswer
from knowledge_graph.pipeline.service import Platform

LOGGER = get_logger("fashion_brain.api")
CHAT_PAGE = Path(__file__).with_name("chat.html")
_SAMPLE_FILE = re.compile(r"^[a-f0-9]{32}\.(png|jpe?g|webp)$", re.IGNORECASE)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = ""


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list)
    conversation_id: str | None = None
    graph_id: str | None = None


class RegisterRequest(BaseModel):
    email: str
    password: str = Field(min_length=8, max_length=200)
    role: Literal["user", "admin"] = "user"
    clearance: Literal["public", "internal", "confidential"] = "internal"
    bootstrap_secret: str | None = None


class LoginRequest(BaseModel):
    email: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str | None = None


class MeUpdateRequest(BaseModel):
    clearance: Literal["public", "internal", "confidential"] | None = None


class ConversationCreate(BaseModel):
    title: str | None = None
    graph_id: str | None = None


class ConversationUpdate(BaseModel):
    title: str | None = None
    archived: bool | None = None
    graph_id: str | None = None


class ConversationMessageRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    graph_id: str | None = None


class ChatApiRequest(BaseModel):
    """Plan-shaped chat body: POST /api/v1/chat."""

    message: str = Field(min_length=3, max_length=2000)
    session_id: str | None = None
    graph_id: str | None = None


class MemoryCreate(BaseModel):
    content: str = Field(min_length=1, max_length=4000)


class MemoryUpdate(BaseModel):
    content: str = Field(min_length=1, max_length=4000)


class GraphQueryBody(BaseModel):
    template: str
    params: dict[str, str] = Field(default_factory=dict)
    limit: int = 20
    offset: int = 0


class PipelineBody(BaseModel):
    kind: str | None = None
    text: str | None = None
    html: str | None = None
    filename: str | None = None
    media_type: str | None = None
    use_model: bool = True
    clean: bool = False
    auto_commit: bool = False
    data_b64: str | None = None
    graph_id: str | None = None
    ontology_id: str | None = None


class ExtractBody(BaseModel):
    kind: str | None = None
    text: str | None = None
    filename: str | None = None
    media_type: str | None = None
    json_data: dict | list | None = None
    data_base64: str | None = None
    clean: bool = False


class ProposeBody(BaseModel):
    text: str = ""
    title: str | None = None
    kind: str | None = None
    graph_id: str | None = None
    use_model: bool = True
    structured: dict | list | None = None
    filename: str | None = None


class StructureBody(BaseModel):
    source_kind: str | None = None
    filename: str | None = None
    title: str | None = None
    text: str | None = None
    graph_id: str | None = None
    ontology_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    relationships: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class QueueReviewBody(BaseModel):
    source_kind: str | None = None
    filename: str | None = None
    title: str | None = None
    text: str | None = None
    graph_id: str | None = None
    ontology_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    relationships: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ScrapingChunkBody(BaseModel):
    file_id: str | None = None
    text: str | None = None
    filename: str | None = None
    kind: str | None = None


class ScrapingProposeBody(BaseModel):
    file_id: str
    graph_id: str | None = None
    ontology_id: str | None = None
    use_model: bool = True


class ScrapingQueueBody(BaseModel):
    file_id: str
    graph_id: str | None = None
    ontology_id: str | None = None
    relationships: list[dict[str, Any]] | None = None
    warnings: list[str] = Field(default_factory=list)


class ScrapingCommitBody(BaseModel):
    review_id: str
    ontology: dict[str, Any] = Field(default_factory=dict)
    graph: dict[str, Any] = Field(default_factory=dict)
    graph_id: str | None = None


class ReviewUpdateBody(BaseModel):
    decision: Literal["approved", "rejected", "pending"] | None = None
    edits: list[dict[str, Any]] = Field(default_factory=list)
    relationship_ids: list[str] = Field(default_factory=list)


class IngestRunBody(BaseModel):
    sources: list[str] | None = None


class ToolBody(BaseModel):
    payload: dict[str, Any] = Field(default_factory=dict)


def create_app(platform: Platform | None = None, *, ingest_on_startup: bool | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.platform is None:
            settings = load_settings()
            configure_logging(settings.log_level, settings.log_json)
            app.state.platform = Platform(settings)
        settings = app.state.platform.settings
        if getattr(app.state, "store", None) is None:
            app.state.store = FashionBrainStore(settings.fashion_brain_db_file)
        if getattr(app.state, "auth", None) is None:
            app.state.auth = AuthService(app.state.store, settings)
        if uses_kg_api(settings):
            status = ping_kg(settings)
            LOGGER.info("knowledge graph api status=%s", status)
            app.state.kg_status = status
        else:
            app.state.kg_status = "disabled"
            if app.state.ingest_on_startup and settings.env == "development":
                app.state.platform.load_catalog()
        yield

    app = FastAPI(
        title="Fashion Brain",
        version="0.2.0",
        description="Backend for the Fashion Brain frontend: user chat + admin KG-making proxies.",
        lifespan=lifespan,
    )
    app.state.platform = platform
    app.state.ingest_on_startup = (platform is None) if ingest_on_startup is None else ingest_on_startup
    app.state.kg_status = "unknown"
    if platform is not None:
        app.state.store = FashionBrainStore(platform.settings.fashion_brain_db_file)
        app.state.auth = AuthService(app.state.store, platform.settings)
    else:
        app.state.store = None
        app.state.auth = None

    # Frontend is a separate static project; allow browser calls from local UI servers.
    # Opening the UI via LAN IP (e.g. http://192.168.x.x:5173) is cross-origin vs :8000.
    settings_for_cors = platform.settings if platform is not None else load_settings()
    cors_origins = [origin.strip() for origin in str(settings_for_cors.cors_origins or "").split(",") if origin.strip()]
    if not cors_origins:
        cors_origins = [
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "http://[::1]:5173",
            "http://127.0.0.1:5500",
            "http://localhost:5500",
            "http://127.0.0.1:3000",
            "http://localhost:3000",
        ]
    is_dev = str(getattr(settings_for_cors, "env", "development") or "").lower() in {
        "development",
        "dev",
        "local",
    }
    # Dev: also allow private-LAN hosts so phone/other-PC access to npm serve works.
    cors_origin_regex = None
    if is_dev:
        cors_origin_regex = (
            r"https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?"
            r"|https?://("
            r"192\.168(?:\.\d{1,3}){2}"
            r"|10(?:\.\d{1,3}){3}"
            r"|172\.(1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2}"
            r")(:\d+)?"
        )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_origin_regex=cors_origin_regex,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_request(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        LOGGER.info(
            "http %s %s %s %.3fs",
            request.method,
            request.url.path,
            response.status_code,
            time.perf_counter() - started,
        )
        return response

    @app.exception_handler(FashionGPTError)
    async def handle_error(_request: Request, exc: FashionGPTError):
        return JSONResponse(status_code=exc.status_code, content={"error": exc.code, "message": exc.message})

    def current_platform(request: Request) -> Platform:
        current = request.app.state.platform
        if current is None:
            raise FashionGPTError("Platform is not initialized.")
        return current

    def current_store(request: Request) -> FashionBrainStore:
        store = request.app.state.store
        if store is None:
            raise FashionGPTError("Fashion Brain store is not initialized.")
        return store

    def current_auth(request: Request) -> AuthService:
        auth = request.app.state.auth
        if auth is None:
            raise FashionGPTError("Fashion Brain auth is not initialized.")
        return auth

    def current_principal(
        request: Request,
        authorization: Annotated[str | None, Header()] = None,
        x_clearance: Annotated[str | None, Header()] = None,
    ) -> Principal:
        return principal_for(current_platform(request), current_auth(request), authorization, x_clearance)

    def current_admin(principal: Principal = Depends(current_principal)) -> Principal:
        require_admin(principal)
        return principal

    # --- public / health ---
    # UI lives in the separate `frontend/` project. This process is API-only (+ optional legacy chat).

    @app.get("/")
    def root():
        return {
            "service": "fashion_brain",
            "role": "api",
            "docs": "/docs",
            "health": "/health",
            "frontend": "Run the separate frontend project (see frontend/README.md).",
            "legacy_chat": "/legacy/chat",
        }

    @app.get("/legacy/chat")
    def legacy_chat_page():
        return FileResponse(CHAT_PAGE)

    @app.get("/samples/{name}")
    def sample_image(name: str, platform: Platform = Depends(current_platform)):
        if not _SAMPLE_FILE.fullmatch(name):
            raise NotFoundError("Sample image was not found.")
        folder = (platform.settings.generated / "samples").resolve()
        path = (folder / name).resolve()
        if path.parent != folder or not path.is_file():
            raise NotFoundError("Sample image was not found.")
        return FileResponse(path)

    @app.get("/health")
    def health(request: Request, platform: Platform = Depends(current_platform)):
        if uses_kg_api(platform.settings):
            kg_status = ping_kg(platform.settings)
            graph_status = "remote"
        else:
            kg_status = getattr(request.app.state, "kg_status", "disabled")
            graph_status = "ok"
            try:
                platform.graph.ping()
            except FashionGPTError as exc:
                graph_status = exc.message
        if platform.settings.model_provider in {"", "none", "null"}:
            model_status = "disabled"
        elif not resolve_model_api_key(platform.settings):
            model_status = "missing_api_key"
        else:
            model_status = "configured"
        store = getattr(request.app.state, "store", None)
        chat_db = "ok" if store and store.path.exists() else "missing"
        degraded = kg_status == "unreachable" or (graph_status not in {"ok", "remote"})
        return {
            "status": "degraded" if degraded else "ok",
            "service": "fashion_brain",
            "kg": kg_status,
            "graph_backend": platform.settings.graph_backend,
            "graph": graph_status,
            "model_provider": platform.settings.model_provider,
            "model_name": platform.settings.model_name,
            "model": model_status,
            "extraction_mode": platform.settings.extraction_mode,
            "sql_enabled": platform.settings.sql_enabled,
            "chat_db": chat_db,
        }

    # --- Wave 1: Auth ---

    @app.post("/v1/auth/register")
    def register(body: RegisterRequest, auth: AuthService = Depends(current_auth)):
        user = auth.register(
            body.email,
            body.password,
            role=body.role,
            clearance=body.clearance,
            bootstrap_secret=body.bootstrap_secret,
        )
        return {"user": user}

    @app.post("/v1/auth/login")
    def login(body: LoginRequest, auth: AuthService = Depends(current_auth)):
        return auth.login(body.email, body.password)

    @app.post("/v1/auth/refresh")
    def refresh(body: RefreshRequest, auth: AuthService = Depends(current_auth)):
        return auth.refresh(body.refresh_token)

    @app.post("/v1/auth/logout")
    def logout(
        body: LogoutRequest,
        auth: AuthService = Depends(current_auth),
        principal: Principal = Depends(current_principal),
    ):
        return auth.logout(body.refresh_token, user_id=principal.subject if principal.subject not in {"api", "anonymous"} else None)

    @app.get("/v1/me")
    def me(principal: Principal = Depends(current_principal), store: FashionBrainStore = Depends(current_store)):
        if principal.subject in {"api", "anonymous"}:
            return {
                "id": principal.subject,
                "email": None,
                "role": "admin" if "admin" in principal.roles else "user",
                "clearance": principal.clearance,
                "roles": principal.roles,
            }
        user = store.get_user(principal.subject)
        if not user:
            raise Unauthorized("User was not found.")
        payload = store.public_user(user)
        payload["roles"] = principal.roles
        return payload

    @app.patch("/v1/me")
    def update_me(
        body: MeUpdateRequest,
        principal: Principal = Depends(current_principal),
        store: FashionBrainStore = Depends(current_store),
    ):
        if principal.subject in {"api", "anonymous"}:
            raise ValidationFailed("Profile updates require a user account.")
        if body.clearance is None:
            user = store.get_user(principal.subject)
            if not user:
                raise NotFoundError("User was not found.")
            return store.public_user(user)
        user = store.update_user(principal.subject, clearance=body.clearance)
        if not user:
            raise NotFoundError("User was not found.")
        return store.public_user(user)

    # --- Wave 2: conversations + memory ---

    @app.get("/v1/settings")
    def settings_view(platform: Platform = Depends(current_platform), principal: Principal = Depends(current_principal)):
        return {
            "clearance_default": principal.clearance,
            "model_provider": platform.settings.model_provider,
            "model_name": platform.settings.model_name,
            "kg_url": platform.settings.kg_url,
        }

    @app.get("/v1/graphs")
    def list_user_graphs(platform: Platform = Depends(current_platform), principal: Principal = Depends(current_principal)):
        """Active individual KGs for chat scope (shared/master listed as interconnect)."""

        if uses_kg_api(platform.settings):
            return kg_request(platform.settings, "GET", "/graphs", clearance=principal.clearance)
        return list_graphs(platform.settings, clearance=principal.clearance)

    @app.get("/v1/conversations")
    def list_conversations(
        archived: bool = False,
        principal: Principal = Depends(current_principal),
        store: FashionBrainStore = Depends(current_store),
    ):
        user_id = _user_id(principal)
        items = store.list_conversations(user_id, archived=archived)
        return {"conversations": [_conversation_public(item) for item in items]}

    @app.post("/v1/conversations")
    def create_conversation(
        body: ConversationCreate,
        principal: Principal = Depends(current_principal),
        store: FashionBrainStore = Depends(current_store),
    ):
        user_id = _user_id(principal)
        item = store.create_conversation(
            user_id,
            title=body.title or "New chat",
            clearance=principal.clearance,
            graph_id=body.graph_id,
        )
        return _conversation_public(item)

    @app.get("/v1/conversations/{conversation_id}")
    def get_conversation(
        conversation_id: str,
        principal: Principal = Depends(current_principal),
        store: FashionBrainStore = Depends(current_store),
    ):
        user_id = _user_id(principal)
        item = store.get_conversation(conversation_id, user_id)
        if not item:
            raise NotFoundError("Conversation was not found.")
        messages = store.list_messages(conversation_id)
        return {**_conversation_public(item), "messages": messages}

    @app.patch("/v1/conversations/{conversation_id}")
    def patch_conversation(
        conversation_id: str,
        body: ConversationUpdate,
        principal: Principal = Depends(current_principal),
        store: FashionBrainStore = Depends(current_store),
    ):
        user_id = _user_id(principal)
        item = store.update_conversation(
            conversation_id,
            user_id,
            title=body.title,
            archived=body.archived,
            graph_id=body.graph_id,
            clear_graph_id=body.graph_id == "",
        )
        if not item:
            raise NotFoundError("Conversation was not found.")
        return _conversation_public(item)

    @app.delete("/v1/conversations/{conversation_id}")
    def delete_conversation(
        conversation_id: str,
        principal: Principal = Depends(current_principal),
        store: FashionBrainStore = Depends(current_store),
    ):
        user_id = _user_id(principal)
        if not store.delete_conversation(conversation_id, user_id):
            raise NotFoundError("Conversation was not found.")
        return {"status": "ok"}

    @app.post("/v1/conversations/{conversation_id}/messages")
    async def conversation_message(
        conversation_id: str,
        body: ConversationMessageRequest,
        platform: Platform = Depends(current_platform),
        principal: Principal = Depends(current_principal),
        store: FashionBrainStore = Depends(current_store),
    ):
        user_id = _user_id(principal)
        conversation = store.get_conversation(conversation_id, user_id)
        if not conversation:
            raise NotFoundError("Conversation was not found.")
        history = store.history_for_ask(conversation_id)
        history = _with_memory(history, store, user_id)
        if conversation["title"] in {"", "New chat"}:
            store.touch_conversation(conversation_id, title=_title_from(body.question))
        store.add_message(conversation_id, "user", body.question)
        graph_id = body.graph_id or conversation.get("graph_id") or None
        if body.graph_id is not None and body.graph_id != conversation.get("graph_id"):
            store.update_conversation(
                conversation_id,
                user_id,
                graph_id=body.graph_id or None,
                clear_graph_id=body.graph_id == "",
            )
        answer = await _run(platform, answer_question, platform, body.question, principal, history, graph_id)
        evidence = [item.model_dump() for item in (answer.evidence or [])]
        assistant = store.add_message(
            conversation_id,
            "assistant",
            answer.answer,
            meta={
                "routes": list(answer.routes or []),
                "evidence": evidence,
                "caveats": list(answer.caveats or []),
                "url_citations": [
                    {
                        "url": item.get("url") or item.get("record_id") or "",
                        "title": item.get("title") or item.get("url") or "Source",
                        "excerpt": item.get("excerpt") or "",
                    }
                    for item in evidence
                    if (item.get("source_id") == "web" or item.get("url"))
                ],
            },
        )
        store.touch_conversation(conversation_id)
        return {
            "conversation_id": conversation_id,
            "message": assistant,
            "answer": answer.model_dump() if hasattr(answer, "model_dump") else answer,
        }

    @app.post("/v1/conversations/{conversation_id}/stop")
    def stop_conversation(conversation_id: str, principal: Principal = Depends(current_principal), store: FashionBrainStore = Depends(current_store)):
        user_id = _user_id(principal)
        if not store.get_conversation(conversation_id, user_id):
            raise NotFoundError("Conversation was not found.")
        return {"status": "accepted", "message": "Stop is cooperative; close the websocket to cancel streaming."}

    @app.post("/api/v1/chat")
    async def api_v1_chat(
        body: ChatApiRequest,
        platform: Platform = Depends(current_platform),
        principal: Principal = Depends(current_principal),
        store: FashionBrainStore = Depends(current_store),
    ):
        """Plan alias: authenticate via token, then chat inside a session/conversation."""

        user_id = _user_id(principal)
        conversation_id = body.session_id
        if conversation_id:
            conversation = store.get_conversation(conversation_id, user_id)
            if not conversation:
                raise NotFoundError("Conversation was not found.")
        else:
            conversation = store.create_conversation(
                user_id,
                title=_title_from(body.message),
                clearance=principal.clearance,
                graph_id=body.graph_id,
            )
            conversation_id = conversation["id"]
        message_body = ConversationMessageRequest(question=body.message, graph_id=body.graph_id)
        result = await conversation_message(conversation_id, message_body, platform, principal, store)
        answer = result.get("answer") or {}
        assistant = result.get("message") or {}
        return {
            "session_id": conversation_id,
            "message_id": assistant.get("id"),
            "intent": (answer.get("routes") or [None])[0],
            "selected_graphs": [route for route in (answer.get("routes") or []) if route and route != (answer.get("routes") or [None])[0]],
            "status": "completed",
            "response": answer.get("answer") or assistant.get("content") or "",
            "result_id": (assistant.get("meta") or {}).get("result_id"),
            "answer": answer,
            "message": assistant,
        }

    @app.get("/v1/memory")
    def list_memory(principal: Principal = Depends(current_principal), store: FashionBrainStore = Depends(current_store)):
        return {"memories": store.list_memories(_user_id(principal))}

    @app.post("/v1/memory")
    def create_memory(
        body: MemoryCreate,
        principal: Principal = Depends(current_principal),
        store: FashionBrainStore = Depends(current_store),
    ):
        return store.add_memory(_user_id(principal), body.content)

    @app.patch("/v1/memory/{memory_id}")
    def patch_memory(
        memory_id: str,
        body: MemoryUpdate,
        principal: Principal = Depends(current_principal),
        store: FashionBrainStore = Depends(current_store),
    ):
        item = store.update_memory(memory_id, _user_id(principal), body.content)
        if not item:
            raise NotFoundError("Memory was not found.")
        return item

    @app.delete("/v1/memory/{memory_id}")
    def delete_memory(
        memory_id: str,
        principal: Principal = Depends(current_principal),
        store: FashionBrainStore = Depends(current_store),
    ):
        if not store.delete_memory(memory_id, _user_id(principal)):
            raise NotFoundError("Memory was not found.")
        return {"status": "ok"}

    @app.post("/ask", response_model=GroundedAnswer)
    async def ask(
        body: AskRequest,
        platform: Platform = Depends(current_platform),
        principal: Principal = Depends(current_principal),
        store: FashionBrainStore = Depends(current_store),
    ):
        history, conversation_id = _resolve_history(body, principal, store)
        graph_id = body.graph_id
        if graph_id is None and conversation_id:
            conv = store.get_conversation(conversation_id, _user_id(principal))
            graph_id = (conv or {}).get("graph_id")
        answer = await _run(platform, answer_question, platform, body.question, principal, history, graph_id)
        if conversation_id:
            store.add_message(conversation_id, "user", body.question)
            store.add_message(conversation_id, "assistant", answer.answer)
            store.touch_conversation(conversation_id)
        return answer

    @app.websocket("/ws/ask")
    async def ask_socket(websocket: WebSocket):
        platform = websocket.app.state.platform
        auth: AuthService | None = websocket.app.state.auth
        store: FashionBrainStore | None = websocket.app.state.store
        await websocket.accept()
        if platform is None or auth is None or store is None:
            await _socket_error(websocket, FashionGPTError("Platform is not initialized."))
            await websocket.close(code=1011)
            return
        authorization = websocket.headers.get("authorization")
        clearance = websocket.headers.get("x-clearance") or websocket.query_params.get("clearance")
        if not authorization and websocket.query_params.get("token"):
            authorization = f"Bearer {websocket.query_params['token']}"
        try:
            principal = principal_for(platform, auth, authorization, clearance)
        except FashionGPTError as exc:
            await _socket_error(websocket, exc)
            await websocket.close(code=1008)
            return
        session: list[dict[str, str]] = []
        try:
            while True:
                started = time.perf_counter()
                try:
                    raw = await _receive_json(websocket)
                except WebSocketDisconnect:
                    return
                except Exception:
                    await websocket.send_json({"error": "validation_failed", "message": "Send a JSON object with a question."})
                    continue
                try:
                    body = AskRequest.model_validate(raw)
                    history, conversation_id = _resolve_history(body, principal, store, fallback=session)
                    graph_id = body.graph_id
                    if graph_id is None and conversation_id:
                        conv = store.get_conversation(conversation_id, _user_id(principal))
                        graph_id = (conv or {}).get("graph_id")
                    streamed = await _stream_ask(
                        websocket, platform, body.question, principal, history, graph_id=graph_id
                    )
                    answer_text = streamed.get("answer_text") or ""
                    answer_payload = streamed.get("answer") or {}
                    if answer_text or answer_payload:
                        session = [
                            *history,
                            {"role": "user", "content": body.question},
                            {"role": "assistant", "content": answer_text},
                        ]
                        session = session[-8:]
                        if conversation_id:
                            store.add_message(conversation_id, "user", body.question)
                            evidence = list(answer_payload.get("evidence") or [])
                            store.add_message(
                                conversation_id,
                                "assistant",
                                answer_text,
                                meta={
                                    "routes": list(answer_payload.get("routes") or []),
                                    "evidence": evidence,
                                    "caveats": list(answer_payload.get("caveats") or []),
                                    "thinking": streamed.get("thinking") or "",
                                    "reasoning": streamed.get("reasoning") or "",
                                    "url_citations": [
                                        {
                                            "url": item.get("url") or item.get("record_id") or "",
                                            "title": item.get("title") or item.get("url") or "Source",
                                            "excerpt": item.get("excerpt") or "",
                                        }
                                        for item in evidence
                                        if isinstance(item, dict)
                                        and (item.get("source_id") == "web" or item.get("url"))
                                    ],
                                },
                            )
                            store.touch_conversation(conversation_id)
                except ValidationError:
                    await websocket.send_json(
                        {"error": "validation_failed", "message": "question must be a string of 3 to 2000 characters."}
                    )
                    continue
                except FashionGPTError as exc:
                    try:
                        await _socket_error(websocket, exc)
                    except (WebSocketDisconnect, RuntimeError):
                        return
                    continue
                LOGGER.info("ws /ws/ask %.3fs", time.perf_counter() - started)
        except WebSocketDisconnect:
            return

    # --- Wave 3: Admin KG making ---

    @app.get("/v1/admin/kg/health")
    def admin_kg_health(platform: Platform = Depends(current_platform), _admin: Principal = Depends(current_admin)):
        return kg_request(platform.settings, "GET", "/health", clearance=_admin.clearance)

    @app.get("/v1/admin/sources")
    def admin_sources(platform: Platform = Depends(current_platform), _admin: Principal = Depends(current_admin)):
        return kg_request(platform.settings, "GET", "/sources", clearance=_admin.clearance)

    @app.post("/v1/admin/graphs/reload")
    def admin_graphs_reload(platform: Platform = Depends(current_platform), _admin: Principal = Depends(current_admin)):
        """Ask KG to hot-reload snapshots (returns a background job)."""

        return kg_request(platform.settings, "POST", "/graphs/reload", clearance=_admin.clearance, json_body={})

    @app.get("/v1/admin/jobs")
    def admin_jobs(
        kind: str | None = None,
        limit: int = 30,
        platform: Platform = Depends(current_platform),
        _admin: Principal = Depends(current_admin),
    ):
        params: dict[str, object] = {"limit": limit}
        if kind:
            params["kind"] = kind
        return kg_request(platform.settings, "GET", "/jobs", clearance=_admin.clearance, params=params)

    @app.get("/v1/admin/jobs/{job_id}")
    def admin_job(
        job_id: str,
        platform: Platform = Depends(current_platform),
        _admin: Principal = Depends(current_admin),
    ):
        return kg_request(platform.settings, "GET", f"/jobs/{job_id}", clearance=_admin.clearance)

    @app.post("/v1/admin/graphs/{graph_id}/rebuild")
    def admin_graph_rebuild(
        graph_id: str,
        platform: Platform = Depends(current_platform),
        _admin: Principal = Depends(current_admin),
    ):
        return kg_request(
            platform.settings,
            "POST",
            f"/graphs/{graph_id}/rebuild",
            clearance=_admin.clearance,
            json_body={},
        )

    @app.post("/v1/admin/graphs/{graph_id}/sync")
    def admin_graph_sync(
        graph_id: str,
        platform: Platform = Depends(current_platform),
        _admin: Principal = Depends(current_admin),
    ):
        return kg_request(
            platform.settings,
            "POST",
            f"/graphs/{graph_id}/sync",
            clearance=_admin.clearance,
            json_body={},
        )

    @app.get("/v1/admin/graphs")
    def admin_graphs(platform: Platform = Depends(current_platform), _admin: Principal = Depends(current_admin)):
        return list_graphs(platform.settings, clearance=_admin.clearance)

    @app.get("/v1/admin/ontologies")
    def admin_ontologies(platform: Platform = Depends(current_platform), _admin: Principal = Depends(current_admin)):
        return kg_request(platform.settings, "GET", "/ontologies", clearance=_admin.clearance)

    @app.get("/v1/admin/ontologies/{ontology_id}")
    def admin_ontology_detail(
        ontology_id: str,
        platform: Platform = Depends(current_platform),
        _admin: Principal = Depends(current_admin),
    ):
        return kg_request(platform.settings, "GET", f"/ontologies/{ontology_id}", clearance=_admin.clearance)

    @app.get("/v1/admin/ingestion/status")
    def admin_ingestion_status(platform: Platform = Depends(current_platform), _admin: Principal = Depends(current_admin)):
        return kg_request(platform.settings, "GET", "/ingestion/status", clearance=_admin.clearance)

    @app.post("/v1/admin/ingestion/run")
    def admin_ingestion_run(
        body: IngestRunBody,
        platform: Platform = Depends(current_platform),
        _admin: Principal = Depends(current_admin),
    ):
        return kg_request(
            platform.settings,
            "POST",
            "/ingestion/run",
            clearance=_admin.clearance,
            json_body=body.model_dump(exclude_none=True),
        )

    @app.post("/v1/admin/pipeline")
    def admin_pipeline(
        body: PipelineBody,
        platform: Platform = Depends(current_platform),
        _admin: Principal = Depends(current_admin),
    ):
        payload = body.model_dump(exclude_none=True)
        if payload.get("html") and not payload.get("text"):
            payload["text"] = payload.pop("html")
        elif "html" in payload:
            payload.pop("html", None)
        if payload.get("data_b64") and not payload.get("data_base64"):
            payload["data_base64"] = payload.pop("data_b64")
        elif "data_b64" in payload:
            payload.pop("data_b64", None)
        if platform.settings.env not in {"development", "test"} and payload.get("auto_commit"):
            raise PermissionDenied("auto_commit is disabled outside development.")
        return kg_request(platform.settings, "POST", "/kg/pipeline", clearance=_admin.clearance, json_body=payload)

    @app.post("/v1/admin/extract")
    def admin_extract(
        body: ExtractBody,
        platform: Platform = Depends(current_platform),
        _admin: Principal = Depends(current_admin),
    ):
        return kg_request(
            platform.settings,
            "POST",
            "/extract",
            clearance=_admin.clearance,
            json_body=body.model_dump(exclude_none=True),
        )

    @app.post("/v1/admin/relationships/propose")
    def admin_propose(
        body: ProposeBody,
        platform: Platform = Depends(current_platform),
        _admin: Principal = Depends(current_admin),
    ):
        return kg_request(
            platform.settings,
            "POST",
            "/relationships/propose",
            clearance=_admin.clearance,
            json_body=body.model_dump(exclude_none=True),
        )

    @app.post("/v1/admin/structure")
    def admin_structure(
        body: StructureBody,
        platform: Platform = Depends(current_platform),
        _admin: Principal = Depends(current_admin),
    ):
        return kg_request(
            platform.settings,
            "POST",
            "/kg/tools/structure_data",
            clearance=_admin.clearance,
            json_body=body.model_dump(exclude_none=True),
        )

    @app.post("/v1/admin/reviews/queue")
    def admin_queue_review(
        body: QueueReviewBody,
        platform: Platform = Depends(current_platform),
        _admin: Principal = Depends(current_admin),
    ):
        return kg_request(
            platform.settings,
            "POST",
            "/kg/tools/queue_review",
            clearance=_admin.clearance,
            json_body=body.model_dump(exclude_none=True),
        )

    @app.post("/v1/admin/scraping/extract")
    def admin_scraping_extract(
        body: ExtractBody,
        platform: Platform = Depends(current_platform),
        _admin: Principal = Depends(current_admin),
    ):
        return kg_request(
            platform.settings,
            "POST",
            "/scraping/extract",
            clearance=_admin.clearance,
            json_body=body.model_dump(exclude_none=True),
        )

    @app.post("/v1/admin/scraping/chunk")
    def admin_scraping_chunk(
        body: ScrapingChunkBody,
        platform: Platform = Depends(current_platform),
        _admin: Principal = Depends(current_admin),
    ):
        return kg_request(
            platform.settings,
            "POST",
            "/scraping/chunk",
            clearance=_admin.clearance,
            json_body=body.model_dump(exclude_none=True),
        )

    @app.post("/v1/admin/scraping/propose")
    def admin_scraping_propose(
        body: ScrapingProposeBody,
        platform: Platform = Depends(current_platform),
        _admin: Principal = Depends(current_admin),
    ):
        return kg_request(
            platform.settings,
            "POST",
            "/scraping/propose",
            clearance=_admin.clearance,
            json_body=body.model_dump(exclude_none=True),
        )

    @app.post("/v1/admin/scraping/queue")
    def admin_scraping_queue(
        body: ScrapingQueueBody,
        platform: Platform = Depends(current_platform),
        _admin: Principal = Depends(current_admin),
    ):
        return kg_request(
            platform.settings,
            "POST",
            "/scraping/queue",
            clearance=_admin.clearance,
            json_body=body.model_dump(exclude_none=True),
        )

    @app.post("/v1/admin/scraping/commit")
    def admin_scraping_commit(
        body: ScrapingCommitBody,
        platform: Platform = Depends(current_platform),
        _admin: Principal = Depends(current_admin),
    ):
        result = kg_request(
            platform.settings,
            "POST",
            "/scraping/commit",
            clearance=_admin.clearance,
            json_body=body.model_dump(exclude_none=True),
        )
        # KG already has in-memory nodes from commit; reload keeps disk/registry aligned.
        reload = reload_kg_graphs(platform.settings, clearance=_admin.clearance)
        if isinstance(result, dict):
            result = {**result, "reload": reload}
        return result

    @app.post("/v1/admin/pipeline/upload")
    async def admin_pipeline_upload(
        file: UploadFile = File(...),
        kind: str | None = Form(None),
        use_model: bool = Form(True),
        clean: bool = Form(False),
        auto_commit: bool = Form(False),
        graph_id: str | None = Form(None),
        ontology_id: str | None = Form(None),
        platform: Platform = Depends(current_platform),
        _admin: Principal = Depends(current_admin),
    ):
        if platform.settings.env not in {"development", "test"} and auto_commit:
            raise PermissionDenied("auto_commit is disabled outside development.")
        data = await file.read()
        if not data:
            raise ValidationFailed("Uploaded file is empty.")
        files = {"file": (file.filename or "upload.bin", data, file.content_type or "application/octet-stream")}
        form = {
            "kind": kind or "",
            "use_model": str(use_model).lower(),
            "clean": str(clean).lower(),
            "auto_commit": str(auto_commit).lower(),
            "graph_id": graph_id or "",
            "ontology_id": ontology_id or "",
        }
        return kg_request(
            platform.settings,
            "POST",
            "/kg/pipeline/upload",
            clearance=_admin.clearance,
            data=form,
            files=files,
        )

    @app.get("/v1/admin/reviews")
    def admin_reviews(platform: Platform = Depends(current_platform), _admin: Principal = Depends(current_admin)):
        return kg_request(platform.settings, "GET", "/kg/reviews", clearance=_admin.clearance)

    @app.get("/v1/admin/reviews/{review_id}")
    def admin_review(review_id: str, platform: Platform = Depends(current_platform), _admin: Principal = Depends(current_admin)):
        return kg_request(platform.settings, "GET", f"/kg/reviews/{review_id}", clearance=_admin.clearance)

    @app.patch("/v1/admin/reviews/{review_id}")
    def admin_review_update(
        review_id: str,
        body: ReviewUpdateBody,
        platform: Platform = Depends(current_platform),
        _admin: Principal = Depends(current_admin),
    ):
        return kg_request(
            platform.settings,
            "POST",
            f"/kg/reviews/{review_id}",
            clearance=_admin.clearance,
            json_body=body.model_dump(exclude_none=True),
        )

    @app.post("/v1/admin/reviews/{review_id}/approve")
    def admin_review_approve(
        review_id: str,
        platform: Platform = Depends(current_platform),
        _admin: Principal = Depends(current_admin),
    ):
        return kg_request(
            platform.settings,
            "POST",
            f"/kg/reviews/{review_id}",
            clearance=_admin.clearance,
            json_body={"decision": "approved"},
        )

    @app.post("/v1/admin/reviews/{review_id}/commit")
    def admin_review_commit(
        review_id: str,
        platform: Platform = Depends(current_platform),
        _admin: Principal = Depends(current_admin),
    ):
        result = kg_request(
            platform.settings,
            "POST",
            f"/kg/reviews/{review_id}/commit",
            clearance=_admin.clearance,
            json_body={},
        )
        reload = reload_kg_graphs(platform.settings, clearance=_admin.clearance)
        if isinstance(result, dict):
            result = {**result, "reload": reload}
        return result

    @app.post("/v1/admin/graphs/{graph_name}/query")
    def admin_graph_query(
        graph_name: str,
        body: GraphQueryBody,
        platform: Platform = Depends(current_platform),
        _admin: Principal = Depends(current_admin),
    ):
        return kg_request(
            platform.settings,
            "POST",
            f"/graphs/{graph_name}/query",
            clearance=_admin.clearance,
            json_body=body.model_dump(),
        )

    @app.post("/v1/admin/tools/{tool_name}")
    def admin_tool(
        tool_name: str,
        body: ToolBody,
        platform: Platform = Depends(current_platform),
        _admin: Principal = Depends(current_admin),
    ):
        return kg_request(
            platform.settings,
            "POST",
            f"/kg/tools/{tool_name}",
            clearance=_admin.clearance,
            json_body=body.payload,
        )

    return app


def principal_for(
    platform: Platform,
    auth: AuthService | None,
    authorization: str | None,
    clearance: str | None,
) -> Principal:
    settings = platform.settings
    if auth is not None:
        jwt_principal = auth.principal_from_authorization(authorization, clearance)
        if jwt_principal is not None:
            return jwt_principal
    if settings.api_token:
        if authorization != f"Bearer {settings.api_token}":
            raise Unauthorized("Authentication is required.")
        level = clearance or "internal"
        if level not in CLEARANCE_RANK:
            raise ValidationFailed("Clearance must be public, internal, or confidential.")
        return Principal(subject="api", clearance=level, roles=["admin", "user"])
    if authorization and authorization.lower().startswith("bearer "):
        # JWT present but invalid / expired
        raise Unauthorized("Authentication is required.")
    if settings.env not in {"development", "test"}:
        raise Unauthorized("An API token or login is required outside development.")
    # Local demo: product graphs are classified internal; default matches conversation clearance.
    level = clearance or "internal"
    if level not in CLEARANCE_RANK:
        raise ValidationFailed("Clearance must be public, internal, or confidential.")
    return Principal(subject="api", clearance=level, roles=["admin", "user"])


def _user_id(principal: Principal) -> str:
    if principal.subject in {"", "anonymous"}:
        raise Unauthorized("Login is required.")
    return principal.subject


def _conversation_public(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item["id"],
        "title": item["title"],
        "clearance": item["clearance"],
        "archived": bool(item["archived"]),
        "graph_id": item.get("graph_id") or None,
        "created_at": item["created_at"],
        "updated_at": item["updated_at"],
    }


def _title_from(text: str) -> str:
    clean = " ".join((text or "").split())
    return clean[:48] + ("…" if len(clean) > 48 else "")


def _with_memory(history: list[dict[str, str]], store: FashionBrainStore, user_id: str) -> list[dict[str, str]]:
    notes = store.memory_texts(user_id, limit=10)
    if not notes:
        return history
    block = "Saved user notes:\n- " + "\n- ".join(notes)
    return [{"role": "user", "content": block}, {"role": "assistant", "content": "I will use those notes when relevant."}, *history]


def _resolve_history(
    body: AskRequest,
    principal: Principal,
    store: FashionBrainStore,
    fallback: list[dict[str, str]] | None = None,
) -> tuple[list[dict[str, str]], str | None]:
    if body.conversation_id:
        user_id = _user_id(principal)
        conversation = store.get_conversation(body.conversation_id, user_id)
        if not conversation:
            raise NotFoundError("Conversation was not found.")
        history = store.history_for_ask(body.conversation_id)
        history = _with_memory(history, store, user_id)
        if conversation["title"] in {"", "New chat"}:
            store.touch_conversation(body.conversation_id, title=_title_from(body.question))
        return history, body.conversation_id
    history = [turn.model_dump() for turn in body.history] or list(fallback or [])
    if principal.subject not in {"api", "anonymous"}:
        history = _with_memory(history, store, principal.subject)
    return history, None


async def _receive_json(websocket: WebSocket):
    message = await websocket.receive()
    if message["type"] == "websocket.disconnect":
        raise WebSocketDisconnect(message.get("code", 1000))
    if message.get("text") is not None:
        raw = message["text"]
    elif message.get("bytes") is not None:
        raw = message["bytes"].decode("utf-8")
    else:
        raise ValueError("empty frame")
    return json.loads(raw)


async def _socket_error(websocket: WebSocket, exc: FashionGPTError) -> None:
    await websocket.send_json({"error": exc.code, "message": exc.message})


def _visible_pieces(event: dict) -> list[dict]:
    text = event.get("text") or ""
    kind = event.get("type")
    if kind == "source":
        return [event]
    if kind != "answer" or len(text) <= 12:
        return [event]
    return [{"type": "answer", "text": text[index : index + 12]} for index in range(0, len(text), 12)]


def _emit(send_stream, event: dict) -> bool:
    try:
        anyio.from_thread.run(send_stream.send, event)
    except (anyio.BrokenResourceError, anyio.ClosedResourceError, anyio.EndOfStream):
        return False
    return True


async def _stream_ask(
    websocket: WebSocket,
    platform: Platform,
    question: str,
    principal: Principal,
    history: list | None = None,
    graph_id: str | None = None,
) -> dict:
    send_stream, receive_stream = anyio.create_memory_object_stream(64)
    result = {"answer_text": "", "answer": {}, "thinking": "", "reasoning": ""}

    def produce() -> None:
        try:
            for event in iter_ask_events(platform, question, principal, history, graph_id=graph_id):
                pieces = _visible_pieces(event)
                for piece in pieces:
                    if not _emit(send_stream, piece):
                        return
                    if piece.get("type") == "answer" and piece.get("text"):
                        time.sleep(0.03)
                    elif piece.get("type") == "source":
                        time.sleep(0.05)
        except (anyio.BrokenResourceError, anyio.ClosedResourceError, anyio.EndOfStream):
            return
        except FashionGPTError as exc:
            _emit(send_stream, {"type": "error", "error": exc.code, "message": exc.message})
        except Exception as exc:
            _emit(send_stream, {"type": "error", "error": "error", "message": f"{exc.__class__.__name__}: {exc}"})
        finally:
            try:
                anyio.from_thread.run(send_stream.aclose)
            except (anyio.BrokenResourceError, anyio.ClosedResourceError, RuntimeError):
                return

    try:
        with anyio.fail_after(platform.settings.request_timeout_seconds):
            async with anyio.create_task_group() as group:
                group.start_soon(anyio.to_thread.run_sync, produce)
                async with receive_stream:
                    async for event in receive_stream:
                        kind = event.get("type")
                        text = str(event.get("text") or "")
                        if kind == "thinking" and text:
                            result["thinking"] += text
                        elif kind == "reasoning" and text:
                            result["reasoning"] += text
                        elif kind == "done":
                            answer = event.get("answer") or {}
                            result["answer"] = answer if isinstance(answer, dict) else {}
                            result["answer_text"] = str(result["answer"].get("answer") or "")
                        await websocket.send_json(event)
    except (WebSocketDisconnect, anyio.BrokenResourceError, anyio.ClosedResourceError):
        return result
    except TimeoutError as exc:
        raise RequestTimeout("The request exceeded the configured timeout.") from exc
    return result


async def _run(platform: Platform, func, *args, **kwargs):
    try:
        with anyio.fail_after(platform.settings.request_timeout_seconds):
            return await anyio.to_thread.run_sync(lambda: func(*args, **kwargs))
    except TimeoutError as exc:
        raise RequestTimeout("The request exceeded the configured timeout.") from exc
