"""Fashion Brain model providers: OpenRouter, Ollama, or any OpenAI-compatible API."""

from __future__ import annotations

import base64
import json
import os
from typing import Protocol
from urllib import error, request

from knowledge_graph.config import Settings
from knowledge_graph.errors import ModelUnavailableError

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_MODEL = "~openai/gpt-sol-latest"
OPENROUTER_DEFAULT_IMAGE_MODEL = "google/gemini-3.1-flash-image"
OLLAMA_DEFAULT_BASE = "http://127.0.0.1:11434"
OLLAMA_DEFAULT_MODEL = "llama3.1:8b"


class ModelProvider(Protocol):
    name: str

    def complete(
        self,
        prompt: str,
        *,
        timeout: float | None,
        system: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Return model text. Raise ModelUnavailableError when the provider cannot run."""


class NullModelProvider:
    name = "none"

    def complete(
        self,
        prompt: str,
        *,
        timeout: float | None,
        system: str | None = None,
        temperature: float | None = None,
    ) -> str:
        raise ModelUnavailableError("No model provider is configured. Set FASHIONGPT_MODEL_PROVIDER to replace this.")

    def chat(self, messages: list[dict], *, timeout: float | None, tools: list[dict] | None = None, tool_choice: str | dict | None = None) -> dict:
        raise ModelUnavailableError("No model provider is configured. Set FASHIONGPT_MODEL_PROVIDER to replace this.")

    def iter_chat(self, messages: list[dict], *, timeout: float | None, tools: list[dict] | None = None, tool_choice: str | dict | None = None, sink: dict | None = None):
        raise ModelUnavailableError("No model provider is configured. Set FASHIONGPT_MODEL_PROVIDER to replace this.")
        yield ("answer", "")


class OpenAICompatibleProvider:
    """Chat-completions client. The API key stays in the environment, not in source code."""

    def __init__(self, base_url: str, api_key: str, model: str, *, extra_headers: dict[str, str] | None = None):
        self.name = "openai-compatible"
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.extra_headers = extra_headers or {}

    def complete(
        self,
        prompt: str,
        *,
        timeout: float | None,
        system: str | None = None,
        temperature: float | None = None,
    ) -> str:
        system_text = system or (
            "You are Fashion Brain. Answer only from the supplied product evidence. "
            "Write a short natural briefing for a colleague — not a Q&A dump or FAQ. "
            "If evidence is missing, say so plainly. Do not invent products, counts, or citations."
        )
        # Timing is recorded inside chat() — avoid double-counting.
        body = self.chat(
            [
                {"role": "system", "content": system_text},
                {"role": "user", "content": prompt},
            ],
            timeout=timeout,
            temperature=temperature,
        )
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelUnavailableError("Model provider returned an unexpected payload.") from exc
        if not content:
            raise ModelUnavailableError("Model provider returned an empty answer.")
        return str(content)

    def chat(
        self,
        messages: list[dict],
        *,
        timeout: float | None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        temperature: float | None = None,
    ) -> dict:
        """Post one chat completion and return the JSON body, including tool calls."""

        from fashion_brain.stage_timing import estimate_tokens, timed

        prompt_chars = sum(len(str(item.get("content") or "")) for item in messages)
        with timed(
            "llm_chat",
            provider=self.name,
            model=self.model,
            messages=len(messages),
            tools=bool(tools),
            prompt_tokens_est=estimate_tokens(*(str(item.get("content") or "") for item in messages)),
            prompt_chars=prompt_chars,
        ):
            return self._post(
                messages,
                timeout=timeout,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
            )

    def generate_image(
        self,
        prompt: str,
        *,
        model: str,
        timeout: float | None,
        aspect_ratio: str | None = None,
        seed: int | None = None,
    ) -> tuple[bytes, str]:
        """Generate one image. Returns the file bytes and a media type such as image/png."""

        image_model = _openrouter_model(model)
        payload: dict = {
            "model": image_model,
            "prompt": prompt,
            "aspect_ratio": (aspect_ratio or "3:4").strip() or "3:4",
            "output_format": "png",
        }
        if seed is not None:
            payload["seed"] = int(seed)
        try:
            body = self._post_json("/images", payload, timeout=timeout)
        except ModelUnavailableError as exc:
            if "HTTP 400" not in exc.message:
                raise
            # Retry without aspect_ratio (some models reject it) but keep seed so
            # /advanced_garment_design views stay correlated.
            retry: dict = {"model": image_model, "prompt": prompt, "output_format": "png"}
            if seed is not None:
                retry["seed"] = int(seed)
            body = self._post_json("/images", retry, timeout=timeout)
        return _decode_image(body)

    def iter_chat(
        self,
        messages: list[dict],
        *,
        timeout: float | None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        sink: dict | None = None,
    ):
        """Yield (thinking|reasoning|answer, text) deltas from one streamed completion."""

        content: list[str] = []
        reasoning: list[str] = []
        thinking: list[str] = []
        annotations: list = []
        tool_calls: dict[int, dict] = {}
        from fashion_brain.stage_timing import estimate_tokens, timed

        with timed(
            "llm_stream",
            provider=self.name,
            model=self.model,
            messages=len(messages),
            tools=bool(tools),
            prompt_tokens_est=estimate_tokens(*(str(item.get("content") or "") for item in messages)),
        ):
            try:
                with self._open(messages, timeout=timeout, tools=tools, tool_choice=tool_choice, stream=True) as response:
                    for kind, text, extra in _iter_sse(response):
                        if kind == "annotation" and isinstance(extra, list):
                            annotations.extend(extra)
                            continue
                        if kind == "tool_call" and isinstance(extra, dict):
                            _merge_tool_call(tool_calls, extra)
                            continue
                        if not text:
                            continue
                        if kind == "thinking":
                            thinking.append(text)
                        elif kind == "reasoning":
                            reasoning.append(text)
                        else:
                            kind = "answer"
                            content.append(text)
                        yield kind, text
            except error.HTTPError as exc:
                raise _model_http_error(exc) from exc
            except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                raise ModelUnavailableError(f"Model provider is unavailable: {exc.__class__.__name__}.") from exc
        if sink is not None:
            message: dict = {"role": "assistant", "content": "".join(content)}
            if reasoning:
                message["reasoning"] = "".join(reasoning)
            if thinking:
                message["thinking"] = "".join(thinking)
            if annotations:
                message["annotations"] = annotations
            if tool_calls:
                message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
            sink["body"] = {"choices": [{"message": message}]}

    def _post(
        self,
        messages: list[dict],
        *,
        timeout: float | None,
        tools: list[dict] | None,
        tool_choice: str | dict | None,
        temperature: float | None = None,
    ) -> dict:
        try:
            with self._open(
                messages,
                timeout=timeout,
                tools=tools,
                tool_choice=tool_choice,
                stream=False,
                temperature=temperature,
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            raise _model_http_error(exc) from exc
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ModelUnavailableError(f"Model provider is unavailable: {exc.__class__.__name__}.") from exc

    def _post_json(self, path: str, payload: dict, *, timeout: float | None) -> dict:
        if not self.base_url or not self.api_key:
            raise ModelUnavailableError("Model base URL and API key are required.")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        req = request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            raise _model_http_error(exc) from exc
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ModelUnavailableError(f"Model provider is unavailable: {exc.__class__.__name__}.") from exc

    def _open(
        self,
        messages: list[dict],
        *,
        timeout: float | None,
        tools: list[dict] | None,
        tool_choice: str | dict | None,
        stream: bool,
        temperature: float | None = None,
    ):
        if not self.base_url or not self.model:
            raise ModelUnavailableError("Model base URL and model name are required.")
        if not self.api_key and self.name not in {"ollama"}:
            raise ModelUnavailableError("Model API key is required.")
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else 0,
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if stream:
            payload["stream"] = True
        headers = {
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        return request.urlopen(req, timeout=timeout)


class OpenRouterProvider(OpenAICompatibleProvider):
    def __init__(self, api_key: str, model: str, *, base_url: str, app_url: str, app_title: str):
        super().__init__(
            base_url or OPENROUTER_BASE_URL,
            api_key,
            _openrouter_model(model or OPENROUTER_DEFAULT_MODEL),
            extra_headers={
                "HTTP-Referer": app_url,
                "X-OpenRouter-Title": app_title,
            },
        )
        self.name = "openrouter"


class OllamaProvider(OpenAICompatibleProvider):
    """Local Ollama via its OpenAI-compatible ``/v1/chat/completions`` API."""

    def __init__(self, base_url: str, model: str, *, timeout_seconds: float | None = None):
        root = (base_url or OLLAMA_DEFAULT_BASE).rstrip("/")
        if root.endswith("/v1"):
            api_base = root
        else:
            api_base = f"{root}/v1"
        super().__init__(
            api_base,
            api_key="ollama",  # Ollama ignores auth; keep a non-empty token for shared client paths
            model=(model or OLLAMA_DEFAULT_MODEL).strip(),
        )
        self.name = "ollama"
        self.timeout_seconds = timeout_seconds

    def complete(
        self,
        prompt: str,
        *,
        timeout: float | None,
        system: str | None = None,
        temperature: float | None = None,
    ) -> str:
        return super().complete(
            prompt,
            timeout=timeout or self.timeout_seconds,
            system=system,
            temperature=temperature,
        )

    def chat(
        self,
        messages: list[dict],
        *,
        timeout: float | None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        temperature: float | None = None,
    ) -> dict:
        return super().chat(
            messages,
            timeout=timeout or self.timeout_seconds,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
        )

    def iter_chat(
        self,
        messages: list[dict],
        *,
        timeout: float | None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        sink: dict | None = None,
    ):
        yield from super().iter_chat(
            messages,
            timeout=timeout or self.timeout_seconds,
            tools=tools,
            tool_choice=tool_choice,
            sink=sink,
        )


def _decode_image(body: dict) -> tuple[bytes, str]:
    try:
        item = body["data"][0]
        raw = base64.b64decode(item["b64_json"])
        media = str(item.get("media_type") or "image/png")
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ModelUnavailableError("Image provider returned an unexpected payload.") from exc
    if not raw:
        raise ModelUnavailableError("Image provider returned an empty image.")
    return raw, media


def _openrouter_model(model: str) -> str:
    text = model.strip()
    if text.startswith("~"):
        text = text[1:]
    return text


def _model_http_error(exc: error.HTTPError) -> ModelUnavailableError:
    detail = _http_error_detail(exc)
    message = f"Model provider returned HTTP {exc.code}."
    if detail:
        message = f"{message} {detail}"
    return ModelUnavailableError(message)


def _iter_sse(response):
    for raw in response:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            if data == "[DONE]":
                break
            continue
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        try:
            choice = chunk["choices"][0]
        except (KeyError, IndexError, TypeError):
            continue
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            continue
        annotations = delta.get("annotations") or choice.get("annotations") or []
        if annotations:
            yield "annotation", "", list(annotations)
        for call in delta.get("tool_calls") or []:
            if isinstance(call, dict):
                yield "tool_call", "", call
        if delta.get("thinking"):
            yield "thinking", str(delta["thinking"]), None
        reasoning = delta.get("reasoning_content")
        if reasoning is None:
            reasoning = delta.get("reasoning")
        if reasoning:
            yield "reasoning", str(reasoning), None
        if delta.get("content"):
            yield "answer", str(delta["content"]), None


def _merge_tool_call(tool_calls: dict[int, dict], call: dict) -> None:
    index = int(call.get("index") or 0)
    slot = tool_calls.setdefault(
        index,
        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
    )
    if call.get("id"):
        slot["id"] = call["id"]
    function = call.get("function") or {}
    if function.get("name"):
        slot["function"]["name"] += function["name"]
    if function.get("arguments"):
        slot["function"]["arguments"] += function["arguments"]


def _http_error_detail(exc: error.HTTPError) -> str:
    try:
        payload = json.loads(exc.read().decode("utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return ""
    message = payload.get("error", payload) if isinstance(payload, dict) else ""
    if isinstance(message, dict):
        message = message.get("message", "")
    return " ".join(str(message).split())[:200]


def resolve_model_api_key(settings: Settings) -> str:
    return settings.model_api_key or os.environ.get("OPENROUTER_API_KEY", "")


def active_model_info(settings: Settings) -> dict[str, str]:
    """Human-readable active provider/model for logs and health."""

    provider = str(settings.model_provider or "").strip().casefold() or "none"
    if provider == "ollama":
        model = str(getattr(settings, "ollama_model", None) or settings.model_name or OLLAMA_DEFAULT_MODEL)
        base = str(getattr(settings, "ollama_base_url", None) or OLLAMA_DEFAULT_BASE)
    elif provider in {"", "none", "null"}:
        model = ""
        base = ""
        provider = "none"
    else:
        model = str(settings.model_name or "")
        if model.startswith("~"):
            model = model[1:]
        base = str(settings.model_base_url or "")
    return {"provider": provider, "model": model, "base_url": base}


def format_active_model(settings: Settings) -> str:
    info = active_model_info(settings)
    if info["provider"] == "none":
        return "provider=none"
    return f"provider={info['provider']} model={info['model']} base={info['base_url']}"


def build_model_provider(settings: Settings) -> ModelProvider:
    provider = str(settings.model_provider or "").strip().casefold()
    if provider in {"", "none", "null"}:
        return NullModelProvider()
    if provider == "openrouter":
        return OpenRouterProvider(
            resolve_model_api_key(settings),
            settings.model_name,
            base_url=settings.model_base_url,
            app_url=settings.model_app_url,
            app_title=settings.model_app_title,
        )
    if provider == "ollama":
        model = (getattr(settings, "ollama_model", None) or settings.model_name or OLLAMA_DEFAULT_MODEL).strip()
        if model.startswith("~"):
            model = model[1:]
        # Prefer dedicated Ollama model id (llama3.1:8b), not an OpenRouter slug.
        if "/" in model and ":" not in model:
            model = getattr(settings, "ollama_model", None) or OLLAMA_DEFAULT_MODEL
        return OllamaProvider(
            str(getattr(settings, "ollama_base_url", None) or OLLAMA_DEFAULT_BASE),
            model,
            timeout_seconds=float(getattr(settings, "ollama_timeout_seconds", 60) or 60),
        )
    if provider == "openai-compatible":
        return OpenAICompatibleProvider(settings.model_base_url, resolve_model_api_key(settings), settings.model_name)
    raise ModelUnavailableError(
        f"Unknown model provider '{settings.model_provider}'. "
        "Use 'openrouter', 'ollama', 'openai-compatible', or 'none'."
    )


def resolve_image_model(settings: Settings) -> str:
    """OpenRouter image model id for garment samples."""

    return str(getattr(settings, "image_model", None) or "").strip() or OPENROUTER_DEFAULT_IMAGE_MODEL


def build_image_provider(settings: Settings) -> OpenRouterProvider:
    """Always OpenRouter for sample images — independent of chat model_provider (e.g. ollama)."""

    key = resolve_model_api_key(settings)
    if not key:
        raise ModelUnavailableError(
            "Sample images need an OpenRouter API key. Set FASHIONGPT_MODEL_API_KEY or OPENROUTER_API_KEY."
        )
    return OpenRouterProvider(
        key,
        resolve_image_model(settings),
        base_url=settings.model_base_url or OPENROUTER_BASE_URL,
        app_url=settings.model_app_url,
        app_title=settings.model_app_title,
    )
