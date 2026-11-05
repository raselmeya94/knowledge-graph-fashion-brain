"""Local Ollama client used by extraction and relationship tools."""

from __future__ import annotations

import base64
import json
import re
from urllib import error, request

from knowledge_graph.config import Settings
from knowledge_graph.errors import ModelUnavailableError


_TEXT_PREFERENCE = ("llama3.1:8b", "llama3.2:3b", "gemma4:31b", "qwen2.5:0.5b")
_VISION_MARKERS = ("llava", "moondream", "bakllava", "minicpm-v", "vision")


def _base_name(name: str) -> str:
    return name.strip().lower().split(":", 1)[0]


def _find_installed(wanted: str, installed: list[str]) -> str | None:
    if not wanted.strip():
        return None
    for name in installed:
        if name.lower() == wanted.lower():
            return name
    wanted_base = _base_name(wanted)
    for name in installed:
        if _base_name(name) == wanted_base:
            return name
    return None


class OllamaClient:
    """One chat client. Tools call it; they do not open their own HTTP clients."""

    def __init__(self, settings: Settings):
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_model
        self.vision_model = settings.ollama_vision_model
        self.timeout = settings.ollama_timeout_seconds
        self._installed: list[str] | None = None

    def health(self) -> dict:
        try:
            installed = self._model_names()
        except ModelUnavailableError as exc:
            return {"status": "unavailable", "message": exc.message, "base_url": self.base_url}
        text_model = self._choose_text_model(installed)
        vision_model = self._choose_vision_model(installed)
        return {
            "status": "ok",
            "base_url": self.base_url,
            "model": text_model,
            "vision_model": vision_model,
            "models": installed,
        }

    def complete(self, prompt: str, *, system: str | None = None, json_mode: bool = False) -> str:
        return self._chat(
            prompt,
            system=system,
            images=None,
            json_mode=json_mode,
            model=self._choose_text_model(self._model_names()),
        )

    def describe_image(self, image: bytes, *, prompt: str) -> str:
        vision_model = self._choose_vision_model(self._model_names())
        if not vision_model:
            installed = ", ".join(self._model_names()) or "none"
            raise ModelUnavailableError(
                f"No vision model is installed. Installed models: {installed}. "
                "Text and relationship tools use the installed chat model."
            )
        encoded = base64.b64encode(image).decode("ascii")
        return self._chat(prompt, system=None, images=[encoded], json_mode=False, model=vision_model)

    def _chat(
        self,
        prompt: str,
        *,
        system: str | None,
        images: list[str] | None,
        json_mode: bool,
        model: str,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        user: dict = {"role": "user", "content": prompt}
        if images:
            user["images"] = images
        messages.append(user)
        payload: dict = {"model": model, "messages": messages, "stream": False}
        if json_mode:
            payload["format"] = "json"
        body = self._post("/api/chat", payload, timeout=self.timeout)
        try:
            content = body["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise ModelUnavailableError("Ollama returned an unexpected payload.") from exc
        if not str(content).strip():
            raise ModelUnavailableError("Ollama returned an empty response.")
        return str(content)

    def _model_names(self) -> list[str]:
        if self._installed is None:
            body = self._get("/api/tags", timeout=2)
            self._installed = [item.get("name", "") for item in body.get("models", []) if isinstance(item, dict) and item.get("name")]
        return list(self._installed)

    def _choose_text_model(self, installed: list[str]) -> str:
        found = _find_installed(self.model, installed)
        if found:
            return found
        for preferred in _TEXT_PREFERENCE:
            found = _find_installed(preferred, installed)
            if found:
                return found
        return installed[0] if installed else self.model

    def _choose_vision_model(self, installed: list[str]) -> str:
        found = _find_installed(self.vision_model, installed)
        if found:
            return found
        for name in installed:
            lowered = name.lower()
            if any(marker in lowered for marker in _VISION_MARKERS):
                return name
        return ""

    def _get(self, path: str, *, timeout: float) -> dict:
        return self._open(path, None, timeout=timeout)

    def _post(self, path: str, payload: dict, *, timeout: float) -> dict:
        raw = json.dumps(payload).encode("utf-8")
        return self._open(path, raw, timeout=timeout)

    def _open(self, path: str, data: bytes | None, *, timeout: float) -> dict:
        req = request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"} if data is not None else {},
            method="POST" if data is not None else "GET",
        )
        try:
            with request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise ModelUnavailableError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise ModelUnavailableError(f"Ollama is not reachable at {self.base_url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise ModelUnavailableError("Ollama timed out.") from exc
        except json.JSONDecodeError as exc:
            raise ModelUnavailableError("Ollama returned invalid JSON.") from exc


def parse_model_json(text: str) -> dict:
    """Accept a JSON object, including one wrapped in a markdown fence."""

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Model response did not contain a JSON object.")
    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Model response JSON must be an object.")
    return payload


class OllamaTool:
    name = "ollama"
    description = "Call the local Ollama server for text completion or image understanding."

    def run(self, payload: dict, ctx) -> dict:
        action = str(payload.get("action") or "complete")
        if action == "health":
            return ctx.ollama.health()
        if action == "describe_image":
            data = payload.get("data")
            if not isinstance(data, bytes):
                from knowledge_graph.construction.detect import payload_bytes

                data = payload_bytes(payload)
            prompt = str(payload.get("prompt") or "Describe the image in plain text. Include names, products, and visible relationships.")
            return {"text": ctx.ollama.describe_image(data, prompt=prompt), "model": ctx.ollama.vision_model}
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            from knowledge_graph.errors import ValidationFailed

            raise ValidationFailed("prompt is required.")
        text = ctx.ollama.complete(prompt, system=payload.get("system"), json_mode=bool(payload.get("json_mode")))
        return {"text": text, "model": ctx.ollama.model}
