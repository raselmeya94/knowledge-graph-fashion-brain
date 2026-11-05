import json
from io import BytesIO

import pytest

from fashion_brain.llm import build_model_provider
from knowledge_graph.config import Settings
from knowledge_graph.errors import ModelUnavailableError


def test_zero_request_timeout_means_no_deadline():
    assert Settings(request_timeout_seconds=0).request_timeout_seconds is None
    assert Settings(request_timeout_seconds="unlimited").request_timeout_seconds is None


def test_openrouter_requires_a_key_before_calling_the_network(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("FASHIONGPT_MODEL_API_KEY", raising=False)
    settings = Settings(model_provider="openrouter", model_api_key="", answer_mode="llm")
    provider = build_model_provider(settings)
    assert provider.name == "openrouter"
    with pytest.raises(ModelUnavailableError, match="API key"):
        provider.complete("Which materials are used?", timeout=1)


def test_openrouter_posts_to_chat_completions(monkeypatch):
    captured = {}

    class Response(BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["headers"] = {key.lower(): value for key, value in req.header_items()}
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return Response(json.dumps({"choices": [{"message": {"content": "Cotton is used by PLMEX-DEMO-001."}}]}).encode())

    monkeypatch.setattr("fashion_brain.llm.request.urlopen", fake_urlopen)
    settings = Settings(
        model_provider="openrouter",
        model_api_key="test-key",
        model_name="openai/gpt-4o-mini",
        model_base_url="https://openrouter.ai/api/v1",
        answer_mode="llm",
    )
    answer = build_model_provider(settings).complete("Question: materials", timeout=5)
    assert answer == "Cotton is used by PLMEX-DEMO-001."
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["headers"]["authorization"] == "Bearer test-key"
    assert captured["headers"]["x-openrouter-title"] == "Fashion Brain"
    assert captured["body"]["model"] == "openai/gpt-4o-mini"
    assert captured["body"]["temperature"] == 0


def test_stream_splits_thinking_reasoning_and_answer(monkeypatch):
    captured = {}
    sse = "\n".join(
        [
            'data: {"choices":[{"delta":{"thinking":"Plan. "}}]}',
            'data: {"choices":[{"delta":{"reasoning_content":"Because. "}}]}',
            'data: {"choices":[{"delta":{"content":"Four."},"annotations":[{"type":"url_citation","url_citation":{"url":"https://example.com","title":"Note"}}]}]}',
            "data: [DONE]",
            "",
        ]
    ).encode()

    class Response:
        def __iter__(self):
            return iter(sse.splitlines(keepends=True))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(req, timeout):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return Response()

    monkeypatch.setattr("fashion_brain.llm.request.urlopen", fake_urlopen)
    settings = Settings(
        model_provider="openrouter",
        model_api_key="test-key",
        model_name="deepseek/deepseek-v4-pro",
        model_base_url="https://openrouter.ai/api/v1",
        answer_mode="llm",
    )
    sink = {}
    events = list(build_model_provider(settings).iter_chat([{"role": "user", "content": "2+2"}], timeout=5, sink=sink))
    assert captured["body"]["stream"] is True
    assert events == [("thinking", "Plan. "), ("reasoning", "Because. "), ("answer", "Four.")]
    message = sink["body"]["choices"][0]["message"]
    assert message["content"] == "Four."
    assert message["reasoning"] == "Because. "
    assert message["annotations"][0]["url_citation"]["url"] == "https://example.com"
