"""Tests for scry.llm — LLM provider abstraction (workstream W5a).

Covers:
- OllamaProvider: happy-path, json_mode, usage, error cases
- OpenAIProvider: happy-path, json_mode, usage, error cases
- AnthropicProvider: happy-path, json_mode, usage, error cases
- LiteLLMProvider: import guard raises LLMError when litellm missing
- Error cases: 4xx / 5xx / timeout / invalid JSON → LLMError subclasses
- make_provider: factory returns correct concrete type
- Default config (Ollama unreachable): LLMNetworkError with guidance message
- Concurrency: 10 concurrent complete() calls all succeed (mock transport)
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from scry.llm import (
    AnthropicProvider,
    LiteLLMProvider,
    LLMError,
    LLMNetworkError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    LLMResponseError,
    OllamaProvider,
    OpenAIProvider,
    make_provider,
)
from scry.models import LLMConfig

# ──────────────────────────────────────────────────────────────────────
# Mock transport helpers
# ──────────────────────────────────────────────────────────────────────


class _MockTransport(httpx.AsyncBaseTransport):
    """Returns a fixed JSON response for every request."""

    def __init__(self, status_code: int, body: dict[str, Any]) -> None:
        self._status_code = status_code
        self._body = json.dumps(body).encode()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(self._status_code, content=self._body)


class _RawTransport(httpx.AsyncBaseTransport):
    """Returns a fixed raw-bytes response (for testing non-JSON bodies)."""

    def __init__(self, status_code: int, body: bytes) -> None:
        self._status_code = status_code
        self._body = body

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(self._status_code, content=self._body)


class _ErrorTransport(httpx.AsyncBaseTransport):
    """Raises the given httpx exception for every request."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise self._exc


class _RecordingTransport(httpx.AsyncBaseTransport):
    """Records the last request body and returns a fixed response."""

    def __init__(self, status_code: int, body: dict[str, Any]) -> None:
        self._status_code = status_code
        self._body_bytes = json.dumps(body).encode()
        self.last_request: httpx.Request | None = None
        self.last_request_json: dict[str, Any] | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.last_request = request
        self.last_request_json = json.loads(request.content)
        return httpx.Response(self._status_code, content=self._body_bytes)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

_BASE_REQUEST = LLMRequest(
    system="You are a helpful assistant.",
    messages=[{"role": "user", "content": "Hello"}],
)

_OLLAMA_OK = {
    "model": "llama3.2",
    "message": {"role": "assistant", "content": "Hi there!"},
    "done_reason": "stop",
    "prompt_eval_count": 10,
    "eval_count": 5,
}

# Same shape as _OLLAMA_OK but with valid JSON content for json_mode tests.
_OLLAMA_OK_JSON = {
    **_OLLAMA_OK,
    "message": {"role": "assistant", "content": '{"reply": "Hi there!"}'},
}

_OPENAI_OK = {
    "id": "chatcmpl-abc",
    "model": "gpt-4o-mini",
    "choices": [
        {
            "message": {"role": "assistant", "content": "Hi there!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 15,
        "completion_tokens": 8,
        "total_tokens": 23,
    },
}

_OPENAI_OK_JSON = {
    **_OPENAI_OK,
    "choices": [
        {
            "message": {"role": "assistant", "content": '{"reply": "Hi there!"}'},
            "finish_reason": "stop",
        }
    ],
}

_ANTHROPIC_OK = {
    "id": "msg_abc",
    "model": "claude-3-5-haiku-20241022",
    "content": [{"type": "text", "text": "Hi there!"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 20, "output_tokens": 6},
}

_ANTHROPIC_OK_JSON = {
    **_ANTHROPIC_OK,
    "content": [{"type": "text", "text": '{"reply": "Hi there!"}'}],
}

_OPENAI_KEY = "sk-test-fake-key"
_ANTHROPIC_KEY = "ant-test-fake-key"


def _ollama_config(**kw: Any) -> LLMConfig:
    return LLMConfig(provider="ollama", model="llama3.2", **kw)


def _openai_config(**kw: Any) -> LLMConfig:
    return LLMConfig(provider="openai", model="gpt-4o-mini", **kw)


def _anthropic_config(**kw: Any) -> LLMConfig:
    return LLMConfig(provider="anthropic", model="claude-3-5-haiku-20241022", **kw)


# ──────────────────────────────────────────────────────────────────────
# OllamaProvider
# ──────────────────────────────────────────────────────────────────────


class TestOllamaProvider:
    async def test_happy_path_returns_correct_fields(self) -> None:
        transport = _MockTransport(200, _OLLAMA_OK)
        provider = OllamaProvider(_ollama_config(), _transport=transport)
        resp = await provider.complete(_BASE_REQUEST)
        assert isinstance(resp, LLMResponse)
        assert resp.text == "Hi there!"
        assert resp.model == "llama3.2"
        assert resp.provider == "ollama"
        assert resp.finish_reason == "stop"

    async def test_usage_parsed(self) -> None:
        transport = _MockTransport(200, _OLLAMA_OK)
        provider = OllamaProvider(_ollama_config(), _transport=transport)
        resp = await provider.complete(_BASE_REQUEST)
        assert resp.usage == {"prompt": 10, "completion": 5, "total": 15}

    async def test_usage_absent_when_not_in_response(self) -> None:
        body = dict(_OLLAMA_OK)
        body.pop("prompt_eval_count")
        body.pop("eval_count")
        transport = _MockTransport(200, body)
        provider = OllamaProvider(_ollama_config(), _transport=transport)
        resp = await provider.complete(_BASE_REQUEST)
        assert resp.usage is None

    async def test_json_mode_sets_format_field(self) -> None:
        recording = _RecordingTransport(200, _OLLAMA_OK_JSON)
        provider = OllamaProvider(_ollama_config(), _transport=recording)
        req = LLMRequest(system=None, messages=[{"role": "user", "content": "hi"}], json_mode=True)
        await provider.complete(req)
        assert recording.last_request_json is not None
        assert recording.last_request_json.get("format") == "json"

    async def test_json_mode_false_omits_format(self) -> None:
        recording = _RecordingTransport(200, _OLLAMA_OK)
        provider = OllamaProvider(_ollama_config(), _transport=recording)
        await provider.complete(_BASE_REQUEST)
        assert recording.last_request_json is not None
        assert "format" not in recording.last_request_json

    async def test_system_prepended_to_messages(self) -> None:
        recording = _RecordingTransport(200, _OLLAMA_OK)
        provider = OllamaProvider(_ollama_config(), _transport=recording)
        await provider.complete(_BASE_REQUEST)
        assert recording.last_request_json is not None
        msgs = recording.last_request_json["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    async def test_no_system_omits_system_message(self) -> None:
        recording = _RecordingTransport(200, _OLLAMA_OK)
        provider = OllamaProvider(_ollama_config(), _transport=recording)
        req = LLMRequest(system=None, messages=[{"role": "user", "content": "hi"}])
        await provider.complete(req)
        assert recording.last_request_json is not None
        msgs = recording.last_request_json["messages"]
        assert all(m["role"] != "system" for m in msgs)

    async def test_timeout_raises_llm_network_error(self) -> None:
        transport = _ErrorTransport(httpx.TimeoutException("timed out"))
        provider = OllamaProvider(_ollama_config(), _transport=transport)
        with pytest.raises(LLMNetworkError, match="Ollama"):
            await provider.complete(_BASE_REQUEST)

    async def test_connection_error_raises_llm_network_error(self) -> None:
        transport = _ErrorTransport(httpx.ConnectError("refused"))
        provider = OllamaProvider(_ollama_config(), _transport=transport)
        with pytest.raises(LLMNetworkError, match="Ollama"):
            await provider.complete(_BASE_REQUEST)

    async def test_network_error_message_includes_url(self) -> None:
        transport = _ErrorTransport(httpx.ConnectError("refused"))
        provider = OllamaProvider(_ollama_config(), _transport=transport)
        with pytest.raises(LLMNetworkError) as exc_info:
            await provider.complete(_BASE_REQUEST)
        assert "http://localhost:11434" in str(exc_info.value)

    async def test_custom_base_url_in_error_message(self) -> None:
        cfg = LLMConfig(provider="ollama", model="llama3.2", base_url="http://myhost:11434")
        transport = _ErrorTransport(httpx.ConnectError("refused"))
        provider = OllamaProvider(cfg, _transport=transport)
        with pytest.raises(LLMNetworkError) as exc_info:
            await provider.complete(_BASE_REQUEST)
        assert "myhost" in str(exc_info.value)

    async def test_http_429_raises_rate_limit_error(self) -> None:
        transport = _MockTransport(429, {"error": "rate limited"})
        provider = OllamaProvider(_ollama_config(), _transport=transport)
        with pytest.raises(LLMRateLimitError):
            await provider.complete(_BASE_REQUEST)

    async def test_http_500_raises_response_error(self) -> None:
        transport = _MockTransport(500, {"error": "server error"})
        provider = OllamaProvider(_ollama_config(), _transport=transport)
        with pytest.raises(LLMResponseError, match="500"):
            await provider.complete(_BASE_REQUEST)

    async def test_invalid_json_raises_response_error(self) -> None:
        transport = _RawTransport(200, b"not-json!!!")
        provider = OllamaProvider(_ollama_config(), _transport=transport)
        with pytest.raises(LLMResponseError, match="non-JSON"):
            await provider.complete(_BASE_REQUEST)

    async def test_provider_name(self) -> None:
        transport = _MockTransport(200, _OLLAMA_OK)
        p = OllamaProvider(_ollama_config(), _transport=transport)
        assert p.name == "ollama"

    async def test_concurrency_ten_calls_all_succeed(self) -> None:
        transport = _MockTransport(200, _OLLAMA_OK)
        provider = OllamaProvider(_ollama_config(), _transport=transport)
        req = LLMRequest(system=None, messages=[{"role": "user", "content": "hi"}])
        results = await asyncio.gather(*[provider.complete(req) for _ in range(10)])
        assert len(results) == 10
        assert all(r.text == "Hi there!" for r in results)


# ──────────────────────────────────────────────────────────────────────
# OpenAIProvider
# ──────────────────────────────────────────────────────────────────────


class TestOpenAIProvider:
    def _make(self, transport: httpx.AsyncBaseTransport) -> OpenAIProvider:
        with patch.dict(os.environ, {"OPENAI_API_KEY": _OPENAI_KEY}):
            return OpenAIProvider(_openai_config(), _transport=transport)

    def test_missing_api_key_raises_llm_error(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with (
            patch.dict(os.environ, env, clear=True),
            pytest.raises(LLMError, match="OPENAI_API_KEY"),
        ):
            OpenAIProvider(_openai_config())

    async def test_happy_path_returns_correct_fields(self) -> None:
        provider = self._make(_MockTransport(200, _OPENAI_OK))
        resp = await provider.complete(_BASE_REQUEST)
        assert resp.text == "Hi there!"
        assert resp.model == "gpt-4o-mini"
        assert resp.provider == "openai"
        assert resp.finish_reason == "stop"

    async def test_usage_parsed(self) -> None:
        provider = self._make(_MockTransport(200, _OPENAI_OK))
        resp = await provider.complete(_BASE_REQUEST)
        assert resp.usage == {"prompt": 15, "completion": 8, "total": 23}

    async def test_json_mode_sets_response_format(self) -> None:
        recording = _RecordingTransport(200, _OPENAI_OK_JSON)
        provider = self._make(recording)
        req = LLMRequest(system=None, messages=[{"role": "user", "content": "hi"}], json_mode=True)
        await provider.complete(req)
        assert recording.last_request_json is not None
        assert recording.last_request_json.get("response_format") == {"type": "json_object"}

    async def test_json_mode_false_omits_response_format(self) -> None:
        recording = _RecordingTransport(200, _OPENAI_OK)
        provider = self._make(recording)
        await provider.complete(_BASE_REQUEST)
        assert recording.last_request_json is not None
        assert "response_format" not in recording.last_request_json

    async def test_max_tokens_forwarded(self) -> None:
        recording = _RecordingTransport(200, _OPENAI_OK)
        provider = self._make(recording)
        req = LLMRequest(system=None, messages=[{"role": "user", "content": "hi"}], max_tokens=128)
        await provider.complete(req)
        assert recording.last_request_json is not None
        assert recording.last_request_json["max_tokens"] == 128

    async def test_timeout_raises_llm_network_error(self) -> None:
        provider = self._make(_ErrorTransport(httpx.TimeoutException("timed out")))
        with pytest.raises(LLMNetworkError, match="OpenAI"):
            await provider.complete(_BASE_REQUEST)

    async def test_http_429_raises_rate_limit_error(self) -> None:
        provider = self._make(_MockTransport(429, {"error": {"message": "rate limit"}}))
        with pytest.raises(LLMRateLimitError):
            await provider.complete(_BASE_REQUEST)

    async def test_http_500_raises_response_error(self) -> None:
        provider = self._make(_MockTransport(500, {"error": "oops"}))
        with pytest.raises(LLMResponseError, match="500"):
            await provider.complete(_BASE_REQUEST)

    async def test_no_choices_raises_response_error(self) -> None:
        body = dict(_OPENAI_OK)
        body["choices"] = []
        provider = self._make(_MockTransport(200, body))
        with pytest.raises(LLMResponseError, match="no choices"):
            await provider.complete(_BASE_REQUEST)

    async def test_invalid_json_raises_response_error(self) -> None:
        transport = _RawTransport(200, b"not json")
        provider = self._make(transport)
        with pytest.raises(LLMResponseError, match="non-JSON"):
            await provider.complete(_BASE_REQUEST)

    async def test_provider_name(self) -> None:
        provider = self._make(_MockTransport(200, _OPENAI_OK))
        assert provider.name == "openai"

    async def test_concurrency_ten_calls_all_succeed(self) -> None:
        provider = self._make(_MockTransport(200, _OPENAI_OK))
        req = LLMRequest(system=None, messages=[{"role": "user", "content": "hi"}])
        results = await asyncio.gather(*[provider.complete(req) for _ in range(10)])
        assert len(results) == 10
        assert all(r.provider == "openai" for r in results)


# ──────────────────────────────────────────────────────────────────────
# AnthropicProvider
# ──────────────────────────────────────────────────────────────────────


class TestAnthropicProvider:
    def _make(self, transport: httpx.AsyncBaseTransport) -> AnthropicProvider:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": _ANTHROPIC_KEY}):
            return AnthropicProvider(_anthropic_config(), _transport=transport)

    def test_missing_api_key_raises_llm_error(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with (
            patch.dict(os.environ, env, clear=True),
            pytest.raises(LLMError, match="ANTHROPIC_API_KEY"),
        ):
            AnthropicProvider(_anthropic_config())

    async def test_happy_path_returns_correct_fields(self) -> None:
        provider = self._make(_MockTransport(200, _ANTHROPIC_OK))
        resp = await provider.complete(_BASE_REQUEST)
        assert resp.text == "Hi there!"
        assert resp.model == "claude-3-5-haiku-20241022"
        assert resp.provider == "anthropic"
        # Anthropic 'end_turn' normalizes to common 'stop' (review-w5a MEDIUM).
        assert resp.finish_reason == "stop"

    async def test_usage_parsed(self) -> None:
        provider = self._make(_MockTransport(200, _ANTHROPIC_OK))
        resp = await provider.complete(_BASE_REQUEST)
        assert resp.usage == {"prompt": 20, "completion": 6, "total": 26}

    async def test_json_mode_injects_system_instruction(self) -> None:
        recording = _RecordingTransport(200, _ANTHROPIC_OK_JSON)
        provider = self._make(recording)
        req = LLMRequest(system=None, messages=[{"role": "user", "content": "hi"}], json_mode=True)
        await provider.complete(req)
        assert recording.last_request_json is not None
        system = recording.last_request_json.get("system", "")
        assert "JSON" in system

    async def test_json_mode_appends_to_existing_system(self) -> None:
        recording = _RecordingTransport(200, _ANTHROPIC_OK_JSON)
        provider = self._make(recording)
        req = LLMRequest(
            system="You are a coder.",
            messages=[{"role": "user", "content": "hi"}],
            json_mode=True,
        )
        await provider.complete(req)
        assert recording.last_request_json is not None
        system = recording.last_request_json.get("system", "")
        assert "coder" in system
        assert "JSON" in system

    async def test_json_mode_false_no_json_instruction(self) -> None:
        recording = _RecordingTransport(200, _ANTHROPIC_OK)
        provider = self._make(recording)
        req = LLMRequest(
            system="You are helpful.",
            messages=[{"role": "user", "content": "hi"}],
            json_mode=False,
        )
        await provider.complete(req)
        assert recording.last_request_json is not None
        system = recording.last_request_json.get("system", "")
        assert system == "You are helpful."

    async def test_default_max_tokens_set(self) -> None:
        recording = _RecordingTransport(200, _ANTHROPIC_OK)
        provider = self._make(recording)
        req = LLMRequest(system=None, messages=[{"role": "user", "content": "hi"}])
        await provider.complete(req)
        assert recording.last_request_json is not None
        assert recording.last_request_json["max_tokens"] == 4096

    async def test_explicit_max_tokens_forwarded(self) -> None:
        recording = _RecordingTransport(200, _ANTHROPIC_OK)
        provider = self._make(recording)
        req = LLMRequest(system=None, messages=[{"role": "user", "content": "hi"}], max_tokens=256)
        await provider.complete(req)
        assert recording.last_request_json is not None
        assert recording.last_request_json["max_tokens"] == 256

    async def test_timeout_raises_llm_network_error(self) -> None:
        provider = self._make(_ErrorTransport(httpx.TimeoutException("timed out")))
        with pytest.raises(LLMNetworkError, match="Anthropic"):
            await provider.complete(_BASE_REQUEST)

    async def test_http_429_raises_rate_limit_error(self) -> None:
        provider = self._make(_MockTransport(429, {"error": {"message": "rate limit"}}))
        with pytest.raises(LLMRateLimitError):
            await provider.complete(_BASE_REQUEST)

    async def test_http_401_raises_response_error(self) -> None:
        provider = self._make(_MockTransport(401, {"error": {"message": "unauthorized"}}))
        with pytest.raises(LLMResponseError, match="401"):
            await provider.complete(_BASE_REQUEST)

    async def test_invalid_json_raises_response_error(self) -> None:
        transport = _RawTransport(200, b"nope")
        provider = self._make(transport)
        with pytest.raises(LLMResponseError, match="non-JSON"):
            await provider.complete(_BASE_REQUEST)

    async def test_provider_name(self) -> None:
        provider = self._make(_MockTransport(200, _ANTHROPIC_OK))
        assert provider.name == "anthropic"

    async def test_concurrency_ten_calls_all_succeed(self) -> None:
        provider = self._make(_MockTransport(200, _ANTHROPIC_OK))
        req = LLMRequest(system=None, messages=[{"role": "user", "content": "hi"}])
        results = await asyncio.gather(*[provider.complete(req) for _ in range(10)])
        assert len(results) == 10
        assert all(r.provider == "anthropic" for r in results)


# ──────────────────────────────────────────────────────────────────────
# LiteLLMProvider
# ──────────────────────────────────────────────────────────────────────


class TestLiteLLMProvider:
    def test_import_error_raises_llm_error(self) -> None:
        """LLMError is raised at construction time when litellm is absent."""
        cfg = LLMConfig(provider="litellm", model="openai/gpt-4o-mini")

        import builtins

        real_import = builtins.__import__

        def _blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "litellm":
                raise ImportError("No module named 'litellm'")
            return real_import(name, *args, **kwargs)

        with (
            patch.object(builtins, "__import__", side_effect=_blocked_import),
            pytest.raises(LLMError, match="litellm"),
        ):
            LiteLLMProvider(cfg)

    def test_provider_name(self) -> None:
        assert LiteLLMProvider.name == "litellm"


# ──────────────────────────────────────────────────────────────────────
# make_provider factory
# ──────────────────────────────────────────────────────────────────────


class TestMakeProvider:
    def test_ollama_config_returns_ollama_provider(self) -> None:
        provider = make_provider(_ollama_config())
        assert isinstance(provider, OllamaProvider)

    def test_openai_config_returns_openai_provider(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": _OPENAI_KEY}):
            provider = make_provider(_openai_config())
        assert isinstance(provider, OpenAIProvider)

    def test_anthropic_config_returns_anthropic_provider(self) -> None:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": _ANTHROPIC_KEY}):
            provider = make_provider(_anthropic_config())
        assert isinstance(provider, AnthropicProvider)

    def test_litellm_config_raises_when_not_installed(self) -> None:
        """make_provider propagates the LLMError from LiteLLMProvider.__init__."""
        cfg = LLMConfig(provider="litellm", model="openai/gpt-4o-mini")

        import builtins

        real_import = builtins.__import__

        def _blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "litellm":
                raise ImportError("No module named 'litellm'")
            return real_import(name, *args, **kwargs)

        with (
            patch.object(builtins, "__import__", side_effect=_blocked_import),
            pytest.raises(LLMError, match="litellm"),
        ):
            make_provider(cfg)

    def test_default_config_is_ollama(self) -> None:
        """LLMConfig() defaults to provider='ollama'."""
        cfg = LLMConfig()
        assert cfg.provider == "ollama"
        provider = make_provider(cfg)
        assert isinstance(provider, OllamaProvider)


# ──────────────────────────────────────────────────────────────────────
# Default config: Ollama unreachable → helpful error
# ──────────────────────────────────────────────────────────────────────


class TestOllamaUnreachable:
    async def test_connection_refused_message_guides_user(self) -> None:
        """When Ollama is unreachable, the error message contains install guidance."""
        transport = _ErrorTransport(httpx.ConnectError("Connection refused"))
        provider = OllamaProvider(_ollama_config(), _transport=transport)
        with pytest.raises(LLMNetworkError) as exc_info:
            await provider.complete(_BASE_REQUEST)
        msg = str(exc_info.value)
        assert "ollama" in msg.lower() or "Ollama" in msg
        assert "ollama.com" in msg or "ollama serve" in msg
        assert "config.yaml" in msg

    async def test_timeout_message_guides_user(self) -> None:
        transport = _ErrorTransport(httpx.TimeoutException("timeout"))
        provider = OllamaProvider(_ollama_config(), _transport=transport)
        with pytest.raises(LLMNetworkError) as exc_info:
            await provider.complete(_BASE_REQUEST)
        msg = str(exc_info.value)
        assert "config.yaml" in msg


# ──────────────────────────────────────────────────────────────────────
# LLMConfig model validation
# ──────────────────────────────────────────────────────────────────────


class TestLLMConfig:
    def test_defaults(self) -> None:
        cfg = LLMConfig()
        assert cfg.provider == "ollama"
        assert cfg.model == "llama3.2"
        assert cfg.base_url is None
        assert cfg.timeout == 60.0

    def test_custom_values(self) -> None:
        cfg = LLMConfig(
            provider="openai",
            model="gpt-4o",
            base_url="https://my-proxy.example.com/v1",
            timeout=30.0,
        )
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-4o"
        assert cfg.base_url == "https://my-proxy.example.com/v1"
        assert cfg.timeout == 30.0

    def test_invalid_provider_raises(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LLMConfig(provider="unknown")  # type: ignore[arg-type]

    def test_extra_fields_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LLMConfig(**{"unknown_field": "value"})  # type: ignore[arg-type]

# uat-r5-5 pr-d noise
