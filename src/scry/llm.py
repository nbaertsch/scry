"""LLM provider abstraction for scry (DESIGN.md §11, Wave 5a).

LiteLLM-style pattern mirroring ``embed.py``.  Supports Ollama (default,
local-first), OpenAI, Anthropic, and LiteLLM routing.

Secrets: API keys come from env vars (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``).
         Never store keys in ``.scry/config.yaml``.

Default provider is ``'ollama'``.  When unreachable, :class:`LLMNetworkError`
is raised with install guidance.  No remote API is called without explicit config.

Optional extras:
    ``openai`` / ``anthropic`` — raw httpx, no extra dep.
    ``litellm``                — ``pip install "scry-cli[litellm]"``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from scry.models import LLMConfig

__all__ = [
    "AnthropicProvider",
    "FinishReason",
    "LLMError",
    "LLMJSONModeError",
    "LLMNetworkError",
    "LLMProvider",
    "LLMRateLimitError",
    "LLMRequest",
    "LLMResponse",
    "LLMResponseError",
    "LiteLLMProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "make_provider",
]


# ─── Exceptions ────────────────────────────────────────────────────────


class LLMError(Exception):
    """Base exception for all LLM provider failures.

    Callers catch this (or a subclass) and decide whether to retry.
    Providers never silently fall back to an alternative backend.
    """


class LLMNetworkError(LLMError):
    """Network-level failure: connection refused, timeout, DNS error, etc."""


class LLMRateLimitError(LLMError):
    """HTTP 429 — upstream provider rate limit exceeded."""


class LLMResponseError(LLMError):
    """Unexpected or malformed response from the provider (4xx/5xx, bad JSON)."""


class LLMJSONModeError(LLMResponseError):
    """JSON mode was requested but the response is not valid JSON.

    Raised when ``LLMRequest.json_mode=True`` and the model returns
    non-JSON text (review-w5a BLOCKING fix: was previously unenforced
    for Anthropic; now enforced for ALL providers as defense-in-depth).
    """


# ─── Request / Response ────────────────────────────────────────────────


@dataclass
class LLMRequest:
    """Input to a single LLM chat completion (OpenAI-style messages format)."""

    system: str | None
    messages: list[dict[str, str]]  # [{"role": ..., "content": ...}, ...]
    temperature: float = 0.0
    max_tokens: int | None = None
    json_mode: bool = False
    """Force JSON-parseable output.  Each provider uses native mechanisms:
    Ollama → ``format="json"``; OpenAI → ``response_format``; Anthropic → system
    instruction.  The response is **always validated** with ``json.loads()`` when
    ``json_mode=True``; invalid JSON raises :class:`LLMJSONModeError`."""


# ─── Normalised finish_reason enum ─────────────────────────────────────

# Per review-w5a MEDIUM: provider-specific finish_reason values
# (``stop``, ``end_turn``, ``length``, ``max_tokens``, etc.) are
# normalised to a small common set so callers don't need provider
# logic.
FinishReason = str  # "stop" | "length" | "tool_use" | "content_filter" | "other" | None

_FINISH_REASON_MAP: dict[str, FinishReason] = {
    # OpenAI
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "content_filter",
    # Anthropic
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_use",
    # Ollama
    "load": "other",
    "unload": "other",
}


def _normalize_finish_reason(raw: str | None) -> FinishReason | None:
    """Map provider-specific finish_reason values to the common set.

    Unknown values map to ``"other"`` so callers always see one of:
    ``stop``, ``length``, ``tool_use``, ``content_filter``, ``other``,
    or ``None`` when the provider didn't report it.
    """
    if raw is None or raw == "":
        return None
    return _FINISH_REASON_MAP.get(raw, "other")


def _validate_json_mode(text: str, provider: str) -> None:
    """Raise :class:`LLMJSONModeError` when *text* is not valid JSON.

    Called by every provider's ``complete()`` whenever ``req.json_mode``
    is True.  The body is parsed with the stdlib :func:`json.loads`;
    any parse failure produces an actionable error including the
    leading ~200 chars of the offending response (review-w5a BLOCKING).
    """
    import json as _json

    try:
        _json.loads(text)
    except (ValueError, TypeError) as exc:
        raise LLMJSONModeError(
            f"{provider}: json_mode=True but response is not valid JSON "
            f"({exc}). Response head: {text[:200]!r}"
        ) from exc


@dataclass
class LLMResponse:
    """Output from a single LLM chat completion."""

    text: str
    model: str
    provider: str
    usage: dict[str, int] | None  # {prompt, completion, total}; None when not reported
    finish_reason: str | None


# ─── Protocol ─────────────────────────────────────────────────────────


class LLMProvider(Protocol):
    """Structural Protocol every LLM backend satisfies.

    Implementations are safe to use from multiple concurrent coroutines.
    """

    name: str

    async def complete(self, req: LLMRequest) -> LLMResponse:
        """Execute a chat completion.  Raises LLMError subclasses on failure."""
        ...


# ─── Shared helpers ────────────────────────────────────────────────────


def _build_messages(req: LLMRequest) -> list[dict[str, str]]:
    """Prepend the system message (if present) and return the full list."""
    msgs: list[dict[str, str]] = []
    if req.system:
        msgs.append({"role": "system", "content": req.system})
    msgs.extend(req.messages)
    return msgs


def _check_status(resp: httpx.Response, *, provider: str) -> None:
    """Raise the appropriate :class:`LLMError` subclass for non-2xx responses."""
    if resp.status_code == 429:
        raise LLMRateLimitError(
            f"{provider}: rate limit exceeded (HTTP 429). Back off and retry, or upgrade your plan."
        )
    if resp.status_code >= 400:
        raise LLMResponseError(f"{provider}: HTTP {resp.status_code}: {resp.text[:400]}")


# ─── OllamaProvider ────────────────────────────────────────────────────

_OLLAMA_DEFAULT_URL = "http://localhost:11434"
_OLLAMA_UNREACHABLE = (
    "Ollama is not reachable at {url}. "
    "Install Ollama (https://ollama.com) and run `ollama serve`, "
    "or configure `llm:` in .scry/config.yaml to use a different provider."
)


class OllamaProvider:
    """Local Ollama HTTP API — no API key, no cost.  Default local-first provider.

    ``_transport`` is for test injection only.
    """

    def __init__(
        self,
        config: LLMConfig,
        *,
        _transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.name: str = "ollama"
        self._base_url = config.base_url or _OLLAMA_DEFAULT_URL
        self._model = config.model
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            transport=_transport,
            timeout=config.timeout,
        )

    async def complete(self, req: LLMRequest) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": _build_messages(req),
            "stream": False,
            "options": {"temperature": req.temperature},
        }
        if req.max_tokens is not None:
            body["options"]["num_predict"] = req.max_tokens
        if req.json_mode:
            body["format"] = "json"

        try:
            resp = await self._client.post("/api/chat", json=body)
        except httpx.TimeoutException as exc:
            raise LLMNetworkError(_OLLAMA_UNREACHABLE.format(url=self._base_url)) from exc
        except httpx.NetworkError as exc:
            raise LLMNetworkError(_OLLAMA_UNREACHABLE.format(url=self._base_url)) from exc

        _check_status(resp, provider="ollama")

        try:
            data: dict[str, Any] = resp.json()
        except Exception as exc:
            raise LLMResponseError(f"Ollama returned non-JSON: {resp.text[:200]}") from exc

        text: str = data.get("message", {}).get("content", "")
        if req.json_mode:
            _validate_json_mode(text, "ollama")
        usage: dict[str, int] | None = None
        if "prompt_eval_count" in data or "eval_count" in data:
            prompt_toks = int(data.get("prompt_eval_count", 0))
            completion_toks = int(data.get("eval_count", 0))
            usage = {
                "prompt": prompt_toks,
                "completion": completion_toks,
                "total": prompt_toks + completion_toks,
            }
        return LLMResponse(
            text=text,
            model=self._model,
            provider="ollama",
            usage=usage,
            finish_reason=_normalize_finish_reason(data.get("done_reason")),
        )


# ─── OpenAIProvider ────────────────────────────────────────────────────

_OPENAI_DEFAULT_URL = "https://api.openai.com/v1"


class OpenAIProvider:
    """OpenAI Chat Completions REST API (raw httpx, no SDK).

    Reads ``OPENAI_API_KEY`` from env at construction — raises :class:`LLMError`
    if absent.  ``_transport`` is for test injection only.
    """

    def __init__(
        self,
        config: LLMConfig,
        *,
        _transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.name: str = "openai"
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise LLMError(
                "OPENAI_API_KEY environment variable is not set. "
                "Export it in your shell before using the OpenAI provider. "
                "Never store API keys in .scry/config.yaml."
            )
        base_url = config.base_url or _OPENAI_DEFAULT_URL
        self._model = config.model
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            transport=_transport,
            timeout=config.timeout,
        )

    async def complete(self, req: LLMRequest) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": _build_messages(req),
            "temperature": req.temperature,
        }
        if req.max_tokens is not None:
            body["max_tokens"] = req.max_tokens
        if req.json_mode:
            body["response_format"] = {"type": "json_object"}

        try:
            resp = await self._client.post("/chat/completions", json=body)
        except httpx.TimeoutException as exc:
            raise LLMNetworkError("OpenAI request timed out") from exc
        except httpx.NetworkError as exc:
            raise LLMNetworkError(f"OpenAI network error: {exc}") from exc

        _check_status(resp, provider="openai")

        try:
            data: dict[str, Any] = resp.json()
        except Exception as exc:
            raise LLMResponseError(f"OpenAI returned non-JSON: {resp.text[:200]}") from exc

        choices: list[Any] = data.get("choices", [])
        if not choices:
            raise LLMResponseError("OpenAI returned no choices in the response")
        choice: dict[str, Any] = choices[0]
        text: str = choice.get("message", {}).get("content", "") or ""
        if req.json_mode:
            _validate_json_mode(text, "openai")

        usage_raw: dict[str, Any] = data.get("usage") or {}
        usage: dict[str, int] | None = None
        if usage_raw:
            usage = {
                "prompt": int(usage_raw.get("prompt_tokens", 0)),
                "completion": int(usage_raw.get("completion_tokens", 0)),
                "total": int(usage_raw.get("total_tokens", 0)),
            }
        return LLMResponse(
            text=text,
            model=str(data.get("model", self._model)),
            provider="openai",
            usage=usage,
            finish_reason=_normalize_finish_reason(choice.get("finish_reason")),
        )


# ─── AnthropicProvider ────────────────────────────────────────────────

_ANTHROPIC_DEFAULT_URL = "https://api.anthropic.com"
_ANTHROPIC_API_VERSION = "2023-06-01"
_ANTHROPIC_DEFAULT_MAX_TOKENS = 4096
_ANTHROPIC_JSON_INSTRUCTION = (
    "Respond with valid JSON only. Do not include any explanation, "
    "markdown code fences, or text outside the JSON object."
)


class AnthropicProvider:
    """Anthropic Messages REST API (raw httpx, no SDK).

    Reads ``ANTHROPIC_API_KEY`` from env at construction — raises :class:`LLMError`
    if absent.  JSON mode is implemented via a system-prompt instruction.
    ``_transport`` is for test injection only.
    """

    def __init__(
        self,
        config: LLMConfig,
        *,
        _transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.name: str = "anthropic"
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Export it in your shell before using the Anthropic provider. "
                "Never store API keys in .scry/config.yaml."
            )
        base_url = config.base_url or _ANTHROPIC_DEFAULT_URL
        self._model = config.model
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "x-api-key": api_key,
                "anthropic-version": _ANTHROPIC_API_VERSION,
                "Content-Type": "application/json",
            },
            transport=_transport,
            timeout=config.timeout,
        )

    async def complete(self, req: LLMRequest) -> LLMResponse:
        # Anthropic does not accept "system" inside the messages list.
        system = req.system or ""
        if req.json_mode:
            system = (
                f"{system}\n\n{_ANTHROPIC_JSON_INSTRUCTION}".strip()
                if system
                else _ANTHROPIC_JSON_INSTRUCTION
            )

        body: dict[str, Any] = {
            "model": self._model,
            "messages": req.messages,  # Anthropic uses the same role/content shape
            "temperature": req.temperature,
            # Anthropic requires max_tokens; default to a generous value.
            "max_tokens": req.max_tokens
            if req.max_tokens is not None
            else _ANTHROPIC_DEFAULT_MAX_TOKENS,
        }
        if system:
            body["system"] = system

        try:
            resp = await self._client.post("/v1/messages", json=body)
        except httpx.TimeoutException as exc:
            raise LLMNetworkError("Anthropic request timed out") from exc
        except httpx.NetworkError as exc:
            raise LLMNetworkError(f"Anthropic network error: {exc}") from exc

        _check_status(resp, provider="anthropic")

        try:
            data: dict[str, Any] = resp.json()
        except Exception as exc:
            raise LLMResponseError(f"Anthropic returned non-JSON: {resp.text[:200]}") from exc

        # Anthropic content is a list of typed blocks; extract the first text block.
        text = ""
        for block in data.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                text = str(block.get("text", ""))
                break
        if req.json_mode:
            # Anthropic JSON mode is prompt-based (no native response_format),
            # so validation here is the ONLY enforcement (review-w5a BLOCKING fix).
            _validate_json_mode(text, "anthropic")

        usage_raw: dict[str, Any] = data.get("usage") or {}
        usage: dict[str, int] | None = None
        if usage_raw:
            prompt_toks = int(usage_raw.get("input_tokens", 0))
            completion_toks = int(usage_raw.get("output_tokens", 0))
            usage = {
                "prompt": prompt_toks,
                "completion": completion_toks,
                "total": prompt_toks + completion_toks,
            }
        return LLMResponse(
            text=text,
            model=str(data.get("model", self._model)),
            provider="anthropic",
            usage=usage,
            finish_reason=_normalize_finish_reason(data.get("stop_reason")),
        )


# ─── LiteLLMProvider ──────────────────────────────────────────────────


class LiteLLMProvider:
    """Routes via LiteLLM (``pip install "scry-cli[litellm]"``).

    ``config.model`` follows LiteLLM's ``<provider>/<model>`` naming.
    Raises :class:`LLMError` at construction if litellm is not installed.
    """

    name: str = "litellm"

    def __init__(self, config: LLMConfig) -> None:
        try:
            import litellm
        except ImportError as exc:
            raise LLMError(
                "litellm is not installed. "
                'Run `pip install "scry-cli[litellm]"` to enable LiteLLM routing.'
            ) from exc
        self._litellm: Any = litellm
        self._model = config.model
        self._timeout = config.timeout

    async def complete(self, req: LLMRequest) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": _build_messages(req),
            "temperature": req.temperature,
            "timeout": self._timeout,
        }
        if req.max_tokens is not None:
            kwargs["max_tokens"] = req.max_tokens
        if req.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            resp: Any = await self._litellm.acompletion(**kwargs)
        except Exception as exc:
            # LiteLLM raises its own exception hierarchy; normalize to LLMError.
            raise LLMError(f"LiteLLM error: {exc}") from exc

        # Wrap response parsing too — incompatible LiteLLM versions can raise
        # AttributeError/IndexError instead of returning a typed response
        # (review-w5a MEDIUM fix).
        try:
            text: str = str(resp.choices[0].message.content or "")
            raw_finish: Any = resp.choices[0].finish_reason
        except (AttributeError, IndexError, TypeError) as exc:
            raise LLMResponseError(
                f"LiteLLM response shape mismatch (incompatible version?): {exc}"
            ) from exc

        if req.json_mode:
            _validate_json_mode(text, "litellm")

        usage_raw: Any = resp.usage
        usage: dict[str, int] | None = None
        if usage_raw is not None:
            usage = {
                "prompt": int(getattr(usage_raw, "prompt_tokens", 0) or 0),
                "completion": int(getattr(usage_raw, "completion_tokens", 0) or 0),
                "total": int(getattr(usage_raw, "total_tokens", 0) or 0),
            }
        return LLMResponse(
            text=text,
            model=str(resp.model or self._model),
            provider="litellm",
            usage=usage,
            finish_reason=_normalize_finish_reason(str(raw_finish) if raw_finish else None),
        )


# ─── Factory ─────────────────────────────────────────────────────────


def make_provider(config: LLMConfig) -> LLMProvider:
    """Dispatch on ``config.provider`` and return a ready :class:`LLMProvider`.

    ``'ollama'`` (default) → :class:`OllamaProvider` — local, no API key.
    ``'openai'``           → :class:`OpenAIProvider` — reads ``OPENAI_API_KEY``.
    ``'anthropic'``        → :class:`AnthropicProvider` — reads ``ANTHROPIC_API_KEY``.
    ``'litellm'``          → :class:`LiteLLMProvider` — requires ``scry[litellm]``.

    API keys are read from env vars only — never from ``.scry/config.yaml``.
    """
    if config.provider == "ollama":
        return OllamaProvider(config)
    if config.provider == "openai":
        return OpenAIProvider(config)
    if config.provider == "anthropic":
        return AnthropicProvider(config)
    if config.provider == "litellm":
        return LiteLLMProvider(config)
    # Exhaustive: LLMConfig.provider is a Literal so mypy catches unknown values.
    raise LLMError(f"Unknown LLM provider: {config.provider!r}")  # pragma: no cover
