"""LLM provider abstraction layer.

Provides a protocol-based abstraction over LLM APIs so the planner can
be tested with a ``MockProvider`` and run in production with the
``AnthropicProvider``.  Uses httpx directly — zero additional dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Protocol, runtime_checkable

import httpx

from app.core.config import get_settings


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMMessage:
    """A single message in a conversation."""

    role: str  # "user" | "assistant"
    content: str


@dataclass(frozen=True)
class LLMResponse:
    """Result of an LLM completion call."""

    content: str  # raw text from the model
    model: str
    usage: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Raised when an LLM call fails (network, auth, rate-limit, etc.)."""


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMProvider(Protocol):
    async def complete(
        self,
        *,
        messages: list[LLMMessage],
        system: str,
        model: str | None = None,
    ) -> LLMResponse: ...


# ---------------------------------------------------------------------------
# Anthropic provider (httpx)
# ---------------------------------------------------------------------------


class AnthropicProvider:
    """Calls the Anthropic Messages API via httpx."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        default_model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
    ) -> None:
        if not api_key:
            raise LLMError("LLM_API_KEY is required for AnthropicProvider")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._max_tokens = max_tokens

    async def complete(
        self,
        *,
        messages: list[LLMMessage],
        system: str,
        model: str | None = None,
    ) -> LLMResponse:
        url = f"{self._base_url}/v1/messages"
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body: dict[str, Any] = {
            "model": model or self._default_model,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                resp = await client.post(url, json=body, headers=headers)
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise LLMError(
                    f"Anthropic API error {exc.response.status_code}: {exc.response.text}"
                ) from exc
            except httpx.RequestError as exc:
                raise LLMError(f"Anthropic API request failed: {exc}") from exc

        data = resp.json()
        content_blocks = data.get("content", [])
        text = "".join(
            block.get("text", "") for block in content_blocks if block.get("type") == "text"
        )
        return LLMResponse(
            content=text,
            model=data.get("model", model or self._default_model),
            usage=data.get("usage", {}),
        )


# ---------------------------------------------------------------------------
# Mock provider (for tests)
# ---------------------------------------------------------------------------


class MockProvider:
    """Returns pre-set responses in order.  No network calls."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses: list[str] = list(responses or [])
        self._calls: list[dict[str, Any]] = []

    async def complete(
        self,
        *,
        messages: list[LLMMessage],
        system: str,
        model: str | None = None,
    ) -> LLMResponse:
        self._calls.append({"messages": messages, "system": system, "model": model})
        if not self._responses:
            raise LLMError("MockProvider: no more preset responses")
        return LLMResponse(
            content=self._responses.pop(0),
            model=model or "mock-model",
            usage={"input_tokens": 0, "output_tokens": 0},
        )

    @property
    def call_count(self) -> int:
        return len(self._calls)

    @property
    def last_call(self) -> dict[str, Any] | None:
        return self._calls[-1] if self._calls else None


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider:
    """Return the configured LLM provider (singleton)."""
    settings = get_settings()
    if settings.llm_provider == "anthropic":
        return AnthropicProvider(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            default_model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
        )
    # Default: mock provider (safe for dev/test)
    return MockProvider(responses=[])
