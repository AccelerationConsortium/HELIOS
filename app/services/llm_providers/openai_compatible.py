"""OpenAI-compatible chat-completions provider.

One adapter for any vendor exposing the classic ``/chat/completions`` schema —
Moonshot / Kimi, DeepSeek, OpenAI (chat), and most self-hosted gateways — by
supplying ``base_url`` + ``model``. Implements the shared ``LLMProvider``
protocol from ``llm_gateway`` (async ``complete`` -> ``LLMResponse``).

The LLM is a pluggable tool: nothing about HELIOS's decisions depends on which
vendor sits behind this adapter.
"""
from __future__ import annotations

from typing import Any

import httpx

from app.services.llm_gateway import LLMError, LLMMessage, LLMResponse

__all__ = ["OpenAICompatibleProvider"]


class OpenAICompatibleProvider:
    """Call any OpenAI-compatible ``/chat/completions`` endpoint via httpx."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        default_model: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise LLMError("api_key is required for OpenAICompatibleProvider")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._transport = transport

    async def complete(
        self,
        *,
        messages: list[LLMMessage],
        system: str,
        model: str | None = None,
    ) -> LLMResponse:
        url = f"{self._base_url}/chat/completions"
        headers = {
            "authorization": f"Bearer {self._api_key}",
            "content-type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": model or self._default_model,
            "messages": [
                {"role": "system", "content": system},
                *({"role": m.role, "content": m.content} for m in messages),
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

        async with httpx.AsyncClient(
            timeout=self._timeout, transport=self._transport
        ) as client:
            try:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise LLMError(
                    f"LLM API error {exc.response.status_code}: {exc.response.text}"
                ) from exc
            except httpx.RequestError as exc:
                raise LLMError(f"LLM API request failed: {exc}") from exc

        data = resp.json()
        content = _extract_content(data)
        usage = data.get("usage", {}) or {}
        return LLMResponse(
            content=content,
            model=data.get("model", model or self._default_model),
            usage={
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
        )


def _extract_content(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        raise LLMError("LLM response contained no choices")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if not isinstance(content, str):
        raise LLMError("LLM response content was not a string")
    return content
