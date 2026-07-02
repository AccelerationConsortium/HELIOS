"""Record-once / replay-forever LLM providers.

Capture real LLM responses a single time with ``RecordingProvider`` (wrapping any
real provider), persist the ``recordings`` dict, then develop and test entirely
offline and deterministically with ``ReplayProvider``. This decouples LLM feature
development from live API access after one capture.
"""
from __future__ import annotations

import hashlib
import json

from app.services.llm_gateway import LLMError, LLMMessage, LLMResponse

__all__ = ["RecordingProvider", "ReplayProvider", "recording_key"]


def recording_key(system: str, messages: list[LLMMessage], model: str | None) -> str:
    """Stable key for a completion request (system + messages + model)."""
    payload = {
        "system": system,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "model": model,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class RecordingProvider:
    """Wrap a provider; record each response into ``recordings`` by request key."""

    def __init__(self, *, inner: object, recordings: dict[str, str]) -> None:
        self._inner = inner
        self._recordings = recordings

    async def complete(
        self,
        *,
        messages: list[LLMMessage],
        system: str,
        model: str | None = None,
    ) -> LLMResponse:
        response = await self._inner.complete(messages=messages, system=system, model=model)
        self._recordings[recording_key(system, messages, model)] = response.content
        return response


class ReplayProvider:
    """Return previously recorded responses; no network. Raises on a cache miss."""

    def __init__(self, recordings: dict[str, str]) -> None:
        self._recordings = dict(recordings)

    async def complete(
        self,
        *,
        messages: list[LLMMessage],
        system: str,
        model: str | None = None,
    ) -> LLMResponse:
        key = recording_key(system, messages, model)
        if key not in self._recordings:
            raise LLMError("ReplayProvider: no recording for this request")
        return LLMResponse(content=self._recordings[key], model=model or "replay", usage={})
