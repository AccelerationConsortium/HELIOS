"""Pluggable LLM provider adapters.

The LLM is a tool: HELIOS's decisions never depend on which vendor is behind
these adapters. All implement the ``LLMProvider`` protocol from ``llm_gateway``.

- ``OpenAICompatibleProvider`` — any ``/chat/completions`` vendor (Moonshot/Kimi,
  DeepSeek, OpenAI-chat, self-hosted) via base_url + model.
- ``RecordingProvider`` / ``ReplayProvider`` — capture once, replay offline.
- ``build_openai_compatible`` — vendor presets + custom base_url.
"""
from __future__ import annotations

from app.services.llm_providers.openai_compatible import OpenAICompatibleProvider
from app.services.llm_providers.replay import (
    RecordingProvider,
    ReplayProvider,
    recording_key,
)

__all__ = [
    "OPENAI_COMPATIBLE_PRESETS",
    "OpenAICompatibleProvider",
    "RecordingProvider",
    "ReplayProvider",
    "build_openai_compatible",
    "recording_key",
]

#: Base URLs for known OpenAI-compatible vendors.
OPENAI_COMPATIBLE_PRESETS: dict[str, str] = {
    "moonshot": "https://api.moonshot.ai/v1",
    "kimi": "https://api.moonshot.ai/v1",
    "moonshot_cn": "https://api.moonshot.cn/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "openai": "https://api.openai.com/v1",
}


def build_openai_compatible(
    vendor: str,
    *,
    api_key: str,
    model: str,
    base_url: str | None = None,
    **kwargs: object,
) -> OpenAICompatibleProvider:
    """Build an OpenAI-compatible provider from a vendor preset or custom URL."""
    resolved = base_url or OPENAI_COMPATIBLE_PRESETS.get(vendor)
    if resolved is None:
        raise ValueError(
            f"Unknown vendor '{vendor}'; pass base_url or use one of "
            f"{sorted(OPENAI_COMPATIBLE_PRESETS)}"
        )
    return OpenAICompatibleProvider(
        api_key=api_key, base_url=resolved, default_model=model, **kwargs
    )
