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
    "resolve_proposer_provider",
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


def resolve_proposer_provider(settings: object | None = None) -> OpenAICompatibleProvider | None:
    """Build the configured LLM proposer provider from settings, or None.

    Returns None (proposer fail-opens to the classical path) when no provider is
    configured, the vendor is unknown, or no API key is available. api_key falls
    back to ``llm_api_key``; ``llm_proposer_base_url`` overrides the preset.
    """
    if settings is None:
        from app.core.config import get_settings

        settings = get_settings()

    vendor = (getattr(settings, "llm_proposer_provider", "") or "").strip()
    if not vendor:
        return None
    api_key = (
        getattr(settings, "llm_proposer_api_key", "")
        or getattr(settings, "llm_api_key", "")
    )
    model = getattr(settings, "llm_proposer_model", "")
    base_url = getattr(settings, "llm_proposer_base_url", "") or None
    if not api_key or not model:
        return None
    if base_url is None and vendor not in OPENAI_COMPATIBLE_PRESETS:
        return None
    return build_openai_compatible(
        vendor, api_key=api_key, model=model, base_url=base_url
    )
