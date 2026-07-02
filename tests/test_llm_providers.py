from __future__ import annotations

import json

import httpx
import pytest

from app.services.llm_gateway import LLMError, LLMMessage
from app.services.llm_providers import (
    OpenAICompatibleProvider,
    RecordingProvider,
    ReplayProvider,
    build_openai_compatible,
)


def _ok_handler(captured: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "kimi-test",
                "choices": [{"message": {"role": "assistant", "content": "hello"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 5},
            },
        )

    return handler


async def test_openai_compatible_builds_request_and_parses_response():
    captured: dict = {}
    provider = OpenAICompatibleProvider(
        api_key="testkey",
        base_url="https://api.moonshot.ai/v1",
        default_model="kimi-test",
        transport=httpx.MockTransport(_ok_handler(captured)),
    )

    resp = await provider.complete(
        messages=[LLMMessage(role="user", content="propose points")],
        system="you are an optimizer",
    )

    assert resp.content == "hello"
    assert resp.model == "kimi-test"
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["auth"] == "Bearer testkey"
    assert captured["body"]["model"] == "kimi-test"
    assert captured["body"]["messages"][0] == {"role": "system", "content": "you are an optimizer"}
    assert captured["body"]["messages"][1] == {"role": "user", "content": "propose points"}


async def test_openai_compatible_raises_on_http_error():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    provider = OpenAICompatibleProvider(
        api_key="k",
        base_url="https://x/v1",
        default_model="m",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LLMError):
        await provider.complete(messages=[LLMMessage(role="user", content="q")], system="s")


def test_openai_compatible_requires_api_key():
    with pytest.raises(LLMError):
        OpenAICompatibleProvider(api_key="", base_url="https://x/v1", default_model="m")


def test_build_openai_compatible_presets():
    moonshot = build_openai_compatible("moonshot", api_key="k", model="kimi-test")
    assert moonshot._base_url == "https://api.moonshot.ai/v1"

    kimi = build_openai_compatible("kimi", api_key="k", model="kimi-test")
    assert kimi._base_url == "https://api.moonshot.ai/v1"

    deepseek = build_openai_compatible("deepseek", api_key="k", model="deepseek-chat")
    assert deepseek._base_url == "https://api.deepseek.com/v1"

    custom = build_openai_compatible(
        "custom", api_key="k", model="m", base_url="https://my.host/v1"
    )
    assert custom._base_url == "https://my.host/v1"


async def test_record_then_replay_round_trip():
    inner = _ScriptedProvider(["recorded-content"])
    recordings: dict[str, str] = {}
    recorder = RecordingProvider(inner=inner, recordings=recordings)

    first = await recorder.complete(
        messages=[LLMMessage(role="user", content="q")], system="s"
    )
    assert first.content == "recorded-content"
    assert recordings  # populated

    replay = ReplayProvider(recordings)
    again = await replay.complete(
        messages=[LLMMessage(role="user", content="q")], system="s"
    )
    assert again.content == "recorded-content"


async def test_replay_unknown_key_raises():
    replay = ReplayProvider({})
    with pytest.raises(LLMError):
        await replay.complete(messages=[LLMMessage(role="user", content="q")], system="s")


class _ScriptedProvider:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    async def complete(self, *, messages, system, model=None):
        from app.services.llm_gateway import LLMResponse

        return LLMResponse(content=self._responses.pop(0), model=model or "scripted", usage={})


async def test_adapter_drives_proposer_end_to_end():
    from app.services.candidate_gen import space_from_dimensions
    from app.services.llm_candidate_proposer import LLMCandidateProposer, validate_proposal

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "kimi-test",
                "choices": [
                    {"message": {"content": json.dumps(
                        {"proposals": [{"params": {"x": 0.3}, "reason": "explore"}]}
                    )}}
                ],
                "usage": {},
            },
        )

    provider = OpenAICompatibleProvider(
        api_key="k",
        base_url="https://api.moonshot.ai/v1",
        default_model="kimi-test",
        transport=httpx.MockTransport(handler),
    )
    space = space_from_dimensions(
        [{"param_name": "x", "param_type": "number", "min_value": 0.0, "max_value": 1.0}]
    )

    proposal = await LLMCandidateProposer(provider=provider).propose(
        campaign_id="c",
        round_index=0,
        space=space,
        objective_kpi="k",
        direction="maximize",
        trigger_reason="plateau",
    )
    validated = validate_proposal(proposal, space=space)

    assert proposal.points[0].params == {"x": 0.3}
    assert validated.accepted_points == [{"x": 0.3}]


def test_import_smoke():
    import app.services.llm_providers  # noqa: F401
