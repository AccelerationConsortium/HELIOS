from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel


class RuntimeInput(BaseModel):
    round_number: int
    value: str = "ok"


class RuntimeOutput(BaseModel):
    saw_context: bool
    prompt_block: str = ""
    value: str = ""


class RuntimeAgent:
    name = "runtime_agent"
    layer = "test"
    capabilities: ClassVar[list] = []

    def __init__(self) -> None:
        from app.agents.base import AgentCapability, BaseAgent

        class _Agent(BaseAgent[RuntimeInput, RuntimeOutput]):
            name = "runtime_agent"
            layer = "test"
            capabilities: ClassVar[list] = [
                AgentCapability("runtime.test", "Runtime integration test agent")
            ]

            def validate_input(self, input_data: RuntimeInput) -> list[str]:
                return []

            async def process(self, input_data: RuntimeInput) -> RuntimeOutput:
                from app.services.agent_context import get_current_agent_context

                ctx = get_current_agent_context()
                return RuntimeOutput(
                    saw_context=ctx is not None,
                    prompt_block=ctx.prompt_block if ctx else "",
                    value=input_data.value,
                )

        self.agent = _Agent()


async def test_control_plane_injects_runtime_context_from_knowledge_bus():
    from app.agents.control_plane import ControlPlane
    from app.services.knowledge_bus import KnowledgeEvent, get_bus

    campaign_id = "camp-runtime-test"
    bus = get_bus(campaign_id)
    bus.publish(
        KnowledgeEvent(
            source_agent="sensor",
            key="instrument.ot2.tip_failure_rate",
            delta_text="Tip failure rate increased.",
            confidence=0.9,
            round_id="2",
        )
    )

    cp = ControlPlane()
    cp.set_campaign_id(campaign_id)
    cp.register(RuntimeAgent().agent)

    result = await cp.call(
        "runtime_agent",
        RuntimeInput(round_number=2),
        caller="test",
    )

    assert result.success
    assert result.output is not None
    assert result.output.saw_context is True
    assert "Tip failure rate increased" in result.output.prompt_block


async def test_llm_gateway_appends_current_agent_context_to_system_prompt():
    from app.services.agent_context import (
        AgentRuntimeContext,
        reset_current_agent_context,
        set_current_agent_context,
    )
    from app.services.llm_gateway import LLMMessage, MockProvider

    token = set_current_agent_context(
        AgentRuntimeContext(
            campaign_id="camp-ctx",
            agent_name="planner_agent",
            caller="test",
            trace_id="trace",
            round_number=1,
            prompt_block="## Soft Knowledge Updates\n- prior observation",
        )
    )
    try:
        provider = MockProvider(responses=["done"])
        await provider.complete(
            messages=[LLMMessage(role="user", content="plan")],
            system="You are a planner.",
        )
        assert provider.last_call is not None
        assert "prior observation" in provider.last_call["system"]
    finally:
        reset_current_agent_context(token)


async def test_dynamic_swarm_uses_control_plane_when_available():
    from app.agents.control_plane import ControlPlane
    from app.agents.swarm import DynamicSwarm, SwarmContext

    cp = ControlPlane()
    cp.set_campaign_id("camp-swarm-test")
    cp.register(RuntimeAgent().agent)

    context = SwarmContext(
        campaign_id="camp-swarm-test",
        round_number=3,
        extra={"control_plane": cp},
    )
    swarm = DynamicSwarm.for_task(["runtime"], context, cp)
    result = await swarm.run(input_data=RuntimeInput(round_number=3))

    assert result.success
    assert "runtime_agent" in result.aggregated_output["selected_agents"]


def test_mcp_manifest_exposes_tools_resources_and_prompts():
    from app.services.mcp_manifest import build_mcp_manifest

    manifest = build_mcp_manifest()

    assert manifest["protocol"] == "mcp-style-manifest"
    assert isinstance(manifest["tools"], list)
    assert isinstance(manifest["resources"], list)
    assert manifest["prompts"][0]["name"] == "campaign_context"
    assert any(resource["uri"].startswith("helios://agents/") for resource in manifest["resources"])
