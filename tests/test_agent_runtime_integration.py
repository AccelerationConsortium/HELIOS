from __future__ import annotations

import uuid
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
    from app.core.db import init_db
    from app.services.campaign_events import replay_events
    from app.services.campaign_state import create_campaign
    from app.services.knowledge_bus import KnowledgeEvent, get_bus

    init_db()
    campaign_id = f"camp-runtime-test-{uuid.uuid4().hex[:8]}"
    create_campaign(campaign_id, {"contract_id": "contract"}, direction="maximize")
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
    events = replay_events(campaign_id)
    assert any(ev["event_type"] == "agent_context_snapshot" for ev in events)
    assert any(ev["event_type"] == "agent_trace_span" for ev in events)


async def test_context_read_does_not_create_empty_blackboard():
    from app.agents.blackboard import manager
    from app.services.agent_context import build_agent_context

    campaign_id = "camp-no-empty-blackboard"
    await build_agent_context(
        campaign_id=campaign_id,
        agent_name="runtime_agent",
        caller="test",
        trace_id="trace",
        input_data=RuntimeInput(round_number=99),
    )

    assert manager.get_existing("99", campaign_id) is None


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
    assert manifest["version"] == "0.2"
    assert isinstance(manifest["tools"], list)
    assert isinstance(manifest["resources"], list)
    assert manifest["transports"][0]["type"] == "http"
    assert manifest["resourceTemplates"][0]["uriTemplate"].startswith("helios://campaigns/")
    assert manifest["prompts"][0]["name"] == "campaign_context"
    assert any(resource["uri"].startswith("helios://agents/") for resource in manifest["resources"])


async def test_stage_runner_executes_control_plane_calls_as_stage():
    from app.agents.control_plane import ControlPlane
    from app.agents.stage import AgentStageCall, AgentStageRunner

    cp = ControlPlane()
    cp.set_campaign_id("camp-stage-test")
    cp.register(RuntimeAgent().agent)

    runner = AgentStageRunner(cp)
    emitted: list[dict] = []
    result = await runner.run_parallel(
        "runtime",
        [AgentStageCall("runtime_agent", RuntimeInput(round_number=1))],
        emit=emitted.append,
    )

    assert result.success
    assert result.outputs["runtime_agent"]["saw_context"] is True
    assert [event["type"] for event in emitted] == [
        "agent_stage_start",
        "agent_stage_end",
    ]


async def test_orchestrator_plan_only_emits_stage_graph_events():
    from app.agents.orchestrator import OrchestratorAgent, OrchestratorInput
    from app.core.db import init_db
    from app.services.campaign_events import replay_events

    init_db()
    campaign_id = f"camp-stage-graph-{uuid.uuid4().hex[:8]}"
    orchestrator = OrchestratorAgent()

    result = await orchestrator.process(
        OrchestratorInput(
            contract_id="contract",
            objective_kpi="yield",
            direction="maximize",
            max_rounds=1,
            batch_size=1,
            dimensions=[
                {"name": "temperature", "type": "float", "min": 20, "max": 80},
            ],
            protocol_template={"steps": []},
            campaign_id=campaign_id,
            plan_only=True,
        )
    )

    assert result.status == "planned"
    events = replay_events(campaign_id)
    event_types = [event["event_type"] for event in events]
    assert "agent_stage_graph" in event_types
    assert "agent_stage_start" in event_types
    assert "agent_stage_end" in event_types
    graph_payload = next(
        event["payload"]["graph"]
        for event in events
        if event["event_type"] == "agent_stage_graph"
    )
    assert [node["name"] for node in graph_payload["nodes"]] == [
        "planning",
        "design",
        "compile",
        "safety",
        "analyze",
    ]


async def test_durable_backend_status_cancel_and_events():
    from app.agents.orchestrator import OrchestratorInput
    from app.services.durable_execution import InProcessDurableBackend

    backend = InProcessDurableBackend()
    campaign_id = "camp-durable-cancel"
    orch_input = OrchestratorInput(
        contract_id="contract",
        objective_kpi="yield",
        direction="maximize",
        max_rounds=1,
        batch_size=1,
        dimensions=[],
        protocol_template={"steps": []},
        campaign_id=campaign_id,
    )

    await backend.start_campaign(orch_input)
    status = await backend.get_status(campaign_id)
    assert status is not None
    assert status.status == "running"

    cancelled = await backend.cancel_campaign(campaign_id)
    assert cancelled is not None
    assert cancelled.status in {"cancelling", "cancelled"}
    events = await backend.list_events(campaign_id)
    event_types = [event.type for event in events]
    assert event_types[0] == "durable_run_started"
    assert "durable_run_cancel_requested" in event_types


async def test_orchestrate_api_uses_durable_backend_contract():
    from app.api.v1.endpoints import orchestrate
    from app.services.durable_execution import (
        DurableRunEvent,
        DurableRunHandle,
        DurableRunStatus,
        get_durable_backend,
        set_durable_backend,
    )

    class FakeBackend:
        name = "fake"

        def __init__(self) -> None:
            self.campaign_id = ""
            self.cancelled = False

        async def start_campaign(self, input_data, **kwargs):
            self.campaign_id = input_data.campaign_id
            return DurableRunHandle(self.campaign_id, self.name, "running")

        async def get_status(self, campaign_id):
            if campaign_id != self.campaign_id:
                return None
            return DurableRunStatus(
                campaign_id=campaign_id,
                backend=self.name,
                status="cancelled" if self.cancelled else "running",
            )

        async def get_result(self, campaign_id):
            return None

        async def cancel_campaign(self, campaign_id):
            if campaign_id != self.campaign_id:
                return None
            self.cancelled = True
            return await self.get_status(campaign_id)

        async def list_events(self, campaign_id):
            return [
                DurableRunEvent(
                    campaign_id=campaign_id,
                    type="durable_run_started",
                    payload={"backend": self.name},
                    created_at="2026-01-01T00:00:00+00:00",
                )
            ]

    previous = get_durable_backend()
    fake = FakeBackend()
    set_durable_backend(fake)
    try:
        response = await orchestrate.orchestrate_start(
            orchestrate.OrchestrateRequest(
                contract_id="contract",
                objective_kpi="yield",
                direction="maximize",
            )
        )
        status = await orchestrate.orchestrate_status(response.campaign_id)
        stopped = await orchestrate.orchestrate_stop(response.campaign_id)
        events = await orchestrate.orchestrate_durable_events(response.campaign_id)

        assert status.status == "running"
        assert stopped["status"] == "cancelled"
        assert events["backend"] == "fake"
        assert events["events"][0]["type"] == "durable_run_started"
    finally:
        set_durable_backend(previous)


async def test_orchestrate_http_lifecycle_start_status_events_stop_resume():
    import httpx

    from app.main import app
    from app.services.campaign_state import create_campaign
    from app.services.durable_execution import (
        DurableRunEvent,
        DurableRunHandle,
        DurableRunStatus,
        get_durable_backend,
        set_durable_backend,
    )

    class FakeBackend:
        name = "fake_http"

        def __init__(self) -> None:
            self.campaign_id = ""
            self.status = "running"
            self.starts = 0

        async def start_campaign(self, input_data, **kwargs):
            self.campaign_id = input_data.campaign_id
            self.status = "running"
            self.starts += 1
            return DurableRunHandle(self.campaign_id, self.name, "running")

        async def get_status(self, campaign_id):
            if campaign_id != self.campaign_id:
                return None
            return DurableRunStatus(
                campaign_id=campaign_id,
                backend=self.name,
                status=self.status,
                result={"starts": self.starts},
            )

        async def get_result(self, campaign_id):
            return None

        async def cancel_campaign(self, campaign_id):
            if campaign_id != self.campaign_id:
                return None
            self.status = "cancelled"
            return await self.get_status(campaign_id)

        async def list_events(self, campaign_id):
            return [
                DurableRunEvent(
                    campaign_id=campaign_id,
                    type="durable_run_started",
                    payload={"backend": self.name, "starts": self.starts},
                    created_at="2026-01-01T00:00:00+00:00",
                )
            ]

    previous = get_durable_backend()
    fake = FakeBackend()
    set_durable_backend(fake)
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            start_response = await client.post(
                "/api/v1/orchestrate/start",
                json={
                    "contract_id": "contract",
                    "objective_kpi": "yield",
                    "direction": "maximize",
                    "max_rounds": 1,
                    "batch_size": 1,
                    "dimensions": [
                        {"name": "temperature", "type": "float", "min": 20, "max": 80}
                    ],
                    "protocol_template": {"steps": []},
                },
            )
            assert start_response.status_code == 200
            campaign_id = start_response.json()["campaign_id"]

            status_response = await client.get(f"/api/v1/orchestrate/{campaign_id}/status")
            events_response = await client.get(
                f"/api/v1/orchestrate/{campaign_id}/durable-events"
            )
            stop_response = await client.post(f"/api/v1/orchestrate/{campaign_id}/stop")

            create_campaign(
                campaign_id,
                {
                    "contract_id": "contract",
                    "objective_kpi": "yield",
                    "direction": "maximize",
                    "max_rounds": 1,
                    "batch_size": 1,
                    "strategy": "lhs",
                    "target_value": None,
                    "dimensions": [
                        {"name": "temperature", "type": "float", "min": 20, "max": 80}
                    ],
                    "protocol_template": {"steps": []},
                    "policy_snapshot": {},
                    "protocol_pattern_id": "",
                    "dry_run": False,
                    "plan_only": False,
                },
                direction="maximize",
            )
            resume_response = await client.post(
                f"/api/v1/orchestrate/{campaign_id}/resume"
            )

        assert status_response.status_code == 200
        assert status_response.json()["status"] == "running"
        assert events_response.status_code == 200
        assert events_response.json()["events"][0]["type"] == "durable_run_started"
        assert stop_response.status_code == 200
        assert stop_response.json()["status"] == "cancelled"
        assert resume_response.status_code == 200
        assert resume_response.json()["status"] == "resuming"
        assert fake.starts == 2
    finally:
        set_durable_backend(previous)
