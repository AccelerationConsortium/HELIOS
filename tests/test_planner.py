"""Tests for the LLM planning pipeline (Phase C: LLM Reasoning Engine).

Covers:
- LLM gateway (MockProvider)
- Planner (system prompt, parse, plan_from_intent)
- Plan grounding (PlanResult → protocol JSON)
- Plan validator (static analysis)
- Agent endpoint (E2E via httpx AsyncClient)
"""
from __future__ import annotations

import json
import os
import tempfile

# Isolate test DB before any app imports.
_tmpdir = tempfile.mkdtemp(prefix="otbot_planner_test_")
os.environ["DATA_DIR"] = _tmpdir
os.environ["DB_PATH"] = os.path.join(_tmpdir, "planner_test.db")
os.environ["OBJECT_STORE_DIR"] = os.path.join(_tmpdir, "obj")
os.environ["LLM_PROVIDER"] = "mock"

import pytest  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.core.db import init_db  # noqa: E402
from app.services.llm_gateway import LLMError, MockProvider  # noqa: E402
from app.services.plan_grounding import GroundingResult, ground_plan  # noqa: E402
from app.services.plan_validator import validate_plan  # noqa: E402
from app.services.planner import (  # noqa: E402
    PlanParseError,
    PlanResult,
    PlanStep,
    build_system_prompt,
    parse_plan_response,
    plan_from_intent,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _setup_db():
    """Initialize a fresh DB for each test."""
    get_settings.cache_clear()
    init_db()


# ---------------------------------------------------------------------------
# Helper: sample LLM JSON response
# ---------------------------------------------------------------------------

SAMPLE_PLAN_JSON = json.dumps({
    "reasoning": "Home the robot first, then aspirate.",
    "steps": [
        {"id": "s1", "primitive": "robot.home", "params": {}, "depends_on": []},
        {
            "id": "s2",
            "primitive": "robot.aspirate",
            "params": {"volume_ul": 100, "well": "A1"},
            "depends_on": ["s1"],
        },
    ],
})

SIMPLE_PLAN_JSON = json.dumps({
    "steps": [
        {"id": "s1", "primitive": "robot.home", "params": {}, "depends_on": []},
        {"id": "s2", "primitive": "heat", "params": {"temp_c": 60}, "depends_on": ["s1"]},
    ],
})


# ===========================================================================
# 1. LLM Gateway — MockProvider
# ===========================================================================


@pytest.mark.anyio
async def test_mock_provider_returns_preset():
    """MockProvider returns responses in order."""
    provider = MockProvider(responses=["hello", "world"])
    from app.services.llm_gateway import LLMMessage

    r1 = await provider.complete(
        messages=[LLMMessage(role="user", content="test")],
        system="sys",
    )
    assert r1.content == "hello"
    assert r1.model == "mock-model"

    r2 = await provider.complete(
        messages=[LLMMessage(role="user", content="test2")],
        system="sys",
    )
    assert r2.content == "world"
    assert provider.call_count == 2


@pytest.mark.anyio
async def test_mock_provider_empty_raises():
    """MockProvider raises LLMError when no responses left."""
    provider = MockProvider(responses=[])
    from app.services.llm_gateway import LLMMessage

    with pytest.raises(LLMError, match="no more preset responses"):
        await provider.complete(
            messages=[LLMMessage(role="user", content="test")],
            system="sys",
        )


# ===========================================================================
# 2. Planner — system prompt + parse + plan_from_intent
# ===========================================================================


def test_build_system_prompt_contains_capabilities():
    """System prompt should include primitives from the registry."""
    prompt = build_system_prompt()
    # Should contain at least some known primitives
    assert "robot.home" in prompt or "aspirate" in prompt
    # Should contain output format instructions
    assert "steps" in prompt
    assert "primitive" in prompt


def test_parse_plan_response_valid_json():
    """Parse a valid JSON plan response."""
    result = parse_plan_response(SAMPLE_PLAN_JSON, model="test-model")
    assert isinstance(result, PlanResult)
    assert len(result.steps) == 2
    assert result.steps[0].id == "s1"
    assert result.steps[0].primitive == "robot.home"
    assert result.steps[1].id == "s2"
    assert result.steps[1].primitive == "robot.aspirate"
    assert result.steps[1].depends_on == ["s1"]
    assert result.reasoning == "Home the robot first, then aspirate."
    assert result.model == "test-model"


def test_parse_plan_response_code_block():
    """Parse JSON wrapped in a ```json code block."""
    wrapped = f"Here is the plan:\n```json\n{SAMPLE_PLAN_JSON}\n```\nDone."
    result = parse_plan_response(wrapped)
    assert len(result.steps) == 2
    assert result.steps[0].primitive == "robot.home"


def test_parse_plan_response_invalid_json():
    """Invalid JSON raises PlanParseError."""
    with pytest.raises(PlanParseError, match="No JSON object found"):
        parse_plan_response("not json at all {broken")


def test_parse_plan_response_no_steps():
    """Missing 'steps' array raises PlanParseError."""
    with pytest.raises(PlanParseError, match="non-empty 'steps'"):
        parse_plan_response('{"reasoning": "no steps here"}')


def test_parse_plan_response_empty_steps():
    """Empty 'steps' array raises PlanParseError."""
    with pytest.raises(PlanParseError, match="non-empty 'steps'"):
        parse_plan_response('{"steps": []}')


def test_parse_plan_response_missing_primitive():
    """Step without 'primitive' raises PlanParseError."""
    bad = json.dumps({"steps": [{"id": "s1", "params": {}}]})
    with pytest.raises(PlanParseError, match="primitive.*required"):
        parse_plan_response(bad)


@pytest.mark.anyio
async def test_plan_from_intent_end_to_end():
    """Full flow: MockProvider → plan_from_intent → PlanResult."""
    provider = MockProvider(responses=[SAMPLE_PLAN_JSON])
    result = await plan_from_intent("Home robot and aspirate 100uL", provider=provider)

    assert isinstance(result, PlanResult)
    assert len(result.steps) == 2
    assert result.steps[0].primitive == "robot.home"
    assert provider.call_count == 1

    # Verify the system prompt was passed
    assert provider.last_call is not None
    assert "system" in provider.last_call
    assert len(provider.last_call["system"]) > 100  # non-trivial prompt


# ===========================================================================
# 3. Plan Grounding
# ===========================================================================


def test_ground_valid_plan():
    """Ground a plan with known primitives → valid protocol JSON."""
    plan = PlanResult(
        steps=[
            PlanStep(id="s1", primitive="robot.home", params={}),
            PlanStep(id="s2", primitive="heat", params={"temp_c": 60}, depends_on=["s1"]),
        ],
        raw_response="test",
        model="test",
    )
    result = ground_plan(plan)
    assert result.ok, f"Grounding errors: {result.errors}"
    assert "steps" in result.protocol
    assert len(result.protocol["steps"]) == 2
    assert result.protocol["steps"][0]["primitive"] == "robot.home"
    assert result.protocol["steps"][1]["depends_on"] == ["s1"]


def test_ground_unknown_primitive():
    """Unknown primitive → error in grounding result."""
    plan = PlanResult(
        steps=[PlanStep(id="s1", primitive="quantum.teleport", params={})],
        raw_response="test",
        model="test",
    )
    result = ground_plan(plan)
    assert not result.ok
    assert any("unknown primitive" in e for e in result.errors)


def test_ground_type_coercion():
    """String numbers should be coerced to numeric types."""
    # plc.dispense_ml has pump (integer) and volume_ml (number)
    plan = PlanResult(
        steps=[PlanStep(id="s1", primitive="plc.dispense_ml", params={
            "pump": "1",        # string → integer coercion
            "volume_ml": "5.0",  # string → number coercion
        })],
        raw_response="test",
        model="test",
    )
    result = ground_plan(plan)
    assert result.ok, f"Grounding errors: {result.errors}"
    # Check that coercion warnings were generated
    coercion_warnings = [w for w in result.warnings if "coerced" in w.lower()]
    assert len(coercion_warnings) >= 1
    # Verify values were actually coerced
    step = result.protocol["steps"][0]
    assert step["params"]["pump"] == 1
    assert step["params"]["volume_ml"] == 5.0


def test_ground_auto_resource_mapping():
    """Resources should be auto-mapped from the registry."""
    plan = PlanResult(
        steps=[PlanStep(id="s1", primitive="robot.home", params={})],
        raw_response="test",
        model="test",
    )
    result = ground_plan(plan)
    assert result.ok
    step = result.protocol["steps"][0]
    # robot.home should have a resource_id from the registry
    # (if the skill file specifies one)
    assert isinstance(step["resources"], list)


# ===========================================================================
# 4. Plan Validator
# ===========================================================================


def test_validate_empty_protocol():
    """Empty protocol produces a warning."""
    result = validate_plan({"steps": []})
    assert any("no steps" in w for w in result.warnings)


def test_validate_unreachable_step():
    """Step depending on non-existent step produces warning."""
    protocol = {
        "steps": [
            {"id": "s1", "primitive": "aspirate", "params": {}, "depends_on": ["nonexistent"]},
        ]
    }
    result = validate_plan(protocol)
    assert any("nonexistent" in w for w in result.warnings)


def test_validate_no_root():
    """All steps having dependencies (no root) produces warning."""
    protocol = {
        "steps": [
            {"id": "s1", "primitive": "aspirate", "params": {}, "depends_on": ["s2"]},
            {"id": "s2", "primitive": "heat", "params": {}, "depends_on": ["s1"]},
        ]
    }
    result = validate_plan(protocol)
    assert any("no root" in w for w in result.warnings)


def test_validate_redundant_operations():
    """Consecutive identical primitives produce warning."""
    protocol = {
        "steps": [
            {"id": "s1", "primitive": "aspirate", "params": {}, "depends_on": []},
            {"id": "s2", "primitive": "aspirate", "params": {}, "depends_on": ["s1"]},
        ]
    }
    result = validate_plan(protocol)
    assert any("consecutive" in w for w in result.warnings)


def test_validate_ok_is_always_true():
    """ValidationResult.ok is always True (warnings don't block)."""
    protocol = {
        "steps": [
            {"id": "s1", "primitive": "unknown.bad", "depends_on": ["bogus"]},
        ]
    }
    result = validate_plan(protocol)
    assert result.ok is True


# ===========================================================================
# 5. Agent Endpoint (E2E)
# ===========================================================================


@pytest.fixture
async def client():
    from app.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.anyio
async def test_agent_plan_endpoint_success(client: AsyncClient):
    """E2E: POST /agent/plan with valid primitives → run in awaiting_approval."""
    from app.api.v1.endpoints.agent import set_test_provider

    # Use primitives that exist in the capabilities registry
    mock_response = json.dumps({
        "steps": [
            {"id": "s1", "primitive": "robot.home", "params": {}, "depends_on": []},
            {"id": "s2", "primitive": "heat", "params": {"temp_c": 60}, "depends_on": ["s1"]},
        ],
    })
    provider = MockProvider(responses=[mock_response])
    set_test_provider(provider)

    try:
        resp = await client.post("/api/v1/agent/plan", json={
            "intent": "Aspirate 100uL and heat to 60C",
        })
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["status"] == "awaiting_approval"
        assert data["run_id"]
        assert len(data["plan_steps"]) == 2
        assert isinstance(data["grounding_warnings"], list)
        assert isinstance(data["validation_warnings"], list)
        assert isinstance(data["validation_info"], list)
    finally:
        set_test_provider(None)


@pytest.mark.anyio
async def test_agent_plan_unknown_primitive_returns_422(client: AsyncClient):
    """LLM returns unknown primitive → 422."""
    from app.api.v1.endpoints.agent import set_test_provider

    mock_response = json.dumps({
        "steps": [{"id": "s1", "primitive": "quantum.teleport", "params": {}}],
    })
    provider = MockProvider(responses=[mock_response])
    set_test_provider(provider)

    try:
        resp = await client.post("/api/v1/agent/plan", json={
            "intent": "Teleport the sample",
        })
        assert resp.status_code == 422
    finally:
        set_test_provider(None)


@pytest.mark.anyio
async def test_agent_plan_always_requires_approval(client: AsyncClient):
    """Agent runs always have status=awaiting_approval, even with approval=False policy."""
    from app.api.v1.endpoints.agent import set_test_provider

    mock_response = json.dumps({
        "steps": [
            {"id": "s1", "primitive": "robot.home", "params": {}, "depends_on": []},
        ],
    })
    provider = MockProvider(responses=[mock_response])
    set_test_provider(provider)

    try:
        resp = await client.post("/api/v1/agent/plan", json={
            "intent": "Home the robot",
            "policy_snapshot": {
                "max_temp_c": 95,
                "max_volume_ul": 1000,
                "allowed_primitives": ["robot.home", "heat"],
                "require_human_approval": False,  # explicitly false
            },
        })
        assert resp.status_code == 200
        data = resp.json()
        # Despite policy saying no approval needed, agent runs ALWAYS require it
        assert data["status"] == "awaiting_approval"
    finally:
        set_test_provider(None)


@pytest.mark.anyio
async def test_agent_plan_llm_error_returns_502(client: AsyncClient):
    """LLM call failure → 502."""
    from app.api.v1.endpoints.agent import set_test_provider

    # Empty responses → LLMError
    provider = MockProvider(responses=[])
    set_test_provider(provider)

    try:
        resp = await client.post("/api/v1/agent/plan", json={
            "intent": "Do something",
        })
        assert resp.status_code == 502
    finally:
        set_test_provider(None)


@pytest.mark.anyio
async def test_agent_plan_bad_json_returns_422(client: AsyncClient):
    """LLM returns unparseable response → 422."""
    from app.api.v1.endpoints.agent import set_test_provider

    provider = MockProvider(responses=["This is not JSON at all"])
    set_test_provider(provider)

    try:
        resp = await client.post("/api/v1/agent/plan", json={
            "intent": "Do something",
        })
        assert resp.status_code == 422
    finally:
        set_test_provider(None)


@pytest.mark.anyio
async def test_agent_event_audit_trail(client: AsyncClient):
    """Agent run should have provenance events with trigger_type='agent'."""
    from app.api.v1.endpoints.agent import set_test_provider

    mock_response = json.dumps({
        "steps": [
            {"id": "s1", "primitive": "robot.home", "params": {}, "depends_on": []},
        ],
    })
    provider = MockProvider(responses=[mock_response])
    set_test_provider(provider)

    try:
        resp = await client.post("/api/v1/agent/plan", json={
            "intent": "Home the robot",
        })
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]

        # Check events
        events_resp = await client.get(f"/api/v1/runs/{run_id}/events")
        assert events_resp.status_code == 200
        events = events_resp.json()
        assert len(events) >= 1

        # Find the run.created event
        created_events = [e for e in events if e["action"] == "run.created"]
        assert len(created_events) == 1
        assert created_events[0]["details"]["trigger_type"] == "agent"
    finally:
        set_test_provider(None)


@pytest.mark.anyio
async def test_full_pipeline_integration():
    """Full pipeline: MockProvider → plan → ground → validate → create_run."""
    provider = MockProvider(responses=[SIMPLE_PLAN_JSON])

    # 1. Plan
    plan = await plan_from_intent("Home the robot and heat", provider=provider)
    assert len(plan.steps) == 2

    # 2. Ground
    grounding = ground_plan(plan)
    assert grounding.ok, f"Grounding errors: {grounding.errors}"

    # 3. Validate
    validation = validate_plan(grounding.protocol)
    assert validation.ok

    # 4. Create run (directly, not via endpoint)
    from app.services.run_service import create_run, default_policy

    policy = default_policy()
    policy["require_human_approval"] = True

    run = create_run(
        trigger_type="agent",
        trigger_payload={"intent": "test", "raw_llm_response": plan.raw_response},
        campaign_id=None,
        protocol=grounding.protocol,
        inputs={},
        policy_snapshot=policy,
        actor="agent-planner",
    )
    assert run["status"] == "awaiting_approval"
    assert run["trigger_type"] == "agent"
    assert len(run["steps"]) == 2


@pytest.mark.anyio
async def test_agent_endpoint_registered():
    """Agent endpoint appears in the OpenAPI schema."""
    from app.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/openapi.json")
        assert resp.status_code == 200
        paths = resp.json()["paths"]
        assert "/api/v1/agent/plan" in paths
