"""Agent planning endpoint — converts natural language intent to a run.

``POST /api/v1/agent/plan`` accepts a natural language experiment description,
runs it through the LLM planner, grounds the result against the capabilities
registry, validates it, and creates a run in AWAITING_APPROVAL status.

Agent-generated runs **always** require human approval regardless of the
policy snapshot.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.llm_gateway import LLMError, LLMProvider
from app.services.plan_grounding import ground_plan
from app.services.plan_validator import validate_plan
from app.services.planner import PlanParseError, plan_from_intent
from app.services.run_service import DomainError, create_run, default_policy

router = APIRouter(prefix="/agent", tags=["agent"])

# Module-level provider override for testing (None = use default singleton)
_test_provider: LLMProvider | None = None


def set_test_provider(provider: LLMProvider | None) -> None:
    """Set a test LLM provider override.  Pass ``None`` to clear."""
    global _test_provider
    _test_provider = provider


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class PlanRequest(BaseModel):
    """Natural language experiment intent."""

    intent: str = Field(..., min_length=1, description="Natural language experiment description")
    inputs: dict[str, Any] = Field(default_factory=dict, description="Optional input bindings")
    policy_snapshot: dict[str, Any] | None = Field(
        None, description="Optional policy override (approval is always forced)"
    )


class PlanResponse(BaseModel):
    """Result of agent planning — includes run_id and diagnostics."""

    run_id: str
    status: str  # always "awaiting_approval"
    plan_steps: list[dict[str, Any]]
    grounding_warnings: list[str]
    validation_warnings: list[str]
    validation_info: list[str]


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/plan")
async def create_plan(request: PlanRequest) -> PlanResponse:
    """Convert natural language intent into an executable run.

    Pipeline:
    1. LLM call → PlanResult (structured steps)
    2. Ground → protocol JSON (validate primitives, coerce params)
    3. Validate → warnings (static analysis)
    4. Create run via existing compiler + safety pipeline
    5. Return run_id + diagnostics

    The run is always created with ``require_human_approval=True``.
    """
    # 1. Intent → Plan (LLM)
    try:
        plan = await plan_from_intent(request.intent, provider=_test_provider)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}") from exc
    except PlanParseError as exc:
        raise HTTPException(
            status_code=422, detail=f"Could not parse LLM response: {exc}"
        ) from exc

    # 2. Ground: PlanResult → protocol JSON
    grounding = ground_plan(plan)
    if not grounding.ok:
        raise HTTPException(status_code=422, detail=grounding.errors)

    # 3. Validate (warnings only — never blocks)
    validation = validate_plan(grounding.protocol)

    # 4. Force human approval for all agent-generated runs
    policy = request.policy_snapshot or default_policy()
    policy["require_human_approval"] = True

    # 5. Create run via existing pipeline
    try:
        run = create_run(
            trigger_type="agent",
            trigger_payload={
                "intent": request.intent,
                "raw_llm_response": plan.raw_response,
                "model": plan.model,
            },
            campaign_id=None,
            protocol=grounding.protocol,
            inputs=request.inputs,
            policy_snapshot=policy,
            actor="agent-planner",
        )
    except DomainError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PlanResponse(
        run_id=run["id"],
        status=run["status"],
        plan_steps=[asdict(s) for s in plan.steps],
        grounding_warnings=grounding.warnings,
        validation_warnings=validation.warnings,
        validation_info=validation.info,
    )
