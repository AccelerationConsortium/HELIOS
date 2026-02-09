"""Workflow import endpoint — accepts battery-lab JSON and creates a run."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.run_service import DomainError, create_run
from app.services.workflow_translator import TranslationError, translate_battery_workflow

router = APIRouter(prefix="/workflows", tags=["workflows"])


class WorkflowImportRequest(BaseModel):
    """Body for importing a battery-lab workflow."""

    workflow: dict[str, Any] = Field(
        ..., description="Full battery-lab workflow JSON (with 'phases' key)"
    )
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Run inputs (instrument_id, etc.)",
    )
    policy_snapshot: dict[str, Any] | None = Field(
        default=None,
        description="Optional safety policy override",
    )
    actor: str = Field(default="api", description="Identity of the requester")


@router.post("/import")
def import_workflow(body: WorkflowImportRequest) -> dict[str, Any]:
    """Translate a battery-lab workflow and create a scheduled run.

    The workflow is translated from the phase-based format to OTbot's
    flat step-list format, compiled, safety-checked, and persisted.
    """
    try:
        protocol = translate_battery_workflow(body.workflow)
    except TranslationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        run = create_run(
            trigger_type="workflow_import",
            trigger_payload={"source": "battery_lab_workflow"},
            campaign_id=None,
            protocol=protocol,
            inputs=body.inputs,
            policy_snapshot=body.policy_snapshot,
            actor=body.actor,
        )
    except DomainError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {"run_id": run["id"], "status": run["status"], "steps": len(run["steps"])}
