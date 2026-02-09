"""Evolution Engine API endpoints — priors, templates, proposals."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.api.v1.schemas import ProposalDecisionRequest, TemplateCreateRequest
from app.services.evolution import (
    approve_proposal,
    create_template,
    get_active_evolved_prior,
    get_proposal,
    get_template,
    list_evolved_priors,
    list_proposals,
    list_templates,
    reject_proposal,
)

router = APIRouter(prefix="/evolution", tags=["evolution"])


# ---------------------------------------------------------------------------
# Evolved Priors
# ---------------------------------------------------------------------------


@router.get("/priors")
async def list_priors_endpoint(
    primitive: str | None = Query(None),
    active_only: bool = Query(True),
) -> list[dict]:
    """List evolved priors with optional filters."""
    return list_evolved_priors(primitive=primitive, active_only=active_only)


@router.get("/priors/{primitive}/{param_name}")
async def get_prior_endpoint(primitive: str, param_name: str) -> dict:
    """Get the active evolved prior for a (primitive, param_name)."""
    prior = get_active_evolved_prior(primitive, param_name)
    if prior is None:
        raise HTTPException(status_code=404, detail="evolved prior not found")
    return {
        "id": prior.id,
        "primitive": prior.primitive,
        "param_name": prior.param_name,
        "evolved_min": prior.evolved_min,
        "evolved_max": prior.evolved_max,
        "confidence": prior.confidence,
        "source_run_id": prior.source_run_id,
        "proposal_id": prior.proposal_id,
        "generation": prior.generation,
        "is_active": prior.is_active,
    }


# ---------------------------------------------------------------------------
# Protocol Templates
# ---------------------------------------------------------------------------


@router.post("/templates")
async def create_template_endpoint(payload: TemplateCreateRequest) -> dict:
    """Create a protocol template manually."""
    try:
        return create_template(
            name=payload.name,
            protocol=payload.protocol,
            parent_template_id=payload.parent_template_id,
            tags=payload.tags,
            created_by=payload.created_by,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/templates")
async def list_templates_endpoint(
    name: str | None = Query(None),
    is_active: bool | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
) -> list[dict]:
    """List protocol templates."""
    return list_templates(name=name, is_active=is_active, limit=limit)


@router.get("/templates/{template_id}")
async def get_template_endpoint(template_id: str) -> dict:
    """Get a single protocol template."""
    template = get_template(template_id)
    if template is None:
        raise HTTPException(status_code=404, detail="template not found")
    return template


# ---------------------------------------------------------------------------
# Evolution Proposals (Human Gate)
# ---------------------------------------------------------------------------


@router.get("/proposals")
async def list_proposals_endpoint(
    run_id: str | None = Query(None),
    status: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
) -> list[dict]:
    """List evolution proposals."""
    return list_proposals(run_id=run_id, status=status, limit=limit)


@router.get("/proposals/{proposal_id}")
async def get_proposal_endpoint(proposal_id: str) -> dict:
    """Get a single evolution proposal."""
    proposal = get_proposal(proposal_id)
    if proposal is None:
        raise HTTPException(status_code=404, detail="proposal not found")
    return proposal


@router.post("/proposals/{proposal_id}/approve")
async def approve_proposal_endpoint(
    proposal_id: str, payload: ProposalDecisionRequest
) -> dict:
    """Approve a pending evolution proposal."""
    try:
        return approve_proposal(proposal_id, payload.reviewer, payload.reason)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/proposals/{proposal_id}/reject")
async def reject_proposal_endpoint(
    proposal_id: str, payload: ProposalDecisionRequest
) -> dict:
    """Reject a pending evolution proposal."""
    try:
        return reject_proposal(proposal_id, payload.reviewer, payload.reason)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
