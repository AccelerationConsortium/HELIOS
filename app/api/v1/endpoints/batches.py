"""Batch candidate generation API endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.api.v1.schemas import BatchGenerateRequest
from app.services.candidate_gen import (
    ParameterSpace,
    SearchDimension,
    generate_batch,
    get_batch,
    list_batches,
    list_candidates,
)

router = APIRouter(prefix="/batches", tags=["batches"])


@router.post("/generate")
async def generate_batch_endpoint(payload: BatchGenerateRequest) -> dict:
    """Generate a batch of candidate parameter sets."""
    try:
        dimensions = tuple(
            SearchDimension(
                param_name=d["param_name"],
                param_type=d.get("param_type", "number"),
                min_value=d.get("min_value"),
                max_value=d.get("max_value"),
                log_scale=d.get("log_scale", False),
                choices=tuple(d["choices"]) if d.get("choices") else None,
                step_key=d.get("step_key"),
                primitive=d.get("primitive"),
            )
            for d in payload.dimensions
        )
        space = ParameterSpace(
            dimensions=dimensions,
            protocol_template=payload.protocol_template,
        )
        result = generate_batch(
            space=space,
            strategy=payload.strategy,
            n_candidates=payload.n_candidates,
            seed=payload.seed,
            created_by=payload.created_by,
            campaign_id=payload.campaign_id,
        )
        return get_batch(result.batch_id)  # type: ignore[return-value]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{batch_id}")
async def get_batch_endpoint(batch_id: str) -> dict:
    """Return a batch request with all its candidates."""
    batch = get_batch(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="batch not found")
    return batch


@router.get("/{batch_id}/candidates")
async def list_candidates_endpoint(batch_id: str) -> list[dict]:
    """Return all candidates for a batch."""
    batch = get_batch(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="batch not found")
    return list_candidates(batch_id)


@router.get("")
async def list_batches_endpoint(
    campaign_id: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
) -> list[dict]:
    """List batch requests, optionally filtered by campaign."""
    return list_batches(campaign_id=campaign_id, limit=limit)
