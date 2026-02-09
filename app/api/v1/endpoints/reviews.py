"""Run review API endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.services.reviewer import get_run_review
from app.services.run_service import get_run

router = APIRouter(prefix="/reviews", tags=["reviews"])


@router.get("/runs/{run_id}")
async def get_run_review_endpoint(run_id: str) -> dict:
    """Return the LLM review for a given run, or 404 if not yet reviewed."""
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    review = get_run_review(run_id)
    if review is None:
        raise HTTPException(status_code=404, detail="review not found")
    return review
