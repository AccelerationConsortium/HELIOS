"""KPI metrics API endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.services.metrics import get_kpi_summary, get_run_kpis
from app.services.run_service import get_run

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/runs/{run_id}/kpis")
async def get_run_kpis_endpoint(run_id: str) -> list[dict]:
    """Return all KPIs extracted for a given run."""
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return get_run_kpis(run_id)


@router.get("/kpis/{kpi_name}")
async def get_kpi_summary_endpoint(
    kpi_name: str,
    schema_version: str | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
) -> list[dict]:
    """Return recent values for a given KPI name across runs."""
    return get_kpi_summary(
        kpi_name=kpi_name,
        schema_version=schema_version,
        limit=limit,
    )
