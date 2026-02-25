"""Typed query DSL — Layer 1 of double-layer query validation.

Provides a deterministic, LLM-free compilation path for structured lab
data queries.  The DSL maps directly to canonical views (v_experiment_*)
via app/services/query_compiler.py.

Design goals:
  - Type-safe : all fields validated by Pydantic
  - Deterministic : same DSL → same SQL, every time
  - Whitelist-only : only approved columns appear in ORDER BY / GROUP BY
  - No injection surface : all user values use ? placeholders in SQL
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Entity → canonical view / table mapping
# ---------------------------------------------------------------------------

Entity = Literal["runs", "candidates", "kpis", "artifacts", "rounds"]

ENTITY_SOURCE: dict[str, str] = {
    "runs":       "v_experiment_runs",
    "candidates": "campaign_candidates",
    "kpis":       "v_experiment_metrics",
    "artifacts":  "v_experiment_artifacts",
    "rounds":     "campaign_rounds",
}


# ---------------------------------------------------------------------------
# Per-entity column whitelists  (ORDER BY / GROUP BY safe columns only)
# ---------------------------------------------------------------------------

_VALID_COLUMNS: dict[str, frozenset[str]] = {
    "runs": frozenset({
        "run_id", "campaign_id", "round_number", "candidate_index",
        "status", "created_at", "updated_at",
        "kpi_value", "kpi_name", "params_json",
        "experiment_id", "protocol_id",
    }),
    "candidates": frozenset({
        "candidate_id", "campaign_id", "round_number", "candidate_index",
        "params_json", "predicted_kpi", "acquisition_score",
        "status", "created_at",
    }),
    "kpis": frozenset({
        "run_id", "campaign_id", "round_number",
        "kpi_name", "kpi_value", "qc_status",
        "review_verdict", "created_at",
    }),
    "artifacts": frozenset({
        "artifact_id", "run_id", "campaign_id",
        "artifact_type", "uri", "content_hash", "created_at",
    }),
    "rounds": frozenset({
        "round_id", "campaign_id", "round_number",
        "n_candidates", "best_kpi", "mean_kpi",
        "status", "started_at", "finished_at",
    }),
}


def is_valid_column(entity: str, col: str) -> bool:
    """Return True if ``col`` is whitelisted for ``entity``."""
    return col in _VALID_COLUMNS.get(entity, frozenset())


# ---------------------------------------------------------------------------
# Aggregation spec
# ---------------------------------------------------------------------------

AggFunc = Literal["count", "mean", "max", "min", "best"]


class AggSpec(BaseModel):
    """Optional aggregation over a numeric column."""

    func: AggFunc = "count"
    on: str = ""                          # column to aggregate (e.g. "kpi_value")
    direction: Literal["maximize", "minimize"] = "maximize"  # for func="best"
    group_by: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Experiment query DSL
# ---------------------------------------------------------------------------


class ExperimentQuery(BaseModel):
    """Typed query DSL for lab data retrieval.

    Covers ~80 % of routine queries without touching the LLM path.
    Unsupported queries fall through to the NL path in QueryAgent.

    Examples
    --------
    # All completed runs in campaign C-001
    ExperimentQuery(entity="runs", campaign_id="C-001", status="completed")

    # Best KPI per round for overpotential
    ExperimentQuery(
        entity="kpis",
        campaign_id="C-001",
        kpi_name="overpotential",
        agg=AggSpec(func="best", on="kpi_value", direction="minimize",
                    group_by=["round_number"]),
    )
    """

    # What to query
    entity: Entity = "runs"

    # --- Filters ---
    campaign_id:     str | None = None
    run_id:          str | None = None
    round_number:    int | None = None      # exact match
    round_min:       int | None = None      # round_number >= round_min
    round_max:       int | None = None      # round_number <= round_max
    candidate_index: int | None = None
    status:          str | None = None      # "completed" | "failed" | "running" …
    kpi_name:        str | None = None
    kpi_min:         float | None = None    # kpi_value >= kpi_min
    kpi_max:         float | None = None    # kpi_value <= kpi_max
    qc_passed_only:  bool = False           # filter to qc_status = 'passed'

    # --- Aggregation (optional) ---
    agg: AggSpec | None = None

    # --- Ordering & pagination ---
    order_by:  str | None = None
    order_dir: Literal["ASC", "DESC"] = "DESC"
    limit:     int = Field(default=100, ge=1, le=10000)
