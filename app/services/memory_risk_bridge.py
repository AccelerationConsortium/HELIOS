"""Memory-enriched risk assessment — bridges semantic memory + failure history
into agent-level risk scoring and granularity decisions.

Three capabilities:
1. **PrimitiveRiskProfile** — per-primitive risk score from semantic memory
   (success_rate) + failure_frequency table (historical failure count).
2. **WorkflowRiskReport** — aggregated risk for a full workflow, used by
   ExecutionAgent._assess_risk() and SafetyAgent.
3. **GranularityAdvisor** — recommends Granularity (FINE/COARSE/ADAPTIVE)
   based on historical failure signals.

All reads are advisory — wrapped in try/except, never block.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from app.core.db import connection, parse_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrimitiveRiskProfile:
    """Risk profile for a single primitive, drawn from memory."""

    primitive: str
    success_rate: float          # 0.0–1.0 from semantic memory
    total_observations: int      # how many times we've seen this primitive
    failure_count: int           # from failure_frequency table
    dominant_failure_type: str   # most common failure type, or ""
    dominant_cause: str          # most common likely_cause, or ""
    param_risk_signals: dict[str, float]  # param_name → deviation risk (0–1)

    @property
    def failure_rate(self) -> float:
        return 1.0 - self.success_rate

    @property
    def is_well_known(self) -> bool:
        """Has enough historical data to be confident."""
        return self.total_observations >= 10

    @property
    def risk_score(self) -> float:
        """Composite risk score: 0.0 = safe, 1.0 = dangerous."""
        # Weight failure rate more for well-known primitives
        if self.total_observations == 0:
            return 0.5  # Unknown = moderate risk
        # Blend failure_rate with a novelty penalty
        novelty = max(0.0, 1.0 - self.total_observations / 20.0)
        return min(1.0, self.failure_rate * 0.6 + novelty * 0.4)


@dataclass
class WorkflowRiskReport:
    """Aggregated risk report for a complete workflow."""

    primitive_profiles: list[PrimitiveRiskProfile]
    overall_risk: float = 0.0          # weighted average
    high_risk_primitives: list[str] = field(default_factory=list)
    novel_primitives: list[str] = field(default_factory=list)
    recommended_granularity: str = "coarse"
    explanation: str = ""


# ---------------------------------------------------------------------------
# Core: build risk profile for a single primitive
# ---------------------------------------------------------------------------


def get_primitive_risk_profile(
    primitive: str,
    workflow_params: dict[str, Any] | None = None,
) -> PrimitiveRiskProfile:
    """Build a risk profile for *primitive* from memory + failure history.

    Parameters
    ----------
    primitive : str
        The primitive name (e.g. "robot.aspirate", "heat").
    workflow_params : dict | None
        Actual parameters for this step — used to check deviation from
        historical mean (param_risk_signals).
    """
    success_rate = 1.0
    total_obs = 0
    failure_count = 0
    dominant_failure_type = ""
    dominant_cause = ""
    param_risk_signals: dict[str, float] = {}

    try:
        with connection() as conn:
            # ── 1. Semantic memory: success rate + param stats ─────────
            sem_rows = conn.execute(
                "SELECT param_name, mean, stddev, sample_count, "
                "       success_rate, success_count, total_count "
                "FROM memory_semantic WHERE primitive = ?",
                (primitive,),
            ).fetchall()

            if sem_rows:
                # Use the max total_count across params as observation count
                total_obs = max(r["total_count"] for r in sem_rows)
                # Average success_rate across all params
                rates = [r["success_rate"] for r in sem_rows if r["total_count"] > 0]
                if rates:
                    success_rate = sum(rates) / len(rates)

                # Check param deviation from historical mean
                if workflow_params:
                    for row in sem_rows:
                        pname = row["param_name"]
                        if pname in workflow_params and row["stddev"] > 0:
                            actual = workflow_params[pname]
                            if isinstance(actual, (int, float)):
                                z_score = abs(float(actual) - row["mean"]) / row["stddev"]
                                # Clamp to 0–1: z=0 → 0 risk, z>=3 → 1.0 risk
                                param_risk_signals[pname] = min(1.0, z_score / 3.0)

            # ── 2. Failure frequency: historical failure patterns ──────
            ff_rows = conn.execute(
                "SELECT failure_type, likely_cause, occurrence_count "
                "FROM failure_frequency WHERE primitive = ? "
                "ORDER BY occurrence_count DESC",
                (primitive,),
            ).fetchall()

            if ff_rows:
                failure_count = sum(r["occurrence_count"] for r in ff_rows)
                dominant_failure_type = ff_rows[0]["failure_type"]
                dominant_cause = ff_rows[0]["likely_cause"]

    except Exception:
        logger.debug("Failed to build risk profile for %s", primitive, exc_info=True)

    return PrimitiveRiskProfile(
        primitive=primitive,
        success_rate=success_rate,
        total_observations=total_obs,
        failure_count=failure_count,
        dominant_failure_type=dominant_failure_type,
        dominant_cause=dominant_cause,
        param_risk_signals=param_risk_signals,
    )


# ---------------------------------------------------------------------------
# Workflow-level risk report
# ---------------------------------------------------------------------------

# Thresholds for risk classification
_HIGH_RISK_THRESHOLD = 0.4      # primitive risk_score above this = high risk
_NOVEL_OBS_THRESHOLD = 5        # fewer observations = novel
_FINE_GRANULARITY_THRESHOLD = 0.5
_ADAPTIVE_GRANULARITY_THRESHOLD = 0.25


def assess_workflow_risk(
    primitives_with_params: list[tuple[str, dict[str, Any]]],
) -> WorkflowRiskReport:
    """Build an aggregated risk report for a workflow.

    Parameters
    ----------
    primitives_with_params : list of (primitive_name, step_params) tuples
        One entry per workflow step.

    Returns
    -------
    WorkflowRiskReport
        Overall risk, per-primitive profiles, and recommended granularity.
    """
    profiles: list[PrimitiveRiskProfile] = []
    high_risk: list[str] = []
    novel: list[str] = []

    for prim, params in primitives_with_params:
        profile = get_primitive_risk_profile(prim, params)
        profiles.append(profile)
        if profile.risk_score > _HIGH_RISK_THRESHOLD:
            high_risk.append(prim)
        if profile.total_observations < _NOVEL_OBS_THRESHOLD:
            novel.append(prim)

    # Weighted average risk (weight by 1/success_rate so riskier primitives dominate)
    if profiles:
        weights = [max(0.1, p.risk_score) for p in profiles]
        overall = sum(p.risk_score * w for p, w in zip(profiles, weights)) / sum(weights)
    else:
        overall = 0.0

    # Granularity recommendation
    if overall > _FINE_GRANULARITY_THRESHOLD or len(high_risk) > 0:
        granularity = "fine"
    elif overall > _ADAPTIVE_GRANULARITY_THRESHOLD or len(novel) > 0:
        granularity = "adaptive"
    else:
        granularity = "coarse"

    # Explanation
    parts: list[str] = []
    if high_risk:
        parts.append(f"high-risk primitives: {high_risk}")
    if novel:
        parts.append(f"novel/low-data primitives: {novel}")
    if profiles:
        worst = max(profiles, key=lambda p: p.risk_score)
        if worst.dominant_failure_type:
            parts.append(
                f"worst: {worst.primitive} "
                f"(failure_rate={worst.failure_rate:.0%}, "
                f"dominant={worst.dominant_failure_type}/{worst.dominant_cause})"
            )
    explanation = " | ".join(parts) if parts else "all primitives nominal"

    return WorkflowRiskReport(
        primitive_profiles=profiles,
        overall_risk=overall,
        high_risk_primitives=high_risk,
        novel_primitives=novel,
        recommended_granularity=granularity,
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Convenience: get_primitive_success_rate (for simple lookups)
# ---------------------------------------------------------------------------


def get_primitive_success_rate(primitive: str) -> float | None:
    """Quick lookup: return success_rate from semantic memory, or None."""
    try:
        with connection() as conn:
            row = conn.execute(
                "SELECT success_rate FROM memory_semantic "
                "WHERE primitive = ? LIMIT 1",
                (primitive,),
            ).fetchone()
            return row["success_rate"] if row else None
    except Exception:
        return None


def get_failure_frequency(primitive: str) -> int:
    """Total failure count for *primitive* across all runs."""
    try:
        with connection() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(occurrence_count), 0) as total "
                "FROM failure_frequency WHERE primitive = ?",
                (primitive,),
            ).fetchone()
            return row["total"] if row else 0
    except Exception:
        return 0
