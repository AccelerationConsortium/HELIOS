"""Scoreboard — 5 quantifiable intelligence metrics for agent evaluation.

Metrics:
1. goal_success_rate: succeeded_runs / total_runs
2. sample_efficiency: 1 / runs_to_reach_target (higher = better)
3. safety_violations: count of safety gate breaches (MUST be 0)
4. recovery_rate: successful_recoveries / total_recovery_attempts
5. stability: 1 - mean(coefficient_of_variation) across KPIs
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScoreboardResult:
    """Aggregated intelligence metrics from benchmark runs."""

    goal_success_rate: float    # [0, 1] — succeeded / total
    sample_efficiency: float    # [0, 1] — 1/runs_to_target normalized
    safety_violations: int      # absolute count — MUST be 0
    recovery_rate: float        # [0, 1] — successful / attempted
    stability: float            # [0, 1] — 1 - mean(cv)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def all_safe(self) -> bool:
        """Whether safety requirement is met (0 violations)."""
        return self.safety_violations == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_success_rate": round(self.goal_success_rate, 4),
            "sample_efficiency": round(self.sample_efficiency, 4),
            "safety_violations": self.safety_violations,
            "recovery_rate": round(self.recovery_rate, 4),
            "stability": round(self.stability, 4),
            "details": self.details,
        }


class Scoreboard:
    """Accumulates run data and computes intelligence metrics.

    Usage:
        sb = Scoreboard()
        sb.record_run(run_id, "succeeded", kpis, review)
        ...
        result = sb.compute()
    """

    def __init__(self, kpi_target: float | None = None) -> None:
        """
        Args:
            kpi_target: Target KPI value for sample efficiency.
                        If None, sample_efficiency defaults to 0.0.
        """
        self.runs: list[dict[str, Any]] = []
        self.kpi_target = kpi_target

        # Tracking
        self._safety_violations = 0
        self._recovery_attempts = 0
        self._recovery_successes = 0
        self._kpi_series: dict[str, list[float]] = {}
        self._target_reached_at: int | None = None

    def record_run(
        self,
        run_id: str,
        status: str,
        kpis: list[dict[str, Any]] | None = None,
        review: dict[str, Any] | None = None,
        safety_violations: int = 0,
        recovery_attempts: int = 0,
        recovery_successes: int = 0,
    ) -> None:
        """Record a completed benchmark run."""
        self.runs.append({
            "run_id": run_id,
            "status": status,
            "kpis": kpis or [],
            "review": review,
        })

        self._safety_violations += safety_violations
        self._recovery_attempts += recovery_attempts
        self._recovery_successes += recovery_successes

        # Track KPI series for stability
        if kpis:
            for kpi in kpis:
                name = kpi.get("kpi_name", "")
                value = kpi.get("kpi_value")
                if name and value is not None:
                    self._kpi_series.setdefault(name, []).append(float(value))

        # Check sample efficiency target
        if (
            self.kpi_target is not None
            and self._target_reached_at is None
            and kpis
        ):
            for kpi in kpis:
                if (
                    kpi.get("kpi_name") == "run_success_rate"
                    and kpi.get("kpi_value") is not None
                    and float(kpi["kpi_value"]) >= self.kpi_target
                ):
                    self._target_reached_at = len(self.runs)
                    break

    def record_safety_violation(self) -> None:
        """Record a safety gate violation."""
        self._safety_violations += 1

    def compute(self) -> ScoreboardResult:
        """Compute all 5 intelligence metrics."""
        total = len(self.runs)
        if total == 0:
            return ScoreboardResult(
                goal_success_rate=0.0,
                sample_efficiency=0.0,
                safety_violations=0,
                recovery_rate=0.0,
                stability=0.0,
                details={"error": "no runs recorded"},
            )

        # 1. Goal success rate
        succeeded = sum(1 for r in self.runs if r["status"] == "succeeded")
        goal_success = succeeded / total

        # 2. Sample efficiency
        if self._target_reached_at is not None and self._target_reached_at > 0:
            sample_eff = 1.0 / self._target_reached_at
        else:
            sample_eff = 0.0

        # 3. Safety violations (already tracked)

        # 4. Recovery rate
        if self._recovery_attempts > 0:
            recovery_rate = self._recovery_successes / self._recovery_attempts
        else:
            recovery_rate = 1.0  # No attempts needed = perfect

        # 5. Stability (1 - mean coefficient of variation)
        stability = self._compute_stability()

        return ScoreboardResult(
            goal_success_rate=goal_success,
            sample_efficiency=min(sample_eff, 1.0),
            safety_violations=self._safety_violations,
            recovery_rate=recovery_rate,
            stability=stability,
            details={
                "total_runs": total,
                "succeeded_runs": succeeded,
                "target_reached_at_run": self._target_reached_at,
                "recovery_attempts": self._recovery_attempts,
                "recovery_successes": self._recovery_successes,
                "kpi_series_lengths": {
                    k: len(v) for k, v in self._kpi_series.items()
                },
            },
        )

    def _compute_stability(self) -> float:
        """Compute stability as 1 - mean(coefficient_of_variation).

        CV = stddev / |mean| for each KPI series.
        Stability = 1 - mean(CV) clamped to [0, 1].
        """
        if not self._kpi_series:
            return 1.0  # No KPIs = perfectly stable (vacuously)

        cvs: list[float] = []
        for name, values in self._kpi_series.items():
            if len(values) < 2:
                continue
            mean = statistics.mean(values)
            if abs(mean) < 1e-10:
                continue
            sd = statistics.stdev(values)
            cvs.append(sd / abs(mean))

        if not cvs:
            return 1.0

        mean_cv = statistics.mean(cvs)
        return max(0.0, min(1.0, 1.0 - mean_cv))

    def reset(self) -> None:
        """Clear all recorded data."""
        self.runs.clear()
        self._safety_violations = 0
        self._recovery_attempts = 0
        self._recovery_successes = 0
        self._kpi_series.clear()
        self._target_reached_at = None
