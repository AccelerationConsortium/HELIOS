"""Campaign-level Metrics Engine — Discovery Velocity, Cost-per-Insight, Success Rate.

Implements the outer meta-RL feedback signals described in the OTbot AI4X paper.
Three campaign-level KPIs feed back to the L3 Orchestrator as meta-optimization
signals for adjusting round cadence, budget allocation, and strategy selection.

KPIs
----
1. **Discovery Velocity**: objective improvement per unit time (best_kpi Δ / Δt)
2. **Cost-per-Insight**: total experiments / number of actionable findings
3. **Success Rate**: experiments producing valid KPI values / total experiments

Integration
-----------
- Callable from ``campaign_loop.run_campaign()`` after round evaluation
- Optional event bus listener on ``run.completed`` for async background updates
- Results persisted to ``campaign_metrics`` table for cross-campaign learning

All operations are advisory — wrapped in try/except, never block campaign execution.
Pure Python stdlib only.  No LLM in the critical path.
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.core.db import connection, json_dumps, parse_json, run_txn, utcnow_iso

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema versioning
# ---------------------------------------------------------------------------

CAMPAIGN_METRICS_SCHEMA_VERSION = "1"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CampaignMetricsSnapshot:
    """Point-in-time campaign metrics computed after a round completes.

    Attributes:
        campaign_id: Campaign identifier.
        round_number: Round at which metrics were computed.
        discovery_velocity: Objective improvement per hour (|Δbest_kpi| / Δt_hours).
        cost_per_insight: Total experiments / actionable findings (lower is better).
        success_rate: Fraction of experiments with valid KPI [0.0, 1.0].
        details: Extra context (e.g. timestamps, counts) for debugging.
    """

    campaign_id: str
    round_number: int
    discovery_velocity: float
    cost_per_insight: float
    success_rate: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetaRLSignal:
    """Formatted signal for the L3 Orchestrator's outer meta-RL loop.

    Feeds into strategy selection, budget allocation, and round cadence.

    Attributes:
        campaign_id: Campaign identifier.
        round_number: Current round.
        discovery_velocity: Objective improvement per hour.
        cost_per_insight: Experiments per actionable finding.
        success_rate: Valid-result ratio.
        velocity_trend: "improving" | "stable" | "declining" based on recent history.
        efficiency_trend: "improving" | "stable" | "declining" for cost-per-insight.
        recommended_action: Advisory hint: "accelerate" | "maintain" | "decelerate" | "pivot".
    """

    campaign_id: str
    round_number: int
    discovery_velocity: float
    cost_per_insight: float
    success_rate: float
    velocity_trend: str
    efficiency_trend: str
    recommended_action: str


# ---------------------------------------------------------------------------
# Core computation functions (pure, testable)
# ---------------------------------------------------------------------------


def _parse_iso_dt(iso_str: str | None) -> datetime | None:
    """Parse ISO datetime string to timezone-aware datetime."""
    if not iso_str:
        return None
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def compute_discovery_velocity(campaign_id: str) -> tuple[float, dict[str, Any]]:
    """Compute discovery velocity: objective improvement per hour.

    Queries campaign_state for best_kpi and timestamps, then:
    velocity = |best_kpi_improvement| / elapsed_hours

    Returns
    -------
    (velocity, details_dict)
    """
    with connection() as conn:
        state = conn.execute(
            """SELECT best_kpi, direction, kpi_history_json,
                      created_at, updated_at
               FROM campaign_state WHERE campaign_id = ?""",
            (campaign_id,),
        ).fetchone()

    if state is None:
        return 0.0, {"error": "campaign_not_found"}

    kpi_history = parse_json(state["kpi_history_json"], [])
    if len(kpi_history) < 2:
        return 0.0, {"reason": "insufficient_history", "n_points": len(kpi_history)}

    direction = state["direction"]
    best_kpi = state["best_kpi"]
    initial_kpi = kpi_history[0] if kpi_history else best_kpi

    # Guard against NULL KPI values from the database
    if best_kpi is None or initial_kpi is None:
        return 0.0, {
            "error": "null_kpi_values",
            "best_kpi": best_kpi,
            "initial_kpi": initial_kpi,
        }

    # Compute KPI improvement magnitude
    if direction == "minimize":
        improvement = float(initial_kpi) - float(best_kpi)
    else:
        improvement = float(best_kpi) - float(initial_kpi)

    # Compute elapsed time in hours
    created_at = _parse_iso_dt(state["created_at"])
    updated_at = _parse_iso_dt(state["updated_at"])

    if created_at and updated_at and updated_at > created_at:
        elapsed_hours = (updated_at - created_at).total_seconds() / 3600.0
    else:
        elapsed_hours = 1.0  # Default 1 hour to avoid division by zero

    velocity = max(improvement, 0.0) / max(elapsed_hours, 0.001)

    details = {
        "initial_kpi": initial_kpi,
        "best_kpi": best_kpi,
        "improvement": improvement,
        "elapsed_hours": round(elapsed_hours, 4),
        "direction": direction,
        "n_history_points": len(kpi_history),
    }
    return round(velocity, 6), details


def compute_cost_per_insight(campaign_id: str) -> tuple[float, dict[str, Any]]:
    """Compute cost-per-insight: experiments per actionable finding.

    An "actionable finding" is a candidate whose KPI improved upon
    the running best at the time it was evaluated.

    Returns
    -------
    (cost_per_insight, details_dict)
        Lower values are better (fewer experiments per insight).
    """
    with connection() as conn:
        state = conn.execute(
            "SELECT direction, kpi_history_json FROM campaign_state WHERE campaign_id = ?",
            (campaign_id,),
        ).fetchone()

        candidates = conn.execute(
            """SELECT kpi_value, round_number, candidate_index
               FROM campaign_candidates
               WHERE campaign_id = ? AND status = 'done' AND kpi_value IS NOT NULL
               ORDER BY round_number ASC, candidate_index ASC""",
            (campaign_id,),
        ).fetchall()

    if state is None:
        return float("inf"), {"error": "campaign_not_found"}

    direction = state["direction"]
    total_completed = len(candidates)

    if total_completed == 0:
        return float("inf"), {"reason": "no_completed_candidates"}

    # Iterate candidates to identify "actionable findings" (improved running best)
    n_insights = 0
    running_best: float | None = None

    for cand in candidates:
        kpi_val = cand["kpi_value"]
        if kpi_val is None:
            continue

        if running_best is None:
            running_best = kpi_val
            n_insights += 1  # First valid result counts as an insight (baseline)
            continue

        is_improvement = (
            (kpi_val < running_best) if direction == "minimize"
            else (kpi_val > running_best)
        )
        if is_improvement:
            n_insights += 1
            running_best = kpi_val

    cost = total_completed / max(n_insights, 1)

    details = {
        "total_completed": total_completed,
        "n_insights": n_insights,
        "direction": direction,
        "final_best": running_best,
    }
    return round(cost, 4), details


def compute_success_rate(campaign_id: str) -> tuple[float, dict[str, Any]]:
    """Compute success rate: fraction of experiments with valid KPI values.

    Success = candidate completed with a non-NULL kpi_value and status='done'.

    Returns
    -------
    (success_rate, details_dict)
        Value in [0.0, 1.0]; higher is better.
    """
    with connection() as conn:
        row = conn.execute(
            """SELECT
                 COUNT(*) AS total,
                 SUM(CASE WHEN status = 'done' AND kpi_value IS NOT NULL THEN 1 ELSE 0 END) AS success
               FROM campaign_candidates
               WHERE campaign_id = ?""",
            (campaign_id,),
        ).fetchone()

    if row is None or row["total"] == 0:
        return 0.0, {"reason": "no_candidates"}

    total = row["total"]
    success = row["success"]
    rate = success / total

    details = {
        "total_candidates": total,
        "successful": success,
        "failed": total - success,
    }
    return round(rate, 4), details


# ---------------------------------------------------------------------------
# Trend analysis (for meta-RL signal)
# ---------------------------------------------------------------------------


def _compute_trend(values: list[float], window: int = 3) -> str:
    """Compute trend from recent metric history.

    Args:
        values: Metric values ordered by time.
        window: Number of recent points to consider.

    Returns:
        "improving" | "stable" | "declining"
    """
    if len(values) < 2:
        return "stable"

    recent = values[-window:] if len(values) >= window else values
    if len(recent) < 2:
        return "stable"

    # Simple linear trend via finite differences
    deltas = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
    avg_delta = sum(deltas) / len(deltas)

    # Threshold: 5% of mean is considered significant change
    mean_val = sum(recent) / len(recent)
    threshold = abs(mean_val * 0.05) if mean_val != 0 else 0.01

    if avg_delta > threshold:
        return "improving"
    elif avg_delta < -threshold:
        return "declining"
    return "stable"


def _recommend_action(
    velocity_trend: str,
    efficiency_trend: str,
    success_rate: float,
) -> str:
    """Generate advisory action based on trends.

    Returns
    -------
    "accelerate" | "maintain" | "decelerate" | "pivot"
    """
    # Very low success rate → pivot (switch strategy)
    if success_rate < 0.3:
        return "pivot"

    # Both velocity and efficiency declining → decelerate (finer exploration needed)
    if velocity_trend == "declining" and efficiency_trend == "declining":
        return "decelerate"

    # Velocity improving → accelerate
    if velocity_trend == "improving":
        return "accelerate"

    # Default: maintain current pace
    return "maintain"


# ---------------------------------------------------------------------------
# Main computation entry point
# ---------------------------------------------------------------------------


def compute_campaign_metrics(
    campaign_id: str,
    round_number: int,
) -> CampaignMetricsSnapshot:
    """Compute all three campaign-level KPIs for the current state.

    This is the primary entry point called from ``campaign_loop``
    after each round evaluation.

    Parameters
    ----------
    campaign_id:
        Campaign to compute metrics for.
    round_number:
        Current round number (for snapshot tracking).

    Returns
    -------
    CampaignMetricsSnapshot
        Frozen snapshot of all three KPIs with details.
    """
    velocity, v_details = compute_discovery_velocity(campaign_id)
    cost, c_details = compute_cost_per_insight(campaign_id)
    rate, r_details = compute_success_rate(campaign_id)

    details = {
        "discovery_velocity": v_details,
        "cost_per_insight": c_details,
        "success_rate": r_details,
    }

    return CampaignMetricsSnapshot(
        campaign_id=campaign_id,
        round_number=round_number,
        discovery_velocity=velocity,
        cost_per_insight=cost,
        success_rate=rate,
        details=details,
    )


def get_meta_rl_signal(campaign_id: str) -> MetaRLSignal:
    """Generate a formatted meta-RL signal for the L3 Orchestrator.

    Reads historical metrics snapshots to compute trends,
    then produces an advisory signal for strategy adjustment.

    Parameters
    ----------
    campaign_id:
        Campaign to generate signal for.

    Returns
    -------
    MetaRLSignal
        Complete signal with trends and recommended action.
    """
    history = get_campaign_metrics_history(campaign_id)

    if not history:
        return MetaRLSignal(
            campaign_id=campaign_id,
            round_number=0,
            discovery_velocity=0.0,
            cost_per_insight=float("inf"),
            success_rate=0.0,
            velocity_trend="stable",
            efficiency_trend="stable",
            recommended_action="maintain",
        )

    latest = history[-1]

    # Trend analysis over metric history
    velocity_history = [s["discovery_velocity"] for s in history]
    # Invert cost-per-insight for trend (lower cost = higher efficiency)
    # Filter out infinite values that would produce misleading trends
    efficiency_history = [
        1.0 / max(s["cost_per_insight"], 0.01)
        for s in history
        if s["cost_per_insight"] != float("inf")
    ]

    velocity_trend = _compute_trend(velocity_history)
    efficiency_trend = _compute_trend(efficiency_history)

    recommended_action = _recommend_action(
        velocity_trend, efficiency_trend, latest["success_rate"]
    )

    return MetaRLSignal(
        campaign_id=campaign_id,
        round_number=latest["round_number"],
        discovery_velocity=latest["discovery_velocity"],
        cost_per_insight=latest["cost_per_insight"],
        success_rate=latest["success_rate"],
        velocity_trend=velocity_trend,
        efficiency_trend=efficiency_trend,
        recommended_action=recommended_action,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def store_campaign_metrics(snapshot: CampaignMetricsSnapshot) -> str:
    """Persist a campaign metrics snapshot to the database.

    Returns the generated row ID.
    """
    row_id = str(uuid.uuid4())
    now = utcnow_iso()

    def _insert(conn: sqlite3.Connection) -> None:
        conn.execute(
            """INSERT INTO campaign_metrics
               (id, campaign_id, round_number,
                discovery_velocity, cost_per_insight, success_rate,
                meta_rl_signal_json, details_json, schema_version, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                row_id,
                snapshot.campaign_id,
                snapshot.round_number,
                snapshot.discovery_velocity,
                snapshot.cost_per_insight,
                snapshot.success_rate,
                json_dumps({}),  # meta_rl_signal populated by subsequent call
                json_dumps(snapshot.details),
                CAMPAIGN_METRICS_SCHEMA_VERSION,
                now,
            ),
        )

    run_txn(_insert)
    logger.debug(
        "Stored campaign metrics for %s round %d: vel=%.4f cpi=%.2f sr=%.2f",
        snapshot.campaign_id,
        snapshot.round_number,
        snapshot.discovery_velocity,
        snapshot.cost_per_insight,
        snapshot.success_rate,
    )
    return row_id


def get_campaign_metrics_history(
    campaign_id: str,
) -> list[dict[str, Any]]:
    """Retrieve all metrics snapshots for a campaign, ordered by round.

    Returns
    -------
    list of dicts with keys:
        round_number, discovery_velocity, cost_per_insight, success_rate,
        details, created_at
    """
    with connection() as conn:
        rows = conn.execute(
            """SELECT round_number, discovery_velocity, cost_per_insight,
                      success_rate, details_json, created_at
               FROM campaign_metrics
               WHERE campaign_id = ?
               ORDER BY round_number ASC""",
            (campaign_id,),
        ).fetchall()

    return [
        {
            "round_number": r["round_number"],
            "discovery_velocity": r["discovery_velocity"],
            "cost_per_insight": r["cost_per_insight"],
            "success_rate": r["success_rate"],
            "details": parse_json(r["details_json"], {}),
            "created_at": r["created_at"],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Campaign loop integration — callable hook
# ---------------------------------------------------------------------------


def compute_and_store_metrics(
    campaign_id: str,
    round_number: int,
) -> CampaignMetricsSnapshot | None:
    """Compute, store, and return campaign metrics.

    Called from ``campaign_loop.run_campaign()`` after round evaluation.
    Advisory — exceptions are caught and logged, never block campaign.

    Parameters
    ----------
    campaign_id:
        Campaign being run.
    round_number:
        Just-completed round number.

    Returns
    -------
    CampaignMetricsSnapshot or None if computation failed.
    """
    try:
        snapshot = compute_campaign_metrics(campaign_id, round_number)
        store_campaign_metrics(snapshot)
        logger.info(
            "Campaign %s round %d metrics: "
            "velocity=%.4f cost_per_insight=%.2f success_rate=%.2f",
            campaign_id,
            round_number,
            snapshot.discovery_velocity,
            snapshot.cost_per_insight,
            snapshot.success_rate,
        )
        return snapshot
    except Exception:
        logger.warning(
            "Campaign metrics computation failed for %s round %d",
            campaign_id,
            round_number,
            exc_info=True,
        )
        return None


# ---------------------------------------------------------------------------
# Event bus listener — async background updates
# ---------------------------------------------------------------------------

_listener_task: asyncio.Task[None] | None = None


async def _on_run_completed(run_id: str) -> None:
    """Recompute campaign metrics when a run completes.

    Looks up the campaign_id from the run, then recomputes metrics
    for the current round.
    """
    try:
        with connection() as conn:
            row = conn.execute(
                "SELECT campaign_id FROM runs WHERE id = ?", (run_id,)
            ).fetchone()

        if row is None or row["campaign_id"] is None:
            return  # Not a campaign run, skip

        campaign_id = row["campaign_id"]

        # Look up current round number
        with connection() as conn:
            state = conn.execute(
                "SELECT current_round FROM campaign_state WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()

        if state is None:
            return

        round_number = state["current_round"]
        compute_and_store_metrics(campaign_id, round_number)

    except Exception:
        logger.warning(
            "Campaign metrics update failed for run %s", run_id, exc_info=True
        )


async def start_campaign_metrics_listener(bus: Any) -> Any:
    """Subscribe to event bus for async campaign metrics updates.

    Returns the Subscription handle for cleanup.
    """
    global _listener_task

    sub = await bus.subscribe(run_id=None)  # Global subscription

    async def _listen() -> None:
        async for event in sub:
            if event.action == "run.completed":
                run_id = event.run_id
                if run_id:
                    await _on_run_completed(run_id)

    _listener_task = asyncio.create_task(_listen())
    logger.info("Campaign metrics listener started")
    return sub


async def stop_campaign_metrics_listener(sub: Any, bus: Any) -> None:
    """Cancel the campaign metrics listener and unsubscribe."""
    global _listener_task

    sub.cancel()
    await bus.unsubscribe(sub)

    if _listener_task is not None:
        _listener_task.cancel()
        try:
            await _listener_task
        except asyncio.CancelledError:
            pass
        _listener_task = None
    logger.info("Campaign metrics listener stopped")
