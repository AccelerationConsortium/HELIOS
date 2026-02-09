"""Goal-Generate-Execute-Evaluate-Evolve intelligent campaign loop orchestrator.

Drives multi-round optimization campaigns that:
1. **Goal**: Define what to optimize (KPI, direction, target, budget)
2. **Generate**: Produce candidate parameter sets via ``candidate_gen``
3. **Execute**: Run candidates through ``execute_fn`` callback (online or offline)
4. **Evaluate**: Extract KPIs, detect convergence, decide next action
5. **Evolve**: Trigger prior tightening and template creation via ``evolution``

Supports two modes:
- **Online**: ``run_campaign()`` with a real ``execute_fn`` that creates and runs experiments
- **Offline**: ``run_campaign_offline()`` with a ``sim_fn`` that returns KPI dicts directly

All operations are advisory -- wrapped in try/except, never block.
Pure Python stdlib only.  No LLM in the critical path.

Pipeline::

    goal = CampaignGoal(objective_kpi="overpotential_mv", direction="minimize", ...)
    result = run_campaign(goal, space, execute_fn)
    # or
    result = run_campaign_offline(goal, space, sim_fn)
"""
from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from app.core.db import utcnow_iso
from app.services.candidate_gen import BatchResult, ParameterSpace, generate_batch
from app.services.convergence import (
    ConvergenceConfig,
    ConvergenceStatus,
    detect_convergence,
)
from app.services.failure_signatures import learn_from_run
from app.services.metrics import get_run_kpis

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CampaignGoal:
    """Defines what the campaign is optimizing for.

    Attributes:
        objective_kpi: KPI name to optimize (e.g. "overpotential_mv").
        direction: "maximize" or "minimize".
        target_value: Optional absolute target; campaign stops when reached.
        max_rounds: Hard budget limit on number of rounds.
        batch_size: Number of candidates generated per round.
        strategy: Sampling strategy for candidate generation.
    """

    objective_kpi: str
    direction: str  # "maximize" | "minimize"
    target_value: float | None
    max_rounds: int
    batch_size: int
    strategy: str  # "lhs" | "bayesian" | "prior_guided" | "random" | "grid"

    def __post_init__(self) -> None:
        if self.direction not in ("maximize", "minimize"):
            raise ValueError(
                f"direction must be 'maximize' or 'minimize', got {self.direction!r}"
            )
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")

    @property
    def maximize(self) -> bool:
        """Whether higher KPI values are better."""
        return self.direction == "maximize"

    def is_target_reached(self, best_value: float) -> bool:
        """Check whether the target has been reached.

        Returns ``False`` if no target is set.
        """
        if self.target_value is None:
            return False
        if self.direction == "maximize":
            return best_value >= self.target_value
        return best_value <= self.target_value

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "objective_kpi": self.objective_kpi,
            "direction": self.direction,
            "target_value": self.target_value,
            "max_rounds": self.max_rounds,
            "batch_size": self.batch_size,
            "strategy": self.strategy,
        }


@dataclass(frozen=True)
class RoundResult:
    """Result of one campaign round.

    Attributes:
        round_number: Zero-based round index.
        run_ids: Run IDs produced in this round.
        kpi_values: Extracted objective KPI values (one per run).
        best_kpi: Best KPI observed in this round.
        convergence_status: Status string from ``ConvergenceStatus``.
        timestamp: ISO-8601 timestamp when the round completed.
    """

    round_number: int
    run_ids: tuple[str, ...]
    kpi_values: tuple[float, ...]
    best_kpi: float
    convergence_status: str
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "round_number": self.round_number,
            "run_ids": list(self.run_ids),
            "kpi_values": list(self.kpi_values),
            "best_kpi": self.best_kpi,
            "convergence_status": self.convergence_status,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class CampaignResult:
    """Final result of a campaign run.

    Attributes:
        goal: The campaign goal that was optimized.
        rounds: All round results in order.
        best_kpi: Best KPI observed across the entire campaign.
        best_round: Round index where the best KPI was observed.
        total_runs: Total number of runs executed.
        converged: Whether the campaign converged.
        target_reached: Whether the target value was reached.
        stop_reason: Why the campaign stopped.
    """

    goal: CampaignGoal
    rounds: tuple[RoundResult, ...]
    best_kpi: float
    best_round: int
    total_runs: int
    converged: bool
    target_reached: bool
    stop_reason: str  # "target_reached" | "converged" | "budget_exhausted" | "diverging"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "goal": self.goal.to_dict(),
            "rounds": [r.to_dict() for r in self.rounds],
            "best_kpi": self.best_kpi,
            "best_round": self.best_round,
            "total_runs": self.total_runs,
            "converged": self.converged,
            "target_reached": self.target_reached,
            "stop_reason": self.stop_reason,
        }


# ---------------------------------------------------------------------------
# Pure-logic evaluation (no DB dependency)
# ---------------------------------------------------------------------------


def evaluate_round_pure(
    kpi_values: list[float],
    all_history: list[float],
    goal: CampaignGoal,
    convergence_config: ConvergenceConfig | None = None,
) -> tuple[float, ConvergenceStatus]:
    """Evaluate a round of KPI values against the full campaign history.

    Pure logic -- no database access.  Suitable for offline / testing use.

    Parameters
    ----------
    kpi_values:
        KPI values from the current round.
    all_history:
        All KPI values observed so far (including current round).
    goal:
        The campaign goal defining optimization direction.
    convergence_config:
        Optional convergence detection thresholds.

    Returns
    -------
    tuple of (best_kpi_this_round, convergence_status)
    """
    if not kpi_values:
        return (
            float("inf") if goal.direction == "minimize" else float("-inf"),
            ConvergenceStatus(
                status="insufficient_data",
                confidence=0.0,
                details={"reason": "no KPI values in round"},
            ),
        )

    if goal.maximize:
        best_this_round = max(kpi_values)
    else:
        best_this_round = min(kpi_values)

    convergence = detect_convergence(
        all_history,
        config=convergence_config,
        maximize=goal.maximize,
    )

    return best_this_round, convergence


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------


_PLATEAU_CONFIDENCE_THRESHOLD = 0.7
_DIVERGING_CONFIDENCE_THRESHOLD = 0.8


def decide_next_action(
    goal: CampaignGoal,
    rounds: list[RoundResult],
    convergence_status: ConvergenceStatus,
) -> str:
    """Decide whether to continue or stop the campaign.

    Returns one of:
    - ``"continue"`` -- generate another round
    - ``"stop_target"`` -- target value reached
    - ``"stop_budget"`` -- max_rounds exhausted
    - ``"stop_converged"`` -- KPI has plateaued
    - ``"stop_diverging"`` -- KPI is diverging (getting worse)

    Parameters
    ----------
    goal:
        Campaign goal with target and budget constraints.
    rounds:
        All completed rounds so far.
    convergence_status:
        Latest convergence analysis result.
    """
    if not rounds:
        return "continue"

    # Compute best KPI across all rounds
    if goal.maximize:
        best_so_far = max(r.best_kpi for r in rounds)
    else:
        best_so_far = min(r.best_kpi for r in rounds)

    # Check target
    if goal.is_target_reached(best_so_far):
        return "stop_target"

    # Check budget
    if len(rounds) >= goal.max_rounds:
        return "stop_budget"

    # Check convergence
    if (
        convergence_status.status == "plateau"
        and convergence_status.confidence > _PLATEAU_CONFIDENCE_THRESHOLD
    ):
        return "stop_converged"

    # Check divergence
    if (
        convergence_status.status == "diverging"
        and convergence_status.confidence > _DIVERGING_CONFIDENCE_THRESHOLD
    ):
        return "stop_diverging"

    return "continue"


# ---------------------------------------------------------------------------
# DB-integrated round evaluation
# ---------------------------------------------------------------------------


def evaluate_round(
    run_ids: list[str],
    goal: CampaignGoal,
    all_kpi_history: list[float],
    round_number: int,
    convergence_config: ConvergenceConfig | None = None,
) -> RoundResult:
    """Evaluate a completed round by extracting KPIs from the database.

    For each run_id, fetches KPIs via ``get_run_kpis()`` and extracts the
    objective KPI.  Computes convergence against the full campaign history.

    Parameters
    ----------
    run_ids:
        Run IDs from this round.
    goal:
        Campaign goal defining which KPI to extract.
    all_kpi_history:
        All objective KPI values from previous rounds (mutable -- this
        function appends the current round's values).
    round_number:
        Zero-based round index.
    convergence_config:
        Optional convergence detection thresholds.

    Returns
    -------
    RoundResult
    """
    round_kpis: list[float] = []

    for run_id in run_ids:
        try:
            kpis = get_run_kpis(run_id)
            for kpi in kpis:
                if kpi["kpi_name"] == goal.objective_kpi:
                    value = kpi.get("kpi_value")
                    if value is not None and not (
                        isinstance(value, float) and math.isnan(value)
                    ):
                        round_kpis.append(float(value))
        except Exception:
            logger.warning(
                "Failed to extract KPIs for run %s -- skipping",
                run_id,
                exc_info=True,
            )

    # Append to history for convergence detection
    all_kpi_history.extend(round_kpis)

    best_this_round, convergence = evaluate_round_pure(
        round_kpis, all_kpi_history, goal, convergence_config
    )

    return RoundResult(
        round_number=round_number,
        run_ids=tuple(run_ids),
        kpi_values=tuple(round_kpis),
        best_kpi=best_this_round,
        convergence_status=convergence.status,
        timestamp=utcnow_iso(),
    )


# ---------------------------------------------------------------------------
# Main campaign loop (DB-integrated)
# ---------------------------------------------------------------------------


def run_campaign(
    goal: CampaignGoal,
    space: ParameterSpace,
    execute_fn: Callable[[dict[str, Any]], str],
    campaign_id: str | None = None,
    convergence_config: ConvergenceConfig | None = None,
) -> CampaignResult:
    """Run a full Goal-Generate-Execute-Evaluate-Evolve campaign loop.

    This is the primary entry point for DB-integrated campaign execution.

    Parameters
    ----------
    goal:
        Defines what KPI to optimize, direction, target, and budget.
    space:
        Parameter space for candidate generation.
    execute_fn:
        Callback that takes a params dict and returns a run_id after
        execution is complete.  For online mode, this creates and executes
        a real run.  For offline mode, use ``run_campaign_offline`` instead.
    campaign_id:
        Optional campaign identifier for batch tracking.
    convergence_config:
        Optional convergence detection configuration.

    Returns
    -------
    CampaignResult
        Final campaign outcome with all rounds, best KPI, and stop reason.
    """
    if campaign_id is None:
        campaign_id = str(uuid.uuid4())

    rounds: list[RoundResult] = []
    all_kpi_history: list[float] = []

    logger.info(
        "Starting campaign %s: kpi=%s direction=%s target=%s max_rounds=%d batch=%d strategy=%s",
        campaign_id,
        goal.objective_kpi,
        goal.direction,
        goal.target_value,
        goal.max_rounds,
        goal.batch_size,
        goal.strategy,
    )

    for round_number in range(goal.max_rounds):
        logger.info("Campaign %s: starting round %d", campaign_id, round_number)

        # --- 1. Generate candidates ---
        try:
            batch: BatchResult = generate_batch(
                space,
                goal.strategy,
                goal.batch_size,
                seed=round_number,
                campaign_id=campaign_id,
                kpi_name=goal.objective_kpi,
            )
        except Exception:
            logger.error(
                "Campaign %s round %d: candidate generation failed",
                campaign_id,
                round_number,
                exc_info=True,
            )
            break

        # --- 2. Execute each candidate ---
        run_ids: list[str] = []
        for candidate in batch.candidates:
            try:
                run_id = execute_fn(candidate.params)
                run_ids.append(run_id)
            except Exception:
                logger.warning(
                    "Campaign %s round %d: execution failed for candidate %d",
                    campaign_id,
                    round_number,
                    candidate.index,
                    exc_info=True,
                )

        if not run_ids:
            logger.warning(
                "Campaign %s round %d: no runs produced -- stopping",
                campaign_id,
                round_number,
            )
            break

        # --- 3. Learn from failures (advisory) ---
        for run_id in run_ids:
            try:
                learn_from_run(run_id)
            except Exception:
                logger.debug(
                    "Failure learning skipped for run %s", run_id, exc_info=True
                )

        # --- 4. Evaluate round ---
        round_result = evaluate_round(
            run_ids, goal, all_kpi_history, round_number, convergence_config
        )
        rounds.append(round_result)

        logger.info(
            "Campaign %s round %d: best_kpi=%.6f convergence=%s (%d values)",
            campaign_id,
            round_number,
            round_result.best_kpi,
            round_result.convergence_status,
            len(round_result.kpi_values),
        )

        # --- 5. Decide next action ---
        convergence = detect_convergence(
            all_kpi_history,
            config=convergence_config,
            maximize=goal.maximize,
        )
        action = decide_next_action(goal, rounds, convergence)

        if action != "continue":
            logger.info(
                "Campaign %s: stopping after round %d -- reason=%s",
                campaign_id,
                round_number,
                action,
            )
            return _build_result(goal, rounds, action)

        # --- 6. Trigger evolution (advisory, never blocks) ---
        _trigger_evolution(run_ids)

    # Budget exhausted (loop completed without early stop)
    stop_reason = "budget_exhausted" if rounds else "budget_exhausted"
    return _build_result(goal, rounds, stop_reason)


# ---------------------------------------------------------------------------
# Offline / benchmark convenience wrapper
# ---------------------------------------------------------------------------


def run_campaign_offline(
    goal: CampaignGoal,
    space: ParameterSpace,
    sim_fn: Callable[[dict[str, Any]], dict[str, float]],
    campaign_id: str | None = None,
    convergence_config: ConvergenceConfig | None = None,
) -> CampaignResult:
    """Run a campaign in offline / benchmark mode with a simulation function.

    Instead of creating real runs and extracting KPIs from the database,
    ``sim_fn`` returns KPI values directly.  No database writes are performed
    for the runs themselves -- only the campaign loop logic executes.

    Parameters
    ----------
    goal:
        Defines what KPI to optimize, direction, target, and budget.
    space:
        Parameter space for candidate generation.
    sim_fn:
        Callable that takes a params dict and returns a dict of
        ``{kpi_name: kpi_value}``.
    campaign_id:
        Optional campaign identifier.
    convergence_config:
        Optional convergence detection configuration.

    Returns
    -------
    CampaignResult
    """
    if campaign_id is None:
        campaign_id = str(uuid.uuid4())

    rounds: list[RoundResult] = []
    all_kpi_history: list[float] = []

    logger.info(
        "Starting offline campaign %s: kpi=%s direction=%s target=%s",
        campaign_id,
        goal.objective_kpi,
        goal.direction,
        goal.target_value,
    )

    for round_number in range(goal.max_rounds):
        # --- 1. Generate candidates ---
        try:
            batch = generate_batch(
                space,
                goal.strategy,
                goal.batch_size,
                seed=round_number,
                campaign_id=campaign_id,
                kpi_name=goal.objective_kpi,
            )
        except Exception:
            logger.error(
                "Offline campaign %s round %d: generation failed",
                campaign_id,
                round_number,
                exc_info=True,
            )
            break

        # --- 2. Simulate each candidate ---
        round_kpis: list[float] = []
        run_ids: list[str] = []

        for candidate in batch.candidates:
            try:
                kpi_results = sim_fn(candidate.params)
                value = kpi_results.get(goal.objective_kpi)
                if value is not None and not math.isnan(value):
                    round_kpis.append(float(value))
                run_ids.append(f"sim-{campaign_id[:8]}-r{round_number}-c{candidate.index}")
            except Exception:
                logger.warning(
                    "Offline campaign %s round %d: sim failed for candidate %d",
                    campaign_id,
                    round_number,
                    candidate.index,
                    exc_info=True,
                )

        if not round_kpis:
            logger.warning(
                "Offline campaign %s round %d: no KPIs produced -- stopping",
                campaign_id,
                round_number,
            )
            break

        # --- 3. Evaluate ---
        all_kpi_history.extend(round_kpis)

        best_this_round, convergence = evaluate_round_pure(
            round_kpis, all_kpi_history, goal, convergence_config
        )

        round_result = RoundResult(
            round_number=round_number,
            run_ids=tuple(run_ids),
            kpi_values=tuple(round_kpis),
            best_kpi=best_this_round,
            convergence_status=convergence.status,
            timestamp=utcnow_iso(),
        )
        rounds.append(round_result)

        logger.info(
            "Offline campaign %s round %d: best=%.6f convergence=%s",
            campaign_id,
            round_number,
            best_this_round,
            convergence.status,
        )

        # --- 4. Decide ---
        action = decide_next_action(goal, rounds, convergence)
        if action != "continue":
            logger.info(
                "Offline campaign %s: stopping -- reason=%s",
                campaign_id,
                action,
            )
            return _build_result(goal, rounds, action)

    return _build_result(goal, rounds, "budget_exhausted")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_result(
    goal: CampaignGoal,
    rounds: list[RoundResult],
    stop_reason: str,
) -> CampaignResult:
    """Assemble a CampaignResult from completed rounds."""
    if not rounds:
        return CampaignResult(
            goal=goal,
            rounds=(),
            best_kpi=float("inf") if goal.direction == "minimize" else float("-inf"),
            best_round=0,
            total_runs=0,
            converged=False,
            target_reached=False,
            stop_reason=stop_reason,
        )

    # Find best KPI across all rounds
    if goal.maximize:
        best_round_result = max(rounds, key=lambda r: r.best_kpi)
    else:
        best_round_result = min(rounds, key=lambda r: r.best_kpi)

    total_runs = sum(len(r.run_ids) for r in rounds)

    return CampaignResult(
        goal=goal,
        rounds=tuple(rounds),
        best_kpi=best_round_result.best_kpi,
        best_round=best_round_result.round_number,
        total_runs=total_runs,
        converged=(stop_reason == "stop_converged"),
        target_reached=(stop_reason == "stop_target"),
        stop_reason=stop_reason,
    )


def _trigger_evolution(run_ids: list[str]) -> None:
    """Trigger evolution engine for completed runs.  Advisory -- never raises."""
    from app.services.evolution import process_review_event

    for run_id in run_ids:
        try:
            process_review_event(run_id)
        except Exception:
            logger.debug(
                "Evolution trigger skipped for run %s", run_id, exc_info=True
            )
