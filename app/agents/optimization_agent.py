"""OptimizationAgent — strategy selection and candidate generation.

Wraps existing strategy_router and optimization services to generate next
candidate points based on campaign history and observations.

Layer: L2
"""
from __future__ import annotations

import itertools
import logging
import random
import uuid
from typing import Any

from pydantic import BaseModel, Field

from app.agents.base import BaseAgent
from app.agents.observation_agent import ObservationPacket
from app.contracts.task_contract import DimensionDef, ObjectiveSpec

logger = logging.getLogger(__name__)


# ── Output Models ──────────────────────────────────────────────────────────

class CandidatePoint(BaseModel):
    """A single candidate point to evaluate in the next round."""

    point_id: str = Field(default_factory=lambda: f"cp-{uuid.uuid4().hex[:12]}")
    parameters: dict[str, Any]
    strategy_used: str
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


# ── Input Model ────────────────────────────────────────────────────────────

class OptimizationInput(BaseModel):
    """Input for OptimizationAgent."""

    campaign_id: str
    round_number: int
    observation: ObservationPacket
    kpi_history: list[float] = Field(default_factory=list)
    parameter_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Historical parameter dicts from previous rounds (parallel to kpi_history)",
    )
    dimensions: list[DimensionDef] = Field(default_factory=list)
    objective: ObjectiveSpec | None = None
    current_strategy: str = "bayesian"
    batch_size: int = Field(default=8, ge=1, le=100)


# ── Output Model ───────────────────────────────────────────────────────────

class OptimizationOutput(BaseModel):
    """Output from OptimizationAgent."""

    candidates: list[CandidatePoint] = Field(default_factory=list)
    strategy_selected: str
    strategy_rationale: str
    convergence_signal: float = Field(ge=0.0, le=1.0)
    decision_nodes: list[dict[str, Any]] = Field(default_factory=list)


# ── Agent Implementation ───────────────────────────────────────────────────

class OptimizationAgent(BaseAgent[OptimizationInput, OptimizationOutput]):
    """Generate next candidate points using strategy routing.

    Selects a strategy (bayesian, random, grid, etc.) via strategy_router,
    then generates batch_size candidate points for the next round.
    """

    name = "optimization_agent"
    description = "Strategy selection and candidate point generation"
    layer = "L2"

    def validate_input(self, input_data: OptimizationInput) -> list[str]:
        errors: list[str] = []
        if not input_data.campaign_id:
            errors.append("campaign_id is required")
        if not input_data.observation:
            errors.append("observation is required")
        if input_data.batch_size < 1:
            errors.append("batch_size must be >= 1")
        return errors

    async def process(self, input_data: OptimizationInput) -> OptimizationOutput:
        campaign_id = input_data.campaign_id
        round_number = input_data.round_number
        observation = input_data.observation
        dimensions = input_data.dimensions
        objective = input_data.objective
        batch_size = input_data.batch_size

        # ── Phase 1: Build campaign snapshot for strategy selection ────────
        from app.services.strategy_models import CampaignSnapshot

        direction = objective.direction if objective else "minimize"
        kpi_history = tuple(input_data.kpi_history)
        param_history: tuple[dict[str, Any], ...] = tuple(input_data.parameter_history)

        best_so_far: float | None = None
        if kpi_history:
            best_so_far = min(kpi_history) if direction == "minimize" else max(kpi_history)

        # last_batch_kpis / last_batch_params: use last observed round if available
        last_batch_kpis: tuple[float, ...] = (
            (kpi_history[-1],) if kpi_history else ()
        )
        last_batch_params: tuple[dict[str, Any], ...] = (
            (param_history[-1],) if param_history else ()
        )

        snapshot = CampaignSnapshot(
            round_number=round_number,
            max_rounds=100,  # placeholder; caller may pass via metadata in future
            n_observations=len(kpi_history),
            n_dimensions=len(dimensions),
            has_categorical=any(d.param_type == "categorical" for d in dimensions),
            has_log_scale=any(d.log_scale for d in dimensions),
            kpi_history=kpi_history,
            direction=direction,
            last_batch_kpis=last_batch_kpis,
            last_batch_params=last_batch_params,
            best_kpi_so_far=best_so_far,
            all_params=param_history,
            all_kpis=kpi_history,
            qc_fail_rate=0.0 if observation.qc_passed else 0.5,
        )

        # ── Phase 2: Select strategy via router ─────────────────────────────
        strategy_selected = input_data.current_strategy
        strategy_rationale = f"Using {strategy_selected} strategy for round {round_number}"

        try:
            from app.services.strategy_router import StrategyRouter
            router = StrategyRouter()
            decision = router.select_strategy(snapshot, campaign_id)

            if hasattr(decision, "backend_name"):
                strategy_selected = decision.backend_name
            if hasattr(decision, "reason"):
                strategy_rationale = decision.reason

        except Exception as exc:
            logger.debug(
                "optimization_agent: strategy router failed, using fallback: %s",
                exc,
            )

        # ── Phase 3: Generate candidate points ──────────────────────────────
        candidates: list[CandidatePoint] = []

        try:
            candidates = self._generate_candidates(
                campaign_id=campaign_id,
                round_number=round_number,
                strategy=strategy_selected,
                dimensions=dimensions,
                batch_size=batch_size,
                kpi_history=input_data.kpi_history,
                objective=objective,
            )
        except Exception as exc:
            logger.error(
                "optimization_agent: candidate generation failed: %s",
                exc, exc_info=True,
            )
            # Return minimal fallback candidates
            candidates = self._generate_random_candidates(
                dimensions, batch_size
            )

        # ── Phase 4: Compute convergence signal ─────────────────────────────
        convergence_signal = self._compute_convergence_signal(
            input_data.kpi_history, observation.qc_passed
        )

        logger.info(
            "optimization_agent: campaign=%s round=%d strategy=%s candidates=%d convergence=%.2f",
            campaign_id, round_number, strategy_selected,
            len(candidates), convergence_signal,
            extra={"campaign_id": campaign_id},
        )

        return OptimizationOutput(
            candidates=candidates,
            strategy_selected=strategy_selected,
            strategy_rationale=strategy_rationale,
            convergence_signal=convergence_signal,
            decision_nodes=[],
        )

    # ── Candidate generation ───────────────────────────────────────────────

    def _generate_candidates(
        self,
        campaign_id: str,
        round_number: int,
        strategy: str,
        dimensions: list[DimensionDef],
        batch_size: int,
        kpi_history: list[float],
        objective: ObjectiveSpec | None = None,
    ) -> list[CandidatePoint]:
        """Generate candidate points using the selected strategy."""

        if strategy == "random":
            return self._generate_random_candidates(dimensions, batch_size)
        elif strategy == "grid":
            return self._generate_grid_candidates(dimensions, batch_size)
        elif strategy == "bayesian":
            return self._generate_bayesian_candidates(
                dimensions, batch_size, kpi_history, objective
            )
        elif strategy == "lhs":
            return self._generate_lhs_candidates(dimensions, batch_size)
        else:
            logger.warning(
                "Unknown strategy %s, falling back to random", strategy
            )
            return self._generate_random_candidates(dimensions, batch_size)

    def _generate_random_candidates(
        self,
        dimensions: list[DimensionDef],
        batch_size: int,
    ) -> list[CandidatePoint]:
        """Generate random candidate points."""
        candidates: list[CandidatePoint] = []

        for _ in range(batch_size):
            params: dict[str, Any] = {}

            for dim in dimensions:
                if dim.param_type == "number":
                    if dim.min_value is not None and dim.max_value is not None:
                        val = random.uniform(dim.min_value, dim.max_value)
                    else:
                        val = random.uniform(0, 1)
                elif dim.param_type == "integer":
                    min_v = int(dim.min_value or 0)
                    max_v = int(dim.max_value or 100)
                    val = random.randint(min_v, max_v)
                elif dim.param_type == "categorical":
                    val = random.choice(dim.choices or ["default"])
                elif dim.param_type == "boolean":
                    val = random.choice([True, False])
                else:
                    val = 0

                params[dim.param_name] = val

            candidates.append(CandidatePoint(
                parameters=params,
                strategy_used="random",
                rationale="Exploratory random sampling",
                confidence=0.5,
            ))

        return candidates

    def _generate_grid_candidates(
        self,
        dimensions: list[DimensionDef],
        batch_size: int,
    ) -> list[CandidatePoint]:
        """Generate grid-based candidate points (uniform sampling)."""
        candidates: list[CandidatePoint] = []

        # Create grids for each dimension
        grids: list[list[Any]] = []
        for dim in dimensions:
            if dim.param_type == "number":
                min_v = dim.min_value or 0
                max_v = dim.max_value or 1
                step = (max_v - min_v) / max(batch_size - 1, 1)
                grid = [min_v + i * step for i in range(batch_size)]
            elif dim.param_type == "integer":
                min_v = int(dim.min_value or 0)
                max_v = int(dim.max_value or 100)
                grid = list(range(min_v, min(max_v + 1, min_v + batch_size)))
            elif dim.param_type == "categorical":
                grid = dim.choices or ["default"]
            else:
                grid = [0]

            grids.append(grid)

        # Generate combinations
        for combo in itertools.islice(
            itertools.product(*grids), batch_size
        ):
            params = {}
            for dim, val in zip(dimensions, combo):
                params[dim.param_name] = val

            candidates.append(CandidatePoint(
                parameters=params,
                strategy_used="grid",
                rationale="Grid-based uniform sampling",
                confidence=0.6,
            ))

        return candidates

    def _generate_lhs_candidates(
        self,
        dimensions: list[DimensionDef],
        batch_size: int,
    ) -> list[CandidatePoint]:
        """Generate Latin Hypercube Sampling candidates."""
        candidates: list[CandidatePoint] = []

        # Simple LHS: divide each dimension into batch_size bins,
        # then randomly sample one value per bin
        for i in range(batch_size):
            params: dict[str, Any] = {}

            for dim in dimensions:
                if dim.param_type == "number":
                    min_v = dim.min_value or 0
                    max_v = dim.max_value or 1
                    # i-th bin
                    bin_min = min_v + (max_v - min_v) * (i / batch_size)
                    bin_max = min_v + (max_v - min_v) * ((i + 1) / batch_size)
                    val = random.uniform(bin_min, bin_max)
                elif dim.param_type == "integer":
                    min_v = int(dim.min_value or 0)
                    max_v = int(dim.max_value or 100)
                    bin_min = int(min_v + (max_v - min_v) * (i / batch_size))
                    bin_max = int(min_v + (max_v - min_v) * ((i + 1) / batch_size))
                    bin_max = max(bin_min, bin_max)  # guard against rounding collapse
                    val = random.randint(bin_min, bin_max)
                elif dim.param_type == "categorical":
                    val = random.choice(dim.choices or ["default"])
                else:
                    val = 0

                params[dim.param_name] = val

            candidates.append(CandidatePoint(
                parameters=params,
                strategy_used="lhs",
                rationale="Latin Hypercube Sampling for balanced exploration",
                confidence=0.7,
            ))

        return candidates

    def _generate_bayesian_candidates(
        self,
        dimensions: list[DimensionDef],
        batch_size: int,
        kpi_history: list[float],
        objective: ObjectiveSpec | None = None,
    ) -> list[CandidatePoint]:
        """Generate Bayesian optimization candidates (falls back to LHS if no history).

        With sparse history, defaults to LHS to avoid overfitting.
        """
        if len(kpi_history) < 3:
            logger.debug(
                "optimization_agent: insufficient history (%d < 3) for bayesian, using LHS",
                len(kpi_history),
            )
            return self._generate_lhs_candidates(dimensions, batch_size)

        # Try to use a real Bayesian optimizer if available
        try:
            from app.services.bayesian_opt import BayesianOptimizer

            optimizer = BayesianOptimizer(dimensions=dimensions)
            candidates = optimizer.generate_candidates(
                kpi_history=kpi_history,
                batch_size=batch_size,
                direction=objective.direction if objective else "minimize",
            )

            return [
                CandidatePoint(
                    parameters=cand,
                    strategy_used="bayesian",
                    rationale="Bayesian optimization with Gaussian Process",
                    confidence=0.8,
                )
                for cand in candidates
            ]

        except (ImportError, Exception):
            logger.debug(
                "optimization_agent: bayesian optimizer unavailable, using LHS fallback"
            )
            return self._generate_lhs_candidates(dimensions, batch_size)

    # ── Convergence signal ─────────────────────────────────────────────────

    @staticmethod
    def _compute_convergence_signal(
        kpi_history: list[float],
        qc_passed: bool,
    ) -> float:
        """Compute a 0-1 convergence signal based on KPI trend and QC status.

        0.0 = exploring, 1.0 = converged
        """
        if not kpi_history:
            return 0.0

        if len(kpi_history) < 3:
            return 0.2  # Early stage

        # Check if improvements are slowing down
        recent = kpi_history[-5:]  # Last 5 rounds
        recent_improvement = (max(recent) - min(recent)) / (abs(min(recent)) + 1e-9)

        if recent_improvement < 0.01:  # <1% improvement in last 5 rounds
            convergence = 0.8
        elif recent_improvement < 0.05:  # <5% improvement
            convergence = 0.6
        else:
            convergence = 0.3

        # QC failures reduce convergence confidence
        if not qc_passed:
            convergence *= 0.7

        return min(1.0, max(0.0, convergence))
