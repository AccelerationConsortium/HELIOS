"""Campaign Planner Agent — L2 planning layer.

Produces a multi-round campaign plan from a TaskContract.
Plans resource requirements, parameter batches for each round,
and contingency strategies.
"""
from __future__ import annotations

import uuid
import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.agents.base import BaseAgent, DecisionNode

logger = logging.getLogger(__name__)


class PlannerInput(BaseModel):
    """Input for campaign planning."""
    # From TaskContract
    contract_id: str
    objective_kpi: str
    direction: Literal["minimize", "maximize"]
    max_rounds: int
    batch_size: int
    strategy: str = "lhs"
    target_value: float | None = None

    # Parameter space
    dimensions: list[dict[str, Any]]
    protocol_template: dict[str, Any]

    # Optional: pre-existing history for adaptive planning
    kpi_history: list[float] = Field(default_factory=list)
    completed_rounds: int = 0


class PlannedRound(BaseModel):
    """Plan for a single experiment round."""
    round_number: int
    strategy: str
    batch_size: int
    resource_estimate: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""


class PlannerOutput(BaseModel):
    """Output from campaign planning."""
    plan_id: str
    contract_id: str
    planned_rounds: list[PlannedRound]
    total_planned_runs: int
    strategy_schedule: dict[int, str] = Field(default_factory=dict)
    estimated_tip_usage: int = 0
    notes: str = ""
    decision_nodes: list[dict[str, Any]] = Field(default_factory=list)


class PlannerAgent(BaseAgent[PlannerInput, PlannerOutput]):
    name = "planner_agent"
    description = "Multi-round campaign planning"
    layer = "L2"

    def validate_input(self, input_data: PlannerInput) -> list[str]:
        errors = []
        if input_data.max_rounds < 1:
            errors.append("max_rounds must be >= 1")
        if input_data.batch_size < 1:
            errors.append("batch_size must be >= 1")
        if not input_data.dimensions:
            errors.append("At least one dimension required")
        if not input_data.objective_kpi:
            errors.append("objective_kpi is required")
        return errors

    async def process(self, input_data: PlannerInput) -> PlannerOutput:
        plan_id = f"cp-{uuid.uuid4().hex[:12]}"

        remaining_rounds = input_data.max_rounds - input_data.completed_rounds

        # Decision node 1: budget check
        budget_node = DecisionNode(
            id="budget_check",
            label="Budget check",
            options=["Stop (no rounds remaining)", "Proceed with planning"],
            selected="Stop (no rounds remaining)" if remaining_rounds <= 0 else "Proceed with planning",
            reason=f"max_rounds={input_data.max_rounds}, completed={input_data.completed_rounds}",
            outcome=f"{remaining_rounds} round(s) remaining",
        )

        if remaining_rounds <= 0:
            return PlannerOutput(
                plan_id=plan_id,
                contract_id=input_data.contract_id,
                planned_rounds=[],
                total_planned_runs=0,
                notes="No rounds remaining",
                decision_nodes=[budget_node.to_dict()],
            )

        # Build strategy schedule:
        # - First 20% of rounds: exploration (LHS or random)
        # - Middle 60%: exploitation (bayesian or prior_guided)
        # - Last 20%: refinement (bayesian with tighter exploration)
        # When strategy="adaptive", uses the intelligent selector.
        strategy_schedule = _build_strategy_schedule(
            base_strategy=input_data.strategy,
            total_rounds=remaining_rounds,
            has_history=len(input_data.kpi_history) > 0,
            dimensions=input_data.dimensions,
        )

        # Decision node 2: strategy mode
        is_adaptive = input_data.strategy == "adaptive"
        strategy_node = DecisionNode(
            id="strategy_mode",
            label="Strategy schedule mode",
            options=["Adaptive (intelligent per-round selector)", "Phase-based (20% explore / 60% exploit / 20% refine)"],
            selected="Adaptive (intelligent per-round selector)" if is_adaptive else "Phase-based (20% explore / 60% exploit / 20% refine)",
            reason=f"base_strategy='{input_data.strategy}', has_history={len(input_data.kpi_history) > 0}, dims={len(input_data.dimensions)}",
        )

        planned_rounds = []
        round_strategy_nodes: list[DecisionNode] = []
        _all_strategies = sorted({input_data.strategy, "lhs", "bayesian", "prior_guided", "random"})

        for i in range(remaining_rounds):
            round_num = input_data.completed_rounds + i + 1
            strategy = strategy_schedule.get(round_num, input_data.strategy)

            # Estimate resources per round
            n_aspirates = len([
                d for d in input_data.dimensions
                if d.get("primitive", "").startswith("robot.")
            ])
            tips_per_round = max(input_data.batch_size, input_data.batch_size * max(n_aspirates, 1))

            planned_rounds.append(PlannedRound(
                round_number=round_num,
                strategy=strategy,
                batch_size=input_data.batch_size,
                resource_estimate={
                    "tips_needed": tips_per_round,
                    "estimated_duration_minutes": input_data.batch_size * 5,  # rough: 5 min per run
                    "instruments": _infer_instruments(input_data.dimensions),
                },
                notes=_round_notes(round_num, remaining_rounds, strategy),
            ))

            round_pct = round_num / remaining_rounds * 100
            if round_pct <= 20:
                phase = "exploration"
            elif round_pct <= 80:
                phase = "exploitation"
            else:
                phase = "refinement"
            round_strategy_nodes.append(DecisionNode(
                id=f"round_{round_num}_strategy",
                label=f"Round {round_num} strategy",
                options=_all_strategies,
                selected=strategy,
                reason=f"{phase} phase ({round_pct:.0f}% of campaign), has_history={len(input_data.kpi_history) > 0}",
            ))

        total_runs = sum(r.batch_size for r in planned_rounds)
        total_tips = sum(r.resource_estimate.get("tips_needed", 0) for r in planned_rounds)

        strategy_node_with_children = DecisionNode(
            id=strategy_node.id,
            label=strategy_node.label,
            options=strategy_node.options,
            selected=strategy_node.selected,
            reason=strategy_node.reason,
            outcome=f"{len(planned_rounds)} rounds scheduled",
            children=tuple(round_strategy_nodes),
        )

        return PlannerOutput(
            plan_id=plan_id,
            contract_id=input_data.contract_id,
            planned_rounds=planned_rounds,
            total_planned_runs=total_runs,
            strategy_schedule=strategy_schedule,
            estimated_tip_usage=total_tips,
            notes=f"Planned {len(planned_rounds)} rounds, {total_runs} total runs",
            decision_nodes=[budget_node.to_dict(), strategy_node_with_children.to_dict()],
        )


def _build_strategy_schedule(
    base_strategy: str,
    total_rounds: int,
    has_history: bool,
    *,
    dimensions: list[dict[str, Any]] | None = None,
) -> dict[int, str]:
    """Build a strategy schedule that transitions from exploration to exploitation.

    When ``base_strategy == "adaptive"``, uses the intelligent strategy
    selector to pick the best backend for each round based on phase,
    dimensionality, and available optimization packages.

    Otherwise falls back to the simple rule-based schedule.
    """
    schedule: dict[int, str] = {}

    if total_rounds <= 3:
        # Too few rounds to transition — use base strategy throughout
        return schedule

    if base_strategy == "adaptive":
        return _build_adaptive_schedule(
            total_rounds, has_history, dimensions=dimensions or [],
        )

    explore_end = max(1, int(total_rounds * 0.2))
    exploit_end = max(explore_end + 1, int(total_rounds * 0.8))

    for i in range(1, total_rounds + 1):
        if i <= explore_end:
            # Exploration phase
            if has_history:
                schedule[i] = "prior_guided"
            else:
                schedule[i] = base_strategy if base_strategy in ("lhs", "random") else "lhs"
        elif i <= exploit_end:
            # Exploitation phase
            schedule[i] = "bayesian" if has_history or i > 3 else "prior_guided"
        else:
            # Refinement phase
            schedule[i] = "bayesian"

    return schedule


def _build_adaptive_schedule(
    total_rounds: int,
    has_history: bool,
    *,
    dimensions: list[dict[str, Any]],
) -> dict[int, str]:
    """Use the strategy selector to build an intelligent per-round schedule.

    Pre-computes the strategy for each round assuming no KPI history
    (the orchestrator can override at runtime with real convergence data).
    """
    try:
        from app.services.strategy_selector import (
            CampaignSnapshot,
            select_strategy,
        )
        from app.services.optimization_backends import list_backends

        available = list_backends()
        n_dims = len(dimensions)
        has_categorical = any(
            d.get("choices") is not None for d in dimensions
        )
        has_log_scale = any(d.get("log_scale", False) for d in dimensions)

        schedule: dict[int, str] = {}
        for i in range(1, total_rounds + 1):
            snapshot = CampaignSnapshot(
                round_number=i,
                max_rounds=total_rounds,
                n_observations=max(0, (i - 1) * 5) if has_history else 0,
                n_dimensions=n_dims,
                has_categorical=has_categorical,
                has_log_scale=has_log_scale,
                kpi_history=(),
                direction="maximize",
                available_backends=available,
            )
            decision = select_strategy(snapshot)

            # Map backend names back to candidate_gen strategy names
            _BACKEND_TO_STRATEGY = {
                "lhs": "lhs",
                "random_sampling": "random",
                "built_in": "bayesian",
                "optuna_tpe": "adaptive",
                "optuna_cmaes": "adaptive",
                "scipy_de": "adaptive",
                "pymoo_nsga2": "adaptive",
            }
            schedule[i] = _BACKEND_TO_STRATEGY.get(
                decision.backend_name, "adaptive"
            )

        return schedule

    except Exception:
        logger.warning(
            "Adaptive schedule failed, falling back to simple schedule",
            exc_info=True,
        )
        return _build_strategy_schedule(
            "lhs", total_rounds, has_history, dimensions=dimensions,
        )


def _infer_instruments(dimensions: list[dict[str, Any]]) -> list[str]:
    """Infer which instruments are needed from dimension primitives."""
    instruments = set()
    for d in dimensions:
        primitive = d.get("primitive", "")
        if primitive.startswith("robot."):
            instruments.add("ot2")
        elif primitive.startswith("plc."):
            instruments.add("plc")
        elif primitive.startswith("relay."):
            instruments.add("relay")
        elif primitive.startswith("squidstat."):
            instruments.add("squidstat")
        elif primitive == "heat":
            instruments.add("furnace")
    return sorted(instruments)


def _round_notes(round_num: int, total: int, strategy: str) -> str:
    """Generate helpful notes for each round."""
    pct = round_num / total * 100
    if pct <= 20:
        return f"Exploration phase ({strategy}): space-filling sampling"
    elif pct <= 80:
        return f"Exploitation phase ({strategy}): focusing on promising regions"
    else:
        return f"Refinement phase ({strategy}): fine-tuning near optimum"
