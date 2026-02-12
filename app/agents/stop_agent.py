"""Stop/Continue Decision Agent -- L0 layer.

Wraps convergence detection and campaign loop decision logic.
Decides whether to continue, stop, or escalate.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.agents.base import BaseAgent


class StopInput(BaseModel):
    """Input for stop/continue decision."""
    kpi_history: list[float]
    current_round: int
    max_rounds: int
    target_value: float | None = None
    direction: Literal["minimize", "maximize"] = "minimize"
    plateau_threshold: float = 0.01
    budget_limit_runs: int | None = None
    total_runs_so_far: int = 0


class StopOutput(BaseModel):
    """Output from stop/continue decision."""
    decision: Literal["continue", "stop_target", "stop_budget", "stop_converged", "stop_diverging"]
    confidence: float = 0.0
    convergence_status: str = "insufficient_data"
    best_kpi: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class StopAgent(BaseAgent[StopInput, StopOutput]):
    name = "stop_agent"
    description = "Stop/continue campaign decision"
    layer = "L0"

    def validate_input(self, input_data: StopInput) -> list[str]:
        errors: list[str] = []
        if input_data.max_rounds < 1:
            errors.append("max_rounds must be >= 1")
        return errors

    async def process(self, input_data: StopInput) -> StopOutput:
        from app.services.convergence import ConvergenceConfig, detect_convergence

        maximize = input_data.direction == "maximize"

        config = ConvergenceConfig(
            plateau_threshold=input_data.plateau_threshold,
        )

        convergence = detect_convergence(
            input_data.kpi_history,
            config=config,
            maximize=maximize,
        )

        # Compute best KPI
        best_kpi: float | None = None
        if input_data.kpi_history:
            best_kpi = max(input_data.kpi_history) if maximize else min(input_data.kpi_history)

        # Decision logic
        decision: str = "continue"

        # Check target
        if best_kpi is not None and input_data.target_value is not None:
            if maximize and best_kpi >= input_data.target_value:
                decision = "stop_target"
            elif not maximize and best_kpi <= input_data.target_value:
                decision = "stop_target"

        # Check budget
        if decision == "continue" and input_data.current_round >= input_data.max_rounds:
            decision = "stop_budget"

        if decision == "continue" and input_data.budget_limit_runs is not None:
            if input_data.total_runs_so_far >= input_data.budget_limit_runs:
                decision = "stop_budget"

        # Check convergence
        # IMPORTANT: Only allow convergence-based stopping if:
        #   1. No target value is set (exploratory optimization), OR
        #   2. Target value is set AND already achieved
        if decision == "continue" and convergence.status == "plateau" and convergence.confidence > 0.7:
            # Check if we have a target and whether it's been met
            if input_data.target_value is not None and best_kpi is not None:
                target_met = (
                    (maximize and best_kpi >= input_data.target_value) or
                    (not maximize and best_kpi <= input_data.target_value)
                )
                if target_met:
                    # Target achieved AND converged → can stop
                    decision = "stop_converged"
                # else: target NOT met → continue optimizing despite convergence
            else:
                # No target set → allow convergence-based stopping
                decision = "stop_converged"

        if decision == "continue" and convergence.status == "diverging" and convergence.confidence > 0.8:
            decision = "stop_diverging"

        return StopOutput(
            decision=decision,
            confidence=convergence.confidence,
            convergence_status=convergence.status,
            best_kpi=best_kpi,
            details=convergence.details,
        )
