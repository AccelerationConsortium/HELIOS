"""Analyzer Agent — per-round KPI analysis and diagnostics.

Runs once per round after all candidates have been executed.  Combines
existing services (strategy_diagnostics, convergence) into a single agent
that generates a human-readable narrative for scientists.

Emits:
  - ``agent_thinking``  — one message per analysis phase
  - ``agent_result``    — AnalyzerOutput (with narrative) at the end
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from app.agents.base import BaseAgent, DecisionNode
from app.services.convergence import ConvergenceStatus, detect_convergence
from app.services.strategy_diagnostics import compute_diagnostics
from app.services.strategy_models import CampaignSnapshot, DiagnosticSignals

logger = logging.getLogger(__name__)

# Type alias for the SSE emit callback passed in from the orchestrator
EmitCallback = Callable[[dict[str, Any]], None]


# ---------------------------------------------------------------------------
# I/O models
# ---------------------------------------------------------------------------


class AnalyzerInput(BaseModel):
    """Input for the per-round Analyzer agent."""

    round_number: int
    direction: str  # "maximize" | "minimize"
    kpi_name: str

    # Current round results
    round_kpis: list[float] = Field(default_factory=list)
    round_params: list[dict[str, Any]] = Field(default_factory=list)

    # Full campaign history (for diagnostics + convergence)
    all_kpis: list[float] = Field(default_factory=list)
    all_params: list[dict[str, Any]] = Field(default_factory=list)
    all_rounds: list[int] = Field(default_factory=list)

    # Campaign context
    qc_fail_rate: float = 0.0
    max_rounds: int = 10
    n_dimensions: int = 1
    has_categorical: bool = False
    has_log_scale: bool = False
    step_history: list[dict[str, Any]] = Field(default_factory=list)

    # Optional SSE emit callback — set by orchestrator before calling run()
    emit: EmitCallback | None = Field(default=None, exclude=True)

    model_config = {"arbitrary_types_allowed": True}


class AnalyzerOutput(BaseModel):
    """Output from the per-round Analyzer agent."""

    round_number: int
    # Rich diagnostic signals from strategy_diagnostics
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    # Convergence status from convergence.py
    convergence_status: str = "insufficient_data"
    convergence_confidence: float = 0.0
    # Round-level statistics
    round_mean_kpi: float = 0.0
    round_best_kpi: float = 0.0
    round_cv: float = 0.0
    improvement_vs_last: float | None = None
    aleatoric_std: float = 0.0
    # Human-readable 1-2 sentence summary
    narrative: str = ""
    decision_nodes: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((v - m) ** 2 for v in values) / len(values)
    return math.sqrt(max(variance, 0.0))


def _cv(values: list[float]) -> float:
    """Coefficient of variation (std / |mean|)."""
    if not values:
        return 0.0
    m = _mean(values)
    s = _std(values)
    return s / abs(m) if abs(m) > 1e-12 else 0.0


def _best(values: list[float], direction: str) -> float:
    if not values:
        return float("nan")
    return max(values) if direction == "maximize" else min(values)


def _improvement_pct(current: float, previous: float, direction: str) -> float:
    """Signed improvement percentage vs previous round best."""
    if abs(previous) < 1e-12:
        return 0.0
    if direction == "maximize":
        return (current - previous) / abs(previous) * 100
    else:
        return (previous - current) / abs(previous) * 100  # lower is better


# ---------------------------------------------------------------------------
# Narrative builder
# ---------------------------------------------------------------------------


def _build_narrative(
    round_number: int,
    kpi_name: str,
    round_best: float,
    improvement_pct: float | None,
    diagnostics: DiagnosticSignals,
    convergence: ConvergenceStatus,
    aleatoric_std: float,
) -> str:
    """Build a 1-2 sentence human-readable summary of the round."""
    parts: list[str] = []

    # Part 1: performance summary
    best_str = f"{round_best:.4g}"
    if improvement_pct is not None:
        sign = "↑" if improvement_pct >= 0 else "↓"
        parts.append(
            f"Round {round_number}: best {kpi_name} = {best_str} "
            f"({sign}{abs(improvement_pct):.1f}% vs R{round_number - 1})."
        )
    else:
        parts.append(f"Round {round_number}: best {kpi_name} = {best_str} (first round).")

    # Part 2: phase + convergence
    phase_tokens: list[str] = []

    conv_status = convergence.status
    conv_conf = convergence.confidence

    if conv_status == "insufficient_data":
        phase_tokens.append("gathering data")
    elif conv_status == "improving":
        phase_tokens.append("still improving")
    elif conv_status == "plateau":
        phase_tokens.append("plateau detected")
    elif conv_status == "diverging":
        phase_tokens.append("diverging — check noise")

    if conv_conf > 0:
        phase_tokens.append(f"conf={conv_conf:.2f}")

    # Model uncertainty token
    if diagnostics.model_uncertainty is not None:
        uncertainty_str = f"σ={diagnostics.model_uncertainty:.3g}"
        phase_tokens.append(uncertainty_str)

    # Phase label (epistemic vs exploitation)
    cov = diagnostics.space_coverage
    if cov < 0.3:
        phase_tokens.append("exploration phase")
    elif diagnostics.improvement_velocity is not None and diagnostics.improvement_velocity > 0.02:
        phase_tokens.append("exploitation phase")

    if phase_tokens:
        parts.append(" ".join(phase_tokens).capitalize() + ".")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Agent implementation
# ---------------------------------------------------------------------------


class AnalyzerAgent(BaseAgent[AnalyzerInput, AnalyzerOutput]):
    """Per-round campaign analyzer.

    Builds a CampaignSnapshot from the current campaign state, computes
    rich DiagnosticSignals and ConvergenceStatus, and distills them into
    a human-readable narrative for the scientist.
    """

    name = "analyzer_agent"
    description = "Per-round KPI diagnostics, convergence, and narrative generation"
    layer = "L2"

    def validate_input(self, input_data: AnalyzerInput) -> list[str]:
        errors: list[str] = []
        if input_data.round_number < 1:
            errors.append("round_number must be >= 1")
        if input_data.direction not in ("maximize", "minimize"):
            errors.append("direction must be 'maximize' or 'minimize'")
        return errors

    async def process(self, input_data: AnalyzerInput) -> AnalyzerOutput:  # noqa: C901
        emit = input_data.emit

        # ── Phase 1: Round-level statistics ───────────────────────────────
        if emit:
            emit({
                "type": "agent_thinking",
                "agent": "analyzer",
                "round": input_data.round_number,
                "message": f"Computing round {input_data.round_number} statistics...",
            })

        round_mean = _mean(input_data.round_kpis)
        round_best = _best(input_data.round_kpis, input_data.direction)
        round_cv_val = _cv(input_data.round_kpis)
        aleatoric_std = _std(input_data.round_kpis)

        # Improvement vs. last round
        improvement: float | None = None
        if input_data.round_number > 1 and len(input_data.all_kpis) > len(input_data.round_kpis):
            prev_kpis = input_data.all_kpis[: -len(input_data.round_kpis)]
            if prev_kpis:
                prev_best = _best(prev_kpis, input_data.direction)
                if not math.isnan(prev_best) and not math.isnan(round_best):
                    improvement = _improvement_pct(round_best, prev_best, input_data.direction)

        if emit:
            emit({
                "type": "agent_thinking",
                "agent": "analyzer",
                "round": input_data.round_number,
                "message": (
                    f"  Round stats: mean={round_mean:.4g}, best={round_best:.4g}, "
                    f"CV={round_cv_val:.3f}, σ={aleatoric_std:.4g}"
                ),
            })

        # ── Phase 2: Build CampaignSnapshot ───────────────────────────────
        if emit:
            emit({
                "type": "agent_thinking",
                "agent": "analyzer",
                "round": input_data.round_number,
                "message": "Building CampaignSnapshot for diagnostic signals...",
            })

        # Determine best so far across all rounds
        best_so_far: float | None = None
        if input_data.all_kpis:
            best_so_far = _best(input_data.all_kpis, input_data.direction)

        snapshot = CampaignSnapshot(
            round_number=input_data.round_number,
            max_rounds=input_data.max_rounds,
            n_observations=len(input_data.all_kpis),
            n_dimensions=max(input_data.n_dimensions, 1),
            has_categorical=input_data.has_categorical,
            has_log_scale=input_data.has_log_scale,
            kpi_history=tuple(input_data.all_kpis),
            direction=input_data.direction,
            last_batch_kpis=tuple(input_data.round_kpis),
            last_batch_params=tuple(input_data.round_params),
            best_kpi_so_far=best_so_far,
            all_params=tuple(input_data.all_params),
            all_kpis=tuple(input_data.all_kpis),
            qc_fail_rate=input_data.qc_fail_rate,
        )

        # ── Phase 3: Compute diagnostic signals ───────────────────────────
        if emit:
            emit({
                "type": "agent_thinking",
                "agent": "analyzer",
                "round": input_data.round_number,
                "message": "Computing diagnostic signals (epistemic / aleatoric / saturation)...",
            })

        # Decision node 1: improvement trend (built now, used in final return)
        if improvement is not None:
            trend_selected = f"Δ={improvement:+.1f}% vs R{input_data.round_number - 1}"
            trend_reason = f"round_best={round_best:.4g}, prev_best computed from {len(input_data.all_kpis) - len(input_data.round_kpis)} prior observations"
        else:
            trend_selected = "First round (no comparison)"
            trend_reason = "round_number=1 or insufficient prior history"
        improvement_node = DecisionNode(
            id="improvement_trend",
            label="Improvement vs last round",
            options=["First round (no comparison)", f"Δ={improvement:+.1f}% vs R{input_data.round_number - 1}"] if improvement is not None else ["First round (no comparison)", "No prior data"],
            selected=trend_selected,
            reason=trend_reason,
            outcome=f"round_best={round_best:.4g}",
        )

        try:
            diagnostics: DiagnosticSignals = compute_diagnostics(snapshot)
        except Exception as exc:
            logger.warning("Diagnostics computation failed: %s", exc, exc_info=True)
            diag_fail_node = DecisionNode(
                id="diagnostics",
                label="Diagnostic signals",
                options=["OK", "Unavailable (exception)"],
                selected="Unavailable (exception)",
                reason=str(exc),
                outcome="Returning minimal output",
            )
            # Return a minimal output rather than propagating the failure
            return AnalyzerOutput(
                round_number=input_data.round_number,
                round_mean_kpi=round_mean,
                round_best_kpi=round_best,
                round_cv=round_cv_val,
                improvement_vs_last=improvement,
                aleatoric_std=aleatoric_std,
                narrative=f"Round {input_data.round_number}: diagnostics unavailable.",
                decision_nodes=[improvement_node.to_dict(), diag_fail_node.to_dict()],
            )

        if emit:
            cov_str = f"{diagnostics.space_coverage:.2f}"
            unc_str = (
                f"{diagnostics.model_uncertainty:.4g}"
                if diagnostics.model_uncertainty is not None
                else "n/a"
            )
            emit({
                "type": "agent_thinking",
                "agent": "analyzer",
                "round": input_data.round_number,
                "message": (
                    f"  Coverage={cov_str}, uncertainty={unc_str}, "
                    f"conv={diagnostics.convergence_status}"
                ),
            })

        # ── Phase 4: Convergence detection ────────────────────────────────
        if emit:
            emit({
                "type": "agent_thinking",
                "agent": "analyzer",
                "round": input_data.round_number,
                "message": "Running convergence detector...",
            })

        try:
            convergence: ConvergenceStatus = detect_convergence(
                list(input_data.all_kpis),
                maximize=input_data.direction == "maximize",
            )
        except Exception as exc:
            logger.warning("Convergence detection failed: %s", exc, exc_info=True)
            from app.services.convergence import ConvergenceStatus as CS
            convergence = CS(
                status="insufficient_data",
                confidence=0.0,
                details={"error": str(exc)},
            )

        if emit:
            emit({
                "type": "agent_thinking",
                "agent": "analyzer",
                "round": input_data.round_number,
                "message": (
                    f"  Convergence: {convergence.status} "
                    f"(confidence={convergence.confidence:.2f})"
                ),
            })

        # ── Phase 5: Build narrative ───────────────────────────────────────
        if emit:
            emit({
                "type": "agent_thinking",
                "agent": "analyzer",
                "round": input_data.round_number,
                "message": "Generating narrative summary...",
            })

        narrative = _build_narrative(
            round_number=input_data.round_number,
            kpi_name=input_data.kpi_name,
            round_best=round_best,
            improvement_pct=improvement,
            diagnostics=diagnostics,
            convergence=convergence,
            aleatoric_std=aleatoric_std,
        )

        if emit:
            emit({
                "type": "agent_thinking",
                "agent": "analyzer",
                "round": input_data.round_number,
                "message": f"  → {narrative}",
            })

        # Decision node 2: diagnostics status
        diag_ok_node = DecisionNode(
            id="diagnostics",
            label="Diagnostic signals",
            options=["OK", "Unavailable (exception)"],
            selected="OK",
            reason=f"coverage={diagnostics.space_coverage:.2f}, uncertainty={diagnostics.model_uncertainty}",
        )

        # Decision node 3: convergence
        conv_node = DecisionNode(
            id="convergence",
            label="Convergence status",
            options=["insufficient_data", "improving", "plateau", "diverging"],
            selected=convergence.status,
            reason=f"confidence={convergence.confidence:.2f}, {len(input_data.all_kpis)} observations",
            outcome=f"conf={convergence.confidence:.2f}",
        )

        # Decision node 4: campaign phase
        cov = diagnostics.space_coverage
        vel = diagnostics.improvement_velocity
        if cov < 0.3:
            phase_selected = "Exploration"
            phase_reason = f"space_coverage={cov:.2f} < 0.3"
        elif vel is not None and vel > 0.02:
            phase_selected = "Exploitation"
            phase_reason = f"improvement_velocity={vel:.4f} > 0.02"
        else:
            phase_selected = "Transition"
            phase_reason = f"coverage={cov:.2f}, velocity={vel}"
        phase_node = DecisionNode(
            id="campaign_phase",
            label="Campaign phase",
            options=["Exploration", "Exploitation", "Transition"],
            selected=phase_selected,
            reason=phase_reason,
        )

        return AnalyzerOutput(
            round_number=input_data.round_number,
            diagnostics=_diagnostics_to_dict(diagnostics),
            convergence_status=convergence.status,
            convergence_confidence=convergence.confidence,
            round_mean_kpi=round_mean,
            round_best_kpi=round_best,
            round_cv=round_cv_val,
            improvement_vs_last=improvement,
            aleatoric_std=aleatoric_std,
            narrative=narrative,
            decision_nodes=[
                improvement_node.to_dict(),
                diag_ok_node.to_dict(),
                conv_node.to_dict(),
                phase_node.to_dict(),
            ],
        )


# ---------------------------------------------------------------------------
# Serialization helper
# ---------------------------------------------------------------------------


def _diagnostics_to_dict(d: DiagnosticSignals) -> dict[str, Any]:
    """Convert DiagnosticSignals frozen dataclass to a plain dict."""
    return {
        "space_coverage": d.space_coverage,
        "model_uncertainty": d.model_uncertainty,
        "noise_ratio": d.noise_ratio,
        "replicate_need_score": d.replicate_need_score,
        "batch_kpi_cv": d.batch_kpi_cv,
        "improvement_velocity": d.improvement_velocity,
        "ei_decay_proxy": d.ei_decay_proxy,
        "kpi_var_ratio": d.kpi_var_ratio,
        "convergence_status": d.convergence_status,
        "convergence_confidence": d.convergence_confidence,
        "local_smoothness": d.local_smoothness,
        "batch_param_spread": d.batch_param_spread,
        "calibration_factor": d.calibration_factor,
        "drift_score": d.drift_score,
    }
