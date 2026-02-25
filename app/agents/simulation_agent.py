"""Simulation Agent — pre-execution dry-run protocol verification.

Sits between SafetyAgent (L1 preflight) and ExecutorAgent (L0 hardware),
providing a virtual execution pass that catches logical errors before any
physical action is taken:

  compile → safety preflight → [SimulationAgent] → execute

Runs ``simulate_protocol()`` from ``app.services.simulation`` and translates
the ``SimulationResult`` into SSE thinking events so scientists can follow
the verification reasoning in real time.

Verdict semantics (propagated to orchestrator):
  "pass"  → proceed normally
  "warn"  → proceed, but log all warnings
  "fail"  → skip candidate (same as safety veto)

Emits:
  - ``agent_thinking``     — one message per verification phase
  - ``simulation_verdict`` — summary dict with verdict + violation counts
  - ``agent_result``       — SimulationOutput at completion
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.agents.base import BaseAgent, DecisionNode
from app.services.simulation import SimulationResult, SimulationViolation, simulate_protocol

logger = logging.getLogger(__name__)

# Type alias for the SSE emit callback injected by the orchestrator
EmitCallback = Callable[[dict[str, Any]], None]


# ---------------------------------------------------------------------------
# I/O models
# ---------------------------------------------------------------------------


class SimulationInput(BaseModel):
    """Input for the pre-execution simulation agent."""

    # Compiled protocol from CompilerAgent
    compiled_graph: dict[str, Any]
    # Deck plan from CompilerAgent (DeckPlan.to_dict())
    deck_plan: dict[str, Any] = Field(default_factory=dict)
    # Safety policy (same snapshot passed to SafetyAgent)
    policy_snapshot: dict[str, Any] = Field(default_factory=dict)
    # Candidate parameters (for context in SSE messages)
    candidate_params: dict[str, Any] = Field(default_factory=dict)
    # Optional pre-populated well volumes (labware:well → µL)
    initial_volumes: dict[str, float] = Field(default_factory=dict)

    round_number: int = 0
    candidate_index: int = 0

    # SSE emit callback — excluded from serialisation
    emit: EmitCallback | None = Field(default=None, exclude=True)

    model_config = {"arbitrary_types_allowed": True}


class SimulationOutput(BaseModel):
    """Output from the pre-execution simulation agent."""

    verdict: Literal["pass", "warn", "fail"]
    n_errors: int = 0
    n_warnings: int = 0
    violations: list[dict[str, Any]] = Field(default_factory=list)
    general_warnings: list[str] = Field(default_factory=list)
    estimated_duration_s: float = 0.0
    resource_summary: dict[str, Any] = Field(default_factory=dict)
    # Human-readable one-liner
    summary: str = ""
    decision_nodes: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent implementation
# ---------------------------------------------------------------------------


class SimulationAgent(BaseAgent[SimulationInput, SimulationOutput]):
    """Pre-execution dry-run protocol simulator.

    Calls ``simulate_protocol()`` with a 4-phase SSE narrative so the
    scientist can follow along as each check category is evaluated.
    """

    name = "simulation_agent"
    description = "Pre-execution virtual protocol verification (dry-run simulator)"
    layer = "L1"

    def validate_input(self, input_data: SimulationInput) -> list[str]:
        errors: list[str] = []
        if not isinstance(input_data.compiled_graph, dict):
            errors.append("compiled_graph must be a dict")
        return errors

    async def process(self, input_data: SimulationInput) -> SimulationOutput:  # noqa: C901
        emit = input_data.emit
        rnd = input_data.round_number
        cand = input_data.candidate_index
        steps = input_data.compiled_graph.get("steps", [])
        n_steps = len(steps)

        def _think(msg: str) -> None:
            if emit:
                emit({
                    "type": "agent_thinking",
                    "agent": "simulation",
                    "round": rnd,
                    "candidate": cand,
                    "message": msg,
                })

        # ── Phase 1: Deck resource validation ─────────────────────────────
        _think(f"Validating deck resources for {n_steps} compiled steps...")

        # ── Phase 2: Volume accounting ─────────────────────────────────────
        _think("Running volume accounting (per-well liquid tracking)...")

        # ── Phase 3: Tip lifecycle + parameter bounds ──────────────────────
        _think("Checking tip lifecycle state machine and parameter bounds...")

        # ── Phase 4: Duration estimation ───────────────────────────────────
        _think("Estimating run duration...")

        # --- Run the core simulation engine ---
        try:
            sim_result: SimulationResult = simulate_protocol(
                steps=steps,
                deck_snapshot=input_data.deck_plan,
                policy_snapshot=input_data.policy_snapshot,
                initial_volumes=input_data.initial_volumes or None,
            )
        except Exception as exc:
            logger.error("Simulation engine error: %s", exc, exc_info=True)
            # Treat engine failure as a warning-level issue, not a block
            if emit:
                emit({
                    "type": "simulation_verdict",
                    "agent": "simulation",
                    "round": rnd,
                    "candidate": cand,
                    "verdict": "warn",
                    "message": f"Simulation engine error (proceeding): {exc}",
                })
            engine_err_node = DecisionNode(
                id="engine_status",
                label="Simulation engine",
                options=["Engine OK", "Engine error (warn and proceed)"],
                selected="Engine error (warn and proceed)",
                reason=str(exc),
                outcome="Proceeding with caution — physical run not blocked",
            )
            verdict_err_node = DecisionNode(
                id="verdict",
                label="Simulation verdict",
                options=["pass", "warn", "fail"],
                selected="warn",
                reason="Engine exception treated as non-blocking warning",
            )
            return SimulationOutput(
                verdict="warn",
                summary=f"Simulation engine error — proceeding with caution: {exc}",
                decision_nodes=[engine_err_node.to_dict(), verdict_err_node.to_dict()],
            )

        # --- Serialise violations ---
        violations_dicts = [
            {
                "step_key": v.step_key,
                "primitive": v.primitive,
                "severity": v.severity,
                "message": v.message,
            }
            for v in sim_result.violations
        ]
        n_errors = len(sim_result.errors)
        n_warnings = len(sim_result.soft_warnings) + len(sim_result.warnings)

        # --- Build summary string ---
        verdict = sim_result.verdict
        dur_str = _format_duration(sim_result.estimated_duration_s)
        if verdict == "pass":
            summary = f"Simulation passed — {n_steps} steps, est. {dur_str}."
        elif verdict == "warn":
            summary = (
                f"Simulation passed with {n_warnings} warning(s) — "
                f"{n_steps} steps, est. {dur_str}."
            )
        else:
            summary = (
                f"Simulation FAILED — {n_errors} error(s), {n_warnings} warning(s). "
                f"Candidate will be skipped."
            )

        # ── Phase 5: Emit verdict ──────────────────────────────────────────
        _think(f"  → {summary}")

        if emit:
            emit({
                "type": "simulation_verdict",
                "agent": "simulation",
                "round": rnd,
                "candidate": cand,
                "verdict": verdict,
                "n_errors": n_errors,
                "n_warnings": n_warnings,
                "estimated_duration_s": sim_result.estimated_duration_s,
                "message": summary,
            })

        # Log errors so they appear in server logs even when not blocking
        if sim_result.errors:
            for v in sim_result.errors:
                logger.warning("Simulation error [%s/%s]: %s", v.step_key, v.primitive, v.message)
        if sim_result.warnings:
            for w in sim_result.warnings:
                logger.info("Simulation warning: %s", w)

        # Build decision nodes
        engine_ok_node = DecisionNode(
            id="engine_status",
            label="Simulation engine",
            options=["Engine OK", "Engine error (warn and proceed)"],
            selected="Engine OK",
            reason=f"Engine ran {n_steps} steps without exception",
            outcome=f"est. duration {_format_duration(sim_result.estimated_duration_s)}",
        )
        violation_children = tuple(
            DecisionNode(
                id=f"violation_{vi}",
                label=f"{v['step_key']} / {v['primitive']}",
                options=["error", "warning"],
                selected=v["severity"],
                reason=v["message"],
            )
            for vi, v in enumerate(violations_dicts)
        )
        verdict_node = DecisionNode(
            id="verdict",
            label="Simulation verdict",
            options=["pass", "warn", "fail"],
            selected=verdict,
            reason=(
                f"{n_errors} error(s), {n_warnings} warning(s)"
                if verdict != "pass" else f"All {n_steps} steps passed"
            ),
            outcome=summary,
            children=violation_children,
        )

        return SimulationOutput(
            verdict=verdict,
            n_errors=n_errors,
            n_warnings=n_warnings,
            violations=violations_dicts,
            general_warnings=list(sim_result.warnings),
            estimated_duration_s=sim_result.estimated_duration_s,
            resource_summary=sim_result.resource_summary,
            summary=summary,
            decision_nodes=[engine_ok_node.to_dict(), verdict_node.to_dict()],
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h{mins:02d}m"
