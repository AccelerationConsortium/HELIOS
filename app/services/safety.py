from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.services.action_contracts import (
    ActionContract,
    Effect,
    Precondition,
    SafetyClass,
    _apply_effect_dict,
    _evaluate_predicate_dict,
)


# All primitives supported by the battery-lab hardware + original OTbot primitives
BATTERY_LAB_PRIMITIVES: list[str] = [
    # Original OTbot
    "aspirate", "heat", "eis", "upload_artifact", "wait",
    # Robot
    "robot.home", "robot.load_pipettes", "robot.set_lights",
    "robot.load_labware", "robot.load_custom_labware",
    "robot.move_to_well", "robot.pick_up_tip", "robot.drop_tip",
    "robot.aspirate", "robot.dispense", "robot.blowout",
    # PLC
    "plc.dispense_ml", "plc.set_pump_on_timer", "plc.set_ultrasonic_on_timer",
    # Relay
    "relay.set_channel", "relay.turn_on", "relay.turn_off", "relay.switch_to",
    # Squidstat
    "squidstat.run_experiment", "squidstat.get_data",
    "squidstat.save_snapshot", "squidstat.reset_plot",
    # High-level
    "cleanup.run_full",
    "sample.prepare_from_csv",
    # SSH
    "ssh.start_stream", "ssh.stop_stream",
    # Utility
    "log",
]


@dataclass
class SafetyResult:
    allowed: bool
    violations: list[str]
    requires_approval: bool = False


def evaluate_preflight(
    *, compiled_graph: dict[str, Any], policy_snapshot: dict[str, Any]
) -> SafetyResult:
    """Evaluate preflight safety checks (parameter limits, allowed primitives).

    This is the original preflight check — parameter bounds and primitive allowlist.
    For contract-based precondition validation, use ``evaluate_contract_preflight()``.
    """
    violations: list[str] = []

    max_temp = float(policy_snapshot.get("max_temp_c", 95.0))
    max_volume = float(policy_snapshot.get("max_volume_ul", 1000.0))
    allowed_primitives = set(
        policy_snapshot.get("allowed_primitives", BATTERY_LAB_PRIMITIVES)
    )

    for step in compiled_graph.get("steps", []):
        primitive = step["primitive"]
        params = step.get("params", {})

        if primitive not in allowed_primitives:
            violations.append(f"primitive not allowed: {primitive}")

        if primitive == "heat":
            temp = float(params.get("temp_c", 0.0))
            if temp > max_temp:
                violations.append(
                    f"step {step['step_key']}: temp_c={temp} exceeds max {max_temp}"
                )

        # Volume check for both original 'aspirate' and dotted 'robot.aspirate'
        if primitive in ("aspirate", "robot.aspirate"):
            volume = float(params.get("volume_ul", params.get("volume", 0.0)))
            if volume > max_volume:
                violations.append(
                    f"step {step['step_key']}: volume_ul={volume} exceeds max {max_volume}"
                )

        if primitive == "robot.dispense":
            volume = float(params.get("volume_ul", params.get("volume", 0.0)))
            if volume > max_volume:
                violations.append(
                    f"step {step['step_key']}: dispense volume={volume} exceeds max {max_volume}"
                )

    requires_approval = bool(policy_snapshot.get("require_human_approval", False))
    return SafetyResult(
        allowed=not violations,
        violations=violations,
        requires_approval=requires_approval,
    )


def evaluate_contract_preflight(
    *,
    steps: list[dict[str, Any]],
    contracts: dict[str, ActionContract],
) -> SafetyResult:
    """Forward-simulate preconditions and effects to catch contract violations.

    Walks through the steps in order, checking each step's preconditions
    against a simulated state, then applying its effects. This catches
    cases like "aspirate before pick_up_tip" at compile time.

    Args:
        steps: List of step dicts, each with 'step_key', 'primitive', 'params'.
        contracts: Mapping of primitive name → ActionContract.

    Returns:
        SafetyResult with any precondition violations found.
    """
    violations: list[str] = []
    # Simulated state for forward checking
    sim_state: dict[str, Any] = {}

    for step in steps:
        primitive = step["primitive"]
        params = step.get("params", {})
        step_key = step.get("step_key", primitive)

        contract = contracts.get(primitive)
        if contract is None:
            continue

        # Check preconditions against simulated state
        for precond in contract.preconditions:
            rendered = precond.render(params)
            if not _evaluate_predicate_dict(rendered, sim_state):
                violations.append(
                    f"step {step_key}: precondition '{rendered}' not satisfied"
                )

        # Apply effects to simulated state (regardless of precondition results,
        # to avoid cascading false-positive violations)
        for effect in contract.effects:
            rendered = effect.render(params)
            _apply_effect_dict(rendered, sim_state)

    return SafetyResult(
        allowed=not violations,
        violations=violations,
    )


def evaluate_runtime_step(
    *, step: dict[str, Any], policy_snapshot: dict[str, Any], interlock_state: dict[str, Any]
) -> SafetyResult:
    if not interlock_state.get("hardware_interlock_ok", True):
        return SafetyResult(
            allowed=False,
            violations=["hardware interlock is not engaged"],
            requires_approval=False,
        )

    if step["primitive"] == "heat" and not interlock_state.get("cooling_ok", True):
        return SafetyResult(
            allowed=False,
            violations=["cooling subsystem not healthy for heat step"],
            requires_approval=False,
        )

    # Runtime uses same threshold policy as preflight for deterministic guardrails.
    return evaluate_preflight(compiled_graph={"steps": [step]}, policy_snapshot=policy_snapshot)
