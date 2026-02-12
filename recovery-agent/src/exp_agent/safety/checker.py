"""Safety checker for runtime action validation.

This module provides functions to check recovery actions against
the SafetyPacket constraints before execution.

Based on plan.md section 3(2):
- RecoveryAction 检查器: action + SafetyPacket + current_state → ALLOW/BLOCK/REQUIRE_HUMAN
"""

from typing import Dict, Any, List, Optional
from exp_agent.core.types import Action, DeviceState
from exp_agent.core.safety_types import (
    SafetyPacket,
    SafetyConstraint,
    SafetyThreshold,
    ActionSafetyCheck,
    SafetyCheckResult,
)


def check_action_safety(
    action: Action,
    packet: SafetyPacket,
    state: DeviceState,
) -> ActionSafetyCheck:
    """Check if an action is safe to execute given current state and safety constraints.

    This function implements the safety_check(action, SafetyPacket, current_state)
    interface described in plan.md section 3(2).

    Args:
        action: The recovery action to check.
        packet: SafetyPacket from pre-flight assessment.
        state: Current device state with telemetry.

    Returns:
        ActionSafetyCheck with result ("allow", "block", "require_human"),
        violated constraints, and rationale.

    Decision Logic (from plan.md):
        - Device timeout: Generally ALLOW AUTO-RETRY
        - If SafetyPacket marks "flammable solvent + closed system + needs ventilation":
            - "increase heating / add more reagent" → BLOCK
        - If "spill/exposure" event detected:
            - Jump to EVACUATE (safety agent's emergency scenario)
    """
    violated_constraints: List[SafetyConstraint] = []
    violated_thresholds: List[SafetyThreshold] = []
    triggered_playbooks: List[str] = []
    rationale_parts: List[str] = []

    # Extract telemetry
    telemetry = state.telemetry or {}
    action_name = action.name.lower()
    action_params = action.params or {}

    # --- Check against constraints ---
    for constraint in packet.constraints:
        violation = _check_constraint(constraint, action, telemetry)
        if violation:
            violated_constraints.append(constraint)
            rationale_parts.append(violation)

    # --- Check against thresholds ---
    for threshold in packet.thresholds:
        violation = _check_threshold(threshold, action, telemetry)
        if violation:
            violated_thresholds.append(threshold)
            rationale_parts.append(violation)

    # --- Check action type against emergency scenarios ---
    for playbook in packet.emergency_playbooks:
        if _triggers_emergency(playbook.scenario, action, telemetry, state):
            triggered_playbooks.append(playbook.scenario)
            if not playbook.recovery_possible:
                rationale_parts.append(
                    f"Emergency scenario '{playbook.scenario}' active - automated recovery not possible"
                )

    # --- Determine result ---
    result: SafetyCheckResult = "allow"
    alternative_actions: List[str] = []

    # Critical violations → BLOCK
    if any(c.mandatory for c in violated_constraints):
        result = "block"
        alternative_actions = _suggest_alternatives(action, violated_constraints)

    if any(t.severity == "critical" for t in violated_thresholds):
        result = "block"
        # Add threshold-based alternatives
        for t in violated_thresholds:
            if t.severity == "critical":
                alternative_actions.append(t.action)

    # Emergency scenarios that require human
    if triggered_playbooks:
        for playbook in packet.emergency_playbooks:
            if playbook.scenario in triggered_playbooks:
                if playbook.requires_human or playbook.requires_evacuation:
                    result = "require_human"
                if not playbook.recovery_possible:
                    result = "block"
                    alternative_actions = ["safe_shutdown", "evacuate", "ask_human"]

    # Non-mandatory violations → warn but allow
    if result == "allow" and (violated_constraints or violated_thresholds):
        if not any(c.mandatory for c in violated_constraints):
            # Only warnings, not blocks
            rationale_parts.append("Non-mandatory constraints violated - proceeding with caution")

    # Build rationale
    if result == "allow":
        rationale = "Action passed safety checks."
    else:
        rationale = "; ".join(rationale_parts) if rationale_parts else "Safety constraint violated."

    return ActionSafetyCheck(
        action_name=action.name,
        result=result,
        violated_constraints=violated_constraints,
        violated_thresholds=violated_thresholds,
        triggered_playbooks=triggered_playbooks,
        rationale=rationale,
        alternative_actions=list(set(alternative_actions)),  # dedupe
    )


def _check_constraint(
    constraint: SafetyConstraint,
    action: Action,
    telemetry: Dict[str, Any]
) -> Optional[str]:
    """Check if an action violates a constraint. Returns violation message or None."""

    action_name = action.name.lower()
    action_params = action.params or {}

    # Temperature limit constraints
    if constraint.type == "temperature_limit":
        # Block actions that would increase temperature
        if any(kw in action_name for kw in ["heat", "warm", "increase_temp"]):
            target_temp = action_params.get("target", action_params.get("temperature"))
            if target_temp and constraint.value and target_temp > constraint.value:
                return f"Action would exceed temperature limit ({constraint.value}{constraint.unit}): {constraint.description}"

        # Check if current temp already exceeds limit
        current_temp = telemetry.get("temperature")
        if current_temp and constraint.value and current_temp > constraint.value:
            if any(kw in action_name for kw in ["heat", "warm", "continue"]):
                return f"Current temperature ({current_temp}°C) exceeds limit ({constraint.value}°C)"

    # Ventilation requirement
    elif constraint.type == "ventilation_required":
        # Can't check this directly - flag for human verification
        # In practice, would check environment sensors
        if any(kw in action_name for kw in ["heat", "evaporate", "distill"]):
            return f"Ventilation required for this action: {constraint.description}"

    # No heating constraint
    elif constraint.type == "no_heating":
        if any(kw in action_name for kw in ["heat", "warm", "increase_temp", "set_temp"]):
            return f"Heating prohibited: {constraint.rationale}"

    # Rate limit constraints
    elif constraint.type == "rate_limit":
        rate_param = action_params.get("rate", action_params.get("flow_rate"))
        if rate_param and constraint.value and rate_param > constraint.value:
            return f"Rate ({rate_param}) exceeds limit ({constraint.value}): {constraint.description}"

    # Supervision required
    elif constraint.type == "supervision_required":
        # Flag any autonomous action for human review
        if action.effect == "write" and action.irreversible:
            return f"Supervision required for irreversible actions: {constraint.description}"

    # PPE required - informational only
    elif constraint.type == "ppe_required":
        # Cannot enforce programmatically - log warning
        pass

    return None


def _check_threshold(
    threshold: SafetyThreshold,
    action: Action,
    telemetry: Dict[str, Any]
) -> Optional[str]:
    """Check if current telemetry violates a threshold. Returns violation message or None.

    Threshold semantics:
    - operator="<" with value=100 means "threshold is violated when value >= 100"
      (i.e., the safe condition is value < 100)
    - operator=">" with value=100 means "threshold is violated when value > 100"
      (i.e., the check triggers an alert when value exceeds 100)

    For maximum thresholds like "temperature must not exceed 100°C":
    - Use operator=">" with value=100
    - This means: if current_value > threshold.value → violated
    """

    current_value = telemetry.get(threshold.variable)
    if current_value is None:
        return None  # Can't check without data

    violated = False
    op = threshold.operator

    # Threshold semantics: the operator defines when the threshold is VIOLATED
    # e.g., ">" means violated when current_value > threshold.value
    if op == "<" and current_value < threshold.value:
        violated = True  # Violated when value is below minimum
    elif op == "<=" and current_value <= threshold.value:
        violated = True  # Violated when value is at or below minimum
    elif op == ">" and current_value > threshold.value:
        violated = True  # Violated when value exceeds maximum
    elif op == ">=" and current_value >= threshold.value:
        violated = True  # Violated when value is at or above maximum
    elif op == "==" and current_value == threshold.value:
        violated = True  # Violated when value equals forbidden value
    elif op == "!=" and current_value != threshold.value:
        violated = True  # Violated when value doesn't match required value
    elif op == "in_range":
        # Violated when value is OUTSIDE the range
        if threshold.value_max is not None:
            if not (threshold.value <= current_value <= threshold.value_max):
                violated = True

    if violated:
        return (
            f"{threshold.variable} ({current_value}{threshold.unit}) violates threshold "
            f"({op} {threshold.value}{threshold.unit}): {threshold.rationale or 'Safety limit exceeded'}"
        )

    # Also check if action would cause violation
    action_params = action.params or {}
    if threshold.variable == "temperature":
        target = action_params.get("target", action_params.get("temperature"))
        if target is not None:
            # Check if the target would violate the threshold
            if op == ">" and target > threshold.value:
                return f"Action target ({target}°C) would violate threshold (>{threshold.value}°C)"
            elif op == ">=" and target >= threshold.value:
                return f"Action target ({target}°C) would violate threshold (>={threshold.value}°C)"
            elif op == "<" and target < threshold.value:
                return f"Action target ({target}°C) would violate threshold (<{threshold.value}°C)"
            elif op == "<=" and target <= threshold.value:
                return f"Action target ({target}°C) would violate threshold (<={threshold.value}°C)"

    return None


def _triggers_emergency(
    scenario: str,
    action: Action,
    telemetry: Dict[str, Any],
    state: DeviceState
) -> bool:
    """Check if current state matches an emergency scenario."""

    # Check for spill/exposure indicators
    if scenario in ["spill", "chemical_spill"]:
        # Would be detected by sensors - check telemetry
        if telemetry.get("spill_detected") or telemetry.get("leak_detected"):
            return True
        # Check error context
        if state.status == "error":
            return False  # Would need error type

    if scenario in ["skin_contact", "eye_contact", "exposure"]:
        # Cannot detect automatically - would be reported by human
        pass

    if scenario == "fire":
        # Check for fire/smoke indicators
        if telemetry.get("fire_detected") or telemetry.get("smoke_detected"):
            return True
        if telemetry.get("temperature", 0) > 200:  # Very high temp
            return True

    if scenario == "overheat":
        # Check for thermal runaway
        temp = telemetry.get("temperature", 0)
        temp_rate = telemetry.get("temperature_rate", 0)  # °C/min
        if temp_rate > 10:  # Rapid temperature rise
            return True

    return False


def _suggest_alternatives(
    action: Action,
    violated_constraints: List[SafetyConstraint]
) -> List[str]:
    """Suggest alternative actions based on violated constraints."""

    alternatives: List[str] = []
    action_name = action.name.lower()

    for constraint in violated_constraints:
        if constraint.type == "temperature_limit":
            if "heat" in action_name:
                alternatives.append("reduce_heating")
                alternatives.append("wait_for_cooling")
            alternatives.append("safe_shutdown")

        elif constraint.type == "no_heating":
            alternatives.append("skip_heating_step")
            alternatives.append("abort")

        elif constraint.type == "ventilation_required":
            alternatives.append("verify_ventilation")
            alternatives.append("ask_human")

        elif constraint.type == "rate_limit":
            alternatives.append("reduce_rate")

        elif constraint.type == "supervision_required":
            alternatives.append("ask_human")

    if not alternatives:
        alternatives = ["ask_human", "safe_shutdown"]

    return alternatives


def check_chemical_safety_event(
    error_type: str,
    telemetry: Dict[str, Any],
    packet: SafetyPacket
) -> Optional[str]:
    """Check if an error is a chemical safety event requiring immediate action.

    Chemical safety events give SafetyAgent FINAL VETO POWER.
    RecoveryAgent can only choose: SAFE_SHUTDOWN, EVACUATE, ASK_HUMAN.

    Returns the required action if this is a chemical safety event, None otherwise.
    """

    # Chemical safety event types (from plan.md section 5)
    CHEMICAL_SAFETY_EVENTS = {
        "spill_detected": "safe_shutdown",
        "leak_detected": "safe_shutdown",
        "exposure_detected": "evacuate",
        "fire_detected": "evacuate",
        "smoke_detected": "evacuate",
        "incompatible_mix": "safe_shutdown",
        "pressure_buildup": "safe_shutdown",
        "off_gas_detected": "evacuate",
    }

    # Check error type
    if error_type in CHEMICAL_SAFETY_EVENTS:
        return CHEMICAL_SAFETY_EVENTS[error_type]

    # Check telemetry for chemical safety indicators
    for key, action in CHEMICAL_SAFETY_EVENTS.items():
        if telemetry.get(key):
            return action

    # Check for overheat beyond chemical threshold
    temp = telemetry.get("temperature")
    if temp is not None:
        for threshold in packet.thresholds:
            if threshold.variable == "temperature" and threshold.severity == "critical":
                if threshold.operator in [">", ">="] and temp > threshold.value:
                    return "safe_shutdown"
                if threshold.operator in ["<", "<="] and temp < threshold.value:
                    # Freezing scenario - less common
                    return "safe_shutdown"

    return None
