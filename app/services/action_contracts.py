"""Action Contracts — Capability DSL for lab automation primitives.

Replaces the binary CRITICAL/BYPASS error classification with a 4-tier
safety system and adds declarative preconditions, effects, and
timeout/retry configuration per primitive.

Design sources:
- PDDL: preconditions + effects + typed predicates
- BehaviorTree.CPP: inline condition evaluation
- PyLabRobot: validate → execute → commit transaction pattern
- Opentrons issue #13197: check preconditions BEFORE physical motion
- IEC 61508: multi-level safety classification
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


# ---------------------------------------------------------------------------
# Safety Classification (4-tier, replacing binary CRITICAL/BYPASS)
# ---------------------------------------------------------------------------


class SafetyClass(IntEnum):
    """4-tier safety classification for lab automation primitives.

    Level 0 — INFORMATIONAL: Log and continue. Pure software / non-state-affecting.
    Level 1 — REVERSIBLE:    Log and continue. Affects hardware state but correctable.
    Level 2 — CAREFUL:       Retry if configured, then abort. Physical but retryable.
    Level 3 — HAZARDOUS:     Abort immediately, no retry. Irreversible liquid/experiment.
    """

    INFORMATIONAL = 0
    REVERSIBLE = 1
    CAREFUL = 2
    HAZARDOUS = 3

    @classmethod
    def from_string(cls, s: str) -> SafetyClass:
        """Parse a safety class from string (case-insensitive)."""
        return cls[s.strip().upper()]

    def to_legacy(self) -> str:
        """Map back to legacy CRITICAL/BYPASS for backward compatibility."""
        if self <= SafetyClass.REVERSIBLE:
            return "BYPASS"
        return "CRITICAL"


# Legacy mapping table: primitive name → SafetyClass
# Used when skill files don't have safety_class in frontmatter.
LEGACY_SAFETY_MAP: dict[str, SafetyClass] = {
    # BYPASS → INFORMATIONAL (pure software / non-state-affecting)
    "robot.home": SafetyClass.INFORMATIONAL,
    "robot.load_pipettes": SafetyClass.INFORMATIONAL,
    "robot.set_lights": SafetyClass.INFORMATIONAL,
    "wait": SafetyClass.INFORMATIONAL,
    "log": SafetyClass.INFORMATIONAL,
    "upload_artifact": SafetyClass.INFORMATIONAL,
    "ssh.start_stream": SafetyClass.INFORMATIONAL,
    "ssh.stop_stream": SafetyClass.INFORMATIONAL,
    "squidstat.get_data": SafetyClass.INFORMATIONAL,
    "squidstat.save_snapshot": SafetyClass.INFORMATIONAL,
    "squidstat.reset_plot": SafetyClass.INFORMATIONAL,
    # BYPASS → REVERSIBLE (affects hardware state but correctable)
    "robot.blowout": SafetyClass.REVERSIBLE,
    "heat": SafetyClass.REVERSIBLE,
    "plc.dispense_ml": SafetyClass.REVERSIBLE,
    "relay.set_channel": SafetyClass.REVERSIBLE,
    "relay.turn_on": SafetyClass.REVERSIBLE,
    "relay.turn_off": SafetyClass.REVERSIBLE,
    "relay.switch_to": SafetyClass.REVERSIBLE,
    # CRITICAL → CAREFUL (physical but retryable)
    "robot.load_labware": SafetyClass.CAREFUL,
    "robot.load_custom_labware": SafetyClass.CAREFUL,
    "robot.move_to_well": SafetyClass.CAREFUL,
    "robot.pick_up_tip": SafetyClass.CAREFUL,
    "robot.drop_tip": SafetyClass.CAREFUL,
    "plc.set_pump_on_timer": SafetyClass.CAREFUL,
    "plc.set_ultrasonic_on_timer": SafetyClass.CAREFUL,
    "cleanup.run_full": SafetyClass.CAREFUL,
    "cleanup.ultrasonic_water": SafetyClass.CAREFUL,
    "cleanup.ultrasonic_acid": SafetyClass.CAREFUL,
    "cleanup.water_flush": SafetyClass.CAREFUL,
    "cleanup.electrode_clean": SafetyClass.CAREFUL,
    # CRITICAL → HAZARDOUS (irreversible liquid/experiment)
    "robot.aspirate": SafetyClass.HAZARDOUS,
    "robot.dispense": SafetyClass.HAZARDOUS,
    "squidstat.run_experiment": SafetyClass.HAZARDOUS,
    "sample.prepare_from_csv": SafetyClass.HAZARDOUS,
    # Original OTbot primitives
    "aspirate": SafetyClass.HAZARDOUS,
    "eis": SafetyClass.HAZARDOUS,
}

# Defaults for primitives not in the map
_DEFAULT_BYPASS = SafetyClass.REVERSIBLE
_DEFAULT_CRITICAL = SafetyClass.CAREFUL


# ---------------------------------------------------------------------------
# Template substitution helper
# ---------------------------------------------------------------------------

_TEMPLATE_RE = re.compile(r"\{(\w+)\}")


def _render_template(template: str, params: dict[str, Any]) -> str:
    """Replace {param_name} placeholders with actual values from params."""

    def _replace(match: re.Match) -> str:
        key = match.group(1)
        val = params.get(key, match.group(0))  # keep original if not found
        return str(val)

    return _TEMPLATE_RE.sub(_replace, template)


# ---------------------------------------------------------------------------
# Precondition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Precondition:
    """A single precondition predicate with {param} template references.

    Predicate patterns:
    - "labware_loaded:{labware}"  → check state labware is loaded
    - "tip_on:{pipette}"          → check tip is attached
    - "tip_off:{pipette}"         → check tip is NOT attached
    - "pipettes_loaded"           → check pipettes are initialized
    - "robot_homed"               → check robot is homed
    - "experiment_idle:{channel}" → check squidstat channel is idle
    """

    predicate: str

    def render(self, params: dict[str, Any]) -> str:
        """Return predicate with {param} placeholders filled in."""
        return _render_template(self.predicate, params)

    def evaluate(self, state: dict[str, Any], params: dict[str, Any]) -> bool:
        """Substitute params, then check against state dict.

        The state dict is a flat representation of RunContext fields.
        Delegates actual evaluation to RunContext.check_precondition().
        """
        from app.services.run_context import RunContext

        rendered = self.render(params)
        if isinstance(state, RunContext):
            return state.check_precondition(rendered)
        # Fallback for plain dict (testing)
        return _evaluate_predicate_dict(rendered, state)


def _evaluate_predicate_dict(rendered: str, state: dict[str, Any]) -> bool:
    """Evaluate a rendered predicate against a plain dict (for testing)."""
    parts = rendered.split(":")

    if parts[0] == "labware_loaded" and len(parts) >= 2:
        return bool(state.get("labware_loaded", {}).get(parts[1]))
    if parts[0] == "tip_on" and len(parts) >= 2:
        return state.get("tip_state", {}).get(parts[1]) == "on"
    if parts[0] == "tip_off" and len(parts) >= 2:
        return state.get("tip_state", {}).get(parts[1]) != "on"
    if parts[0] == "pipettes_loaded":
        return bool(state.get("pipettes_loaded"))
    if parts[0] == "robot_homed":
        return bool(state.get("robot_homed"))
    if parts[0] == "experiment_idle" and len(parts) >= 2:
        return not state.get("experiment_running", {}).get(parts[1], False)

    # Unknown predicate → fail safe (treat as unsatisfied)
    return False


# ---------------------------------------------------------------------------
# Effect
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Effect:
    """A single state mutation operation with {param} template references.

    Operation patterns:
    - "set:key:value"                   → state[key] = value
    - "set:key:subkey:value"            → state[key][subkey] = value
    - "increase:key:subkey:amount"      → state[key][subkey] += amount
    - "decrease:key:subkey:amount"      → state[key][subkey] -= amount
    """

    operation: str

    def render(self, params: dict[str, Any]) -> str:
        """Return operation with {param} placeholders filled in."""
        return _render_template(self.operation, params)

    def apply(self, state: dict[str, Any], params: dict[str, Any]) -> None:
        """Substitute params and apply mutation to state.

        Delegates to RunContext.apply_effect() if state is a RunContext,
        otherwise applies to plain dict (for testing).
        """
        from app.services.run_context import RunContext

        rendered = self.render(params)
        if isinstance(state, RunContext):
            state.apply_effect(rendered)
        else:
            _apply_effect_dict(rendered, state)


def _coerce_value(val: str) -> Any:
    """Convert a string value to the appropriate Python type."""
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    try:
        return float(val)
    except (ValueError, TypeError):
        return val


def _apply_effect_dict(rendered: str, state: dict[str, Any]) -> None:
    """Apply a rendered effect to a plain dict (for testing)."""
    parts = rendered.split(":")
    if len(parts) < 3:
        return

    op = parts[0]

    if op == "set":
        # "set:key:value" or "set:key:subkey:value" or "set:key:s1:s2:value"
        value = _coerce_value(parts[-1])
        keys = parts[1:-1]
        target = state
        for k in keys[:-1]:
            target = target.setdefault(k, {})
        target[keys[-1]] = value

    elif op in ("increase", "decrease"):
        # "increase:key:subkey:amount" or "decrease:key:subkey:amount"
        amount = float(parts[-1])
        keys = parts[1:-1]
        target = state
        for k in keys[:-1]:
            target = target.setdefault(k, {})
        current = float(target.get(keys[-1], 0.0))
        if op == "increase":
            target[keys[-1]] = current + amount
        else:
            target[keys[-1]] = current - amount


# ---------------------------------------------------------------------------
# Timeout / Retry Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimeoutConfig:
    """Per-primitive timeout and retry configuration."""

    seconds: float = 300.0  # default 5 minutes
    retries: int = 0  # default: no retries


# ---------------------------------------------------------------------------
# Action Contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionContract:
    """Full contract for a single primitive action.

    Combines preconditions, effects, timeout/retry config, and safety class.
    """

    preconditions: tuple[Precondition, ...] = ()
    effects: tuple[Effect, ...] = ()
    timeout: TimeoutConfig = field(default_factory=TimeoutConfig)
    safety_class: SafetyClass = SafetyClass.CAREFUL


# ---------------------------------------------------------------------------
# Parsing functions (from YAML frontmatter dicts)
# ---------------------------------------------------------------------------


def parse_contract(raw: dict[str, Any] | None) -> tuple[Precondition, ...] | None:
    """Parse the 'contract' block from YAML frontmatter.

    Returns (preconditions_tuple, effects_tuple) or None if no contract.
    """
    if not raw or not isinstance(raw, dict):
        return None

    preconditions = tuple(
        Precondition(predicate=str(p))
        for p in raw.get("preconditions", [])
        if p
    )
    effects = tuple(
        Effect(operation=str(e))
        for e in raw.get("effects", [])
        if e
    )
    return preconditions, effects  # type: ignore[return-value]


def parse_timeout(raw: dict[str, Any] | None) -> TimeoutConfig:
    """Parse the 'timeout' block from YAML frontmatter."""
    if not raw or not isinstance(raw, dict):
        return TimeoutConfig()
    return TimeoutConfig(
        seconds=float(raw.get("seconds", 300.0)),
        retries=int(raw.get("retries", 0)),
    )


def parse_safety_class(
    raw_safety: str | None,
    error_class: str = "CRITICAL",
    primitive_name: str = "",
) -> SafetyClass:
    """Parse safety_class, falling back to error_class + legacy mapping.

    Priority:
    1. Explicit safety_class string in frontmatter
    2. Legacy mapping table (primitive_name → SafetyClass)
    3. Default based on error_class
    """
    if raw_safety and isinstance(raw_safety, str):
        try:
            return SafetyClass.from_string(raw_safety)
        except (KeyError, ValueError):
            pass

    # Fallback to legacy mapping
    if primitive_name in LEGACY_SAFETY_MAP:
        return LEGACY_SAFETY_MAP[primitive_name]

    # Default based on error_class
    if error_class == "BYPASS":
        return _DEFAULT_BYPASS
    return _DEFAULT_CRITICAL


def build_action_contract(
    raw_contract: dict[str, Any] | None,
    raw_timeout: dict[str, Any] | None,
    safety_class: SafetyClass,
) -> ActionContract:
    """Build a complete ActionContract from parsed frontmatter components."""
    preconditions: tuple[Precondition, ...] = ()
    effects: tuple[Effect, ...] = ()

    parsed = parse_contract(raw_contract)
    if parsed is not None:
        preconditions, effects = parsed

    timeout = parse_timeout(raw_timeout)

    return ActionContract(
        preconditions=preconditions,
        effects=effects,
        timeout=timeout,
        safety_class=safety_class,
    )
