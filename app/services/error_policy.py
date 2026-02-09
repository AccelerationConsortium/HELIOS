"""Error policy module — CRITICAL / BYPASS classification for step failures.

The battery-lab dispatcher uses two error patterns:

* **CRITICAL**: The handler ``raise``s on failure, meaning the step *must*
  succeed or the whole run is aborted.  Examples: labware loading, tip
  pick-up, aspirate/dispense, squidstat experiment start.

* **BYPASS**: The handler catches the exception, logs a warning, and returns
  ``False``.  The workflow continues even if the step fails.  Examples:
  robot homing, light control, relay switching, blowout.

This module formalises those conventions into an ``ErrorPolicy`` that the
worker consults when a step raises.  It also provides the upgraded 4-tier
``classify_step_safety()`` function using ``SafetyClass``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.services.action_contracts import (
    LEGACY_SAFETY_MAP,
    ActionContract,
    SafetyClass,
)


# ---------------------------------------------------------------------------
# Primitive → severity mapping (legacy, preserved for backward compatibility)
# ---------------------------------------------------------------------------

#: Primitives whose failure is CRITICAL — any exception aborts the run.
CRITICAL_PRIMITIVES: frozenset[str] = frozenset(
    {
        # Robot — physical liquid handling (wrong volumes / positions are unrecoverable)
        "robot.load_labware",
        "robot.load_custom_labware",
        "robot.move_to_well",
        "robot.pick_up_tip",
        "robot.drop_tip",
        "robot.aspirate",
        "robot.dispense",
        # PLC — timed pump operations
        "plc.set_pump_on_timer",
        "plc.set_ultrasonic_on_timer",
        # Squidstat — electrochemical experiments
        "squidstat.run_experiment",
        # Cleanup — orchestrated multi-device cleanup
        "cleanup.run_full",
        # Sample preparation
        "sample.prepare_from_csv",
        # Original OTbot primitives
        "aspirate",
        "eis",
    }
)

#: Primitives whose failure can be BYPASSed — log and continue.
BYPASS_PRIMITIVES: frozenset[str] = frozenset(
    {
        # Robot — non-critical helpers
        "robot.home",
        "robot.load_pipettes",
        "robot.set_lights",
        "robot.blowout",
        # PLC — best-effort dispensing
        "plc.dispense_ml",
        # Relay — channel switching (often retryable at network level)
        "relay.set_channel",
        "relay.turn_on",
        "relay.turn_off",
        "relay.switch_to",
        # Squidstat — data retrieval / plots
        "squidstat.get_data",
        "squidstat.save_snapshot",
        "squidstat.reset_plot",
        # SSH streaming
        "ssh.start_stream",
        "ssh.stop_stream",
        # Utilities
        "wait",
        "log",
        "heat",
        "upload_artifact",
    }
)


# ---------------------------------------------------------------------------
# 4-tier safety classification
# ---------------------------------------------------------------------------


def classify_step_safety(
    primitive: str,
    contract: ActionContract | None = None,
) -> SafetyClass:
    """Return the ``SafetyClass`` for a given primitive.

    Priority:
    1. Contract's safety_class (if contract provided)
    2. LEGACY_SAFETY_MAP lookup
    3. Default based on CRITICAL/BYPASS membership
    4. CAREFUL (fail-safe default for completely unknown primitives)
    """
    # 1. From contract
    if contract is not None:
        return contract.safety_class

    # 2. From legacy map
    if primitive in LEGACY_SAFETY_MAP:
        return LEGACY_SAFETY_MAP[primitive]

    # 3. From CRITICAL/BYPASS sets
    if primitive in CRITICAL_PRIMITIVES:
        return SafetyClass.CAREFUL
    if primitive in BYPASS_PRIMITIVES:
        return SafetyClass.REVERSIBLE

    # 4. Fail-safe
    return SafetyClass.CAREFUL


def classify_step_error(primitive: str, exc: Exception) -> str:
    """Return ``"CRITICAL"`` or ``"BYPASS"`` for a given primitive failure.

    Delegates to ``classify_step_safety()`` and maps back to legacy labels.

    * Known CRITICAL primitives always return ``"CRITICAL"``.
    * Known BYPASS primitives return ``"BYPASS"``.
    * Unknown primitives default to ``"CRITICAL"`` (fail-safe).
    """
    # Keep the original fast-path for full backward compatibility
    if primitive in CRITICAL_PRIMITIVES:
        return "CRITICAL"
    if primitive in BYPASS_PRIMITIVES:
        return "BYPASS"
    # Unknown primitive → fail-safe
    return "CRITICAL"


# ---------------------------------------------------------------------------
# ErrorPolicy dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ErrorPolicy:
    """Run-level error policy governing how step failures are handled.

    Attributes:
        allow_bypass: If ``True``, steps classified as BYPASS may be skipped
            instead of aborting the run.  When ``False``, every failure is
            treated as CRITICAL regardless of the primitive classification.
    """

    allow_bypass: bool = True

    @classmethod
    def from_policy_snapshot(cls, policy: dict[str, Any]) -> ErrorPolicy:
        """Build an ErrorPolicy from the run's policy_snapshot dict.

        Recognised keys:
            ``error_policy.allow_bypass`` (bool, default ``True``)
        """
        ep = policy.get("error_policy", {})
        if isinstance(ep, dict):
            allow_bypass = bool(ep.get("allow_bypass", True))
        else:
            allow_bypass = True
        return cls(allow_bypass=allow_bypass)
