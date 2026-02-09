"""FaultInjector — Composable fault injection for SimLab.

Supports 8 fault types that can be stacked and configured independently:
- disconnection, timeout, tip_shortage, liquid_insufficient
- deck_conflict, sensor_drift, temp_hysteresis, file_missing

Each fault type has configurable probability and trigger primitives.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

# Sentinel for non-exception faults (e.g. sensor_drift modifies result)
_NOT_AN_EXCEPTION = object()


@dataclass(frozen=True)
class FaultConfig:
    """Configuration for a single fault type.

    Args:
        fault_type: One of the 8 supported fault types.
        trigger_primitives: Which primitives can trigger this fault.
            Empty tuple means any primitive can trigger.
        probability: Per-step injection probability (0.0–1.0).
        params: Fault-specific parameters (e.g. drift_pct for sensor_drift).
    """

    fault_type: str
    trigger_primitives: tuple[str, ...] = ()
    probability: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fault type constants
# ---------------------------------------------------------------------------

FAULT_DISCONNECTION = "disconnection"
FAULT_TIMEOUT = "timeout"
FAULT_TIP_SHORTAGE = "tip_shortage"
FAULT_LIQUID_INSUFFICIENT = "liquid_insufficient"
FAULT_DECK_CONFLICT = "deck_conflict"
FAULT_SENSOR_DRIFT = "sensor_drift"
FAULT_TEMP_HYSTERESIS = "temp_hysteresis"
FAULT_FILE_MISSING = "file_missing"

ALL_FAULT_TYPES = (
    FAULT_DISCONNECTION,
    FAULT_TIMEOUT,
    FAULT_TIP_SHORTAGE,
    FAULT_LIQUID_INSUFFICIENT,
    FAULT_DECK_CONFLICT,
    FAULT_SENSOR_DRIFT,
    FAULT_TEMP_HYSTERESIS,
    FAULT_FILE_MISSING,
)


class FaultInjector:
    """Composable fault injector for SimAdapter.

    Multiple FaultConfig instances can be registered; each is evaluated
    per step. The first matching fault (by primitive + probability) fires.

    Args:
        faults: List of fault configurations.
        rng: Seeded Random instance for deterministic injection.
    """

    def __init__(
        self,
        faults: list[FaultConfig],
        rng: random.Random,
    ) -> None:
        self.faults = list(faults)
        self.rng = rng
        self.injection_log: list[dict[str, Any]] = []

    def maybe_inject(
        self,
        primitive: str,
        params: dict[str, Any],
        world: Any = None,
    ) -> Exception | None:
        """Check all fault configs and maybe raise an exception.

        Returns None if no fault triggers, otherwise returns the exception
        to be raised by SimAdapter.
        """
        for fc in self.faults:
            # Check trigger primitives match
            if fc.trigger_primitives and primitive not in fc.trigger_primitives:
                continue

            # Probability gate
            if self.rng.random() > fc.probability:
                continue

            exc = self._create_fault(fc, primitive, params, world)
            if exc is not None:
                self.injection_log.append({
                    "fault_type": fc.fault_type,
                    "primitive": primitive,
                    "params": params,
                })
                return exc

        return None

    def _create_fault(
        self,
        fc: FaultConfig,
        primitive: str,
        params: dict[str, Any],
        world: Any,
    ) -> Exception | None:
        """Create the appropriate exception for a fault type."""
        ft = fc.fault_type

        if ft == FAULT_DISCONNECTION:
            return ConnectionError(
                f"instrument disconnected during {primitive}"
            )

        if ft == FAULT_TIMEOUT:
            timeout_s = fc.params.get("timeout_s", 300)
            return TimeoutError(
                f"step timed out after {timeout_s}s during {primitive}"
            )

        if ft == FAULT_TIP_SHORTAGE:
            return RuntimeError(
                f"no tips available in rack for {primitive}"
            )

        if ft == FAULT_LIQUID_INSUFFICIENT:
            requested = params.get("volume_ul", params.get("volume", 0))
            return RuntimeError(
                f"insufficient liquid in well: requested {requested} µL"
            )

        if ft == FAULT_DECK_CONFLICT:
            slot = params.get("slot", "unknown")
            return RuntimeError(
                f"deck slot {slot} already occupied"
            )

        if ft == FAULT_SENSOR_DRIFT:
            # Sensor drift is a non-crash fault — modifies world state instead
            # The SimAdapter will handle this via world modification
            drift_pct = fc.params.get("drift_pct", 0.10)
            if world is not None and hasattr(world, "temps"):
                for ch in world.temps:
                    world.temps[ch] *= (1.0 + drift_pct)
            # Don't raise — let execution continue with drifted state
            return None

        if ft == FAULT_TEMP_HYSTERESIS:
            overshoot_c = fc.params.get("overshoot_c", 3.0)
            if world is not None and hasattr(world, "temps"):
                for ch in world.temps:
                    world.temps[ch] += overshoot_c
            return None

        if ft == FAULT_FILE_MISSING:
            filename = params.get("filename", params.get("path", "unknown"))
            return FileNotFoundError(
                f"artifact file not found: {filename}"
            )

        return None
