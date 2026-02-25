"""Protocol dry-run simulator — pre-execution virtual verification.

Simulates a compiled protocol step-by-step without touching hardware:
  1. Deck resource validation  — all labware/well references exist in deck plan
  2. Volume accounting         — per-well liquid tracking, catches aspirate underflows
  3. Tip lifecycle tracking    — pick_up / drop_tip state machine
  4. Parameter bounds check    — volume, temp, current vs. policy limits
  5. Duration estimation       — per-primitive time budget, total vs. policy max

Returns a SimulationResult with a three-level verdict:
  "pass"  → proceed to execution
  "warn"  → proceed with logged warnings (non-blocking violations)
  "fail"  → block execution; orchestrator should abort candidate
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-primitive duration estimates (seconds)
# ---------------------------------------------------------------------------

_PRIMITIVE_DURATION_S: dict[str, float] = {
    # Liquid handling
    "robot.aspirate": 3.0,
    "robot.dispense": 3.0,
    "robot.blowout": 2.0,
    "robot.mix": 10.0,
    "aspirate": 3.0,
    # Tip handling
    "robot.pick_up_tip": 2.0,
    "robot.drop_tip": 2.0,
    "robot.move_to_well": 2.0,
    # Robot setup
    "robot.home": 5.0,
    "robot.load_pipettes": 2.0,
    "robot.load_labware": 2.0,
    "robot.load_custom_labware": 2.0,
    "robot.set_lights": 0.5,
    # Thermal
    "heat": 60.0,
    "cool": 60.0,
    # Electrochemistry
    "squidstat.run_experiment": 120.0,
    "squidstat.get_data": 5.0,
    "squidstat.save_snapshot": 2.0,
    "squidstat.reset_plot": 1.0,
    "eis": 120.0,
    # Hardware
    "plc.dispense_ml": 10.0,
    "plc.set_pump_on_timer": 5.0,
    "plc.set_ultrasonic_on_timer": 30.0,
    "relay.set_channel": 0.5,
    "relay.turn_on": 0.5,
    "relay.turn_off": 0.5,
    "relay.switch_to": 0.5,
    # Utilities
    "wait": 5.0,
    "log": 0.1,
    "upload_artifact": 2.0,
    "ssh.start_stream": 1.0,
    "ssh.stop_stream": 1.0,
    # High-level
    "sample.prepare_from_csv": 30.0,
    "cleanup.run_full": 180.0,
    "cleanup.ultrasonic_water": 60.0,
    "cleanup.ultrasonic_acid": 90.0,
    "cleanup.water_flush": 30.0,
    "cleanup.electrode_clean": 45.0,
}

_DEFAULT_STEP_DURATION_S = 5.0  # fallback for unknown primitives

# Tip state machine: primitives that change tip state
_TIP_PICK_PRIMITIVES = {"robot.pick_up_tip"}
_TIP_DROP_PRIMITIVES = {"robot.drop_tip"}
# Primitives that require a tip to be held
_TIP_REQUIRED_PRIMITIVES = {
    "robot.aspirate",
    "robot.dispense",
    "robot.blowout",
    "robot.mix",
    "robot.move_to_well",
    "aspirate",
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class SimulationViolation:
    """A single issue found during simulation."""

    step_key: str
    primitive: str
    severity: Literal["error", "warning"]
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.step_key} ({self.primitive}): {self.message}"


@dataclass(frozen=True)
class SimulationResult:
    """Full result of a protocol dry-run simulation."""

    verdict: Literal["pass", "warn", "fail"]
    violations: tuple[SimulationViolation, ...]
    warnings: tuple[str, ...]
    estimated_duration_s: float
    resource_summary: dict[str, Any]

    @property
    def errors(self) -> list[SimulationViolation]:
        return [v for v in self.violations if v.severity == "error"]

    @property
    def soft_warnings(self) -> list[SimulationViolation]:
        return [v for v in self.violations if v.severity == "warning"]


# ---------------------------------------------------------------------------
# Core simulation engine
# ---------------------------------------------------------------------------


def simulate_protocol(
    steps: list[dict[str, Any]],
    deck_snapshot: dict[str, Any],
    policy_snapshot: dict[str, Any],
    *,
    initial_volumes: dict[str, float] | None = None,
) -> SimulationResult:
    """Simulate a compiled protocol without hardware.

    Parameters
    ----------
    steps:
        List of compiled step dicts (``step_key``, ``primitive``, ``params``).
    deck_snapshot:
        Deck plan dict as produced by ``DeckPlan.to_dict()``.
        Keys: ``slots`` (dict slot_num → {labware_name, role, ...}).
    policy_snapshot:
        Safety policy dict (``max_volume_ul``, ``max_temp_c``, etc.).
    initial_volumes:
        Optional pre-set well volumes in µL keyed by ``"labware_name:well"``
        (e.g. ``{"src_plate:A1": 500.0}``).

    Returns
    -------
    SimulationResult
        verdict, violations, estimated_duration_s, resource_summary.
    """
    violations: list[SimulationViolation] = []
    general_warnings: list[str] = []

    # Build labware name set from deck_snapshot for resource checks
    known_labware: set[str] = set()
    deck_slots = deck_snapshot.get("slots", {})
    for _slot_num, slot_info in deck_slots.items():
        lname = slot_info.get("labware_name")
        if lname:
            known_labware.add(lname)

    # Volume state: well_key → float (µL available)
    well_volumes: dict[str, float] = dict(initial_volumes or {})
    total_duration_s = 0.0
    has_tip = False
    tip_picks = 0
    tip_drops = 0

    max_volume_ul = float(policy_snapshot.get("max_volume_ul", 1000.0))
    max_temp_c = float(policy_snapshot.get("max_temp_c", 95.0))
    max_current_ma = float(policy_snapshot.get("max_current_ma", 100.0))
    # max_run_duration_s: use a dedicated policy key if provided; otherwise fall back to
    # per-step cap × step count as a conservative upper bound.
    if "max_run_duration_s" in policy_snapshot:
        max_run_duration_s = float(policy_snapshot["max_run_duration_s"])
    else:
        max_run_duration_s = float(policy_snapshot.get("max_step_duration_s", 300.0)) * len(steps)

    for step in steps:
        step_key = step.get("step_key", "?")
        primitive = step.get("primitive", "")
        params: dict[str, Any] = step.get("params") or {}

        # --- 1. Parameter bounds check ---
        try:
            _check_parameter_bounds(
                step_key, primitive, params,
                max_volume_ul, max_temp_c, max_current_ma,
                violations,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Parameter bounds check failed for step %s: %s", step_key, exc)

        # --- 2. Deck resource validation ---
        try:
            _check_deck_resources(
                step_key, primitive, params, known_labware, violations, general_warnings
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Deck resource check failed for step %s: %s", step_key, exc)

        # --- 3. Volume accounting ---
        try:
            _update_volume_state(
                step_key, primitive, params, well_volumes, violations
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Volume state update failed for step %s: %s", step_key, exc)

        # --- 4. Tip lifecycle ---
        has_tip, tip_picks, tip_drops = _update_tip_state(
            step_key, primitive, has_tip, tip_picks, tip_drops, violations
        )

        # --- 5. Duration accumulation ---
        wait_s = float(params.get("wait_s", params.get("wait", 0)))
        prim_duration = _PRIMITIVE_DURATION_S.get(primitive, _DEFAULT_STEP_DURATION_S)
        # For wait primitives, use the actual wait time
        if primitive == "wait" and wait_s > 0:
            prim_duration = wait_s
        total_duration_s += prim_duration

    # --- Tip not dropped at end ---
    if has_tip:
        general_warnings.append("Protocol ends with tip still held (tip not dropped)")

    # --- Duration feasibility ---
    if total_duration_s > max_run_duration_s and max_run_duration_s > 0:
        general_warnings.append(
            f"Estimated run time {total_duration_s:.0f}s exceeds policy cap "
            f"{max_run_duration_s:.0f}s"
        )

    # Build resource summary
    resource_summary: dict[str, Any] = {
        "known_labware": sorted(known_labware),
        "well_volumes_after": dict(well_volumes),
        "tip_picks": tip_picks,
        "tip_drops": tip_drops,
        "estimated_duration_s": total_duration_s,
    }

    # Determine verdict
    errors = [v for v in violations if v.severity == "error"]
    soft = [v for v in violations if v.severity == "warning"]

    if errors:
        verdict: Literal["pass", "warn", "fail"] = "fail"
    elif soft or general_warnings:
        verdict = "warn"
    else:
        verdict = "pass"

    return SimulationResult(
        verdict=verdict,
        violations=tuple(violations),
        warnings=tuple(general_warnings),
        estimated_duration_s=total_duration_s,
        resource_summary=resource_summary,
    )


# ---------------------------------------------------------------------------
# Internal check helpers
# ---------------------------------------------------------------------------


def _check_parameter_bounds(
    step_key: str,
    primitive: str,
    params: dict[str, Any],
    max_volume_ul: float,
    max_temp_c: float,
    max_current_ma: float,
    violations: list[SimulationViolation],
) -> None:
    """Check numeric params against policy bounds."""
    vol = _get_float_param(params, ("volume", "volume_ul", "volume_uL"))
    if vol is not None and vol > max_volume_ul:
        violations.append(SimulationViolation(
            step_key=step_key, primitive=primitive, severity="error",
            message=f"volume {vol:.1f}µL exceeds policy max {max_volume_ul:.1f}µL",
        ))

    temp = _get_float_param(params, ("temperature", "temp", "temperature_c", "temp_c"))
    if temp is not None and temp > max_temp_c:
        violations.append(SimulationViolation(
            step_key=step_key, primitive=primitive, severity="error",
            message=f"temperature {temp:.1f}°C exceeds policy max {max_temp_c:.1f}°C",
        ))

    current = _get_float_param(params, ("max_current", "current", "current_ma"))
    if current is not None and current > max_current_ma:
        violations.append(SimulationViolation(
            step_key=step_key, primitive=primitive, severity="error",
            message=f"current {current:.1f}mA exceeds policy max {max_current_ma:.1f}mA",
        ))


def _check_deck_resources(
    step_key: str,
    primitive: str,
    params: dict[str, Any],
    known_labware: set[str],
    violations: list[SimulationViolation],
    warnings: list[str],
) -> None:
    """Check that referenced labware names exist on the deck."""
    if not known_labware:
        # No deck info provided — skip silently
        return

    for key in ("labware", "labware_name", "source_labware", "dest_labware"):
        lname = params.get(key)
        if lname and isinstance(lname, str) and lname not in known_labware:
            violations.append(SimulationViolation(
                step_key=step_key, primitive=primitive, severity="warning",
                message=f"labware '{lname}' not found in deck plan",
            ))


def _update_volume_state(
    step_key: str,
    primitive: str,
    params: dict[str, Any],
    well_volumes: dict[str, float],
    violations: list[SimulationViolation],
) -> None:
    """Track per-well volumes for aspirate / dispense steps."""
    vol = _get_float_param(params, ("volume", "volume_ul", "volume_uL"))
    if vol is None or vol <= 0:
        return

    labware = params.get("labware", params.get("source_labware", ""))
    well = params.get("well", params.get("source_well", ""))
    well_key = f"{labware}:{well}" if labware and well else None

    if primitive in ("robot.aspirate", "aspirate"):
        if well_key:
            available = well_volumes.get(well_key, math.inf)
            if available < vol:
                violations.append(SimulationViolation(
                    step_key=step_key, primitive=primitive, severity="warning",
                    message=(
                        f"aspirating {vol:.1f}µL from {well_key} but only "
                        f"{'∞' if math.isinf(available) else f'{available:.1f}'}µL tracked"
                    ),
                ))
            # Only subtract if we have tracked volume (inf means unknown)
            if not math.isinf(available):
                well_volumes[well_key] = max(available - vol, 0.0)

    elif primitive in ("robot.dispense",):
        dest_labware = params.get("dest_labware", params.get("labware", ""))
        dest_well = params.get("dest_well", params.get("well", ""))
        dest_key = f"{dest_labware}:{dest_well}" if dest_labware and dest_well else None
        if dest_key:
            well_volumes[dest_key] = well_volumes.get(dest_key, 0.0) + vol


def _update_tip_state(
    step_key: str,
    primitive: str,
    has_tip: bool,
    tip_picks: int,
    tip_drops: int,
    violations: list[SimulationViolation],
) -> tuple[bool, int, int]:
    """Update tip lifecycle state machine and append violations as needed.

    Returns the updated (has_tip, tip_picks, tip_drops) tuple.
    """
    if primitive in _TIP_PICK_PRIMITIVES:
        if has_tip:
            violations.append(SimulationViolation(
                step_key=step_key,
                primitive=primitive,
                severity="error",
                message="pick_up_tip called while already holding a tip",
            ))
        has_tip = True
        tip_picks += 1
    elif primitive in _TIP_DROP_PRIMITIVES:
        if not has_tip:
            violations.append(SimulationViolation(
                step_key=step_key,
                primitive=primitive,
                severity="warning",
                message="drop_tip called without holding a tip",
            ))
        has_tip = False
        tip_drops += 1
    elif primitive in _TIP_REQUIRED_PRIMITIVES and not has_tip:
        violations.append(SimulationViolation(
            step_key=step_key,
            primitive=primitive,
            severity="error",
            message=f"{primitive} requires a tip but none is held",
        ))
    return has_tip, tip_picks, tip_drops


def _get_float_param(
    params: dict[str, Any],
    keys: tuple[str, ...],
) -> float | None:
    """Return the first matching numeric param, or None."""
    for key in keys:
        val = params.get(key)
        if val is not None:
            try:
                f = float(val)
                if not math.isnan(f):
                    return f
            except (ValueError, TypeError):
                pass
    return None
