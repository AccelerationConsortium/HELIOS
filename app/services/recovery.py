"""Adaptive recovery engine — mid-run repair via procedural memory.

When a step fails after exhausting its retry loop, the recovery engine:
1. Queries procedural memory for matching repair recipes
2. Validates all recipe steps are safe (SafetyClass < HAZARDOUS)
3. Executes recovery steps via the adapter
4. Returns success/failure so the caller can retry the original step

All operations are advisory — failures are swallowed and logged.
Recovery never blocks the existing error classification path.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from typing import Any

from app.adapters.base import InstrumentAdapter
from app.core.db import run_txn
from app.services.action_contracts import LEGACY_SAFETY_MAP, SafetyClass
from app.services.audit import record_event
from app.services.error_policy import classify_step_safety
from app.services.memory import get_repair_recipes, increment_recipe_hit_count
from app.services.run_context import RunContext
from app.services.safety import evaluate_runtime_step

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecoveryPolicy:
    """Run-level policy for adaptive recovery.

    Parsed from the run's ``policy_snapshot["recovery_policy"]`` dict.
    Defaults: enabled, max 2 recovery attempts per step.
    """

    enabled: bool = True
    max_attempts_per_step: int = 2

    @classmethod
    def from_policy_snapshot(cls, policy: dict[str, Any]) -> RecoveryPolicy:
        rp = policy.get("recovery_policy", {})
        if not isinstance(rp, dict):
            return cls()
        return cls(
            enabled=bool(rp.get("enabled", True)),
            max_attempts_per_step=int(rp.get("max_attempts_per_step", 2)),
        )


@dataclass
class RecoveryResult:
    """Outcome of a recovery attempt."""

    attempted: bool  # Whether recovery was attempted at all
    succeeded: bool  # Whether all recovery steps executed successfully
    recipe_used: str | None  # trigger_error_pattern of the recipe used
    steps_executed: int  # Number of recovery steps that ran
    error: str | None  # Error message if recovery itself failed


_NOT_ATTEMPTED = RecoveryResult(
    attempted=False, succeeded=False, recipe_used=None, steps_executed=0, error=None
)


# ---------------------------------------------------------------------------
# Safety validation helpers
# ---------------------------------------------------------------------------


def _get_safety_class(primitive: str) -> SafetyClass:
    """Look up the SafetyClass for a primitive from registry or legacy map.

    Falls back to CAREFUL (safe default) for unknown primitives.
    """
    try:
        from app.services.primitives_registry import get_registry

        registry = get_registry()
        spec = registry.get(primitive)
        if spec is not None and spec.contract is not None:
            return spec.contract.safety_class
    except Exception:
        pass

    return LEGACY_SAFETY_MAP.get(primitive, SafetyClass.CAREFUL)


def _validate_recipe_safety(steps: list[dict[str, Any]]) -> bool:
    """Return True if ALL steps in the recipe have SafetyClass < HAZARDOUS.

    Allows INFORMATIONAL (0), REVERSIBLE (1), and CAREFUL (2).
    Blocks HAZARDOUS (3) — irreversible liquid/experiment operations.
    """
    for step in steps:
        primitive = step.get("primitive", "")
        safety = _get_safety_class(primitive)
        if safety >= SafetyClass.HAZARDOUS:
            logger.debug(
                "Recovery recipe step %s is HAZARDOUS — rejecting recipe", primitive,
            )
            return False
    return True


def _get_contract_for_step(primitive: str) -> Any:
    """Look up the ActionContract for a primitive from the registry."""
    try:
        from app.services.primitives_registry import get_registry

        registry = get_registry()
        spec = registry.get(primitive)
        if spec is not None:
            return spec.contract
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Provenance event recording
# ---------------------------------------------------------------------------


def _record_recovery_event(
    run_id: str, action: str, details: dict[str, Any],
) -> None:
    """Record a recovery provenance event in a new transaction."""
    try:

        def _txn(conn: sqlite3.Connection) -> None:
            record_event(
                conn,
                run_id=run_id,
                actor="recovery-engine",
                action=action,
                details=details,
            )

        run_txn(_txn)
    except Exception:
        logger.debug("Failed to record recovery event %s", action, exc_info=True)


# ---------------------------------------------------------------------------
# Core recovery function
# ---------------------------------------------------------------------------


def attempt_recovery(
    *,
    primitive: str,
    error_text: str,
    run_id: str,
    instrument_id: str,
    adapter: InstrumentAdapter,
    run_context: RunContext,
    policy_snapshot: dict[str, Any],
    step_key: str,
    recovery_policy: RecoveryPolicy,
    recovery_attempt_counts: dict[str, int],
) -> RecoveryResult:
    """Attempt adaptive recovery for a failed step.

    Queries procedural memory for repair recipes matching the failed
    primitive and error pattern.  If a safe recipe is found, executes
    the recovery steps and returns success so the caller can retry.

    All errors are caught — recovery is advisory and never blocks
    the existing error classification path.
    """
    # --- Guard: policy disabled ---
    if not recovery_policy.enabled:
        return _NOT_ATTEMPTED

    # --- Guard: already at max attempts for this step ---
    current_attempts = recovery_attempt_counts.get(step_key, 0)
    if current_attempts >= recovery_policy.max_attempts_per_step:
        logger.debug(
            "step %s: recovery attempt limit reached (%d/%d)",
            step_key, current_attempts, recovery_policy.max_attempts_per_step,
        )
        return _NOT_ATTEMPTED

    # --- Recipe lookup (advisory — never raises) ---
    try:
        recipes = get_repair_recipes(primitive)
    except Exception:
        logger.debug(
            "step %s: memory lookup failed — skipping recovery",
            step_key, exc_info=True,
        )
        return _NOT_ATTEMPTED

    if not recipes:
        return _NOT_ATTEMPTED

    # --- Find matching recipe by error pattern ---
    error_lower = error_text.lower()
    matched_recipe = None
    for recipe in recipes:
        if recipe.trigger_error_pattern in error_lower:
            matched_recipe = recipe
            break

    if matched_recipe is None:
        _record_recovery_event(run_id, "recovery.skipped", {
            "step_key": step_key,
            "primitive": primitive,
            "reason": "no matching error pattern",
            "error_text": error_text[:200],
        })
        return _NOT_ATTEMPTED

    # --- Safety validation ---
    if not _validate_recipe_safety(matched_recipe.steps):
        _record_recovery_event(run_id, "recovery.skipped", {
            "step_key": step_key,
            "primitive": primitive,
            "reason": "recipe contains HAZARDOUS steps",
            "recipe_pattern": matched_recipe.trigger_error_pattern,
        })
        return _NOT_ATTEMPTED

    # --- Record attempt ---
    recovery_attempt_counts[step_key] = current_attempts + 1

    _record_recovery_event(run_id, "recovery.attempted", {
        "step_key": step_key,
        "primitive": primitive,
        "error_text": error_text[:200],
        "recipe_pattern": matched_recipe.trigger_error_pattern,
        "recipe_steps": len(matched_recipe.steps),
    })

    # --- Execute recovery steps ---
    steps_executed = 0
    for i, recipe_step in enumerate(matched_recipe.steps):
        rec_primitive = recipe_step.get("primitive", "")
        rec_params = recipe_step.get("params", {})

        try:
            # Safety gate for each recovery step
            interlock_state = {"hardware_interlock_ok": True, "cooling_ok": True}
            runtime_gate = evaluate_runtime_step(
                step={"primitive": rec_primitive, "params": rec_params},
                policy_snapshot=policy_snapshot,
                interlock_state=interlock_state,
            )
            if not runtime_gate.allowed:
                violations = "; ".join(runtime_gate.violations)
                _record_recovery_event(run_id, "recovery.failed", {
                    "step_key": step_key,
                    "primitive": primitive,
                    "recipe_step_index": i,
                    "recipe_primitive": rec_primitive,
                    "reason": f"safety gate blocked: {violations}",
                })
                return RecoveryResult(
                    attempted=True, succeeded=False,
                    recipe_used=matched_recipe.trigger_error_pattern,
                    steps_executed=steps_executed,
                    error=f"safety gate blocked: {violations}",
                )

            # Execute the recovery step
            adapter.execute_primitive(
                instrument_id=instrument_id,
                primitive=rec_primitive,
                params=rec_params,
            )

            # Apply effects to RunContext
            contract = _get_contract_for_step(rec_primitive)
            if contract is not None:
                for effect in contract.effects:
                    rendered = effect.render(rec_params)
                    run_context.apply_effect(rendered)

            steps_executed += 1

            _record_recovery_event(run_id, "recovery.step_executed", {
                "step_key": step_key,
                "recipe_step_index": i,
                "recipe_primitive": rec_primitive,
                "recipe_params": rec_params,
                "success": True,
            })

        except Exception as exc:
            _record_recovery_event(run_id, "recovery.failed", {
                "step_key": step_key,
                "primitive": primitive,
                "recipe_step_index": i,
                "recipe_primitive": rec_primitive,
                "error": str(exc),
            })
            return RecoveryResult(
                attempted=True, succeeded=False,
                recipe_used=matched_recipe.trigger_error_pattern,
                steps_executed=steps_executed,
                error=str(exc),
            )

    # --- All recovery steps succeeded ---
    _record_recovery_event(run_id, "recovery.succeeded", {
        "step_key": step_key,
        "primitive": primitive,
        "recipe_pattern": matched_recipe.trigger_error_pattern,
        "steps_executed": steps_executed,
    })

    # Increment hit count (advisory — never raises)
    try:
        increment_recipe_hit_count(
            matched_recipe.trigger_primitive,
            matched_recipe.trigger_error_pattern,
        )
    except Exception:
        logger.debug("Failed to increment recipe hit_count", exc_info=True)

    return RecoveryResult(
        attempted=True, succeeded=True,
        recipe_used=matched_recipe.trigger_error_pattern,
        steps_executed=steps_executed,
        error=None,
    )
