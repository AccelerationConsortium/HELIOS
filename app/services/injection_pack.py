"""Injection pack builder — cross-object validation and campaign creation.

Converts a confirmed :class:`InjectionPack` (produced by the conversation
engine) into a live campaign, mapping the 6 structured objects to the
existing ``create_campaign()``, ``CampaignGoal``, and ``ParameterSpace``
APIs.

Also provides:
* Cross-object validation (KPI-instrument compat, param-safety bounds)
* Diff summary vs. defaults
* Campaign creation from injection pack
"""
from __future__ import annotations

import logging
from typing import Any

from app.api.v1.schemas_init import InjectionPack, ParamSpaceSpec, SafetyRulesSpec
from app.services.protocol_patterns import build_protocol_from_pattern, get_pattern

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cross-object validation
# ---------------------------------------------------------------------------

# KPI → required instrument mapping (mirrors conversation_engine)
_KPI_INSTRUMENT_MAP: dict[str, str] = {
    "overpotential_mv": "squidstat",
    "current_density_ma_cm2": "squidstat",
    "coulombic_efficiency": "squidstat",
    "stability_decay_pct": "squidstat",
    "charge_passed_c": "squidstat",
    "impedance_ohm": "squidstat",
    "volume_accuracy_pct": "ot2",
    "temp_accuracy_c": "furnace",
}


def validate_injection_pack(pack: InjectionPack) -> list[str]:
    """Run cross-object validation on a complete injection pack.

    Returns a list of error strings (empty = valid).
    """
    errors: list[str] = []

    # 1. KPI-instrument compatibility
    _validate_kpi_instrument(pack, errors)

    # 2. Parameter bounds within safety limits
    _validate_param_safety(pack, errors)

    # 3. Pattern exists and is valid
    _validate_pattern(pack, errors)

    # 4. Human gate policy consistency
    _validate_human_gate(pack, errors)

    return errors


def _validate_kpi_instrument(pack: InjectionPack, errors: list[str]) -> None:
    """Check that the primary KPI's required instrument is in allowed_primitives."""
    kpi = pack.kpi_config.primary_kpi
    required_instrument = _KPI_INSTRUMENT_MAP.get(kpi)
    if not required_instrument:
        return

    # Infer instruments from allowed_primitives
    instrument_prefixes: dict[str, str] = {
        "squidstat": "squidstat.",
        "ot2": "robot.",
        "plc": "plc.",
        "relay": "relay.",
        "furnace": "heat",
    }
    prefix = instrument_prefixes.get(required_instrument, "")
    if prefix and not any(p.startswith(prefix) for p in pack.safety.allowed_primitives):
        errors.append(
            f"KPI '{kpi}' requires instrument '{required_instrument}', "
            f"but no '{prefix}*' primitives found in allowed_primitives"
        )


def _validate_param_safety(pack: InjectionPack, errors: list[str]) -> None:
    """Check that parameter ranges respect safety constraints."""
    max_temp = pack.safety.max_temp_c

    for dim in pack.param_space.dimensions:
        # Temperature params must not exceed safety limit
        if "temp" in dim.param_name.lower() and dim.max_value is not None:
            if dim.max_value > max_temp:
                errors.append(
                    f"Parameter '{dim.param_name}' max_value={dim.max_value} "
                    f"exceeds safety max_temp_c={max_temp}"
                )

        # Volume params must not exceed safety limit
        if "volume" in dim.param_name.lower() and dim.max_value is not None:
            if dim.max_value > pack.safety.max_volume_ul:
                errors.append(
                    f"Parameter '{dim.param_name}' max_value={dim.max_value} "
                    f"exceeds safety max_volume_ul={pack.safety.max_volume_ul}"
                )


def _validate_pattern(pack: InjectionPack, errors: list[str]) -> None:
    """Check that the protocol pattern exists and is valid."""
    pattern = get_pattern(pack.protocol.pattern_id)
    if pattern is None:
        errors.append(f"Unknown protocol pattern '{pack.protocol.pattern_id}'")


def _validate_human_gate(pack: InjectionPack, errors: list[str]) -> None:
    """Basic consistency checks on human gate policy."""
    if pack.human_gate.max_rounds < 1:
        errors.append("max_rounds must be >= 1")

    if pack.human_gate.budget_limit_runs is not None:
        if pack.human_gate.budget_limit_runs < pack.param_space.batch_size:
            errors.append(
                f"budget_limit_runs ({pack.human_gate.budget_limit_runs}) "
                f"is less than batch_size ({pack.param_space.batch_size})"
            )


# ---------------------------------------------------------------------------
# Diff summary vs defaults
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    "objective_type": "oer_screening",
    "direction": "minimize",
    "acceptable_range_pct": 10.0,
    "max_temp_c": 95.0,
    "max_volume_ul": 1000.0,
    "require_human_approval": False,
    "strategy": "lhs",
    "batch_size": 10,
    "max_rounds": 20,
    "plateau_threshold": 0.01,
    "auto_approve_magnitude": 0.3,
}


def build_diff_summary(pack: InjectionPack) -> list[dict[str, Any]]:
    """Generate a diff of the injection pack vs. default values.

    Returns a list of ``{"field": str, "default": Any, "actual": Any}`` dicts
    for every field that differs from the default.
    """
    diffs: list[dict[str, Any]] = []

    actual_values: dict[str, Any] = {
        "objective_type": pack.goal.objective_type,
        "direction": pack.goal.direction,
        "acceptable_range_pct": pack.goal.acceptable_range_pct,
        "max_temp_c": pack.safety.max_temp_c,
        "max_volume_ul": pack.safety.max_volume_ul,
        "require_human_approval": pack.safety.require_human_approval,
        "strategy": pack.param_space.strategy,
        "batch_size": pack.param_space.batch_size,
        "max_rounds": pack.human_gate.max_rounds,
        "plateau_threshold": pack.human_gate.plateau_threshold,
        "auto_approve_magnitude": pack.human_gate.auto_approve_magnitude,
    }

    for field_name, default in _DEFAULTS.items():
        actual = actual_values.get(field_name)
        if actual != default:
            diffs.append({
                "field": field_name,
                "default": default,
                "actual": actual,
            })

    # Always include these non-default-able fields
    diffs.append({
        "field": "objective_kpi",
        "default": None,
        "actual": pack.goal.objective_kpi,
    })
    diffs.append({
        "field": "pattern_id",
        "default": None,
        "actual": pack.protocol.pattern_id,
    })

    n_dims = len(pack.param_space.dimensions)
    diffs.append({
        "field": "optimizable_dimensions",
        "default": None,
        "actual": n_dims,
    })

    if pack.goal.target_value is not None:
        diffs.append({
            "field": "target_value",
            "default": None,
            "actual": pack.goal.target_value,
        })

    if pack.human_gate.budget_limit_runs is not None:
        diffs.append({
            "field": "budget_limit_runs",
            "default": None,
            "actual": pack.human_gate.budget_limit_runs,
        })

    return diffs


# ---------------------------------------------------------------------------
# Map InjectionPack → existing APIs
# ---------------------------------------------------------------------------


def injection_pack_to_campaign_args(pack: InjectionPack) -> dict[str, Any]:
    """Convert an InjectionPack into arguments for ``create_campaign()``.

    Returns a dict with keys: name, cadence_seconds, protocol, inputs,
    policy_snapshot, actor.
    """
    # Build protocol JSON from pattern
    pattern = get_pattern(pack.protocol.pattern_id)
    if pattern is None:
        raise ValueError(f"Unknown pattern '{pack.protocol.pattern_id}'")

    protocol = pattern.to_protocol_json({})
    protocol["name"] = f"{pattern.name} (auto-configured)"
    protocol["version"] = pattern.version

    # Build inputs from goal
    inputs: dict[str, Any] = {
        "objective_kpi": pack.goal.objective_kpi,
        "direction": pack.goal.direction,
    }
    if pack.goal.target_value is not None:
        inputs["target_value"] = pack.goal.target_value

    # Build policy snapshot from safety + human gate
    policy_snapshot: dict[str, Any] = {
        "max_temp_c": pack.safety.max_temp_c,
        "max_volume_ul": pack.safety.max_volume_ul,
        "allowed_primitives": pack.safety.allowed_primitives,
        "require_human_approval": pack.safety.require_human_approval,
        "hazardous_reagents": pack.safety.hazardous_reagents,
        "auto_approve_magnitude": pack.human_gate.auto_approve_magnitude,
        "human_gate_triggers": pack.human_gate.human_gate_triggers,
        "plateau_threshold": pack.human_gate.plateau_threshold,
        "max_rounds": pack.human_gate.max_rounds,
    }
    if pack.human_gate.budget_limit_runs is not None:
        policy_snapshot["budget_limit_runs"] = pack.human_gate.budget_limit_runs

    # Derive campaign name
    name = f"{pack.goal.objective_type}_{pack.goal.objective_kpi}"

    # Cadence default: 5 minutes between scheduled firings
    cadence_seconds = 300

    return {
        "name": name,
        "cadence_seconds": cadence_seconds,
        "protocol": protocol,
        "inputs": inputs,
        "policy_snapshot": policy_snapshot,
        "actor": pack.metadata.created_by,
    }


def injection_pack_to_campaign_goal(pack: InjectionPack) -> dict[str, Any]:
    """Convert InjectionPack to CampaignGoal constructor args."""
    return {
        "objective_kpi": pack.goal.objective_kpi,
        "direction": pack.goal.direction,
        "target_value": pack.goal.target_value,
        "max_rounds": pack.human_gate.max_rounds,
        "batch_size": pack.param_space.batch_size,
        "strategy": pack.param_space.strategy,
    }


def create_campaign_from_pack(pack: InjectionPack) -> dict[str, Any]:
    """Create a campaign in the DB from a confirmed injection pack.

    Returns the campaign dict from ``create_campaign()``.

    Raises ValueError on validation failure.
    """
    # Validate cross-object constraints
    errors = validate_injection_pack(pack)
    if errors:
        raise ValueError(
            "Injection pack validation failed: " + "; ".join(errors)
        )

    from app.services.run_service import create_campaign

    args = injection_pack_to_campaign_args(pack)
    campaign = create_campaign(**args)
    logger.info("Created campaign '%s' from injection pack (session=%s)",
                campaign.get("id"), pack.metadata.session_id)
    return campaign
