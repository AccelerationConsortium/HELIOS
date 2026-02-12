"""Bridge between InjectionPack and TaskContract.

Converts the conversation engine output (InjectionPack) into the
new TaskContract format used by the agent system. This allows the
existing UI conversation flow to feed directly into the orchestrator.
"""
from __future__ import annotations

import logging
from typing import Any

from app.api.v1.schemas_init import InjectionPack
from app.contracts.task_contract import (
    DimensionDef,
    ExplorationSpace,
    HumanGatePolicy,
    ObjectiveSpec,
    SafetyEnvelope,
    StopCondition,
    TaskContract,
    new_task_contract_id,
)
from app.core.db import utcnow_iso

logger = logging.getLogger(__name__)


def _validate_primitives_against_registry(
    allowed_primitives: list[str],
) -> list[str]:
    """Check that allowed_primitives are known to the PrimitivesRegistry.

    Returns a list of warning strings for any unrecognised primitives.
    This is a soft check — warnings are logged but never block execution.
    """
    warnings: list[str] = []
    try:
        from app.services.primitives_registry import get_registry

        registry = get_registry()
        known_names = set(registry.list_primitive_names())
        # Always-available utility primitives that don't require skills
        known_names.update({"wait", "log"})

        for prim in allowed_primitives:
            if prim not in known_names:
                warnings.append(
                    f"Primitive '{prim}' not found in registry — "
                    f"instrument may need onboarding"
                )
    except Exception:
        # Registry unavailable — skip validation
        pass
    return warnings


def injection_pack_to_task_contract(pack: InjectionPack) -> TaskContract:
    """Convert an InjectionPack to a TaskContract.

    Maps the 6 InjectionPack objects to the TaskContract's unified schema.
    The TaskContract adds explicit stop conditions and exploration space
    that are inferred from the InjectionPack's various fields.
    """
    # Cross-validate primitives against the registry
    prim_warnings = _validate_primitives_against_registry(
        pack.safety.allowed_primitives,
    )
    for w in prim_warnings:
        logger.warning("contract_bridge: %s", w)
    # Map dimensions
    dimensions = []
    for dim in pack.param_space.dimensions:
        dimensions.append(DimensionDef(
            param_name=dim.param_name,
            param_type=dim.param_type,
            min_value=dim.min_value,
            max_value=dim.max_value,
            log_scale=dim.log_scale,
            choices=dim.choices,
            step_key=dim.step_key,
            primitive=dim.primitive,
            unit=dim.unit,
        ))

    # Build objective
    objective = ObjectiveSpec(
        objective_type=pack.goal.objective_type,
        primary_kpi=pack.kpi_config.primary_kpi,
        direction=pack.goal.direction,
        secondary_kpis=pack.kpi_config.secondary_kpis,
        acceptable_range_pct=pack.goal.acceptable_range_pct,
    )

    # Build exploration space
    exploration_space = ExplorationSpace(
        dimensions=dimensions,
        forbidden_combinations=pack.param_space.forbidden_combinations,
        strategy=pack.param_space.strategy,
        batch_size=pack.param_space.batch_size,
    )

    # Build stop conditions
    stop_conditions = StopCondition(
        max_rounds=pack.human_gate.max_rounds,
        max_total_runs=pack.human_gate.budget_limit_runs,
        target_kpi_value=pack.goal.target_value,
        target_kpi_direction=pack.goal.direction,
        plateau_threshold=pack.human_gate.plateau_threshold,
    )

    # Build safety envelope
    safety_envelope = SafetyEnvelope(
        max_temp_c=pack.safety.max_temp_c,
        max_volume_ul=pack.safety.max_volume_ul,
        allowed_primitives=pack.safety.allowed_primitives,
        hazardous_reagents=pack.safety.hazardous_reagents,
        require_human_approval=pack.safety.require_human_approval,
    )

    # Build human gate policy
    human_gate = HumanGatePolicy(
        auto_approve_magnitude=pack.human_gate.auto_approve_magnitude,
        triggers=pack.human_gate.human_gate_triggers,
    )

    return TaskContract(
        contract_id=new_task_contract_id(),
        version="1.0",
        created_at=utcnow_iso(),
        created_by=pack.metadata.created_by,
        objective=objective,
        exploration_space=exploration_space,
        stop_conditions=stop_conditions,
        safety_envelope=safety_envelope,
        human_gate=human_gate,
        protocol_pattern_id=pack.protocol.pattern_id,
        protocol_optional_steps=pack.protocol.optional_steps,
        source_session_id=pack.metadata.session_id,
        checksum=pack.metadata.checksum,
    )


def task_contract_to_orchestrator_input(
    contract: TaskContract,
    protocol_template: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert a TaskContract to OrchestratorInput kwargs.

    If protocol_template is not provided, builds one from the
    registered pattern.
    """
    if protocol_template is None:
        from app.services.protocol_patterns import get_pattern
        pattern = get_pattern(contract.protocol_pattern_id)
        if pattern is not None:
            protocol_template = pattern.to_protocol_json({})
        else:
            protocol_template = {"steps": []}

    # Convert dimensions to dict format
    dims = [
        {
            "param_name": d.param_name,
            "param_type": d.param_type,
            "min_value": d.min_value,
            "max_value": d.max_value,
            "log_scale": d.log_scale,
            "choices": d.choices,
            "step_key": d.step_key,
            "primitive": d.primitive,
            "unit": d.unit,
        }
        for d in contract.exploration_space.dimensions
    ]

    # Build policy snapshot from safety envelope
    policy_snapshot = {
        "max_temp_c": contract.safety_envelope.max_temp_c,
        "max_volume_ul": contract.safety_envelope.max_volume_ul,
        "allowed_primitives": contract.safety_envelope.allowed_primitives,
        "hazardous_reagents": contract.safety_envelope.hazardous_reagents,
        "require_human_approval": contract.safety_envelope.require_human_approval,
    }

    return {
        "contract_id": contract.contract_id,
        "objective_kpi": contract.objective.primary_kpi,
        "direction": contract.objective.direction,
        "max_rounds": contract.stop_conditions.max_rounds,
        "batch_size": contract.exploration_space.batch_size,
        "strategy": contract.exploration_space.strategy,
        "target_value": contract.stop_conditions.target_kpi_value,
        "dimensions": dims,
        "protocol_template": protocol_template,
        "policy_snapshot": policy_snapshot,
        "protocol_pattern_id": contract.protocol_pattern_id,
    }
