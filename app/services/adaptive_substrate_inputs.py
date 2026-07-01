"""Read-only input mappers for the adaptive campaign substrate shadow track.

These pure functions translate live round state into the typed inputs the
``adaptive_campaign_substrate`` builder expects. They are deliberately kept
out of the orchestrator so the wiring there stays thin and these mappings stay
unit-testable in isolation.

Boundaries: read-only. No registry mutation, no execution, no routing. The
mapped objects feed a shadow artifact only.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from app.services.action_contracts import SafetyClass
from app.services.dynamic_action_space import ActionSpec
from app.services.failure_attribution import (
    FailureAttributionDistribution,
    attribute_failure,
)
from app.services.failure_signatures import FailureSignature, classify_failure
from app.services.objective_state import ObjectiveState, StoppingCriteria
from app.services.primitives_registry import PrimitiveSpec

__all__ = [
    "action_specs_from_registry",
    "available_capabilities",
    "campaign_instruments_from_protocol",
    "failure_attribution_from_events",
    "objective_state_from_input",
]

#: Severity ranking for choosing the dominant failure event (higher = worse).
_SEVERITY_RANK: dict[str, int] = {
    "CRITICAL": 4,
    "HIGH": 3,
    "MEDIUM": 2,
    "LOW": 1,
    "BYPASS": 0,
}


class _RegistryLike(Protocol):
    def list_primitives(self) -> list[PrimitiveSpec]: ...
    def get_primitive(self, name: str) -> PrimitiveSpec | None: ...
    def list_instruments(self) -> list[str]: ...


def objective_state_from_input(
    *,
    campaign_id: str,
    objective_kpi: str,
    max_rounds: int | None = None,
    target_value: float | None = None,  # noqa: ARG001 (reserved; not a confidence target)
    created_at: datetime | None = None,
) -> ObjectiveState:
    """Construct a per-round snapshot ObjectiveState from campaign input.

    This is a snapshot, not the evolving Phase-1 state: there is no revision
    history and confidence is the neutral default. Cross-round evolution is a
    later step.
    """
    stopping = StoppingCriteria(max_rounds=max_rounds) if max_rounds is not None else None
    fields: dict[str, Any] = {
        "campaign_id": campaign_id,
        "primary_objective": objective_kpi,
        "stopping_criteria": stopping,
    }
    if created_at is not None:
        fields["created_at"] = created_at
    return ObjectiveState(**fields)


def failure_attribution_from_events(
    events: list[dict[str, Any]],
    *,
    now: datetime | None = None,
) -> FailureAttributionDistribution | None:
    """Attribute the most severe failure event, or None when there are none."""
    if not events:
        return None

    signatures = [_signature_from_event(event) for event in events]
    dominant = max(signatures, key=lambda sig: _SEVERITY_RANK.get(sig.severity, 0))
    return attribute_failure(dominant, now=now)


def campaign_instruments_from_protocol(
    protocol_template: dict[str, Any] | None,
    registry: _RegistryLike,
) -> set[str]:
    """Return the set of instruments referenced by the protocol's steps."""
    instruments: set[str] = set()
    if not isinstance(protocol_template, dict):
        return instruments
    for step in protocol_template.get("steps", []) or []:
        if not isinstance(step, dict):
            continue
        primitive_name = step.get("primitive")
        if not primitive_name:
            continue
        spec = registry.get_primitive(str(primitive_name))
        if spec is not None and spec.instrument:
            instruments.add(spec.instrument)
    return instruments


def action_specs_from_registry(
    registry: _RegistryLike,
    *,
    instruments: set[str] | None = None,
    cap: int | None = None,
) -> list[ActionSpec]:
    """Map registry primitives to ActionSpec, filtered to campaign instruments.

    Instrument-less primitives (report / informational / diagnostic-style) are
    always kept. When ``cap`` is set it bounds only the instrumented (experiment)
    actions so the special actions are never dropped. Order is deterministic
    (by primitive name).
    """
    specs = sorted(registry.list_primitives(), key=lambda spec: spec.name)
    instrumentless: list[ActionSpec] = []
    instrumented: list[ActionSpec] = []
    for spec in specs:
        if spec.instrument is None:
            instrumentless.append(_action_spec_from_primitive(spec))
        elif instruments is None or spec.instrument in instruments:
            instrumented.append(_action_spec_from_primitive(spec))
    if cap is not None:
        instrumented = instrumented[:cap]
    return instrumentless + instrumented


def available_capabilities(
    *,
    campaign_instruments: set[str],
    registry: _RegistryLike,
) -> tuple[list[str], str]:
    """Return (capabilities, source). Prefer the campaign subset, else registry."""
    if campaign_instruments:
        return sorted(campaign_instruments), "campaign_deck"
    return sorted(registry.list_instruments()), "registry_fallback"


def _signature_from_event(event: dict[str, Any]) -> FailureSignature:
    return classify_failure(
        step_key=str(event.get("step") or event.get("step_key") or ""),
        primitive=str(event.get("primitive") or ""),
        error_message=str(event.get("error") or event.get("message") or ""),
    )


def _action_spec_from_primitive(spec: PrimitiveSpec) -> ActionSpec:
    contract = spec.contract
    latency = contract.timeout.seconds if contract is not None else None
    context_dependencies = (
        [precondition.predicate for precondition in contract.preconditions]
        if contract is not None
        else []
    )
    return ActionSpec(
        name=spec.name,
        kind=_kind_for_primitive(spec),
        safety_class=spec.safety_class.name.lower(),
        required_capabilities=[spec.instrument] if spec.instrument else [],
        latency=latency,
        context_dependencies=context_dependencies,
    )


#: Single-instrument hardware ops that don't yet declare an ``instrument`` in
#: their skill metadata. Classified as experiments so their missing-instrument
#: metadata gap keeps surfacing downstream. Temporary bridge until the skill
#: files declare an instrument; remove entries as they are fixed.
_INSTRUMENT_PENDING_EXPERIMENTS: frozenset[str] = frozenset({"heat"})


def _kind_for_primitive(spec: PrimitiveSpec) -> str:
    name = spec.name.lower()
    if spec.safety_class == SafetyClass.INFORMATIONAL:
        return "report"
    if name.startswith("cleanup."):
        return "cleanup"
    if name.startswith("sample."):
        return "preparation"
    if "calibrat" in name:
        return "calibration"
    if "diagnos" in name:
        return "diagnostic"
    if spec.instrument is not None:
        return "experiment"
    if name in _INSTRUMENT_PENDING_EXPERIMENTS:
        return "experiment"
    return "workflow"
