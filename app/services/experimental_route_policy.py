"""HELIOS-owned policy and execution gates for experimental-route selection.

The Nexus report is evidence, never an instruction.  HELIOS scores all
reachable alternatives, applies local capability/safety/budget/approval gates,
and only mutates the live route behind an explicit authority flag.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class ExperimentalRouteRuntime:
    node_id: str
    dimensions: tuple[dict[str, Any], ...]
    protocol_template: dict[str, Any]
    protocol_pattern_id: str
    uses_campaign_default: bool = False


@dataclass(frozen=True)
class ExperimentalRouteOption:
    node_id: str
    score: float | None
    eligible: bool
    is_current: bool
    transition_id: str | None = None
    rejection_reasons: tuple[str, ...] = ()
    score_components: dict[str, float] = field(default_factory=dict)
    runtime: ExperimentalRouteRuntime | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.runtime is not None:
            data["runtime"]["dimensions"] = list(self.runtime.dimensions)
        return data


@dataclass(frozen=True)
class ExperimentalRouteDecision:
    active_node_id: str | None
    selected_node_id: str | None
    authority_enabled: bool
    execution_allowed: bool
    applied: bool
    changed: bool
    reason: str
    options: tuple[ExperimentalRouteOption, ...]
    nexus_contract_version: str | None
    nexus_authority: str | None

    @property
    def selected_option(self) -> ExperimentalRouteOption | None:
        return next(
            (option for option in self.options if option.node_id == self.selected_node_id),
            None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_node_id": self.active_node_id,
            "selected_node_id": self.selected_node_id,
            "authority_enabled": self.authority_enabled,
            "execution_allowed": self.execution_allowed,
            "applied": self.applied,
            "changed": self.changed,
            "reason": self.reason,
            "options": [option.to_dict() for option in self.options],
            "nexus_contract_version": self.nexus_contract_version,
            "nexus_authority": self.nexus_authority,
        }


DEFAULT_WEIGHTS = {
    "objective": 0.35,
    "information_gain": 0.25,
    "prior": 0.12,
    "evidence": 0.12,
    "failure": 0.20,
    "safety": 0.22,
    "cost": 0.08,
    "duration": 0.05,
    "switch_cost": 0.08,
    "switch_duration": 0.05,
}


def select_experimental_route(
    *,
    report: dict[str, Any],
    execution_graph: dict[str, Any],
    campaign_dimensions: list[dict[str, Any]],
    campaign_protocol_template: dict[str, Any],
    campaign_protocol_pattern_id: str,
    direction: str,
    authority_enabled: bool,
    available_capabilities: list[str] | None,
    policy_snapshot: dict[str, Any] | None = None,
) -> ExperimentalRouteDecision:
    """Evaluate a Nexus report under HELIOS policy and return an audit bundle."""
    policy = dict(policy_snapshot or {})
    # The report graph is external evidence and must never supply executable
    # protocol content.  Only the campaign graph accepted by HELIOS can do so.
    graph = dict(execution_graph or {})
    active_node_id = _text(graph.get("active_node_id"))
    contract = _text(report.get("contract_version"))
    nexus_authority = _text(report.get("authority"))
    if contract != "experimental_route_intelligence.v1" or nexus_authority != "advisory_only":
        return ExperimentalRouteDecision(
            active_node_id=active_node_id,
            selected_node_id=active_node_id,
            authority_enabled=authority_enabled,
            execution_allowed=False,
            applied=False,
            changed=False,
            reason="Nexus route evidence contract or authority is invalid; current route retained.",
            options=(),
            nexus_contract_version=contract,
            nexus_authority=nexus_authority,
        )

    nodes = {
        str(node.get("node_id")): node
        for node in graph.get("nodes", []) or []
        if isinstance(node, dict) and node.get("node_id")
    }
    assessments = {
        str(item.get("node_id")): item
        for item in report.get("node_assessments", []) or []
        if isinstance(item, dict) and item.get("node_id")
    }
    nexus_transitions = {
        str(item.get("target_id")): item
        for item in report.get("available_transitions", []) or []
        if isinstance(item, dict) and item.get("target_id")
    }
    local_transitions = {
        str(item.get("target_id")): item
        for item in graph.get("transitions", []) or []
        if (
            isinstance(item, dict)
            and item.get("target_id")
            and str(item.get("source_id")) == active_node_id
        )
    }
    transitions = {
        target_id: {
            **local_transition,
            "transition_id": f"{active_node_id}->{target_id}",
            "target_status": nexus_transitions.get(target_id, {}).get(
                "target_status"
            ),
            "target_evidence_strength": nexus_transitions.get(target_id, {}).get(
                "target_evidence_strength"
            ),
            "target_missing_capabilities": nexus_transitions.get(target_id, {}).get(
                "target_missing_capabilities", []
            ),
            "target_capability_status": nexus_transitions.get(target_id, {}).get(
                "target_capability_status", "unknown"
            ),
        }
        for target_id, local_transition in local_transitions.items()
        if target_id in nexus_transitions
    }
    candidate_ids = set(transitions)
    if active_node_id:
        candidate_ids.add(active_node_id)

    weights = dict(DEFAULT_WEIGHTS)
    supplied_weights = policy.get("experimental_route_weights")
    if isinstance(supplied_weights, dict):
        for name in weights:
            value = supplied_weights.get(name)
            if (
                isinstance(value, int | float)
                and isfinite(float(value))
                and value >= 0
            ):
                weights[name] = float(value)

    objective_utilities = _objective_utilities(assessments, direction)
    max_safety = _number(policy.get("experimental_route_max_safety_risk"), 0.6)
    max_cost = _optional_number(policy.get("experimental_route_max_expected_cost"))
    max_duration = _optional_number(
        policy.get("experimental_route_max_expected_duration_s")
    )
    approved = {
        str(item)
        for item in policy.get("approved_experimental_route_transitions", []) or []
    }
    capability_inventory_supplied = available_capabilities is not None
    local_capabilities = set(available_capabilities or [])

    options: list[ExperimentalRouteOption] = []
    for node_id in sorted(candidate_ids):
        node = nodes.get(node_id, {})
        assessment = assessments.get(node_id, {})
        transition = transitions.get(node_id)
        is_current = node_id == active_node_id
        transition_id = _text((transition or {}).get("transition_id"))
        rejection_reasons: list[str] = []

        capability_status = _text((transition or {}).get("target_capability_status"))
        required_capabilities = {
            str(item) for item in node.get("required_capabilities", []) or []
        }
        locally_missing_caps = sorted(required_capabilities - local_capabilities)
        missing_caps = list(
            (transition or {}).get("target_missing_capabilities")
            or assessment.get("missing_capabilities")
            or []
        )
        if not is_current and not capability_inventory_supplied:
            rejection_reasons.append("capability_inventory_unknown")
        if not is_current and capability_status == "unknown":
            rejection_reasons.append("target_capability_status_unknown")
        if not is_current and (
            capability_status == "missing"
            or missing_caps
            or locally_missing_caps
        ):
            rejection_reasons.append("missing_required_capabilities")
        if assessment.get("status") == "capability_blocked":
            rejection_reasons.append("capability_blocked")

        safety_risk = _number(node.get("safety_risk"), 0.0)
        expected_cost = _number(node.get("expected_cost"), 1.0)
        expected_duration = _number(node.get("expected_duration_s"), 0.0)
        if safety_risk > max_safety:
            rejection_reasons.append("safety_risk_above_policy")
        if max_cost is not None and expected_cost > max_cost:
            rejection_reasons.append("expected_cost_above_budget")
        if max_duration is not None and expected_duration > max_duration:
            rejection_reasons.append("expected_duration_above_budget")

        runtime = resolve_experimental_route_runtime(
            node=node,
            is_current=is_current,
            campaign_dimensions=campaign_dimensions,
            campaign_protocol_template=campaign_protocol_template,
            campaign_protocol_pattern_id=campaign_protocol_pattern_id,
        )
        if runtime is None:
            rejection_reasons.append("route_has_no_executable_helios_mapping")

        if (
            not is_current
            and bool((transition or {}).get("approval_required", True))
            and transition_id not in approved
            and f"{active_node_id}->{node_id}" not in approved
        ):
            rejection_reasons.append("operator_approval_required")

        failure_rate = _number(assessment.get("failure_rate"), 0.0)
        info_gap = _number(assessment.get("information_gap"), 1.0)
        prior = _number(assessment.get("normalized_prior"), 0.0)
        evidence = _number(assessment.get("evidence_strength"), 0.0)
        switch_cost = _number((transition or {}).get("switch_cost"), 0.0)
        switch_duration = _number((transition or {}).get("switch_duration_s"), 0.0)
        components = {
            "objective": weights["objective"] * objective_utilities.get(node_id, 0.5),
            "information_gain": weights["information_gain"] * info_gap,
            "prior": weights["prior"] * prior,
            "evidence": weights["evidence"] * evidence,
            "failure_penalty": -weights["failure"] * failure_rate,
            "safety_penalty": -weights["safety"] * safety_risk,
            "cost_penalty": -weights["cost"] * _bounded_cost(expected_cost),
            "duration_penalty": -weights["duration"] * _bounded_cost(expected_duration / 3600.0),
            "switch_cost_penalty": -weights["switch_cost"] * _bounded_cost(switch_cost),
            "switch_duration_penalty": -weights["switch_duration"]
            * _bounded_cost(switch_duration / 3600.0),
        }
        score = round(sum(components.values()), 9)
        options.append(
            ExperimentalRouteOption(
                node_id=node_id,
                score=score,
                eligible=not rejection_reasons,
                is_current=is_current,
                transition_id=transition_id,
                rejection_reasons=tuple(sorted(set(rejection_reasons))),
                score_components=components,
                runtime=runtime,
            )
        )

    eligible = [option for option in options if option.eligible]
    selected = (
        max(
            eligible,
            key=lambda item: (
                item.score if item.score is not None else float("-inf"),
                item.node_id,
            ),
        )
        if eligible
        else None
    )
    selected_id = selected.node_id if selected is not None else None
    changed = selected_id is not None and selected_id != active_node_id
    applied = bool(authority_enabled and selected is not None and changed)
    current_option = next(
        (option for option in options if option.node_id == active_node_id),
        None,
    )
    execution_allowed = bool(
        (selected is not None and selected.eligible)
        if applied
        else (current_option is not None and current_option.eligible)
    )
    if selected is None:
        reason = "No route passed HELIOS capability, safety, budget, approval, and execution gates."
    elif not authority_enabled and changed:
        reason = f"Shadow policy prefers {selected_id}; live route authority is disabled."
    elif applied:
        reason = f"HELIOS selected and applied experimental route {selected_id}."
    else:
        reason = f"HELIOS retained experimental route {selected_id}."
    return ExperimentalRouteDecision(
        active_node_id=active_node_id,
        selected_node_id=selected_id,
        authority_enabled=authority_enabled,
        execution_allowed=execution_allowed,
        applied=applied,
        changed=applied,
        reason=reason,
        options=tuple(options),
        nexus_contract_version=contract,
        nexus_authority=nexus_authority,
    )


def resolve_experimental_route_runtime(
    *,
    node: dict[str, Any],
    is_current: bool,
    campaign_dimensions: list[dict[str, Any]],
    campaign_protocol_template: dict[str, Any],
    campaign_protocol_pattern_id: str,
) -> ExperimentalRouteRuntime | None:
    """Resolve a graph node into concrete HELIOS design/compile inputs."""
    node_id = _text(node.get("node_id"))
    if node_id is None:
        return None
    raw_dimensions = node.get("parameter_space")
    dimensions = normalize_route_dimensions(raw_dimensions) if raw_dimensions else []
    protocol_ref = node.get("protocol_ref") if isinstance(node.get("protocol_ref"), dict) else {}
    template = protocol_ref.get("protocol_template", protocol_ref.get("template"))
    pattern_id = _text(protocol_ref.get("protocol_pattern_id")) or ""
    use_default = bool(protocol_ref.get("use_campaign_default")) or (
        is_current and not raw_dimensions and not protocol_ref
    )
    if not dimensions and use_default:
        dimensions = [dict(item) for item in campaign_dimensions]
    if not isinstance(template, dict) and use_default:
        template = dict(campaign_protocol_template)
    if not pattern_id and use_default:
        pattern_id = campaign_protocol_pattern_id
    if (
        not dimensions
        or not all(_dimension_is_executable(item) for item in dimensions)
        or (not isinstance(template, dict) and not pattern_id)
    ):
        return None
    if pattern_id:
        from app.services.protocol_patterns import get_pattern

        if get_pattern(pattern_id) is None:
            return None
    return ExperimentalRouteRuntime(
        node_id=node_id,
        dimensions=tuple(dimensions),
        protocol_template=dict(template or {}),
        protocol_pattern_id=pattern_id,
        uses_campaign_default=use_default and not raw_dimensions,
    )


def normalize_route_dimensions(raw_dimensions: Any) -> list[dict[str, Any]]:
    """Accept Nexus-style or HELIOS-style parameter-space dictionaries."""
    normalized: list[dict[str, Any]] = []
    if not isinstance(raw_dimensions, list):
        return normalized
    for raw in raw_dimensions:
        if not isinstance(raw, dict):
            continue
        name = raw.get("param_name", raw.get("name"))
        if not name:
            continue
        item = {
            "param_name": str(name),
            "param_type": raw.get("param_type", raw.get("type", "number")),
            "min_value": raw.get("min_value", raw.get("min", raw.get("lower"))),
            "max_value": raw.get("max_value", raw.get("max", raw.get("upper"))),
            "log_scale": bool(raw.get("log_scale", False)),
        }
        for key in ("choices", "step_key", "primitive"):
            if key in raw:
                item[key] = raw[key]
        normalized.append(item)
    return normalized


def _objective_utilities(
    assessments: dict[str, dict[str, Any]], direction: str
) -> dict[str, float]:
    best_by_node: dict[str, float] = {}
    for node_id, assessment in assessments.items():
        summaries = assessment.get("objective_summaries", []) or []
        if summaries and isinstance(summaries[0], dict):
            value = summaries[0].get("best")
            if isinstance(value, int | float):
                best_by_node[node_id] = float(value)
    if not best_by_node:
        return {}
    low, high = min(best_by_node.values()), max(best_by_node.values())
    if high == low:
        return {node_id: 0.5 for node_id in best_by_node}
    utilities = {
        node_id: (value - low) / (high - low)
        for node_id, value in best_by_node.items()
    }
    if direction == "minimize":
        utilities = {node_id: 1.0 - value for node_id, value in utilities.items()}
    return utilities


def _bounded_cost(value: float) -> float:
    value = max(0.0, value)
    return value / (1.0 + value)


def _dimension_is_executable(dimension: dict[str, Any]) -> bool:
    choices = dimension.get("choices")
    if choices is not None:
        return isinstance(choices, list | tuple) and bool(choices)
    if dimension.get("param_type") == "boolean":
        return True
    low = dimension.get("min_value")
    high = dimension.get("max_value")
    return (
        isinstance(low, int | float)
        and isinstance(high, int | float)
        and isfinite(float(low))
        and isfinite(float(high))
        and float(low) <= float(high)
    )


def _number(value: Any, default: float) -> float:
    if isinstance(value, int | float) and isfinite(float(value)):
        return float(value)
    return default


def _optional_number(value: Any) -> float | None:
    if isinstance(value, int | float) and isfinite(float(value)):
        return float(value)
    return None


def _text(value: Any) -> str | None:
    return str(value) if value is not None and str(value) else None
