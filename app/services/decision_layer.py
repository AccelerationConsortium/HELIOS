"""Thin campaign decision layer above dynamic strategy selection.

Phase 1 is an envelope adapter only. It does not change campaign runtime
behavior or reimplement the existing strategy selector.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from pydantic import ValidationError

from app.contracts.scientific_evidence import (
    ScientificEvidencePolicyMode,
    ScientificEvidenceRecommendedAction,
)
from app.services.decision_models import (
    CampaignContextRequest,
    CampaignDecisionAction,
    CampaignDecisionEvidence,
    CampaignDecisionPlan,
    CampaignRoundContext,
    ConstraintPatch,
    ObjectivePatch,
)

__all__ = ["CampaignDecisionLayer", "strategy_decision_to_payload"]


class CampaignDecisionLayer:
    """Wrap round context and strategy-selection output in a decision envelope."""

    def decide(self, context: CampaignRoundContext) -> CampaignDecisionPlan:
        """Return a shadow-only campaign decision plan for one round."""
        if context.stop_requested:
            return CampaignDecisionPlan(
                action_type=CampaignDecisionAction.STOP_CAMPAIGN,
                rationale="Stop was requested; campaign should stop before further actions.",
                confidence=1.0,
                shadow_only=True,
                evidence=[_summary_evidence("operator_control", "stop_requested", context.metadata)],
            )

        if _is_blocking_failure(context.failure_summary):
            return CampaignDecisionPlan(
                action_type=CampaignDecisionAction.RECOVER_FAILURE,
                rationale="Blocking failure requires recovery before candidate generation.",
                confidence=0.9,
                shadow_only=True,
                evidence=[_summary_evidence("failure_summary", "blocking_failure", context.failure_summary)],
            )

        if _is_high_safety_risk(context.safety_summary):
            return CampaignDecisionPlan(
                action_type=CampaignDecisionAction.TIGHTEN_CONSTRAINTS,
                constraint_patch=ConstraintPatch(
                    reason="Safety risk requires constraint update before proceeding.",
                    proposed_changes=dict(context.safety_summary),
                    shadow_only=True,
                ),
                rationale="Safety risk requires constraint update before proceeding.",
                confidence=0.85,
                shadow_only=True,
                evidence=[_summary_evidence("safety_summary", "constraint_risk", context.safety_summary)],
            )

        if _drift_requires_context_review(context.drift_summary):
            return CampaignDecisionPlan(
                action_type=CampaignDecisionAction.REQUEST_HUMAN_OBSERVATION,
                route_target="decision_context_review",
                context_requests=[
                    CampaignContextRequest(
                        request_type="decision_context_completion",
                        reason=(
                            "Closed-loop drift monitoring found missing decision, "
                            "override, or failure context from prior rounds."
                        ),
                        priority="high",
                        target="decision_memory",
                        payload={
                            "drift_report_id": context.drift_summary.get(
                                "report_id"
                            ),
                            "omissions": list(
                                context.decision_memory.get("omissions", [])
                            )[:8],
                        },
                    )
                ],
                rationale=(
                    "Missing prior-round context can amplify closed-loop error; "
                    "complete the context before generating more candidates."
                ),
                confidence=_drift_confidence(context.drift_summary),
                shadow_only=True,
                evidence=[
                    _summary_evidence(
                        "closed_loop_drift",
                        "context_drift",
                        context.drift_summary,
                    )
                ],
            )

        if _drift_requires_objective_review(context.drift_summary):
            return CampaignDecisionPlan(
                action_type=CampaignDecisionAction.REVISE_OBJECTIVE,
                route_target="objective_revision",
                objective_patch=ObjectivePatch(
                    reason=(
                        "Closed-loop monitoring found an expanding gap between "
                        "the proxy and the scientific objective."
                    ),
                    proposed_changes={
                        "drift_report": _json_safe(context.drift_summary),
                        "auto_applied": False,
                    },
                    shadow_only=True,
                ),
                rationale=(
                    "Target drift requires objective review before further proxy optimization."
                ),
                confidence=_drift_confidence(context.drift_summary),
                shadow_only=True,
                evidence=[
                    _summary_evidence(
                        "closed_loop_drift",
                        "target_drift",
                        context.drift_summary,
                    )
                ],
            )

        if _drift_requires_validation(context.drift_summary):
            return CampaignDecisionPlan(
                action_type=CampaignDecisionAction.RUN_VALIDATION,
                route_target="closed_loop_drift_validation",
                rationale=(
                    "Closed-loop drift signals require validation before more candidate generation."
                ),
                confidence=_drift_confidence(context.drift_summary),
                shadow_only=True,
                evidence=[
                    _summary_evidence(
                        "closed_loop_drift",
                        "validation_required",
                        context.drift_summary,
                    )
                ],
                metadata={"drift_report_id": context.drift_summary.get("report_id")},
            )

        scientific_evidence_plan = _scientific_evidence_plan(context)
        if scientific_evidence_plan is not None:
            return scientific_evidence_plan

        if _is_validation_due(context.validation_summary):
            return CampaignDecisionPlan(
                action_type=CampaignDecisionAction.RUN_VALIDATION,
                route_target="validation",
                rationale="Validation is due before more candidate generation.",
                confidence=_confidence_with_default(context.validation_summary, 0.7),
                shadow_only=True,
                evidence=[
                    CampaignDecisionEvidence(
                        source="validation_summary",
                        kind="validation_due",
                        summary="Validation summary indicated validation is due.",
                        payload=dict(context.validation_summary),
                    )
                ],
            )

        if _has_conflicting_objective_signals(context.objective_summary):
            return CampaignDecisionPlan(
                action_type=CampaignDecisionAction.REQUEST_HUMAN_OBSERVATION,
                route_target="human_observation",
                context_requests=[
                    CampaignContextRequest(
                        request_type="objective_disambiguation",
                        reason=(
                            "Objective summary contains conflicting signals; "
                            "human observation is needed before objective routing."
                        ),
                        priority="high",
                        target="objective_summary",
                        payload=dict(context.objective_summary),
                    )
                ],
                rationale=(
                    "Conflicting objective signals make campaign-level routing "
                    "unreliable without human context."
                ),
                confidence=_confidence_with_default(context.objective_summary, 0.65),
                shadow_only=True,
                evidence=[
                    CampaignDecisionEvidence(
                        source="objective_summary",
                        kind="objective_signal_conflict",
                        summary="Objective summary indicated conflicting signals.",
                        payload=dict(context.objective_summary),
                    )
                ],
            )

        proxy_gap_score = _objective_proxy_gap_score(context.objective_summary)
        if _is_high_objective_proxy_gap(context.objective_summary, proxy_gap_score):
            return CampaignDecisionPlan(
                action_type=CampaignDecisionAction.REVISE_OBJECTIVE,
                route_target="objective_revision",
                objective_patch=ObjectivePatch(
                    reason=(
                        "Active objective is too far from functional scientific "
                        "performance."
                    ),
                    proposed_changes={
                        "proxy_gap_assessment": _json_safe(context.objective_summary)
                    },
                    shadow_only=True,
                ),
                rationale=(
                    "High objective proxy gap suggests revising the active "
                    "objective before more candidate generation."
                ),
                confidence=_objective_proxy_gap_confidence(
                    context.objective_summary,
                    proxy_gap_score,
                ),
                shadow_only=True,
                evidence=[
                    CampaignDecisionEvidence(
                        source="objective_summary",
                        kind="proxy_gap",
                        summary="Objective summary indicated a high proxy gap.",
                        payload=dict(context.objective_summary),
                    )
                ],
            )

        if _has_plateau_with_missing_literature(context):
            return CampaignDecisionPlan(
                action_type=CampaignDecisionAction.QUERY_LITERATURE,
                route_target="literature",
                context_requests=[
                    CampaignContextRequest(
                        request_type="literature_context",
                        reason=(
                            "Campaign appears plateaued and literature context "
                            "is missing."
                        ),
                        priority="high",
                        target="literature_summary",
                        payload={
                            "nexus_diagnostics": dict(context.nexus_diagnostics),
                            "objective_summary": dict(context.objective_summary),
                            "literature_summary": dict(context.literature_summary),
                        },
                    )
                ],
                rationale=(
                    "Plateau with missing literature context should request "
                    "literature before more candidate generation."
                ),
                confidence=_plateau_confidence(context),
                shadow_only=True,
                evidence=[
                    CampaignDecisionEvidence(
                        source="contextual_decision_layer",
                        kind="plateau_literature_missing",
                        summary="Plateau signal was present while literature context was missing.",
                        payload={
                            "nexus_diagnostics": dict(context.nexus_diagnostics),
                            "objective_summary": dict(context.objective_summary),
                            "literature_summary": dict(context.literature_summary),
                        },
                    )
                ],
            )

        if _has_low_failure_attribution_confidence(context.failure_summary):
            return CampaignDecisionPlan(
                action_type=CampaignDecisionAction.REQUEST_HUMAN_OBSERVATION,
                route_target="human_observation",
                context_requests=[
                    CampaignContextRequest(
                        request_type="failure_attribution",
                        reason=(
                            "Failure attribution confidence is low; human "
                            "observation is needed before routing."
                        ),
                        priority="high",
                        target="failure_summary",
                        payload=dict(context.failure_summary),
                    )
                ],
                rationale=(
                    "Low failure-attribution confidence makes the next campaign "
                    "route unreliable without human observation."
                ),
                confidence=_failure_attribution_request_confidence(
                    context.failure_summary
                ),
                shadow_only=True,
                evidence=[
                    CampaignDecisionEvidence(
                        source="failure_summary",
                        kind="low_failure_attribution_confidence",
                        summary="Failure summary had low attribution confidence.",
                        payload=dict(context.failure_summary),
                    )
                ],
            )

        plan = self._wrap_strategy_result(context.strategy_selection_result)
        if _needs_backend_memory_context(context):
            _add_backend_memory_context_request(plan, context)
        return _attach_scientific_evidence(plan, context)

    def _wrap_strategy_result(self, result: dict[str, Any]) -> CampaignDecisionPlan:
        evidence = _strategy_evidence(result.get("evidence"))
        evidence.insert(
            0,
            CampaignDecisionEvidence(
                source="dynamic_strategy_selector",
                kind="strategy_selection",
                summary=(
                    "Existing dynamic strategy-selection result was wrapped "
                    "as a campaign decision envelope."
                ),
                payload={
                    "campaign_intent": _first_present(result, "campaign_intent", "intent"),
                    "optimization_mode": _first_present(result, "optimization_mode", "mode"),
                    "candidate_generation_backend": _first_present(
                        result,
                        "candidate_generation_backend",
                        "backend",
                        "selected_backend",
                    ),
                },
            ),
        )
        confidence = _confidence(result)
        return CampaignDecisionPlan(
            action_type=CampaignDecisionAction.PROPOSE_CANDIDATES,
            campaign_intent=_first_present(result, "campaign_intent", "intent"),
            optimization_mode=_first_present(result, "optimization_mode", "mode"),
            candidate_generation_backend=_first_present(
                result,
                "candidate_generation_backend",
                "backend",
                "selected_backend",
            ),
            strategy_trace=_strategy_trace(result),
            evidence=evidence,
            rationale="Dynamic strategy-selection result supports proposing the next candidate batch.",
            confidence=confidence,
            fallback_action=_fallback_action(result.get("fallback_action")),
            shadow_only=True,
            metadata={"wrapped_from": "dynamic_strategy_selector"},
        )


def strategy_decision_to_payload(decision: Any) -> dict[str, Any]:
    """Convert a StrategyDecision-like object to a JSON-compatible payload."""
    if decision is None:
        return {}
    if is_dataclass(decision):
        raw = asdict(decision)
    elif isinstance(decision, dict):
        raw = dict(decision)
    else:
        raw = {
            "backend": getattr(decision, "backend_name", None),
            "phase": getattr(decision, "phase", None),
            "reason": getattr(decision, "reason", None),
            "confidence": getattr(decision, "confidence", None),
            "strategy_trace": getattr(decision, "strategy_trace", None),
            "evidence": getattr(decision, "evidence", None),
        }
    payload = _json_safe(raw)
    if isinstance(payload, dict) and "backend" not in payload and payload.get("backend_name") is not None:
        payload["backend"] = payload["backend_name"]
    return payload


def _is_blocking_failure(summary: dict[str, Any]) -> bool:
    return (
        summary.get("blocking") is True
        or summary.get("severity") == "blocking"
        or summary.get("requires_recovery") is True
    )


def _is_high_safety_risk(summary: dict[str, Any]) -> bool:
    return (
        summary.get("blocking") is True
        or summary.get("risk_level") in {"high", "blocking", "critical"}
        or summary.get("requires_constraint_update") is True
    )


def _is_validation_due(summary: dict[str, Any]) -> bool:
    return (
        summary.get("validation_due") is True
        or summary.get("requires_validation") is True
        or summary.get("due") is True
        or summary.get("status") == "due"
    )


def _drift_requires_validation(summary: dict[str, Any]) -> bool:
    return summary.get("requires_validation") is True


def _drift_requires_objective_review(summary: dict[str, Any]) -> bool:
    return summary.get("requires_objective_review") is True


def _drift_requires_context_review(summary: dict[str, Any]) -> bool:
    return summary.get("requires_context_review") is True


def _drift_confidence(summary: dict[str, Any]) -> float:
    scores = [
        _as_float(signal.get("score"))
        for signal in summary.get("signals", [])
        if isinstance(signal, dict) and signal.get("score") is not None
    ]
    finite_scores = [score for score in scores if score is not None]
    if finite_scores:
        return min(0.95, max(0.5, sum(finite_scores) / len(finite_scores)))
    status = summary.get("overall_status")
    if status == "drift":
        return 0.85
    if status == "watch":
        return 0.6
    return 0.5


def _scientific_evidence_plan(
    context: CampaignRoundContext,
) -> CampaignDecisionPlan | None:
    assessment = context.scientific_evidence_assessment
    if (
        assessment is None
        or assessment.policy_mode != ScientificEvidencePolicyMode.BOUNDED
    ):
        return None

    action = assessment.recommended_action
    if action == ScientificEvidenceRecommendedAction.NONE:
        return None
    payload = assessment.model_dump(mode="json")
    evidence = CampaignDecisionEvidence(
        source="pas_scientific_evidence",
        kind="scientific_evidence_assessment",
        summary=(
            f"PAS evidence assessment recommended {action.value}; "
            "HELIOS retained execution authority."
        ),
        payload=payload,
    )
    metadata = {
        "scientific_evidence_policy_mode": assessment.policy_mode.value,
        "scientific_evidence_bundle_id": assessment.bundle_id,
    }
    if action == ScientificEvidenceRecommendedAction.RUN_VALIDATION:
        return CampaignDecisionPlan(
            action_type=CampaignDecisionAction.RUN_VALIDATION,
            route_target="scientific_evidence_validation",
            rationale="Bounded scientific evidence requires validation review.",
            confidence=max(assessment.support_strength, assessment.contradiction_strength),
            shadow_only=True,
            evidence=[evidence],
            metadata=metadata,
        )
    if action == ScientificEvidenceRecommendedAction.QUERY_LITERATURE:
        return CampaignDecisionPlan(
            action_type=CampaignDecisionAction.QUERY_LITERATURE,
            route_target="literature",
            context_requests=[
                CampaignContextRequest(
                    request_type="scientific_evidence_refresh",
                    reason="PAS evidence is stale or insufficient.",
                    priority="high",
                    target="scientific_evidence",
                    payload={"bundle_id": assessment.bundle_id},
                )
            ],
            rationale="Bounded scientific evidence requires a literature refresh.",
            confidence=max(0.5, assessment.support_strength),
            shadow_only=True,
            evidence=[evidence],
            metadata=metadata,
        )
    if action == ScientificEvidenceRecommendedAction.REQUEST_HUMAN_OBSERVATION:
        return CampaignDecisionPlan(
            action_type=CampaignDecisionAction.REQUEST_HUMAN_OBSERVATION,
            route_target="scientific_evidence_review",
            context_requests=[
                CampaignContextRequest(
                    request_type="scientific_evidence_applicability",
                    reason="PAS evidence applicability requires operator review.",
                    priority="high",
                    target="scientific_evidence",
                    payload={"bundle_id": assessment.bundle_id},
                )
            ],
            rationale="Bounded scientific evidence requires human applicability review.",
            confidence=max(0.5, assessment.applicability_score),
            shadow_only=True,
            evidence=[evidence],
            metadata=metadata,
        )
    return None


def _attach_scientific_evidence(
    plan: CampaignDecisionPlan,
    context: CampaignRoundContext,
) -> CampaignDecisionPlan:
    assessment = context.scientific_evidence_assessment
    if assessment is None or assessment.policy_mode == ScientificEvidencePolicyMode.OFF:
        return plan
    plan.metadata.update(
        {
            "scientific_evidence_policy_mode": assessment.policy_mode.value,
            "scientific_evidence_bundle_id": assessment.bundle_id,
        }
    )
    plan.evidence.append(
        CampaignDecisionEvidence(
            source="pas_scientific_evidence",
            kind="scientific_evidence_assessment",
            summary=(
                f"PAS evidence was recorded in {assessment.policy_mode.value} mode "
                "without changing live execution authority."
            ),
            payload=assessment.model_dump(mode="json"),
        )
    )
    return plan


def _is_high_objective_proxy_gap(
    summary: dict[str, Any],
    proxy_gap_score: float | None,
) -> bool:
    return (
        _lower(summary.get("proxy_gap_level")) == "high"
        or _lower(summary.get("proxy_gap")) == "high"
        or _nested_proxy_gap_level(summary) == "high"
        or (proxy_gap_score is not None and proxy_gap_score >= 0.6)
    )


def _has_conflicting_objective_signals(summary: dict[str, Any]) -> bool:
    return (
        summary.get("conflicting_signals") is True
        or summary.get("objective_conflict") is True
        or summary.get("conflict") is True
        or _has_items(summary.get("conflicts"))
        or _has_items(summary.get("signal_conflicts"))
    )


def _has_plateau_with_missing_literature(context: CampaignRoundContext) -> bool:
    return _has_plateau_signal(context) and _is_literature_missing(
        context.literature_summary
    )


def _has_plateau_signal(context: CampaignRoundContext) -> bool:
    summaries = (
        context.nexus_diagnostics,
        context.objective_summary,
        context.strategy_selection_result,
        context.metadata,
    )
    return any(_summary_has_plateau(summary) for summary in summaries)


def _summary_has_plateau(summary: dict[str, Any]) -> bool:
    return (
        summary.get("plateau") is True
        or summary.get("is_plateau") is True
        or _lower(summary.get("status")) == "plateau"
        or _lower(summary.get("convergence_status")) == "plateau"
        or _lower(summary.get("trend")) == "plateau"
    )


def _is_literature_missing(summary: dict[str, Any]) -> bool:
    if not summary:
        return True
    if summary.get("missing") is True or summary.get("available") is False:
        return True
    if summary.get("literature_missing") is True:
        return True
    for key in ("count", "n_papers", "papers"):
        value = summary.get(key)
        if isinstance(value, int | float) and value <= 0:
            return True
        if isinstance(value, list | tuple | set) and len(value) == 0:
            return True
    return False


def _plateau_confidence(context: CampaignRoundContext) -> float:
    for summary in (
        context.nexus_diagnostics,
        context.objective_summary,
        context.strategy_selection_result,
    ):
        value = _as_float(
            summary.get(
                "plateau_confidence",
                summary.get("convergence_confidence", summary.get("confidence")),
            )
        )
        if value is not None:
            return _clamp_confidence(value)
    return 0.7


def _has_low_failure_attribution_confidence(summary: dict[str, Any]) -> bool:
    confidence = _failure_attribution_confidence(summary)
    if confidence is None:
        return False
    return _has_failure_signal(summary) and confidence < 0.5


def _failure_attribution_confidence(summary: dict[str, Any]) -> float | None:
    for key in (
        "attribution_confidence",
        "failure_attribution_confidence",
        "confidence",
    ):
        value = _as_float(summary.get(key))
        if value is not None:
            return value
    attribution = summary.get("attribution")
    if isinstance(attribution, dict):
        return _as_float(attribution.get("confidence"))
    return None


def _has_failure_signal(summary: dict[str, Any]) -> bool:
    return (
        summary.get("has_failure") is True
        or summary.get("failure") is True
        or _has_items(summary.get("events"))
        or _as_float(summary.get("failure_count")) not in (None, 0.0)
    )


def _failure_attribution_request_confidence(summary: dict[str, Any]) -> float:
    confidence = _failure_attribution_confidence(summary)
    if confidence is None:
        return 0.65
    return _clamp_confidence(1.0 - confidence)


def _needs_backend_memory_context(context: CampaignRoundContext) -> bool:
    return _is_backend_memory_missing(
        context.backend_memory_summary
    ) and _has_repeated_failure(context.failure_summary)


def _is_backend_memory_missing(summary: dict[str, Any]) -> bool:
    if not summary:
        return True
    if summary.get("missing") is True or summary.get("available") is False:
        return True
    if summary.get("backend_memory_missing") is True:
        return True
    for key in ("record_count", "n_records", "records", "history"):
        value = summary.get(key)
        if isinstance(value, int | float) and value <= 0:
            return True
        if isinstance(value, list | tuple | set) and len(value) == 0:
            return True
    return False


def _has_repeated_failure(summary: dict[str, Any]) -> bool:
    if summary.get("repeated_failure") is True or summary.get("repeated_failures") is True:
        return True
    for key in ("repeated_failure_count", "failure_count", "n_failures"):
        value = _as_float(summary.get(key))
        if value is not None and value >= 2:
            return True
    events = summary.get("events")
    return isinstance(events, list | tuple) and len(events) >= 2


def _add_backend_memory_context_request(
    plan: CampaignDecisionPlan,
    context: CampaignRoundContext,
) -> None:
    plan.context_requests.append(
        CampaignContextRequest(
            request_type="backend_memory",
            reason=(
                "Backend memory is missing while failures repeat; backend "
                "history is needed to interpret the strategy route."
            ),
            priority="high",
            target="backend_memory_summary",
            payload={
                "backend_memory_summary": dict(context.backend_memory_summary),
                "failure_summary": dict(context.failure_summary),
            },
        )
    )
    plan.evidence.append(
        CampaignDecisionEvidence(
            source="contextual_decision_layer",
            kind="backend_memory_missing_repeated_failure",
            summary=(
                "Backend memory was missing while repeated failures were present."
            ),
            payload={
                "backend_memory_summary": dict(context.backend_memory_summary),
                "failure_summary": dict(context.failure_summary),
            },
        )
    )
    plan.rationale = (
        f"{plan.rationale} Backend memory is missing for repeated failures, "
        "so this shadow plan requests additional context."
    )
    plan.metadata = {
        **dict(plan.metadata),
        "context_enriched": True,
        "context_request_reason": "backend_memory_missing_repeated_failure",
    }


def _objective_proxy_gap_score(summary: dict[str, Any]) -> float | None:
    score = _as_float(summary.get("proxy_gap_score"))
    if score is not None:
        return score

    assessment = summary.get("proxy_gap_assessment")
    if isinstance(assessment, dict):
        nested_score = _as_float(assessment.get("score"))
        if nested_score is not None:
            return nested_score
    return None


def _nested_proxy_gap_level(summary: dict[str, Any]) -> str | None:
    assessment = summary.get("proxy_gap_assessment")
    if not isinstance(assessment, dict):
        return None
    return _lower(assessment.get("level"))


def _objective_proxy_gap_confidence(
    summary: dict[str, Any],
    proxy_gap_score: float | None,
) -> float:
    confidence = _as_float(summary.get("confidence"))
    if confidence is not None:
        return _clamp_confidence(confidence)
    if proxy_gap_score is not None:
        return _clamp_confidence(proxy_gap_score)
    return 0.65


def _first_present(payload: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return str(value)
    return None


def _strategy_trace(payload: dict[str, Any]) -> dict[str, Any]:
    trace = payload.get("strategy_trace", payload.get("trace", {}))
    return _json_safe(trace) if isinstance(trace, dict) else {"value": _json_safe(trace)}


def _confidence(payload: dict[str, Any]) -> float:
    value = payload.get("confidence", payload.get("score", payload.get("selection_confidence", 0.5)))
    try:
        return _clamp_confidence(float(value))
    except (TypeError, ValueError):
        return 0.5


def _confidence_with_default(payload: dict[str, Any], default: float) -> float:
    value = _as_float(payload.get("confidence"))
    if value is None:
        return default
    return _clamp_confidence(value)


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, value))


def _lower(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).lower()


def _has_items(value: Any) -> bool:
    return isinstance(value, list | tuple | set) and len(value) > 0


def _fallback_action(value: Any) -> CampaignDecisionAction | None:
    if value is None:
        return None
    try:
        return CampaignDecisionAction(value)
    except ValueError:
        return None


def _strategy_evidence(raw: Any) -> list[CampaignDecisionEvidence]:
    if raw is None:
        return []
    if isinstance(raw, list) and all(isinstance(item, dict) for item in raw):
        try:
            return [CampaignDecisionEvidence.model_validate(item) for item in raw]
        except ValidationError:
            pass
    return [
        CampaignDecisionEvidence(
            source="dynamic_strategy_selector",
            kind="raw_evidence",
            summary="Raw evidence from dynamic strategy selector",
            payload={"evidence": _json_safe(raw)},
        )
    ]


def _summary_evidence(source: str, kind: str, payload: dict[str, Any]) -> CampaignDecisionEvidence:
    return CampaignDecisionEvidence(
        source=source,
        kind=kind,
        summary=f"{source} indicated {kind.replace('_', ' ')}.",
        payload=dict(payload),
    )


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        return enum_value
    return value
