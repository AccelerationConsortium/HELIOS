"""Thin campaign decision layer above dynamic strategy selection.

Phase 1 is an envelope adapter only. It does not change campaign runtime
behavior or reimplement the existing strategy selector.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from pydantic import ValidationError

from app.services.decision_models import (
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

        return self._wrap_strategy_result(context.strategy_selection_result)

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
