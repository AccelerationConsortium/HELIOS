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
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


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
