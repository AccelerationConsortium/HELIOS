"""Runtime adapter for closed-loop drift monitoring.

This module owns the bounded translation between orchestrator/runtime payloads
and the pure drift monitor. It may persist campaign context and emit a report,
but it never changes campaign strategy, objectives, parameter space, or routes.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)

__all__ = [
    "assess_and_persist_closed_loop_drift",
    "bounded_proxy_gap",
    "bounded_runtime_state",
    "current_proxy_gap_delta",
    "extract_closed_loop_runtime_signals",
    "human_override_from_steps",
    "record_closed_loop_observation",
    "sanitize_closed_loop_observation",
]


def human_override_from_steps(
    steps: list[dict[str, Any]],
) -> tuple[bool | None, str | None]:
    """Return an auditable operator-override signal from round step results."""
    reasons: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        status = str(step.get("status") or "")
        reason = str(step.get("human_override_reason") or step.get("reason") or step.get("rejection_reason") or "")
        explicit_override = step.get("human_override") is True
        human_reason = any(token in reason.lower() for token in ("operator", "human", "manual", "user"))
        if explicit_override or (status == "rejected" and human_reason):
            reasons.append(reason or "operator rejected candidate")
        elif status == "approval_timeout":
            reasons.append("operator approval timed out")
    if not reasons:
        return None, None
    return True, "; ".join(dict.fromkeys(reasons))


def current_proxy_gap_delta(campaign_context: dict[str, Any], prior_summary: dict[str, Any]) -> float | None:
    """Calculate the current observed proxy-gap change for outcome accounting."""
    gaps: list[float] = []
    for observation in campaign_context.get("closed_loop_observations", []) or []:
        if not isinstance(observation, dict):
            continue
        proxy = observation.get("proxy_value", observation.get("proxy_kpi"))
        scientific = observation.get(
            "scientific_value",
            observation.get(
                "scientific_objective_value",
                observation.get("functional_outcome"),
            ),
        )
        if (
            isinstance(proxy, int | float)
            and not isinstance(proxy, bool)
            and isinstance(scientific, int | float)
            and not isinstance(scientific, bool)
            and math.isfinite(float(proxy))
            and math.isfinite(float(scientific))
        ):
            scale = max(abs(float(proxy)), abs(float(scientific)), 1e-9)
            gaps.append(min(1.0, abs(float(proxy) - float(scientific)) / scale))

    current: float | None = None
    if gaps:
        current = sum(gaps[-2:]) / min(len(gaps), 2)
    else:
        assessment = campaign_context.get("proxy_gap_assessment")
        score = assessment.get("score") if isinstance(assessment, dict) else None
        if isinstance(score, int | float) and not isinstance(score, bool):
            current = max(0.0, min(1.0, float(score)))
    if current is None:
        return None

    previous: float | None = None
    for signal in prior_summary.get("signals", []) or []:
        if not isinstance(signal, dict) or signal.get("name") != "objective_proxy_gap":
            continue
        value = signal.get("current_value")
        if isinstance(value, int | float) and not isinstance(value, bool):
            previous = float(value)
            break
    if previous is None and len(gaps) > 2:
        previous = sum(gaps[:-2]) / len(gaps[:-2])
    return current - previous if previous is not None else None


def assess_and_persist_closed_loop_drift(
    *,
    campaign_id: str,
    round_index: int,
    campaign_context: dict[str, Any],
    parameters: list[dict[str, Any]],
    parameter_rounds: list[int],
    dimensions: list[dict[str, Any]],
    emit: Callable[[dict[str, Any]], None],
    force: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Persist and emit a best-effort report for the next decision context."""
    existing_report = dict(campaign_context.get("closed_loop_drift_report") or {})
    existing_memory = dict(campaign_context.get("decision_memory") or {})
    try:
        if not getattr(get_settings(), "closed_loop_drift_monitor_enabled", True):
            return {}, existing_memory
        if not force and existing_report and existing_report.get("round_index") == round_index:
            return existing_report, existing_memory

        from app.services.campaign_state import (
            load_all_candidates,
            save_campaign_context,
        )
        from app.services.closed_loop_drift import (
            assess_closed_loop_drift,
            build_next_round_decision_memory,
        )
        from app.services.decision_trajectory import load_trajectories

        trajectories = load_trajectories(campaign_id)
        decision_memory = build_next_round_decision_memory(trajectories)
        report = assess_closed_loop_drift(
            campaign_id=campaign_id,
            round_index=round_index,
            parameters=parameters,
            parameter_rounds=parameter_rounds,
            dimensions=dimensions,
            trajectories=trajectories,
            campaign_context=campaign_context,
            candidate_records=load_all_candidates(campaign_id),
            decision_memory=decision_memory,
        )
        report_payload = report.model_dump(mode="json")
        campaign_context["decision_memory"] = decision_memory
        campaign_context["closed_loop_drift_report"] = report_payload
        history = list(campaign_context.get("closed_loop_drift_history", []) or [])
        history.append(
            {
                "report_id": report.report_id,
                "round_index": report.round_index,
                "overall_status": report.overall_status.value,
                "requires_validation": report.requires_validation,
                "requires_objective_review": report.requires_objective_review,
                "requires_context_review": report.requires_context_review,
                "safe_for_memory_reuse": report.safe_for_memory_reuse,
            }
        )
        campaign_context["closed_loop_drift_history"] = history[-50:]
        save_campaign_context(campaign_id, campaign_context)
        emit(
            {
                "type": "closed_loop_drift_report",
                "round": round_index,
                "report_id": report.report_id,
                "status": report.overall_status.value,
                "requires_validation": report.requires_validation,
                "requires_objective_review": report.requires_objective_review,
                "requires_context_review": report.requires_context_review,
                "safe_for_memory_reuse": report.safe_for_memory_reuse,
                "signals": [
                    {
                        "name": signal.name,
                        "drift_type": signal.drift_type,
                        "status": signal.status.value,
                        "score": signal.score,
                    }
                    for signal in report.signals
                ],
                "message": (
                    f"Closed-loop drift: {report.overall_status.value}; "
                    f"actions={','.join(report.recommended_actions) or 'none'}"
                ),
            }
        )
        return report_payload, decision_memory
    except Exception:
        logger.warning(
            "Closed-loop drift monitor failed; preserving prior context",
            exc_info=True,
        )
        return existing_report, existing_memory


def extract_closed_loop_runtime_signals(outputs: Any) -> dict[str, Any]:
    """Extract a bounded, explicit drift contract from worker outputs."""
    if not isinstance(outputs, dict):
        return {}
    nested = outputs.get("closed_loop_signals")
    nested = dict(nested) if isinstance(nested, dict) else {}
    result: dict[str, Any] = {}
    telemetry = _bounded_numeric_runtime_mapping(nested.get("telemetry", outputs.get("telemetry")))
    calibration = bounded_runtime_state(nested.get("calibration", outputs.get("calibration")))
    instrument_state = bounded_runtime_state(nested.get("instrument_state", outputs.get("instrument_state")))
    for key in ("calibration_id", "calibration_confidence", "calibrated_at"):
        if key in calibration and key not in instrument_state:
            instrument_state[key] = calibration[key]
    if telemetry:
        result["telemetry"] = telemetry
    if calibration:
        result["calibration"] = calibration
    if instrument_state:
        result["instrument_state"] = instrument_state
    for key in (
        "predicted_value",
        "predicted_kpi",
        "proxy_value",
        "proxy_kpi",
        "scientific_value",
        "scientific_objective_value",
        "functional_outcome",
        "calibration_confidence",
    ):
        value = nested.get(key, outputs.get(key))
        if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(float(value)):
            result[key] = float(value)
    return result


_RUNTIME_STATE_KEYS = {
    "instrument_id",
    "calibration_id",
    "calibration_confidence",
    "calibrated_at",
    "firmware_version",
}


def bounded_runtime_state(value: Any) -> dict[str, Any]:
    """Keep only small scalar instrument/calibration identity fields."""
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for key in _RUNTIME_STATE_KEYS:
        item = value.get(key)
        if isinstance(item, bool) or item is None:
            continue
        if isinstance(item, int | float) and math.isfinite(float(item)):
            result[key] = float(item)
        elif isinstance(item, str):
            result[key] = item[:256]
    return result


def _bounded_numeric_runtime_mapping(value: Any) -> dict[str, float]:
    """Flatten at most 32 finite numeric telemetry fields to a depth of three."""
    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}

    def _walk(node: dict[str, Any], prefix: str, depth: int) -> None:
        if depth > 3 or len(result) >= 32:
            return
        for raw_key, item in node.items():
            if len(result) >= 32:
                break
            key = str(raw_key)[:80]
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(item, bool):
                continue
            if isinstance(item, int | float) and math.isfinite(float(item)):
                result[name] = float(item)
            elif isinstance(item, dict):
                _walk(item, name, depth + 1)

    _walk(value, "", 0)
    return result


def sanitize_closed_loop_observation(value: Any) -> dict[str, Any]:
    """Normalize user-seeded observations to the bounded runtime contract."""
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for key in (
        "round_index",
        "candidate_index",
        "outcome_value",
        "predicted_value",
        "predicted_kpi",
        "proxy_value",
        "proxy_kpi",
        "scientific_value",
        "scientific_objective_value",
        "functional_outcome",
        "calibration_confidence",
    ):
        item = value.get(key)
        if isinstance(item, int | float) and not isinstance(item, bool) and math.isfinite(float(item)):
            result[key] = float(item)
    for key in ("strategy", "backend", "failure_reason", "run_id", "recorded_at"):
        item = value.get(key)
        if isinstance(item, str):
            result[key] = item[:500]
    telemetry = _bounded_numeric_runtime_mapping(value.get("telemetry"))
    if telemetry:
        result["telemetry"] = telemetry
    return result


def bounded_proxy_gap(value: Any) -> dict[str, Any]:
    """Normalize an explicit proxy-gap assessment to a compact safe shape."""
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    score = value.get("score")
    if isinstance(score, int | float) and not isinstance(score, bool) and math.isfinite(float(score)):
        result["score"] = max(0.0, min(1.0, float(score)))
    for key in ("level", "source", "reason", "recorded_at"):
        item = value.get(key)
        if isinstance(item, str):
            result[key] = item[:500]
    return result


def record_closed_loop_observation(
    *,
    campaign_id: str,
    campaign_context: dict[str, Any],
    round_number: int,
    candidate_index: int,
    parameters: dict[str, Any],
    kpi: float | None,
    step_result: dict[str, Any],
    strategy: str,
    backend: str | None,
    failure_reason: str | None = None,
) -> None:
    """Persist one bounded, explicitly recognized closed-loop observation."""
    runtime = dict(step_result.get("closed_loop_signals") or {})
    telemetry: dict[str, Any] = {}
    for source in (runtime.get("telemetry"), runtime.get("calibration")):
        if isinstance(source, dict):
            telemetry.update(source)
    observation = {
        "round_index": round_number,
        "candidate_index": candidate_index,
        "parameters": dict(parameters),
        "outcome_value": kpi,
        "predicted_value": runtime.get("predicted_value", runtime.get("predicted_kpi")),
        "proxy_value": runtime.get("proxy_value", runtime.get("proxy_kpi")),
        "scientific_value": runtime.get(
            "scientific_value",
            runtime.get("scientific_objective_value", runtime.get("functional_outcome")),
        ),
        "calibration_confidence": runtime.get("calibration_confidence"),
        "telemetry": telemetry,
        "strategy": strategy,
        "backend": backend,
        "failure_reason": failure_reason,
        "run_id": step_result.get("run_id"),
        "recorded_at": time.time(),
    }
    observations = campaign_context.setdefault("closed_loop_observations", [])
    observations.append({key: value for key, value in observation.items() if value not in (None, {}, [])})
    if len(observations) > 200:
        del observations[:-200]
    instrument_state = runtime.get("instrument_state")
    if isinstance(instrument_state, dict):
        campaign_context["instrument_state"] = {
            **dict(campaign_context.get("instrument_state") or {}),
            **bounded_runtime_state(instrument_state),
        }
    try:
        from app.services.campaign_state import save_campaign_context

        save_campaign_context(campaign_id, campaign_context)
    except Exception:
        logger.debug(
            "Failed to checkpoint closed-loop drift observation",
            exc_info=True,
        )
