"""Typed, replayable monitoring for long-horizon closed-loop drift.

The monitor is deliberately pure: callers provide persisted campaign state and
decision trajectories, and receive a report.  It never changes a strategy,
objective, parameter space, or hardware route.  The orchestrator may feed the
report into the next round's decision context; the existing campaign-authority
gate remains the only live promotion boundary.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import UTC, datetime
from enum import StrEnum
from statistics import fmean, median, pstdev
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

__all__ = [
    "CLOSED_LOOP_DRIFT_SCHEMA_VERSION",
    "ClosedLoopDriftMonitor",
    "ClosedLoopDriftReport",
    "DriftSignal",
    "DriftStatus",
    "assess_closed_loop_drift",
    "build_candidate_applicability_context",
    "build_next_round_decision_memory",
]


CLOSED_LOOP_DRIFT_SCHEMA_VERSION = "closed_loop_drift.v1"
_MIN_WINDOW = 2
_RECENT_ROUNDS = 2
_MAX_MEMORY_RECORDS = 8


class DriftStatus(StrEnum):
    """Evidence strength for one monitored quantity."""

    INSUFFICIENT = "insufficient"
    STABLE = "stable"
    WATCH = "watch"
    DRIFT = "drift"


class DriftSignal(BaseModel):
    """One auditable drift metric and the evidence used to calculate it."""

    name: str
    drift_type: str
    status: DriftStatus
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    sample_count: int = Field(default=0, ge=0)
    baseline_count: int = Field(default=0, ge=0)
    recent_count: int = Field(default=0, ge=0)
    threshold: float | None = None
    baseline_value: float | None = None
    current_value: float | None = None
    trend: float | None = None
    evidence: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClosedLoopDriftReport(BaseModel):
    """Round-scoped report threaded into the next decision context."""

    report_id: str = Field(default_factory=lambda: f"cld-{uuid4().hex}")
    schema_version: str = CLOSED_LOOP_DRIFT_SCHEMA_VERSION
    campaign_id: str
    round_index: int = Field(ge=0)
    overall_status: DriftStatus
    signals: list[DriftSignal]
    requires_validation: bool = False
    requires_objective_review: bool = False
    requires_context_review: bool = False
    safe_for_memory_reuse: bool = True
    recommended_actions: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClosedLoopDriftMonitor:
    """Assess the six closed-loop drift modes from four primary quantities."""

    def assess(
        self,
        *,
        campaign_id: str,
        round_index: int,
        parameters: list[dict[str, Any]] | None = None,
        parameter_rounds: list[int] | None = None,
        dimensions: list[dict[str, Any]] | None = None,
        trajectories: list[dict[str, Any]] | None = None,
        campaign_context: dict[str, Any] | None = None,
        candidate_records: list[dict[str, Any]] | None = None,
        decision_memory: dict[str, Any] | None = None,
    ) -> ClosedLoopDriftReport:
        context = dict(campaign_context or {})
        candidates = list(candidate_records or [])
        recorded_parameters = [dict(row["params"]) for row in candidates if isinstance(row.get("params"), dict)]
        recorded_rounds = [
            int(row["round_number"])
            for row in candidates
            if isinstance(row.get("params"), dict) and isinstance(row.get("round_number"), int)
        ]
        if len(recorded_parameters) == len(recorded_rounds) and recorded_parameters:
            observed_parameters = recorded_parameters
            observed_rounds = recorded_rounds
        else:
            observed_parameters = list(parameters or [])
            observed_rounds = list(parameter_rounds or [])
        rows = _campaign_trajectories(trajectories or [])
        memory = dict(decision_memory or build_next_round_decision_memory(rows))
        signals = [
            _observation_distribution_signal(observed_parameters, observed_rounds, dimensions or []),
            _prediction_residual_signal(rows, context),
            _objective_proxy_gap_signal(context, rows),
            _replay_policy_signal(rows),
            _measurement_drift_signal(context),
            _context_drift_signal(memory),
            _memory_applicability_signal(candidates, context),
        ]
        overall = _overall_status(signals)
        by_name = {signal.name: signal for signal in signals}

        direct_validation_names = {
            "prediction_outcome_residual",
            "replay_policy_performance",
            "measurement_telemetry",
        }
        requires_validation = any(
            signal.status == DriftStatus.DRIFT and signal.name in direct_validation_names for signal in signals
        )
        requires_validation = requires_validation or (
            by_name["observation_distribution"].status == DriftStatus.DRIFT
            and any(
                by_name[name].status in {DriftStatus.WATCH, DriftStatus.DRIFT}
                for name in (
                    "prediction_outcome_residual",
                    "replay_policy_performance",
                )
            )
        )
        requires_objective_review = by_name["objective_proxy_gap"].status == DriftStatus.DRIFT
        requires_context_review = by_name["decision_context_completeness"].status == DriftStatus.DRIFT
        safe_for_memory_reuse = by_name["candidate_memory_applicability"].status not in {
            DriftStatus.WATCH,
            DriftStatus.DRIFT,
        }
        recommended_actions: list[str] = []
        if by_name["measurement_telemetry"].status == DriftStatus.DRIFT:
            recommended_actions.append("validate_instrument_calibration")
        if requires_validation:
            recommended_actions.append("run_validation_before_more_candidates")
        if requires_objective_review:
            recommended_actions.append("review_proxy_against_scientific_objective")
        if requires_context_review:
            recommended_actions.append("complete_missing_decision_context")
        if not safe_for_memory_reuse:
            recommended_actions.append("block_unqualified_candidate_memory_reuse")

        return ClosedLoopDriftReport(
            campaign_id=campaign_id,
            round_index=round_index,
            overall_status=overall,
            signals=signals,
            requires_validation=requires_validation,
            requires_objective_review=requires_objective_review,
            requires_context_review=requires_context_review,
            safe_for_memory_reuse=safe_for_memory_reuse,
            recommended_actions=list(dict.fromkeys(recommended_actions)),
            metadata={
                "trajectory_count": len(rows),
                "parameter_count": len(observed_parameters),
                "decision_memory_count": int(memory.get("record_count", 0) or 0),
            },
        )


def assess_closed_loop_drift(**kwargs: Any) -> ClosedLoopDriftReport:
    """Assess drift with the default monitor."""
    return ClosedLoopDriftMonitor().assess(**kwargs)


def build_next_round_decision_memory(
    trajectories: list[dict[str, Any]], *, limit: int = _MAX_MEMORY_RECORDS
) -> dict[str, Any]:
    """Project recent outcomes into bounded, next-round decision context.

    The projection keeps the reasons that are commonly lost in a closed loop:
    strategy rationale, human-override reason, failure reasons, route changes,
    and the applicability context under which the decision was made.
    """
    rows = _campaign_trajectories(trajectories)[-max(1, limit) :]
    records: list[dict[str, Any]] = []
    omissions: list[dict[str, Any]] = []
    for row in rows:
        trajectory = _mapping(row.get("trajectory"))
        trace = _mapping(trajectory.get("trace"))
        plan = _mapping(trace.get("decision_plan"))
        outcome = _mapping(trajectory.get("outcome"))
        outcome_metadata = _mapping(outcome.get("metadata"))
        trace_context = _mapping(trace.get("context"))
        failure_count = int(outcome.get("failure_count", 0) or 0)
        failure_reasons = _bounded_text_list(
            outcome_metadata.get("failure_reasons"), limit=8, max_length=500
        )
        record = {
            "trace_id": row.get("trace_id") or trace.get("trace_id"),
            "round_index": row.get("round_index", trace.get("round_index")),
            "selected_action": _bounded_text(plan.get("action_type"), 120),
            "selected_backend": _bounded_text(plan.get("candidate_generation_backend"), 120),
            "strategy_change_reason": _bounded_text(plan.get("rationale"), 1000),
            "observed_action": _bounded_text(outcome.get("observed_action"), 120),
            "observed_backend": _bounded_text(outcome.get("observed_backend"), 120),
            "objective_delta": outcome.get("objective_delta"),
            "proxy_gap_delta": outcome.get("proxy_gap_delta"),
            "reward": row.get("reward"),
            "failure_count": failure_count,
            "failure_reasons": failure_reasons,
            "human_override": outcome.get("human_override"),
            "human_override_reason": _bounded_text(outcome_metadata.get("human_override_reason"), 500),
            "route_changed": bool(trace.get("would_change_route", False)),
            "context_requests": _bounded_context_requests(plan.get("context_requests", [])),
            "applicability_context": _applicability_from_trace_context(trace_context),
        }
        trace_id = str(record["trace_id"] or "unknown")
        if record["human_override"] is True and not record["human_override_reason"]:
            omissions.append({"trace_id": trace_id, "missing": "human_override_reason"})
        if failure_count > 0 and not failure_reasons:
            omissions.append({"trace_id": trace_id, "missing": "failure_reasons"})
        if record["route_changed"] and not record["strategy_change_reason"]:
            omissions.append({"trace_id": trace_id, "missing": "strategy_change_reason"})
        records.append(record)
    return {
        "schema_version": "decision_memory_context.v1",
        "record_count": len(records),
        "records": records,
        "omissions": omissions,
        "latest_round": records[-1]["round_index"] if records else None,
    }


def _bounded_context_requests(value: Any) -> list[dict[str, Any]]:
    """Retain request intent without recursively copying prior context payloads."""
    if not isinstance(value, list):
        return []
    requests: list[dict[str, Any]] = []
    for item in value[:8]:
        if not isinstance(item, dict):
            continue
        request = _drop_empty(
            {
                "request_type": item.get("request_type"),
                "reason": str(item.get("reason") or "")[:500],
                "priority": item.get("priority"),
                "target": item.get("target"),
            }
        )
        requests.append(request)
    return requests


def build_candidate_applicability_context(
    *,
    objective_kpi: str,
    direction: str,
    campaign_context: dict[str, Any] | None = None,
    protocol_pattern_id: str | None = None,
    strategy: str | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    """Build a compact context fingerprint stored beside a candidate outcome."""
    context = dict(campaign_context or {})
    instrument_state = _mapping(context.get("instrument_state"))
    drift = _mapping(context.get("closed_loop_drift_report"))
    return _drop_empty(
        {
            "schema_version": "candidate_applicability.v1",
            "objective_kpi": objective_kpi,
            "direction": direction,
            "current_objective_level": context.get("current_objective_level"),
            "material_family": context.get("material_family"),
            "active_experimental_node_id": context.get("active_experimental_node_id"),
            "protocol_pattern_id": protocol_pattern_id,
            "strategy": strategy,
            "backend": backend,
            "instrument_id": instrument_state.get("instrument_id"),
            "calibration_id": instrument_state.get("calibration_id"),
            "drift_status": drift.get("overall_status"),
            "drift_report_id": drift.get("report_id"),
        }
    )


def _observation_distribution_signal(
    parameters: list[dict[str, Any]],
    rounds: list[int],
    dimensions: list[dict[str, Any]],
) -> DriftSignal:
    if len(parameters) != len(rounds) or not rounds:
        return _insufficient(
            "observation_distribution",
            "strategy_drift",
            len(parameters),
            "Aligned parameter and round histories are required.",
        )
    latest_round = max(rounds)
    cutoff = latest_round - _RECENT_ROUNDS + 1
    baseline = [params for params, round_no in zip(parameters, rounds, strict=True) if round_no < cutoff]
    recent = [params for params, round_no in zip(parameters, rounds, strict=True) if round_no >= cutoff]
    if len(baseline) < _MIN_WINDOW or len(recent) < _MIN_WINDOW:
        return _insufficient_windows(
            "observation_distribution",
            "strategy_drift",
            baseline,
            recent,
            "Need at least two baseline and two recent candidates.",
        )

    scores: list[float] = []
    evidence: list[str] = []
    for dim in dimensions:
        name = str(dim.get("param_name") or dim.get("name") or "")
        if not name:
            continue
        base_values = [item[name] for item in baseline if name in item]
        recent_values = [item[name] for item in recent if name in item]
        if not base_values or not recent_values:
            continue
        if _all_numeric(base_values + recent_values):
            low = _as_float(dim.get("min_value", dim.get("min")))
            high = _as_float(dim.get("max_value", dim.get("max")))
            span = abs(high - low) if low is not None and high is not None else None
            if not span:
                combined = [float(value) for value in base_values + recent_values]
                span = max(combined) - min(combined)
            if not span:
                score = 0.0
            else:
                mean_shift = (
                    abs(fmean(float(value) for value in recent_values) - fmean(float(value) for value in base_values))
                    / span
                )
                spread_shift = (
                    abs(pstdev(float(value) for value in recent_values) - pstdev(float(value) for value in base_values))
                    / span
                )
                score = _clamp(mean_shift + 0.5 * spread_shift)
        else:
            score = _categorical_total_variation(base_values, recent_values)
        scores.append(score)
        if len(evidence) < 16:
            evidence.append(f"{name} distribution shift={score:.3g}")
    if not scores:
        return _insufficient_windows(
            "observation_distribution",
            "strategy_drift",
            baseline,
            recent,
            "No comparable parameter dimensions were available.",
        )
    score = round(fmean(scores), 10)
    return _scored_signal(
        name="observation_distribution",
        drift_type="strategy_drift",
        score=score,
        sample_count=len(parameters),
        baseline_count=len(baseline),
        recent_count=len(recent),
        watch_threshold=0.2,
        drift_threshold=0.4,
        evidence=evidence,
        metadata={"latest_round": latest_round, "recent_round_cutoff": cutoff},
    )


def _prediction_residual_signal(rows: list[dict[str, Any]], context: dict[str, Any]) -> DriftSignal:
    runtime_pairs = [
        (predicted, outcome)
        for observation in context.get("closed_loop_observations", []) or []
        if isinstance(observation, dict)
        for predicted, outcome in [
            (
                _as_float(observation.get("predicted_value", observation.get("predicted_kpi"))),
                _as_float(observation.get("outcome_value")),
            )
        ]
        if predicted is not None and outcome is not None
    ]
    if len(runtime_pairs) >= _MIN_WINDOW * 2:
        nonzero_outcomes = [abs(outcome) for _, outcome in runtime_pairs if outcome]
        outcome_scale = median(nonzero_outcomes) if nonzero_outcomes else 1.0
        values = [_clamp(abs(predicted - outcome) / outcome_scale) for predicted, outcome in runtime_pairs]
        return _residual_shift_signal(
            values,
            scale=outcome_scale,
            source="runtime_prediction",
        )

    residuals: list[tuple[float, float]] = []
    for row in rows:
        trajectory = _mapping(row.get("trajectory"))
        trace = _mapping(trajectory.get("trace"))
        plan = _mapping(trace.get("decision_plan"))
        strategy_trace = _mapping(plan.get("strategy_trace"))
        outcome = _mapping(trajectory.get("outcome"))
        expected = _selected_expected_improvement(
            strategy_trace,
            selected_backend=plan.get("candidate_generation_backend"),
        )
        delta = _as_float(outcome.get("objective_delta"))
        if expected is None or delta is None:
            continue
        residuals.append((expected, delta))
    if len(residuals) < _MIN_WINDOW * 2:
        return _insufficient(
            "prediction_outcome_residual",
            "model_drift",
            len(residuals),
            "Need four decisions with expected improvement and final objective delta.",
        )
    nonzero_deltas = [abs(delta) for _expected, delta in residuals if delta != 0]
    delta_scale = median(nonzero_deltas) if nonzero_deltas else 1.0
    values = [abs(expected - _clamp(max(delta, 0.0) / delta_scale)) for expected, delta in residuals]
    return _residual_shift_signal(
        values,
        scale=delta_scale,
        source="strategy_expected_improvement",
    )


def _residual_shift_signal(values: list[float], *, scale: float, source: str) -> DriftSignal:
    baseline, recent = _split_series(values)
    baseline_mean = fmean(baseline)
    recent_mean = fmean(recent)
    degradation = max(0.0, recent_mean - baseline_mean)
    score = _clamp(max(recent_mean, degradation * 2.0))
    return _scored_signal(
        name="prediction_outcome_residual",
        drift_type="model_drift",
        score=score,
        sample_count=len(values),
        baseline_count=len(baseline),
        recent_count=len(recent),
        watch_threshold=0.3,
        drift_threshold=0.5,
        baseline_value=_round(baseline_mean),
        current_value=_round(recent_mean),
        trend=_round(recent_mean - baseline_mean),
        evidence=[
            f"recent mean normalized residual={recent_mean:.3g}",
            f"historical mean normalized residual={baseline_mean:.3g}",
        ],
        metadata={"residual_scale": _round(scale), "source": source},
    )


def _objective_proxy_gap_signal(context: dict[str, Any], rows: list[dict[str, Any]]) -> DriftSignal:
    current = _proxy_gap_score(context)
    history = [
        value
        for value in (
            _as_float(item.get("score")) for item in context.get("proxy_gap_history", []) if isinstance(item, dict)
        )
        if value is not None
    ]
    divergences = _proxy_scientific_divergences(context)
    if divergences:
        current = fmean(divergences[-_MIN_WINDOW:])
        history = divergences[:-_MIN_WINDOW]
    if current is None:
        proxy_deltas = [
            _as_float(_mapping(_mapping(row.get("trajectory")).get("outcome")).get("proxy_gap_delta")) for row in rows
        ]
        observed = [value for value in proxy_deltas if value is not None]
        if observed:
            current = _clamp(0.5 + fmean(observed[-_MIN_WINDOW:]))
            history = [_clamp(0.5 + value) for value in observed[:-_MIN_WINDOW]]
    if current is None:
        return _insufficient(
            "objective_proxy_gap",
            "target_drift",
            0,
            "No proxy-gap assessment or paired proxy/scientific outcomes were recorded.",
        )
    baseline = fmean(history) if history else None
    trend = current - baseline if baseline is not None else None
    score = _clamp(max(current, (trend or 0.0) * 2.0))
    return _scored_signal(
        name="objective_proxy_gap",
        drift_type="target_drift",
        score=score,
        sample_count=len(history) + 1,
        baseline_count=len(history),
        recent_count=1,
        watch_threshold=0.35,
        drift_threshold=0.6,
        baseline_value=_round(baseline) if baseline is not None else None,
        current_value=_round(current),
        trend=_round(trend) if trend is not None else None,
        evidence=[
            f"current proxy-to-scientific gap={current:.3g}",
            "high gap or an expanding gap indicates target drift",
        ],
    )


def _replay_policy_signal(rows: list[dict[str, Any]]) -> DriftSignal:
    scored = [row for row in rows if _as_float(row.get("reward")) is not None]
    if len(scored) < 6:
        return _insufficient(
            "replay_policy_performance",
            "strategy_drift",
            len(scored),
            "Need at least six scored trajectories for recent-versus-history replay.",
        )
    split = max(3, len(scored) - 3)
    historical = scored[:split]
    recent = scored[split:]
    current_policy = _policy_key(recent[-1])
    current_rows = [row for row in recent if _policy_key(row) == current_policy]
    if len(current_rows) < _MIN_WINDOW:
        return _insufficient_windows(
            "replay_policy_performance",
            "strategy_drift",
            historical,
            current_rows,
            "Need two recent scored outcomes from the same current policy.",
        )
    historical_by_policy: dict[str, list[float]] = defaultdict(list)
    for row in historical:
        reward = _as_float(row.get("reward"))
        if reward is not None:
            historical_by_policy[_policy_key(row)].append(reward)
    eligible = {policy: rewards for policy, rewards in historical_by_policy.items() if len(rewards) >= _MIN_WINDOW}
    if not eligible:
        return _insufficient_windows(
            "replay_policy_performance",
            "strategy_drift",
            historical,
            current_rows,
            "Historical policies lack two comparable outcomes.",
        )
    best_policy, best_rewards = max(eligible.items(), key=lambda item: fmean(item[1]))
    current_mean = fmean(float(row["reward"]) for row in current_rows)
    historical_mean = fmean(best_rewards)
    underperformance = max(0.0, historical_mean - current_mean)
    score = _clamp(underperformance / 2.0)
    return _scored_signal(
        name="replay_policy_performance",
        drift_type="strategy_drift",
        score=score,
        sample_count=len(scored),
        baseline_count=len(best_rewards),
        recent_count=len(current_rows),
        watch_threshold=0.15,
        drift_threshold=0.3,
        baseline_value=_round(historical_mean),
        current_value=_round(current_mean),
        trend=_round(current_mean - historical_mean),
        evidence=[
            f"recent policy={current_policy} mean reward={current_mean:.3g}",
            f"historical policy={best_policy} mean reward={historical_mean:.3g}",
            "This is replay evidence, not a causal counterfactual.",
        ],
        metadata={"current_policy": current_policy, "historical_policy": best_policy},
    )


def _measurement_drift_signal(context: dict[str, Any]) -> DriftSignal:
    observations = [item for item in context.get("closed_loop_observations", []) if isinstance(item, dict)]
    numeric: dict[str, list[float]] = defaultdict(list)
    for observation in observations:
        telemetry = _mapping(observation.get("telemetry"))
        for key, value in _flatten_numeric(telemetry).items():
            numeric[key].append(value)
    confidence = _calibration_confidence(context, observations)
    field_scores: list[tuple[str, float, float, float]] = []
    for key, values in numeric.items():
        if len(values) < _MIN_WINDOW * 2:
            continue
        baseline, recent = _split_series(values)
        base_mean = fmean(baseline)
        recent_mean = fmean(recent)
        scale = max(pstdev(baseline), abs(base_mean) * 0.05, 1e-9)
        score = _clamp(abs(recent_mean - base_mean) / (4.0 * scale))
        field_scores.append((key, score, base_mean, recent_mean))
    if confidence is None and not field_scores:
        return _insufficient(
            "measurement_telemetry",
            "measurement_drift",
            len(observations),
            "No calibration confidence or repeated numeric telemetry was recorded.",
        )
    score = max((item[1] for item in field_scores), default=0.0)
    if confidence is not None:
        score = max(score, _clamp(1.0 - confidence))
    evidence = [
        f"{key}: baseline={baseline:.3g}, recent={recent:.3g}, shift={field_score:.3g}"
        for key, field_score, baseline, recent in sorted(field_scores, key=lambda item: item[1], reverse=True)[:5]
    ]
    if confidence is not None:
        evidence.append(f"current calibration confidence={confidence:.3g}")
    return _scored_signal(
        name="measurement_telemetry",
        drift_type="measurement_drift",
        score=score,
        sample_count=len(observations),
        baseline_count=max(0, len(observations) - _MIN_WINDOW),
        recent_count=min(_MIN_WINDOW, len(observations)),
        watch_threshold=0.3,
        drift_threshold=0.5,
        current_value=_round(confidence) if confidence is not None else None,
        evidence=evidence,
    )


def _context_drift_signal(memory: dict[str, Any]) -> DriftSignal:
    omissions = [item for item in memory.get("omissions", []) if isinstance(item, dict)]
    record_count = int(memory.get("record_count", 0) or 0)
    if record_count == 0:
        return _insufficient(
            "decision_context_completeness",
            "context_drift",
            0,
            "No prior decisions exist yet.",
        )
    ratio = len(omissions) / max(record_count, 1)
    score = _clamp(ratio)
    return _scored_signal(
        name="decision_context_completeness",
        drift_type="context_drift",
        score=score,
        sample_count=record_count,
        baseline_count=record_count,
        recent_count=1,
        watch_threshold=0.1,
        drift_threshold=0.3,
        current_value=_round(ratio),
        evidence=[f"{item.get('trace_id')}: missing {item.get('missing')}" for item in omissions[:8]]
        or ["Recent decision records retain required reasons and failure context."],
        metadata={"omission_count": len(omissions)},
    )


def _memory_applicability_signal(records: list[dict[str, Any]], context: dict[str, Any]) -> DriftSignal:
    successful = [row for row in records if row.get("status") == "completed" and row.get("kpi_value") is not None]
    if not successful:
        return _insufficient(
            "candidate_memory_applicability",
            "memory_eviction_error",
            0,
            "No successful candidate-memory records exist yet.",
        )
    current = _current_applicability_context(context)
    missing: list[dict[str, Any]] = []
    mismatched: list[dict[str, Any]] = []
    for row in successful:
        stored = _mapping(row.get("applicability_context"))
        if not stored:
            missing.append(row)
            continue
        if any(
            key in current and key in stored and current[key] != stored[key] for key in _APPLICABILITY_COMPARISON_KEYS
        ):
            mismatched.append(row)
    unsafe_count = len(missing) + len(mismatched)
    ratio = unsafe_count / len(successful)
    return _scored_signal(
        name="candidate_memory_applicability",
        drift_type="memory_eviction_error",
        score=ratio,
        sample_count=len(successful),
        baseline_count=len(successful),
        recent_count=len(successful),
        watch_threshold=0.01,
        drift_threshold=0.5,
        current_value=_round(ratio),
        evidence=[
            f"{len(missing)} of {len(successful)} successful candidates lack applicability context.",
            f"{len(mismatched)} of {len(successful)} successful candidates mismatch the current context.",
        ],
        metadata={
            "unqualified_candidate_count": unsafe_count,
            "missing_context_count": len(missing),
            "mismatched_context_count": len(mismatched),
        },
    )


_APPLICABILITY_COMPARISON_KEYS = (
    "objective_kpi",
    "direction",
    "current_objective_level",
    "material_family",
    "active_experimental_node_id",
    "instrument_id",
    "calibration_id",
)


def _current_applicability_context(context: dict[str, Any]) -> dict[str, Any]:
    hierarchy = [item for item in context.get("objective_hierarchy", []) if isinstance(item, dict)]
    objective = hierarchy[0] if hierarchy else {}
    instrument = _mapping(context.get("instrument_state"))
    return _drop_empty(
        {
            "objective_kpi": objective.get("metric") or context.get("scientific_goal"),
            "direction": objective.get("direction"),
            "current_objective_level": context.get("current_objective_level"),
            "material_family": context.get("material_family"),
            "active_experimental_node_id": context.get("active_experimental_node_id"),
            "instrument_id": instrument.get("instrument_id"),
            "calibration_id": instrument.get("calibration_id"),
        }
    )


def _selected_expected_improvement(strategy_trace: dict[str, Any], *, selected_backend: Any = None) -> float | None:
    selected = str(strategy_trace.get("selected_mode") or strategy_trace.get("selected_intent") or "")
    actions = [action for action in strategy_trace.get("available_actions", []) or [] if isinstance(action, dict)]
    for action in actions:
        if selected and str(action.get("name")) != selected:
            continue
        value = _as_float(action.get("expected_improvement"))
        if value is not None:
            return _clamp(value)
    for action in actions:
        if selected_backend and str(action.get("backend_name")) != str(selected_backend):
            continue
        value = _as_float(action.get("expected_improvement"))
        if value is not None:
            return _clamp(value)
    return None


def _proxy_gap_score(context: dict[str, Any]) -> float | None:
    candidates = [
        context.get("objective_proxy_gap"),
        context.get("proxy_gap_score"),
        _mapping(context.get("proxy_gap_assessment")).get("score"),
        _mapping(context.get("objective_summary")).get("proxy_gap_score"),
        _mapping(_mapping(context.get("objective_summary")).get("proxy_gap_assessment")).get("score"),
    ]
    for value in candidates:
        parsed = _as_float(value)
        if parsed is not None:
            return _clamp(parsed)
    return None


def _proxy_scientific_divergences(context: dict[str, Any]) -> list[float]:
    divergences: list[float] = []
    for item in context.get("closed_loop_observations", []) or []:
        if not isinstance(item, dict):
            continue
        proxy = _as_float(item.get("proxy_value"))
        scientific = _as_float(item.get("scientific_value"))
        if proxy is None or scientific is None:
            continue
        scale = max(abs(proxy), abs(scientific), 1e-9)
        divergences.append(_clamp(abs(proxy - scientific) / scale))
    return divergences


def _calibration_confidence(context: dict[str, Any], observations: list[dict[str, Any]]) -> float | None:
    instrument_state = _mapping(context.get("instrument_state"))
    candidates: list[Any] = []
    for observation in reversed(observations):
        candidates.extend(
            [
                observation.get("calibration_confidence"),
                _mapping(observation.get("telemetry")).get("calibration_confidence"),
            ]
        )
    candidates.extend(
        [
            instrument_state.get("calibration_confidence"),
            context.get("calibration_confidence"),
        ]
    )
    for value in candidates:
        parsed = _as_float(value)
        if parsed is not None:
            return _clamp(parsed)
    return None


def _policy_key(row: dict[str, Any]) -> str:
    trajectory = _mapping(row.get("trajectory"))
    outcome = _mapping(trajectory.get("outcome"))
    trace = _mapping(trajectory.get("trace"))
    plan = _mapping(trace.get("decision_plan"))
    return str(
        outcome.get("observed_backend")
        or plan.get("candidate_generation_backend")
        or plan.get("action_type")
        or "unknown"
    )


def _campaign_trajectories(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("layer", "campaign") == "campaign"]


def _applicability_from_trace_context(context: dict[str, Any]) -> dict[str, Any]:
    objective = _mapping(context.get("objective_summary"))
    metadata = _mapping(context.get("metadata"))
    return _drop_empty(
        {
            "objective_kpi": objective.get("objective_kpi"),
            "direction": objective.get("direction"),
            "target_value": objective.get("target_value"),
            "round_strategy": metadata.get("round_strategy"),
            "planned_strategy": metadata.get("planned_strategy"),
            "drift_status": _mapping(context.get("drift_summary")).get("overall_status"),
        }
    )


def _flatten_numeric(value: dict[str, Any], prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, bool):
            continue
        if isinstance(item, int | float) and math.isfinite(float(item)):
            result[name] = float(item)
        elif isinstance(item, dict):
            result.update(_flatten_numeric(item, name))
    return result


def _categorical_total_variation(baseline: list[Any], recent: list[Any]) -> float:
    values = {str(value) for value in baseline + recent}
    return 0.5 * sum(
        abs(
            sum(1 for item in baseline if str(item) == value) / len(baseline)
            - sum(1 for item in recent if str(item) == value) / len(recent)
        )
        for value in values
    )


def _split_series(values: list[float]) -> tuple[list[float], list[float]]:
    recent_count = max(_MIN_WINDOW, min(3, len(values) // 2))
    return values[:-recent_count], values[-recent_count:]


def _scored_signal(
    *,
    name: str,
    drift_type: str,
    score: float,
    sample_count: int,
    baseline_count: int,
    recent_count: int,
    watch_threshold: float,
    drift_threshold: float,
    baseline_value: float | None = None,
    current_value: float | None = None,
    trend: float | None = None,
    evidence: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> DriftSignal:
    score = _clamp(score)
    if score >= drift_threshold:
        status = DriftStatus.DRIFT
    elif score >= watch_threshold:
        status = DriftStatus.WATCH
    else:
        status = DriftStatus.STABLE
    return DriftSignal(
        name=name,
        drift_type=drift_type,
        status=status,
        score=_round(score),
        sample_count=sample_count,
        baseline_count=baseline_count,
        recent_count=recent_count,
        threshold=drift_threshold,
        baseline_value=baseline_value,
        current_value=current_value,
        trend=trend,
        evidence=list(evidence or []),
        metadata=dict(metadata or {}),
    )


def _insufficient(name: str, drift_type: str, sample_count: int, reason: str) -> DriftSignal:
    return DriftSignal(
        name=name,
        drift_type=drift_type,
        status=DriftStatus.INSUFFICIENT,
        sample_count=sample_count,
        evidence=[reason],
    )


def _insufficient_windows(
    name: str,
    drift_type: str,
    baseline: list[Any],
    recent: list[Any],
    reason: str,
) -> DriftSignal:
    signal = _insufficient(name, drift_type, len(baseline) + len(recent), reason)
    signal.baseline_count = len(baseline)
    signal.recent_count = len(recent)
    return signal


def _overall_status(signals: list[DriftSignal]) -> DriftStatus:
    if any(signal.status == DriftStatus.DRIFT for signal in signals):
        return DriftStatus.DRIFT
    if any(signal.status == DriftStatus.WATCH for signal in signals):
        return DriftStatus.WATCH
    if any(signal.status == DriftStatus.STABLE for signal in signals):
        return DriftStatus.STABLE
    return DriftStatus.INSUFFICIENT


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _drop_empty(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item not in (None, "", [], {})}


def _bounded_text(value: Any, max_length: int) -> str | None:
    if value is None:
        return None
    return str(value)[:max_length]


def _bounded_text_list(value: Any, *, limit: int, max_length: int) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [str(item)[:max_length] for item in value[:limit]]


def _all_numeric(values: list[Any]) -> bool:
    return all(
        not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(float(value))
        for value in values
    )


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 10)
