from __future__ import annotations

from typing import Any

import pytest

from app.services.closed_loop_drift import (
    DriftStatus,
    assess_closed_loop_drift,
    build_next_round_decision_memory,
)


def _trajectory(
    index: int,
    *,
    backend: str,
    reward: float,
    expected_improvement: float = 0.8,
    objective_delta: float = 0.8,
    proxy_gap_delta: float | None = None,
    human_override: bool | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": f"row-{index}",
        "layer": "campaign",
        "trace_id": f"trace-{index}",
        "round_index": index,
        "reward": reward,
        "trajectory": {
            "trace": {
                "trace_id": f"trace-{index}",
                "round_index": index,
                "would_change_route": False,
                "context": {
                    "objective_summary": {
                        "objective_kpi": "yield",
                        "direction": "maximize",
                    },
                    "metadata": {"round_strategy": "adaptive"},
                },
                "decision_plan": {
                    "action_type": "propose_candidates",
                    "candidate_generation_backend": backend,
                    "rationale": f"selected {backend} because replay evidence supported it",
                    "context_requests": [],
                    "strategy_trace": {
                        "selected_mode": "exploit",
                        "available_actions": [
                            {
                                "name": "exploit",
                                "expected_improvement": expected_improvement,
                            }
                        ],
                    },
                },
            },
            "outcome": {
                "observed_action": "propose_candidates",
                "observed_backend": backend,
                "objective_delta": objective_delta,
                "proxy_gap_delta": proxy_gap_delta,
                "failure_count": 0,
                "human_override": human_override,
                "metadata": dict(metadata or {}),
            },
            "reward": {"reward": reward},
        },
    }


def _signals(report):
    return {signal.name: signal for signal in report.signals}


def test_empty_history_is_insufficient_and_never_requests_live_action():
    report = assess_closed_loop_drift(
        campaign_id="empty",
        round_index=1,
    )

    assert report.overall_status == DriftStatus.INSUFFICIENT
    assert report.requires_validation is False
    assert report.requires_objective_review is False
    assert report.requires_context_review is False
    assert report.recommended_actions == []


def test_monitor_detects_four_primary_quantities_and_six_drift_modes():
    trajectories = [
        *[_trajectory(i, backend="historical_good", reward=0.8, objective_delta=0.8) for i in range(1, 4)],
        *[_trajectory(i, backend="current_bad", reward=-0.8, objective_delta=0.0) for i in range(4, 7)],
    ]
    parameters = [
        {"x": 0.2},
        {"x": 0.4},
        {"x": 0.5},
        {"x": 0.6},
        {"x": 9.2},
        {"x": 9.4},
        {"x": 9.5},
        {"x": 9.8},
    ]
    context = {
        "proxy_gap_assessment": {"score": 0.8, "level": "high"},
        "instrument_state": {"calibration_confidence": 0.2},
        "closed_loop_observations": [
            *[{"telemetry": {"sensor_zero": value}} for value in (1.0, 1.1, 1.0, 1.1)],
            *[{"telemetry": {"sensor_zero": value}} for value in (4.0, 4.1)],
        ],
    }
    decision_memory = {
        "record_count": 3,
        "records": [{}, {}, {}],
        "omissions": [{"trace_id": "trace-4", "missing": "human_override_reason"}],
    }
    candidates = [
        {
            "status": "completed",
            "kpi_value": 1.0,
            "applicability_context": {},
        }
    ]

    report = assess_closed_loop_drift(
        campaign_id="drifting",
        round_index=7,
        parameters=parameters,
        parameter_rounds=[1, 1, 2, 2, 3, 3, 4, 4],
        dimensions=[
            {
                "param_name": "x",
                "param_type": "number",
                "min_value": 0.0,
                "max_value": 10.0,
            }
        ],
        trajectories=trajectories,
        campaign_context=context,
        candidate_records=candidates,
        decision_memory=decision_memory,
    )
    signals = _signals(report)

    assert signals["observation_distribution"].status == DriftStatus.DRIFT
    assert signals["prediction_outcome_residual"].status == DriftStatus.DRIFT
    assert signals["objective_proxy_gap"].status == DriftStatus.DRIFT
    assert signals["replay_policy_performance"].status == DriftStatus.DRIFT
    assert signals["measurement_telemetry"].status == DriftStatus.DRIFT
    assert signals["decision_context_completeness"].status == DriftStatus.DRIFT
    assert signals["candidate_memory_applicability"].status == DriftStatus.DRIFT
    assert report.requires_validation is True
    assert report.requires_objective_review is True
    assert report.requires_context_review is True
    assert report.safe_for_memory_reuse is False
    assert "run_validation_before_more_candidates" in report.recommended_actions
    assert "block_unqualified_candidate_memory_reuse" in report.recommended_actions
    assert report.model_dump(mode="json")["schema_version"] == "closed_loop_drift.v1"


def test_replay_signal_is_explicitly_non_causal():
    trajectories = [
        *[_trajectory(i, backend="old", reward=0.9) for i in range(1, 4)],
        *[_trajectory(i, backend="new", reward=-0.5) for i in range(4, 7)],
    ]

    report = assess_closed_loop_drift(
        campaign_id="replay",
        round_index=7,
        trajectories=trajectories,
    )
    replay = _signals(report)["replay_policy_performance"]

    assert replay.status == DriftStatus.DRIFT
    assert replay.metadata == {
        "current_policy": "new",
        "historical_policy": "old",
    }
    assert any("not a causal counterfactual" in line for line in replay.evidence)


def test_replay_does_not_compare_a_single_current_policy_outcome():
    trajectories = [
        *[_trajectory(i, backend="old", reward=0.9) for i in range(1, 5)],
        _trajectory(5, backend="intermediate", reward=0.1),
        _trajectory(6, backend="new", reward=-0.9),
    ]

    replay = _signals(
        assess_closed_loop_drift(
            campaign_id="replay-small-current",
            round_index=7,
            trajectories=trajectories,
        )
    )["replay_policy_performance"]

    assert replay.status == DriftStatus.INSUFFICIENT
    assert replay.recent_count == 1


def test_runtime_prediction_residual_shift_takes_precedence_over_strategy_proxy():
    report = assess_closed_loop_drift(
        campaign_id="runtime-residual",
        round_index=5,
        campaign_context={
            "closed_loop_observations": [
                {"predicted_value": 10.0, "outcome_value": 10.0},
                {"predicted_value": 10.2, "outcome_value": 10.0},
                {"predicted_value": 18.0, "outcome_value": 10.0},
                {"predicted_value": 19.0, "outcome_value": 10.0},
            ]
        },
    )

    residual = _signals(report)["prediction_outcome_residual"]
    assert residual.status == DriftStatus.DRIFT
    assert residual.trend is not None and residual.trend > 0.7
    assert residual.metadata["source"] == "runtime_prediction"
    assert report.requires_validation is True


def test_sampling_region_shift_alone_is_observed_without_forcing_validation():
    report = assess_closed_loop_drift(
        campaign_id="intentional-exploitation",
        round_index=4,
        parameters=[{"x": 0.0}, {"x": 0.1}, {"x": 9.8}, {"x": 10.0}],
        parameter_rounds=[1, 1, 3, 3],
        dimensions=[
            {
                "param_name": "x",
                "param_type": "number",
                "min_value": 0.0,
                "max_value": 10.0,
            }
        ],
    )

    assert _signals(report)["observation_distribution"].status == DriftStatus.DRIFT
    assert report.requires_validation is False


def test_latest_runtime_calibration_overrides_stale_initial_belief():
    report = assess_closed_loop_drift(
        campaign_id="measurement",
        round_index=3,
        campaign_context={
            "instrument_state": {"calibration_confidence": 0.95},
            "closed_loop_observations": [
                {"calibration_confidence": 0.8},
                {"calibration_confidence": 0.2},
            ],
        },
    )

    measurement = _signals(report)["measurement_telemetry"]
    assert measurement.status == DriftStatus.DRIFT
    assert measurement.current_value == 0.2


def test_runtime_signal_contract_is_allowlisted_and_bounded():
    from app.services.closed_loop_runtime import (
        extract_closed_loop_runtime_signals as _extract_closed_loop_runtime_signals,
    )

    payload = _extract_closed_loop_runtime_signals(
        {
            "closed_loop_signals": {
                "telemetry": {
                    **{f"sensor_{index}": index for index in range(50)},
                    "raw_blob": "not persisted",
                },
                "calibration": {
                    "calibration_id": "cal-2",
                    "calibration_confidence": 0.4,
                    "untrusted_blob": "not persisted",
                },
            }
        }
    )

    assert len(payload["telemetry"]) == 32
    assert "raw_blob" not in payload["telemetry"]
    assert payload["instrument_state"]["calibration_id"] == "cal-2"
    assert "untrusted_blob" not in payload["calibration"]


def test_only_explicit_human_rejections_are_recorded_as_overrides():
    from app.services.closed_loop_runtime import (
        human_override_from_steps as _human_override_from_steps,
    )

    assert _human_override_from_steps([{"status": "rejected", "reason": "safety envelope violation"}]) == (None, None)
    assert _human_override_from_steps(
        [
            {
                "status": "rejected",
                "human_override": True,
                "human_override_reason": "operator saw precipitation",
            }
        ]
    ) == (True, "operator saw precipitation")


def test_current_proxy_gap_delta_uses_new_runtime_observations():
    from app.services.closed_loop_runtime import (
        current_proxy_gap_delta as _current_proxy_gap_delta,
    )

    delta = _current_proxy_gap_delta(
        {
            "closed_loop_observations": [
                {"proxy_value": 10.0, "scientific_value": 9.0},
                {"proxy_value": 10.0, "scientific_value": 2.0},
            ]
        },
        {
            "signals": [
                {
                    "name": "objective_proxy_gap",
                    "current_value": 0.1,
                }
            ]
        },
    )

    assert delta == pytest.approx(0.35)


def test_memory_report_rejects_success_from_a_different_calibration_context():
    report = assess_closed_loop_drift(
        campaign_id="memory-context",
        round_index=3,
        campaign_context={
            "scientific_goal": "yield",
            "objective_hierarchy": [{"metric": "yield", "direction": "maximize"}],
            "instrument_state": {"calibration_id": "cal-new"},
        },
        candidate_records=[
            {
                "status": "completed",
                "kpi_value": 0.9,
                "applicability_context": {
                    "objective_kpi": "yield",
                    "direction": "maximize",
                    "calibration_id": "cal-old",
                },
            }
        ],
    )

    memory = _signals(report)["candidate_memory_applicability"]
    assert memory.status == DriftStatus.DRIFT
    assert memory.metadata["mismatched_context_count"] == 1
    assert report.safe_for_memory_reuse is False


def test_decision_memory_carries_reasons_and_flags_missing_override_reason():
    complete = _trajectory(
        1,
        backend="bo",
        reward=0.5,
        human_override=True,
        metadata={
            "human_override_reason": "operator saw bubbles",
            "failure_reasons": ["sensor timeout"],
        },
    )
    missing = _trajectory(
        2,
        backend="lhs",
        reward=-0.2,
        human_override=True,
    )
    missing["trajectory"]["outcome"]["failure_count"] = 1

    memory = build_next_round_decision_memory([complete, missing])

    assert memory["record_count"] == 2
    assert memory["records"][0]["human_override_reason"] == "operator saw bubbles"
    assert memory["records"][0]["failure_reasons"] == ["sensor timeout"]
    assert memory["records"][0]["strategy_change_reason"]
    assert memory["records"][0]["applicability_context"]["objective_kpi"] == "yield"
    assert {item["missing"] for item in memory["omissions"]} == {
        "human_override_reason",
        "failure_reasons",
    }


def test_decision_memory_does_not_recursively_copy_context_request_payloads():
    row = _trajectory(1, backend="bo", reward=0.2)
    row["trajectory"]["trace"]["decision_plan"]["context_requests"] = [
        {
            "request_type": "decision_context_completion",
            "reason": "missing operator rationale",
            "priority": "high",
            "target": "decision_memory",
            "payload": {
                "decision_memory": {"records": [{"nested": "old memory"}]},
                "drift_summary": {"signals": ["large prior report"]},
            },
        }
    ]

    memory = build_next_round_decision_memory([row])
    request = memory["records"][0]["context_requests"][0]

    assert request == {
        "request_type": "decision_context_completion",
        "reason": "missing operator rationale",
        "priority": "high",
        "target": "decision_memory",
    }
    assert "payload" not in request


def test_decision_memory_bounds_free_text_and_failure_lists():
    row = _trajectory(
        1,
        backend="bo",
        reward=-0.2,
        human_override=True,
        metadata={
            "human_override_reason": "h" * 900,
            "failure_reasons": [f"failure-{index}-" + "x" * 700 for index in range(20)],
        },
    )
    row["trajectory"]["outcome"]["failure_count"] = 20
    row["trajectory"]["trace"]["decision_plan"]["rationale"] = "r" * 2000

    record = build_next_round_decision_memory([row])["records"][0]

    assert len(record["strategy_change_reason"]) == 1000
    assert len(record["human_override_reason"]) == 500
    assert len(record["failure_reasons"]) == 8
    assert all(len(reason) == 500 for reason in record["failure_reasons"])


@pytest.mark.parametrize(
    ("summary", "expected_action"),
    [
        (
            {"requires_validation": True, "signals": [{"status": "drift", "score": 0.8}]},
            "run_validation",
        ),
        (
            {"requires_objective_review": True, "signals": [{"status": "drift", "score": 0.8}]},
            "revise_objective",
        ),
        (
            {"requires_context_review": True, "signals": [{"status": "drift", "score": 0.8}]},
            "request_human_observation",
        ),
    ],
)
def test_decision_layer_consumes_drift_only_as_bounded_review_action(summary, expected_action):
    from app.services.decision_layer import CampaignDecisionLayer
    from app.services.round_context import build_campaign_round_context

    context = build_campaign_round_context(
        campaign_id="decision",
        round_index=3,
        drift_summary=summary,
        decision_memory={"records": []},
        strategy_selection_result={"backend": "bo"},
    )

    plan = CampaignDecisionLayer().decide(context)

    assert plan.action_type.value == expected_action
    assert plan.shadow_only is True
    if plan.objective_patch is not None:
        assert plan.objective_patch.proposed_changes["auto_applied"] is False


def test_monitor_decision_layer_and_authority_form_a_bounded_validation_chain():
    from app.services.campaign_decision_authority import (
        evaluate_campaign_decision_authority,
    )
    from app.services.decision_layer import CampaignDecisionLayer
    from app.services.round_context import build_campaign_round_context

    report = assess_closed_loop_drift(
        campaign_id="authority-chain",
        round_index=3,
        campaign_context={"instrument_state": {"calibration_confidence": 0.1}},
    )
    plan = CampaignDecisionLayer().decide(
        build_campaign_round_context(
            campaign_id="authority-chain",
            round_index=3,
            drift_summary=report.model_dump(mode="json"),
        )
    )

    disabled = evaluate_campaign_decision_authority(plan, enabled=False)
    enabled = evaluate_campaign_decision_authority(plan, enabled=True)

    assert plan.action_type.value == "run_validation"
    assert disabled.proceed_to_candidates is True
    assert enabled.proceed_to_candidates is False
    assert enabled.round_status == "deferred"
    assert any(update.update_type == "validation_request" for update in enabled.state_updates)


async def test_orchestrator_threads_report_outcome_and_applicability_into_next_round(
    monkeypatch,
    tmp_path,
):
    from app.agents.orchestrator import OrchestratorAgent, OrchestratorInput
    from app.core.config import get_settings
    from app.core.db import init_db
    from app.services.campaign_events import replay_events
    from app.services.campaign_state import load_all_candidates, load_campaign
    from app.services.decision_trajectory import load_trajectories

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "data" / "orchestrator.db"))
    monkeypatch.setenv("OBJECT_STORE_DIR", str(tmp_path / "objects"))
    monkeypatch.setenv("SCIENTIFIC_LEDGER_ENABLED", "false")
    monkeypatch.setenv("CONTEXTUAL_DECISION_SHADOW_ENABLED", "false")
    monkeypatch.setenv("CAMPAIGN_DECISION_AUTHORITY_ENABLED", "false")
    monkeypatch.setenv("CLOSED_LOOP_DRIFT_MONITOR_ENABLED", "true")
    get_settings.cache_clear()
    init_db()

    campaign_id = "camp-drift-integration"
    result = await OrchestratorAgent().process(
        OrchestratorInput(
            contract_id="contract-drift",
            objective_kpi="yield",
            direction="maximize",
            max_rounds=1,
            batch_size=2,
            strategy="lhs",
            dry_run=True,
            campaign_id=campaign_id,
            dimensions=[
                {
                    "param_name": "temperature_c",
                    "param_type": "number",
                    "min_value": 20,
                    "max_value": 100,
                }
            ],
            protocol_template={"steps": [{"primitive": "log", "params": {}}]},
            closed_loop_context={
                "instrument_state": {
                    "instrument_id": "reader-1",
                    "calibration_id": "cal-2026-07",
                    "calibration_confidence": 0.9,
                },
                "proxy_gap_assessment": {"score": 0.2, "level": "low"},
            },
        )
    )

    assert result.status == "completed"
    payloads = [event["payload"] for event in replay_events(campaign_id)]
    reports = [payload for payload in payloads if payload.get("type") == "closed_loop_drift_report"]
    assert reports
    assert reports[0]["round"] == 1
    assert reports[-1]["round"] == 2

    trajectories = load_trajectories(campaign_id)
    assert any(row["layer"] == "campaign" for row in trajectories)
    campaign_trajectory = next(row["trajectory"] for row in trajectories if row["layer"] == "campaign")
    assert "drift_summary" in campaign_trajectory["trace"]["context"]

    state = load_campaign(campaign_id)
    assert state is not None
    next_context = state["campaign_context"]
    assert next_context["closed_loop_drift_report"]["round_index"] == 2
    assert next_context["decision_memory"]["record_count"] == 1

    candidates = load_all_candidates(campaign_id)
    assert candidates
    assert all(row["applicability_context"]["objective_kpi"] == "yield" for row in candidates)
    assert all(row["applicability_context"]["calibration_id"] == "cal-2026-07" for row in candidates)

    get_settings.cache_clear()
