from __future__ import annotations

from app.services.campaign_decision_authority import (
    evaluate_campaign_decision_authority,
)
from app.services.decision_models import (
    CampaignContextRequest,
    CampaignDecisionAction,
    CampaignDecisionPlan,
    ConstraintPatch,
    ObjectivePatch,
)


def _plan(
    action: CampaignDecisionAction,
    **kwargs,
) -> CampaignDecisionPlan:
    return CampaignDecisionPlan(
        action_type=action,
        rationale=kwargs.pop("rationale", f"{action.value} rationale"),
        **kwargs,
    )


def test_disabled_authority_never_consumes_shadow_plan():
    verdict = evaluate_campaign_decision_authority(
        _plan(CampaignDecisionAction.STOP_CAMPAIGN),
        enabled=False,
    )

    assert verdict.consumed is False
    assert verdict.proceed_to_candidates is True
    assert verdict.terminal is False
    assert verdict.reason == "campaign decision authority disabled"


def test_propose_candidates_continues_when_authority_enabled():
    verdict = evaluate_campaign_decision_authority(
        _plan(
            CampaignDecisionAction.PROPOSE_CANDIDATES,
            candidate_generation_backend="gp_backend",
        ),
        enabled=True,
    )

    assert verdict.consumed is False
    assert verdict.proceed_to_candidates is True
    assert verdict.terminal is False


def test_stop_campaign_becomes_terminal_authority_verdict():
    verdict = evaluate_campaign_decision_authority(
        _plan(CampaignDecisionAction.STOP_CAMPAIGN),
        enabled=True,
    )

    assert verdict.consumed is True
    assert verdict.proceed_to_candidates is False
    assert verdict.terminal is True
    assert verdict.stop_reason == "campaign_decision_authority_stop"
    assert verdict.round_status == "completed"


def test_objective_and_constraint_actions_emit_persistable_updates():
    verdict = evaluate_campaign_decision_authority(
        _plan(
            CampaignDecisionAction.REVISE_OBJECTIVE,
            objective_patch=ObjectivePatch(
                reason="proxy gap too high",
                proposed_changes={"active_objective": "functional_kpi"},
            ),
            constraint_patch=ConstraintPatch(
                reason="tighten unsafe region",
                proposed_changes={"temperature_c": {"max": 80}},
            ),
        ),
        enabled=True,
    )

    assert verdict.consumed is True
    assert verdict.proceed_to_candidates is False
    updates = {update.update_type: update.payload for update in verdict.state_updates}
    assert updates["objective_transition"]["proposed_changes"] == {
        "active_objective": "functional_kpi"
    }
    assert updates["objective_transition"]["auto_applied"] is False
    assert updates["space_revision"]["revision_type"] == "constraint_update"
    assert updates["space_revision"]["approval_required"] is True


def test_context_actions_synthesize_or_preserve_context_requests():
    literature = evaluate_campaign_decision_authority(
        _plan(CampaignDecisionAction.QUERY_LITERATURE),
        enabled=True,
    )
    assert literature.state_updates[0].update_type == "context_request"
    assert literature.state_updates[0].payload["request_type"] == "literature_context"

    human = evaluate_campaign_decision_authority(
        _plan(
            CampaignDecisionAction.REQUEST_HUMAN_OBSERVATION,
            context_requests=[
                CampaignContextRequest(
                    request_type="failure_attribution",
                    reason="low confidence",
                    priority="high",
                    target="failure_summary",
                )
            ],
        ),
        enabled=True,
    )
    assert human.state_updates[0].payload["request_type"] == "failure_attribution"
    assert human.state_updates[0].payload["target"] == "failure_summary"


def test_validation_and_recovery_emit_action_requests():
    validation = evaluate_campaign_decision_authority(
        _plan(CampaignDecisionAction.RUN_VALIDATION),
        enabled=True,
    )
    recovery = evaluate_campaign_decision_authority(
        _plan(CampaignDecisionAction.RECOVER_FAILURE),
        enabled=True,
    )

    validation_types = [update.update_type for update in validation.state_updates]
    recovery_types = [update.update_type for update in recovery.state_updates]
    assert "validation_request" in validation_types
    assert "recovery_request" in recovery_types


def test_campaign_decision_authority_config_defaults_off(monkeypatch):
    from app.core.config import get_settings

    monkeypatch.delenv("CAMPAIGN_DECISION_AUTHORITY_ENABLED", raising=False)
    get_settings.cache_clear()
    assert get_settings().campaign_decision_authority_enabled is False

    monkeypatch.setenv("CAMPAIGN_DECISION_AUTHORITY_ENABLED", "true")
    get_settings.cache_clear()
    assert get_settings().campaign_decision_authority_enabled is True

    get_settings.cache_clear()


async def test_orchestrator_consumes_enabled_authority_before_candidate_generation(
    monkeypatch,
    tmp_path,
):
    from app.agents.orchestrator import OrchestratorAgent, OrchestratorInput
    from app.core.config import get_settings
    from app.core.db import init_db
    from app.services.campaign_events import replay_events

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "data" / "orchestrator.db"))
    monkeypatch.setenv("OBJECT_STORE_DIR", str(tmp_path / "objects"))
    monkeypatch.setenv("CAMPAIGN_DECISION_AUTHORITY_ENABLED", "true")
    monkeypatch.setenv("SCIENTIFIC_LEDGER_ENABLED", "false")
    get_settings.cache_clear()
    init_db()

    campaign_id = "camp-authority-integration"
    orchestrator = OrchestratorAgent()
    result = await orchestrator.process(
        OrchestratorInput(
            contract_id="contract-authority",
            objective_kpi="yield",
            direction="maximize",
            max_rounds=1,
            batch_size=2,
            strategy="lhs",
            dry_run=True,
            campaign_id=campaign_id,
            policy_snapshot={"risk_level": "high"},
            dimensions=[
                {
                    "param_name": "temperature_c",
                    "param_type": "number",
                    "min_value": 20,
                    "max_value": 100,
                }
            ],
            protocol_template={"steps": [{"primitive": "log", "params": {}}]},
        )
    )

    assert result.status == "completed"
    events = replay_events(campaign_id)
    payloads = [event["payload"] for event in events]
    authority = [
        payload
        for payload in payloads
        if payload.get("type") == "campaign_decision_authority"
    ]
    assert authority
    assert authority[0]["consumed"] is True
    assert authority[0]["action_type"] == "tighten_constraints"
    assert any(payload.get("type") == "round_deferred" for payload in payloads)
    assert not any(
        payload.get("type") == "agent_result" and payload.get("agent") == "design"
        for payload in payloads
    )
    from app.services.decision_trajectory import load_trajectories

    rows = load_trajectories(campaign_id)
    assert len(rows) == 1
    trajectory = rows[0]["trajectory"]
    assert trajectory["trace"]["actual_action"] == "tighten_constraints"
    assert trajectory["trace"]["comparison"]["would_change_route"] is False
    assert trajectory["outcome"]["observed_action"] == "tighten_constraints"

    get_settings.cache_clear()
