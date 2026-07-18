from __future__ import annotations

from app.services.decision_markdown import (
    LedgerProvenance,
    redact_sensitive,
    render_completed_decision,
    render_nexus_snapshot,
    render_pending_decision,
    render_policy_snapshot,
    render_trajectory_figure,
)
from tests.fixtures.scientific_ledger import decision_accounting, decision_trace


def _provenance() -> LedgerProvenance:
    return LedgerProvenance(
        code_commit="abc123",
        policy_id="campaign-meta-controller",
        policy_version="v4",
        nexus_contract_version="early_stage_system_characterization.v1",
        rubric_version="v0.1_static",
    )


def test_pending_bundle_is_deterministic_and_complete():
    trace = decision_trace()
    first = render_pending_decision(trace, provenance=_provenance())
    second = render_pending_decision(trace, provenance=_provenance())

    assert first == second
    assert set(first.files) == {
        "rounds/003/objective.md",
        "rounds/003/observations.md",
        "rounds/003/decision_003.md",
        "rounds/003/strategy.md",
        "rounds/003/evidence.md",
        "rounds/003/failure.md",
        "rounds/003/recovery.md",
        "rounds/003/summary.md",
    }
    card = first.files["rounds/003/decision_003.md"]
    assert "status: pending" in card
    assert "## Candidate Actions" in card
    assert "Information gain" in card
    assert "validation" in card
    assert "Pending" in card
    assert "sk-test-secret-value" not in card
    assert "[REDACTED]" in card


def test_completed_card_contains_outcome_reward_failure_and_recovery():
    accounting = decision_accounting()
    bundle = render_completed_decision(
        accounting,
        provenance=_provenance(),
        observations=[{"measurement": "yield", "value": 0.84}],
        failures=[{"failure_type": "hardware", "root_cause": "pipette offset"}],
        recovery_events=[{"fix": "increase z offset 0.5mm", "result": "pass"}],
    )
    card = bundle.files["rounds/003/decision_003.md"]
    assert "status: completed" in card
    assert "Execution success | yes" in card
    assert "Total reward:" in card
    assert "Verifier | Passed | Score" in card
    assert "Failure events: 1" in card
    assert "Recovery events: 1" in card
    assert "Bearer super-secret-token" not in "\n".join(bundle.files.values())
    assert "pipette offset" in bundle.files["rounds/003/failure.md"]
    assert "increase z offset 0.5mm" in bundle.files["rounds/003/recovery.md"]


def test_redaction_recurses_through_nested_payloads_and_strings():
    value = {
        "api_key": "secret",
        "nested": {
            "Authorization": "Bearer abcdefghijklmnop",
            "note": "token sk-1234567890abcdefghijkl",
        },
    }
    redacted = redact_sensitive(value)
    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["nested"]["Authorization"] == "[REDACTED]"
    assert "sk-" not in redacted["nested"]["note"]


def test_policy_nexus_and_trajectory_are_markdown_artifacts():
    provenance = _provenance()
    policy = render_policy_snapshot(
        campaign_id="campaign-32",
        policy={"rule": "BO if stable", "failure_recovery": True},
        provenance=provenance,
    )
    nexus = render_nexus_snapshot(
        campaign_id="campaign-32",
        diagnostics={"entropy_score": 0.2, "failure_attribution": {"offset": 0.8}},
        provenance=provenance,
    )
    trajectory = render_trajectory_figure(
        campaign_id="campaign-32",
        decision_rows=[
            {"round_index": 1, "action": "validation", "status": "completed", "reward": 0.4},
            {"round_index": 2, "action": "propose_candidates", "status": "completed", "reward": 0.6},
        ],
    )
    assert "# Decision Policy" in policy and "BO if stable" in policy
    assert "# Nexus Optimization Health" in nexus and "failure_attribution.offset" in nexus
    assert "```mermaid" in trajectory and "d0 --> d1" in trajectory


def test_trajectory_uses_decision_card_front_matter_names():
    trajectory = render_trajectory_figure(
        campaign_id="campaign-32",
        decision_rows=[
            {
                "round_index": 3,
                "selected_action": "validate",
                "selected_backend": "bo_mcp",
                "status": "completed",
                "reward": 0.8,
            }
        ],
    )

    assert "R3 · validate · completed · reward=0.8" in trajectory


def test_renderer_tolerates_scalar_campaign_objective_metadata():
    trace = decision_trace().model_copy(deep=True)
    trace.context.objective_summary = {}

    bundle = render_pending_decision(
        trace,
        provenance=_provenance(),
        campaign_metadata={"objective": "maximize yield"},
    )

    assert "**value:** maximize yield" in bundle.files["rounds/003/objective.md"]
