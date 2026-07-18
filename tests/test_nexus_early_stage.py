from __future__ import annotations

import io
import json
from urllib.error import HTTPError

from app.core.config import Settings
from app.services.campaign_mode import CampaignMode
from app.services.dynamic_action_space import (
    ActionShadowLabel,
    ActionSpec,
    build_action_space_snapshot,
)
from app.services.nexus_early_stage import (
    NexusEarlyStageAdapter,
    NexusEarlyStageClient,
    NexusEarlyStageErrorType,
)


class _FakeHTTPResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self.payload = payload
        self.status = status

    def __enter__(self):  # noqa: ANN204
        return self

    def __exit__(self, *args):  # noqa: ANN002, ANN204
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def _http_error(status: int, payload: dict) -> HTTPError:
    return HTTPError(
        url="http://nexus.test/api/early-stage/analyze",
        code=status,
        msg="bad",
        hdrs={},
        fp=io.BytesIO(json.dumps(payload).encode("utf-8")),
    )


def _report(**overrides):
    base = {
        "contract_version": "early_stage_system_characterization.v1",
        "recommended_campaign_mode": "optimization_ready",
        "confidence": 0.8,
        "risk_flags": [],
        "diagnostic_recommendations": [],
    }
    base.update(overrides)
    return base


def test_settings_normalizes_nexus_url(monkeypatch):
    monkeypatch.setenv("NEXUS_URL", "http://nexus.test")
    assert Settings().nexus_url == "http://nexus.test/api"

    monkeypatch.setenv("NEXUS_URL", "http://nexus.test/api")
    assert Settings().nexus_url == "http://nexus.test/api"


def test_client_intake_success_extracts_analysis_report(monkeypatch):
    def fake_urlopen(request, timeout):  # noqa: ANN001, ANN202
        assert request.get_method() == "POST"
        assert timeout == 1.5
        return _FakeHTTPResponse(
            {
                "analysis_report": _report(
                    recommended_campaign_mode="controllability_mapping",
                    risk_flags=["poor_controllability"],
                ),
                "observations": [{"iteration": 0}],
                "parameter_specs": [{"name": "flow_rate"}],
            }
        )

    monkeypatch.setattr("app.services.nexus_early_stage.urlopen", fake_urlopen)

    response = NexusEarlyStageClient(
        base_url="http://nexus.test/api",
        timeout_seconds=1.5,
    ).intake({"campaign_id": "camp-1"})

    assert response.ok is True
    assert response.campaign_id == "camp-1"
    assert response.contract_version == "early_stage_system_characterization.v1"
    assert response.recommended_campaign_mode == "controllability_mapping"
    assert response.risk_flags == ("poor_controllability",)
    assert response.observations == ({"iteration": 0},)


def test_client_400_returns_typed_bad_request(monkeypatch):
    def fake_urlopen(request, timeout):  # noqa: ANN001, ANN202, ARG001
        raise _http_error(400, {"detail": "bad mapping"})

    monkeypatch.setattr("app.services.nexus_early_stage.urlopen", fake_urlopen)

    response = NexusEarlyStageClient(base_url="http://nexus.test/api").analyze(
        {"campaign_id": "camp-1"}
    )

    assert response.ok is False
    assert response.status_code == 400
    assert response.error_type == NexusEarlyStageErrorType.BAD_REQUEST
    assert response.error_message == "bad mapping"


def test_client_404_returns_typed_not_found(monkeypatch):
    def fake_urlopen(request, timeout):  # noqa: ANN001, ANN202, ARG001
        raise _http_error(404, {"detail": "missing campaign"})

    monkeypatch.setattr("app.services.nexus_early_stage.urlopen", fake_urlopen)

    response = NexusEarlyStageClient(base_url="http://nexus.test/api").report("camp-1")

    assert response.ok is False
    assert response.error_type == NexusEarlyStageErrorType.NOT_FOUND
    assert response.error_message == "missing campaign"


def test_client_timeout_returns_typed_timeout(monkeypatch):
    def fake_urlopen(request, timeout):  # noqa: ANN001, ANN202, ARG001
        raise TimeoutError("slow nexus")

    monkeypatch.setattr("app.services.nexus_early_stage.urlopen", fake_urlopen)

    response = NexusEarlyStageClient(base_url="http://nexus.test/api").analyze(
        {"campaign_id": "camp-1"}
    )

    assert response.ok is False
    assert response.error_type == NexusEarlyStageErrorType.TIMEOUT


def test_client_unsupported_contract_degrades_without_dropping_report(monkeypatch):
    def fake_urlopen(request, timeout):  # noqa: ANN001, ANN202, ARG001
        return _FakeHTTPResponse(
            {"report": _report(contract_version="early_stage_system_characterization.v2")}
        )

    monkeypatch.setattr("app.services.nexus_early_stage.urlopen", fake_urlopen)

    response = NexusEarlyStageClient(base_url="http://nexus.test/api").report("camp-1")

    assert response.ok is False
    assert response.error_type == NexusEarlyStageErrorType.UNSUPPORTED_CONTRACT_VERSION
    assert response.report is not None
    assert response.contract_version == "early_stage_system_characterization.v2"


def test_adapter_unsupported_contract_requires_review_without_evidence():
    advice = NexusEarlyStageAdapter().adapt(
        _report(
            contract_version="early_stage_system_characterization.v2",
            risk_flags=["poor_controllability"],
        )
    )

    assert advice.requires_operator_approval is True
    assert advice.campaign_mode_hint is None
    assert advice.evidence == ()
    assert advice.audit_metadata["error_type"] == (
        NexusEarlyStageErrorType.UNSUPPORTED_CONTRACT_VERSION
    )


def test_adapter_gates_bo_for_poor_controllability_and_preserves_audit():
    advice = NexusEarlyStageAdapter().adapt(
        _report(
            recommended_campaign_mode="optimization_ready",
            risk_flags=["poor_controllability"],
            diagnostic_recommendations=[
                {"action_type": "run_controllability_mapping", "priority": "high"}
            ],
            insights=["actual temperature lags target"],
        ),
        endpoint_used="/early-stage/analyze",
        campaign_id="camp-1",
    )

    assert advice.campaign_mode_hint == CampaignMode.CONTROLLABILITY_MAPPING
    assert advice.ordinary_bo_allowed is False
    assert advice.requires_operator_approval is True
    assert advice.audit_metadata["contract_version"] == (
        "early_stage_system_characterization.v1"
    )
    assert advice.audit_metadata["risk_flags"] == ["poor_controllability"]
    assert advice.audit_metadata["top_diagnostic_recommendations"] == [
        {"action_type": "run_controllability_mapping", "priority": "high"}
    ]
    assert any(e.target_action == "controllability_mapping" for e in advice.evidence)
    assert "actual temperature lags target" in advice.operator_messages


def test_adapter_prioritizes_objective_missing_over_other_risks():
    advice = NexusEarlyStageAdapter().adapt(
        _report(
            recommended_campaign_mode="hardware_feasibility_discovery",
            risk_flags=["hardware_failures_dominate", "objective_missing"],
        )
    )

    assert advice.campaign_mode_hint == CampaignMode.OBJECTIVE_DISCOVERY
    assert advice.ordinary_bo_allowed is False


def test_adapter_maps_hardware_failures_and_danger_zone_adjustments():
    advice = NexusEarlyStageAdapter().adapt(
        _report(
            recommended_campaign_mode="optimization_ready",
            risk_flags=["hardware_failures_dominate"],
            feasibility_summary={
                "danger_zones": [{"parameter": "flow_rate", "lower": 7.0, "upper": 10.0}]
            },
            hardware_summary={"worst_design_id": "thin-wall-alpha"},
        )
    )

    assert advice.campaign_mode_hint == CampaignMode.HARDWARE_FEASIBILITY_DISCOVERY
    assert advice.ordinary_bo_allowed is False
    adjustment_types = {item.adjustment_type for item in advice.action_space_adjustments}
    assert "reject_or_annotate_danger_zone" in adjustment_types
    assert "route_by_hardware_design" in adjustment_types
    assert any(item.reject_by_default for item in advice.action_space_adjustments)


def test_adapter_optimization_ready_allows_guarded_bo():
    advice = NexusEarlyStageAdapter().adapt(_report(confidence=0.9))

    assert advice.campaign_mode_hint == CampaignMode.BO_OPTIMIZATION
    assert advice.ordinary_bo_allowed is True
    assert advice.requires_operator_approval is False
    assert any(e.target_action == "exploit" for e in advice.evidence)


def test_new_campaign_modes_are_reachable_in_action_space():
    mode_decision = NexusEarlyStageAdapter().adapt(
        _report(risk_flags=["objective_missing"])
    ).campaign_mode_hint
    assert mode_decision == CampaignMode.OBJECTIVE_DISCOVERY

    from app.services.campaign_mode import CampaignModeDecision

    snapshot = build_action_space_snapshot(
        mode_decision=CampaignModeDecision(
            campaign_id="camp-1",
            round_index=0,
            mode=mode_decision,
            priority_rank=1,
            reason="objective missing",
        ),
        actions=[
            ActionSpec(name="rank_kpis", kind="objective_discovery"),
            ActionSpec(name="run_bo", kind="optimization"),
        ],
        available_capabilities=[],
    )

    labels = {item.name: item.label for item in snapshot.assessments}
    assert labels["rank_kpis"] == ActionShadowLabel.PREFERRED
    assert labels["run_bo"] == ActionShadowLabel.RISKY
