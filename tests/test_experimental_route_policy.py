from __future__ import annotations

import json
from copy import deepcopy
from unittest.mock import patch

from app.services.experimental_route_policy import (
    normalize_route_dimensions,
    select_experimental_route,
)
from app.services.nexus_experimental_routes import (
    NexusExperimentalRouteClient,
    NexusExperimentalRouteErrorType,
    build_experimental_route_payload,
)


def _report(*, approval_required: bool = False, capability_status: str = "available"):
    template = {"steps": [{"primitive": "log", "params": {}}]}
    return {
        "campaign_id": "camp-route",
        "contract_version": "experimental_route_intelligence.v1",
        "authority": "advisory_only",
        "confidence": 0.7,
        "capability_inventory_supplied": True,
        "risk_flags": [],
        "graph": {
            "graph_id": "routes",
            "active_node_id": "baseline",
            "nodes": [
                {
                    "node_id": "baseline",
                    "label": "Baseline",
                    "parameter_space": [],
                    "protocol_ref": {"use_campaign_default": True},
                    "expected_cost": 1.0,
                    "expected_duration_s": 10.0,
                    "safety_risk": 0.1,
                },
                {
                    "node_id": "alternate",
                    "label": "Alternate synthesis",
                    "parameter_space": [
                        {"name": "temperature", "lower": 300, "upper": 700}
                    ],
                    "protocol_ref": {"protocol_template": template},
                    "required_capabilities": ["furnace"],
                    "expected_cost": 1.0,
                    "expected_duration_s": 10.0,
                    "safety_risk": 0.1,
                },
            ],
            "transitions": [
                {
                    "source_id": "baseline",
                    "target_id": "alternate",
                    "approval_required": approval_required,
                }
            ],
        },
        "node_assessments": [
            {
                "node_id": "baseline",
                "status": "high_failure",
                "failure_rate": 0.9,
                "information_gap": 0.0,
                "normalized_prior": 0.1,
                "evidence_strength": 0.8,
                "missing_capabilities": [],
                "objective_summaries": [],
            },
            {
                "node_id": "alternate",
                "status": "unobserved",
                "failure_rate": 0.0,
                "information_gap": 1.0,
                "normalized_prior": 0.9,
                "evidence_strength": 0.0,
                "missing_capabilities": [],
                "objective_summaries": [],
            },
        ],
        "available_transitions": [
            {
                "source_id": "baseline",
                "target_id": "alternate",
                "transition_id": "baseline->alternate",
                "switch_cost": 0.0,
                "switch_duration_s": 0.0,
                "approval_required": approval_required,
                "target_capability_status": capability_status,
                "target_missing_capabilities": [],
            }
        ],
    }


def _select(
    report,
    *,
    enabled: bool,
    policy=None,
    execution_graph=None,
    available_capabilities=None,
):
    return select_experimental_route(
        report=report,
        execution_graph=execution_graph or report["graph"],
        campaign_dimensions=[
            {"param_name": "voltage", "min_value": 0.1, "max_value": 1.0}
        ],
        campaign_protocol_template={"steps": []},
        campaign_protocol_pattern_id="",
        direction="maximize",
        authority_enabled=enabled,
        available_capabilities=(
            ["furnace"]
            if available_capabilities is None
            else available_capabilities
        ),
        policy_snapshot=policy,
    )


def test_shadow_policy_records_preferred_route_without_applying_it() -> None:
    decision = _select(_report(), enabled=False)

    assert decision.selected_node_id == "alternate"
    assert decision.applied is False
    assert decision.changed is False
    assert "Shadow policy prefers alternate" in decision.reason


def test_live_policy_applies_executable_approved_route() -> None:
    decision = _select(_report(), enabled=True)

    assert decision.applied is True
    assert decision.changed is True
    assert decision.selected_option is not None
    assert decision.selected_option.runtime is not None
    assert decision.selected_option.runtime.dimensions[0]["param_name"] == "temperature"


def test_operator_approval_and_capability_inventory_are_hard_gates() -> None:
    pending = _select(_report(approval_required=True), enabled=True)
    alternate = next(item for item in pending.options if item.node_id == "alternate")
    assert pending.applied is False
    assert "operator_approval_required" in alternate.rejection_reasons

    approved = _select(
        _report(approval_required=True),
        enabled=True,
        policy={"approved_experimental_route_transitions": ["baseline->alternate"]},
    )
    assert approved.applied is True

    unknown = _select(_report(capability_status="unknown"), enabled=True)
    alternate = next(item for item in unknown.options if item.node_id == "alternate")
    assert unknown.applied is False
    assert "target_capability_status_unknown" in alternate.rejection_reasons


def test_nexus_cannot_replace_helios_execution_mapping() -> None:
    report = _report()
    helios_graph = deepcopy(report["graph"])
    report["graph"]["nodes"][1]["protocol_ref"] = {
        "protocol_template": {"steps": [{"primitive": "untrusted.execute"}]}
    }

    decision = _select(
        report,
        enabled=True,
        execution_graph=helios_graph,
    )

    assert decision.applied is True
    assert decision.selected_option is not None
    assert decision.selected_option.runtime is not None
    assert decision.selected_option.runtime.protocol_template == {
        "steps": [{"primitive": "log", "params": {}}]
    }


def test_nexus_cannot_claim_a_locally_missing_capability_is_available() -> None:
    decision = _select(
        _report(capability_status="available"),
        enabled=True,
        available_capabilities=[],
    )

    alternate = next(item for item in decision.options if item.node_id == "alternate")
    assert decision.applied is False
    assert "missing_required_capabilities" in alternate.rejection_reasons


def test_non_finite_policy_threshold_cannot_bypass_safety_gate() -> None:
    report = _report()
    report["graph"]["nodes"][1]["safety_risk"] = 0.9

    decision = _select(
        report,
        enabled=True,
        policy={"experimental_route_max_safety_risk": float("nan")},
    )

    alternate = next(item for item in decision.options if item.node_id == "alternate")
    assert decision.applied is False
    assert "safety_risk_above_policy" in alternate.rejection_reasons


def test_invalid_route_parameter_space_is_not_promoted_live() -> None:
    report = _report()
    report["graph"]["nodes"][1]["parameter_space"] = [
        {"name": "temperature", "lower": 700, "upper": 300}
    ]

    decision = _select(report, enabled=True)

    alternate = next(item for item in decision.options if item.node_id == "alternate")
    assert decision.applied is False
    assert "route_has_no_executable_helios_mapping" in alternate.rejection_reasons


def test_policy_blocks_execution_when_current_and_alternate_routes_are_ineligible() -> None:
    report = _report(approval_required=True)
    report["graph"]["nodes"][0]["safety_risk"] = 0.9

    decision = _select(report, enabled=True)

    assert decision.selected_node_id is None
    assert decision.execution_allowed is False
    assert decision.applied is False


def test_route_dimension_normalizer_accepts_nexus_bounds() -> None:
    assert normalize_route_dimensions(
        [{"name": "temperature", "lower": 20, "upper": 80}]
    ) == [
        {
            "param_name": "temperature",
            "param_type": "number",
            "min_value": 20,
            "max_value": 80,
            "log_scale": False,
        }
    ]


def test_experimental_route_gates_default_off(monkeypatch) -> None:
    from app.core.config import get_settings

    monkeypatch.delenv("NEXUS_EXPERIMENTAL_ROUTES_ENABLED", raising=False)
    monkeypatch.delenv("EXPERIMENTAL_ROUTE_AUTHORITY_ENABLED", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.nexus_experimental_routes_enabled is False
    assert settings.experimental_route_authority_enabled is False
    get_settings.cache_clear()


class _FakeHTTPResponse:
    status = 200

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit=-1):
        return json.dumps(self.payload).encode()


def test_nexus_client_sends_api_key_and_rejects_non_advisory_authority() -> None:
    report = _report()
    with patch(
        "app.services.nexus_experimental_routes.urlopen",
        return_value=_FakeHTTPResponse({"report": report}),
    ) as opener:
        response = NexusExperimentalRouteClient(
            base_url="http://nexus.test/api", api_key="secret"
        ).analyze({"campaign_id": "camp-route"})

    assert response.ok is True
    assert opener.call_args.args[0].headers["X-api-key"] == "secret"

    report["authority"] = "route_selector"
    with patch(
        "app.services.nexus_experimental_routes.urlopen",
        return_value=_FakeHTTPResponse({"report": report}),
    ):
        rejected = NexusExperimentalRouteClient(
            base_url="http://nexus.test/api"
        ).analyze({"campaign_id": "camp-route"})
    assert rejected.ok is False
    assert rejected.error_type == NexusExperimentalRouteErrorType.INVALID_AUTHORITY


def test_payload_builder_strips_server_generated_transition_id() -> None:
    payload = build_experimental_route_payload(
        campaign_id="c",
        graph={
            "nodes": [{"node_id": "a", "label": "A", "unexpected": True}],
            "transitions": [
                {
                    "source_id": "a",
                    "target_id": "b",
                    "transition_id": "a->b",
                }
            ],
        },
        observations=[],
        objective="yield",
        direction="maximize",
        available_capabilities=None,
    )

    assert "unexpected" not in payload["graph"]["nodes"][0]
    assert "transition_id" not in payload["graph"]["transitions"][0]


def test_payload_builder_caps_observation_history() -> None:
    observations = [{"iteration": index} for index in range(10_005)]
    payload = build_experimental_route_payload(
        campaign_id="c",
        graph={"nodes": [{"node_id": "a", "label": "A"}]},
        observations=observations,
        objective="yield",
        direction="maximize",
        available_capabilities=None,
    )

    assert len(payload["observations"]) == 10_000
    assert payload["observations"][0]["iteration"] == 5


async def test_orchestrator_applies_route_and_records_route_labelled_observation(
    monkeypatch, tmp_path
) -> None:
    from app.agents.orchestrator import OrchestratorAgent, OrchestratorInput
    from app.core.config import get_settings
    from app.core.db import init_db
    from app.services.campaign_events import replay_events
    from app.services.campaign_state import load_all_candidates, load_campaign
    from app.services.nexus_experimental_routes import (
        NexusExperimentalRouteResponse,
    )

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "data" / "orchestrator.db"))
    monkeypatch.setenv("OBJECT_STORE_DIR", str(tmp_path / "objects"))
    monkeypatch.setenv("SCIENTIFIC_LEDGER_ENABLED", "true")
    monkeypatch.setenv(
        "SCIENTIFIC_LEDGER_ROOT", str(tmp_path / "scientific-ledger")
    )
    monkeypatch.setenv("SCIENTIFIC_LEDGER_GIT_ENABLED", "false")
    monkeypatch.setenv("NEXUS_EXPERIMENTAL_ROUTES_ENABLED", "true")
    monkeypatch.setenv("EXPERIMENTAL_ROUTE_AUTHORITY_ENABLED", "true")
    get_settings.cache_clear()
    init_db()

    report = _report()
    monkeypatch.setattr(
        "app.services.nexus_experimental_routes.NexusExperimentalRouteClient.analyze",
        lambda _self, _payload: NexusExperimentalRouteResponse(
            ok=True,
            endpoint="http://nexus.test/api/experimental-routes/analyze",
            status_code=200,
            campaign_id="camp-route-live",
            report=report,
            raw={"report": report},
        ),
    )
    graph = dict(report["graph"])
    graph["transitions"] = [
        {
            "source_id": "baseline",
            "target_id": "alternate",
            "approval_required": False,
        }
    ]
    result = await OrchestratorAgent().process(
        OrchestratorInput(
            contract_id="contract-route-live",
            objective_kpi="yield",
            direction="maximize",
            max_rounds=1,
            batch_size=1,
            strategy="lhs",
            dry_run=True,
            campaign_id="camp-route-live",
            dimensions=[
                {
                    "param_name": "voltage",
                    "param_type": "number",
                    "min_value": 0.1,
                    "max_value": 1.0,
                }
            ],
            protocol_template={"steps": [{"primitive": "log", "params": {}}]},
            experimental_route_graph=graph,
            available_capabilities=["furnace"],
        )
    )

    assert result.status == "completed"
    saved = load_campaign("camp-route-live")
    assert saved is not None
    context = saved["campaign_context"]
    assert context["active_experimental_node_id"] == "alternate"
    assert context["experimental_route_decisions"][0]["applied"] is True
    assert context["experimental_route_observations"][0]["metadata"][
        "experimental_node_id"
    ] == "alternate"
    candidates = load_all_candidates("camp-route-live")
    assert "temperature" in candidates[0]["params"]
    events = [item["payload"] for item in replay_events("camp-route-live")]
    assert any(
        event.get("type") == "experimental_route_decision"
        and event.get("selected_node_id") == "alternate"
        and event.get("applied") is True
        for event in events
    )
    from app.services.decision_trajectory import load_trajectories

    trajectory = next(
        row["trajectory"]
        for row in load_trajectories("camp-route-live")
        if row["layer"] == "campaign"
    )
    route_trace = trajectory["trace"]["metadata"]["experimental_route_decision"]
    assert route_trace["selected_node_id"] == "alternate"
    assert route_trace["applied"] is True
    get_settings.cache_clear()
