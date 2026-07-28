"""Live contract test across the sibling Nexus and HELIOS checkouts."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from app.services.experimental_route_policy import select_experimental_route


def _nexus_repo() -> Path:
    return Path(os.getenv("NEXUS_REPO_PATH", "/Users/sissifeng/Nexus"))


def test_nexus_report_is_consumed_by_helios_route_authority(tmp_path) -> None:
    nexus_repo = _nexus_repo()
    if not (nexus_repo / "optimization_copilot").is_dir():
        pytest.skip("Sibling Nexus checkout is unavailable")
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    sys.path.insert(0, str(nexus_repo))
    try:
        from optimization_copilot.api.app import create_app

        client = TestClient(create_app(workspace_dir=str(tmp_path / "nexus-workspace")))
        response = client.post(
            "/api/experimental-routes/analyze",
            json={
                "campaign_id": "helios-nexus-joint",
                "graph": {
                    "graph_id": "joint-routes",
                    "active_node_id": "baseline",
                    "nodes": [
                        {
                            "node_id": "baseline",
                            "label": "Baseline synthesis",
                            "parameter_space": [
                                {"name": "voltage", "lower": 0.1, "upper": 1.0}
                            ],
                            "protocol_ref": {"use_campaign_default": True},
                            "required_capabilities": ["potentiostat"],
                            "prior_weight": 0.1,
                        },
                        {
                            "node_id": "alternate",
                            "label": "Alternate synthesis",
                            "parameter_space": [
                                {"name": "temperature", "lower": 300, "upper": 700}
                            ],
                            "protocol_ref": {
                                "protocol_template": {
                                    "steps": [
                                        {"primitive": "robot.dispense", "params": {}}
                                    ]
                                }
                            },
                            "required_capabilities": ["furnace"],
                            "prior_weight": 5.0,
                        },
                    ],
                    "transitions": [
                        {
                            "source_id": "baseline",
                            "target_id": "alternate",
                            "approval_required": False,
                        }
                    ],
                },
                "available_capabilities": ["potentiostat", "furnace"],
                "objectives": ["yield"],
                "objective_directions": ["maximize"],
                "observations": [
                    {
                        "iteration": 1,
                        "parameters": {"voltage": 0.5},
                        "kpi_values": {"yield": 0.1},
                        "qc_passed": False,
                        "is_failure": True,
                        "failure_reason": "baseline failed",
                        "metadata": {"experimental_node_id": "baseline"},
                    }
                ],
            },
        )
    finally:
        sys.path.remove(str(nexus_repo))

    assert response.status_code == 200, response.text
    report = response.json()["report"]
    assert report["authority"] == "advisory_only"
    assert report["contract_version"] == "experimental_route_intelligence.v1"

    decision = select_experimental_route(
        report=report,
        execution_graph={
            **report["graph"],
            "nodes": [
                {
                    **node,
                    "protocol_ref": (
                        {"use_campaign_default": True}
                        if node["node_id"] == "baseline"
                        else {
                            "protocol_template": {
                                "steps": [
                                    {"primitive": "robot.dispense", "params": {}}
                                ]
                            }
                        }
                    ),
                }
                for node in report["graph"]["nodes"]
            ],
        },
        campaign_dimensions=[
            {"param_name": "voltage", "min_value": 0.1, "max_value": 1.0}
        ],
        campaign_protocol_template={"steps": []},
        campaign_protocol_pattern_id="",
        direction="maximize",
        authority_enabled=True,
        available_capabilities=["potentiostat", "furnace"],
        policy_snapshot={},
    )

    assert decision.applied is True
    assert decision.selected_node_id == "alternate"
    assert decision.selected_option is not None
    assert decision.selected_option.runtime is not None
    assert decision.selected_option.runtime.dimensions[0]["param_name"] == "temperature"
