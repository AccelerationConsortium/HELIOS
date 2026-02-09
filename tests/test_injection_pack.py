"""Tests for the injection pack builder — cross-object validation and mapping."""
from __future__ import annotations

import os
import tempfile

_tmpdir = tempfile.mkdtemp(prefix="otbot_pack_test_")
os.environ["DATA_DIR"] = _tmpdir
os.environ["DB_PATH"] = os.path.join(_tmpdir, "pack_test.db")
os.environ["OBJECT_STORE_DIR"] = os.path.join(_tmpdir, "obj")
os.environ["LLM_PROVIDER"] = "mock"

import pytest  # noqa: E402

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.core.db import init_db  # noqa: E402
from app.api.v1.schemas_init import (  # noqa: E402
    DimensionSpec,
    GoalSpec,
    HumanGatePolicySpec,
    InjectionPack,
    InjectionPackMetadata,
    KPIConfigSpec,
    ParamSpaceSpec,
    ProtocolPatternSpec,
    SafetyRulesSpec,
)
from app.services.injection_pack import (  # noqa: E402
    build_diff_summary,
    injection_pack_to_campaign_args,
    injection_pack_to_campaign_goal,
    validate_injection_pack,
)


@pytest.fixture(autouse=True)
def _setup_db():
    get_settings.cache_clear()
    init_db()


def _make_pack(**overrides) -> InjectionPack:
    """Build a valid injection pack with optional overrides."""
    goal = overrides.get("goal", GoalSpec(
        objective_type="oer_screening",
        objective_kpi="overpotential_mv",
        direction="minimize",
        acceptable_range_pct=10.0,
    ))
    protocol = overrides.get("protocol", ProtocolPatternSpec(
        pattern_id="oer_screening",
        mandatory_steps=["synthesis", "deposition", "annealing", "electrochem_test"],
    ))
    param_space = overrides.get("param_space", ParamSpaceSpec(
        dimensions=[
            DimensionSpec(
                param_name="precursor_ratio",
                param_type="number",
                min_value=0.1,
                max_value=10.0,
                optimizable=True,
                step_key="synthesis",
                primitive="robot.aspirate",
            ),
        ],
        strategy="lhs",
        batch_size=10,
    ))
    safety = overrides.get("safety", SafetyRulesSpec(
        max_temp_c=95.0,
        max_volume_ul=1000.0,
        allowed_primitives=[
            "robot.aspirate", "robot.dispense", "robot.home",
            "squidstat.run_experiment", "squidstat.get_data",
            "heat", "wait", "log",
        ],
        hazardous_reagents=["KOH"],
    ))
    kpi_config = overrides.get("kpi_config", KPIConfigSpec(
        primary_kpi="overpotential_mv",
    ))
    human_gate = overrides.get("human_gate", HumanGatePolicySpec(
        max_rounds=20,
        plateau_threshold=0.01,
        auto_approve_magnitude=0.3,
        human_gate_triggers=["safety_boundary_change"],
    ))
    metadata = overrides.get("metadata", InjectionPackMetadata(
        session_id="test-session",
        created_at="2025-01-01T00:00:00Z",
        created_by="tester",
        checksum="abc123",
    ))
    return InjectionPack(
        goal=goal,
        protocol=protocol,
        param_space=param_space,
        safety=safety,
        kpi_config=kpi_config,
        human_gate=human_gate,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Cross-object validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_valid_pack_passes(self):
        pack = _make_pack()
        errors = validate_injection_pack(pack)
        assert len(errors) == 0

    def test_missing_instrument_for_kpi(self):
        """KPI overpotential_mv needs squidstat. Remove squidstat primitives."""
        pack = _make_pack(safety=SafetyRulesSpec(
            max_temp_c=95.0,
            max_volume_ul=1000.0,
            allowed_primitives=["robot.aspirate", "robot.dispense", "heat", "wait", "log"],
            hazardous_reagents=["KOH"],
        ))
        errors = validate_injection_pack(pack)
        assert any("squidstat" in e for e in errors)

    def test_param_temp_exceeds_safety(self):
        """A temperature parameter that exceeds safety max_temp_c."""
        pack = _make_pack(
            param_space=ParamSpaceSpec(
                dimensions=[
                    DimensionSpec(
                        param_name="annealing_temp_c",
                        param_type="number",
                        min_value=100.0,
                        max_value=999.0,  # exceeds safety limit of 95.0
                        optimizable=True,
                    ),
                ],
                strategy="lhs",
                batch_size=10,
            ),
        )
        errors = validate_injection_pack(pack)
        assert any("temp" in e.lower() and "exceed" in e.lower() for e in errors)

    def test_param_volume_exceeds_safety(self):
        pack = _make_pack(
            param_space=ParamSpaceSpec(
                dimensions=[
                    DimensionSpec(
                        param_name="solvent_volume_ul",
                        param_type="number",
                        min_value=50.0,
                        max_value=5000.0,  # exceeds safety limit of 1000.0
                        optimizable=True,
                    ),
                ],
                strategy="lhs",
                batch_size=10,
            ),
        )
        errors = validate_injection_pack(pack)
        assert any("volume" in e.lower() and "exceed" in e.lower() for e in errors)

    def test_unknown_pattern_detected(self):
        pack = _make_pack(protocol=ProtocolPatternSpec(
            pattern_id="nonexistent_pattern",
        ))
        errors = validate_injection_pack(pack)
        assert any("nonexistent_pattern" in e for e in errors)

    def test_budget_limit_less_than_batch_size(self):
        pack = _make_pack(human_gate=HumanGatePolicySpec(
            max_rounds=20,
            budget_limit_runs=5,  # less than batch_size=10
            auto_approve_magnitude=0.3,
            human_gate_triggers=["safety_boundary_change"],
        ))
        errors = validate_injection_pack(pack)
        assert any("budget" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# Diff summary
# ---------------------------------------------------------------------------


class TestDiffSummary:
    def test_diff_includes_non_default_fields(self):
        pack = _make_pack(goal=GoalSpec(
            objective_type="stability_testing",  # not default "oer_screening"
            objective_kpi="stability_decay_pct",
            direction="maximize",  # not default "minimize"
            acceptable_range_pct=10.0,
        ))
        diffs = build_diff_summary(pack)
        field_names = [d["field"] for d in diffs]
        assert "objective_type" in field_names
        assert "direction" in field_names

    def test_diff_always_includes_kpi_and_pattern(self):
        pack = _make_pack()
        diffs = build_diff_summary(pack)
        field_names = [d["field"] for d in diffs]
        assert "objective_kpi" in field_names
        assert "pattern_id" in field_names
        assert "optimizable_dimensions" in field_names


# ---------------------------------------------------------------------------
# Campaign mapping
# ---------------------------------------------------------------------------


class TestCampaignMapping:
    def test_campaign_args_has_required_keys(self):
        pack = _make_pack()
        args = injection_pack_to_campaign_args(pack)
        assert "name" in args
        assert "cadence_seconds" in args
        assert "protocol" in args
        assert "inputs" in args
        assert "policy_snapshot" in args
        assert "actor" in args

    def test_campaign_args_protocol_has_steps(self):
        pack = _make_pack()
        args = injection_pack_to_campaign_args(pack)
        assert "steps" in args["protocol"]
        assert len(args["protocol"]["steps"]) > 0

    def test_campaign_goal_args_correct(self):
        pack = _make_pack()
        goal_args = injection_pack_to_campaign_goal(pack)
        assert goal_args["objective_kpi"] == "overpotential_mv"
        assert goal_args["direction"] == "minimize"
        assert goal_args["max_rounds"] == 20
        assert goal_args["batch_size"] == 10
        assert goal_args["strategy"] == "lhs"

    def test_policy_snapshot_has_safety_fields(self):
        pack = _make_pack()
        args = injection_pack_to_campaign_args(pack)
        policy = args["policy_snapshot"]
        assert "max_temp_c" in policy
        assert "allowed_primitives" in policy
        assert "human_gate_triggers" in policy
