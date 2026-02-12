"""Tests for the InjectionPack -> TaskContract bridge."""
import pytest
from app.api.v1.schemas_init import (
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
from app.services.contract_bridge import (
    injection_pack_to_task_contract,
    task_contract_to_orchestrator_input,
)


def _make_pack() -> InjectionPack:
    """Create a minimal valid InjectionPack for testing."""
    return InjectionPack(
        goal=GoalSpec(
            objective_type="oer_screening",
            objective_kpi="overpotential_mv",
            direction="minimize",
            target_value=300.0,
            acceptable_range_pct=10.0,
        ),
        protocol=ProtocolPatternSpec(
            pattern_id="oer_screening",
            optional_steps=[],
            mandatory_steps=["synthesis", "deposition", "annealing", "electrochem_test"],
        ),
        param_space=ParamSpaceSpec(
            dimensions=[
                DimensionSpec(
                    param_name="precursor_ratio",
                    param_type="number",
                    min_value=0.1,
                    max_value=10.0,
                    optimizable=True,
                    step_key="synthesis",
                    primitive="robot.aspirate",
                    unit="ratio",
                ),
            ],
            strategy="lhs",
            batch_size=10,
        ),
        safety=SafetyRulesSpec(
            max_temp_c=95.0,
            max_volume_ul=1000.0,
            allowed_primitives=["robot.home", "robot.aspirate", "robot.dispense", "heat", "squidstat.run_experiment"],
            require_human_approval=False,
            hazardous_reagents=["KOH"],
        ),
        kpi_config=KPIConfigSpec(
            primary_kpi="overpotential_mv",
            secondary_kpis=["current_density_ma_cm2"],
            target_value=300.0,
        ),
        human_gate=HumanGatePolicySpec(
            auto_approve_magnitude=0.3,
            human_gate_triggers=["safety_boundary_change"],
            plateau_threshold=0.01,
            max_rounds=20,
            budget_limit_runs=200,
        ),
        metadata=InjectionPackMetadata(
            session_id="test-session-123",
            version="1.0",
            created_at="2024-01-01T00:00:00Z",
            created_by="test_user",
            checksum="abc123",
        ),
    )


class TestInjectionPackToTaskContract:
    def test_basic_conversion(self):
        pack = _make_pack()
        contract = injection_pack_to_task_contract(pack)

        assert contract.contract_id.startswith("tc-")
        assert contract.objective.primary_kpi == "overpotential_mv"
        assert contract.objective.direction == "minimize"
        assert contract.stop_conditions.max_rounds == 20
        assert contract.stop_conditions.max_total_runs == 200
        assert contract.stop_conditions.target_kpi_value == 300.0
        assert contract.safety_envelope.max_temp_c == 95.0
        assert len(contract.exploration_space.dimensions) == 1
        assert contract.exploration_space.dimensions[0].param_name == "precursor_ratio"
        assert contract.protocol_pattern_id == "oer_screening"
        assert contract.source_session_id == "test-session-123"

    def test_preserves_safety(self):
        pack = _make_pack()
        contract = injection_pack_to_task_contract(pack)

        assert "KOH" in contract.safety_envelope.hazardous_reagents
        assert "robot.home" in contract.safety_envelope.allowed_primitives
        assert not contract.safety_envelope.require_human_approval

    def test_preserves_human_gate(self):
        pack = _make_pack()
        contract = injection_pack_to_task_contract(pack)

        assert contract.human_gate.auto_approve_magnitude == 0.3
        assert "safety_boundary_change" in contract.human_gate.triggers


class TestTaskContractToOrchestratorInput:
    def test_round_trip(self):
        pack = _make_pack()
        contract = injection_pack_to_task_contract(pack)
        orch_input = task_contract_to_orchestrator_input(contract)

        assert orch_input["contract_id"] == contract.contract_id
        assert orch_input["objective_kpi"] == "overpotential_mv"
        assert orch_input["direction"] == "minimize"
        assert orch_input["max_rounds"] == 20
        assert orch_input["batch_size"] == 10
        assert len(orch_input["dimensions"]) == 1
        assert orch_input["dimensions"][0]["param_name"] == "precursor_ratio"

    def test_policy_snapshot(self):
        pack = _make_pack()
        contract = injection_pack_to_task_contract(pack)
        orch_input = task_contract_to_orchestrator_input(contract)

        policy = orch_input["policy_snapshot"]
        assert policy["max_temp_c"] == 95.0
        assert policy["max_volume_ul"] == 1000.0
