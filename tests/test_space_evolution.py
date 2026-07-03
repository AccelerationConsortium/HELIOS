"""Phase C / C1+C4: space-evolution advisor + group-relative ranking."""
from __future__ import annotations

from app.contracts.task_contract import (
    DimensionDef,
    ExplorationSpace,
    HumanGatePolicy,
    ObjectiveSpec,
    SafetyEnvelope,
    StopCondition,
    TaskContract,
)
from app.services.space_evolution import SpaceEvolutionAdvisor, group_relative_rank


def _contract():
    return TaskContract(
        contract_id="c-1",
        created_at="2026-07-03T00:00:00Z",
        created_by="test",
        objective=ObjectiveSpec(
            objective_type="single", primary_kpi="overpotential", direction="minimize"
        ),
        exploration_space=ExplorationSpace(
            dimensions=[
                DimensionDef(
                    param_name="fe_ratio", param_type="number",
                    min_value=0.0, max_value=1.0,
                ),
            ]
        ),
        stop_conditions=StopCondition(max_rounds=30),
        safety_envelope=SafetyEnvelope(),
        human_gate=HumanGatePolicy(),
        protocol_pattern_id="p-1",
    )


# --- C1: advisor ---------------------------------------------------------


def test_plateau_proposes_boundary_widening():
    proposals = SpaceEvolutionAdvisor().propose({"plateaued": True}, _contract())
    assert len(proposals) == 1
    p = proposals[0]
    assert p.boundary_overlays[0].param_name == "fe_ratio"
    # widened 50% each side of range 1.0
    assert p.boundary_overlays[0].new_min == -0.5
    assert p.boundary_overlays[0].new_max == 1.5


def test_proxy_mismatch_proposes_objective_reframe():
    proposals = SpaceEvolutionAdvisor().propose({"proxy_gap": "high"}, _contract())
    assert len(proposals) == 1
    assert proposals[0].objective_overlay.add_secondary_kpis == [
        "overpotential_robustness"
    ]


def test_both_signals_produce_two_proposals():
    proposals = SpaceEvolutionAdvisor().propose(
        {"plateaued": True, "proxy_mismatch": True}, _contract()
    )
    assert len(proposals) == 2


def test_quiet_fingerprint_proposes_nothing():
    assert SpaceEvolutionAdvisor().propose({}, _contract()) == []


def test_suggested_secondary_kpi_is_used():
    proposals = SpaceEvolutionAdvisor().propose(
        {"proxy_gap": "high", "suggested_secondary_kpi": "faradaic_efficiency"},
        _contract(),
    )
    assert proposals[0].objective_overlay.add_secondary_kpis == ["faradaic_efficiency"]


# --- C4: group-relative ranking -----------------------------------------


def test_ranks_by_reward_with_advantage():
    ranked = group_relative_rank(
        [{"id": "a", "reward": 0.2}, {"id": "b", "reward": 0.8}, {"id": "c", "reward": 0.5}]
    )
    assert [r.id for r in ranked] == ["b", "c", "a"]
    assert [r.rank for r in ranked] == [1, 2, 3]
    # advantage is reward - mean(0.5)
    by_id = {r.id: r.advantage for r in ranked}
    assert by_id["b"] == 0.3
    assert by_id["a"] == -0.3
    assert by_id["c"] == 0.0


def test_empty_group_returns_empty():
    assert group_relative_rank([]) == []
