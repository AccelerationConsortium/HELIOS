"""Scenario Pack — 30 benchmark scenarios across 6 categories.

Categories:
- c2: Metrics Store validation (5 scenarios)
- c3: Reviewer validation (5 scenarios)
- c4: Candidate Gen validation (5 scenarios)
- c5: Evolution Engine validation (5 scenarios)
- fault: Fault injection validation (5 scenarios)
- intelligence: Intelligence metric validation (5 scenarios)

Each scenario is a BenchmarkScenario with acceptance criteria.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from benchmarks.fault_injector import (
    FAULT_DECK_CONFLICT,
    FAULT_DISCONNECTION,
    FAULT_FILE_MISSING,
    FAULT_LIQUID_INSUFFICIENT,
    FAULT_SENSOR_DRIFT,
    FAULT_TEMP_HYSTERESIS,
    FAULT_TIMEOUT,
    FAULT_TIP_SHORTAGE,
    FaultConfig,
)


# ---------------------------------------------------------------------------
# Scenario data structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AcceptanceCriterion:
    """A single acceptance criterion for a scenario."""

    name: str
    metric: str       # what to measure
    operator: str     # "==", ">=", "<=", "<", ">", "!="
    threshold: float  # expected value
    description: str = ""


@dataclass(frozen=True)
class BenchmarkScenario:
    """A complete benchmark scenario specification.

    Args:
        id: Unique scenario identifier (e.g. "c2_replay_consistency").
        name: Human-readable name.
        category: One of "c2", "c3", "c4", "c5", "fault", "intelligence".
        description: What this scenario tests.
        protocol: Protocol JSON for the run.
        policy: Policy overrides.
        faults: Fault injection configurations.
        seed: Base random seed.
        repeat: Number of repetitions.
        acceptance: List of acceptance criteria.
        tags: Searchable tags.
    """

    id: str
    name: str
    category: str
    description: str
    protocol: dict[str, Any] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)
    faults: list[FaultConfig] = field(default_factory=list)
    seed: int = 42
    repeat: int = 1
    acceptance: list[AcceptanceCriterion] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

SCENARIO_PACK: dict[str, BenchmarkScenario] = {}


def register(scenario: BenchmarkScenario) -> None:
    """Register a scenario in the global pack."""
    if scenario.id in SCENARIO_PACK:
        raise ValueError(f"duplicate scenario ID: {scenario.id}")
    SCENARIO_PACK[scenario.id] = scenario


def get_scenarios(category: str | None = None) -> list[BenchmarkScenario]:
    """Get scenarios, optionally filtered by category."""
    if category is None:
        return list(SCENARIO_PACK.values())
    return [s for s in SCENARIO_PACK.values() if s.category == category]


def get_scenario(scenario_id: str) -> BenchmarkScenario | None:
    """Get a specific scenario by ID."""
    return SCENARIO_PACK.get(scenario_id)


# ---------------------------------------------------------------------------
# Standard protocol templates for scenarios
# ---------------------------------------------------------------------------

def _simple_protocol(n_wells: int = 4, volume_ul: float = 100.0) -> dict:
    """Generate a simple aspirate/dispense protocol."""
    steps = [
        {"key": "s0", "primitive": "robot.home", "params": {}},
        {"key": "s1", "primitive": "robot.load_pipettes",
         "params": {"pipettes": ["left"]}, "depends_on": ["s0"]},
        {"key": "s2", "primitive": "robot.load_labware",
         "params": {"labware": "plate1", "slot": "1"}, "depends_on": ["s1"]},
    ]
    idx = 3
    for i in range(n_wells):
        well = f"A{i + 1}"
        tip_key = f"s{idx}"
        steps.append({
            "key": tip_key,
            "primitive": "robot.pick_up_tip",
            "params": {"pipette": "left"},
            "depends_on": [f"s{idx - 1}"] if i == 0 else [f"s{idx - 1}"],
        })
        idx += 1

        asp_key = f"s{idx}"
        steps.append({
            "key": asp_key,
            "primitive": "robot.aspirate",
            "params": {"pipette": "left", "volume_ul": volume_ul,
                       "labware": "plate1", "well": well},
            "depends_on": [tip_key],
        })
        idx += 1

        disp_key = f"s{idx}"
        steps.append({
            "key": disp_key,
            "primitive": "robot.dispense",
            "params": {"pipette": "left", "volume_ul": volume_ul,
                       "labware": "plate1", "well": well},
            "depends_on": [asp_key],
        })
        idx += 1

        drop_key = f"s{idx}"
        steps.append({
            "key": drop_key,
            "primitive": "robot.drop_tip",
            "params": {"pipette": "left"},
            "depends_on": [disp_key],
        })
        idx += 1

    return {"name": "simple_pipetting", "steps": steps}


def _heat_protocol(target_c: float = 37.0) -> dict:
    """Generate a heat-then-measure protocol."""
    return {
        "name": "heat_and_measure",
        "steps": [
            {"key": "s0", "primitive": "robot.home", "params": {}},
            {"key": "s1", "primitive": "heat",
             "params": {"target_temp_c": target_c}, "depends_on": ["s0"]},
            {"key": "s2", "primitive": "wait",
             "params": {"seconds": 5.0}, "depends_on": ["s1"]},
            {"key": "s3", "primitive": "squidstat.run_experiment",
             "params": {"channel": "0"}, "depends_on": ["s2"]},
        ],
    }


def _multi_step_protocol() -> dict:
    """Generate a complex multi-step protocol for stress testing."""
    return {
        "name": "multi_step",
        "steps": [
            {"key": "s0", "primitive": "robot.home", "params": {}},
            {"key": "s1", "primitive": "robot.load_pipettes",
             "params": {"pipettes": ["left"]}, "depends_on": ["s0"]},
            {"key": "s2", "primitive": "robot.load_labware",
             "params": {"labware": "plate1", "slot": "1"}, "depends_on": ["s1"]},
            {"key": "s3", "primitive": "robot.load_labware",
             "params": {"labware": "plate2", "slot": "2"}, "depends_on": ["s1"]},
            {"key": "s4", "primitive": "heat",
             "params": {"target_temp_c": 37.0}, "depends_on": ["s2"]},
            {"key": "s5", "primitive": "robot.pick_up_tip",
             "params": {"pipette": "left"}, "depends_on": ["s2"]},
            {"key": "s6", "primitive": "robot.aspirate",
             "params": {"pipette": "left", "volume_ul": 50.0,
                        "labware": "plate1", "well": "A1"},
             "depends_on": ["s5"]},
            {"key": "s7", "primitive": "robot.dispense",
             "params": {"pipette": "left", "volume_ul": 50.0,
                        "labware": "plate2", "well": "A1"},
             "depends_on": ["s6"]},
            {"key": "s8", "primitive": "robot.drop_tip",
             "params": {"pipette": "left"}, "depends_on": ["s7"]},
            {"key": "s9", "primitive": "squidstat.run_experiment",
             "params": {"channel": "0"}, "depends_on": ["s4"]},
            {"key": "s10", "primitive": "wait",
             "params": {"seconds": 2.0}, "depends_on": ["s8", "s9"]},
        ],
    }


# ---------------------------------------------------------------------------
# Default policy
# ---------------------------------------------------------------------------

_DEFAULT_POLICY = {
    "max_temp_c": 95.0,
    "max_volume_ul": 1000.0,
    "require_human_approval": False,
    "recovery_policy": {"enabled": True, "max_attempts_per_step": 2},
}


# ---------------------------------------------------------------------------
# C2: Metrics Store scenarios (5)
# ---------------------------------------------------------------------------

register(BenchmarkScenario(
    id="c2_replay_consistency",
    name="Replay Consistency",
    category="c2",
    description="Same protocol ×10 with same seed → identical KPIs",
    protocol=_simple_protocol(n_wells=4),
    policy=_DEFAULT_POLICY,
    seed=42,
    repeat=10,
    acceptance=[
        AcceptanceCriterion(
            name="kpi_deterministic_stddev",
            metric="kpi_stddev",
            operator="==",
            threshold=0.0,
            description="Deterministic KPIs (step_duration_s) must have stddev=0",
        ),
    ],
    tags=["c2", "consistency", "deterministic"],
))

register(BenchmarkScenario(
    id="c2_version_compat",
    name="Version Compatibility",
    category="c2",
    description="KPI extraction across schema versions — all fields present",
    protocol=_simple_protocol(n_wells=2),
    policy=_DEFAULT_POLICY,
    seed=42,
    repeat=1,
    acceptance=[
        AcceptanceCriterion(
            name="schema_fields_present",
            metric="field_completeness",
            operator=">=",
            threshold=1.0,
            description="All KPI schema v1 fields must be present",
        ),
    ],
    tags=["c2", "schema", "compatibility"],
))

register(BenchmarkScenario(
    id="c2_bad_data_robustness",
    name="Bad Data Robustness",
    category="c2",
    description="Malformed artifacts/missing fields → no crash, graceful skip",
    protocol=_simple_protocol(n_wells=1),
    policy=_DEFAULT_POLICY,
    seed=42,
    repeat=1,
    acceptance=[
        AcceptanceCriterion(
            name="no_crash",
            metric="crash_count",
            operator="==",
            threshold=0.0,
            description="No unhandled exceptions during KPI extraction",
        ),
    ],
    tags=["c2", "robustness", "error-handling"],
))

register(BenchmarkScenario(
    id="c2_latency",
    name="KPI Extraction Latency",
    category="c2",
    description="P95 KPI extraction time < 2s for 20-step protocol",
    protocol=_multi_step_protocol(),
    policy=_DEFAULT_POLICY,
    seed=42,
    repeat=20,
    acceptance=[
        AcceptanceCriterion(
            name="p95_latency",
            metric="extraction_p95_s",
            operator="<",
            threshold=2.0,
            description="P95 extraction latency must be under 2 seconds",
        ),
    ],
    tags=["c2", "performance", "latency"],
))

register(BenchmarkScenario(
    id="c2_cross_run_query",
    name="Cross-Run KPI Query",
    category="c2",
    description="get_kpi_summary() across 50 runs → correct aggregation",
    protocol=_simple_protocol(n_wells=2),
    policy=_DEFAULT_POLICY,
    seed=42,
    repeat=50,
    acceptance=[
        AcceptanceCriterion(
            name="aggregation_correct",
            metric="aggregation_error",
            operator="==",
            threshold=0.0,
            description="Cross-run KPI aggregation must be exact",
        ),
    ],
    tags=["c2", "query", "aggregation"],
))


# ---------------------------------------------------------------------------
# C3: Reviewer scenarios (5)
# ---------------------------------------------------------------------------

register(BenchmarkScenario(
    id="c3_classification_accuracy",
    name="Classification Accuracy",
    category="c3",
    description="20 runs with known failure types → top-1 failure_type ≥ 0.75",
    protocol=_multi_step_protocol(),
    policy=_DEFAULT_POLICY,
    seed=42,
    repeat=20,
    acceptance=[
        AcceptanceCriterion(
            name="top1_accuracy",
            metric="failure_type_accuracy",
            operator=">=",
            threshold=0.75,
            description="Top-1 failure type classification accuracy ≥ 75%",
        ),
    ],
    tags=["c3", "classification", "accuracy"],
))

register(BenchmarkScenario(
    id="c3_suggestion_completeness",
    name="Suggestion Completeness",
    category="c3",
    description="Structured field coverage ≥ 0.95",
    protocol=_multi_step_protocol(),
    policy=_DEFAULT_POLICY,
    seed=42,
    repeat=10,
    acceptance=[
        AcceptanceCriterion(
            name="field_completeness",
            metric="suggestion_completeness",
            operator=">=",
            threshold=0.95,
            description="Review suggestion structured field completeness ≥ 95%",
        ),
    ],
    tags=["c3", "completeness", "structured"],
))

register(BenchmarkScenario(
    id="c3_honesty",
    name="Honesty (Hallucination Check)",
    category="c3",
    description="Hallucination rate < 2% — no fabricated step references",
    protocol=_multi_step_protocol(),
    policy=_DEFAULT_POLICY,
    seed=42,
    repeat=10,
    acceptance=[
        AcceptanceCriterion(
            name="hallucination_rate",
            metric="hallucination_rate",
            operator="<",
            threshold=0.02,
            description="Hallucination rate must be under 2%",
        ),
    ],
    tags=["c3", "honesty", "hallucination"],
))

register(BenchmarkScenario(
    id="c3_score_consistency",
    name="Score Consistency",
    category="c3",
    description="Same run reviewed 5× → score stddev < 5",
    protocol=_simple_protocol(n_wells=4),
    policy=_DEFAULT_POLICY,
    seed=42,
    repeat=5,
    acceptance=[
        AcceptanceCriterion(
            name="score_stddev",
            metric="review_score_stddev",
            operator="<",
            threshold=5.0,
            description="Review score standard deviation < 5 across repeats",
        ),
    ],
    tags=["c3", "consistency", "score"],
))

register(BenchmarkScenario(
    id="c3_degradation_detection",
    name="Degradation Detection",
    category="c3",
    description="Deliberately degraded runs → verdict='failed' or 'degraded'",
    protocol=_simple_protocol(n_wells=2),
    policy=_DEFAULT_POLICY,
    faults=[FaultConfig(
        fault_type=FAULT_TIP_SHORTAGE,
        trigger_primitives=("robot.pick_up_tip",),
        probability=1.0,
    )],
    seed=42,
    repeat=5,
    acceptance=[
        AcceptanceCriterion(
            name="degradation_detected",
            metric="degradation_detection_rate",
            operator=">=",
            threshold=1.0,
            description="All degraded runs must be detected as failed/degraded",
        ),
    ],
    tags=["c3", "degradation", "detection"],
))


# ---------------------------------------------------------------------------
# C4: Candidate Gen scenarios (5)
# ---------------------------------------------------------------------------

register(BenchmarkScenario(
    id="c4_hard_constraints",
    name="Hard Constraint Satisfaction",
    category="c4",
    description="All generated candidates within specified bounds → 0 violations",
    protocol=_simple_protocol(n_wells=2),
    policy=_DEFAULT_POLICY,
    seed=42,
    repeat=1,
    acceptance=[
        AcceptanceCriterion(
            name="zero_violations",
            metric="constraint_violations",
            operator="==",
            threshold=0.0,
            description="Zero out-of-bounds candidates",
        ),
    ],
    tags=["c4", "constraints", "bounds"],
))

register(BenchmarkScenario(
    id="c4_lhs_coverage",
    name="LHS Space Coverage",
    category="c4",
    description="Latin Hypercube coverage metric — min_distance > threshold",
    protocol=_simple_protocol(n_wells=2),
    policy=_DEFAULT_POLICY,
    seed=42,
    repeat=1,
    acceptance=[
        AcceptanceCriterion(
            name="lhs_min_distance",
            metric="lhs_min_distance",
            operator=">",
            threshold=0.0,
            description="LHS samples must have non-zero minimum distance",
        ),
    ],
    tags=["c4", "coverage", "lhs"],
))

register(BenchmarkScenario(
    id="c4_quality_improvement",
    name="Prior-Guided Quality",
    category="c4",
    description="Prior-guided candidates vs random → median KPI ≥ 20% better",
    protocol=_simple_protocol(n_wells=4),
    policy=_DEFAULT_POLICY,
    seed=42,
    repeat=1,
    acceptance=[
        AcceptanceCriterion(
            name="kpi_improvement",
            metric="median_kpi_improvement_pct",
            operator=">=",
            threshold=20.0,
            description="Prior-guided median KPI improvement ≥ 20% over random",
        ),
    ],
    tags=["c4", "quality", "prior-guided"],
))

register(BenchmarkScenario(
    id="c4_dedup",
    name="Duplicate Detection",
    category="c4",
    description="Duplicate candidate rate < 1%",
    protocol=_simple_protocol(n_wells=2),
    policy=_DEFAULT_POLICY,
    seed=42,
    repeat=1,
    acceptance=[
        AcceptanceCriterion(
            name="dedup_rate",
            metric="duplicate_rate",
            operator="<",
            threshold=0.01,
            description="Duplicate candidate rate must be under 1%",
        ),
    ],
    tags=["c4", "dedup", "uniqueness"],
))

register(BenchmarkScenario(
    id="c4_strategy_comparison",
    name="Strategy Comparison",
    category="c4",
    description="All 4 strategies (random, lhs, grid, prior_guided) produce valid output",
    protocol=_simple_protocol(n_wells=2),
    policy=_DEFAULT_POLICY,
    seed=42,
    repeat=1,
    acceptance=[
        AcceptanceCriterion(
            name="all_strategies_valid",
            metric="strategy_error_count",
            operator="==",
            threshold=0.0,
            description="All 4 sampling strategies must produce error-free output",
        ),
    ],
    tags=["c4", "strategies", "validity"],
))


# ---------------------------------------------------------------------------
# C5: Evolution Engine scenarios (5)
# ---------------------------------------------------------------------------

register(BenchmarkScenario(
    id="c5_learning_curve",
    name="Learning Curve (3 conditions)",
    category="c5",
    description="no evolution vs priors-only vs full C5 — full C5 ≥ priors-only ≥ no evolution",
    protocol=_simple_protocol(n_wells=4),
    policy=_DEFAULT_POLICY,
    seed=42,
    repeat=10,
    acceptance=[
        AcceptanceCriterion(
            name="learning_ordering",
            metric="learning_curve_ordering",
            operator="==",
            threshold=1.0,
            description="full C5 goal_success ≥ priors-only ≥ no evolution",
        ),
    ],
    tags=["c5", "learning", "comparison"],
))

register(BenchmarkScenario(
    id="c5_sample_efficiency",
    name="Sample Efficiency Improvement",
    category="c5",
    description="Runs-to-target with evolution < baseline",
    protocol=_simple_protocol(n_wells=4),
    policy=_DEFAULT_POLICY,
    seed=42,
    repeat=20,
    acceptance=[
        AcceptanceCriterion(
            name="efficiency_improvement",
            metric="evolved_vs_baseline_runs",
            operator="<",
            threshold=1.0,
            description="Evolved runs-to-target < baseline runs-to-target (ratio < 1)",
        ),
    ],
    tags=["c5", "efficiency", "evolution"],
))

register(BenchmarkScenario(
    id="c5_stability",
    name="Stability Non-Degradation",
    category="c5",
    description="KPI stddev non-degradation after evolution",
    protocol=_simple_protocol(n_wells=4),
    policy=_DEFAULT_POLICY,
    seed=42,
    repeat=10,
    acceptance=[
        AcceptanceCriterion(
            name="stability_preserved",
            metric="stddev_ratio",
            operator="<=",
            threshold=1.1,
            description="stddev_after ≤ stddev_before × 1.1",
        ),
    ],
    tags=["c5", "stability", "non-degradation"],
))

register(BenchmarkScenario(
    id="c5_template_audit",
    name="Template Audit Trail",
    category="c5",
    description="Template lineage chain traces correctly via parent_template_id",
    protocol=_simple_protocol(n_wells=4),
    policy=_DEFAULT_POLICY,
    seed=42,
    repeat=3,
    acceptance=[
        AcceptanceCriterion(
            name="lineage_valid",
            metric="lineage_integrity",
            operator="==",
            threshold=1.0,
            description="All template parent references resolve correctly",
        ),
    ],
    tags=["c5", "templates", "audit"],
))

register(BenchmarkScenario(
    id="c5_human_gate",
    name="Human Gate Enforcement",
    category="c5",
    description="Large magnitude change → pending status (100% enforcement)",
    protocol=_simple_protocol(n_wells=4),
    policy=_DEFAULT_POLICY,
    seed=42,
    repeat=5,
    acceptance=[
        AcceptanceCriterion(
            name="gate_enforcement",
            metric="human_gate_enforcement_rate",
            operator="==",
            threshold=1.0,
            description="100% of large magnitude changes require human approval",
        ),
    ],
    tags=["c5", "human-gate", "enforcement"],
))


# ---------------------------------------------------------------------------
# Fault Injection scenarios (5)
# ---------------------------------------------------------------------------

register(BenchmarkScenario(
    id="fault_disconnection",
    name="Disconnection Recovery",
    category="fault",
    description="Recovery from instrument disconnection",
    protocol=_simple_protocol(n_wells=2),
    policy=_DEFAULT_POLICY,
    faults=[FaultConfig(
        fault_type=FAULT_DISCONNECTION,
        trigger_primitives=("robot.aspirate",),
        probability=0.5,
    )],
    seed=42,
    repeat=5,
    acceptance=[
        AcceptanceCriterion(
            name="recovery_attempted",
            metric="recovery_rate",
            operator=">",
            threshold=0.0,
            description="At least one recovery attempt made",
        ),
    ],
    tags=["fault", "disconnection", "recovery"],
))

register(BenchmarkScenario(
    id="fault_timeout",
    name="Timeout Handling",
    category="fault",
    description="Timeout handling — no hang, step marked failed",
    protocol=_simple_protocol(n_wells=2),
    policy=_DEFAULT_POLICY,
    faults=[FaultConfig(
        fault_type=FAULT_TIMEOUT,
        trigger_primitives=("robot.aspirate",),
        probability=0.5,
        params={"timeout_s": 5},
    )],
    seed=42,
    repeat=5,
    acceptance=[
        AcceptanceCriterion(
            name="no_hang",
            metric="hung_steps",
            operator="==",
            threshold=0.0,
            description="No steps should hang indefinitely",
        ),
    ],
    tags=["fault", "timeout", "handling"],
))

register(BenchmarkScenario(
    id="fault_tip_shortage",
    name="Tip Shortage Recovery",
    category="fault",
    description="Tip shortage → recovery attempted",
    protocol=_simple_protocol(n_wells=4),
    policy=_DEFAULT_POLICY,
    faults=[FaultConfig(
        fault_type=FAULT_TIP_SHORTAGE,
        trigger_primitives=("robot.pick_up_tip",),
        probability=0.3,
    )],
    seed=42,
    repeat=5,
    acceptance=[
        AcceptanceCriterion(
            name="recovery_attempted",
            metric="recovery_attempted",
            operator=">=",
            threshold=1.0,
            description="Recovery must be attempted for tip shortage",
        ),
    ],
    tags=["fault", "tip-shortage", "recovery"],
))

register(BenchmarkScenario(
    id="fault_liquid_insufficient",
    name="Liquid Insufficient Detection",
    category="fault",
    description="Volume check → error detected before damage",
    protocol=_simple_protocol(n_wells=2),
    policy=_DEFAULT_POLICY,
    faults=[FaultConfig(
        fault_type=FAULT_LIQUID_INSUFFICIENT,
        trigger_primitives=("robot.aspirate",),
        probability=0.5,
    )],
    seed=42,
    repeat=5,
    acceptance=[
        AcceptanceCriterion(
            name="error_detected",
            metric="errors_detected",
            operator=">=",
            threshold=1.0,
            description="Liquid insufficient error must be detected",
        ),
    ],
    tags=["fault", "liquid", "detection"],
))

register(BenchmarkScenario(
    id="fault_multi_fault",
    name="Multi-Fault Resilience",
    category="fault",
    description="2+ faults in one run → run completes with correct status",
    protocol=_multi_step_protocol(),
    policy=_DEFAULT_POLICY,
    faults=[
        FaultConfig(
            fault_type=FAULT_TIMEOUT,
            trigger_primitives=("robot.aspirate",),
            probability=0.3,
            params={"timeout_s": 5},
        ),
        FaultConfig(
            fault_type=FAULT_SENSOR_DRIFT,
            trigger_primitives=("squidstat.run_experiment",),
            probability=0.5,
            params={"drift_pct": 0.10},
        ),
    ],
    seed=42,
    repeat=5,
    acceptance=[
        AcceptanceCriterion(
            name="run_completes",
            metric="completion_rate",
            operator=">=",
            threshold=0.8,
            description="≥80% of runs complete (not hang/crash)",
        ),
    ],
    tags=["fault", "multi-fault", "resilience"],
))


# ---------------------------------------------------------------------------
# Intelligence Metric scenarios (5)
# ---------------------------------------------------------------------------

register(BenchmarkScenario(
    id="intel_goal_success",
    name="Goal Success Rate",
    category="intelligence",
    description="Multi-round campaign → track goal success rate",
    protocol=_simple_protocol(n_wells=4),
    policy=_DEFAULT_POLICY,
    seed=42,
    repeat=20,
    acceptance=[
        AcceptanceCriterion(
            name="goal_success",
            metric="goal_success_rate",
            operator=">=",
            threshold=0.5,
            description="Goal success rate ≥ 50%",
        ),
    ],
    tags=["intelligence", "goal-success"],
))

register(BenchmarkScenario(
    id="intel_sample_efficiency",
    name="Sample Efficiency",
    category="intelligence",
    description="How fast agent reaches target KPI value",
    protocol=_simple_protocol(n_wells=4),
    policy=_DEFAULT_POLICY,
    seed=42,
    repeat=20,
    acceptance=[
        AcceptanceCriterion(
            name="sample_efficiency",
            metric="sample_efficiency",
            operator=">",
            threshold=0.0,
            description="Sample efficiency must be positive (target reached)",
        ),
    ],
    tags=["intelligence", "sample-efficiency"],
))

register(BenchmarkScenario(
    id="intel_safety",
    name="Safety Violations",
    category="intelligence",
    description="Safety violations count MUST be 0 across all runs",
    protocol=_multi_step_protocol(),
    policy=_DEFAULT_POLICY,
    seed=42,
    repeat=10,
    acceptance=[
        AcceptanceCriterion(
            name="zero_safety_violations",
            metric="safety_violations",
            operator="==",
            threshold=0.0,
            description="Zero safety violations across all benchmark runs",
        ),
    ],
    tags=["intelligence", "safety"],
))

register(BenchmarkScenario(
    id="intel_recovery",
    name="Recovery Rate",
    category="intelligence",
    description="Recovery success rate from injected faults",
    protocol=_simple_protocol(n_wells=4),
    policy=_DEFAULT_POLICY,
    faults=[FaultConfig(
        fault_type=FAULT_DISCONNECTION,
        trigger_primitives=("robot.aspirate",),
        probability=0.3,
    )],
    seed=42,
    repeat=10,
    acceptance=[
        AcceptanceCriterion(
            name="recovery_rate",
            metric="recovery_rate",
            operator=">=",
            threshold=0.5,
            description="Recovery success rate ≥ 50%",
        ),
    ],
    tags=["intelligence", "recovery"],
))

register(BenchmarkScenario(
    id="intel_stability",
    name="Stability",
    category="intelligence",
    description="KPI stability across repeated runs (low coefficient of variation)",
    protocol=_simple_protocol(n_wells=4),
    policy=_DEFAULT_POLICY,
    seed=42,
    repeat=20,
    acceptance=[
        AcceptanceCriterion(
            name="stability",
            metric="stability",
            operator=">=",
            threshold=0.8,
            description="Stability metric ≥ 0.8 (low variance across repeats)",
        ),
    ],
    tags=["intelligence", "stability"],
))
