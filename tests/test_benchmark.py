"""Tests for Offline Benchmark Framework (Phase D1: SimLab + LogReplay + Scoreboard).

Covers:
- SimWorld (state tracking, deterministic seed, noise model)
- SimAdapter (primitive simulation: aspirate, heat, eis, tip)
- FaultInjector (fault triggering, probability, stacking)
- LogReplay (replay fidelity, error replay)
- ScenarioPack (all scenarios loadable, no duplicate IDs)
- Scoreboard (metric computation, safety=0 check, edge cases)
- Runner (single scenario E2E, acceptance checking)
- Reporter (JSON + MD output format)
"""
from __future__ import annotations

import json
import os
import tempfile

# Isolate test DB before any app imports.
_tmpdir = tempfile.mkdtemp(prefix="otbot_benchmark_test_")
os.environ["DATA_DIR"] = _tmpdir
os.environ["DB_PATH"] = os.path.join(_tmpdir, "benchmark_test.db")
os.environ["OBJECT_STORE_DIR"] = os.path.join(_tmpdir, "obj")
os.environ["LLM_PROVIDER"] = "mock"

import pytest  # noqa: E402

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.core.db import connection, init_db  # noqa: E402

from benchmarks.fault_injector import (  # noqa: E402
    FAULT_DISCONNECTION,
    FAULT_FILE_MISSING,
    FAULT_LIQUID_INSUFFICIENT,
    FAULT_SENSOR_DRIFT,
    FAULT_TEMP_HYSTERESIS,
    FAULT_TIMEOUT,
    FAULT_TIP_SHORTAGE,
    FaultConfig,
    FaultInjector,
)
from benchmarks.log_replay import (  # noqa: E402
    LogScenario,
    ReplayAdapter,
    ReplayStep,
    make_error_recovery_scenario,
    make_simple_pipetting_scenario,
)
from benchmarks.runner import AcceptanceResult, BenchmarkRunner, ScenarioResult  # noqa: E402
from benchmarks.scenarios import (  # noqa: E402
    AcceptanceCriterion,
    BenchmarkScenario,
    SCENARIO_PACK,
    get_scenarios,
)
from benchmarks.scoreboard import Scoreboard, ScoreboardResult  # noqa: E402
from benchmarks.simlab import SimAdapter, SimWorld, add_noise  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _setup_db():
    """Re-init DB and clear tables for each test."""
    get_settings.cache_clear()
    init_db()
    yield
    with connection() as conn:
        # FK-safe cleanup order
        for table in (
            "evolved_priors", "evolution_proposals", "protocol_templates",
            "run_reviews", "run_kpis", "artifacts", "run_steps",
            "batch_candidates", "batch_requests",
            "memory_semantic", "provenance_events", "approvals",
            "runs", "campaigns",
        ):
            try:
                conn.execute(f"DELETE FROM {table}")
            except Exception:
                pass
        conn.commit()


# ===========================================================================
# TestSimWorld
# ===========================================================================


class TestSimWorld:
    """Tests for SimWorld state tracking and determinism."""

    def test_deterministic_seed(self):
        """Same seed produces identical RNG sequences."""
        w1 = SimWorld(seed=42)
        w2 = SimWorld(seed=42)

        vals1 = [w1.rng.random() for _ in range(10)]
        vals2 = [w2.rng.random() for _ in range(10)]
        assert vals1 == vals2

    def test_different_seeds_differ(self):
        """Different seeds produce different RNG sequences."""
        w1 = SimWorld(seed=42)
        w2 = SimWorld(seed=99)

        vals1 = [w1.rng.random() for _ in range(10)]
        vals2 = [w2.rng.random() for _ in range(10)]
        assert vals1 != vals2

    def test_reset(self):
        """Reset returns world to initial state."""
        w = SimWorld(seed=42)
        w.robot_homed = True
        w.labware["plate1"] = True
        w.step_count = 5

        w.reset()
        assert w.robot_homed is False
        assert w.labware == {}
        assert w.step_count == 0


# ===========================================================================
# TestSimAdapter
# ===========================================================================


class TestSimAdapter:
    """Tests for SimAdapter primitive simulation."""

    def test_robot_home(self):
        """robot.home sets homed state."""
        world = SimWorld(seed=42)
        adapter = SimAdapter(world, noise_pct=0.0)
        result = adapter.execute("robot.home", {})
        assert result["homed"] is True
        assert world.robot_homed is True

    def test_aspirate_with_noise(self):
        """robot.aspirate returns measured volume with noise."""
        world = SimWorld(seed=42)
        adapter = SimAdapter(world, noise_pct=0.02)
        result = adapter.execute("robot.aspirate", {
            "pipette": "left",
            "volume_ul": 100.0,
            "labware": "plate1",
            "well": "A1",
        })
        assert "measured_volume_ul" in result
        measured = result["measured_volume_ul"]
        # Should be close to 100 but not exact (noise)
        assert 90.0 < measured < 110.0

    def test_aspirate_no_noise(self):
        """robot.aspirate without noise returns exact volume."""
        world = SimWorld(seed=42)
        adapter = SimAdapter(world, noise_pct=0.0)
        result = adapter.execute("robot.aspirate", {
            "pipette": "left",
            "volume_ul": 100.0,
        })
        assert result["measured_volume_ul"] == 100.0

    def test_heat_simulation(self):
        """heat primitive returns measured temperature."""
        world = SimWorld(seed=42)
        adapter = SimAdapter(world, noise_pct=0.01)
        result = adapter.execute("heat", {"target_temp_c": 37.0})
        assert "measured_temp_c" in result
        assert 35.0 < result["measured_temp_c"] < 39.0

    def test_eis_simulation(self):
        """squidstat.run_experiment returns impedance."""
        world = SimWorld(seed=42)
        adapter = SimAdapter(world, noise_pct=0.02)
        result = adapter.execute("squidstat.run_experiment", {
            "channel": "0",
            "expected_impedance": 1000.0,
        })
        assert "impedance_ohm" in result
        assert 900.0 < result["impedance_ohm"] < 1100.0

    def test_tip_state_tracking(self):
        """Pick up tip / drop tip updates world state."""
        world = SimWorld(seed=42)
        adapter = SimAdapter(world, noise_pct=0.0)
        adapter.execute("robot.pick_up_tip", {"pipette": "left"})
        assert world.tips["left"] == "on"
        adapter.execute("robot.drop_tip", {"pipette": "left"})
        assert world.tips["left"] == "off"

    def test_step_count_increments(self):
        """Each execute() increments step_count."""
        world = SimWorld(seed=42)
        adapter = SimAdapter(world, noise_pct=0.0)
        assert world.step_count == 0
        adapter.execute("wait", {})
        assert world.step_count == 1
        adapter.execute("log", {})
        assert world.step_count == 2


# ===========================================================================
# TestNoiseModel
# ===========================================================================


class TestNoiseModel:
    """Tests for add_noise function."""

    def test_zero_noise(self):
        """Zero noise_pct returns exact value."""
        import random
        rng = random.Random(42)
        assert add_noise(100.0, 0.0, rng) == 100.0

    def test_noise_within_bounds(self):
        """Noise stays within reasonable bounds over many samples."""
        import random
        rng = random.Random(42)
        values = [add_noise(100.0, 0.02, rng) for _ in range(1000)]
        # 99.7% should be within ±3σ = ±6%
        assert all(85.0 < v < 115.0 for v in values)

    def test_zero_value(self):
        """Zero value with noise returns zero."""
        import random
        rng = random.Random(42)
        assert add_noise(0.0, 0.02, rng) == 0.0


# ===========================================================================
# TestFaultInjector
# ===========================================================================


class TestFaultInjector:
    """Tests for FaultInjector fault triggering."""

    def test_disconnection_fault(self):
        """Disconnection fault raises ConnectionError."""
        import random
        rng = random.Random(42)
        fi = FaultInjector(
            faults=[FaultConfig(
                fault_type=FAULT_DISCONNECTION,
                trigger_primitives=("robot.aspirate",),
                probability=1.0,
            )],
            rng=rng,
        )
        exc = fi.maybe_inject("robot.aspirate", {})
        assert isinstance(exc, ConnectionError)
        assert "disconnected" in str(exc)

    def test_no_trigger_for_wrong_primitive(self):
        """Fault doesn't trigger for non-matching primitive."""
        import random
        rng = random.Random(42)
        fi = FaultInjector(
            faults=[FaultConfig(
                fault_type=FAULT_DISCONNECTION,
                trigger_primitives=("robot.aspirate",),
                probability=1.0,
            )],
            rng=rng,
        )
        exc = fi.maybe_inject("robot.home", {})
        assert exc is None

    def test_probability_zero_never_triggers(self):
        """Probability=0.0 never triggers."""
        import random
        rng = random.Random(42)
        fi = FaultInjector(
            faults=[FaultConfig(
                fault_type=FAULT_TIMEOUT,
                probability=0.0,
            )],
            rng=rng,
        )
        for _ in range(100):
            assert fi.maybe_inject("robot.aspirate", {}) is None

    def test_fault_stacking(self):
        """Multiple faults can be registered; first matching fires."""
        import random
        rng = random.Random(42)
        fi = FaultInjector(
            faults=[
                FaultConfig(
                    fault_type=FAULT_TIP_SHORTAGE,
                    trigger_primitives=("robot.pick_up_tip",),
                    probability=1.0,
                ),
                FaultConfig(
                    fault_type=FAULT_TIMEOUT,
                    trigger_primitives=("robot.pick_up_tip",),
                    probability=1.0,
                ),
            ],
            rng=rng,
        )
        exc = fi.maybe_inject("robot.pick_up_tip", {})
        # First matching fault should fire
        assert isinstance(exc, RuntimeError)
        assert "tips" in str(exc).lower()

    def test_sensor_drift_modifies_world(self):
        """Sensor drift modifies world temps, returns None (no exception)."""
        import random
        rng = random.Random(42)
        world = SimWorld(seed=42)
        world.temps["default"] = 37.0

        fi = FaultInjector(
            faults=[FaultConfig(
                fault_type=FAULT_SENSOR_DRIFT,
                probability=1.0,
                params={"drift_pct": 0.10},
            )],
            rng=rng,
        )
        exc = fi.maybe_inject("squidstat.run_experiment", {}, world)
        assert exc is None  # Not an exception
        assert world.temps["default"] > 37.0  # Drifted up

    def test_injection_log(self):
        """Injection log records triggered faults."""
        import random
        rng = random.Random(42)
        fi = FaultInjector(
            faults=[FaultConfig(
                fault_type=FAULT_FILE_MISSING,
                trigger_primitives=("upload_artifact",),
                probability=1.0,
            )],
            rng=rng,
        )
        fi.maybe_inject("upload_artifact", {"filename": "test.csv"})
        assert len(fi.injection_log) == 1
        assert fi.injection_log[0]["fault_type"] == FAULT_FILE_MISSING


# ===========================================================================
# TestLogReplay
# ===========================================================================


class TestLogReplay:
    """Tests for LogScenario and ReplayAdapter."""

    def test_replay_fidelity(self):
        """ReplayAdapter returns recorded results in order."""
        scenario = LogScenario(
            name="test",
            description="test replay",
            steps=(
                ReplayStep(primitive="wait", params={}, result={"waited": True}),
                ReplayStep(primitive="log", params={}, result={"logged": True}),
            ),
        )
        adapter = ReplayAdapter(scenario)
        r1 = adapter.execute("wait", {})
        assert r1["waited"] is True
        r2 = adapter.execute("log", {})
        assert r2["logged"] is True
        assert adapter.steps_remaining == 0

    def test_replay_error(self):
        """ReplayAdapter raises recorded errors."""
        scenario = LogScenario(
            name="test_error",
            description="test error replay",
            steps=(
                ReplayStep(
                    primitive="robot.aspirate",
                    params={},
                    error="instrument disconnected",
                ),
            ),
        )
        adapter = ReplayAdapter(scenario)
        with pytest.raises(RuntimeError, match="disconnected"):
            adapter.execute("robot.aspirate", {})

    def test_replay_exhausted(self):
        """ReplayAdapter raises IndexError when exhausted."""
        scenario = LogScenario(
            name="test_exhaust",
            description="test exhaustion",
            steps=(
                ReplayStep(primitive="wait", params={}, result={"ok": True}),
            ),
        )
        adapter = ReplayAdapter(scenario)
        adapter.execute("wait", {})
        with pytest.raises(IndexError):
            adapter.execute("wait", {})

    def test_built_in_pipetting_scenario(self):
        """make_simple_pipetting_scenario produces valid scenario."""
        scenario = make_simple_pipetting_scenario(n_wells=2)
        assert scenario.name == "simple_pipetting"
        assert len(scenario.steps) > 0
        adapter = ReplayAdapter(scenario)
        for step in scenario.steps:
            try:
                adapter.execute(step.primitive, step.params)
            except RuntimeError:
                pass  # Expected for error steps


# ===========================================================================
# TestScenarioPack
# ===========================================================================


class TestScenarioPack:
    """Tests for scenario pack integrity."""

    def test_all_scenarios_loadable(self):
        """All registered scenarios can be loaded."""
        scenarios = get_scenarios()
        assert len(scenarios) >= 20  # We registered 30

    def test_no_duplicate_ids(self):
        """No duplicate scenario IDs in the pack."""
        ids = [s.id for s in get_scenarios()]
        assert len(ids) == len(set(ids))

    def test_all_categories_present(self):
        """All expected categories have scenarios."""
        categories = {s.category for s in get_scenarios()}
        assert "c2" in categories
        assert "c3" in categories
        assert "c4" in categories
        assert "c5" in categories
        assert "fault" in categories
        assert "intelligence" in categories

    def test_each_scenario_has_acceptance(self):
        """Every scenario has at least one acceptance criterion."""
        for s in get_scenarios():
            assert len(s.acceptance) >= 1, f"Scenario {s.id} has no acceptance criteria"


# ===========================================================================
# TestScoreboard
# ===========================================================================


class TestScoreboard:
    """Tests for Scoreboard intelligence metrics."""

    def test_empty_scoreboard(self):
        """Empty scoreboard returns zeros."""
        sb = Scoreboard()
        result = sb.compute()
        assert result.goal_success_rate == 0.0
        assert result.safety_violations == 0

    def test_perfect_scoreboard(self):
        """All-succeeded runs give high scores."""
        sb = Scoreboard(kpi_target=0.8)
        for i in range(10):
            sb.record_run(
                run_id=f"run-{i}",
                status="succeeded",
                kpis=[{"kpi_name": "run_success_rate", "kpi_value": 1.0}],
            )
        result = sb.compute()
        assert result.goal_success_rate == 1.0
        assert result.safety_violations == 0
        assert result.all_safe

    def test_safety_violation_tracked(self):
        """Safety violations are accumulated."""
        sb = Scoreboard()
        sb.record_run("r1", "succeeded", safety_violations=1)
        sb.record_run("r2", "succeeded", safety_violations=2)
        result = sb.compute()
        assert result.safety_violations == 3
        assert not result.all_safe

    def test_recovery_rate(self):
        """Recovery rate computed correctly."""
        sb = Scoreboard()
        sb.record_run("r1", "succeeded",
                       recovery_attempts=4, recovery_successes=3)
        result = sb.compute()
        assert result.recovery_rate == 0.75

    def test_stability_constant_kpis(self):
        """Constant KPIs give stability=1.0."""
        sb = Scoreboard()
        for i in range(10):
            sb.record_run(
                f"r{i}", "succeeded",
                kpis=[{"kpi_name": "accuracy", "kpi_value": 95.0}],
            )
        result = sb.compute()
        assert result.stability == 1.0

    def test_to_dict(self):
        """ScoreboardResult.to_dict() returns expected structure."""
        result = ScoreboardResult(
            goal_success_rate=0.85,
            sample_efficiency=0.5,
            safety_violations=0,
            recovery_rate=0.9,
            stability=0.95,
        )
        d = result.to_dict()
        assert d["goal_success_rate"] == 0.85
        assert d["safety_violations"] == 0


# ===========================================================================
# TestRunner
# ===========================================================================


class TestRunner:
    """Tests for BenchmarkRunner execution."""

    def test_run_single_scenario(self):
        """Runner can execute a single simple scenario."""
        scenario = BenchmarkScenario(
            id="test_basic",
            name="Test Basic",
            category="intelligence",
            description="Basic test scenario",
            protocol={
                "name": "test",
                "steps": [
                    {"key": "s0", "primitive": "robot.home", "params": {}},
                    {"key": "s1", "primitive": "wait", "params": {"seconds": 1}},
                ],
            },
            policy={"max_temp_c": 95.0, "max_volume_ul": 1000.0},
            seed=42,
            repeat=3,
            acceptance=[
                AcceptanceCriterion(
                    name="success",
                    metric="goal_success_rate",
                    operator=">=",
                    threshold=0.5,
                ),
            ],
        )
        runner = BenchmarkRunner(scenarios=[scenario], seed=42)
        result = runner.run_scenario(scenario)
        assert isinstance(result, ScenarioResult)
        assert result.scenario_id == "test_basic"
        assert result.duration_s > 0

    def test_acceptance_checking(self):
        """Acceptance criteria are evaluated correctly."""
        scenario = BenchmarkScenario(
            id="test_accept",
            name="Test Accept",
            category="intelligence",
            description="Test acceptance",
            protocol={
                "name": "test",
                "steps": [
                    {"key": "s0", "primitive": "robot.home", "params": {}},
                ],
            },
            seed=42,
            repeat=5,
            acceptance=[
                AcceptanceCriterion(
                    name="goal_check",
                    metric="goal_success_rate",
                    operator=">=",
                    threshold=0.5,
                ),
            ],
        )
        runner = BenchmarkRunner(scenarios=[scenario], seed=42)
        result_dict = {"goal_success_rate": 0.8}
        acceptance = runner._check_acceptance(scenario, result_dict)
        assert acceptance.criteria["goal_check"] is True
        assert acceptance.all_passed

    def test_run_all_report_structure(self):
        """run_all returns BenchmarkReport with correct structure."""
        scenario = BenchmarkScenario(
            id="test_report",
            name="Test Report",
            category="intelligence",
            description="Test report",
            protocol={
                "name": "test",
                "steps": [
                    {"key": "s0", "primitive": "wait", "params": {}},
                ],
            },
            seed=42,
            repeat=1,
            acceptance=[
                AcceptanceCriterion(
                    name="always_pass",
                    metric="goal_success_rate",
                    operator=">=",
                    threshold=0.0,
                ),
            ],
        )
        runner = BenchmarkRunner(scenarios=[scenario], seed=42)
        report = runner.run_all()
        assert report.total == 1
        assert report.duration_s > 0
        d = report.to_dict()
        assert "scenarios" in d
        assert "total_scenarios" in d


# ===========================================================================
# TestReporter
# ===========================================================================


class TestReporter:
    """Tests for Reporter output generation."""

    def test_generates_json_and_md(self):
        """Reporter generates both report.json and report.md."""
        from benchmarks.reporter import Reporter
        from benchmarks.runner import BenchmarkReport

        report = BenchmarkReport(
            timestamp="2025-01-01T00:00:00Z",
            seed=42,
        )
        report.results.append(ScenarioResult(
            scenario_id="test_1",
            passed=True,
            duration_s=1.0,
            metrics={"goal_success_rate": 1.0},
            acceptance=AcceptanceResult(
                criteria={"test": True},
                details={"test": "passed"},
            ),
        ))

        out_dir = os.path.join(_tmpdir, "test_reports")
        reporter = Reporter(output_dir=out_dir)
        json_path, md_path = reporter.generate(report)

        assert os.path.exists(json_path)
        assert os.path.exists(md_path)

        with open(json_path) as f:
            data = json.load(f)
        assert data["seed"] == 42
        assert data["total_scenarios"] == 1

        with open(md_path) as f:
            content = f.read()
        assert "OTbot Benchmark Report" in content
        assert "test_1" in content
