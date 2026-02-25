"""Tests for app/services/simulation.py — dry-run protocol simulator."""
from __future__ import annotations

import pytest

from app.services.simulation import (
    SimulationResult,
    SimulationViolation,
    _update_tip_state,
    simulate_protocol,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step(key: str, primitive: str, **params: object) -> dict:
    return {"step_key": key, "primitive": primitive, "params": params}


def _tip_seq(*primitives: str) -> list[dict]:
    """Build a minimal protocol with the given primitive sequence."""
    return [{"step_key": f"s{i}", "primitive": p, "params": {}} for i, p in enumerate(primitives)]


# ---------------------------------------------------------------------------
# simulate_protocol — verdict tiers
# ---------------------------------------------------------------------------


class TestVerdict:
    def test_clean_protocol_passes(self) -> None:
        steps = [
            _step("s1", "robot.pick_up_tip"),
            _step("s2", "robot.aspirate", volume=50),
            _step("s3", "robot.dispense", volume=50),
            _step("s4", "robot.drop_tip"),
        ]
        r = simulate_protocol(steps, {}, {"max_volume_ul": 1000})
        assert r.verdict == "pass"
        assert r.errors == []
        assert r.soft_warnings == []

    def test_volume_exceeds_policy_fails(self) -> None:
        steps = [
            _step("s1", "robot.pick_up_tip"),
            _step("s2", "robot.aspirate", volume=2000),
            _step("s3", "robot.drop_tip"),
        ]
        r = simulate_protocol(steps, {}, {"max_volume_ul": 500})
        assert r.verdict == "fail"
        assert len(r.errors) == 1
        assert "2000" in r.errors[0].message

    def test_temp_exceeds_policy_fails(self) -> None:
        steps = [_step("s1", "heat", temperature=150)]
        r = simulate_protocol(steps, {}, {"max_temp_c": 100})
        assert r.verdict == "fail"
        assert any("150" in v.message for v in r.errors)

    def test_current_exceeds_policy_fails(self) -> None:
        steps = [_step("s1", "squidstat.run_experiment", max_current=200)]
        r = simulate_protocol(steps, {}, {"max_current_ma": 100})
        assert r.verdict == "fail"

    def test_general_warning_gives_warn_not_fail(self) -> None:
        # Tip not dropped → general_warning → "warn"
        steps = [_step("s1", "robot.pick_up_tip")]
        r = simulate_protocol(steps, {}, {})
        assert r.verdict == "warn"
        assert any("tip" in w.lower() for w in r.warnings)

    def test_soft_violation_gives_warn(self) -> None:
        # Labware not in deck → warning (not error)
        steps = [_step("s1", "robot.aspirate", labware="missing_plate", well="A1", volume=50)]
        deck = {"slots": {"1": {"labware_name": "src_plate"}}}
        r = simulate_protocol(steps, deck, {})
        assert r.verdict in ("warn", "fail")  # depends on tip state
        soft = r.soft_warnings
        assert any("missing_plate" in v.message for v in soft)


# ---------------------------------------------------------------------------
# Tip lifecycle state machine
# ---------------------------------------------------------------------------


class TestTipLifecycle:
    def test_double_pick_up_is_error(self) -> None:
        steps = _tip_seq("robot.pick_up_tip", "robot.pick_up_tip")
        r = simulate_protocol(steps, {}, {})
        assert r.verdict == "fail"
        assert any("already holding" in v.message for v in r.errors)

    def test_drop_without_tip_is_warning(self) -> None:
        steps = _tip_seq("robot.drop_tip")
        r = simulate_protocol(steps, {}, {})
        assert r.verdict == "warn"
        assert any("without holding" in v.message for v in r.soft_warnings)

    def test_aspirate_without_tip_is_error(self) -> None:
        steps = [_step("s1", "robot.aspirate", volume=50)]
        r = simulate_protocol(steps, {}, {})
        assert r.verdict == "fail"
        assert any("requires a tip" in v.message for v in r.errors)

    def test_tip_not_dropped_at_end_is_general_warning(self) -> None:
        steps = [_step("s1", "robot.pick_up_tip")]
        r = simulate_protocol(steps, {}, {})
        assert any("not dropped" in w for w in r.warnings)

    def test_correct_tip_lifecycle_passes(self) -> None:
        steps = _tip_seq("robot.pick_up_tip", "robot.drop_tip")
        r = simulate_protocol(steps, {}, {})
        assert r.verdict == "pass"
        assert r.resource_summary["tip_picks"] == 1
        assert r.resource_summary["tip_drops"] == 1


# ---------------------------------------------------------------------------
# Volume accounting
# ---------------------------------------------------------------------------


class TestVolumeAccounting:
    def test_aspirate_from_tracked_well_updates_volume(self) -> None:
        steps = [
            _step("s1", "robot.pick_up_tip"),
            _step("s2", "robot.aspirate", labware="plate", well="A1", volume=100),
            _step("s3", "robot.drop_tip"),
        ]
        r = simulate_protocol(steps, {}, {}, initial_volumes={"plate:A1": 300.0})
        assert r.verdict == "pass"
        assert r.resource_summary["well_volumes_after"]["plate:A1"] == pytest.approx(200.0)

    def test_aspirate_underflow_is_warning(self) -> None:
        steps = [
            _step("s1", "robot.pick_up_tip"),
            _step("s2", "robot.aspirate", labware="plate", well="A1", volume=500),
            _step("s3", "robot.drop_tip"),
        ]
        r = simulate_protocol(steps, {}, {}, initial_volumes={"plate:A1": 50.0})
        assert any("aspirating" in v.message for v in r.soft_warnings)

    def test_aspirate_from_untracked_well_passes(self) -> None:
        """Untracked wells are assumed to have infinite volume — no violation."""
        steps = [
            _step("s1", "robot.pick_up_tip"),
            _step("s2", "robot.aspirate", labware="plate", well="A1", volume=500),
            _step("s3", "robot.drop_tip"),
        ]
        r = simulate_protocol(steps, {}, {})
        # No warnings about volume (only check for volume-related warnings)
        vol_warnings = [v for v in r.soft_warnings if "aspirating" in v.message]
        assert vol_warnings == []


# ---------------------------------------------------------------------------
# Duration estimation
# ---------------------------------------------------------------------------


class TestDuration:
    def test_duration_accumulates(self) -> None:
        steps = [
            _step("s1", "robot.pick_up_tip"),   # 2s
            _step("s2", "robot.aspirate", volume=50),  # 3s
            _step("s3", "robot.drop_tip"),       # 2s
        ]
        r = simulate_protocol(steps, {}, {})
        assert r.estimated_duration_s == pytest.approx(7.0)

    def test_wait_uses_actual_time(self) -> None:
        steps = [_step("s1", "wait", wait_s=120)]
        r = simulate_protocol(steps, {}, {})
        assert r.estimated_duration_s == pytest.approx(120.0)

    def test_max_run_duration_key_takes_precedence(self) -> None:
        steps = [_step(f"s{i}", "wait", wait_s=100) for i in range(10)]
        r = simulate_protocol(steps, {}, {"max_run_duration_s": 500})
        # 10 × 100s = 1000s > 500s cap → warning
        assert r.verdict == "warn"
        assert any("exceeds policy cap" in w for w in r.warnings)

    def test_per_step_fallback_duration_cap(self) -> None:
        # Without max_run_duration_s, fallback = 300 * n_steps
        steps = [_step(f"s{i}", "wait", wait_s=100) for i in range(2)]
        r = simulate_protocol(steps, {}, {"max_step_duration_s": 50})
        # 2 × 100s = 200s > 50 × 2 = 100s → warning
        assert r.verdict == "warn"


# ---------------------------------------------------------------------------
# Robustness / edge cases
# ---------------------------------------------------------------------------


class TestRobustness:
    def test_empty_steps_passes(self) -> None:
        r = simulate_protocol([], {}, {})
        assert r.verdict == "pass"
        assert r.estimated_duration_s == 0.0

    def test_params_none_does_not_crash(self) -> None:
        steps = [{"step_key": "bad", "primitive": "robot.aspirate", "params": None}]
        r = simulate_protocol(steps, {}, {})
        # Should not raise; tip-required violation expected
        assert r.verdict == "fail"

    def test_params_missing_does_not_crash(self) -> None:
        steps = [{"step_key": "s1", "primitive": "robot.pick_up_tip"}]
        r = simulate_protocol(steps, {}, {})
        assert r.verdict in ("pass", "warn", "fail")

    def test_unknown_primitive_gets_default_duration(self) -> None:
        steps = [_step("s1", "some.unknown.primitive")]
        r = simulate_protocol(steps, {}, {})
        assert r.estimated_duration_s == pytest.approx(5.0)  # _DEFAULT_STEP_DURATION_S

    def test_no_deck_info_skips_resource_check(self) -> None:
        steps = [_step("s1", "robot.aspirate", labware="ghost", well="A1", volume=10)]
        r = simulate_protocol(steps, {}, {})
        # No deck info → resource check skipped silently
        resource_violations = [v for v in r.violations if "not found in deck" in v.message]
        assert resource_violations == []


# ---------------------------------------------------------------------------
# _update_tip_state unit tests
# ---------------------------------------------------------------------------


class TestUpdateTipState:
    def test_pick_up_sets_has_tip(self) -> None:
        violations: list[SimulationViolation] = []
        has_tip, picks, drops = _update_tip_state(
            "s1", "robot.pick_up_tip", False, 0, 0, violations
        )
        assert has_tip is True
        assert picks == 1
        assert drops == 0
        assert violations == []

    def test_drop_tip_clears_has_tip(self) -> None:
        violations: list[SimulationViolation] = []
        has_tip, picks, drops = _update_tip_state(
            "s1", "robot.drop_tip", True, 1, 0, violations
        )
        assert has_tip is False
        assert drops == 1
        assert violations == []

    def test_double_pick_records_error(self) -> None:
        violations: list[SimulationViolation] = []
        has_tip, picks, drops = _update_tip_state(
            "s1", "robot.pick_up_tip", True, 1, 0, violations
        )
        assert has_tip is True  # still holding
        assert len(violations) == 1
        assert violations[0].severity == "error"

    def test_noop_primitive_unchanged(self) -> None:
        violations: list[SimulationViolation] = []
        has_tip, picks, drops = _update_tip_state(
            "s1", "log", False, 0, 0, violations
        )
        assert has_tip is False
        assert picks == 0
        assert drops == 0
        assert violations == []
