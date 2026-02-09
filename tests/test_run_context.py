"""Tests for app.services.run_context — Per-run state tracker."""
from __future__ import annotations

import pytest

from app.services.run_context import RunContext


# ---------------------------------------------------------------------------
# RunContext — initialization
# ---------------------------------------------------------------------------


class TestRunContextInit:
    def test_defaults(self) -> None:
        ctx = RunContext()
        assert ctx.labware_loaded == {}
        assert ctx.tip_state == {}
        assert ctx.pipette_volume == {}
        assert ctx.pipettes_loaded is False
        assert ctx.robot_homed is False
        assert ctx.experiment_running == {}
        assert ctx.ssh_streaming is False
        assert ctx.active_relay_channel is None
        assert ctx.well_volume == {}

    def test_custom_init(self) -> None:
        ctx = RunContext(
            labware_loaded={"plate1": True},
            pipettes_loaded=True,
            robot_homed=True,
        )
        assert ctx.labware_loaded["plate1"] is True
        assert ctx.pipettes_loaded is True


# ---------------------------------------------------------------------------
# Precondition evaluation
# ---------------------------------------------------------------------------


class TestCheckPrecondition:
    def test_labware_loaded_true(self) -> None:
        ctx = RunContext(labware_loaded={"plate1": True})
        assert ctx.check_precondition("labware_loaded:plate1") is True

    def test_labware_loaded_false(self) -> None:
        ctx = RunContext()
        assert ctx.check_precondition("labware_loaded:plate1") is False

    def test_tip_on_true(self) -> None:
        ctx = RunContext(tip_state={"left": "on"})
        assert ctx.check_precondition("tip_on:left") is True

    def test_tip_on_false(self) -> None:
        ctx = RunContext(tip_state={"left": "off"})
        assert ctx.check_precondition("tip_on:left") is False

    def test_tip_on_missing(self) -> None:
        ctx = RunContext()
        assert ctx.check_precondition("tip_on:left") is False

    def test_tip_off_true(self) -> None:
        ctx = RunContext(tip_state={"left": "off"})
        assert ctx.check_precondition("tip_off:left") is True

    def test_tip_off_when_on(self) -> None:
        ctx = RunContext(tip_state={"left": "on"})
        assert ctx.check_precondition("tip_off:left") is False

    def test_tip_off_missing_defaults_true(self) -> None:
        ctx = RunContext()
        assert ctx.check_precondition("tip_off:left") is True

    def test_pipettes_loaded(self) -> None:
        ctx = RunContext(pipettes_loaded=True)
        assert ctx.check_precondition("pipettes_loaded") is True

    def test_pipettes_not_loaded(self) -> None:
        ctx = RunContext()
        assert ctx.check_precondition("pipettes_loaded") is False

    def test_robot_homed(self) -> None:
        ctx = RunContext(robot_homed=True)
        assert ctx.check_precondition("robot_homed") is True

    def test_robot_not_homed(self) -> None:
        ctx = RunContext()
        assert ctx.check_precondition("robot_homed") is False

    def test_experiment_idle_true(self) -> None:
        ctx = RunContext(experiment_running={"0": False})
        assert ctx.check_precondition("experiment_idle:0") is True

    def test_experiment_idle_default(self) -> None:
        ctx = RunContext()
        assert ctx.check_precondition("experiment_idle:0") is True

    def test_experiment_busy(self) -> None:
        ctx = RunContext(experiment_running={"0": True})
        assert ctx.check_precondition("experiment_idle:0") is False

    def test_ssh_streaming(self) -> None:
        ctx = RunContext(ssh_streaming=True)
        assert ctx.check_precondition("ssh_streaming") is True

    def test_unknown_predicate(self) -> None:
        ctx = RunContext()
        assert ctx.check_precondition("completely_unknown") is False


# ---------------------------------------------------------------------------
# Effect application — set operations
# ---------------------------------------------------------------------------


class TestApplyEffectSet:
    def test_set_robot_homed(self) -> None:
        ctx = RunContext()
        ctx.apply_effect("set:robot_homed:true")
        assert ctx.robot_homed is True

    def test_set_robot_not_homed(self) -> None:
        ctx = RunContext(robot_homed=True)
        ctx.apply_effect("set:robot_homed:false")
        assert ctx.robot_homed is False

    def test_set_pipettes_loaded(self) -> None:
        ctx = RunContext()
        ctx.apply_effect("set:pipettes_loaded:true")
        assert ctx.pipettes_loaded is True

    def test_set_ssh_streaming(self) -> None:
        ctx = RunContext()
        ctx.apply_effect("set:ssh_streaming:true")
        assert ctx.ssh_streaming is True

    def test_set_active_relay_channel(self) -> None:
        ctx = RunContext()
        ctx.apply_effect("set:active_relay_channel:3")
        assert ctx.active_relay_channel == 3

    def test_set_active_relay_channel_none(self) -> None:
        ctx = RunContext(active_relay_channel=3)
        ctx.apply_effect("set:active_relay_channel:none")
        assert ctx.active_relay_channel is None

    def test_set_labware_loaded(self) -> None:
        ctx = RunContext()
        ctx.apply_effect("set:labware_loaded:plate1:true")
        assert ctx.labware_loaded["plate1"] is True

    def test_set_labware_unloaded(self) -> None:
        ctx = RunContext(labware_loaded={"plate1": True})
        ctx.apply_effect("set:labware_loaded:plate1:false")
        assert ctx.labware_loaded["plate1"] is False

    def test_set_tip_state(self) -> None:
        ctx = RunContext()
        ctx.apply_effect("set:tip_state:left:on")
        assert ctx.tip_state["left"] == "on"

    def test_set_tip_on_shorthand(self) -> None:
        ctx = RunContext()
        ctx.apply_effect("set:tip_on:left:true")
        assert ctx.tip_state["left"] == "on"

    def test_set_tip_on_false_shorthand(self) -> None:
        ctx = RunContext(tip_state={"left": "on"})
        ctx.apply_effect("set:tip_on:left:false")
        assert ctx.tip_state["left"] == "off"

    def test_set_pipette_volume(self) -> None:
        ctx = RunContext()
        ctx.apply_effect("set:pipette_volume:left:0")
        assert ctx.pipette_volume["left"] == 0.0

    def test_set_experiment_running(self) -> None:
        ctx = RunContext()
        ctx.apply_effect("set:experiment_running:0:true")
        assert ctx.experiment_running["0"] is True

    def test_set_well_volume(self) -> None:
        ctx = RunContext()
        ctx.apply_effect("set:well_volume:plate1:A1:200")
        assert ctx.well_volume["plate1"]["A1"] == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# Effect application — numeric operations
# ---------------------------------------------------------------------------


class TestApplyEffectNumeric:
    def test_increase_pipette_volume(self) -> None:
        ctx = RunContext(pipette_volume={"left": 50.0})
        ctx.apply_effect("increase:pipette_volume:left:100")
        assert ctx.pipette_volume["left"] == pytest.approx(150.0)

    def test_decrease_pipette_volume(self) -> None:
        ctx = RunContext(pipette_volume={"left": 150.0})
        ctx.apply_effect("decrease:pipette_volume:left:50")
        assert ctx.pipette_volume["left"] == pytest.approx(100.0)

    def test_increase_from_zero(self) -> None:
        ctx = RunContext()
        ctx.apply_effect("increase:pipette_volume:left:75")
        assert ctx.pipette_volume["left"] == pytest.approx(75.0)

    def test_decrease_below_zero(self) -> None:
        ctx = RunContext(pipette_volume={"left": 10.0})
        ctx.apply_effect("decrease:pipette_volume:left:20")
        assert ctx.pipette_volume["left"] == pytest.approx(-10.0)

    def test_increase_well_volume(self) -> None:
        ctx = RunContext()
        ctx.apply_effect("increase:well_volume:plate1:A1:100")
        assert ctx.well_volume["plate1"]["A1"] == pytest.approx(100.0)

    def test_decrease_well_volume(self) -> None:
        ctx = RunContext(well_volume={"plate1": {"A1": 200.0}})
        ctx.apply_effect("decrease:well_volume:plate1:A1:50")
        assert ctx.well_volume["plate1"]["A1"] == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestApplyEffectEdgeCases:
    def test_too_short_noop(self) -> None:
        ctx = RunContext()
        ctx.apply_effect("set:x")  # len < 3 → no-op
        # Should not crash

    def test_unknown_op_noop(self) -> None:
        ctx = RunContext()
        ctx.apply_effect("multiply:x:y:5")
        # Should not crash

    def test_unknown_key_noop(self) -> None:
        ctx = RunContext()
        ctx.apply_effect("set:unknown_field:value")
        # No crash; unknown key path does nothing harmful


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_empty_snapshot(self) -> None:
        ctx = RunContext()
        snap = ctx.snapshot()
        assert snap["labware_loaded"] == {}
        assert snap["pipettes_loaded"] is False
        assert snap["robot_homed"] is False
        assert snap["active_relay_channel"] is None

    def test_populated_snapshot(self) -> None:
        ctx = RunContext(
            labware_loaded={"plate1": True, "tiprack1": True},
            tip_state={"left": "on"},
            pipette_volume={"left": 100.0},
            pipettes_loaded=True,
            robot_homed=True,
            active_relay_channel=2,
        )
        snap = ctx.snapshot()
        assert snap["labware_loaded"] == {"plate1": True, "tiprack1": True}
        assert snap["tip_state"] == {"left": "on"}
        assert snap["pipette_volume"] == {"left": 100.0}
        assert snap["pipettes_loaded"] is True
        assert snap["robot_homed"] is True
        assert snap["active_relay_channel"] == 2

    def test_snapshot_is_copy(self) -> None:
        ctx = RunContext(labware_loaded={"plate1": True})
        snap = ctx.snapshot()
        snap["labware_loaded"]["plate1"] = False
        # Original should be unchanged
        assert ctx.labware_loaded["plate1"] is True

    def test_snapshot_well_volume(self) -> None:
        ctx = RunContext(well_volume={"plate1": {"A1": 100.0, "A2": 50.0}})
        snap = ctx.snapshot()
        assert snap["well_volume"]["plate1"]["A1"] == 100.0
        assert snap["well_volume"]["plate1"]["A2"] == 50.0


# ---------------------------------------------------------------------------
# Integration: Precondition + Effect chain
# ---------------------------------------------------------------------------


class TestPreconditionEffectChain:
    """Simulate a 3-step protocol: home → load_labware → pick_up_tip → aspirate."""

    def test_three_step_protocol(self) -> None:
        ctx = RunContext()

        # Step 0: home
        assert ctx.check_precondition("robot_homed") is False
        ctx.apply_effect("set:robot_homed:true")
        assert ctx.check_precondition("robot_homed") is True

        # Step 1: load_pipettes
        ctx.apply_effect("set:pipettes_loaded:true")
        assert ctx.check_precondition("pipettes_loaded") is True

        # Step 2: load_labware
        ctx.apply_effect("set:labware_loaded:corning_96:true")
        assert ctx.check_precondition("labware_loaded:corning_96") is True

        # Step 3: pick_up_tip (precondition: tip_off:left)
        assert ctx.check_precondition("tip_off:left") is True
        ctx.apply_effect("set:tip_state:left:on")
        assert ctx.check_precondition("tip_on:left") is True

        # Step 4: aspirate (preconditions: labware_loaded, tip_on, pipettes_loaded)
        assert ctx.check_precondition("labware_loaded:corning_96") is True
        assert ctx.check_precondition("tip_on:left") is True
        assert ctx.check_precondition("pipettes_loaded") is True
        ctx.apply_effect("increase:pipette_volume:left:100")
        assert ctx.pipette_volume["left"] == pytest.approx(100.0)

        # Step 5: dispense (preconditions same as aspirate)
        ctx.apply_effect("decrease:pipette_volume:left:100")
        assert ctx.pipette_volume["left"] == pytest.approx(0.0)

        # Step 6: drop_tip
        assert ctx.check_precondition("tip_on:left") is True
        ctx.apply_effect("set:tip_state:left:off")
        ctx.apply_effect("set:pipette_volume:left:0")
        assert ctx.check_precondition("tip_off:left") is True

    def test_relay_workflow(self) -> None:
        ctx = RunContext()
        assert ctx.active_relay_channel is None

        ctx.apply_effect("set:active_relay_channel:3")
        assert ctx.active_relay_channel == 3

        ctx.apply_effect("set:active_relay_channel:5")
        assert ctx.active_relay_channel == 5

        ctx.apply_effect("set:active_relay_channel:none")
        assert ctx.active_relay_channel is None

    def test_experiment_lifecycle(self) -> None:
        ctx = RunContext()

        # Pre-experiment: idle
        assert ctx.check_precondition("experiment_idle:0") is True

        # Start experiment
        ctx.apply_effect("set:experiment_running:0:true")
        assert ctx.check_precondition("experiment_idle:0") is False

        # Finish experiment
        ctx.apply_effect("set:experiment_running:0:false")
        assert ctx.check_precondition("experiment_idle:0") is True


# ---------------------------------------------------------------------------
# Cross-module integration: Precondition.evaluate with RunContext
# ---------------------------------------------------------------------------


class TestPreconditionWithRunContext:
    """Test that Precondition.evaluate() works with RunContext (not just dicts)."""

    def test_evaluate_with_run_context(self) -> None:
        from app.services.action_contracts import Precondition

        ctx = RunContext(labware_loaded={"plate1": True})
        p = Precondition(predicate="labware_loaded:{labware}")
        assert p.evaluate(ctx, {"labware": "plate1"}) is True

    def test_evaluate_with_run_context_fails(self) -> None:
        from app.services.action_contracts import Precondition

        ctx = RunContext()
        p = Precondition(predicate="labware_loaded:{labware}")
        assert p.evaluate(ctx, {"labware": "plate1"}) is False


# ---------------------------------------------------------------------------
# Cross-module integration: Effect.apply with RunContext
# ---------------------------------------------------------------------------


class TestEffectWithRunContext:
    """Test that Effect.apply() works with RunContext (not just dicts)."""

    def test_apply_with_run_context(self) -> None:
        from app.services.action_contracts import Effect

        ctx = RunContext()
        e = Effect(operation="set:robot_homed:true")
        e.apply(ctx, {})
        assert ctx.robot_homed is True

    def test_apply_increase_with_run_context(self) -> None:
        from app.services.action_contracts import Effect

        ctx = RunContext(pipette_volume={"left": 0.0})
        e = Effect(operation="increase:pipette_volume:{pipette}:{volume}")
        e.apply(ctx, {"pipette": "left", "volume": "100"})
        assert ctx.pipette_volume["left"] == pytest.approx(100.0)
