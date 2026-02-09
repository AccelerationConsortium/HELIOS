"""Tests for instrument adapters (SimulatedAdapter and BatteryLabAdapter dry-run)."""
from __future__ import annotations

import pytest

from app.adapters.simulated_instrument import SimulatedAdapter, execute_primitive
from app.adapters.battery_lab import BatteryLabAdapter
from app.adapters.base import InstrumentAdapter


# ---------------------------------------------------------------------------
# SimulatedAdapter
# ---------------------------------------------------------------------------

class TestSimulatedAdapter:
    def test_implements_protocol(self):
        adapter = SimulatedAdapter()
        assert isinstance(adapter, InstrumentAdapter)

    def test_connect_disconnect(self):
        adapter = SimulatedAdapter()
        adapter.connect()
        assert adapter.health_check()["connected"] is True
        adapter.disconnect()
        assert adapter.health_check()["connected"] is False

    def test_execute_aspirate(self):
        adapter = SimulatedAdapter()
        result = adapter.execute_primitive(
            instrument_id="sim-1",
            primitive="aspirate",
            params={"volume_ul": 100, "duration_s": 0.01},
        )
        assert result["ok"] is True
        assert "measured_volume_ul" in result

    def test_execute_heat(self):
        adapter = SimulatedAdapter()
        result = adapter.execute_primitive(
            instrument_id="sim-1",
            primitive="heat",
            params={"temp_c": 60, "duration_s": 0.01},
        )
        assert result["ok"] is True
        assert "measured_temp_c" in result

    def test_execute_wait(self):
        adapter = SimulatedAdapter()
        result = adapter.execute_primitive(
            instrument_id="sim-1",
            primitive="wait",
            params={"duration_s": 0.01},
        )
        assert result["ok"] is True

    def test_execute_battery_lab_primitives(self):
        """All battery-lab primitives should be accepted by the simulated adapter."""
        adapter = SimulatedAdapter()
        battery_primitives = [
            "robot.home", "robot.load_pipettes", "robot.aspirate",
            "plc.dispense_ml", "relay.switch_to", "log",
        ]
        for prim in battery_primitives:
            result = adapter.execute_primitive(
                instrument_id="sim-1",
                primitive=prim,
                params={"duration_s": 0.01},
            )
            assert result["ok"] is True, f"Failed for primitive: {prim}"

    def test_unsupported_primitive_raises(self):
        adapter = SimulatedAdapter()
        with pytest.raises(RuntimeError, match="unsupported"):
            adapter.execute_primitive(
                instrument_id="sim-1",
                primitive="totally_unknown_action",
                params={"duration_s": 0.01},
            )

    def test_force_fail(self):
        adapter = SimulatedAdapter()
        with pytest.raises(RuntimeError, match="forced failure"):
            adapter.execute_primitive(
                instrument_id="sim-1",
                primitive="wait",
                params={"force_fail": True, "duration_s": 0.01},
            )

    def test_backward_compatible_free_function(self):
        """Legacy execute_primitive() function still works."""
        result = execute_primitive(
            instrument_id="sim-1",
            primitive="wait",
            params={"duration_s": 0.01},
        )
        assert result["ok"] is True


# ---------------------------------------------------------------------------
# BatteryLabAdapter (dry-run only — no real hardware)
# ---------------------------------------------------------------------------

class TestBatteryLabAdapterDryRun:
    def test_implements_protocol(self):
        adapter = BatteryLabAdapter(dry_run=True)
        assert isinstance(adapter, InstrumentAdapter)

    def test_connect_disconnect(self):
        adapter = BatteryLabAdapter(dry_run=True)
        adapter.connect()
        health = adapter.health_check()
        assert health["connected"] is True
        assert health["dry_run"] is True

        adapter.disconnect()
        assert adapter.health_check()["connected"] is False

    def test_execute_robot_home(self):
        adapter = BatteryLabAdapter(dry_run=True)
        adapter.connect()
        result = adapter.execute_primitive(
            instrument_id="ot2-1",
            primitive="robot.home",
            params={},
        )
        assert result["ok"] is True

    def test_execute_wait(self):
        adapter = BatteryLabAdapter(dry_run=True)
        adapter.connect()
        result = adapter.execute_primitive(
            instrument_id="ot2-1",
            primitive="wait",
            params={"duration_seconds": 0.5},
        )
        assert result["ok"] is True

    def test_execute_all_known_actions(self):
        """Every action in the dispatcher should be accepted in dry-run mode."""
        adapter = BatteryLabAdapter(dry_run=True)
        adapter.connect()
        known_actions = [
            "robot.home", "robot.load_pipettes", "robot.set_lights",
            "robot.load_labware", "robot.load_custom_labware",
            "robot.move_to_well", "robot.pick_up_tip", "robot.drop_tip",
            "robot.aspirate", "robot.dispense", "robot.blowout",
            "plc.dispense_ml", "plc.set_pump_on_timer", "plc.set_ultrasonic_on_timer",
            "relay.set_channel", "relay.turn_on", "relay.turn_off", "relay.switch_to",
            "squidstat.run_experiment", "squidstat.get_data",
            "squidstat.save_snapshot", "squidstat.reset_plot",
            "cleanup.run_full", "sample.prepare_from_csv",
            "ssh.start_stream", "ssh.stop_stream",
            "wait", "log",
        ]
        for action in known_actions:
            result = adapter.execute_primitive(
                instrument_id="ot2-1",
                primitive=action,
                params={},
            )
            assert result["ok"] is True, f"Failed for action: {action}"

    def test_unknown_action_raises(self):
        adapter = BatteryLabAdapter(dry_run=True)
        adapter.connect()
        with pytest.raises(ValueError, match="Unknown action"):
            adapter.execute_primitive(
                instrument_id="ot2-1",
                primitive="completely.bogus",
                params={},
            )

    def test_not_connected_raises(self):
        adapter = BatteryLabAdapter(dry_run=True)
        with pytest.raises(RuntimeError, match="not connected"):
            adapter.execute_primitive(
                instrument_id="ot2-1",
                primitive="robot.home",
                params={},
            )
