"""
Tests for Safety Runtime (Phase 3).

Tests cover:
1. Hysteresis/timing tests - state oscillation prevention
2. Combined interlock tests - priority ordering
3. Replay consistency tests - deterministic behavior
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

# Configure pytest-asyncio
pytestmark = pytest.mark.asyncio(loop_scope="function")

from exp_agent.sensing.safety_state import (
    SafetyStateMachine,
    SafetyState,
    SafetyStateUpdate,
    InterlockClass,
    InterlockReason,
    RecommendedAction,
    HysteresisConfig,
    EvidenceChain,
)
from exp_agent.sensing.recovery_gate import (
    RecoveryGate,
    RecoveryAction,
    ActionRiskLevel,
    GateDecision,
)
from exp_agent.sensing.protocol.sensor_event import SensorEvent, SensorType
from exp_agent.sensing.protocol.snapshot import (
    SensorSnapshot,
    SystemSnapshot,
)
from exp_agent.sensing.protocol.health_event import HealthStatus
from exp_agent.sensing.drivers.mock_driver import (
    MockSensorDriver,
    MockSensorConfig,
    TemperatureProfile,
    AirflowProfile,
    PressureProfile,
)
from exp_agent.sensing.hub.sensor_hub import SensorHub, HubConfig
from exp_agent.sensing.simulator.fault_injector import FaultInjector, FaultType, FaultConfig


class TestHysteresis:
    """Tests for state transition hysteresis."""

    def test_upward_transition_immediate(self):
        """Upward state transitions should be immediate."""
        machine = SafetyStateMachine()

        # Create snapshot with temperature over limit
        snapshot = self._create_snapshot({
            "temp_1": {"type": SensorType.TEMPERATURE, "value": 150.0},
        })

        update = machine.process_snapshot(snapshot, {"max_temp": 130})

        # Should immediately transition to INTERLOCKED
        assert update.state >= SafetyState.INTERLOCKED

    def test_downward_transition_requires_hysteresis(self):
        """Downward transitions should require hysteresis."""
        config = HysteresisConfig(
            min_hold_time_ms=100,
            recovery_threshold_readings=3,
            safe_recovery_delay_ms=100,
        )
        machine = SafetyStateMachine(hysteresis=config)

        # First, trigger interlock
        bad_snapshot = self._create_snapshot({
            "temp_1": {"type": SensorType.TEMPERATURE, "value": 150.0},
        })
        machine.process_snapshot(bad_snapshot, {"max_temp": 130})
        assert machine.current_state >= SafetyState.INTERLOCKED

        # Now send good readings - should not immediately recover
        good_snapshot = self._create_snapshot({
            "temp_1": {"type": SensorType.TEMPERATURE, "value": 100.0},
        })

        # First good reading
        update1 = machine.process_snapshot(good_snapshot)
        assert machine.current_state >= SafetyState.INTERLOCKED  # Still interlocked

        # Second good reading
        update2 = machine.process_snapshot(good_snapshot)
        assert machine.current_state >= SafetyState.INTERLOCKED  # Still interlocked

    def test_oscillation_prevention(self):
        """Rapidly changing values should not cause state oscillation."""
        config = HysteresisConfig(min_hold_time_ms=1000)
        machine = SafetyStateMachine(hysteresis=config)

        state_changes = []
        machine.set_state_callback(lambda u: state_changes.append(u.state))

        # Simulate rapid value changes around threshold
        for i in range(10):
            value = 125.0 + (10.0 if i % 2 == 0 else -10.0)  # Oscillate around 130
            snapshot = self._create_snapshot({
                "temp_1": {"type": SensorType.TEMPERATURE, "value": value},
            })
            machine.process_snapshot(snapshot, {"max_temp": 130})

        # Should not have many state changes due to hysteresis
        # First transition to interlocked, then held
        assert len(state_changes) <= 2  # At most initial + one change

    def _create_snapshot(self, sensors: dict) -> SystemSnapshot:
        """Helper to create a snapshot from sensor dict."""
        snapshot = SystemSnapshot()
        for sensor_id, config in sensors.items():
            sensor = SensorSnapshot(
                sensor_id=sensor_id,
                sensor_type=config["type"],
                health_status=config.get("health", HealthStatus.HEALTHY),
            )
            sensor.latest_value = config["value"]
            sensor.latest_event = SensorEvent(
                sensor_id=sensor_id,
                sensor_type=config["type"],
                value=config["value"],
            )
            snapshot.sensors[sensor_id] = sensor
        snapshot._update_aggregates()
        return snapshot


class TestCombinedInterlocks:
    """Tests for combined interlock priority."""

    def test_priority_order_emergency_highest(self):
        """EMERGENCY should take priority over lower states."""
        machine = SafetyStateMachine()

        # Create snapshot with multiple issues:
        # - Temperature high (INTERLOCKED)
        # - E-stop triggered (EMERGENCY)
        snapshot = self._create_snapshot({
            "temp_1": {"type": SensorType.TEMPERATURE, "value": 150.0},
            "estop_1": {"type": SensorType.ESTOP, "value": 1.0},
        })

        update = machine.process_snapshot(snapshot, {"max_temp": 130})

        # Should be EMERGENCY (highest)
        assert update.state == SafetyState.EMERGENCY
        # Should have E-stop as reason or MULTIPLE
        assert update.reason in (InterlockReason.ESTOP_TRIGGERED, InterlockReason.MULTIPLE_INTERLOCKS)

    def test_multiple_interlocks_aggregated(self):
        """Multiple interlocks should all be captured."""
        machine = SafetyStateMachine()

        snapshot = self._create_snapshot({
            "temp_1": {"type": SensorType.TEMPERATURE, "value": 150.0},
            "pressure_1": {"type": SensorType.PRESSURE, "value": 250.0},
            "airflow_1": {"type": SensorType.AIRFLOW, "value": 0.2},
        })

        update = machine.process_snapshot(snapshot, {
            "max_temp": 130,
            "max_pressure": 200,
            "min_airflow": 0.3,
        })

        # Should have all three interlocks
        assert len(update.interlocks) >= 3
        reasons = {i.reason for i in update.interlocks}
        assert InterlockReason.TEMPERATURE_HIGH in reasons
        assert InterlockReason.PRESSURE_HIGH in reasons
        assert InterlockReason.HOOD_AIRFLOW_LOW in reasons

    def test_actions_aggregated_by_priority(self):
        """Recommended actions should be aggregated and prioritized."""
        machine = SafetyStateMachine()

        snapshot = self._create_snapshot({
            "temp_1": {"type": SensorType.TEMPERATURE, "value": 150.0},
            "pressure_1": {"type": SensorType.PRESSURE, "value": 250.0},
        })

        update = machine.process_snapshot(snapshot, {
            "max_temp": 130,
            "max_pressure": 200,
        })

        # Should include actions from both interlocks
        assert RecommendedAction.STOP_HEATING in update.recommended_actions
        assert RecommendedAction.EXECUTE_VENT in update.recommended_actions

        # Higher priority actions should come first
        stop_idx = update.recommended_actions.index(RecommendedAction.STOP_HEATING)
        vent_idx = update.recommended_actions.index(RecommendedAction.EXECUTE_VENT)
        # VENT should be higher priority than STOP_HEATING in our ordering
        assert vent_idx < stop_idx

    def _create_snapshot(self, sensors: dict) -> SystemSnapshot:
        """Helper to create a snapshot from sensor dict."""
        snapshot = SystemSnapshot()
        for sensor_id, config in sensors.items():
            sensor = SensorSnapshot(
                sensor_id=sensor_id,
                sensor_type=config["type"],
                health_status=config.get("health", HealthStatus.HEALTHY),
            )
            sensor.latest_value = config["value"]
            sensor.latest_event = SensorEvent(
                sensor_id=sensor_id,
                sensor_type=config["type"],
                value=config["value"],
            )
            snapshot.sensors[sensor_id] = sensor
        snapshot._update_aggregates()
        return snapshot


class TestReplayConsistency:
    """Tests for replay consistency - same inputs should give same outputs."""

    def test_deterministic_state_transitions(self):
        """Same sequence of snapshots should produce same state sequence."""
        # Create a sequence of snapshots
        snapshots = [
            {"temp_1": {"type": SensorType.TEMPERATURE, "value": 100.0}},
            {"temp_1": {"type": SensorType.TEMPERATURE, "value": 120.0}},
            {"temp_1": {"type": SensorType.TEMPERATURE, "value": 140.0}},  # Over threshold
            {"temp_1": {"type": SensorType.TEMPERATURE, "value": 160.0}},  # Further over
            {"temp_1": {"type": SensorType.TEMPERATURE, "value": 120.0}},  # Back down
        ]

        config = {"max_temp": 130}

        # Run twice with same config
        results1 = self._run_sequence(snapshots, config)
        results2 = self._run_sequence(snapshots, config)

        # Should get same state sequence
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1.state == r2.state
            assert r1.reason == r2.reason

    def test_evidence_chain_reproducible(self):
        """Evidence chain should be reproducible from same inputs."""
        snapshot = self._create_snapshot({
            "temp_1": {"type": SensorType.TEMPERATURE, "value": 150.0},
        })

        machine1 = SafetyStateMachine()
        machine2 = SafetyStateMachine()

        update1 = machine1.process_snapshot(snapshot, {"max_temp": 130})
        update2 = machine2.process_snapshot(snapshot, {"max_temp": 130})

        # Snapshot IDs should be identical
        assert update1.evidence.snapshot_id == update2.evidence.snapshot_id

        # Trigger values should be identical
        assert update1.evidence.trigger_values == update2.evidence.trigger_values

    def _run_sequence(self, snapshot_defs: list[dict], config: dict) -> list[SafetyStateUpdate]:
        """Run a sequence of snapshots through a fresh state machine."""
        machine = SafetyStateMachine(hysteresis=HysteresisConfig(
            min_hold_time_ms=0,  # Disable for deterministic testing
            recovery_threshold_readings=1,
            safe_recovery_delay_ms=0,
        ))
        results = []

        for sdef in snapshot_defs:
            snapshot = self._create_snapshot(sdef)
            update = machine.process_snapshot(snapshot, config)
            results.append(update)

        return results

    def _create_snapshot(self, sensors: dict) -> SystemSnapshot:
        """Helper to create a snapshot from sensor dict."""
        snapshot = SystemSnapshot()
        for sensor_id, config in sensors.items():
            sensor = SensorSnapshot(
                sensor_id=sensor_id,
                sensor_type=config["type"],
                health_status=config.get("health", HealthStatus.HEALTHY),
            )
            sensor.latest_value = config["value"]
            sensor.latest_event = SensorEvent(
                sensor_id=sensor_id,
                sensor_type=config["type"],
                value=config["value"],
            )
            snapshot.sensors[sensor_id] = sensor
        snapshot._update_aggregates()
        return snapshot


class TestRecoveryGate:
    """Tests for RecoveryGate - no blind recovery rule."""

    def test_safe_actions_always_allowed(self):
        """Safe actions should always be allowed."""
        gate = RecoveryGate()
        snapshot = SystemSnapshot()  # Empty snapshot

        for action in [RecoveryAction.SAFE_SHUTDOWN, RecoveryAction.ASK_HUMAN, RecoveryAction.WAIT]:
            decision = gate.check_action(action, snapshot)
            assert decision.allowed
            assert decision.reason == "Action is always safe"

    def test_high_risk_blocked_without_sensors(self):
        """High-risk actions should be blocked without sensor data."""
        gate = RecoveryGate()
        snapshot = SystemSnapshot()  # Empty - no sensors

        decision = gate.check_action(RecoveryAction.START_HEAT, snapshot)

        assert not decision.allowed
        assert RecoveryAction.SAFE_SHUTDOWN in decision.alternative_actions
        assert len(decision.sensor_issues) > 0

    def test_high_risk_blocked_with_stale_sensors(self):
        """High-risk actions should be blocked with stale sensor data."""
        gate = RecoveryGate(max_sensor_age_seconds=5.0)

        # Create snapshot with old reading
        snapshot = SystemSnapshot()
        sensor = SensorSnapshot(
            sensor_id="temp_1",
            sensor_type=SensorType.TEMPERATURE,
            health_status=HealthStatus.HEALTHY,
        )
        sensor.latest_value = 100.0
        # Simulate old reading by setting age > 5s
        sensor.latest_event = SensorEvent(
            sensor_id="temp_1",
            sensor_type=SensorType.TEMPERATURE,
            value=100.0,
            ts=datetime.now(timezone.utc) - timedelta(seconds=20),  # 20s old
        )
        snapshot.sensors["temp_1"] = sensor
        snapshot._update_aggregates()

        decision = gate.check_action(RecoveryAction.START_HEAT, snapshot)

        assert not decision.allowed
        assert "no healthy sensor with reading" in decision.sensor_issues[0]

    def test_high_risk_allowed_with_healthy_sensors(self):
        """High-risk actions should be allowed with healthy, recent sensor data."""
        gate = RecoveryGate()

        snapshot = self._create_healthy_snapshot()
        decision = gate.check_action(RecoveryAction.START_HEAT, snapshot)

        assert decision.allowed

    def test_hard_interlock_blocks_all_risky_actions(self):
        """Hard interlocks should block all risky actions."""
        machine = SafetyStateMachine()
        gate = RecoveryGate(state_machine=machine)

        # Create EMERGENCY state
        snapshot = self._create_snapshot({
            "estop_1": {"type": SensorType.ESTOP, "value": 1.0},
        })
        state_update = machine.process_snapshot(snapshot)

        # Any risky action should be blocked
        decision = gate.check_action(RecoveryAction.RETRY, snapshot, state_update)

        assert not decision.allowed
        assert decision.requires_human

    def test_get_allowed_actions_filters_correctly(self):
        """get_allowed_actions should return only allowed actions."""
        gate = RecoveryGate()
        snapshot = SystemSnapshot()  # Empty - no sensors

        allowed = gate.get_allowed_actions(snapshot)

        # Without sensors: ALWAYS_SAFE and LOW risk actions should be allowed
        # HIGH and MEDIUM risk require sensor verification
        for action in allowed:
            risk = gate.ACTION_RISK[action]
            assert risk in (ActionRiskLevel.ALWAYS_SAFE, ActionRiskLevel.LOW), \
                f"Unexpected {action} with risk {risk} allowed without sensors"

    def test_get_safest_action_fallback(self):
        """get_safest_action should fall back to SAFE_SHUTDOWN."""
        gate = RecoveryGate()
        snapshot = SystemSnapshot()  # Empty - no sensors

        # Propose only high-risk actions
        proposed = [RecoveryAction.START_HEAT, RecoveryAction.ADD_REAGENT]
        safest = gate.get_safest_action(proposed, snapshot)

        assert safest == RecoveryAction.SAFE_SHUTDOWN

    def _create_healthy_snapshot(self) -> SystemSnapshot:
        """Create a snapshot with healthy, recent sensors."""
        snapshot = SystemSnapshot()
        now = datetime.now(timezone.utc)

        for sensor_id, stype in [
            ("temp_1", SensorType.TEMPERATURE),
            ("airflow_1", SensorType.AIRFLOW),
            ("pressure_1", SensorType.PRESSURE),
        ]:
            sensor = SensorSnapshot(
                sensor_id=sensor_id,
                sensor_type=stype,
                health_status=HealthStatus.HEALTHY,
            )
            sensor.latest_value = 100.0
            sensor.latest_event = SensorEvent(
                sensor_id=sensor_id,
                sensor_type=stype,
                value=100.0,
                ts=now,
            )
            snapshot.sensors[sensor_id] = sensor

        snapshot._update_aggregates()
        return snapshot

    def _create_snapshot(self, sensors: dict) -> SystemSnapshot:
        """Helper to create a snapshot from sensor dict."""
        snapshot = SystemSnapshot()
        for sensor_id, config in sensors.items():
            sensor = SensorSnapshot(
                sensor_id=sensor_id,
                sensor_type=config["type"],
                health_status=config.get("health", HealthStatus.HEALTHY),
            )
            sensor.latest_value = config["value"]
            sensor.latest_event = SensorEvent(
                sensor_id=sensor_id,
                sensor_type=config["type"],
                value=config["value"],
            )
            snapshot.sensors[sensor_id] = sensor
        snapshot._update_aggregates()
        return snapshot


class TestEvidenceChain:
    """Tests for evidence chain auditability."""

    def test_snapshot_hash_deterministic(self):
        """Snapshot hash should be deterministic."""
        snapshot1 = self._create_snapshot({
            "temp_1": {"type": SensorType.TEMPERATURE, "value": 100.0},
        })
        snapshot2 = self._create_snapshot({
            "temp_1": {"type": SensorType.TEMPERATURE, "value": 100.0},
        })
        # Force same timestamp for test
        snapshot2.ts = snapshot1.ts

        hash1 = EvidenceChain.compute_snapshot_id(snapshot1)
        hash2 = EvidenceChain.compute_snapshot_id(snapshot2)

        assert hash1 == hash2

    def test_different_values_different_hash(self):
        """Different sensor values should produce different hashes."""
        snapshot1 = self._create_snapshot({
            "temp_1": {"type": SensorType.TEMPERATURE, "value": 100.0},
        })
        snapshot2 = self._create_snapshot({
            "temp_1": {"type": SensorType.TEMPERATURE, "value": 101.0},
        })
        snapshot2.ts = snapshot1.ts

        hash1 = EvidenceChain.compute_snapshot_id(snapshot1)
        hash2 = EvidenceChain.compute_snapshot_id(snapshot2)

        assert hash1 != hash2

    def test_evidence_includes_trigger_events(self):
        """Evidence should include trigger event IDs."""
        machine = SafetyStateMachine()

        snapshot = self._create_snapshot({
            "temp_1": {"type": SensorType.TEMPERATURE, "value": 150.0},
        })

        update = machine.process_snapshot(snapshot, {"max_temp": 130})

        assert update.evidence is not None
        assert len(update.evidence.trigger_event_ids) > 0
        assert update.evidence.trigger_values.get("temp_1") == 150.0

    def _create_snapshot(self, sensors: dict) -> SystemSnapshot:
        """Helper to create a snapshot from sensor dict."""
        snapshot = SystemSnapshot()
        for sensor_id, config in sensors.items():
            sensor = SensorSnapshot(
                sensor_id=sensor_id,
                sensor_type=config["type"],
                health_status=config.get("health", HealthStatus.HEALTHY),
            )
            sensor.latest_value = config["value"]
            sensor.latest_event = SensorEvent(
                sensor_id=sensor_id,
                sensor_type=config["type"],
                value=config["value"],
            )
            snapshot.sensors[sensor_id] = sensor
        snapshot._update_aggregates()
        return snapshot


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
