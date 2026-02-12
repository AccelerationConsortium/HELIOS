"""
Tests for RecoveryAgent with policy-driven decisions.
"""
from exp_agent.core.types import DeviceState, HardwareError, Action
from exp_agent.recovery.recovery_agent import RecoveryAgent


def test_recovery_decision_overshoot_with_context():
    """Overshoot with drift signature and target -> DEGRADE."""
    recovery = RecoveryAgent()

    # Provide state with telemetry
    state = DeviceState(
        name="heater",
        status="error",
        telemetry={"temperature": 128.0, "target": 120.0}
    )
    error = HardwareError(
        device="heater",
        type="overshoot",
        severity="high",
        message="Too hot"
    )

    # Provide history showing drift pattern
    history = [
        DeviceState(name="heater", status="running", telemetry={"temperature": 118.0}),
        DeviceState(name="heater", status="running", telemetry={"temperature": 122.0}),
        DeviceState(name="heater", status="running", telemetry={"temperature": 125.0}),
        DeviceState(name="heater", status="running", telemetry={"temperature": 128.0}),
    ]

    # Provide the action that was being executed
    last_action = Action(
        name="set_temperature",
        effect="write",
        params={"temperature": 120.0}
    )

    decision = recovery.decide(state, error, history=history, last_action=last_action)

    assert decision.kind == "degrade"
    assert len(decision.actions) >= 1
    assert decision.actions[0].name == "cool_down"


def test_recovery_decision_overshoot_no_context():
    """Overshoot without context -> ABORT (safe default)."""
    recovery = RecoveryAgent()
    state = DeviceState(name="heater", status="error")
    error = HardwareError(
        device="heater",
        type="overshoot",
        severity="high",
        message="Too hot"
    )

    # Without history or target, the safer choice is abort
    decision = recovery.decide(state, error)

    assert decision.kind == "abort"
    assert any(a.name == "cool_down" for a in decision.actions)


def test_recovery_decision_sensor_fail():
    """Sensor failure always aborts."""
    recovery = RecoveryAgent()
    state = DeviceState(name="heater", status="error")
    error = HardwareError(
        device="heater",
        type="sensor_fail",
        severity="high",
        message="Sensor died"
    )

    decision = recovery.decide(state, error)

    assert decision.kind == "abort"


def test_recovery_agent_retry_counting():
    """RecoveryAgent tracks retry counts."""
    recovery = RecoveryAgent()
    state = DeviceState(name="heater", status="running", telemetry={"temperature": 100.0})
    error = HardwareError(
        device="heater",
        type="timeout",
        severity="medium",
        message="Timeout"
    )

    # First call increments retry count
    recovery.decide(state, error)
    assert recovery.get_retry_count("timeout") == 1

    # Second call increments again
    recovery.decide(state, error)
    assert recovery.get_retry_count("timeout") == 2

    # Reset clears all counts
    recovery.reset_retry_counts()
    assert recovery.get_retry_count("timeout") == 0


def test_recovery_agent_signature_analysis():
    """RecoveryAgent can analyze fault signatures."""
    recovery = RecoveryAgent()

    # Drift pattern
    history = [
        DeviceState(name="heater", status="running", telemetry={"temperature": 100.0}),
        DeviceState(name="heater", status="running", telemetry={"temperature": 102.0}),
        DeviceState(name="heater", status="running", telemetry={"temperature": 104.0}),
        DeviceState(name="heater", status="running", telemetry={"temperature": 106.0}),
    ]
    assert recovery.analyze_fault_signature(history) == "drift"

    # Stall pattern
    stall_history = [
        DeviceState(name="heater", status="running", telemetry={"temperature": 100.0}),
        DeviceState(name="heater", status="running", telemetry={"temperature": 100.0}),
        DeviceState(name="heater", status="running", telemetry={"temperature": 100.0}),
    ]
    assert recovery.analyze_fault_signature(stall_history) == "stall"
