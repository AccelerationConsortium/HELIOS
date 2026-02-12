"""
Tests for the policy-driven recovery system.

These tests validate the decision logic against expected outcomes.
"""
import pytest
from exp_agent.core.types import DeviceState, HardwareError, Action
from exp_agent.recovery.policy import (
    decide_recovery,
    analyze_signature,
    classify_error,
    SignatureConfig,
    RecoveryConfig,
    SIGNATURE_CONFIG,
    RECOVERY_CONFIG,
)


# ============================================================================
# Signature Analysis Tests
# ============================================================================

class TestSignatureAnalysis:
    """Test fault signature detection."""

    def test_drift_detection(self):
        """Temperature steadily increasing -> drift."""
        history = [
            DeviceState(name="heater", status="running", telemetry={"temperature": 100.0}),
            DeviceState(name="heater", status="running", telemetry={"temperature": 102.0}),
            DeviceState(name="heater", status="running", telemetry={"temperature": 104.0}),
            DeviceState(name="heater", status="running", telemetry={"temperature": 106.0}),
        ]
        result = analyze_signature(history)
        assert result.mode == "drift"
        assert result.confidence > 0.5

    def test_stall_detection(self):
        """Temperature not changing -> stall."""
        history = [
            DeviceState(name="heater", status="running", telemetry={"temperature": 100.0}),
            DeviceState(name="heater", status="running", telemetry={"temperature": 100.0}),
            DeviceState(name="heater", status="running", telemetry={"temperature": 100.0}),
            DeviceState(name="heater", status="running", telemetry={"temperature": 100.0}),
        ]
        result = analyze_signature(history)
        assert result.mode == "stall"

    def test_oscillation_detection(self):
        """Temperature bouncing up and down -> oscillation."""
        history = [
            DeviceState(name="heater", status="running", telemetry={"temperature": 115.0}),
            DeviceState(name="heater", status="running", telemetry={"temperature": 122.0}),
            DeviceState(name="heater", status="running", telemetry={"temperature": 116.0}),
            DeviceState(name="heater", status="running", telemetry={"temperature": 121.0}),
            DeviceState(name="heater", status="running", telemetry={"temperature": 118.0}),
        ]
        result = analyze_signature(history)
        assert result.mode == "oscillation"

    def test_stable_detection(self):
        """Small variations around target -> stable."""
        history = [
            DeviceState(name="heater", status="running", telemetry={"temperature": 120.0}),
            DeviceState(name="heater", status="running", telemetry={"temperature": 120.2}),
            DeviceState(name="heater", status="running", telemetry={"temperature": 119.9}),
            DeviceState(name="heater", status="running", telemetry={"temperature": 120.1}),
        ]
        result = analyze_signature(history)
        assert result.mode == "stable"

    def test_insufficient_history(self):
        """Too few samples -> unknown."""
        history = [
            DeviceState(name="heater", status="running", telemetry={"temperature": 100.0}),
        ]
        result = analyze_signature(history)
        assert result.mode == "unknown"


# ============================================================================
# Error Classification Tests
# ============================================================================

class TestErrorClassification:
    """Test error profile classification."""

    def test_overshoot_classification(self):
        """Overshoot is unsafe but recoverable."""
        error = HardwareError(
            device="heater",
            type="overshoot",
            severity="high",
            message="Temperature exceeded target",
            when="now"
        )
        profile = classify_error(error)
        assert profile.unsafe is True
        assert profile.recoverable is True
        assert profile.default_strategy == "degrade"

    def test_sensor_fail_classification(self):
        """Sensor failure is non-recoverable."""
        error = HardwareError(
            device="heater",
            type="sensor_fail",
            severity="high",
            message="Sensor returned invalid value",
            when="now"
        )
        profile = classify_error(error)
        assert profile.unsafe is True
        assert profile.recoverable is False
        assert profile.default_strategy == "abort"

    def test_timeout_classification(self):
        """Timeout is recoverable via retry."""
        error = HardwareError(
            device="heater",
            type="timeout",
            severity="medium",
            message="Device did not respond",
            when="now"
        )
        profile = classify_error(error)
        assert profile.unsafe is False
        assert profile.recoverable is True
        assert profile.default_strategy == "retry"


# ============================================================================
# Recovery Decision Tests
# ============================================================================

class TestRecoveryDecisions:
    """Test recovery decision logic."""

    def _make_state(self, temp: float, target: float = None) -> DeviceState:
        telemetry = {"temperature": temp}
        if target:
            telemetry["target"] = target
        return DeviceState(name="heater", status="running", telemetry=telemetry)

    def _make_history(self, temps: list) -> list:
        return [
            DeviceState(name="heater", status="running", telemetry={"temperature": t})
            for t in temps
        ]

    def test_overshoot_with_drift_degrade(self):
        """Overshoot + drift signature -> DEGRADE."""
        state = self._make_state(128.0, 120.0)
        error = HardwareError(
            device="heater",
            type="overshoot",
            severity="high",
            message="Temperature exceeded target by 8°C",
            when="now"
        )
        history = self._make_history([118.0, 122.0, 125.0, 128.0])
        last_action = Action(name="set_temperature", effect="write", params={"temperature": 120.0})

        decision = decide_recovery(
            state=state,
            error=error,
            history=history,
            retry_counts={},
            last_action=last_action
        )

        assert decision.kind == "degrade"
        assert decision.sample_status == "compromised"
        assert any(a.name == "cool_down" for a in decision.actions)

    def test_sensor_fail_abort(self):
        """Sensor failure -> ABORT."""
        state = self._make_state(-999)
        error = HardwareError(
            device="heater",
            type="sensor_fail",
            severity="high",
            message="Sensor returned -999",
            when="now"
        )

        decision = decide_recovery(
            state=state,
            error=error,
            history=[],
            retry_counts={}
        )

        assert decision.kind == "abort"
        assert any(a.name == "cool_down" for a in decision.actions)

    def test_timeout_first_retry(self):
        """First timeout -> RETRY with no wait."""
        state = self._make_state(100.0)
        error = HardwareError(
            device="heater",
            type="timeout",
            severity="medium",
            message="Device did not respond",
            when="now"
        )

        decision = decide_recovery(
            state=state,
            error=error,
            history=[],
            retry_counts={"timeout": 0}
        )

        assert decision.kind == "retry"
        assert decision.sample_status == "intact"

    def test_timeout_with_backoff(self):
        """Second timeout -> RETRY with 2s backoff."""
        state = self._make_state(100.0)
        error = HardwareError(
            device="heater",
            type="timeout",
            severity="medium",
            message="Device did not respond",
            when="now"
        )

        decision = decide_recovery(
            state=state,
            error=error,
            history=[],
            retry_counts={"timeout": 1}
        )

        assert decision.kind == "retry"
        # Should have a wait action
        wait_actions = [a for a in decision.actions if a.name == "wait"]
        assert len(wait_actions) == 1
        assert wait_actions[0].params["duration"] == 2.0

    def test_postcondition_stall_abort(self):
        """Postcondition failed + stall -> ABORT."""
        state = self._make_state(100.0)
        error = HardwareError(
            device="heater",
            type="postcondition_failed",
            severity="medium",
            message="Temperature did not reach target",
            when="now"
        )
        history = self._make_history([100.0, 100.0, 100.0, 100.0])

        decision = decide_recovery(
            state=state,
            error=error,
            history=history,
            retry_counts={"postcondition_failed": 1}
        )

        assert decision.kind == "abort"
        assert decision.sample_status == "compromised"

    def test_postcondition_escalation_to_degrade(self):
        """Repeated postcondition failures -> DEGRADE."""
        state = self._make_state(115.0)
        error = HardwareError(
            device="heater",
            type="postcondition_failed",
            severity="medium",
            message="Temperature did not reach target",
            when="now"
        )
        history = self._make_history([112.0, 114.0, 115.0])
        last_action = Action(name="set_temperature", effect="write", params={"temperature": 120.0})

        decision = decide_recovery(
            state=state,
            error=error,
            history=history,
            retry_counts={"postcondition_failed": 2},
            last_action=last_action
        )

        assert decision.kind == "degrade"
        assert decision.sample_status == "compromised"
        # Should include set_temperature with degraded target
        set_temp_actions = [a for a in decision.actions if a.name == "set_temperature"]
        assert len(set_temp_actions) == 1
        assert set_temp_actions[0].params["temperature"] == 110.0  # 120 - 10


# ============================================================================
# Integration Test
# ============================================================================

class TestPolicyIntegration:
    """Test full policy integration scenarios."""

    def test_full_escalation_sequence(self):
        """Test escalation: retry -> retry with wait -> degrade."""
        state = DeviceState(
            name="heater",
            status="running",
            telemetry={"temperature": 115.0}
        )
        error = HardwareError(
            device="heater",
            type="postcondition_failed",
            severity="medium",
            message="Temperature did not reach target",
            when="now"
        )
        history = [
            DeviceState(name="heater", status="running", telemetry={"temperature": 112.0}),
            DeviceState(name="heater", status="running", telemetry={"temperature": 114.0}),
            DeviceState(name="heater", status="running", telemetry={"temperature": 115.0}),
        ]
        last_action = Action(name="set_temperature", effect="write", params={"temperature": 120.0})

        # First failure: retry immediately
        d1 = decide_recovery(state, error, history, {"postcondition_failed": 0}, last_action)
        assert d1.kind == "retry"
        assert d1.actions == []

        # Second failure: retry with wait
        d2 = decide_recovery(state, error, history, {"postcondition_failed": 1}, last_action)
        assert d2.kind == "retry"
        assert any(a.name == "wait" for a in d2.actions)

        # Third failure: degrade
        d3 = decide_recovery(state, error, history, {"postcondition_failed": 2}, last_action)
        assert d3.kind == "degrade"
        assert any(a.name == "cool_down" for a in d3.actions)
        assert any(a.name == "set_temperature" for a in d3.actions)
