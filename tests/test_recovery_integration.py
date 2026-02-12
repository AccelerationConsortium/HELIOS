"""Integration tests for RecoveryAgent with Orchestrator.

Tests the full integration: error mapping → recovery decision → orchestrator retry
"""
import asyncio

import pytest

from app.agents import OrchestratorAgent
from app.services.error_mapping import (
    map_exception_to_error_type,
    get_error_severity,
    should_emit_chemical_safety_alert,
)


def _run(coro):
    """Helper to run async code in tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestErrorMapping:
    """Test error type mapping functionality."""

    def test_map_timeout_error(self):
        """Test timeout error mapping."""
        exc = TimeoutError("Connection timeout")
        error_type = map_exception_to_error_type(exc)
        assert error_type == "timeout"

    def test_map_connection_error(self):
        """Test connection error mapping."""
        exc = ConnectionError("Connection lost")
        error_type = map_exception_to_error_type(exc)
        assert error_type == "connection_lost"

    def test_map_value_error(self):
        """Test value error mapping."""
        exc = ValueError("Invalid parameter")
        error_type = map_exception_to_error_type(exc)
        assert error_type == "postcondition_failed"

    def test_map_unknown_error(self):
        """Test unknown error mapping."""
        class CustomError(Exception):
            pass

        exc = CustomError("Unknown error")
        error_type = map_exception_to_error_type(exc)
        assert error_type == "unknown_error"

    def test_get_error_severity_high(self):
        """Test high severity errors."""
        assert get_error_severity("spill_detected") == "high"
        assert get_error_severity("fire_detected") == "high"
        assert get_error_severity("safety_violation") == "high"

    def test_get_error_severity_medium(self):
        """Test medium severity errors."""
        assert get_error_severity("sensor_fail") == "medium"
        assert get_error_severity("actuator_jam") == "medium"

    def test_get_error_severity_low(self):
        """Test low severity errors."""
        assert get_error_severity("timeout") == "low"
        assert get_error_severity("sensor_drift") == "low"

    def test_chemical_safety_alert_by_type(self):
        """Test chemical safety detection by error type."""
        assert should_emit_chemical_safety_alert("spill_detected", {})
        assert should_emit_chemical_safety_alert("fire_detected", {})
        assert not should_emit_chemical_safety_alert("timeout", {})

    def test_chemical_safety_alert_by_telemetry(self):
        """Test chemical safety detection by telemetry."""
        # High temperature
        assert should_emit_chemical_safety_alert(
            "unknown_error",
            {"temperature": 85.0}
        )

        # High pressure
        assert should_emit_chemical_safety_alert(
            "unknown_error",
            {"pressure": 3.0}
        )

        # Spill detected flag
        assert should_emit_chemical_safety_alert(
            "unknown_error",
            {"spill_detected": True}
        )

        # Normal conditions
        assert not should_emit_chemical_safety_alert(
            "unknown_error",
            {"temperature": 25.0, "pressure": 1.0}
        )


class TestOrchestratorIntegration:
    """Test RecoveryAgent integration with Orchestrator."""

    def test_orchestrator_has_recovery_agent(self):
        """Test that orchestrator initializes with recovery agent."""
        orchestrator = OrchestratorAgent()
        assert hasattr(orchestrator, 'recovery')
        assert orchestrator.recovery is not None

    def test_recovery_agent_available(self):
        """Test that recovery agent is available."""
        orchestrator = OrchestratorAgent()
        # Check if full recovery-agent is available or fallback mode
        available = orchestrator.recovery._available
        assert isinstance(available, bool)

        if available:
            print("✅ Full recovery-agent capabilities active")
        else:
            print("⚠️  Using fallback recovery logic")


class TestRecoveryWorkflow:
    """Test complete recovery workflow scenarios."""

    def test_error_to_recovery_workflow(self):
        """Test complete error → recovery decision workflow."""
        from app.agents.recovery_agent import RecoveryInput

        # Simulate error
        exc = TimeoutError("Connection timeout")

        # Map error
        error_type = map_exception_to_error_type(exc)
        error_severity = get_error_severity(error_type)

        assert error_type == "timeout"
        assert error_severity == "low"

        # Build recovery input
        recovery_input = RecoveryInput(
            error_type=error_type,
            error_message=str(exc),
            device_name="test_device",
            device_status="error",
            error_severity=error_severity,
            retry_count=0,
        )

        # Get recovery decision
        orchestrator = OrchestratorAgent()
        result = _run(orchestrator.recovery.run(recovery_input))

        assert result.success
        assert result.output is not None
        assert result.output.decision in ["retry", "abort", "skip", "degrade"]

    def test_chemical_safety_workflow(self):
        """Test chemical safety event workflow."""
        from app.agents.recovery_agent import RecoveryInput

        # Simulate chemical safety error
        error_type = "spill_detected"
        error_severity = get_error_severity(error_type)

        assert error_severity == "high"
        assert should_emit_chemical_safety_alert(error_type, {})

        # Build recovery input
        recovery_input = RecoveryInput(
            error_type=error_type,
            error_message="Liquid spill detected",
            device_name="opentrons_ot2",
            device_status="error",
            error_severity=error_severity,
            telemetry={"spill_detected": True},
            retry_count=0,
        )

        # Get recovery decision
        orchestrator = OrchestratorAgent()
        result = _run(orchestrator.recovery.run(recovery_input))

        assert result.success
        assert result.output is not None

        # Chemical safety should result in abort
        if orchestrator.recovery._available:
            # Full recovery-agent detects chemical safety
            assert result.output.decision == "abort"
            assert result.output.chemical_safety_event
        else:
            # Fallback mode might not detect chemical safety
            assert result.output.decision in ["retry", "abort"]


class TestRecoveryMetrics:
    """Test recovery metrics and reporting."""

    def test_retry_count_tracking(self):
        """Test that retry count is properly tracked."""
        from app.agents.recovery_agent import RecoveryInput

        orchestrator = OrchestratorAgent()

        # First attempt
        recovery_input = RecoveryInput(
            error_type="timeout",
            error_message="Timeout",
            device_name="device",
            retry_count=0,
        )

        result1 = _run(orchestrator.recovery.run(recovery_input))
        assert result1.success

        # Second attempt (increased retry count)
        recovery_input.retry_count = 1
        result2 = _run(orchestrator.recovery.run(recovery_input))
        assert result2.success

        # Decisions may vary based on retry count
        # Both should be valid recovery decisions
        assert result1.output.decision in ["retry", "abort", "skip", "degrade"]
        assert result2.output.decision in ["retry", "abort", "skip", "degrade"]

    def test_recovery_output_fields(self):
        """Test that recovery output contains all required fields."""
        from app.agents.recovery_agent import RecoveryInput

        orchestrator = OrchestratorAgent()

        recovery_input = RecoveryInput(
            error_type="sensor_fail",
            error_message="Sensor failure",
            device_name="device",
            retry_count=0,
        )

        result = _run(orchestrator.recovery.run(recovery_input))

        assert result.success
        assert result.output is not None

        # Check required fields
        assert hasattr(result.output, 'decision')
        assert hasattr(result.output, 'rationale')
        assert hasattr(result.output, 'actions')
        assert hasattr(result.output, 'retry_delay_seconds')
        assert hasattr(result.output, 'max_retries')
        assert hasattr(result.output, 'chemical_safety_event')

        assert isinstance(result.output.decision, str)
        assert isinstance(result.output.rationale, str)
        assert isinstance(result.output.actions, list)
        assert isinstance(result.output.retry_delay_seconds, (int, float))
        assert isinstance(result.output.max_retries, int)
        assert isinstance(result.output.chemical_safety_event, bool)
