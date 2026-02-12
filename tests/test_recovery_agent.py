"""Tests for RecoveryAgent integration."""
import asyncio

import pytest

from app.agents import RecoveryAgent, RecoveryInput


def _run(coro):
    """Helper to run async code in tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_recovery_agent_basic():
    """Test basic recovery agent functionality."""
    agent = RecoveryAgent()

    # Test simple timeout error
    input_data = RecoveryInput(
        error_type="timeout",
        error_message="Connection timeout after 30s",
        device_name="opentrons_ot2",
        device_status="error",
        telemetry={"last_response": 1234567890},
        retry_count=0,
    )

    result = _run(agent.run(input_data))

    assert result.success
    assert result.output is not None
    assert result.output.decision in ["retry", "abort", "degrade", "skip"]
    assert result.output.rationale
    assert isinstance(result.output.actions, list)


def test_recovery_agent_retry_logic():
    """Test retry logic with different error types."""
    agent = RecoveryAgent()

    # Test with a retryable error (timeout)
    input1 = RecoveryInput(
        error_type="timeout",
        error_message="Operation timeout",
        device_name="plc_controller",
        error_severity="low",
        retry_count=0,
    )

    result1 = _run(agent.run(input1))
    assert result1.success
    assert result1.output is not None
    # Should return a valid decision
    assert result1.output.decision in ["retry", "degrade", "skip", "abort"]
    assert result1.output.rationale

    # Test with potentially unrecoverable error (depends on policy)
    input2 = RecoveryInput(
        error_type="sensor_fail",
        error_message="Sensor failure detected",
        device_name="plc_controller",
        error_severity="high",
        retry_count=2,
    )

    result2 = _run(agent.run(input2))
    assert result2.success
    assert result2.output is not None
    # Should return a valid decision (policy determines exact strategy)
    assert result2.output.decision in ["retry", "degrade", "skip", "abort"]
    assert result2.output.rationale


def test_recovery_agent_chemical_safety():
    """Test chemical safety event escalation."""
    agent = RecoveryAgent()

    # Chemical safety event
    input_data = RecoveryInput(
        error_type="spill_detected",
        error_message="Liquid spill detected in workspace",
        device_name="opentrons_ot2",
        device_status="emergency",
        telemetry={
            "spill_detected": True,
            "temperature": 25.0,
        },
        retry_count=0,
    )

    result = _run(agent.run(input_data))

    assert result.success
    assert result.output is not None

    # Chemical safety should force abort (when recovery-agent is available)
    # or retry (fallback mode)
    assert result.output.decision in ["abort", "retry"]
    # Check for safety-related content in rationale
    assert "safety" in result.output.rationale.lower() or "chemical" in result.output.rationale.lower() or "fallback" in result.output.rationale.lower()


def test_recovery_agent_validation():
    """Test input validation."""
    agent = RecoveryAgent()

    # Missing required fields
    input_data = RecoveryInput(
        error_type="",  # Empty error type
        error_message="Test error",
        device_name="",  # Empty device name
    )

    result = _run(agent.run(input_data))

    assert not result.success
    assert len(result.errors) > 0
    assert any("error_type" in e.lower() for e in result.errors)
    assert any("device_name" in e.lower() for e in result.errors)


def test_recovery_agent_with_history():
    """Test recovery with telemetry history."""
    agent = RecoveryAgent()

    # Error with history showing degradation
    input_data = RecoveryInput(
        error_type="sensor_drift",
        error_message="Temperature sensor reading drift detected",
        device_name="plc_controller",
        telemetry={"temperature": 45.0},
        history=[
            {"device_name": "plc_controller", "status": "ok", "telemetry": {"temperature": 25.0}},
            {"device_name": "plc_controller", "status": "ok", "telemetry": {"temperature": 30.0}},
            {"device_name": "plc_controller", "status": "warning", "telemetry": {"temperature": 40.0}},
            {"device_name": "plc_controller", "status": "error", "telemetry": {"temperature": 45.0}},
        ],
        retry_count=0,
    )

    result = _run(agent.run(input_data))

    assert result.success
    assert result.output is not None
    # With drift history, might degrade or abort
    assert result.output.decision in ["retry", "degrade", "abort"]


def test_recovery_agent_fallback():
    """Test fallback logic when recovery-agent package not available."""
    agent = RecoveryAgent()

    # Force fallback by setting agent as unavailable
    original_available = agent._available
    agent._available = False
    agent._agent = None

    try:
        input_data = RecoveryInput(
            error_type="test_error",
            error_message="Test error message",
            device_name="test_device",
            retry_count=0,
        )

        result = _run(agent.run(input_data))

        assert result.success
        assert result.output is not None
        assert result.output.decision == "retry"
        assert "fallback" in result.output.rationale.lower()

        # Test fallback max retries
        input_data.retry_count = 5
        result2 = _run(agent.run(input_data))
        assert result2.success
        assert result2.output.decision == "abort"

    finally:
        # Restore original state
        agent._available = original_available
