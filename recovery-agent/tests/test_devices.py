"""
Tests for simulated devices.

Validates that each device:
1. Initializes correctly
2. Responds to actions
3. Generates appropriate faults
4. Reports telemetry correctly
"""
import pytest
import time
from exp_agent.core.types import Action, HardwareError
from exp_agent.devices.simulated import SimHeater, SimPump, SimPositioner, SimSpectrometer


# ============================================================================
# SimHeater Tests
# ============================================================================

class TestSimHeater:
    """Test the simulated heater device."""

    def test_init(self):
        """Heater initializes with correct defaults."""
        h = SimHeater(name="test_heater")
        assert h.name == "test_heater"
        assert h.current_temp == 25.0
        assert h.status == "idle"

    def test_set_temperature(self):
        """Can set target temperature."""
        h = SimHeater(name="test_heater")
        h.execute(Action(name="set_temperature", effect="write", params={"temperature": 100}))
        assert h.target_temp == 100
        assert h.heating is True
        assert h.status == "running"

    def test_cool_down(self):
        """Cool down resets to ambient."""
        h = SimHeater(name="test_heater")
        h.execute(Action(name="set_temperature", effect="write", params={"temperature": 100}))
        h.execute(Action(name="cool_down", effect="write"))
        assert h.target_temp == 25.0
        assert h.heating is False
        assert h.status == "idle"

    def test_sensor_fail_fault(self):
        """Sensor fail mode raises HardwareError."""
        h = SimHeater(name="test_heater", fault_mode="sensor_fail")
        # Run some ticks to trigger fault
        for _ in range(10):
            h.tick()
        with pytest.raises(HardwareError) as exc:
            h.read_state()
        assert exc.value.type == "sensor_fail"


# ============================================================================
# SimPump Tests
# ============================================================================

class TestSimPump:
    """Test the simulated pump device."""

    def test_init(self):
        """Pump initializes with correct defaults."""
        p = SimPump(name="test_pump")
        assert p.name == "test_pump"
        assert p.flow_rate == 0.0
        assert p.pressure == 1.0
        assert p.status == "idle"

    def test_set_flow(self):
        """Can set flow rate."""
        p = SimPump(name="test_pump")
        p.execute(Action(name="set_flow", effect="write", params={"flow_rate": 50.0}))
        assert p.target_flow == 50.0
        assert p.running is True
        assert p.status == "running"

    def test_stop_pump(self):
        """Stop pump halts flow."""
        p = SimPump(name="test_pump")
        p.execute(Action(name="set_flow", effect="write", params={"flow_rate": 50.0}))
        p.execute(Action(name="stop_pump", effect="write"))
        assert p.target_flow == 0.0
        assert p.running is False

    def test_prime_pump(self):
        """Prime runs at low flow."""
        p = SimPump(name="test_pump")
        p.execute(Action(name="prime_pump", effect="write"))
        assert p.target_flow == 5.0
        assert p.running is True

    def test_flow_blocked_fault(self):
        """Flow blocked mode raises HardwareError."""
        p = SimPump(name="test_pump", fault_mode="flow_blocked")
        p.execute(Action(name="set_flow", effect="write", params={"flow_rate": 50.0}))
        # Manually set state to simulate fault condition
        p.running = True
        p.flow_rate = 0.0  # Blocked
        p.tick_count = 5
        p.run_start_time = time.time() - 2.0  # Simulate 2s of running
        with pytest.raises(HardwareError) as exc:
            p.read_state()
        assert exc.value.type == "flow_blocked"

    def test_telemetry_fields(self):
        """Pump reports expected telemetry."""
        p = SimPump(name="test_pump")
        state = p.read_state()
        assert "flow_rate" in state.telemetry
        assert "pressure" in state.telemetry
        assert "target_flow" in state.telemetry
        assert "running" in state.telemetry


# ============================================================================
# SimPositioner Tests
# ============================================================================

class TestSimPositioner:
    """Test the simulated positioner device."""

    def test_init(self):
        """Positioner initializes at origin."""
        pos = SimPositioner(name="test_pos")
        assert pos.name == "test_pos"
        assert pos.position == {"x": 0.0, "y": 0.0, "z": 0.0}
        assert pos.status == "idle"

    def test_move_to(self):
        """Can move to absolute position."""
        pos = SimPositioner(name="test_pos")
        pos.execute(Action(name="move_to", effect="write", params={"x": 10.0, "y": 5.0, "z": 20.0}))
        assert pos.target == {"x": 10.0, "y": 5.0, "z": 20.0}
        assert pos.moving is True
        assert pos.status == "moving"

    def test_move_relative(self):
        """Can move relative to current position."""
        pos = SimPositioner(name="test_pos")
        pos.position = {"x": 5.0, "y": 5.0, "z": 5.0}
        pos.execute(Action(name="move_relative", effect="write", params={"dx": 3.0, "dy": -2.0, "dz": 1.0}))
        assert pos.target == {"x": 8.0, "y": 3.0, "z": 6.0}

    def test_home(self):
        """Home returns to origin."""
        pos = SimPositioner(name="test_pos")
        pos.position = {"x": 10.0, "y": 10.0, "z": 10.0}
        pos.execute(Action(name="home", effect="write"))
        assert pos.target == {"x": 0.0, "y": 0.0, "z": 0.0}
        assert pos.moving is True

    def test_stop(self):
        """Emergency stop halts motion."""
        pos = SimPositioner(name="test_pos")
        pos.execute(Action(name="move_to", effect="write", params={"x": 50.0}))
        pos.execute(Action(name="stop", effect="write"))
        assert pos.moving is False
        assert pos.status == "idle"

    def test_retract(self):
        """Retract moves Z up."""
        pos = SimPositioner(name="test_pos")
        pos.position = {"x": 10.0, "y": 10.0, "z": 5.0}
        pos.execute(Action(name="retract", effect="write"))
        assert pos.target["z"] == 15.0  # +10 from current

    def test_collision_fault(self):
        """Collision mode raises HardwareError."""
        pos = SimPositioner(name="test_pos", fault_mode="collision")
        pos.execute(Action(name="move_to", effect="write", params={"x": 30.0}))
        # Manually set collision state
        pos.collision_detected = True
        pos.status = "error"
        with pytest.raises(HardwareError) as exc:
            pos.read_state()
        assert exc.value.type == "collision"

    def test_telemetry_fields(self):
        """Positioner reports expected telemetry."""
        pos = SimPositioner(name="test_pos")
        state = pos.read_state()
        assert "x" in state.telemetry
        assert "y" in state.telemetry
        assert "z" in state.telemetry
        assert "moving" in state.telemetry


# ============================================================================
# SimSpectrometer Tests
# ============================================================================

class TestSimSpectrometer:
    """Test the simulated spectrometer device."""

    def test_init(self):
        """Spectrometer initializes correctly."""
        spec = SimSpectrometer(name="test_spec")
        assert spec.name == "test_spec"
        assert spec.integration_time == 100
        assert spec.lamp_on is False
        assert spec.status == "idle"

    def test_lamp_control(self):
        """Can turn lamp on/off."""
        spec = SimSpectrometer(name="test_spec")
        spec.execute(Action(name="lamp_on", effect="write"))
        assert spec.lamp_on is True
        spec.execute(Action(name="lamp_off", effect="write"))
        assert spec.lamp_on is False

    def test_start_acquisition(self):
        """Can start acquisition."""
        spec = SimSpectrometer(name="test_spec")
        spec.execute(Action(name="lamp_on", effect="write"))
        spec.execute(Action(name="start_acquisition", effect="write"))
        assert spec.acquiring is True
        assert spec.status == "acquiring"

    def test_set_integration_time(self):
        """Can set integration time."""
        spec = SimSpectrometer(name="test_spec")
        spec.execute(Action(name="set_integration_time", effect="write", params={"time_ms": 200}))
        assert spec.integration_time == 200

    def test_reduce_integration(self):
        """Reduce integration for saturation."""
        spec = SimSpectrometer(name="test_spec")
        spec.integration_time = 100
        spec.execute(Action(name="reduce_integration", effect="write", params={"factor": 0.5}))
        assert spec.integration_time == 50

    def test_dark_subtract(self):
        """Dark subtract resets baseline."""
        spec = SimSpectrometer(name="test_spec")
        spec.baseline = 500.0
        spec.execute(Action(name="dark_subtract", effect="write"))
        assert spec.baseline == 100.0

    def test_recalibrate(self):
        """Recalibrate resets wavelength offset."""
        spec = SimSpectrometer(name="test_spec")
        spec.wavelength_offset = 5.0
        spec.execute(Action(name="recalibrate", effect="write"))
        assert spec.wavelength_offset == 0.0

    def test_signal_saturation_fault(self):
        """Signal saturation mode raises HardwareError."""
        spec = SimSpectrometer(name="test_spec", fault_mode="signal_saturated")
        spec.execute(Action(name="lamp_on", effect="write"))
        spec.execute(Action(name="start_acquisition", effect="write"))
        # Simulate time passing
        time.sleep(0.1)
        for _ in range(10):
            spec.tick()
            time.sleep(0.1)
        with pytest.raises(HardwareError) as exc:
            spec.read_state()
        assert exc.value.type == "signal_saturated"

    def test_telemetry_fields(self):
        """Spectrometer reports expected telemetry."""
        spec = SimSpectrometer(name="test_spec")
        state = spec.read_state()
        assert "signal_intensity" in state.telemetry
        assert "peak_wavelength" in state.telemetry
        assert "baseline" in state.telemetry
        assert "integration_time" in state.telemetry
        assert "lamp_on" in state.telemetry


# ============================================================================
# Cross-Device Policy Integration Tests
# ============================================================================

class TestPolicyIntegration:
    """Test that policy correctly classifies errors from all devices."""

    def test_classify_pump_errors(self):
        """Policy classifies pump errors correctly."""
        from exp_agent.recovery.policy import classify_error

        # Flow blocked - unsafe, recoverable
        err = HardwareError(device="pump", type="flow_blocked", severity="high", message="test", when="now")
        profile = classify_error(err)
        assert profile.unsafe is True
        assert profile.recoverable is True

        # Leak detected - unsafe, non-recoverable
        err = HardwareError(device="pump", type="leak_detected", severity="critical", message="test", when="now")
        profile = classify_error(err)
        assert profile.unsafe is True
        assert profile.recoverable is False

    def test_classify_positioner_errors(self):
        """Policy classifies positioner errors correctly."""
        from exp_agent.recovery.policy import classify_error

        # Collision - unsafe, non-recoverable
        err = HardwareError(device="positioner", type="collision", severity="critical", message="test", when="now")
        profile = classify_error(err)
        assert profile.unsafe is True
        assert profile.recoverable is False

        # Position drift - safe, recoverable
        err = HardwareError(device="positioner", type="position_drift", severity="medium", message="test", when="now")
        profile = classify_error(err)
        assert profile.unsafe is False
        assert profile.recoverable is True

    def test_classify_spectrometer_errors(self):
        """Policy classifies spectrometer errors correctly."""
        from exp_agent.recovery.policy import classify_error

        # Signal saturated - safe, recoverable
        err = HardwareError(device="spec", type="signal_saturated", severity="medium", message="test", when="now")
        profile = classify_error(err)
        assert profile.unsafe is False
        assert profile.recoverable is True

        # Lamp failure - safe but non-recoverable (hardware)
        err = HardwareError(device="spec", type="lamp_failure", severity="high", message="test", when="now")
        profile = classify_error(err)
        assert profile.recoverable is False


# ============================================================================
# Multi-Metric Signature Analysis Tests
# ============================================================================

class TestMultiMetricSignature:
    """Test signature analysis with different telemetry types."""

    def test_flow_rate_drift(self):
        """Detect drift in flow rate."""
        from exp_agent.recovery.policy import analyze_signature

        history = [
            {"name": "pump", "status": "running", "telemetry": {"flow_rate": 50.0}},
            {"name": "pump", "status": "running", "telemetry": {"flow_rate": 48.0}},
            {"name": "pump", "status": "running", "telemetry": {"flow_rate": 46.0}},
            {"name": "pump", "status": "running", "telemetry": {"flow_rate": 44.0}},
        ]
        from exp_agent.core.types import DeviceState
        history = [DeviceState(**h) for h in history]

        result = analyze_signature(history, metric="flow_rate")
        assert result.mode == "drift"

    def test_position_stall(self):
        """Detect stall in position."""
        from exp_agent.recovery.policy import analyze_signature
        from exp_agent.core.types import DeviceState

        history = [
            DeviceState(name="pos", status="running", telemetry={"x": 10.0, "y": 5.0, "z": 0.0}),
            DeviceState(name="pos", status="running", telemetry={"x": 10.0, "y": 5.0, "z": 0.0}),
            DeviceState(name="pos", status="running", telemetry={"x": 10.0, "y": 5.0, "z": 0.0}),
            DeviceState(name="pos", status="running", telemetry={"x": 10.0, "y": 5.0, "z": 0.0}),
        ]
        result = analyze_signature(history, metric="x")
        assert result.mode == "stall"

    def test_signal_oscillation(self):
        """Detect oscillation in signal intensity."""
        from exp_agent.recovery.policy import analyze_signature
        from exp_agent.core.types import DeviceState

        history = [
            DeviceState(name="spec", status="running", telemetry={"signal_intensity": 30000}),
            DeviceState(name="spec", status="running", telemetry={"signal_intensity": 35000}),
            DeviceState(name="spec", status="running", telemetry={"signal_intensity": 30500}),
            DeviceState(name="spec", status="running", telemetry={"signal_intensity": 34500}),
            DeviceState(name="spec", status="running", telemetry={"signal_intensity": 31000}),
        ]
        result = analyze_signature(history, metric="signal_intensity")
        assert result.mode == "oscillation"
