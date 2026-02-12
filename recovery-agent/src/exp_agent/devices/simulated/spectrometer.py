"""
SimSpectrometer - Simulated optical spectrometer for SDL fault recovery testing.

Fault modes:
- none: Normal operation
- signal_saturated: Detector saturates (readings max out)
- baseline_drift: Baseline drifts over time, affecting measurements
- calibration_lost: Wavelength calibration becomes unreliable
- low_signal: Signal too weak to measure (below noise floor)
- lamp_failure: Light source fails
"""
import time
import math
import random
from typing import Literal, Optional, List
from ..base import Device
from ...core.types import DeviceState, Action, HardwareError


class SimSpectrometer(Device):
    def __init__(
        self,
        name: str,
        fault_mode: Literal["none", "signal_saturated", "baseline_drift", "calibration_lost", "low_signal", "lamp_failure"] = "none"
    ):
        super().__init__(name)
        self.fault_mode = fault_mode

        # State variables
        self.integration_time = 100  # ms
        self.wavelength_range = (400, 800)  # nm (visible)
        self.signal_intensity = 0.0  # 0-65535 (16-bit ADC)
        self.baseline = 100.0  # Dark counts
        self.lamp_on = False
        self.acquiring = False
        self.status: Literal["idle", "acquiring", "error", "ready"] = "idle"

        # Calibration state
        self.wavelength_offset = 0.0  # nm (calibration error)
        self.gain = 1.0

        # Measurement results
        self.last_spectrum: List[float] = []
        self.peak_wavelength = 0.0
        self.peak_intensity = 0.0

        # Safety/quality thresholds
        self.saturation_level = 60000
        self.noise_floor = 200
        self.max_baseline_drift = 500

        # Simulation state
        self.tick_count = 0
        self.last_update_time = time.time()
        self.acquisition_start_time = 0.0

    def _simulate_spectrum(self) -> List[float]:
        """Generate a simulated spectrum with a peak."""
        n_points = 100
        spectrum = []

        # Simulate a Gaussian peak around 550nm (green)
        peak_center = 550.0 + self.wavelength_offset
        peak_width = 30.0

        for i in range(n_points):
            wavelength = self.wavelength_range[0] + i * (self.wavelength_range[1] - self.wavelength_range[0]) / n_points

            # Gaussian peak
            signal = 30000 * math.exp(-((wavelength - peak_center) ** 2) / (2 * peak_width ** 2))

            # Add baseline and noise
            signal += self.baseline + random.gauss(0, 50)

            # Apply gain
            signal *= self.gain

            spectrum.append(max(0, min(65535, signal)))

        return spectrum

    def _update_physics(self):
        now = time.time()
        dt = (now - self.last_update_time) * 2.0
        self.last_update_time = now

        if self.acquiring:
            if self.acquisition_start_time == 0.0:
                self.acquisition_start_time = now

            duration = now - self.acquisition_start_time

            # Fault injection
            if self.fault_mode == "baseline_drift":
                # Baseline drifts upward over time
                self.baseline += 50 * dt

            elif self.fault_mode == "calibration_lost" and duration > 1.0:
                # Wavelength calibration drifts
                self.wavelength_offset += 2.0 * dt

            elif self.fault_mode == "signal_saturated":
                # Gain too high, signal saturates
                self.gain = 3.0

            elif self.fault_mode == "low_signal":
                # Gain too low or sample issue
                self.gain = 0.01

            elif self.fault_mode == "lamp_failure" and duration > 1.5:
                # Lamp dies
                self.lamp_on = False
                self.gain = 0.0
        else:
            self.acquisition_start_time = 0.0

    def read_state(self) -> DeviceState:
        self._update_physics()

        # Generate spectrum if acquiring
        if self.acquiring and self.lamp_on:
            self.last_spectrum = self._simulate_spectrum()
            self.peak_intensity = max(self.last_spectrum)
            self.peak_wavelength = self.wavelength_range[0] + \
                self.last_spectrum.index(self.peak_intensity) * \
                (self.wavelength_range[1] - self.wavelength_range[0]) / len(self.last_spectrum)
            self.signal_intensity = sum(self.last_spectrum) / len(self.last_spectrum)

        # Check for saturation
        if self.fault_mode == "signal_saturated" and self.peak_intensity > self.saturation_level:
            if self.tick_count > 3:
                raise HardwareError(
                    device=self.name,
                    type="signal_saturated",
                    severity="medium",
                    message=f"Detector saturated: peak={self.peak_intensity:.0f} (max={self.saturation_level})",
                    when=str(time.time()),
                    context={"peak_intensity": self.peak_intensity, "integration_time": self.integration_time}
                )

        # Check for baseline drift
        if self.baseline > self.max_baseline_drift:
            self.status = "error"

        # Check for lamp failure
        if self.fault_mode == "lamp_failure" and not self.lamp_on and self.tick_count > 5:
            raise HardwareError(
                device=self.name,
                type="lamp_failure",
                severity="high",
                message="Light source failure: no signal detected",
                when=str(time.time()),
                context={"lamp_on": self.lamp_on, "signal": self.signal_intensity}
            )

        # Check for low signal
        if self.fault_mode == "low_signal" and self.peak_intensity < self.noise_floor and self.tick_count > 3:
            raise HardwareError(
                device=self.name,
                type="low_signal",
                severity="medium",
                message=f"Signal below noise floor: {self.peak_intensity:.0f} < {self.noise_floor}",
                when=str(time.time()),
                context={"peak_intensity": self.peak_intensity, "noise_floor": self.noise_floor}
            )

        return DeviceState(
            name=self.name,
            status=self.status,
            telemetry={
                "signal_intensity": round(self.signal_intensity, 1),
                "peak_wavelength": round(self.peak_wavelength, 1),
                "peak_intensity": round(self.peak_intensity, 1),
                "baseline": round(self.baseline, 1),
                "integration_time": self.integration_time,
                "lamp_on": self.lamp_on,
                "acquiring": self.acquiring,
                "wavelength_offset": round(self.wavelength_offset, 2),
                "gain": round(self.gain, 3)
            }
        )

    def execute(self, action: Action) -> None:
        self._update_physics()

        if action.name == "lamp_on":
            self.lamp_on = True
            print(f"[{self.name}] Lamp turned ON")

        elif action.name == "lamp_off":
            self.lamp_on = False
            print(f"[{self.name}] Lamp turned OFF")

        elif action.name == "start_acquisition":
            self.acquiring = True
            self.status = "acquiring"
            print(f"[{self.name}] Starting acquisition (integration={self.integration_time}ms)")

        elif action.name == "stop_acquisition":
            self.acquiring = False
            self.status = "idle"
            print(f"[{self.name}] Acquisition stopped")

        elif action.name == "set_integration_time":
            self.integration_time = action.params.get("time_ms", 100)
            print(f"[{self.name}] Integration time set to {self.integration_time}ms")

        elif action.name == "reduce_integration":
            # Reduce integration time to avoid saturation
            factor = action.params.get("factor", 0.5)
            self.integration_time = int(self.integration_time * factor)
            self.gain *= factor
            print(f"[{self.name}] Reduced integration to {self.integration_time}ms")

        elif action.name == "increase_integration":
            # Increase integration time for weak signal
            factor = action.params.get("factor", 2.0)
            self.integration_time = int(self.integration_time * factor)
            self.gain *= factor
            print(f"[{self.name}] Increased integration to {self.integration_time}ms")

        elif action.name == "dark_subtract":
            # Perform dark subtraction to correct baseline
            self.baseline = 100.0  # Reset to nominal
            print(f"[{self.name}] Dark subtraction performed, baseline reset")

        elif action.name == "recalibrate":
            # Reset wavelength calibration
            self.wavelength_offset = 0.0
            print(f"[{self.name}] Wavelength recalibrated")

        elif action.name == "wait":
            duration = action.params.get("duration", 0)
            print(f"[{self.name}] Waiting {duration}s")

        else:
            print(f"[{self.name}] Unknown action: {action.name}")

    def health(self) -> bool:
        return self.status != "error" and self.baseline < self.max_baseline_drift

    def tick(self, dt: float = 1.0):
        self.tick_count += 1
        print(f"  [SimSpectrometer] Tick={self.tick_count} Signal={self.signal_intensity:.0f} "
              f"Peak={self.peak_wavelength:.1f}nm Baseline={self.baseline:.0f}")
