"""
SimPump - Simulated fluid pump device for SDL fault recovery testing.

Fault modes:
- none: Normal operation
- flow_blocked: Flow rate drops to zero (clog or valve issue)
- pressure_drop: Pressure falls below safe threshold
- leak_detected: Pressure oscillates indicating a leak
- cavitation: Unstable flow with bubbles (noisy signature)
"""
import time
from typing import Literal, Optional
from ..base import Device
from ...core.types import DeviceState, Action, HardwareError


class SimPump(Device):
    def __init__(
        self,
        name: str,
        fault_mode: Literal["none", "flow_blocked", "pressure_drop", "leak_detected", "cavitation"] = "none"
    ):
        super().__init__(name)
        self.fault_mode = fault_mode

        # State variables
        self.flow_rate = 0.0          # mL/min
        self.target_flow = 0.0        # mL/min
        self.pressure = 1.0           # bar (atmospheric)
        self.running = False
        self.status: Literal["idle", "running", "error"] = "idle"

        # Safety limits
        self.max_pressure = 10.0      # bar
        self.min_pressure = 0.5       # bar
        self.max_flow = 100.0         # mL/min

        # Simulation state
        self.tick_count = 0
        self.last_update_time = time.time()
        self.run_start_time = 0.0

    def _update_physics(self):
        now = time.time()
        dt = (now - self.last_update_time) * 2.0  # 2x speed
        self.last_update_time = now

        if self.running:
            if self.run_start_time == 0.0:
                self.run_start_time = now

            duration = now - self.run_start_time

            # Normal behavior: ramp up to target flow
            diff = self.target_flow - self.flow_rate
            rate = 20.0  # mL/min per second
            if abs(diff) > 0.1:
                change = min(abs(diff), rate * dt)
                self.flow_rate += change if diff > 0 else -change

            # Pressure follows flow (simplified model)
            self.pressure = 1.0 + (self.flow_rate / self.max_flow) * 5.0

            # Fault injection
            if self.fault_mode == "flow_blocked" and duration > 1.5:
                # Flow suddenly drops to zero
                self.flow_rate = 0.0
                self.pressure = 8.0  # Pressure builds up

            elif self.fault_mode == "pressure_drop" and duration > 1.5:
                # Pressure drops gradually
                self.pressure = max(0.3, self.pressure - 2.0 * dt)
                self.flow_rate = self.flow_rate * 0.9

            elif self.fault_mode == "leak_detected" and duration > 1.5:
                # Pressure oscillates (leak signature)
                import math
                self.pressure = 3.0 + 1.5 * math.sin(duration * 5.0)

            elif self.fault_mode == "cavitation" and duration > 1.5:
                # Noisy, unstable flow
                import random
                self.flow_rate = self.target_flow * (0.5 + 0.5 * random.random())
                self.pressure = 2.0 + random.random() * 2.0
        else:
            self.run_start_time = 0.0
            # Idle: flow stops, pressure returns to atmospheric
            self.flow_rate = max(0, self.flow_rate - 10.0 * dt)
            self.pressure = 1.0 + (self.pressure - 1.0) * 0.9

    def read_state(self) -> DeviceState:
        self._update_physics()

        # Check for critical conditions
        if self.fault_mode == "flow_blocked" and self.running and self.flow_rate < 0.1:
            if self.tick_count > 3:
                raise HardwareError(
                    device=self.name,
                    type="flow_blocked",
                    severity="high",
                    message=f"Flow blocked: rate={self.flow_rate:.1f} mL/min, pressure={self.pressure:.1f} bar",
                    when=str(time.time()),
                    context={"flow_rate": self.flow_rate, "pressure": self.pressure}
                )

        if self.pressure < self.min_pressure:
            self.status = "error"
        elif self.pressure > self.max_pressure:
            self.status = "error"

        return DeviceState(
            name=self.name,
            status=self.status,
            telemetry={
                "flow_rate": round(self.flow_rate, 2),
                "target_flow": self.target_flow,
                "pressure": round(self.pressure, 2),
                "running": self.running
            }
        )

    def execute(self, action: Action) -> None:
        self._update_physics()

        if action.name == "set_flow":
            target = action.params.get("flow_rate", 0.0)
            self.target_flow = min(target, self.max_flow)
            self.running = True
            self.status = "running"
            print(f"[{self.name}] Setting flow rate to {self.target_flow} mL/min")

        elif action.name == "stop_pump":
            self.target_flow = 0.0
            self.running = False
            self.status = "idle"
            print(f"[{self.name}] Pump stopped")

        elif action.name == "prime_pump":
            # Priming: run at low flow to clear air
            self.target_flow = 5.0
            self.running = True
            self.status = "running"
            print(f"[{self.name}] Priming pump at 5 mL/min")

        elif action.name == "reduce_flow":
            # Reduce to percentage of current
            factor = action.params.get("factor", 0.5)
            self.target_flow = self.target_flow * factor
            print(f"[{self.name}] Reducing flow to {self.target_flow:.1f} mL/min")

        elif action.name == "wait":
            duration = action.params.get("duration", 0)
            print(f"[{self.name}] Waiting {duration}s")

        else:
            print(f"[{self.name}] Unknown action: {action.name}")

    def health(self) -> bool:
        return self.status != "error"

    def tick(self, dt: float = 1.0):
        self.tick_count += 1
        print(f"  [SimPump] Tick={self.tick_count} Flow={self.flow_rate:.1f} P={self.pressure:.1f}")
