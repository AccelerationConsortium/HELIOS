import random
import time
from typing import Literal, Optional
from ..base import Device
from ...core.types import DeviceState, Action, HardwareError

class SimHeater(Device):
    def __init__(self, name: str, fault_mode: Literal["none", "random", "timeout", "overshoot", "sensor_fail"] = "none"):
        super().__init__(name)
        self.fault_mode = fault_mode
        self.current_temp = 25.0
        self.target_temp = 25.0
        self.heating = False
        self.status: Literal["idle", "running", "error"] = "idle"
        self.max_safe_temp = 130.0
        self.tick_count = 0
        self.last_update_time = time.time()
        self.heating_start_time = 0.0

    def _update_physics(self):
        now = time.time()
        dt = now - self.last_update_time
        self.last_update_time = now
        
        # Speed up simulation: 1s real = 10s sim? 
        # Or just keep 1:1. 1:1 is fine if we wait 5s.
        # Let's make it 2x faster to be snappy.
        dt = dt * 2.0 

        if self.heating:
            if self.heating_start_time == 0.0:
                 self.heating_start_time = now
            
            # Simple approach to target
            diff = self.target_temp - self.current_temp
            # Rate: 5 degrees per second (fast heater!)
            rate = 10.0 
            if diff > 0:
                change = rate * dt
                if change > diff: change = diff
                self.current_temp += change
            elif diff < 0:
                change = -rate * dt
                if change < diff: change = diff
                self.current_temp += change
            
            # Time-based Fault Injection
            duration = now - self.heating_start_time # This is real time duration
            
            # Physics-based Overshoot Fault (injected)
            # Trigger after 2 seconds of heating
            if self.fault_mode == "overshoot" and duration > 2.0:
                 # Force drift upwards
                 self.current_temp += 20.0 * dt

            # Timeout Fault 
            if self.fault_mode == "timeout" and duration > 2.0:
                if self.fault_mode == "timeout":
                     # Undo progress or drift back
                     self.current_temp -= change * 0.9 

        else:
            self.heating_start_time = 0.0 # Reset
            # Cooling to ambient 25.0
            diff = 25.0 - self.current_temp
            rate = 2.0
            if abs(diff) > 0.1:
                direction = 1.0 if diff > 0 else -1.0
                self.current_temp += direction * rate * dt

    def read_state(self) -> DeviceState:
        # Update physics lazily on read
        self._update_physics()

        # Simulate sensor failure
        if self.fault_mode == "sensor_fail": 
             # Only trigger after some time
             if self.tick_count > 5:
                raise HardwareError(
                    device=self.name,
                    type="sensor_fail",
                    severity="high",
                    message="Temperature sensor reading failed (got -999)",
                    when=str(time.time()),
                    context={"status": self.status, "tick": self.tick_count}
                )
        
        # Check Safety Overshoot implicitly during read (sensor monitoring)
        if self.current_temp > self.max_safe_temp:
            self.status = "error"
            # We used to raise here, but that blocks recovery actions (like cool_down).
            # We return the state, and let the Policy/Executor decide to raise 
            # if we are in a strict monitoring phase.
            # raise HardwareError(...) 


        return DeviceState(
            name=self.name,
            status=self.status,
            telemetry={
                "temperature": self.current_temp,
                "target": self.target_temp,
                "heating": self.heating
            }
        )

    def execute(self, action: Action) -> None:
        self._update_physics()
        if action.name == "set_temperature":
            target = action.params.get("temperature", 25.0)
            self.target_temp = target
            self.heating = True
            self.status = "running"
            print(f"[{self.name}] Setting target to {target}")
            
        elif action.name == "wait":
            duration = action.params.get("duration", 0)
            print(f"[{self.name}] Waiting for {duration} seconds... (Simulated by sleep in logic)")
            # In real device, 'wait' might be a sleep command sent to firmware.
            # Here we just acknowledge.
            pass
            
        elif action.name == "cool_down":
            self.target_temp = 25.0
            self.heating = False
            self.status = "idle"
            print(f"[{self.name}] Cooling down to 25.0")
            
        else:
            print(f"[{self.name}] Unknown action: {action.name}")

    def health(self) -> bool:
        return True

    def tick(self, dt: float = 1.0):
        """External tick mostly for step counting."""
        self.tick_count += 1
        print(f"  [SimHeater] Tick={self.tick_count} T={self.current_temp:.1f} (Target={self.target_temp:.1f})")
