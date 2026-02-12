"""
SimPositioner - Simulated XYZ stage/positioner for SDL fault recovery testing.

Fault modes:
- none: Normal operation
- collision: Hard stop detected (hit limit or obstacle)
- position_drift: Position drifts from target over time
- motor_stall: Motor stops responding (stall signature)
- limit_exceeded: Soft limit violation
- encoder_error: Position feedback becomes unreliable
"""
import time
import math
from typing import Literal, Optional, Dict
from ..base import Device
from ...core.types import DeviceState, Action, HardwareError


class SimPositioner(Device):
    def __init__(
        self,
        name: str,
        fault_mode: Literal["none", "collision", "position_drift", "motor_stall", "limit_exceeded", "encoder_error"] = "none"
    ):
        super().__init__(name)
        self.fault_mode = fault_mode

        # Position state (in mm)
        self.position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.target = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.velocity = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.moving = False
        self.status: Literal["idle", "moving", "error", "homed"] = "idle"

        # Limits (mm)
        self.limits = {
            "x": (-50.0, 50.0),
            "y": (-50.0, 50.0),
            "z": (0.0, 100.0)  # Z typically 0 to positive
        }

        # Speed settings
        self.max_speed = 10.0  # mm/s
        self.acceleration = 20.0  # mm/s²

        # Simulation state
        self.tick_count = 0
        self.last_update_time = time.time()
        self.move_start_time = 0.0
        self.collision_detected = False

    def _update_physics(self):
        now = time.time()
        dt = (now - self.last_update_time) * 2.0  # 2x speed
        self.last_update_time = now

        if self.moving:
            if self.move_start_time == 0.0:
                self.move_start_time = now

            duration = now - self.move_start_time
            all_reached = True

            for axis in ["x", "y", "z"]:
                diff = self.target[axis] - self.position[axis]
                if abs(diff) > 0.01:
                    all_reached = False
                    # Simple motion: accelerate then constant speed
                    direction = 1.0 if diff > 0 else -1.0
                    speed = min(self.max_speed, abs(diff) * 2.0)
                    move = direction * speed * dt
                    if abs(move) > abs(diff):
                        move = diff
                    self.position[axis] += move
                    self.velocity[axis] = speed * direction
                else:
                    self.velocity[axis] = 0.0

            # Fault injection
            if self.fault_mode == "collision" and duration > 1.0:
                # Simulate hitting an obstacle
                self.collision_detected = True
                self.moving = False
                self.velocity = {"x": 0.0, "y": 0.0, "z": 0.0}
                self.status = "error"

            elif self.fault_mode == "position_drift" and duration > 1.5:
                # Position drifts away from target
                for axis in ["x", "y"]:
                    self.position[axis] += 0.5 * dt  # Slow drift

            elif self.fault_mode == "motor_stall" and duration > 1.5:
                # Motor stops, position frozen
                self.velocity = {"x": 0.0, "y": 0.0, "z": 0.0}
                # Position doesn't change

            elif self.fault_mode == "encoder_error" and duration > 1.5:
                # Noisy position readings
                import random
                for axis in ["x", "y", "z"]:
                    self.position[axis] += random.gauss(0, 0.2)

            if all_reached and not self.collision_detected:
                self.moving = False
                self.status = "idle"
                self.velocity = {"x": 0.0, "y": 0.0, "z": 0.0}
        else:
            self.move_start_time = 0.0
            self.velocity = {"x": 0.0, "y": 0.0, "z": 0.0}

    def _check_limits(self) -> Optional[str]:
        """Check if any axis exceeds limits."""
        for axis, (low, high) in self.limits.items():
            if self.position[axis] < low or self.position[axis] > high:
                return axis
        return None

    def read_state(self) -> DeviceState:
        self._update_physics()

        # Check for collision fault
        if self.collision_detected:
            raise HardwareError(
                device=self.name,
                type="collision",
                severity="critical",
                message=f"Collision detected at position ({self.position['x']:.2f}, {self.position['y']:.2f}, {self.position['z']:.2f})",
                when=str(time.time()),
                context={"position": self.position.copy(), "target": self.target.copy()}
            )

        # Check for limit exceeded
        if self.fault_mode == "limit_exceeded":
            exceeded = self._check_limits()
            if exceeded and self.tick_count > 3:
                raise HardwareError(
                    device=self.name,
                    type="limit_exceeded",
                    severity="high",
                    message=f"Soft limit exceeded on {exceeded} axis: {self.position[exceeded]:.2f}",
                    when=str(time.time()),
                    context={"axis": exceeded, "position": self.position[exceeded]}
                )

        # Check for motor stall (no movement despite command)
        if self.fault_mode == "motor_stall" and self.moving and self.tick_count > 5:
            total_velocity = sum(abs(v) for v in self.velocity.values())
            if total_velocity < 0.01:
                raise HardwareError(
                    device=self.name,
                    type="motor_stall",
                    severity="high",
                    message="Motor stall detected: no movement despite active command",
                    when=str(time.time()),
                    context={"position": self.position.copy(), "target": self.target.copy()}
                )

        return DeviceState(
            name=self.name,
            status=self.status,
            telemetry={
                "x": round(self.position["x"], 3),
                "y": round(self.position["y"], 3),
                "z": round(self.position["z"], 3),
                "target_x": self.target["x"],
                "target_y": self.target["y"],
                "target_z": self.target["z"],
                "velocity_x": round(self.velocity["x"], 3),
                "velocity_y": round(self.velocity["y"], 3),
                "velocity_z": round(self.velocity["z"], 3),
                "moving": self.moving
            }
        )

    def execute(self, action: Action) -> None:
        self._update_physics()

        if action.name == "move_to":
            x = action.params.get("x", self.position["x"])
            y = action.params.get("y", self.position["y"])
            z = action.params.get("z", self.position["z"])
            self.target = {"x": x, "y": y, "z": z}
            self.moving = True
            self.status = "moving"
            self.collision_detected = False
            print(f"[{self.name}] Moving to ({x}, {y}, {z})")

        elif action.name == "move_relative":
            dx = action.params.get("dx", 0.0)
            dy = action.params.get("dy", 0.0)
            dz = action.params.get("dz", 0.0)
            self.target["x"] = self.position["x"] + dx
            self.target["y"] = self.position["y"] + dy
            self.target["z"] = self.position["z"] + dz
            self.moving = True
            self.status = "moving"
            print(f"[{self.name}] Moving relative ({dx}, {dy}, {dz})")

        elif action.name == "home":
            self.target = {"x": 0.0, "y": 0.0, "z": 0.0}
            self.moving = True
            self.status = "moving"
            print(f"[{self.name}] Homing to origin")

        elif action.name == "stop":
            self.moving = False
            self.target = self.position.copy()
            self.status = "idle"
            print(f"[{self.name}] Emergency stop")

        elif action.name == "retract":
            # Move Z up to safe position
            self.target["z"] = min(self.position["z"] + 10.0, self.limits["z"][1])
            self.moving = True
            self.status = "moving"
            print(f"[{self.name}] Retracting Z to {self.target['z']}")

        elif action.name == "reduce_speed":
            factor = action.params.get("factor", 0.5)
            self.max_speed = self.max_speed * factor
            print(f"[{self.name}] Reducing speed to {self.max_speed:.1f} mm/s")

        elif action.name == "wait":
            duration = action.params.get("duration", 0)
            print(f"[{self.name}] Waiting {duration}s")

        else:
            print(f"[{self.name}] Unknown action: {action.name}")

    def health(self) -> bool:
        return self.status != "error" and not self.collision_detected

    def tick(self, dt: float = 1.0):
        self.tick_count += 1
        pos = f"({self.position['x']:.2f}, {self.position['y']:.2f}, {self.position['z']:.2f})"
        print(f"  [SimPositioner] Tick={self.tick_count} Pos={pos} Moving={self.moving}")
