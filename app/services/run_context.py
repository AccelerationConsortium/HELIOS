"""RunContext — Per-run state tracker for Action Contract evaluation.

Tracks mutable lab state throughout a single protocol run:
- Labware registry (loaded/unloaded)
- Tip state per pipette (on/off)
- Pipette volumes
- Robot homed status
- Experiment running state per channel
- SSH streaming state
- Active relay channel

Lifecycle: created at execute_run() start, destroyed at run end.
Preconditions are evaluated against this state, effects mutate it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunContext:
    """Mutable state tracker for a single protocol run.

    Each field represents a trackable aspect of the lab's current state.
    Precondition predicates query these fields; effect operations mutate them.
    """

    # Labware loaded on the deck: labware_name → True
    labware_loaded: dict[str, bool] = field(default_factory=dict)

    # Tip state per pipette: pipette_name → "on" | "off"
    tip_state: dict[str, str] = field(default_factory=dict)

    # Current volume in each pipette: pipette_name → float (µL)
    pipette_volume: dict[str, float] = field(default_factory=dict)

    # Whether pipettes have been initialized
    pipettes_loaded: bool = False

    # Whether the robot has been homed
    robot_homed: bool = False

    # Experiment running state per channel: channel_id → bool
    experiment_running: dict[str, bool] = field(default_factory=dict)

    # SSH streaming state
    ssh_streaming: bool = False

    # Active relay channel (None = no active channel)
    active_relay_channel: int | None = None

    # Well volumes: labware_name → well_name → float (µL)
    well_volume: dict[str, dict[str, float]] = field(default_factory=dict)

    # -----------------------------------------------------------------------
    # Precondition evaluation
    # -----------------------------------------------------------------------

    def check_precondition(self, rendered: str) -> bool:
        """Evaluate a rendered predicate string against current state.

        Supported predicates:
        - "labware_loaded:<name>"    → labware_loaded[name] is True
        - "tip_on:<pipette>"         → tip_state[pipette] == "on"
        - "tip_off:<pipette>"        → tip_state[pipette] != "on"
        - "pipettes_loaded"          → pipettes_loaded is True
        - "robot_homed"              → robot_homed is True
        - "experiment_idle:<channel>" → experiment_running[channel] is False
        - "ssh_streaming"            → ssh_streaming is True

        Returns False for unknown predicates (fail-safe).
        """
        parts = rendered.split(":")

        pred = parts[0]

        if pred == "labware_loaded" and len(parts) >= 2:
            return self.labware_loaded.get(parts[1], False)

        if pred == "tip_on" and len(parts) >= 2:
            return self.tip_state.get(parts[1]) == "on"

        if pred == "tip_off" and len(parts) >= 2:
            return self.tip_state.get(parts[1], "off") != "on"

        if pred == "pipettes_loaded":
            return self.pipettes_loaded

        if pred == "robot_homed":
            return self.robot_homed

        if pred == "experiment_idle" and len(parts) >= 2:
            return not self.experiment_running.get(parts[1], False)

        if pred == "ssh_streaming":
            return self.ssh_streaming

        # Unknown predicate → fail safe
        return False

    # -----------------------------------------------------------------------
    # Effect application
    # -----------------------------------------------------------------------

    def apply_effect(self, rendered: str) -> None:
        """Apply a rendered effect operation to this context.

        Supported operations:
        - "set:key:value"                    → setattr/dict set
        - "set:key:subkey:value"             → nested dict set
        - "increase:key:subkey:amount"       → numeric increase
        - "decrease:key:subkey:amount"       → numeric decrease

        Key routing:
        - "labware_loaded" → self.labware_loaded
        - "tip_state" / "tip_on"  → self.tip_state
        - "pipette_volume" → self.pipette_volume
        - "pipettes_loaded" → self.pipettes_loaded (bool)
        - "robot_homed" → self.robot_homed (bool)
        - "experiment_running" → self.experiment_running
        - "ssh_streaming" → self.ssh_streaming (bool)
        - "active_relay_channel" → self.active_relay_channel (int|None)
        - "well_volume" → self.well_volume
        """
        parts = rendered.split(":")
        if len(parts) < 3:
            return

        op = parts[0]

        if op == "set":
            self._apply_set(parts[1:])
        elif op == "increase":
            self._apply_numeric(parts[1:], increase=True)
        elif op == "decrease":
            self._apply_numeric(parts[1:], increase=False)

    def _apply_set(self, parts: list[str]) -> None:
        """Handle 'set' operations.

        parts is everything after 'set:', e.g.:
        - ["robot_homed", "true"]
        - ["labware_loaded", "plate1", "true"]
        - ["tip_state", "left", "on"]
        - ["pipette_volume", "left", "0"]
        - ["active_relay_channel", "3"]
        """
        if len(parts) < 2:
            return

        key = parts[0]
        value_str = parts[-1]
        value = self._coerce(value_str)

        # Simple boolean/scalar fields
        if key == "robot_homed":
            self.robot_homed = bool(value)
            return
        if key == "pipettes_loaded":
            self.pipettes_loaded = bool(value)
            return
        if key == "ssh_streaming":
            self.ssh_streaming = bool(value)
            return
        if key == "active_relay_channel":
            if value_str.lower() == "none":
                self.active_relay_channel = None
            else:
                self.active_relay_channel = int(float(value_str))
            return

        # Dict fields with subkey
        if key == "labware_loaded" and len(parts) >= 3:
            self.labware_loaded[parts[1]] = bool(value)
            return
        if key in ("tip_state", "tip_on") and len(parts) >= 3:
            # "set:tip_state:left:on" or "set:tip_on:left:true"
            if key == "tip_on":
                self.tip_state[parts[1]] = "on" if value else "off"
            else:
                self.tip_state[parts[1]] = str(value_str)
            return
        if key == "pipette_volume" and len(parts) >= 3:
            self.pipette_volume[parts[1]] = float(value_str)
            return
        if key == "experiment_running" and len(parts) >= 3:
            self.experiment_running[parts[1]] = bool(value)
            return

        # Nested dict fields (e.g., well_volume:plate1:A1:100)
        if key == "well_volume" and len(parts) >= 4:
            labware = parts[1]
            well = parts[2]
            if labware not in self.well_volume:
                self.well_volume[labware] = {}
            self.well_volume[labware][well] = float(parts[3])
            return

    def _apply_numeric(self, parts: list[str], *, increase: bool) -> None:
        """Handle 'increase' / 'decrease' operations.

        parts is everything after 'increase:' or 'decrease:', e.g.:
        - ["pipette_volume", "left", "100"]
        - ["well_volume", "plate1", "A1", "50"]
        """
        if len(parts) < 3:
            return

        key = parts[0]
        amount = float(parts[-1])

        if key == "pipette_volume" and len(parts) >= 3:
            pipette = parts[1]
            current = self.pipette_volume.get(pipette, 0.0)
            if increase:
                self.pipette_volume[pipette] = current + amount
            else:
                self.pipette_volume[pipette] = current - amount
            return

        if key == "well_volume" and len(parts) >= 4:
            labware = parts[1]
            well = parts[2]
            if labware not in self.well_volume:
                self.well_volume[labware] = {}
            current = self.well_volume[labware].get(well, 0.0)
            if increase:
                self.well_volume[labware][well] = current + amount
            else:
                self.well_volume[labware][well] = current - amount
            return

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    @staticmethod
    def _coerce(val: str) -> Any:
        """Convert string value to appropriate Python type."""
        low = val.lower()
        if low == "true":
            return True
        if low == "false":
            return False
        if low == "none":
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return val

    def snapshot(self) -> dict[str, Any]:
        """Return a serializable snapshot of current state.

        Useful for audit logging and debugging.
        """
        return {
            "labware_loaded": dict(self.labware_loaded),
            "tip_state": dict(self.tip_state),
            "pipette_volume": dict(self.pipette_volume),
            "pipettes_loaded": self.pipettes_loaded,
            "robot_homed": self.robot_homed,
            "experiment_running": dict(self.experiment_running),
            "ssh_streaming": self.ssh_streaming,
            "active_relay_channel": self.active_relay_channel,
            "well_volume": {
                k: dict(v) for k, v in self.well_volume.items()
            },
        }
