"""SimLab — Behavior-level simulator for offline benchmarking.

Provides deterministic, seed-controlled simulation of lab primitives
so that the full agent stack (metrics, reviewer, candidate gen, evolution)
can be exercised without real hardware.

Key classes:
- SimWorld: mutable world state with seeded RNG
- SimAdapter: primitive → synthetic result mapping
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# SimWorld — deterministic world state
# ---------------------------------------------------------------------------


@dataclass
class SimWorld:
    """Deterministic lab world state for simulation.

    All randomness goes through ``self.rng`` for reproducibility.
    """

    seed: int = 42
    rng: random.Random = field(init=False)

    # Labware on deck: name → True
    labware: dict[str, bool] = field(default_factory=dict)

    # Tip state per pipette: pipette_name → "on" | "off"
    tips: dict[str, str] = field(default_factory=dict)

    # Current volume in each pipette: pipette_name → µL
    volumes: dict[str, float] = field(default_factory=dict)

    # Whether robot has been homed
    robot_homed: bool = False

    # Pipettes loaded
    pipettes_loaded: bool = False

    # Temperatures: channel → °C
    temps: dict[str, float] = field(default_factory=lambda: {"default": 25.0})

    # Well volumes: labware_name → well_name → µL
    well_volumes: dict[str, dict[str, float]] = field(default_factory=dict)

    # Experiment running per channel
    experiment_running: dict[str, bool] = field(default_factory=dict)

    # Track execution history for auditing
    step_count: int = 0

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def reset(self, seed: int | None = None) -> None:
        """Reset world to initial state with optional new seed."""
        if seed is not None:
            self.seed = seed
        self.rng = random.Random(self.seed)
        self.labware.clear()
        self.tips.clear()
        self.volumes.clear()
        self.robot_homed = False
        self.pipettes_loaded = False
        self.temps = {"default": 25.0}
        self.well_volumes.clear()
        self.experiment_running.clear()
        self.step_count = 0


# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------


def add_noise(value: float, noise_pct: float, rng: random.Random) -> float:
    """Add Gaussian noise scaled by noise_pct of the value.

    noise_pct=0.02 means ±2% noise at 1σ.
    """
    if noise_pct <= 0.0 or value == 0.0:
        return value
    sigma = abs(value) * noise_pct
    return value + rng.gauss(0.0, sigma)


# ---------------------------------------------------------------------------
# SimAdapter — primitive execution simulator
# ---------------------------------------------------------------------------


class SimAdapter:
    """Maps primitive + params → synthetic result dict.

    Simulates lab primitive execution by updating SimWorld state and
    returning result dicts compatible with KPI extractors.

    Args:
        world: SimWorld instance for state tracking.
        noise_pct: Gaussian noise as fraction of target (0.02 = 2%).
        fault_injector: Optional FaultInjector for error injection.
    """

    def __init__(
        self,
        world: SimWorld,
        noise_pct: float = 0.02,
        fault_injector: Any | None = None,
    ) -> None:
        self.world = world
        self.noise_pct = noise_pct
        self.fault_injector = fault_injector

    def execute(self, primitive: str, params: dict[str, Any]) -> dict[str, Any]:
        """Execute a primitive in the simulated world.

        Raises on fault injection; otherwise returns result dict.
        """
        # Check fault injection first
        if self.fault_injector is not None:
            fault = self.fault_injector.maybe_inject(primitive, params, self.world)
            if fault is not None:
                raise fault

        self.world.step_count += 1
        return self._simulate(primitive, params)

    def _simulate(self, primitive: str, params: dict[str, Any]) -> dict[str, Any]:
        """Route primitive to specific simulation handler."""
        # Robot primitives
        if primitive == "robot.home":
            return self._sim_robot_home(params)
        if primitive == "robot.load_pipettes":
            return self._sim_load_pipettes(params)
        if primitive == "robot.load_labware":
            return self._sim_load_labware(params)
        if primitive == "robot.load_custom_labware":
            return self._sim_load_labware(params)
        if primitive == "robot.pick_up_tip":
            return self._sim_pick_up_tip(params)
        if primitive == "robot.drop_tip":
            return self._sim_drop_tip(params)
        if primitive == "robot.aspirate":
            return self._sim_aspirate(params)
        if primitive == "robot.dispense":
            return self._sim_dispense(params)
        if primitive == "robot.blowout":
            return self._sim_blowout(params)
        if primitive == "robot.move_to_well":
            return self._sim_move_to_well(params)
        if primitive == "robot.set_lights":
            return {"status": "ok"}

        # Original OTbot primitives
        if primitive == "aspirate":
            return self._sim_aspirate(params)

        # Heat
        if primitive == "heat":
            return self._sim_heat(params)

        # EIS / squidstat
        if primitive == "eis":
            return self._sim_eis(params)
        if primitive == "squidstat.run_experiment":
            return self._sim_eis(params)
        if primitive == "squidstat.get_data":
            return self._sim_squidstat_get_data(params)
        if primitive == "squidstat.save_snapshot":
            return {"status": "ok", "snapshot_saved": True}
        if primitive == "squidstat.reset_plot":
            return {"status": "ok"}

        # PLC
        if primitive == "plc.dispense_ml":
            return self._sim_plc_dispense(params)
        if primitive == "plc.set_pump_on_timer":
            return {"status": "ok", "duration_s": params.get("duration_s", 10.0)}
        if primitive == "plc.set_ultrasonic_on_timer":
            return {"status": "ok", "duration_s": params.get("duration_s", 5.0)}

        # Relay
        if primitive in ("relay.set_channel", "relay.turn_on",
                         "relay.turn_off", "relay.switch_to"):
            return self._sim_relay(primitive, params)

        # Cleanup
        if primitive == "cleanup.run_full":
            return {"status": "ok", "cleaned": True}

        # Sample
        if primitive == "sample.prepare_from_csv":
            return {"status": "ok", "samples_prepared": params.get("count", 1)}

        # SSH
        if primitive in ("ssh.start_stream", "ssh.stop_stream"):
            return {"status": "ok"}

        # Utility
        if primitive == "wait":
            return {"status": "ok", "waited_s": params.get("seconds", 1.0)}
        if primitive == "log":
            return {"status": "ok"}
        if primitive == "upload_artifact":
            return {"status": "ok", "artifact_id": f"sim-artifact-{self.world.step_count}"}

        # Default fallback
        return {"status": "ok"}

    # -- Robot simulation handlers --

    def _sim_robot_home(self, params: dict) -> dict:
        self.world.robot_homed = True
        return {"status": "ok", "homed": True}

    def _sim_load_pipettes(self, params: dict) -> dict:
        self.world.pipettes_loaded = True
        pipettes = params.get("pipettes", ["left"])
        for p in pipettes if isinstance(pipettes, list) else [pipettes]:
            self.world.tips.setdefault(p, "off")
            self.world.volumes.setdefault(p, 0.0)
        return {"status": "ok", "pipettes_loaded": True}

    def _sim_load_labware(self, params: dict) -> dict:
        name = params.get("labware", params.get("name", "plate1"))
        slot = params.get("slot", "1")
        self.world.labware[name] = True
        # Initialize well volumes if not present
        if name not in self.world.well_volumes:
            self.world.well_volumes[name] = {}
        return {"status": "ok", "labware": name, "slot": slot}

    def _sim_pick_up_tip(self, params: dict) -> dict:
        pipette = params.get("pipette", "left")
        self.world.tips[pipette] = "on"
        return {"status": "ok", "pipette": pipette, "tip": "on"}

    def _sim_drop_tip(self, params: dict) -> dict:
        pipette = params.get("pipette", "left")
        self.world.tips[pipette] = "off"
        self.world.volumes[pipette] = 0.0
        return {"status": "ok", "pipette": pipette, "tip": "off"}

    def _sim_aspirate(self, params: dict) -> dict:
        pipette = params.get("pipette", "left")
        volume = float(params.get("volume_ul", params.get("volume", 100.0)))
        labware = params.get("labware", "plate1")
        well = params.get("well", "A1")

        measured = add_noise(volume, self.noise_pct, self.world.rng)
        self.world.volumes[pipette] = self.world.volumes.get(pipette, 0.0) + measured

        # Decrease well volume
        if labware in self.world.well_volumes:
            cur = self.world.well_volumes[labware].get(well, 1000.0)
            self.world.well_volumes[labware][well] = cur - measured

        return {
            "status": "ok",
            "requested_volume_ul": volume,
            "measured_volume_ul": measured,
            "pipette": pipette,
            "labware": labware,
            "well": well,
        }

    def _sim_dispense(self, params: dict) -> dict:
        pipette = params.get("pipette", "left")
        volume = float(params.get("volume_ul", params.get("volume", 100.0)))
        labware = params.get("labware", "plate1")
        well = params.get("well", "A1")

        measured = add_noise(volume, self.noise_pct, self.world.rng)
        self.world.volumes[pipette] = max(
            0.0, self.world.volumes.get(pipette, 0.0) - measured
        )

        # Increase well volume
        if labware in self.world.well_volumes:
            cur = self.world.well_volumes[labware].get(well, 0.0)
            self.world.well_volumes[labware][well] = cur + measured

        return {
            "status": "ok",
            "requested_volume_ul": volume,
            "measured_volume_ul": measured,
            "pipette": pipette,
            "labware": labware,
            "well": well,
        }

    def _sim_blowout(self, params: dict) -> dict:
        pipette = params.get("pipette", "left")
        self.world.volumes[pipette] = 0.0
        return {"status": "ok", "pipette": pipette}

    def _sim_move_to_well(self, params: dict) -> dict:
        return {
            "status": "ok",
            "labware": params.get("labware", "plate1"),
            "well": params.get("well", "A1"),
        }

    # -- Heat simulation --

    def _sim_heat(self, params: dict) -> dict:
        target = float(params.get("target_temp_c", params.get("temperature", 37.0)))
        channel = params.get("channel", "default")
        measured = add_noise(target, self.noise_pct, self.world.rng)
        self.world.temps[channel] = measured
        return {
            "status": "ok",
            "target_temp_c": target,
            "measured_temp_c": measured,
            "channel": channel,
        }

    # -- EIS / Squidstat simulation --

    def _sim_eis(self, params: dict) -> dict:
        channel = params.get("channel", "0")
        # Synthetic impedance: base value ± noise
        base_impedance = float(params.get("expected_impedance", 1000.0))
        measured = add_noise(base_impedance, self.noise_pct, self.world.rng)
        self.world.experiment_running[channel] = False
        return {
            "status": "ok",
            "impedance_ohm": measured,
            "channel": channel,
        }

    def _sim_squidstat_get_data(self, params: dict) -> dict:
        channel = params.get("channel", "0")
        return {
            "status": "ok",
            "data_points": 100,
            "channel": channel,
        }

    # -- PLC simulation --

    def _sim_plc_dispense(self, params: dict) -> dict:
        volume_ml = float(params.get("volume_ml", 1.0))
        measured = add_noise(volume_ml, self.noise_pct, self.world.rng)
        return {
            "status": "ok",
            "requested_volume_ml": volume_ml,
            "measured_volume_ml": measured,
        }

    # -- Relay simulation --

    def _sim_relay(self, primitive: str, params: dict) -> dict:
        channel = params.get("channel", 0)
        if isinstance(channel, str):
            try:
                channel = int(channel)
            except ValueError:
                channel = 0
        if primitive == "relay.set_channel":
            self.world.experiment_running[str(channel)] = False
        elif primitive == "relay.turn_on":
            self.world.experiment_running[str(channel)] = True
        elif primitive == "relay.turn_off":
            self.world.experiment_running[str(channel)] = False
        return {"status": "ok", "channel": channel}
