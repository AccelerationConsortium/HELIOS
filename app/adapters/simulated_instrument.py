"""Simulated instrument adapter for testing without real hardware.

Refactored from the original standalone ``execute_primitive`` function
to implement the ``InstrumentAdapter`` protocol.
"""
from __future__ import annotations

import random
import time
from typing import Any


class SimulatedAdapter:
    """InstrumentAdapter implementation that simulates all primitives.

    Returned measurements include small random noise to mimic real sensors.
    Supports ``force_fail`` param for fault-injection testing.
    """

    def __init__(self) -> None:
        self._connected = False

    # ---- InstrumentAdapter protocol ----

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def execute_primitive(
        self, *, instrument_id: str, primitive: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        if params.get("force_fail"):
            raise RuntimeError(f"step forced failure on primitive={primitive}")

        duration_s = float(params.get("duration_s", 0.2))
        time.sleep(max(0.0, min(duration_s, 2.0)))

        handler = _PRIMITIVE_HANDLERS.get(primitive)
        if handler is None:
            raise RuntimeError(f"unsupported primitive: {primitive}")

        return handler(instrument_id=instrument_id, primitive=primitive, params=params)

    def health_check(self) -> dict[str, Any]:
        return {
            "adapter": "simulated",
            "connected": self._connected,
        }


# ---------------------------------------------------------------------------
# Primitive handlers — pure functions for each simulated primitive
# ---------------------------------------------------------------------------

def _handle_aspirate(*, instrument_id: str, primitive: str, params: dict[str, Any]) -> dict[str, Any]:
    volume = float(params.get("volume_ul", 0))
    return {
        "instrument_id": instrument_id,
        "primitive": primitive,
        "measured_volume_ul": round(volume * random.uniform(0.99, 1.01), 3),
        "ok": True,
    }


def _handle_heat(*, instrument_id: str, primitive: str, params: dict[str, Any]) -> dict[str, Any]:
    temp = float(params.get("temp_c", 25.0))
    return {
        "instrument_id": instrument_id,
        "primitive": primitive,
        "measured_temp_c": round(temp * random.uniform(0.995, 1.005), 3),
        "ok": True,
    }


def _handle_eis(*, instrument_id: str, primitive: str, params: dict[str, Any]) -> dict[str, Any]:
    # Simulated Nyquist spectrum using a simple Randles circuit:
    #   Z = R_sol + R_ct / (1 + j*w*R_ct*C)
    import math

    n_points = 15
    frequencies = [10 ** (i * 0.4) for i in range(n_points)]  # ~1 Hz to ~100 kHz
    r_sol = random.uniform(5, 15)
    r_ct = random.uniform(50, 150)
    c_dl = random.uniform(1e-6, 1e-4)  # double-layer capacitance

    z_real_list: list[float] = []
    z_imag_list: list[float] = []
    for f in frequencies:
        w = 2.0 * math.pi * f
        denom = 1.0 + (w * r_ct * c_dl) ** 2
        z_r = r_sol + r_ct / denom
        z_i = -(w * r_ct**2 * c_dl) / denom
        z_real_list.append(round(z_r, 5))
        z_imag_list.append(round(z_i, 5))

    return {
        "instrument_id": instrument_id,
        "primitive": primitive,
        "impedance_ohm": round(r_sol + r_ct, 5),  # backward compat scalar
        "spectrum": {
            "technique": "eis",
            "frequencies_hz": [round(f, 5) for f in frequencies],
            "z_real": z_real_list,
            "z_imag": z_imag_list,
            "r_sol_ohm": round(r_sol, 5),
            "r_ct_ohm": round(r_ct, 5),
        },
        "ok": True,
    }


def _handle_lsv(*, instrument_id: str, primitive: str, params: dict[str, Any]) -> dict[str, Any]:
    """Simulated Linear Sweep Voltammetry (LSV) curve."""
    import math

    n_points = 20
    e_start = float(params.get("e_start_v", 0.0))
    e_end = float(params.get("e_end_v", 1.0))
    potentials = [e_start + i * (e_end - e_start) / (n_points - 1) for i in range(n_points)]

    # Tafel-like exponential current with noise
    i0 = random.uniform(0.01, 0.1)  # exchange current mA
    b = random.uniform(0.05, 0.15)  # Tafel slope proxy
    currents: list[float] = []
    for e in potentials:
        current = i0 * (math.exp(e / b) - 1.0) + random.gauss(0, 0.005)
        currents.append(round(current, 6))

    return {
        "instrument_id": instrument_id,
        "primitive": primitive,
        "spectrum": {
            "technique": "lsv",
            "potential_v": [round(p, 6) for p in potentials],
            "current_ma": currents,
        },
        "ok": True,
    }


def _handle_wait(*, instrument_id: str, primitive: str, params: dict[str, Any]) -> dict[str, Any]:
    return {"instrument_id": instrument_id, "primitive": primitive, "ok": True}


def _handle_upload_artifact(*, instrument_id: str, primitive: str, params: dict[str, Any]) -> dict[str, Any]:
    return {"instrument_id": instrument_id, "primitive": primitive, "uploaded": True, "ok": True}


# Battery-lab primitives (simulated stubs) --------------------------------

def _handle_generic_ok(*, instrument_id: str, primitive: str, params: dict[str, Any]) -> dict[str, Any]:
    """Catch-all: return ok=True for any battery-lab primitive."""
    return {"instrument_id": instrument_id, "primitive": primitive, "ok": True}


# Mapping of primitive name → handler
_PRIMITIVE_HANDLERS: dict[str, Any] = {
    # Original OTbot primitives
    "aspirate": _handle_aspirate,
    "heat": _handle_heat,
    "eis": _handle_eis,
    "lsv": _handle_lsv,
    "wait": _handle_wait,
    "upload_artifact": _handle_upload_artifact,
    # Battery-lab primitives — all simulated as generic ok
    "robot.home": _handle_generic_ok,
    "robot.load_pipettes": _handle_generic_ok,
    "robot.set_lights": _handle_generic_ok,
    "robot.load_labware": _handle_generic_ok,
    "robot.load_custom_labware": _handle_generic_ok,
    "robot.move_to_well": _handle_generic_ok,
    "robot.pick_up_tip": _handle_generic_ok,
    "robot.drop_tip": _handle_generic_ok,
    "robot.aspirate": _handle_generic_ok,
    "robot.dispense": _handle_generic_ok,
    "robot.blowout": _handle_generic_ok,
    "plc.dispense_ml": _handle_generic_ok,
    "plc.set_pump_on_timer": _handle_generic_ok,
    "plc.set_ultrasonic_on_timer": _handle_generic_ok,
    "relay.set_channel": _handle_generic_ok,
    "relay.turn_on": _handle_generic_ok,
    "relay.turn_off": _handle_generic_ok,
    "relay.switch_to": _handle_generic_ok,
    "squidstat.run_experiment": _handle_generic_ok,
    "squidstat.get_data": _handle_generic_ok,
    "squidstat.save_snapshot": _handle_generic_ok,
    "squidstat.reset_plot": _handle_generic_ok,
    "cleanup.run_full": _handle_generic_ok,
    "sample.prepare_from_csv": _handle_generic_ok,
    "ssh.start_stream": _handle_generic_ok,
    "ssh.stop_stream": _handle_generic_ok,
    "log": _handle_generic_ok,
}


# ---------------------------------------------------------------------------
# Backward-compatible free function (used by existing tests / worker.py)
# ---------------------------------------------------------------------------

_DEFAULT_ADAPTER = SimulatedAdapter()


def execute_primitive(
    *, instrument_id: str, primitive: str, params: dict[str, Any]
) -> dict[str, Any]:
    """Legacy wrapper — delegates to the module-level SimulatedAdapter."""
    return _DEFAULT_ADAPTER.execute_primitive(
        instrument_id=instrument_id, primitive=primitive, params=params,
    )
