"""Deterministic electrochemistry surrogate for research-grade dry runs.

The surrogate is intentionally simple, but it has the properties a benchmark
needs: a known optimum, seeded noise, smooth structure, interaction terms, and
fault modes that produce plausible LSV/EIS artifacts.
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ElectrochemSurrogateConfig:
    seed: int = 0
    noise_std_mv: float = 2.0
    fault_profile: str = "none"  # none | sensor_drift | dropout | fouling


@dataclass(frozen=True)
class ElectrochemObservation:
    overpotential_mv: float
    lsv_curve: dict[str, list[float]]
    eis_summary: dict[str, float]
    fault_flags: list[str] = field(default_factory=list)

    def to_step_result(self) -> dict[str, Any]:
        return {
            "ok": True,
            "overpotential_mv": self.overpotential_mv,
            "simulated_kpi": self.overpotential_mv,
            "lsv_curve": self.lsv_curve,
            "eis_summary": self.eis_summary,
            "fault_flags": list(self.fault_flags),
            "oracle": "electrochem_surrogate",
        }


OPTIMAL_OVERPOTENTIAL_MV = 180.0


def _stable_rng(params: dict[str, Any], seed: int) -> random.Random:
    items = ",".join(f"{k}={params[k]}" for k in sorted(params))
    digest = hashlib.sha256(f"{seed}|{items}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _float_param(params: dict[str, Any], name: str, default: float) -> float:
    try:
        return float(params.get(name, default))
    except (TypeError, ValueError):
        return default


def evaluate_electrochem_candidate(
    params: dict[str, Any],
    *,
    config: ElectrochemSurrogateConfig | None = None,
) -> ElectrochemObservation:
    """Evaluate one electrochemical candidate.

    Lower ``overpotential_mv`` is better. The optimum is near:
    catalyst_conc_mm=50, deposition_time_s=90, electrolyte_ph=13,
    scan_rate_mv_s=20, anneal_temp_c=60.
    """
    cfg = config or ElectrochemSurrogateConfig()
    rng = _stable_rng(params, cfg.seed)

    conc = _float_param(params, "catalyst_conc_mm", 50.0)
    dep_time = _float_param(params, "deposition_time_s", 90.0)
    ph = _float_param(params, "electrolyte_ph", 13.0)
    scan_rate = _float_param(params, "scan_rate_mv_s", 20.0)
    anneal_temp = _float_param(params, "anneal_temp_c", 60.0)

    # Smooth response surface with interaction terms. The coefficients keep
    # the benchmark numerically stable while still penalising poor protocols.
    overpotential = OPTIMAL_OVERPOTENTIAL_MV
    overpotential += 0.030 * (conc - 50.0) ** 2
    overpotential += 0.004 * (dep_time - 90.0) ** 2
    overpotential += 7.0 * (ph - 13.0) ** 2
    overpotential += 0.015 * (scan_rate - 20.0) ** 2
    overpotential += 0.002 * (anneal_temp - 60.0) ** 2
    overpotential += 0.006 * abs(conc - 50.0) * abs(ph - 13.0)
    overpotential += 0.002 * max(0.0, dep_time - 115.0) * max(0.0, conc - 60.0)

    fault_flags: list[str] = []
    if cfg.fault_profile == "sensor_drift":
        drift = 0.18 * (dep_time / 120.0) * scan_rate
        overpotential += drift
        fault_flags.append("sensor_drift")
    elif cfg.fault_profile == "fouling":
        fouling = 12.0 if conc > 70.0 and dep_time > 100.0 else 4.0
        overpotential += fouling
        fault_flags.append("electrode_fouling")
    elif cfg.fault_profile == "dropout":
        # Dropout degrades the observable without making the run unusable.
        overpotential += 18.0
        fault_flags.append("partial_measurement_dropout")

    if cfg.noise_std_mv > 0:
        overpotential += rng.gauss(0.0, cfg.noise_std_mv)
    overpotential = round(max(0.0, overpotential), 6)

    potentials = [round(-0.05 + i * 0.025, 6) for i in range(25)]
    onset_v = overpotential / 1000.0
    current_ma = []
    dropout_cut = 17 if cfg.fault_profile == "dropout" else len(potentials)
    for idx, potential in enumerate(potentials):
        if idx >= dropout_cut:
            current_ma.append(float("nan"))
            continue
        signal = 0.02 * (math.exp(max(0.0, potential - onset_v) / 0.055) - 1.0)
        signal += rng.gauss(0.0, 0.002)
        current_ma.append(round(signal, 8))

    r_ct = 35.0 + max(0.0, overpotential - OPTIMAL_OVERPOTENTIAL_MV) * 0.45
    r_sol = 8.0 + 0.08 * abs(ph - 13.0)
    c_dl = 1.5e-5 + max(0.0, 70.0 - abs(conc - 50.0)) * 2e-8

    return ElectrochemObservation(
        overpotential_mv=overpotential,
        lsv_curve={
            "potential_v": potentials,
            "current_ma": current_ma,
        },
        eis_summary={
            "r_solution_ohm": round(r_sol, 6),
            "r_charge_transfer_ohm": round(r_ct, 6),
            "c_double_layer_f": round(c_dl, 10),
        },
        fault_flags=fault_flags,
    )


def electrochem_search_dimensions() -> list[dict[str, Any]]:
    """Canonical HELIOS electrochem optimization space."""
    return [
        {
            "param_name": "catalyst_conc_mm",
            "param_type": "number",
            "min_value": 10.0,
            "max_value": 90.0,
            "primitive": "robot.dispense",
        },
        {
            "param_name": "deposition_time_s",
            "param_type": "number",
            "min_value": 30.0,
            "max_value": 150.0,
            "primitive": "wait",
        },
        {
            "param_name": "electrolyte_ph",
            "param_type": "number",
            "min_value": 11.0,
            "max_value": 14.0,
            "primitive": "robot.dispense",
        },
        {
            "param_name": "scan_rate_mv_s",
            "param_type": "number",
            "min_value": 5.0,
            "max_value": 80.0,
            "primitive": "squidstat.run_experiment",
        },
        {
            "param_name": "anneal_temp_c",
            "param_type": "number",
            "min_value": 25.0,
            "max_value": 95.0,
            "primitive": "heat",
        },
    ]


def electrochem_protocol_template() -> dict[str, Any]:
    """Small protocol template that exercises compile/safety/simulation stages."""
    return {
        "name": "electrochem_closed_loop_candidate",
        "steps": [
            {"step_key": "s0", "primitive": "robot.home", "params": {}},
            {
                "step_key": "s1",
                "primitive": "robot.load_pipettes",
                "params": {"pipettes": ["left"]},
                "depends_on": ["s0"],
            },
            {
                "step_key": "s2",
                "primitive": "robot.load_labware",
                "params": {"labware": "plate1", "slot": "1"},
                "depends_on": ["s0"],
            },
            {
                "step_key": "s3",
                "primitive": "robot.pick_up_tip",
                "params": {"pipette": "left"},
                "depends_on": ["s1", "s2"],
            },
            {
                "step_key": "s4",
                "primitive": "robot.dispense",
                "params": {"pipette": "left", "volume_ul": 80.0, "labware": "plate1", "well": "A1"},
                "depends_on": ["s3"],
            },
            {
                "step_key": "s5",
                "primitive": "robot.drop_tip",
                "params": {"pipette": "left"},
                "depends_on": ["s4"],
            },
            {
                "step_key": "s6",
                "primitive": "wait",
                "params": {"seconds": 1.0},
                "depends_on": ["s5"],
            },
            {
                "step_key": "s7",
                "primitive": "squidstat.run_experiment",
                "params": {"channel": "0", "technique": "lsv"},
                "depends_on": ["s6"],
            },
        ],
    }
