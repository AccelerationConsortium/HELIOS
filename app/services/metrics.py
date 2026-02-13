"""Run-level KPI extraction and metrics store.

Post-run listener that extracts domain-specific KPIs from completed runs
and stores them in the ``run_kpis`` table with:
- ``kpi_schema_version`` for forward compatibility when definitions change
- ``source_artifact_id`` FK for traceability back to raw measurement data

All operations are advisory — wrapped in try/except, never block
run completion.  Write path is post-run async via event_bus listener.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from app.core.db import connection, json_dumps, parse_json, run_txn, utcnow_iso

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# KPI schema versioning
# ---------------------------------------------------------------------------

KPI_SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class KpiDefinition:
    """Declarative KPI specification."""

    name: str  # e.g. "volume_accuracy_pct"
    unit: str  # e.g. "pct", "celsius", "ohm", "seconds", "ratio", "count"
    scope: str  # "step" | "run"
    primitive: str | None  # None = all steps / run-level
    extractor: str  # function name in the dispatcher dict


@dataclass(frozen=True)
class KpiValue:
    """A single extracted KPI measurement."""

    kpi_name: str
    kpi_value: float | None
    kpi_unit: str
    step_id: str | None
    source_artifact_id: str | None
    details: dict[str, Any]


# ---------------------------------------------------------------------------
# KPI Definitions v1
# ---------------------------------------------------------------------------

KPI_DEFINITIONS_V1: list[KpiDefinition] = [
    KpiDefinition(
        name="volume_accuracy_pct",
        unit="pct",
        scope="step",
        primitive="aspirate",
        extractor="extract_volume_accuracy",
    ),
    KpiDefinition(
        name="temp_accuracy_c",
        unit="celsius",
        scope="step",
        primitive="heat",
        extractor="extract_temp_accuracy",
    ),
    KpiDefinition(
        name="impedance_ohm",
        unit="ohm",
        scope="step",
        primitive="eis",
        extractor="extract_impedance",
    ),
    KpiDefinition(
        name="step_duration_s",
        unit="seconds",
        scope="step",
        primitive=None,  # all steps
        extractor="extract_step_duration",
    ),
    # --- Electrochemistry KPIs ---
    KpiDefinition(
        name="overpotential_mv",
        unit="mV",
        scope="step",
        primitive="squidstat.run_experiment",
        extractor="extract_overpotential",
    ),
    KpiDefinition(
        name="current_density_ma_cm2",
        unit="mA/cm2",
        scope="step",
        primitive="squidstat.run_experiment",
        extractor="extract_current_density",
    ),
    KpiDefinition(
        name="coulombic_efficiency",
        unit="ratio",
        scope="step",
        primitive="squidstat.run_experiment",
        extractor="extract_coulombic_efficiency",
    ),
    KpiDefinition(
        name="stability_decay_pct",
        unit="pct",
        scope="step",
        primitive="squidstat.run_experiment",
        extractor="extract_stability_decay",
    ),
    KpiDefinition(
        name="charge_passed_c",
        unit="C",
        scope="step",
        primitive="squidstat.run_experiment",
        extractor="extract_charge_passed",
    ),
    # --- pH measurement KPIs ---
    KpiDefinition(
        name="ph_value",
        unit="pH",
        scope="step",
        primitive="ph_sensor.read_value",
        extractor="extract_ph_value",
    ),
    KpiDefinition(
        name="ph_std",
        unit="pH",
        scope="step",
        primitive="ph_sensor.read_value",
        extractor="extract_ph_std",
    ),
    KpiDefinition(
        name="delta_ph",
        unit="pH",
        scope="step",
        primitive="ph_sensor.read_value",
        extractor="extract_delta_ph",
    ),
]

KPI_DEFINITIONS_V1_RUN: list[KpiDefinition] = [
    KpiDefinition(
        name="run_success_rate",
        unit="ratio",
        scope="run",
        primitive=None,
        extractor="extract_run_success_rate",
    ),
    KpiDefinition(
        name="run_duration_s",
        unit="seconds",
        scope="run",
        primitive=None,
        extractor="extract_run_duration",
    ),
    KpiDefinition(
        name="recovery_count",
        unit="count",
        scope="run",
        primitive=None,
        extractor="extract_recovery_count",
    ),
]


# ---------------------------------------------------------------------------
# Step-level extractor functions
# ---------------------------------------------------------------------------


def extract_volume_accuracy(
    step: dict[str, Any],
    artifact: dict[str, Any] | None,
    artifact_payload: dict[str, Any] | None,
) -> KpiValue | None:
    """(measured_volume_ul / requested_volume_ul) × 100."""
    if artifact_payload is None:
        return None
    measured = artifact_payload.get("measured_volume_ul")
    if measured is None:
        return None
    params = parse_json(step.get("params_json", "{}"), {})
    requested = params.get("volume_ul") or params.get("volume")
    if not requested or float(requested) == 0:
        return None
    value = (float(measured) / float(requested)) * 100.0
    return KpiValue(
        kpi_name="volume_accuracy_pct",
        kpi_value=round(value, 4),
        kpi_unit="pct",
        step_id=step["id"],
        source_artifact_id=artifact["id"] if artifact else None,
        details={
            "measured_volume_ul": float(measured),
            "requested_volume_ul": float(requested),
        },
    )


def extract_temp_accuracy(
    step: dict[str, Any],
    artifact: dict[str, Any] | None,
    artifact_payload: dict[str, Any] | None,
) -> KpiValue | None:
    """|measured_temp_c - target_temp_c|."""
    if artifact_payload is None:
        return None
    measured = artifact_payload.get("measured_temp_c")
    if measured is None:
        return None
    params = parse_json(step.get("params_json", "{}"), {})
    target = params.get("temp_c")
    if target is None:
        return None
    value = abs(float(measured) - float(target))
    return KpiValue(
        kpi_name="temp_accuracy_c",
        kpi_value=round(value, 4),
        kpi_unit="celsius",
        step_id=step["id"],
        source_artifact_id=artifact["id"] if artifact else None,
        details={
            "measured_temp_c": float(measured),
            "target_temp_c": float(target),
        },
    )


def extract_impedance(
    step: dict[str, Any],
    artifact: dict[str, Any] | None,
    artifact_payload: dict[str, Any] | None,
) -> KpiValue | None:
    """Raw impedance_ohm from EIS step result."""
    if artifact_payload is None:
        return None
    impedance = artifact_payload.get("impedance_ohm")
    if impedance is None:
        return None
    return KpiValue(
        kpi_name="impedance_ohm",
        kpi_value=round(float(impedance), 5),
        kpi_unit="ohm",
        step_id=step["id"],
        source_artifact_id=artifact["id"] if artifact else None,
        details={"impedance_ohm": float(impedance)},
    )


def extract_step_duration(
    step: dict[str, Any],
    artifact: dict[str, Any] | None,
    artifact_payload: dict[str, Any] | None,
) -> KpiValue | None:
    """ended_at - started_at in seconds per step."""
    started = step.get("started_at")
    ended = step.get("ended_at")
    if not started or not ended:
        return None
    try:
        t_start = datetime.fromisoformat(started)
        t_end = datetime.fromisoformat(ended)
        value = (t_end - t_start).total_seconds()
    except (ValueError, TypeError):
        return None
    return KpiValue(
        kpi_name="step_duration_s",
        kpi_value=round(value, 3),
        kpi_unit="seconds",
        step_id=step["id"],
        source_artifact_id=None,
        details={"started_at": started, "ended_at": ended},
    )


# ---------------------------------------------------------------------------
# Electrochemistry step-level extractor functions
# ---------------------------------------------------------------------------


def extract_overpotential(
    step: dict[str, Any],
    artifact: dict[str, Any] | None,
    artifact_payload: dict[str, Any] | None,
) -> KpiValue | None:
    """Overpotential at 10 mA/cm² (η@10mA).

    Reads ``overpotential_mv`` directly from artifact payload, OR computes
    from ``potential_v`` and ``reference_potential_v`` if available.
    Deterministic: same artifact → same KPI value.
    """
    if artifact_payload is None:
        return None
    # Direct field
    eta = artifact_payload.get("overpotential_mv")
    if eta is None:
        # Compute from potential vs reference (E_measured - E_thermodynamic)
        potential = artifact_payload.get("potential_v")
        ref = artifact_payload.get("reference_potential_v", 1.23)  # OER std
        if potential is not None:
            eta = (float(potential) - float(ref)) * 1000.0  # V → mV
    if eta is None:
        return None
    return KpiValue(
        kpi_name="overpotential_mv",
        kpi_value=round(float(eta), 4),
        kpi_unit="mV",
        step_id=step["id"],
        source_artifact_id=artifact["id"] if artifact else None,
        details={"overpotential_mv": float(eta)},
    )


def extract_current_density(
    step: dict[str, Any],
    artifact: dict[str, Any] | None,
    artifact_payload: dict[str, Any] | None,
) -> KpiValue | None:
    """Current density in mA/cm².

    Reads ``current_density_ma_cm2`` directly, OR computes from
    ``current_ma`` / ``electrode_area_cm2``.
    """
    if artifact_payload is None:
        return None
    j = artifact_payload.get("current_density_ma_cm2")
    if j is None:
        current = artifact_payload.get("current_ma")
        area = artifact_payload.get("electrode_area_cm2")
        if current is not None and area is not None and float(area) > 0:
            j = float(current) / float(area)
    if j is None:
        return None
    return KpiValue(
        kpi_name="current_density_ma_cm2",
        kpi_value=round(float(j), 4),
        kpi_unit="mA/cm2",
        step_id=step["id"],
        source_artifact_id=artifact["id"] if artifact else None,
        details={"current_density_ma_cm2": float(j)},
    )


def extract_coulombic_efficiency(
    step: dict[str, Any],
    artifact: dict[str, Any] | None,
    artifact_payload: dict[str, Any] | None,
) -> KpiValue | None:
    """Coulombic efficiency (CE) = charge_out / charge_in.

    Reads ``coulombic_efficiency`` directly, OR computes from
    ``charge_discharge_c`` / ``charge_charge_c``.
    """
    if artifact_payload is None:
        return None
    ce = artifact_payload.get("coulombic_efficiency")
    if ce is None:
        q_out = artifact_payload.get("charge_discharge_c")
        q_in = artifact_payload.get("charge_charge_c")
        if q_out is not None and q_in is not None and float(q_in) > 0:
            ce = float(q_out) / float(q_in)
    if ce is None:
        return None
    return KpiValue(
        kpi_name="coulombic_efficiency",
        kpi_value=round(float(ce), 6),
        kpi_unit="ratio",
        step_id=step["id"],
        source_artifact_id=artifact["id"] if artifact else None,
        details={"coulombic_efficiency": float(ce)},
    )


def extract_stability_decay(
    step: dict[str, Any],
    artifact: dict[str, Any] | None,
    artifact_payload: dict[str, Any] | None,
) -> KpiValue | None:
    """Stability decay percentage over chronoamperometry/chronopotentiometry.

    Reads ``stability_decay_pct`` directly, OR computes from
    ``initial_current_ma`` and ``final_current_ma``:
        decay = (1 - final/initial) × 100
    """
    if artifact_payload is None:
        return None
    decay = artifact_payload.get("stability_decay_pct")
    if decay is None:
        initial = artifact_payload.get("initial_current_ma")
        final = artifact_payload.get("final_current_ma")
        if (
            initial is not None
            and final is not None
            and float(initial) != 0
        ):
            decay = (1.0 - float(final) / float(initial)) * 100.0
    if decay is None:
        return None
    return KpiValue(
        kpi_name="stability_decay_pct",
        kpi_value=round(float(decay), 4),
        kpi_unit="pct",
        step_id=step["id"],
        source_artifact_id=artifact["id"] if artifact else None,
        details={"stability_decay_pct": float(decay)},
    )


def extract_charge_passed(
    step: dict[str, Any],
    artifact: dict[str, Any] | None,
    artifact_payload: dict[str, Any] | None,
) -> KpiValue | None:
    """Total charge passed in Coulombs.

    Reads ``charge_passed_c`` directly, OR computes from
    ``current_ma`` × ``duration_s`` / 1000.
    """
    if artifact_payload is None:
        return None
    charge = artifact_payload.get("charge_passed_c")
    if charge is None:
        current = artifact_payload.get("current_ma")
        duration = artifact_payload.get("duration_s")
        if current is not None and duration is not None:
            charge = float(current) * float(duration) / 1000.0  # mA·s → C
    if charge is None:
        return None
    return KpiValue(
        kpi_name="charge_passed_c",
        kpi_value=round(float(charge), 6),
        kpi_unit="C",
        step_id=step["id"],
        source_artifact_id=artifact["id"] if artifact else None,
        details={"charge_passed_c": float(charge)},
    )


# ---------------------------------------------------------------------------
# pH measurement step-level extractor functions
# ---------------------------------------------------------------------------


def extract_ph_value(
    step: dict[str, Any],
    artifact: dict[str, Any] | None,
    artifact_payload: dict[str, Any] | None,
) -> KpiValue | None:
    """Mean pH reading from colorimetric strip measurement.

    Reads ``ph_mean`` from artifact payload (produced by PhSensorController.read_ph).
    """
    if artifact_payload is None:
        return None
    ph = artifact_payload.get("ph_mean")
    if ph is None:
        return None
    return KpiValue(
        kpi_name="ph_value",
        kpi_value=round(float(ph), 3),
        kpi_unit="pH",
        step_id=step["id"],
        source_artifact_id=artifact["id"] if artifact else None,
        details={
            "ph_mean": float(ph),
            "ph_readings": artifact_payload.get("ph_readings", []),
            "well": artifact_payload.get("well", ""),
            "n_readings": artifact_payload.get("n_readings", 0),
        },
    )


def extract_ph_std(
    step: dict[str, Any],
    artifact: dict[str, Any] | None,
    artifact_payload: dict[str, Any] | None,
) -> KpiValue | None:
    """Standard deviation of pH readings (measurement precision indicator).

    Reads ``ph_std`` from artifact payload.
    """
    if artifact_payload is None:
        return None
    std = artifact_payload.get("ph_std")
    if std is None:
        return None
    return KpiValue(
        kpi_name="ph_std",
        kpi_value=round(float(std), 4),
        kpi_unit="pH",
        step_id=step["id"],
        source_artifact_id=artifact["id"] if artifact else None,
        details={"ph_std": float(std)},
    )


def extract_delta_ph(
    step: dict[str, Any],
    artifact: dict[str, Any] | None,
    artifact_payload: dict[str, Any] | None,
) -> KpiValue | None:
    """Deviation from target pH (|measured - target|).

    Reads ``ph_mean`` from payload and ``target_ph`` from step params.
    Useful for Bayesian optimization objective tracking.
    """
    if artifact_payload is None:
        return None
    measured = artifact_payload.get("ph_mean")
    if measured is None:
        return None
    params = parse_json(step.get("params_json", "{}"), {})
    target = params.get("target_ph")
    if target is None:
        return None
    delta = abs(float(measured) - float(target))
    return KpiValue(
        kpi_name="delta_ph",
        kpi_value=round(delta, 4),
        kpi_unit="pH",
        step_id=step["id"],
        source_artifact_id=artifact["id"] if artifact else None,
        details={
            "measured_ph": float(measured),
            "target_ph": float(target),
            "delta": delta,
        },
    )


# ---------------------------------------------------------------------------
# Run-level extractor functions
# ---------------------------------------------------------------------------


def extract_run_success_rate(
    run: dict[str, Any],
    steps: list[dict[str, Any]],
    conn: Any,
) -> KpiValue | None:
    """succeeded_steps / total_steps."""
    if not steps:
        return None
    succeeded = sum(1 for s in steps if s["status"] == "succeeded")
    total = len(steps)
    return KpiValue(
        kpi_name="run_success_rate",
        kpi_value=round(succeeded / total, 4),
        kpi_unit="ratio",
        step_id=None,
        source_artifact_id=None,
        details={"succeeded": succeeded, "total": total},
    )


def extract_run_duration(
    run: dict[str, Any],
    steps: list[dict[str, Any]],
    conn: Any,
) -> KpiValue | None:
    """run ended_at - started_at in seconds."""
    started = run.get("started_at")
    ended = run.get("ended_at")
    if not started or not ended:
        return None
    try:
        t_start = datetime.fromisoformat(started)
        t_end = datetime.fromisoformat(ended)
        value = (t_end - t_start).total_seconds()
    except (ValueError, TypeError):
        return None
    return KpiValue(
        kpi_name="run_duration_s",
        kpi_value=round(value, 3),
        kpi_unit="seconds",
        step_id=None,
        source_artifact_id=None,
        details={"started_at": started, "ended_at": ended},
    )


def extract_recovery_count(
    run: dict[str, Any],
    steps: list[dict[str, Any]],
    conn: Any,
) -> KpiValue | None:
    """Count of recovery.attempted events in provenance for this run."""
    run_id = run["id"]
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM provenance_events "
        "WHERE run_id = ? AND action = 'recovery.attempted'",
        (run_id,),
    ).fetchone()
    count = row["cnt"] if row else 0
    return KpiValue(
        kpi_name="recovery_count",
        kpi_value=float(count),
        kpi_unit="count",
        step_id=None,
        source_artifact_id=None,
        details={"recovery_attempted_events": count},
    )


# ---------------------------------------------------------------------------
# Dispatcher maps
# ---------------------------------------------------------------------------

_STEP_EXTRACTORS: dict[str, Callable] = {
    "extract_volume_accuracy": extract_volume_accuracy,
    "extract_temp_accuracy": extract_temp_accuracy,
    "extract_impedance": extract_impedance,
    "extract_step_duration": extract_step_duration,
    # Electrochemistry extractors
    "extract_overpotential": extract_overpotential,
    "extract_current_density": extract_current_density,
    "extract_coulombic_efficiency": extract_coulombic_efficiency,
    "extract_stability_decay": extract_stability_decay,
    "extract_charge_passed": extract_charge_passed,
    # pH measurement extractors
    "extract_ph_value": extract_ph_value,
    "extract_ph_std": extract_ph_std,
    "extract_delta_ph": extract_delta_ph,
}

_RUN_EXTRACTORS: dict[str, Callable] = {
    "extract_run_success_rate": extract_run_success_rate,
    "extract_run_duration": extract_run_duration,
    "extract_recovery_count": extract_recovery_count,
}


# ---------------------------------------------------------------------------
# Core extraction + storage
# ---------------------------------------------------------------------------


def extract_and_store_kpis(run_id: str) -> list[KpiValue]:
    """Extract all KPIs for a completed run and persist to run_kpis table.

    Returns the list of extracted KPI values.
    All operations inside a single transaction.
    """
    import sqlite3 as _sqlite3

    extracted: list[KpiValue] = []

    def _txn(conn: _sqlite3.Connection) -> list[KpiValue]:
        # Load run record
        run_row = conn.execute(
            "SELECT * FROM runs WHERE id = ?", (run_id,),
        ).fetchone()
        if run_row is None:
            return []
        run = dict(run_row)

        # Load all steps
        step_rows = conn.execute(
            "SELECT * FROM run_steps WHERE run_id = ? ORDER BY step_key ASC",
            (run_id,),
        ).fetchall()
        steps = [dict(r) for r in step_rows]

        # Load artifacts (kind='primitive_result') keyed by step_id
        artifact_rows = conn.execute(
            "SELECT * FROM artifacts WHERE run_id = ? AND kind = 'primitive_result'",
            (run_id,),
        ).fetchall()
        artifacts_by_step: dict[str, dict[str, Any]] = {}
        for a in artifact_rows:
            artifacts_by_step[a["step_id"]] = dict(a)

        now = utcnow_iso()

        # --- Step-level KPIs ---
        for defn in KPI_DEFINITIONS_V1:
            extractor_fn = _STEP_EXTRACTORS.get(defn.extractor)
            if extractor_fn is None:
                continue
            for step in steps:
                # Filter by primitive if specified
                if defn.primitive is not None and step["primitive"] != defn.primitive:
                    continue
                # Skip non-terminal steps
                if step["status"] not in ("succeeded", "failed", "skipped"):
                    continue

                artifact = artifacts_by_step.get(step["id"])
                artifact_payload = None
                if artifact is not None:
                    try:
                        with open(artifact["uri"], "r") as f:
                            artifact_payload = json.load(f)
                    except Exception:
                        pass

                kpi = extractor_fn(step, artifact, artifact_payload)
                if kpi is not None:
                    conn.execute(
                        "INSERT INTO run_kpis "
                        "(id, run_id, step_id, kpi_name, kpi_value, kpi_unit, "
                        "kpi_schema_version, source_artifact_id, details_json, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            str(uuid.uuid4()),
                            run_id,
                            kpi.step_id,
                            kpi.kpi_name,
                            kpi.kpi_value,
                            kpi.kpi_unit,
                            KPI_SCHEMA_VERSION,
                            kpi.source_artifact_id,
                            json_dumps(kpi.details),
                            now,
                        ),
                    )
                    extracted.append(kpi)

        # --- Run-level KPIs ---
        for defn in KPI_DEFINITIONS_V1_RUN:
            extractor_fn = _RUN_EXTRACTORS.get(defn.extractor)
            if extractor_fn is None:
                continue
            kpi = extractor_fn(run, steps, conn)
            if kpi is not None:
                conn.execute(
                    "INSERT INTO run_kpis "
                    "(id, run_id, step_id, kpi_name, kpi_value, kpi_unit, "
                    "kpi_schema_version, source_artifact_id, details_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(uuid.uuid4()),
                        run_id,
                        kpi.step_id,
                        kpi.kpi_name,
                        kpi.kpi_value,
                        kpi.kpi_unit,
                        KPI_SCHEMA_VERSION,
                        kpi.source_artifact_id,
                        json_dumps(kpi.details),
                        now,
                    ),
                )
                extracted.append(kpi)

        return extracted

    return run_txn(_txn)


# ---------------------------------------------------------------------------
# Read path — advisory queries
# ---------------------------------------------------------------------------


def get_run_kpis(run_id: str) -> list[dict[str, Any]]:
    """Return all KPIs for a given run."""
    import sqlite3 as _sqlite3

    def _txn(conn: _sqlite3.Connection) -> list[dict[str, Any]]:
        rows = conn.execute(
            "SELECT * FROM run_kpis WHERE run_id = ? ORDER BY kpi_name, step_id",
            (run_id,),
        ).fetchall()
        result = []
        for r in rows:
            item = dict(r)
            item["details"] = parse_json(item.pop("details_json"), {})
            result.append(item)
        return result

    return run_txn(_txn)


def get_kpi_summary(
    kpi_name: str,
    schema_version: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return recent values for a given KPI name across runs."""
    import sqlite3 as _sqlite3

    def _txn(conn: _sqlite3.Connection) -> list[dict[str, Any]]:
        if schema_version:
            rows = conn.execute(
                "SELECT k.*, r.status as run_status FROM run_kpis k "
                "JOIN runs r ON k.run_id = r.id "
                "WHERE k.kpi_name = ? AND k.kpi_schema_version = ? "
                "ORDER BY k.created_at DESC LIMIT ?",
                (kpi_name, schema_version, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT k.*, r.status as run_status FROM run_kpis k "
                "JOIN runs r ON k.run_id = r.id "
                "WHERE k.kpi_name = ? "
                "ORDER BY k.created_at DESC LIMIT ?",
                (kpi_name, limit),
            ).fetchall()
        result = []
        for r in rows:
            item = dict(r)
            item["details"] = parse_json(item.pop("details_json"), {})
            result.append(item)
        return result

    return run_txn(_txn)


# ---------------------------------------------------------------------------
# Event listener — async write path
# ---------------------------------------------------------------------------

_listener_task: asyncio.Task[None] | None = None


async def _on_run_completed(run_id: str) -> None:
    """Extract KPIs for a completed run."""
    try:
        kpis = extract_and_store_kpis(run_id)
        logger.debug("KPIs extracted for run %s (%d values)", run_id, len(kpis))
    except Exception:
        logger.warning("KPI extraction failed for run %s", run_id, exc_info=True)


async def start_metrics_listener(bus: Any) -> Any:
    """Subscribe to the event bus and process run.completed events.

    Returns the Subscription handle for cleanup.
    """
    global _listener_task

    sub = await bus.subscribe(run_id=None)  # global subscription

    async def _listen() -> None:
        async for event in sub:
            if event.action == "run.completed":
                run_id = event.run_id
                if run_id:
                    await _on_run_completed(run_id)

    _listener_task = asyncio.create_task(_listen())
    return sub


async def stop_metrics_listener(sub: Any, bus: Any) -> None:
    """Cancel the metrics listener and unsubscribe."""
    global _listener_task

    sub.cancel()
    await bus.unsubscribe(sub)

    if _listener_task is not None:
        _listener_task.cancel()
        try:
            await _listener_task
        except asyncio.CancelledError:
            pass
        _listener_task = None
