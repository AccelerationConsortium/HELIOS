"""QC flags and failure signature persistence.

Provides structured quality annotations for the DB Retrieval Agent:
  - qc_flags: ok | missing | suspect | failed per (run, kpi)
  - run_failure_signatures: machine-readable failure classifications
  - metric_dictionary: canonical metric registry
  - param_schema: campaign-level parameter type/unit storage
  - experiment_index: cross-SDL dimension indexing
"""
from __future__ import annotations

import uuid
from typing import Any

from app.core.db import connection, json_dumps, parse_json, run_txn, utcnow_iso


# ---------------------------------------------------------------------------
# QC Flags
# ---------------------------------------------------------------------------

def store_qc_flag(
    *,
    run_id: str,
    kpi_name: str,
    flag_value: str = "ok",
    step_id: str | None = None,
    measured_value: float | None = None,
    threshold: float | None = None,
    message: str | None = None,
) -> str:
    """Store a QC flag for a (run, kpi) pair. Returns the flag id."""
    flag_id = uuid.uuid4().hex[:12]
    now = utcnow_iso()

    def _txn(conn):
        conn.execute(
            """INSERT INTO qc_flags
               (id, run_id, step_id, kpi_name, flag_value,
                measured_value, threshold, message, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (flag_id, run_id, step_id, kpi_name, flag_value,
             measured_value, threshold, message, now),
        )
    run_txn(_txn)
    return flag_id


def get_qc_flags(run_id: str) -> list[dict[str, Any]]:
    """Get all QC flags for a run."""
    with connection() as conn:
        rows = conn.execute(
            "SELECT * FROM qc_flags WHERE run_id = ? ORDER BY kpi_name",
            (run_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_suspect_runs(kpi_name: str | None = None) -> list[str]:
    """Return run_ids with suspect or failed QC flags."""
    with connection() as conn:
        if kpi_name:
            rows = conn.execute(
                "SELECT DISTINCT run_id FROM qc_flags "
                "WHERE flag_value IN ('suspect', 'failed') AND kpi_name = ?",
                (kpi_name,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT DISTINCT run_id FROM qc_flags "
                "WHERE flag_value IN ('suspect', 'failed')"
            ).fetchall()
    return [r["run_id"] for r in rows]


# ---------------------------------------------------------------------------
# Failure Signatures
# ---------------------------------------------------------------------------

def store_failure_signature(
    *,
    run_id: str,
    failure_type: str,
    severity: str,
    likely_cause: str,
    step_key: str | None = None,
    primitive: str | None = None,
    confidence: float | None = None,
    retryable: bool = False,
    message_code: str | None = None,
    recommended_patch: dict | None = None,
) -> str:
    """Persist a machine-readable failure signature. Returns the id."""
    sig_id = uuid.uuid4().hex[:12]
    now = utcnow_iso()

    def _txn(conn):
        conn.execute(
            """INSERT INTO run_failure_signatures
               (id, run_id, step_key, primitive, failure_type, severity,
                likely_cause, confidence, retryable, message_code,
                recommended_patch_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (sig_id, run_id, step_key, primitive, failure_type, severity,
             likely_cause, confidence, 1 if retryable else 0,
             message_code, json_dumps(recommended_patch) if recommended_patch else None,
             now),
        )
    run_txn(_txn)
    return sig_id


def get_failure_signatures(run_id: str) -> list[dict[str, Any]]:
    """Get all failure signatures for a run."""
    with connection() as conn:
        rows = conn.execute(
            "SELECT * FROM run_failure_signatures WHERE run_id = ? "
            "ORDER BY created_at",
            (run_id,),
        ).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        d["retryable"] = bool(d["retryable"])
        d["recommended_patch"] = parse_json(d.pop("recommended_patch_json"), None)
        results.append(d)
    return results


def get_failure_stats() -> list[dict[str, Any]]:
    """Aggregate failure statistics by type."""
    with connection() as conn:
        rows = conn.execute(
            "SELECT failure_type, severity, COUNT(*) as count, "
            "AVG(confidence) as avg_confidence "
            "FROM run_failure_signatures "
            "GROUP BY failure_type, severity ORDER BY count DESC"
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Metric Dictionary
# ---------------------------------------------------------------------------

def register_metric(
    *,
    metric_name: str,
    unit: str,
    definition: str,
    scope: str = "run",
    extractor_version: str = "1",
) -> None:
    """Register a metric in the dictionary (idempotent)."""
    now = utcnow_iso()

    def _txn(conn):
        conn.execute(
            """INSERT OR REPLACE INTO metric_dictionary
               (metric_name, unit, definition, scope, extractor_version, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (metric_name, unit, definition, scope, extractor_version, now),
        )
    run_txn(_txn)


def get_metric_dictionary() -> list[dict[str, Any]]:
    """Return all registered metrics."""
    with connection() as conn:
        rows = conn.execute(
            "SELECT * FROM metric_dictionary ORDER BY metric_name"
        ).fetchall()
    return [dict(r) for r in rows]


def seed_metric_dictionary() -> int:
    """Seed the metric dictionary from the KPI definitions in metrics.py.

    Returns the number of metrics seeded.
    """
    from app.services.metrics import (
        KPI_DEFINITIONS_V1,
        KPI_DEFINITIONS_V1_RUN,
        KPI_SCHEMA_VERSION,
    )

    all_kpis = list(KPI_DEFINITIONS_V1) + list(KPI_DEFINITIONS_V1_RUN)
    count = 0
    for kpi in all_kpis:
        register_metric(
            metric_name=kpi.name,
            unit=kpi.unit,
            definition=f"{kpi.name} ({kpi.extractor})",
            scope=kpi.scope,
            extractor_version=KPI_SCHEMA_VERSION,
        )
        count += 1
    return count


# ---------------------------------------------------------------------------
# Parameter Schema (Req 5)
# ---------------------------------------------------------------------------

def store_param_schema(
    campaign_id: str,
    dimensions: list[dict[str, Any]],
) -> int:
    """Store parameter type/unit metadata for a campaign.

    Args:
        campaign_id: The campaign these dimensions belong to.
        dimensions: List of DimensionDef-like dicts with param_name, param_type, unit, etc.

    Returns count of stored params.
    """
    count = 0

    def _txn(conn):
        nonlocal count
        for dim in dimensions:
            conn.execute(
                """INSERT OR REPLACE INTO param_schema
                   (campaign_id, param_name, param_type, unit,
                    min_value, max_value, log_scale, choices_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    campaign_id,
                    dim["param_name"],
                    dim.get("param_type", "number"),
                    dim.get("unit", ""),
                    dim.get("min_value"),
                    dim.get("max_value"),
                    1 if dim.get("log_scale") else 0,
                    json_dumps(dim.get("choices")) if dim.get("choices") else None,
                ),
            )
            count += 1
    run_txn(_txn)
    return count


def get_param_schema(campaign_id: str) -> list[dict[str, Any]]:
    """Get parameter schema for a campaign."""
    with connection() as conn:
        rows = conn.execute(
            "SELECT * FROM param_schema WHERE campaign_id = ? ORDER BY param_name",
            (campaign_id,),
        ).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        d["log_scale"] = bool(d["log_scale"])
        d["choices"] = parse_json(d.pop("choices_json"), None)
        results.append(d)
    return results


# ---------------------------------------------------------------------------
# Experiment Index (Req 6)
# ---------------------------------------------------------------------------

def index_experiment(
    *,
    run_id: str,
    domain: str | None = None,
    system_id: str | None = None,
    instrument_set: str | None = None,
    protocol_version: str | None = None,
    workflow_template_id: str | None = None,
    experiment_class: str | None = None,
    tags: list[str] | None = None,
) -> None:
    """Add or update experiment index dimensions for a run."""
    now = utcnow_iso()

    def _txn(conn):
        conn.execute(
            """INSERT OR REPLACE INTO experiment_index
               (run_id, domain, system_id, instrument_set,
                protocol_version, workflow_template_id,
                experiment_class, tags_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, domain, system_id, instrument_set,
             protocol_version, workflow_template_id,
             experiment_class, json_dumps(tags or []), now),
        )
    run_txn(_txn)


def get_experiment_index(run_id: str) -> dict[str, Any] | None:
    """Get experiment index for a run."""
    with connection() as conn:
        row = conn.execute(
            "SELECT * FROM experiment_index WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["tags"] = parse_json(d.pop("tags_json"), [])
    return d


def search_experiments(
    *,
    domain: str | None = None,
    system_id: str | None = None,
    protocol_version: str | None = None,
    workflow_template_id: str | None = None,
) -> list[dict[str, Any]]:
    """Search experiment index by dimension filters."""
    conditions: list[str] = []
    params: list[Any] = []

    if domain:
        conditions.append("domain = ?")
        params.append(domain)
    if system_id:
        conditions.append("system_id = ?")
        params.append(system_id)
    if protocol_version:
        conditions.append("protocol_version = ?")
        params.append(protocol_version)
    if workflow_template_id:
        conditions.append("workflow_template_id = ?")
        params.append(workflow_template_id)

    where = " AND ".join(conditions) if conditions else "1=1"

    with connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM experiment_index WHERE {where} ORDER BY created_at DESC",
            params,
        ).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        d["tags"] = parse_json(d.pop("tags_json"), [])
        results.append(d)
    return results
