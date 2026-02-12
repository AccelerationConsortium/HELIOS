"""Dataset snapshot service: immutable, reproducible query result captures.

Provides:
  - create_snapshot(): execute a QueryPlan, hash the result, persist
  - load_snapshot(): retrieve snapshot metadata + run_ids
  - get_snapshot_rows(): re-execute from snapshot_runs (deterministic)
  - get_db_watermark(): current append-only sequence number

A snapshot is identified by snapshot_id and keyed by:
  query_plan_hash + db_watermark → deterministic at any point in time.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

from app.core.db import connection, json_dumps, parse_json, run_txn, utcnow_iso


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def new_snapshot_id() -> str:
    return f"snap-{uuid.uuid4().hex[:12]}"


def _hash_rows(rows: list[dict[str, Any]]) -> str:
    """Compute deterministic hash of query result rows."""
    # Sort keys in each dict, then sort rows by their JSON repr
    serialised = json.dumps(
        sorted([json.dumps(r, sort_keys=True, separators=(",", ":")) for r in rows])
    )
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()[:32]


# ---------------------------------------------------------------------------
# DB Watermark
# ---------------------------------------------------------------------------

def get_db_watermark() -> int:
    """Return the current DB watermark.

    Uses the max sequence from campaign_events (append-only log).
    Falls back to max rowid from runs if no events exist.
    """
    with connection() as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) AS wm FROM campaign_events"
        ).fetchone()
        wm = row["wm"]
        if wm == 0:
            # Fallback: use max rowid from provenance_events
            row2 = conn.execute(
                "SELECT COUNT(*) AS cnt FROM provenance_events"
            ).fetchone()
            wm = row2["cnt"]
    return wm


# ---------------------------------------------------------------------------
# Snapshot CRUD
# ---------------------------------------------------------------------------

def create_snapshot(
    *,
    rows: list[dict[str, Any]],
    query_plan_hash: str,
    campaign_id: str | None = None,
    query_plan_id: str | None = None,
    snapshot_name: str | None = None,
) -> dict[str, Any]:
    """Create an immutable dataset snapshot from query result rows.

    Returns snapshot metadata dict with snapshot_id.
    """
    snapshot_id = new_snapshot_id()
    watermark = get_db_watermark()
    result_hash = _hash_rows(rows)
    now = utcnow_iso()

    # Extract run_ids from rows if present
    run_ids: list[str] = []
    for row in rows:
        rid = row.get("run_id")
        if rid and rid not in run_ids:
            run_ids.append(rid)

    def _txn(conn):
        conn.execute(
            """INSERT INTO dataset_snapshots
               (snapshot_id, query_plan_id, campaign_id, db_watermark,
                query_plan_hash, result_hash, row_count, run_ids_json,
                snapshot_name, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                snapshot_id, query_plan_id, campaign_id, watermark,
                query_plan_hash, result_hash, len(rows),
                json_dumps(run_ids), snapshot_name, now,
            ),
        )
        # Materialize run associations for JOIN-based retrieval
        for rid in run_ids:
            conn.execute(
                "INSERT OR IGNORE INTO snapshot_runs (snapshot_id, run_id) "
                "VALUES (?, ?)",
                (snapshot_id, rid),
            )
    run_txn(_txn)

    return {
        "snapshot_id": snapshot_id,
        "query_plan_id": query_plan_id,
        "campaign_id": campaign_id,
        "db_watermark": watermark,
        "query_plan_hash": query_plan_hash,
        "result_hash": result_hash,
        "row_count": len(rows),
        "run_ids": run_ids,
        "snapshot_name": snapshot_name,
        "created_at": now,
    }


def load_snapshot(snapshot_id: str) -> dict[str, Any] | None:
    """Load snapshot metadata by ID."""
    with connection() as conn:
        row = conn.execute(
            "SELECT * FROM dataset_snapshots WHERE snapshot_id = ?",
            (snapshot_id,),
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["run_ids"] = parse_json(d.pop("run_ids_json"), [])
    return d


def get_snapshot_run_ids(snapshot_id: str) -> list[str]:
    """Return the materialized run_ids for a snapshot."""
    with connection() as conn:
        rows = conn.execute(
            "SELECT run_id FROM snapshot_runs WHERE snapshot_id = ? ORDER BY run_id",
            (snapshot_id,),
        ).fetchall()
    return [r["run_id"] for r in rows]


def list_snapshots(campaign_id: str | None = None) -> list[dict[str, Any]]:
    """List snapshots, optionally filtered by campaign."""
    with connection() as conn:
        if campaign_id:
            rows = conn.execute(
                "SELECT snapshot_id, campaign_id, db_watermark, result_hash, "
                "row_count, snapshot_name, created_at FROM dataset_snapshots "
                "WHERE campaign_id = ? ORDER BY created_at DESC",
                (campaign_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT snapshot_id, campaign_id, db_watermark, result_hash, "
                "row_count, snapshot_name, created_at FROM dataset_snapshots "
                "ORDER BY created_at DESC"
            ).fetchall()
    return [dict(r) for r in rows]


def verify_snapshot(snapshot_id: str, current_rows: list[dict[str, Any]]) -> bool:
    """Verify that re-executing the query produces the same result hash."""
    snap = load_snapshot(snapshot_id)
    if snap is None:
        return False
    return _hash_rows(current_rows) == snap["result_hash"]
