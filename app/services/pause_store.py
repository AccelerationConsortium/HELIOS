"""Pause persistence service — durable human-in-the-loop state.

Persists PauseRequest/PauseResult to SQLite so that:
1. Paused executions survive process restarts
2. The SSE endpoint can replay pending pauses on reconnect
3. Operator decisions are recorded in the audit trail

DB table: ``pause_requests`` (created by ``ensure_pause_table``).

Typical flow::

    # Agent requests pause → ControlPlane handler calls:
    save_pause(campaign_id, agent_name, request)

    # Operator makes decision via API:
    resolve_pause(pause_id, decision, decided_by, modifications)

    # ControlPlane handler polls:
    status = get_pause_status(pause_id)
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Any

from app.core.db import connection, json_dumps, parse_json, run_txn, utcnow_iso
from app.services.audit import record_event

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pause_requests (
    id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    reason TEXT NOT NULL,
    risk_factors_json TEXT NOT NULL,
    suggested_action TEXT NOT NULL,
    checkpoint_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    decision TEXT,
    decided_by TEXT,
    decided_at TEXT,
    modifications_json TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pause_campaign ON pause_requests(campaign_id);
CREATE INDEX IF NOT EXISTS idx_pause_status ON pause_requests(status);
"""


def ensure_pause_table() -> None:
    """Create the pause_requests table if it doesn't exist.

    Safe to call multiple times (uses IF NOT EXISTS).
    Called from init_db() or at first use.
    """
    with connection() as conn:
        conn.executescript(_SCHEMA)


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------


def save_pause(
    campaign_id: str,
    agent_name: str,
    pause_id: str,
    reason: str,
    risk_factors: dict[str, float],
    suggested_action: str,
    checkpoint: dict[str, Any],
    metadata: dict[str, Any],
    expires_in_s: float,
) -> str:
    """Persist a new pause request.  Returns the pause_id."""
    from datetime import datetime, timedelta, timezone

    now = utcnow_iso()
    expires_at = (
        datetime.now(timezone.utc) + timedelta(seconds=expires_in_s)
    ).isoformat()

    def _txn(conn: sqlite3.Connection) -> str:
        conn.execute(
            """
            INSERT INTO pause_requests (
                id, campaign_id, agent_name, reason, risk_factors_json,
                suggested_action, checkpoint_json, metadata_json,
                expires_at, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
            """,
            (
                pause_id,
                campaign_id,
                agent_name,
                reason,
                json_dumps(risk_factors),
                suggested_action,
                json_dumps(checkpoint),
                json_dumps(metadata),
                expires_at,
                now,
            ),
        )
        record_event(
            conn,
            run_id=None,
            actor=agent_name,
            action="pause.requested",
            details={
                "pause_id": pause_id,
                "campaign_id": campaign_id,
                "reason": reason,
                "risk_factors": risk_factors,
                "suggested_action": suggested_action,
            },
        )
        return pause_id

    return run_txn(_txn)


def resolve_pause(
    pause_id: str,
    decision: str,
    decided_by: str = "",
    modifications: dict[str, Any] | None = None,
) -> bool:
    """Record operator decision for a pause request.

    Returns True if the pause was found and updated, False otherwise.
    """
    now = utcnow_iso()

    def _txn(conn: sqlite3.Connection) -> bool:
        row = conn.execute(
            "SELECT status FROM pause_requests WHERE id = ?", (pause_id,)
        ).fetchone()
        if row is None:
            return False
        if row["status"] != "pending":
            return False  # already resolved

        conn.execute(
            """
            UPDATE pause_requests
            SET status = 'resolved',
                decision = ?,
                decided_by = ?,
                decided_at = ?,
                modifications_json = ?
            WHERE id = ?
            """,
            (
                decision,
                decided_by,
                now,
                json_dumps(modifications or {}),
                pause_id,
            ),
        )
        record_event(
            conn,
            run_id=None,
            actor=decided_by or "operator",
            action="pause.resolved",
            details={
                "pause_id": pause_id,
                "decision": decision,
                "modifications": modifications or {},
            },
        )
        return True

    return run_txn(_txn)


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------


def get_pause_status(pause_id: str) -> dict[str, Any] | None:
    """Get current status of a pause request.

    Returns dict with: status, decision, decided_by, decided_at, modifications.
    Returns None if pause_id not found.
    """
    with connection() as conn:
        row = conn.execute(
            """
            SELECT status, decision, decided_by, decided_at, modifications_json
            FROM pause_requests WHERE id = ?
            """,
            (pause_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "status": row["status"],
            "decision": row["decision"],
            "decided_by": row["decided_by"],
            "decided_at": row["decided_at"],
            "modifications": parse_json(row["modifications_json"], {}),
        }


def list_pending_pauses(campaign_id: str) -> list[dict[str, Any]]:
    """List all pending (unresolved) pause requests for a campaign."""
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT id, agent_name, reason, risk_factors_json,
                   suggested_action, expires_at, created_at
            FROM pause_requests
            WHERE campaign_id = ? AND status = 'pending'
            ORDER BY created_at ASC
            """,
            (campaign_id,),
        ).fetchall()
        return [
            {
                "pause_id": r["id"],
                "agent_name": r["agent_name"],
                "reason": r["reason"],
                "risk_factors": parse_json(r["risk_factors_json"], {}),
                "suggested_action": r["suggested_action"],
                "expires_at": r["expires_at"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]


def get_pause_history(campaign_id: str) -> list[dict[str, Any]]:
    """Get all pause requests (pending + resolved) for a campaign.

    Useful for audit / post-hoc analysis of granularity decisions.
    """
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT id, agent_name, reason, risk_factors_json,
                   suggested_action, status, decision, decided_by,
                   decided_at, expires_at, created_at
            FROM pause_requests
            WHERE campaign_id = ?
            ORDER BY created_at ASC
            """,
            (campaign_id,),
        ).fetchall()
        return [
            {
                "pause_id": r["id"],
                "agent_name": r["agent_name"],
                "reason": r["reason"],
                "risk_factors": parse_json(r["risk_factors_json"], {}),
                "suggested_action": r["suggested_action"],
                "status": r["status"],
                "decision": r["decision"],
                "decided_by": r["decided_by"],
                "decided_at": r["decided_at"],
                "expires_at": r["expires_at"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]
