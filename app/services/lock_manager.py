from __future__ import annotations

from datetime import datetime, timedelta, timezone
import sqlite3
from typing import Any

from app.core.db import utcnow_iso


def _parse_iso(ts: str | None) -> datetime | None:
    if ts is None:
        return None
    return datetime.fromisoformat(ts)


def acquire_lock(
    conn: sqlite3.Connection,
    *,
    resource_id: str,
    run_id: str,
    ttl_seconds: int,
) -> dict[str, Any] | None:
    now = datetime.now(timezone.utc)
    lease_until = (now + timedelta(seconds=ttl_seconds)).isoformat()

    row = conn.execute(
        "SELECT resource_id, owner_run_id, lease_until, fencing_token FROM resource_locks WHERE resource_id = ?",
        (resource_id,),
    ).fetchone()

    if row is None:
        token = 1
        conn.execute(
            """
            INSERT INTO resource_locks (resource_id, owner_run_id, lease_until, fencing_token, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (resource_id, run_id, lease_until, token, utcnow_iso()),
        )
        return {"resource_id": resource_id, "fencing_token": token, "lease_until": lease_until}

    current_owner = row["owner_run_id"]
    current_lease = _parse_iso(row["lease_until"])
    token = int(row["fencing_token"])

    lease_expired = current_lease is None or current_lease <= now
    if current_owner in (None, run_id) or lease_expired:
        token += 1
        conn.execute(
            """
            UPDATE resource_locks
            SET owner_run_id = ?, lease_until = ?, fencing_token = ?, updated_at = ?
            WHERE resource_id = ?
            """,
            (run_id, lease_until, token, utcnow_iso(), resource_id),
        )
        return {"resource_id": resource_id, "fencing_token": token, "lease_until": lease_until}

    return None


def release_lock(conn: sqlite3.Connection, *, resource_id: str, run_id: str) -> None:
    conn.execute(
        """
        UPDATE resource_locks
        SET owner_run_id = NULL, lease_until = NULL, updated_at = ?
        WHERE resource_id = ? AND owner_run_id = ?
        """,
        (utcnow_iso(), resource_id, run_id),
    )
