"""SSE event persistence and replay for campaign runner."""

from __future__ import annotations

from typing import Any

from app.core.db import connection, json_dumps, parse_json, run_txn, utcnow_iso


def log_event(
    campaign_id: str,
    event_type: str,
    payload: dict[str, Any],
) -> int:
    """Persist an SSE event and return its sequence number."""
    now = utcnow_iso()

    def _insert(conn):
        cur = conn.execute(
            """INSERT INTO campaign_events
               (campaign_id, event_type, payload_json, created_at)
               VALUES (?, ?, ?, ?)""",
            (campaign_id, event_type, json_dumps(payload), now),
        )
        return cur.lastrowid

    return run_txn(_insert)


def replay_events(
    campaign_id: str,
    after_seq: int = 0,
) -> list[dict[str, Any]]:
    """Return events with seq > after_seq, ordered by seq ASC.

    Used for Last-Event-ID reconnection replay.
    """
    with connection() as conn:
        rows = conn.execute(
            """SELECT seq, campaign_id, event_type, payload_json, created_at
               FROM campaign_events
               WHERE campaign_id = ? AND seq > ?
               ORDER BY seq ASC""",
            (campaign_id, after_seq),
        ).fetchall()

    result = []
    for r in rows:
        result.append({
            "seq": r["seq"],
            "campaign_id": r["campaign_id"],
            "event_type": r["event_type"],
            "payload": parse_json(r["payload_json"], {}),
            "created_at": r["created_at"],
        })
    return result


def get_latest_seq(campaign_id: str) -> int:
    """Return the highest seq for a campaign, or 0 if none."""
    with connection() as conn:
        row = conn.execute(
            "SELECT MAX(seq) AS max_seq FROM campaign_events WHERE campaign_id = ?",
            (campaign_id,),
        ).fetchone()
    if row is None or row["max_seq"] is None:
        return 0
    return row["max_seq"]
