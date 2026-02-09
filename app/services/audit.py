"""Append-only audit trail for provenance events.

Every state transition in the orchestrator records an event here.
When the event bus is wired (via ``set_event_bus``), events are also
published to SSE subscribers in real time.
"""
from __future__ import annotations

import uuid
import sqlite3
from typing import Any

from app.core.db import json_dumps, utcnow_iso

# ---------------------------------------------------------------------------
# Module-level event bus reference (set during FastAPI lifespan)
# ---------------------------------------------------------------------------

_event_bus: Any = None


def set_event_bus(bus: Any) -> None:
    """Called once during FastAPI lifespan to enable real-time event publishing."""
    global _event_bus
    _event_bus = bus


# ---------------------------------------------------------------------------
# Core audit function
# ---------------------------------------------------------------------------


def record_event(
    conn: sqlite3.Connection,
    *,
    run_id: str | None,
    actor: str,
    action: str,
    details: dict[str, Any],
) -> None:
    event_id = str(uuid.uuid4())
    created_at = utcnow_iso()

    conn.execute(
        """
        INSERT INTO provenance_events (id, run_id, actor, action, details_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (event_id, run_id, actor, action, json_dumps(details), created_at),
    )

    # Publish to event bus for real-time SSE streaming (thread-safe, no-op if bus not set)
    bus = _event_bus
    if bus is not None:
        from app.services.event_bus import EventMessage

        bus.publish(
            EventMessage(
                id=event_id,
                run_id=run_id,
                actor=actor,
                action=action,
                details=details,
                created_at=created_at,
            )
        )
