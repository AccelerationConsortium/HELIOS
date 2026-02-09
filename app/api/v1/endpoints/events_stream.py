"""SSE (Server-Sent Events) streaming endpoints for real-time event delivery.

Provides two SSE endpoints:
- ``GET /events/stream`` — global event stream (all events)
- ``GET /runs/{run_id}/events/stream`` — per-run event stream with historical catch-up

Supports ``Last-Event-ID`` header for reconnection, and sends ``:keepalive``
comments every 15 seconds to prevent proxy/load-balancer timeouts.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from starlette.responses import StreamingResponse

from app.core.db import parse_json, row_to_dict, run_txn
from app.services.event_bus import EventMessage

router = APIRouter(tags=["events-stream"])

KEEPALIVE_INTERVAL = 15  # seconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_event_bus():
    """Lazy import to avoid circular dependency with app.main."""
    from app.main import event_bus

    return event_bus


def _format_sse(event: EventMessage) -> str:
    """Format an EventMessage as an SSE text block."""
    data = json.dumps(
        {
            "id": event.id,
            "run_id": event.run_id,
            "actor": event.actor,
            "action": event.action,
            "details": event.details,
            "created_at": event.created_at,
        },
        separators=(",", ":"),
    )
    return f"id: {event.id}\nevent: {event.action}\ndata: {data}\n\n"


def _format_sse_from_dict(row: dict[str, Any]) -> str:
    """Format a DB row dict as an SSE text block."""
    data = json.dumps(
        {
            "id": row["id"],
            "run_id": row.get("run_id"),
            "actor": row["actor"],
            "action": row["action"],
            "details": row.get("details", {}),
            "created_at": row["created_at"],
        },
        separators=(",", ":"),
    )
    return f"id: {row['id']}\nevent: {row['action']}\ndata: {data}\n\n"


def _fetch_historical_events(run_id: str) -> list[dict[str, Any]]:
    """Fetch all provenance events for *run_id* from the database."""

    def _txn(conn):
        rows = conn.execute(
            "SELECT * FROM provenance_events WHERE run_id = ? ORDER BY created_at ASC",
            (run_id,),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item = row_to_dict(row)
            assert item is not None
            item["details"] = parse_json(item.pop("details_json"), {})
            out.append(item)
        return out

    return run_txn(_txn)


def _run_exists(run_id: str) -> bool:
    """Check whether *run_id* exists in the ``runs`` table."""

    def _txn(conn):
        row = conn.execute("SELECT 1 FROM runs WHERE id = ?", (run_id,)).fetchone()
        return row is not None

    return run_txn(_txn)


# ---------------------------------------------------------------------------
# Async generator
# ---------------------------------------------------------------------------


async def _event_generator(
    request: Request,
    sub,
    historical: list[dict[str, Any]] | None = None,
    last_event_id: str | None = None,
):
    """Yield SSE-formatted strings: historical events first, then live events.

    Parameters
    ----------
    request:
        The incoming HTTP request (used to detect client disconnect).
    sub:
        An ``EventBus.Subscription`` providing live events.
    historical:
        Optional list of historical event dicts to replay before live events.
    last_event_id:
        If provided, skip historical events up to and including this ID.
    """
    bus = _get_event_bus()

    try:
        # 1. Replay historical events (if any)
        if historical:
            skip = last_event_id is not None
            for row in historical:
                if skip:
                    if row["id"] == last_event_id:
                        skip = False
                    continue
                yield _format_sse_from_dict(row)

        # 2. Stream live events with keepalive
        while True:
            if await request.is_disconnected():
                break
            try:
                event = await asyncio.wait_for(sub.queue.get(), timeout=KEEPALIVE_INTERVAL)
            except asyncio.TimeoutError:
                # Send keepalive comment
                yield ": keepalive\n\n"
                continue

            if event is None:
                # Sentinel — bus stopped or subscription cancelled
                break

            # Skip events already seen via Last-Event-ID
            if last_event_id and event.id == last_event_id:
                last_event_id = None  # clear after match
                continue

            yield _format_sse(event)
    finally:
        await bus.unsubscribe(sub)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/events/stream")
async def global_event_stream(
    request: Request,
    last_event_id: str | None = Query(None, alias="Last-Event-ID"),
) -> StreamingResponse:
    """SSE stream of **all** events across all runs."""
    bus = _get_event_bus()

    # Also check the standard SSE header
    header_last_id = request.headers.get("Last-Event-ID") or last_event_id

    sub = await bus.subscribe(run_id=None)
    return StreamingResponse(
        _event_generator(request, sub, last_event_id=header_last_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/runs/{run_id}/events/stream")
async def run_event_stream(
    request: Request,
    run_id: str,
    last_event_id: str | None = Query(None, alias="Last-Event-ID"),
) -> StreamingResponse:
    """SSE stream for a specific run, with historical catch-up.

    The endpoint first subscribes to live events, **then** queries historical
    events from the database. This ordering guarantees no events are lost in
    the gap between subscribing and querying.
    """
    if not _run_exists(run_id):
        raise HTTPException(status_code=404, detail="run not found")

    bus = _get_event_bus()

    header_last_id = request.headers.get("Last-Event-ID") or last_event_id

    # Subscribe FIRST, then query history → gap-free delivery
    sub = await bus.subscribe(run_id=run_id)
    historical = _fetch_historical_events(run_id)

    return StreamingResponse(
        _event_generator(request, sub, historical=historical, last_event_id=header_last_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
