"""SSE endpoint for orchestrator campaign events — agent reasoning stream.

Supports:
- Live streaming via in-memory queues
- Last-Event-ID reconnection replay from campaign_events DB table
- DB-backed campaign existence check (survives restarts)
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from starlette.responses import StreamingResponse

router = APIRouter(prefix="/orchestrate", tags=["orchestrate"])

# In-memory event queues per campaign
_campaign_queues: dict[str, list[asyncio.Queue]] = {}


def publish_campaign_event(campaign_id: str, event: dict[str, Any]) -> None:
    """Publish an event to all SSE subscribers for a campaign."""
    for queue in _campaign_queues.get(campaign_id, []):
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            pass  # Drop if subscriber is slow


async def _campaign_event_generator(
    request: Request,
    campaign_id: str,
    *,
    last_event_id: int | None = None,
):
    """Yield SSE events for a campaign.

    If *last_event_id* is given, replay missed events from DB first,
    then switch to the live in-memory queue.
    """
    # Phase 1: Always replay from DB.
    # - On first connect (last_event_id is None) we default to 0 so the browser
    #   gets all historical events even if the campaign finished before SSE was
    #   established (common in simulated mode where campaigns complete in <1 s).
    # - On reconnect, last_event_id is the last received seq so we only replay
    #   missed events.
    replay_from = last_event_id if last_event_id is not None else 0
    try:
        from app.services.campaign_events import replay_events
        for evt in replay_events(campaign_id, after_seq=replay_from):
            seq = evt["seq"]
            event_type = evt["event_type"]
            data = json.dumps(evt["payload"], separators=(",", ":"))
            yield f"id: {seq}\nevent: {event_type}\ndata: {data}\n\n"
    except Exception:
        pass  # best-effort replay

    # Phase 2: Live stream from in-memory queue
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)

    if campaign_id not in _campaign_queues:
        _campaign_queues[campaign_id] = []
    _campaign_queues[campaign_id].append(queue)

    try:
        while True:
            if await request.is_disconnected():
                break
            try:
                event = await asyncio.wait_for(queue.get(), timeout=15)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue

            if event is None:
                break

            event_type = event.get("type", "agent_event")
            seq = event.get("_seq", "")
            data = json.dumps(event, separators=(",", ":"))
            yield f"id: {seq}\nevent: {event_type}\ndata: {data}\n\n"
    finally:
        queues = _campaign_queues.get(campaign_id, [])
        if queue in queues:
            queues.remove(queue)


@router.get("/{campaign_id}/events/stream")
async def campaign_event_stream(
    request: Request,
    campaign_id: str,
) -> StreamingResponse:
    """SSE stream for orchestrator campaign agent events.

    Supports ``Last-Event-ID`` header for reconnection replay.
    """
    from app.api.v1.endpoints.orchestrate import (
        _running_campaigns,
        _campaign_results,
        _campaign_errors,
    )

    # Check campaign exists — in-memory first, then DB
    in_memory = (
        campaign_id in _running_campaigns
        or campaign_id in _campaign_results
        or campaign_id in _campaign_errors
    )
    if not in_memory:
        # Fall back to DB
        from app.services.campaign_state import load_campaign
        if load_campaign(campaign_id) is None:
            raise HTTPException(
                status_code=404, detail=f"Campaign '{campaign_id}' not found"
            )

    # Parse Last-Event-ID from header
    last_event_id: int | None = None
    raw_id = request.headers.get("Last-Event-ID") or request.headers.get("last-event-id")
    if raw_id is not None:
        try:
            last_event_id = int(raw_id)
        except (ValueError, TypeError):
            pass

    return StreamingResponse(
        _campaign_event_generator(
            request, campaign_id, last_event_id=last_event_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
