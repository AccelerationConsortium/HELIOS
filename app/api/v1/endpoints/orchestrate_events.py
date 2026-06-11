"""SSE endpoint for orchestrator campaign events — agent reasoning stream.

Supports:
- Live streaming via in-memory queues
- Last-Event-ID reconnection replay from campaign_events DB table
- DB-backed campaign existence check (survives restarts)
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from starlette.responses import StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/orchestrate", tags=["orchestrate"])

# In-memory event queues per campaign
_campaign_queues: dict[str, list[asyncio.Queue]] = {}


def publish_campaign_event(campaign_id: str, event: dict[str, Any]) -> None:
    """Publish an event to all SSE subscribers for a campaign.

    If the event has not been persisted yet (no ``_seq`` key), it is also
    written to the campaign_events DB table so the SSE replay phase can
    deliver it to late subscribers. Events coming from the orchestrator's
    ``_emit`` already carry ``_seq`` and are NOT persisted again — doing so
    previously stored every orchestrator event twice, which doubled DB
    writes and duplicated events on reconnection replay.
    """
    # Persist first so subscribers receive the event with its seq attached.
    if "_seq" not in event:
        event_type = event.get("type") or event.get("action") or "agent_event"
        try:
            from app.services.campaign_events import log_event
            event["_seq"] = log_event(campaign_id, event_type, event)
        except Exception:
            logger.debug("Campaign event persistence failed", exc_info=True)

    for queue in _campaign_queues.get(campaign_id, []):
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning(
                "Dropping SSE event for campaign %s: subscriber queue full",
                campaign_id,
            )


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
    # Subscribe to the live queue BEFORE replaying from DB. Subscribing after
    # replay left a window where events published mid-replay were neither in
    # the DB snapshot nor in the queue, and were silently lost. Any overlap
    # (event both replayed and queued) is removed by the seq check below.
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    _campaign_queues.setdefault(campaign_id, []).append(queue)

    # Phase 1: Always replay from DB.
    # - On first connect (last_event_id is None) we default to 0 so the browser
    #   gets all historical events even if the campaign finished before SSE was
    #   established (common in simulated mode where campaigns complete in <1 s).
    # - On reconnect, last_event_id is the last received seq so we only replay
    #   missed events.
    replay_from = last_event_id if last_event_id is not None else 0
    last_seq_sent = replay_from
    try:
        try:
            from app.services.campaign_events import replay_events
            for evt in replay_events(campaign_id, after_seq=replay_from):
                seq = evt["seq"]
                last_seq_sent = max(last_seq_sent, seq)
                event_type = evt["event_type"]
                data = json.dumps(evt["payload"], separators=(",", ":"))
                yield f"id: {seq}\nevent: {event_type}\ndata: {data}\n\n"
        except Exception:
            logger.debug("SSE replay failed for %s", campaign_id, exc_info=True)

        # Phase 2: Live stream from in-memory queue
        while True:
            if await request.is_disconnected():
                break
            try:
                event = await asyncio.wait_for(queue.get(), timeout=15)
            except TimeoutError:
                yield ": keepalive\n\n"
                continue

            if event is None:
                break

            # Skip events already delivered during DB replay.
            seq = event.get("_seq", "")
            if isinstance(seq, int):
                if seq <= last_seq_sent:
                    continue
                last_seq_sent = seq

            event_type = event.get("type", "agent_event")
            data = json.dumps(event, separators=(",", ":"))
            yield f"id: {seq}\nevent: {event_type}\ndata: {data}\n\n"
    finally:
        queues = _campaign_queues.get(campaign_id, [])
        if queue in queues:
            queues.remove(queue)
        if not queues:
            # Drop the campaign key entirely; otherwise every campaign that
            # ever had a subscriber leaks an empty list for the process life.
            _campaign_queues.pop(campaign_id, None)


@router.get("/{campaign_id}/events/stream")
async def campaign_event_stream(
    request: Request,
    campaign_id: str,
) -> StreamingResponse:
    """SSE stream for orchestrator campaign agent events.

    Supports ``Last-Event-ID`` header for reconnection replay.
    """
    # Check campaign exists — durable backend first (covers in-memory and
    # DB-restored campaigns), then demo campaigns, then the campaign_state DB.
    from app.api.v1.endpoints.orchestrate_demo import is_demo_campaign
    from app.services.durable_execution import get_durable_backend

    if not is_demo_campaign(campaign_id) and await get_durable_backend().get_status(campaign_id) is None:
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
