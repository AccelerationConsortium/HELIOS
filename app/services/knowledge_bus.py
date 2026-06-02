"""Lightweight gossip-style knowledge bus for intra-campaign soft knowledge propagation.

Agents publish incremental discoveries during execution; other agents drain the bus
before their next LLM call and inject the knowledge into their system prompt.

Design notes
------------
* One :class:`KnowledgeBus` per campaign, obtained via :func:`get_bus`.
* Events have a TTL (in rounds): stale events are filtered during drain.
* :meth:`drain_for_agent` marks events as seen so each agent receives each
  event exactly once.
* :meth:`format_for_prompt` converts a list of events into a concise Markdown
  block suitable for injection into an LLM system prompt.
* fnmatch-style pattern subscriptions are supported for async notification.

This module depends only on the Python standard library.
"""
from __future__ import annotations

import asyncio
import fnmatch
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "KnowledgeEvent",
    "KnowledgeBus",
    "CampaignKnowledgeBus",
    "get_bus",
]

logger = logging.getLogger("helios.services.knowledge_bus")


@dataclass(frozen=True)
class KnowledgeEvent:
    """A single incremental knowledge update published by an agent.

    Attributes:
        source_agent: Name of the publishing agent.
        key: Logical topic key (e.g. ``"instrument.ot2.tip_failure_rate"``).
        delta_text: Human-readable incremental update, injected into LLM prompts.
        confidence: Publisher-reported confidence in ``[0.0, 1.0]``.
        round_id: Round identifier when the event was published.
        ttl_rounds: How many rounds this knowledge remains valid after publication.
    """

    source_agent: str
    key: str
    delta_text: str
    confidence: float
    round_id: str
    ttl_rounds: int = 3
    event_id: str = field(default_factory=lambda: f"ke-{uuid.uuid4().hex[:12]}")

    def is_expired(self, current_round: int) -> bool:
        """True if this event should no longer be delivered.

        Round numbers are compared numerically when possible; non-numeric
        round IDs never expire (conservative default).
        """
        try:
            pub = int(self.round_id)
            return (current_round - pub) >= self.ttl_rounds
        except (ValueError, TypeError):
            return False


class KnowledgeBus:
    """Per-campaign knowledge bus.

    Thread-safe via :class:`asyncio.Lock`. All mutating operations acquire the
    lock; reads from :meth:`drain_for_agent` also acquire it to ensure
    consistent seen-set bookkeeping.
    """

    def __init__(self, campaign_id: str) -> None:
        self._campaign_id = campaign_id
        self._events: list[KnowledgeEvent] = []
        # agent_name -> stable event ids already delivered
        self._seen: dict[str, set[str]] = {}
        # pattern -> list of queues for async notification
        self._pattern_subs: dict[str, list[asyncio.Queue[KnowledgeEvent]]] = {}
        self._lock = asyncio.Lock()
        self._published = 0
        logger.info("knowledge_bus.created", extra={"campaign_id": campaign_id})

    # ── Publishing ────────────────────────────────────────────────────────────

    def publish(self, event: KnowledgeEvent) -> None:
        """Publish a knowledge event synchronously (no await needed).

        Fan-out to pattern subscribers is done via ``put_nowait``; if a
        bounded queue is full the notification is dropped silently to avoid
        blocking the publisher.
        """
        self._events.append(event)
        self._published += 1
        logger.debug(
            "knowledge_bus.publish",
            extra={
                "campaign_id": self._campaign_id,
                "key": event.key,
                "source": event.source_agent,
                "round_id": event.round_id,
                "confidence": event.confidence,
            },
        )
        # Notify async pattern subscribers
        for pattern, queues in self._pattern_subs.items():
            if fnmatch.fnmatch(event.key, pattern):
                for q in queues:
                    try:
                        q.put_nowait(event)
                    except asyncio.QueueFull:
                        logger.debug(
                            "knowledge_bus.queue_full",
                            extra={"pattern": pattern, "key": event.key},
                        )

    # ── Draining ─────────────────────────────────────────────────────────────

    async def drain_for_agent(
        self, agent_name: str, current_round: int
    ) -> list[KnowledgeEvent]:
        """Return unseen, non-expired events for *agent_name*.

        Each event is delivered to a given agent at most once (idempotent).
        Expired events are filtered but not removed from the internal list
        (garbage collection happens in :meth:`clear_round`).
        """
        async with self._lock:
            seen = self._seen.setdefault(agent_name, set())
            result: list[KnowledgeEvent] = []
            for ev in self._events:
                if ev.event_id in seen:
                    continue
                if ev.is_expired(current_round):
                    continue
                result.append(ev)
                seen.add(ev.event_id)
            return result

    def snapshot(self, *, include_seen: bool = False) -> dict[str, Any]:
        """Return a serializable snapshot for replay/debug surfaces."""
        payload: dict[str, Any] = {
            "campaign_id": self._campaign_id,
            "stats": self.stats(),
            "events": [
                {
                    "event_id": ev.event_id,
                    "source_agent": ev.source_agent,
                    "key": ev.key,
                    "delta_text": ev.delta_text,
                    "confidence": ev.confidence,
                    "round_id": ev.round_id,
                    "ttl_rounds": ev.ttl_rounds,
                }
                for ev in self._events
            ],
        }
        if include_seen:
            payload["seen"] = {
                agent: sorted(event_ids)
                for agent, event_ids in self._seen.items()
            }
        return payload

    # ── Prompt formatting ─────────────────────────────────────────────────────

    @staticmethod
    def format_for_prompt(events: list[KnowledgeEvent]) -> str:
        """Render events as a concise Markdown block for LLM system-prompt injection.

        Returns an empty string when *events* is empty so callers can do a
        simple truthiness check before concatenating.
        """
        if not events:
            return ""
        lines = ["## Soft Knowledge Updates (from peer agents — treat as advisory)\n"]
        for ev in events:
            conf_pct = int(ev.confidence * 100)
            lines.append(f"- **[{ev.key}]** {ev.delta_text} _(confidence {conf_pct}%, from {ev.source_agent})_")
        return "\n".join(lines) + "\n"

    # ── Async subscriptions ───────────────────────────────────────────────────

    def subscribe_pattern(self, pattern: str, queue: asyncio.Queue[KnowledgeEvent]) -> None:
        """Register *queue* for async notifications when a key matches *pattern*.

        *pattern* follows ``fnmatch`` conventions:
        - ``"instrument.*"`` matches all instrument keys
        - ``"spectral.xrd"`` matches exactly that key
        """
        self._pattern_subs.setdefault(pattern, []).append(queue)
        logger.debug(
            "knowledge_bus.subscribe",
            extra={"campaign_id": self._campaign_id, "pattern": pattern},
        )

    def unsubscribe_pattern(self, pattern: str, queue: asyncio.Queue[KnowledgeEvent]) -> None:
        """Remove a previously registered queue for *pattern*."""
        bucket = self._pattern_subs.get(pattern, [])
        try:
            bucket.remove(queue)
        except ValueError:
            pass

    # ── Garbage collection ────────────────────────────────────────────────────

    async def clear_round(self, round_id: str) -> int:
        """Remove events whose round_id matches *round_id*.

        Returns the number of events removed.
        """
        async with self._lock:
            before = len(self._events)
            self._events = [ev for ev in self._events if ev.round_id != round_id]
            removed = before - len(self._events)
            if removed:
                logger.debug(
                    "knowledge_bus.gc",
                    extra={
                        "campaign_id": self._campaign_id,
                        "round_id": round_id,
                        "removed": removed,
                    },
                )
            return removed

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        """Return a snapshot of bus statistics."""
        return {
            "published": self._published,
            "active": len(self._events),
            "subscribers": sum(len(qs) for qs in self._pattern_subs.values()),
        }

    @property
    def campaign_id(self) -> str:
        return self._campaign_id


# ── Process-wide registry ────────────────────────────────────────────────────

class CampaignKnowledgeBus:
    """Registry of per-campaign :class:`KnowledgeBus` instances."""

    def __init__(self) -> None:
        self._buses: dict[str, KnowledgeBus] = {}

    def get_bus(self, campaign_id: str) -> KnowledgeBus:
        """Return the bus for *campaign_id*, creating it if necessary."""
        if campaign_id not in self._buses:
            self._buses[campaign_id] = KnowledgeBus(campaign_id)
        return self._buses[campaign_id]

    def discard(self, campaign_id: str) -> None:
        """Remove the bus for *campaign_id* (called on campaign completion)."""
        self._buses.pop(campaign_id, None)
        logger.info("knowledge_bus.discarded", extra={"campaign_id": campaign_id})

    def active_campaigns(self) -> list[str]:
        return list(self._buses.keys())


_registry = CampaignKnowledgeBus()


def get_bus(campaign_id: str) -> KnowledgeBus:
    """Module-level convenience: get or create the bus for *campaign_id*."""
    return _registry.get_bus(campaign_id)
