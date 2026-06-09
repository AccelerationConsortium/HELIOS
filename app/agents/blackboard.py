"""Round-scoped shared working memory ("Blackboard") for HELIOS.

This module provides a lightweight, async-first publish/subscribe working
memory that lets agents collaborating within a single experimental round share
discoveries in real time -- without routing every message through the
orchestrator.

Design notes
------------
* A :class:`RoundBlackboard` is scoped to exactly one ``(campaign_id, round_id)``
  pair. When the round completes the orchestrator calls
  :meth:`BlackboardManager.discard` to free its subscribers and entries.
* Writes are serialized through an :class:`asyncio.Lock` so that the stored
  snapshot and the fan-out to subscriber queues stay mutually consistent.
* Reads are deliberately non-blocking and lock-free: they operate on the
  current ``dict`` reference, which is only ever swapped under the write lock,
  so a reader always sees a coherent (if possibly slightly stale) entry.
* Subscriptions are delivered via :class:`asyncio.Queue`. Exact-key and
  prefix ("wildcard") subscriptions are both supported.

The module depends only on the Python standard library.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "BlackboardEntry",
    "RoundBlackboard",
    "BlackboardManager",
    "manager",
]

logger = logging.getLogger("helios.agents.blackboard")


@dataclass(slots=True, frozen=True)
class BlackboardEntry:
    """A single immutable record posted to a blackboard.

    Attributes:
        key: Logical address of the discovery (e.g. ``"spectral.peak_nm"``).
        value: Arbitrary payload. Not introspected by the blackboard.
        author: Identifier of the agent that posted the entry.
        confidence: Author-reported confidence in ``[0.0, 1.0]``.
        entry_type: Coarse semantic type such as ``"observation"``,
            ``"decision"``, ``"risk"``, or ``"artifact"``.
        tags: Optional topic tags for filtering and replay surfaces.
        metadata: Optional structured details that should travel with the
            entry but not be rendered as the primary value.
        timestamp: Monotonic timestamp (``time.monotonic``) of the write.
            Monotonic time is used so ordering is robust to wall-clock changes;
            it is only meaningful relative to other entries in the same process.
    """

    key: str
    value: Any
    author: str
    confidence: float = 1.0
    entry_type: str = "observation"
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("key must be a non-empty string")
        if not self.author:
            raise ValueError("author must be a non-empty string")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence!r}"
            )
        if not self.entry_type:
            raise ValueError("entry_type must be a non-empty string")
        object.__setattr__(self, "tags", tuple(self.tags))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable snapshot of the entry."""
        return {
            "key": self.key,
            "value": _json_safe(self.value),
            "author": self.author,
            "confidence": self.confidence,
            "entry_type": self.entry_type,
            "tags": list(self.tags),
            "metadata": _json_safe(self.metadata),
            "timestamp": self.timestamp,
        }


class RoundBlackboard:
    """Async-first, round-scoped shared working memory.

    Instances are normally obtained from
    :meth:`BlackboardManager.get_or_create` rather than constructed directly,
    so that all agents in a round share the same board.
    """

    __slots__ = (
        "_round_id",
        "_campaign_id",
        "_entries",
        "_exact_subs",
        "_prefix_subs",
        "_lock",
    )

    def __init__(self, round_id: str, campaign_id: str) -> None:
        self._round_id: str = round_id
        self._campaign_id: str = campaign_id
        self._entries: dict[str, BlackboardEntry] = {}
        # Exact-key subscriptions: key -> set of queues.
        self._exact_subs: dict[str, set[asyncio.Queue[BlackboardEntry]]] = {}
        # Prefix subscriptions: prefix -> set of queues.
        self._prefix_subs: dict[str, set[asyncio.Queue[BlackboardEntry]]] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

        logger.info(
            "blackboard.created",
            extra={
                "round_id": round_id,
                "campaign_id": campaign_id,
            },
        )

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def round_id(self) -> str:
        """The round this blackboard is scoped to."""
        return self._round_id

    @property
    def campaign_id(self) -> str:
        """The campaign this blackboard belongs to."""
        return self._campaign_id

    @property
    def size(self) -> int:
        """Number of distinct keys currently stored."""
        return len(self._entries)

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    async def write(
        self,
        key: str,
        value: Any,
        *,
        author: str,
        confidence: float = 1.0,
        entry_type: str = "observation",
        tags: tuple[str, ...] | list[str] = (),
        metadata: dict[str, Any] | None = None,
    ) -> BlackboardEntry:
        """Store a discovery and broadcast it to interested subscribers.

        The store update and fan-out are performed atomically under the write
        lock so subscribers can never observe a notification for an entry that
        is not yet readable via :meth:`read`.

        Args:
            key: Logical address of the discovery.
            value: Arbitrary payload.
            author: Identifier of the posting agent.
            confidence: Author-reported confidence in ``[0.0, 1.0]``.
            entry_type: Coarse semantic type for downstream filtering.
            tags: Optional topic tags.
            metadata: Optional structured sidecar payload.

        Returns:
            The stored :class:`BlackboardEntry`.

        Raises:
            ValueError: If ``key`` is empty or ``confidence`` is out of range.
        """
        if not key:
            raise ValueError("key must be a non-empty string")

        entry = BlackboardEntry(
            key=key,
            value=value,
            author=author,
            confidence=confidence,
            entry_type=entry_type,
            tags=tuple(tags),
            metadata=metadata or {},
        )

        async with self._lock:
            previous = self._entries.get(key)
            self._entries[key] = entry
            targets = self._collect_targets(key)

            logger.debug(
                "blackboard.write",
                extra={
                    "round_id": self._round_id,
                    "campaign_id": self._campaign_id,
                    "key": key,
                    "author": author,
                    "confidence": confidence,
                    "entry_type": entry_type,
                    "overwrite": previous is not None,
                    "subscriber_count": len(targets),
                    "size": len(self._entries),
                },
            )

            # Fan out. put_nowait is safe for unbounded queues (the default);
            # if a subscriber opted into a bounded queue and it is full we drop
            # the notification rather than block the writer or other readers.
            for queue in targets:
                try:
                    queue.put_nowait(entry)
                except asyncio.QueueFull:
                    logger.warning(
                        "blackboard.subscriber_queue_full",
                        extra={
                            "round_id": self._round_id,
                            "key": key,
                        },
                    )

        return entry

    def _collect_targets(
        self, key: str
    ) -> set[asyncio.Queue[BlackboardEntry]]:
        """Return the set of queues that should be notified for ``key``.

        Must be called while holding ``self._lock``.
        """
        targets: set[asyncio.Queue[BlackboardEntry]] = set()
        exact = self._exact_subs.get(key)
        if exact:
            targets.update(exact)
        for prefix, queues in self._prefix_subs.items():
            if key.startswith(prefix):
                targets.update(queues)
        return targets

    # ------------------------------------------------------------------ #
    # Reads (non-blocking, lock-free)
    # ------------------------------------------------------------------ #
    def read(self, key: str) -> BlackboardEntry | None:
        """Return the latest entry for ``key``, or ``None`` if absent."""
        return self._entries.get(key)

    def read_all(self) -> dict[str, BlackboardEntry]:
        """Return a shallow snapshot of all current entries.

        The returned dict is a copy, so callers may iterate it freely even if
        concurrent writes occur.
        """
        return dict(self._entries)

    def snapshot(self) -> dict[str, Any]:
        """Return a serializable snapshot of the current board."""
        return {
            "round_id": self._round_id,
            "campaign_id": self._campaign_id,
            "entries": {
                key: entry.to_dict()
                for key, entry in sorted(self._entries.items())
            },
            "size": self.size,
        }

    def keys_with_prefix(self, prefix: str) -> list[str]:
        """Return all stored keys that begin with ``prefix``."""
        return [k for k in self._entries if k.startswith(prefix)]

    # ------------------------------------------------------------------ #
    # Subscriptions
    # ------------------------------------------------------------------ #
    def subscribe(self, key: str) -> asyncio.Queue[BlackboardEntry]:
        """Subscribe to writes for an exact ``key``.

        Returns:
            An unbounded :class:`asyncio.Queue` that receives a
            :class:`BlackboardEntry` on each matching write.
        """
        queue: asyncio.Queue[BlackboardEntry] = asyncio.Queue()
        self._exact_subs.setdefault(key, set()).add(queue)
        logger.debug(
            "blackboard.subscribe",
            extra={
                "round_id": self._round_id,
                "key": key,
                "kind": "exact",
            },
        )
        return queue

    def subscribe_prefix(self, prefix: str) -> asyncio.Queue[BlackboardEntry]:
        """Subscribe to writes for any key beginning with ``prefix``.

        Example:
            ``subscribe_prefix("spectral.")`` is notified on writes to
            ``"spectral.peak_nm"``, ``"spectral.fwhm"``, etc.

        Returns:
            An unbounded :class:`asyncio.Queue` that receives a
            :class:`BlackboardEntry` on each matching write.
        """
        queue: asyncio.Queue[BlackboardEntry] = asyncio.Queue()
        self._prefix_subs.setdefault(prefix, set()).add(queue)
        logger.debug(
            "blackboard.subscribe",
            extra={
                "round_id": self._round_id,
                "prefix": prefix,
                "kind": "prefix",
            },
        )
        return queue

    def unsubscribe(self, q: asyncio.Queue[BlackboardEntry]) -> None:
        """Remove ``q`` from all exact and prefix subscriptions.

        Safe to call with a queue that is not currently subscribed; empty
        subscription buckets are pruned to keep fan-out cheap.
        """
        removed = False
        for table in (self._exact_subs, self._prefix_subs):
            empty_buckets: list[str] = []
            for selector, queues in table.items():
                if q in queues:
                    queues.discard(q)
                    removed = True
                if not queues:
                    empty_buckets.append(selector)
            for selector in empty_buckets:
                del table[selector]

        logger.debug(
            "blackboard.unsubscribe",
            extra={
                "round_id": self._round_id,
                "found": removed,
            },
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"RoundBlackboard(round_id={self._round_id!r}, "
            f"campaign_id={self._campaign_id!r}, size={self.size})"
        )


class BlackboardManager:
    """Process-wide registry of active round blackboards.

    Use the module-level :data:`manager` singleton in application code. The
    class is exported primarily for typing and testing (where an isolated
    instance is convenient).
    """

    __slots__ = ("_boards", "_lock")

    def __init__(self) -> None:
        self._boards: dict[tuple[str, str], RoundBlackboard] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

    @staticmethod
    def _key(round_id: str, campaign_id: str) -> tuple[str, str]:
        return (campaign_id, round_id)

    def get_or_create(
        self, round_id: str, campaign_id: str
    ) -> RoundBlackboard:
        """Return the blackboard for ``round_id``, creating it if needed.

        Note:
            This is synchronous on purpose so that agents can obtain the shared
            board cheaply at startup. The asyncio event loop is single-threaded,
            so the check-then-create below is not subject to task interleaving
            (no ``await`` occurs between the lookup and the insert).
        """
        board_key = self._key(round_id, campaign_id)
        board = self._boards.get(board_key)
        if board is None:
            board = RoundBlackboard(round_id=round_id, campaign_id=campaign_id)
            self._boards[board_key] = board
            logger.info(
                "blackboard.manager.created",
                extra={
                    "round_id": round_id,
                    "campaign_id": campaign_id,
                    "active_count": len(self._boards),
                },
            )
        return board

    def get_existing(
        self, round_id: str, campaign_id: str | None = None
    ) -> RoundBlackboard | None:
        """Return an active board without creating a new one.

        Context injection uses this read-only path so observing a round does
        not create empty boards that must later be cleaned up.
        """
        if campaign_id is not None:
            return self._boards.get(self._key(round_id, campaign_id))

        matches = [
            board
            for (board_campaign_id, board_round_id), board in self._boards.items()
            if board_round_id == round_id
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            logger.warning(
                "blackboard.manager.ambiguous_round_lookup",
                extra={"round_id": round_id, "match_count": len(matches)},
            )
        return None

    def discard(
        self,
        round_id: str,
        campaign_id: str | None = None,
    ) -> RoundBlackboard | None:
        """Drop the blackboard for ``round_id`` (called when a round completes).

        Returns:
            The discarded :class:`RoundBlackboard`, or ``None`` if no board was
            registered under ``round_id``.
        """
        if campaign_id is not None:
            board = self._boards.pop(self._key(round_id, campaign_id), None)
        else:
            matches = [
                key
                for key in self._boards
                if key[1] == round_id
            ]
            if len(matches) != 1:
                if len(matches) > 1:
                    logger.warning(
                        "blackboard.manager.ambiguous_discard",
                        extra={"round_id": round_id, "match_count": len(matches)},
                    )
                board = None
            else:
                board = self._boards.pop(matches[0], None)
        if board is not None:
            logger.info(
                "blackboard.manager.discarded",
                extra={
                    "round_id": round_id,
                    "campaign_id": board.campaign_id,
                    "final_size": board.size,
                    "active_count": len(self._boards),
                },
            )
        else:
            logger.debug(
                "blackboard.manager.discard_missing",
                extra={"round_id": round_id},
            )
        return board

    def active_boards(self) -> dict[tuple[str, str], RoundBlackboard]:
        """Return a snapshot mapping ``(campaign_id, round_id)`` to boards."""
        return dict(self._boards)


# Module-level singleton used throughout the application.
manager: BlackboardManager = BlackboardManager()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json")
        except TypeError:
            return model_dump()
    return repr(value)
