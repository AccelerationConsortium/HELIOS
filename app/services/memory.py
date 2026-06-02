"""Three-layer semantic memory system — episodic, semantic, and procedural.

All memory reads are **advisory only** — wrapped in try/except, never block
planning or grounding.  Write path is post-run async via event_bus listener.

Layers:
1. **Episodic**: Raw per-step records (run X used primitive Y with params Z → succeeded/failed)
2. **Semantic**: Aggregated facts (success rates, parameter priors via Welford's algorithm)
3. **Procedural**: Repair/fallback recipes (if primitive X fails with error Y → recovery steps)

Listener lifecycle
------------------
The async write path is driven by a single :class:`MemoryListener` subscribed
to ``run.completed`` events on the event bus.  Historically this module held a
bare module-level ``_listener_task`` that was overwritten — but never cancelled
— on every ``start_memory_listener()`` call.  When a campaign was resumed or
restarted, each invocation leaked the prior task: the orphaned coroutine kept
its ``Subscription`` registered on the bus, kept consuming dispatched events,
and kept writing to memory tables under a stale campaign context.  Over many
resumes this produced unbounded task growth (OOM risk) and cross-campaign state
leakage (data corruption risk).

:class:`MemoryListener` fixes this by:

1. Tracking the ``campaign_id`` bound to the live listener.
2. Making :meth:`MemoryListener.start` idempotent — a second start for the same
   campaign returns the existing subscription instead of spawning a duplicate.
3. Enforcing a single live listener: starting for a *different* campaign
   cancels and unsubscribes the previous one first (lazy re-initialisation).
4. Exposing explicit lifecycle state so tests can assert clean teardown
   between campaigns.

The module-level :func:`start_memory_listener` / :func:`stop_memory_listener`
helpers remain for backward compatibility with ``app.main`` — they delegate to a
process-wide singleton, but the class itself is dependency-injectable and holds
no hidden global state, so unit tests can construct isolated instances.
"""
from __future__ import annotations

import asyncio
import logging
import math
import uuid
from dataclasses import dataclass
from typing import Any

from app.core.db import json_dumps, parse_json, run_txn, utcnow_iso

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Episode:
    """A single step outcome record."""

    run_id: str
    step_key: str
    primitive: str
    params: dict[str, Any]
    outcome: str  # "succeeded" | "failed"
    error: str | None = None


@dataclass
class ParamPrior:
    """Aggregated parameter statistics from semantic memory."""

    mean: float
    stddev: float
    sample_count: int


@dataclass
class RepairRecipe:
    """A procedural recovery recipe."""

    trigger_primitive: str
    trigger_error_pattern: str
    steps: list[dict[str, Any]]
    source: str
    hit_count: int


# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

_SEED_RECIPES: list[dict[str, Any]] = [
    {
        "trigger_primitive": "robot.aspirate",
        "trigger_error_pattern": "tip",
        "recipe": [
            {"primitive": "robot.drop_tip", "params": {}},
            {"primitive": "robot.pick_up_tip", "params": {}},
        ],
        "source": "seed",
    },
    {
        "trigger_primitive": "robot.dispense",
        "trigger_error_pattern": "tip",
        "recipe": [
            {"primitive": "robot.drop_tip", "params": {}},
            {"primitive": "robot.pick_up_tip", "params": {}},
        ],
        "source": "seed",
    },
]


def seed_initial_recipes() -> None:
    """Insert hardcoded recovery recipes if not already present.

    Idempotent — safe to call on every startup.
    """
    import sqlite3 as _sqlite3

    now = utcnow_iso()

    def _txn(conn: _sqlite3.Connection) -> None:
        for recipe in _SEED_RECIPES:
            existing = conn.execute(
                "SELECT id FROM memory_procedures "
                "WHERE trigger_primitive = ? AND trigger_error_pattern = ? AND source = 'seed'",
                (recipe["trigger_primitive"], recipe["trigger_error_pattern"]),
            ).fetchone()
            if existing is not None:
                continue
            conn.execute(
                "INSERT INTO memory_procedures "
                "(id, trigger_primitive, trigger_error_pattern, recipe_json, source, hit_count, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 0, ?)",
                (
                    str(uuid.uuid4()),
                    recipe["trigger_primitive"],
                    recipe["trigger_error_pattern"],
                    json_dumps(recipe["recipe"]),
                    recipe["source"],
                    now,
                ),
            )

    run_txn(_txn)


# ---------------------------------------------------------------------------
# Layer 1: Episodic — write path
# ---------------------------------------------------------------------------


def extract_episodes(run_id: str) -> list[Episode]:
    """Read run_steps for *run_id* and persist as episodic memory rows.

    Only steps with terminal status (succeeded / failed) are recorded.
    Returns the list of extracted episodes.
    """
    import sqlite3 as _sqlite3

    episodes: list[Episode] = []

    def _txn(conn: _sqlite3.Connection) -> list[Episode]:
        rows = conn.execute(
            "SELECT step_key, primitive, params_json, status, error "
            "FROM run_steps WHERE run_id = ? ORDER BY step_key ASC",
            (run_id,),
        ).fetchall()
        now = utcnow_iso()
        for row in rows:
            status = row["status"]
            if status == "succeeded":
                outcome = "succeeded"
            elif status == "failed":
                outcome = "failed"
            else:
                continue  # skip pending/running/skipped

            ep = Episode(
                run_id=run_id,
                step_key=row["step_key"],
                primitive=row["primitive"],
                params=parse_json(row["params_json"], {}),
                outcome=outcome,
                error=row["error"],
            )
            episodes.append(ep)
            conn.execute(
                "INSERT INTO memory_episodes "
                "(id, run_id, step_key, primitive, params_json, outcome, error, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    ep.run_id,
                    ep.step_key,
                    ep.primitive,
                    json_dumps(ep.params),
                    ep.outcome,
                    ep.error,
                    now,
                ),
            )
        return episodes

    return run_txn(_txn)


# ---------------------------------------------------------------------------
# Layer 2: Semantic — aggregation
# ---------------------------------------------------------------------------


def update_semantic_facts(episodes: list[Episode]) -> None:
    """Update running statistics in memory_semantic using Welford's algorithm.

    For each succeeded episode with numeric params, update the mean/stddev/count.
    Also updates success_rate across all outcomes for each primitive.
    """
    import sqlite3 as _sqlite3

    if not episodes:
        return

    now = utcnow_iso()

    def _txn(conn: _sqlite3.Connection) -> None:
        # Group by primitive for success_rate tracking
        primitive_outcomes: dict[str, dict[str, int]] = {}
        for ep in episodes:
            if ep.primitive not in primitive_outcomes:
                primitive_outcomes[ep.primitive] = {"succeeded": 0, "failed": 0}
            primitive_outcomes[ep.primitive][ep.outcome] = (
                primitive_outcomes[ep.primitive].get(ep.outcome, 0) + 1
            )

        for ep in episodes:
            if ep.outcome != "succeeded":
                continue
            for param_name, value in ep.params.items():
                # Only aggregate numeric values
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue

                float_val = float(value)

                # Read current state
                row = conn.execute(
                    "SELECT mean, stddev, sample_count, success_count, total_count "
                    "FROM memory_semantic WHERE primitive = ? AND param_name = ?",
                    (ep.primitive, param_name),
                ).fetchone()

                if row is None:
                    # First observation
                    conn.execute(
                        "INSERT INTO memory_semantic "
                        "(primitive, param_name, mean, stddev, sample_count, "
                        "success_rate, success_count, total_count, updated_at) "
                        "VALUES (?, ?, ?, 0.0, 1, 1.0, 1, 1, ?)",
                        (ep.primitive, param_name, float_val, now),
                    )
                else:
                    # Welford's online update
                    old_mean = row["mean"]
                    old_count = row["sample_count"]
                    old_success = row["success_count"]
                    old_total = row["total_count"]

                    n_new = old_count + 1
                    delta = float_val - old_mean
                    new_mean = old_mean + delta / n_new
                    # Reconstruct M2 from old stddev
                    old_m2 = (row["stddev"] ** 2) * old_count if old_count > 1 else 0.0
                    new_m2 = old_m2 + delta * (float_val - new_mean)
                    new_stddev = math.sqrt(new_m2 / n_new) if n_new > 1 else 0.0

                    new_success = old_success + 1
                    new_total = old_total + 1
                    new_rate = new_success / new_total if new_total > 0 else 0.0

                    conn.execute(
                        "UPDATE memory_semantic SET "
                        "mean = ?, stddev = ?, sample_count = ?, "
                        "success_rate = ?, success_count = ?, total_count = ?, updated_at = ? "
                        "WHERE primitive = ? AND param_name = ?",
                        (
                            new_mean,
                            new_stddev,
                            n_new,
                            new_rate,
                            new_success,
                            new_total,
                            now,
                            ep.primitive,
                            param_name,
                        ),
                    )

    run_txn(_txn)


# ---------------------------------------------------------------------------
# Layer 3: Procedural — repair pattern detection
# ---------------------------------------------------------------------------


def detect_repair_patterns(episodes: list[Episode]) -> None:
    """Scan episodes for failure → recovery sequences and record as recipes.

    If episode[i] failed and episode[j] (j > i) with the same primitive
    succeeded, record steps[i+1..j-1] as a repair recipe.
    """
    import sqlite3 as _sqlite3

    if len(episodes) < 2:
        return

    now = utcnow_iso()
    run_id = episodes[0].run_id if episodes else "unknown"

    def _txn(conn: _sqlite3.Connection) -> None:
        for i, ep in enumerate(episodes):
            if ep.outcome != "failed" or not ep.error:
                continue

            # Look for a later succeeded step with the same primitive
            for j in range(i + 1, len(episodes)):
                if (
                    episodes[j].primitive == ep.primitive
                    and episodes[j].outcome == "succeeded"
                ):
                    # Extract recovery steps between failure and success
                    recovery_steps = [
                        {"primitive": episodes[k].primitive, "params": episodes[k].params}
                        for k in range(i + 1, j)
                    ]
                    if not recovery_steps:
                        break  # No intermediate steps — not a real repair

                    # Use a simplified error pattern (first word of error)
                    error_pattern = ep.error.split()[0].lower() if ep.error else "unknown"

                    # Check if this recipe already exists
                    existing = conn.execute(
                        "SELECT id FROM memory_procedures "
                        "WHERE trigger_primitive = ? AND trigger_error_pattern = ?",
                        (ep.primitive, error_pattern),
                    ).fetchone()
                    if existing is None:
                        conn.execute(
                            "INSERT INTO memory_procedures "
                            "(id, trigger_primitive, trigger_error_pattern, recipe_json, "
                            "source, hit_count, updated_at) "
                            "VALUES (?, ?, ?, ?, ?, 0, ?)",
                            (
                                str(uuid.uuid4()),
                                ep.primitive,
                                error_pattern,
                                json_dumps(recovery_steps),
                                run_id,
                                now,
                            ),
                        )
                    break  # Only record first recovery per failure

    run_txn(_txn)


# ---------------------------------------------------------------------------
# Read path — advisory queries
# ---------------------------------------------------------------------------


def get_param_priors(primitive: str, param_name: str) -> ParamPrior | None:
    """Query semantic memory for parameter statistics.

    Returns None if no data exists.  Safe to call at any time — never raises.
    """
    import sqlite3 as _sqlite3

    def _txn(conn: _sqlite3.Connection) -> ParamPrior | None:
        row = conn.execute(
            "SELECT mean, stddev, sample_count FROM memory_semantic "
            "WHERE primitive = ? AND param_name = ?",
            (primitive, param_name),
        ).fetchone()
        if row is None:
            return None
        return ParamPrior(
            mean=row["mean"],
            stddev=row["stddev"],
            sample_count=row["sample_count"],
        )

    return run_txn(_txn)


def get_repair_recipes(primitive: str) -> list[RepairRecipe]:
    """Query procedural memory for repair recipes matching *primitive*."""
    import sqlite3 as _sqlite3

    def _txn(conn: _sqlite3.Connection) -> list[RepairRecipe]:
        rows = conn.execute(
            "SELECT trigger_primitive, trigger_error_pattern, recipe_json, "
            "source, hit_count FROM memory_procedures "
            "WHERE trigger_primitive = ?",
            (primitive,),
        ).fetchall()
        return [
            RepairRecipe(
                trigger_primitive=r["trigger_primitive"],
                trigger_error_pattern=r["trigger_error_pattern"],
                steps=parse_json(r["recipe_json"], []),
                source=r["source"],
                hit_count=r["hit_count"],
            )
            for r in rows
        ]

    return run_txn(_txn)


def increment_recipe_hit_count(trigger_primitive: str, trigger_error_pattern: str) -> None:
    """Increment hit_count for a matching procedural recipe.

    Advisory — never raises.  Called by the recovery engine after a
    recipe is successfully applied during adaptive execution.
    """
    import sqlite3 as _sqlite3

    def _txn(conn: _sqlite3.Connection) -> None:
        conn.execute(
            "UPDATE memory_procedures SET hit_count = hit_count + 1, updated_at = ? "
            "WHERE trigger_primitive = ? AND trigger_error_pattern = ?",
            (utcnow_iso(), trigger_primitive, trigger_error_pattern),
        )

    try:
        run_txn(_txn)
    except Exception:
        logger.debug("Failed to increment recipe hit_count", exc_info=True)


def format_memory_for_prompt(primitives: list[str] | None = None) -> str:
    """Build a concise memory context string for the LLM system prompt.

    Returns an empty string if no memory data exists.
    All errors are swallowed — memory is advisory.
    """
    import sqlite3 as _sqlite3

    try:
        sections: list[str] = []

        def _txn(conn: _sqlite3.Connection) -> None:
            # --- Recent outcome summary ---
            if primitives:
                placeholders = ",".join("?" for _ in primitives)
                query = (
                    f"SELECT primitive, outcome, COUNT(*) as cnt "
                    f"FROM memory_episodes WHERE primitive IN ({placeholders}) "
                    f"GROUP BY primitive, outcome ORDER BY primitive"
                )
                rows = conn.execute(query, primitives).fetchall()
            else:
                rows = conn.execute(
                    "SELECT primitive, outcome, COUNT(*) as cnt "
                    "FROM memory_episodes GROUP BY primitive, outcome "
                    "ORDER BY primitive"
                ).fetchall()

            if rows:
                # Aggregate success rates
                prim_stats: dict[str, dict[str, int]] = {}
                for r in rows:
                    prim = r["primitive"]
                    if prim not in prim_stats:
                        prim_stats[prim] = {"succeeded": 0, "failed": 0}
                    prim_stats[prim][r["outcome"]] = r["cnt"]

                outcome_lines: list[str] = []
                for prim, stats in sorted(prim_stats.items()):
                    total = stats["succeeded"] + stats["failed"]
                    rate = stats["succeeded"] / total * 100 if total > 0 else 0
                    outcome_lines.append(f"- {prim}: {rate:.0f}% success rate ({total} runs)")

                if outcome_lines:
                    sections.append("### Recent Outcomes\n" + "\n".join(outcome_lines))

            # --- Repair recipes ---
            if primitives:
                placeholders = ",".join("?" for _ in primitives)
                recipe_rows = conn.execute(
                    f"SELECT trigger_primitive, trigger_error_pattern, recipe_json "
                    f"FROM memory_procedures WHERE trigger_primitive IN ({placeholders})",
                    primitives,
                ).fetchall()
            else:
                recipe_rows = conn.execute(
                    "SELECT trigger_primitive, trigger_error_pattern, recipe_json "
                    "FROM memory_procedures"
                ).fetchall()

            if recipe_rows:
                recipe_lines: list[str] = []
                for r in recipe_rows:
                    steps = parse_json(r["recipe_json"], [])
                    step_names = " -> ".join(s.get("primitive", "?") for s in steps)
                    recipe_lines.append(
                        f"- If {r['trigger_primitive']} fails with "
                        f"\"{r['trigger_error_pattern']}\" error: {step_names}"
                    )
                if recipe_lines:
                    sections.append("### Known Recovery Recipes\n" + "\n".join(recipe_lines))

        run_txn(_txn)

        if not sections:
            return ""

        header = (
            "\n## Memory Context (Advisory)\n"
            "The following is based on past experiment outcomes. "
            "Use as guidance, not requirements.\n"
        )
        return header + "\n\n".join(sections) + "\n"

    except Exception:
        logger.debug("Memory prompt generation failed — returning empty", exc_info=True)
        return ""


# ---------------------------------------------------------------------------
# Event listener — async write path
# ---------------------------------------------------------------------------


async def _on_run_completed(run_id: str) -> None:
    """Process a completed run: extract episodes → update semantics → detect repairs."""
    try:
        episodes = extract_episodes(run_id)
        if episodes:
            update_semantic_facts(episodes)
            detect_repair_patterns(episodes)
        logger.debug("Memory updated for run %s (%d episodes)", run_id, len(episodes))
    except Exception:
        logger.warning("Memory extraction failed for run %s", run_id, exc_info=True)


class MemoryListener:
    """Single-instance, campaign-aware async listener for the memory write path.

    Holds no module-level global state — construct one per process (or per test)
    and inject the event bus into :meth:`start`.  The instance guarantees that
    at most one live subscription + task pair exists at a time, eliminating the
    task-leak / cross-campaign-leakage class of bugs.
    """

    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None
        self._sub: Any | None = None
        self._bus: Any | None = None
        self._campaign_id: str | None = None
        # Serialises concurrent start/stop calls (e.g. overlapping campaign
        # resume + shutdown) so we never race on the task/sub fields.
        self._lock = asyncio.Lock()

    # -- introspection (used by tests + diagnostics) -------------------------

    @property
    def is_running(self) -> bool:
        """True iff a live listener task is currently active."""
        return self._task is not None and not self._task.done()

    @property
    def campaign_id(self) -> str | None:
        """The campaign currently bound to the live listener, if any."""
        return self._campaign_id

    # -- lifecycle -----------------------------------------------------------

    async def start(self, bus: Any, campaign_id: str | None = None) -> Any:
        """Subscribe to the bus and begin processing ``run.completed`` events.

        Idempotent and deduplicating:

        * If a listener is already live for the same *campaign_id* (including
          the ``None`` global case), the existing subscription is returned and
          **no** new task is created.
        * If a listener is live for a *different* campaign, it is cancelled and
          unsubscribed first, then a fresh one is created (lazy re-init).  This
          enforces a single global listener and prevents orphaned tasks from
          accumulating across campaign resumes/restarts.
        """
        async with self._lock:
            if self.is_running:
                if self._campaign_id == campaign_id:
                    logger.debug(
                        "Memory listener already running for campaign %s — reusing",
                        campaign_id,
                    )
                    return self._sub
                logger.info(
                    "Memory listener rebinding campaign %s → %s — cancelling prior listener",
                    self._campaign_id,
                    campaign_id,
                )
                await self._teardown_locked()

            sub = await bus.subscribe(run_id=None)  # global subscription
            self._sub = sub
            self._bus = bus
            self._campaign_id = campaign_id
            self._task = asyncio.create_task(
                self._listen(sub),
                name=f"memory-listener:{campaign_id or 'global'}",
            )
            logger.info("Memory listener started for campaign %s", campaign_id)
            return sub

    async def stop(self) -> None:
        """Cancel the listener task and unsubscribe.  Safe to call when idle."""
        async with self._lock:
            await self._teardown_locked()

    async def _teardown_locked(self) -> None:
        """Cancel + unsubscribe the live listener.  Caller must hold ``_lock``."""
        sub, bus, task = self._sub, self._bus, self._task
        self._sub = None
        self._bus = None
        self._task = None
        self._campaign_id = None

        if sub is not None:
            try:
                sub.cancel()
            except Exception:
                logger.debug("Memory listener subscription cancel failed", exc_info=True)
        if sub is not None and bus is not None:
            try:
                await bus.unsubscribe(sub)
            except Exception:
                logger.debug("Memory listener unsubscribe failed", exc_info=True)
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.debug("Memory listener task raised on shutdown", exc_info=True)

    async def _listen(self, sub: Any) -> None:
        async for event in sub:
            if event.action == "run.completed":
                run_id = event.run_id
                if run_id:
                    await _on_run_completed(run_id)


# ---------------------------------------------------------------------------
# Process-wide singleton + backward-compatible helpers
# ---------------------------------------------------------------------------
#
# ``app.main`` calls the module-level helpers below.  They delegate to a single
# shared :class:`MemoryListener` so the legacy call sites keep working while the
# class remains independently constructible for unit tests.

_default_listener = MemoryListener()


async def start_memory_listener(bus: Any, campaign_id: str | None = None) -> Any:
    """Start the process-wide memory listener.  Returns the Subscription handle.

    Backward compatible with the previous signature ``start_memory_listener(bus)``.
    Repeated calls no longer leak tasks — see :meth:`MemoryListener.start`.
    """
    return await _default_listener.start(bus, campaign_id=campaign_id)


async def stop_memory_listener(sub: Any | None = None, bus: Any | None = None) -> None:
    """Stop the process-wide memory listener.

    *sub* and *bus* are accepted for signature compatibility with the previous
    implementation but are no longer required — the listener owns its own
    subscription and bus references.
    """
    await _default_listener.stop()
