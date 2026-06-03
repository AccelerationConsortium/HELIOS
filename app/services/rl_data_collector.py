"""Historical campaign data collector for RL training.

Extracts training data from completed campaigns stored in the orchestrator
database:
- Campaign snapshots at each round
- Actions taken (strategy decisions)
- Rewards (KPI improvements)
- Terminal outcomes

This module provides two layers:

1. **Streaming layer (async-first)** -- ``stream_campaign_transitions`` and
   ``stream_historical_transitions`` yield one ``RoundTransition`` at a time
   using cursor-based pagination (``LIMIT``/``OFFSET``). Rounds are
   deserialized in bounded chunks so memory stays flat even for campaigns
   with hundreds of rounds and thousands of candidates. A ``BackpressureFn``
   callback lets downstream RL agents pause/throttle ingestion (e.g. while a
   replay buffer flushes or a training step runs).

2. **Materialized layer (backward compatible)** -- ``extract_campaign_trace``
   and ``collect_historical_campaigns`` keep their original synchronous
   signatures and return the same trace dicts as before, but are now thin
   adapters over the streaming layer. Existing scripts continue to work
   unchanged.

Data format for offline training (materialized trace):
    {
        "campaign_id": "camp-abc123",
        "snapshots": [CampaignSnapshot, ...],  # one per round
        "actions": [0, 1, 2, ...],            # RL action indices
        "rewards": [0.05, -0.01, 0.12, ...],  # per-round rewards
        "kpi_history": [98.5, 98.7, 99.1, ...],
        "final_kpi": 99.3,
        "converged": True,
        "target_reached": False,
    }
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services.rl_reward import RewardConfig
from app.services.strategy_models import CampaignSnapshot

logger = logging.getLogger(__name__)

__all__ = [
    "RoundTransition",
    "CampaignTraceMeta",
    "BackpressureFn",
    "stream_campaign_transitions",
    "stream_historical_transitions",
    "collect_historical_campaigns",
    "extract_campaign_trace",
    "action_from_backend_name",
    "save_training_dataset",
    "load_training_dataset",
]


# ---------------------------------------------------------------------------
# Streaming primitives
# ---------------------------------------------------------------------------

# Default number of round rows pulled per DB page. Chosen to keep a single
# page's deserialized JSON well under a few MB even for wide candidate batches.
DEFAULT_ROUND_PAGE_SIZE = 50

# An async callback invoked once per yielded transition. Receivers can ``await``
# inside it to apply backpressure (sleep, wait on a semaphore, flush a buffer).
# Returning ``False`` signals the producer to stop streaming early.
BackpressureFn = Callable[["RoundTransition"], Awaitable[bool | None]]


@dataclass(frozen=True)
class RoundTransition:
    """One lazily-produced (round, snapshot, action, reward) training tuple.

    ``is_terminal`` marks the final round of the campaign so consumers can
    apply terminal-reward bootstrapping without buffering the whole trace.
    """

    campaign_id: str
    round_number: int
    snapshot: CampaignSnapshot
    action: int
    reward: float
    is_terminal: bool
    best_kpi_so_far: float | None


@dataclass(frozen=True)
class CampaignTraceMeta:
    """Per-campaign metadata resolved once before streaming its rounds."""

    campaign_id: str
    status: str
    direction: str
    max_rounds: int
    target_value: float | None
    n_dimensions: int
    has_categorical: bool
    has_log_scale: bool
    dimensions: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Backend name -> RL action mapping
# ---------------------------------------------------------------------------

def action_from_backend_name(backend: str) -> int:
    """Map backend name to RL action index.

    Args:
        backend: Backend name (e.g., "lhs", "built_in", "optuna_tpe")

    Returns:
        Action index (0-3)
    """
    backend_lower = backend.lower()

    if "lhs" in backend_lower or "random" in backend_lower or "grid" in backend_lower:
        return 0  # explore
    elif "bayesian" in backend_lower or "tpe" in backend_lower or "built_in" in backend_lower:
        return 1  # exploit
    elif "cmaes" in backend_lower or "de" in backend_lower or "refine" in backend_lower:
        return 2  # refine
    elif "stabilize" in backend_lower or "replicate" in backend_lower:
        return 3  # stabilize
    else:
        logger.warning("Unknown backend '%s', defaulting to action=1 (exploit)", backend)
        return 1


# ---------------------------------------------------------------------------
# Low-level blocking DB helpers (run via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _open_ro_connection(db_path: str) -> sqlite3.Connection:
    """Open a read-only-friendly connection with row access by column name."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _fetch_campaign_meta(db_path: str, campaign_id: str) -> CampaignTraceMeta | None:
    """Load campaign metadata (single small row). Blocking."""
    conn = _open_ro_connection(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM campaign_state WHERE campaign_id = ?",
            (campaign_id,),
        )
        row = cur.fetchone()
        if not row:
            logger.warning("Campaign %s not found", campaign_id)
            return None

        status = row["status"]
        if status not in ("completed", "failed"):
            logger.debug("Campaign %s not finished (status=%s)", campaign_id, status)
            return None

        input_json = json.loads(row["input_json"])
        dimensions = list(input_json.get("dimensions", []))
        return CampaignTraceMeta(
            campaign_id=campaign_id,
            status=status,
            direction=input_json.get("direction", "maximize"),
            max_rounds=input_json.get("max_rounds", 10),
            target_value=input_json.get("target_value"),
            n_dimensions=len(dimensions),
            has_categorical=any(d.get("choices") is not None for d in dimensions),
            has_log_scale=any(d.get("log_scale", False) for d in dimensions),
            dimensions=dimensions,
        )
    finally:
        conn.close()


def _count_rounds(db_path: str, campaign_id: str) -> int:
    """Count rounds for a campaign (cheap aggregate). Blocking."""
    conn = _open_ro_connection(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) AS n FROM campaign_rounds WHERE campaign_id = ?",
            (campaign_id,),
        )
        return int(cur.fetchone()["n"])
    finally:
        conn.close()


def _fetch_round_page(
    db_path: str,
    campaign_id: str,
    limit: int,
    offset: int,
) -> list[dict[str, Any]]:
    """Fetch one page of rounds, parsing JSON columns. Blocking.

    Returns plain dicts (not sqlite3.Row) so the result is detached from the
    connection and safe to hand back across the thread boundary.
    """
    conn = _open_ro_connection(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT round_number, strategy, n_candidates, kpis, params
            FROM campaign_rounds
            WHERE campaign_id = ?
            ORDER BY round_number
            LIMIT ? OFFSET ?
            """,
            (campaign_id, limit, offset),
        )
        page: list[dict[str, Any]] = []
        for r in cur.fetchall():
            kpis_raw = r["kpis"]
            params_raw = r["params"]
            page.append(
                {
                    "round_number": r["round_number"],
                    "strategy": r["strategy"] or "built_in",
                    "n_candidates": r["n_candidates"],
                    "kpis": json.loads(kpis_raw) if kpis_raw else [],
                    "params": json.loads(params_raw) if params_raw else [],
                }
            )
        return page
    finally:
        conn.close()


def _fetch_finished_campaign_ids(db_path: str) -> list[str]:
    """List completed/failed campaign IDs newest-first. Blocking."""
    conn = _open_ro_connection(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT campaign_id
            FROM campaign_state
            WHERE status IN ('completed', 'failed')
            ORDER BY created_at DESC
            """
        )
        return [row["campaign_id"] for row in cur.fetchall()]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Pure transition builder (no I/O) -- unit-testable in isolation
# ---------------------------------------------------------------------------

def _build_transition(
    meta: CampaignTraceMeta,
    round_row: dict[str, Any],
    *,
    cumulative_kpis: list[float],
    cumulative_params: list[dict[str, Any]],
    kpi_history: list[float],
    prev_best_kpi: float | None,
    is_terminal: bool,
    reward_config: RewardConfig,
) -> tuple[RoundTransition, float | None]:
    """Build a single ``RoundTransition`` from one round row.

    Mutates the passed accumulators in place (matching the original streaming
    semantics) and returns the transition plus the updated ``best_kpi``.

    The accumulators are the *only* unbounded structures; for very large
    campaigns callers may pass bounded/sliding-window lists to cap memory at
    the cost of kNN-signal fidelity.
    """
    round_num = round_row["round_number"]
    strategy = round_row["strategy"]
    round_kpis = round_row["kpis"]
    round_params = round_row["params"]

    cumulative_kpis.extend(round_kpis)
    cumulative_params.extend(round_params)

    if cumulative_kpis:
        best_kpi = max(cumulative_kpis) if meta.direction == "maximize" else min(cumulative_kpis)
    else:
        best_kpi = None

    snapshot = CampaignSnapshot(
        round_number=round_num,
        max_rounds=meta.max_rounds,
        n_observations=len(cumulative_kpis),
        n_dimensions=meta.n_dimensions,
        has_categorical=meta.has_categorical,
        has_log_scale=meta.has_log_scale,
        kpi_history=tuple(kpi_history),
        direction=meta.direction,
        last_batch_kpis=tuple(round_kpis),
        last_batch_params=tuple(round_params),
        best_kpi_so_far=best_kpi,
        all_params=tuple(cumulative_params),
        all_kpis=tuple(cumulative_kpis),
    )

    action = action_from_backend_name(strategy)

    if best_kpi is not None:
        if prev_best_kpi is not None:
            delta = best_kpi - prev_best_kpi
            if meta.direction == "minimize":
                delta = -delta
            reward = delta / reward_config.kpi_scale
        else:
            reward = 0.0  # first round has no improvement baseline
    else:
        reward = reward_config.round_cost  # failed round

    # Terminal bonus folded into the last round's reward.
    if is_terminal and best_kpi is not None and meta.target_value is not None:
        if meta.direction == "maximize":
            target_reached = best_kpi >= meta.target_value
        else:
            target_reached = best_kpi <= meta.target_value
        if target_reached:
            reward += reward_config.convergence_bonus * reward_config.gamma

    # Advance per-round KPI history *after* snapshot construction to preserve
    # the original ordering semantics (snapshot sees prior history only).
    kpi_history.extend(round_kpis)

    transition = RoundTransition(
        campaign_id=meta.campaign_id,
        round_number=round_num,
        snapshot=snapshot,
        action=action,
        reward=reward,
        is_terminal=is_terminal,
        best_kpi_so_far=best_kpi,
    )
    return transition, best_kpi


# ---------------------------------------------------------------------------
# Async streaming layer
# ---------------------------------------------------------------------------

async def stream_campaign_transitions(
    campaign_id: str,
    db_path: str = "helios.db",
    *,
    reward_config: RewardConfig | None = None,
    page_size: int = DEFAULT_ROUND_PAGE_SIZE,
    backpressure: BackpressureFn | None = None,
) -> AsyncIterator[RoundTransition]:
    """Lazily yield ``RoundTransition`` tuples for one campaign.

    Rounds are read in pages of ``page_size`` via ``LIMIT``/``OFFSET`` so the
    database is never asked to materialize the whole campaign at once. All
    blocking ``sqlite3`` work runs in a worker thread, keeping the event loop
    responsive.

    The ``backpressure`` callback (if provided) is awaited after each yielded
    transition; returning ``False`` from it stops the stream early.
    """
    if reward_config is None:
        reward_config = RewardConfig()
    if page_size < 1:
        raise ValueError("page_size must be >= 1")

    meta = await asyncio.to_thread(_fetch_campaign_meta, db_path, campaign_id)
    if meta is None:
        return

    total_rounds = await asyncio.to_thread(_count_rounds, db_path, campaign_id)
    if total_rounds == 0:
        logger.warning("Campaign %s has no rounds", campaign_id)
        return

    cumulative_kpis: list[float] = []
    cumulative_params: list[dict[str, Any]] = []
    kpi_history: list[float] = []
    prev_best_kpi: float | None = None
    emitted = 0
    offset = 0

    while offset < total_rounds:
        page = await asyncio.to_thread(
            _fetch_round_page, db_path, campaign_id, page_size, offset
        )
        if not page:
            break

        for round_row in page:
            emitted += 1
            is_terminal = emitted == total_rounds
            transition, prev_best_kpi = _build_transition(
                meta,
                round_row,
                cumulative_kpis=cumulative_kpis,
                cumulative_params=cumulative_params,
                kpi_history=kpi_history,
                prev_best_kpi=prev_best_kpi,
                is_terminal=is_terminal,
                reward_config=reward_config,
            )
            yield transition

            if backpressure is not None:
                should_continue = await backpressure(transition)
                if should_continue is False:
                    logger.info(
                        "Backpressure requested stop for campaign %s at round %d",
                        campaign_id,
                        transition.round_number,
                    )
                    return

        offset += len(page)

    logger.debug("Streamed %d transitions for campaign %s", emitted, campaign_id)


async def stream_historical_transitions(
    db_path: str = "helios.db",
    *,
    min_rounds: int = 3,
    reward_config: RewardConfig | None = None,
    page_size: int = DEFAULT_ROUND_PAGE_SIZE,
    backpressure: BackpressureFn | None = None,
) -> AsyncIterator[RoundTransition]:
    """Lazily yield transitions across *all* finished campaigns.

    Campaigns shorter than ``min_rounds`` are skipped. Only one campaign's
    accumulators are resident at a time, so memory is bounded by the largest
    single campaign rather than the full corpus.
    """
    if reward_config is None:
        reward_config = RewardConfig()

    campaign_ids = await asyncio.to_thread(_fetch_finished_campaign_ids, db_path)
    logger.info("Found %d finished campaigns", len(campaign_ids))

    for campaign_id in campaign_ids:
        n_rounds = await asyncio.to_thread(_count_rounds, db_path, campaign_id)
        if n_rounds < min_rounds:
            logger.debug(
                "Skipping campaign %s (%d rounds < min_rounds=%d)",
                campaign_id,
                n_rounds,
                min_rounds,
            )
            continue

        async for transition in stream_campaign_transitions(
            campaign_id,
            db_path,
            reward_config=reward_config,
            page_size=page_size,
            backpressure=backpressure,
        ):
            yield transition


# ---------------------------------------------------------------------------
# Materialized adapters (backward compatible, sync API preserved)
# ---------------------------------------------------------------------------

def _materialize_trace(
    meta: CampaignTraceMeta,
    transitions: list[RoundTransition],
) -> dict[str, Any]:
    """Assemble the legacy trace dict from an ordered transition list."""
    snapshots = [t.snapshot for t in transitions]
    actions = [t.action for t in transitions]
    rewards = [t.reward for t in transitions]
    final_kpi = transitions[-1].best_kpi_so_far

    target_reached = False
    if meta.target_value is not None and final_kpi is not None:
        if meta.direction == "maximize":
            target_reached = final_kpi >= meta.target_value
        else:
            target_reached = final_kpi <= meta.target_value

    # Reconstruct the flat kpi_history the legacy format exposed.
    kpi_history: list[float] = []
    for s in snapshots:
        kpi_history.extend(s.last_batch_kpis)

    return {
        "campaign_id": meta.campaign_id,
        "snapshots": snapshots,
        "actions": actions,
        "rewards": rewards,
        "kpi_history": kpi_history,
        "final_kpi": final_kpi,
        "converged": meta.status == "completed",
        "target_reached": target_reached,
        "direction": meta.direction,
        "n_rounds": len(transitions),
    }


async def extract_campaign_trace_async(
    campaign_id: str,
    db_path: str = "helios.db",
    reward_config: RewardConfig | None = None,
    *,
    page_size: int = DEFAULT_ROUND_PAGE_SIZE,
) -> dict[str, Any] | None:
    """Async materializer: drain the stream into the legacy trace dict."""
    if reward_config is None:
        reward_config = RewardConfig()

    meta = await asyncio.to_thread(_fetch_campaign_meta, db_path, campaign_id)
    if meta is None:
        return None

    transitions: list[RoundTransition] = []
    try:
        async for transition in stream_campaign_transitions(
            campaign_id,
            db_path,
            reward_config=reward_config,
            page_size=page_size,
        ):
            transitions.append(transition)
    except Exception as exc:
        logger.error(
            "Failed to extract campaign %s: %s", campaign_id, exc, exc_info=True
        )
        return None

    if not transitions:
        logger.warning("Campaign %s produced no transitions", campaign_id)
        return None

    return _materialize_trace(meta, transitions)


def _run_async(coro: Awaitable[Any]) -> Any:
    """Run an async coroutine from a synchronous context.

    Raises if called from within a running event loop -- callers inside async
    code should use the ``*_async`` variants directly instead.
    """
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is not None:
        raise RuntimeError(
            "extract_campaign_trace/collect_historical_campaigns are synchronous "
            "and cannot be called from a running event loop; use "
            "extract_campaign_trace_async / stream_historical_transitions instead."
        )
    return asyncio.run(coro)


def extract_campaign_trace(
    campaign_id: str,
    db_path: str = "helios.db",
    reward_config: RewardConfig | None = None,
) -> dict[str, Any] | None:
    """Extract training data from a single campaign (backward-compatible sync API).

    Now backed by the streaming layer: rounds are paginated rather than loaded
    in one query. Returns the same trace dict shape as before.
    """
    return _run_async(
        extract_campaign_trace_async(campaign_id, db_path, reward_config)
    )


async def collect_historical_campaigns_async(
    db_path: str = "helios.db",
    min_rounds: int = 3,
    reward_config: RewardConfig | None = None,
    *,
    page_size: int = DEFAULT_ROUND_PAGE_SIZE,
) -> list[dict[str, Any]]:
    """Async collector: materialize one campaign at a time to bound memory."""
    if reward_config is None:
        reward_config = RewardConfig()

    campaign_ids = await asyncio.to_thread(_fetch_finished_campaign_ids, db_path)
    logger.info("Found %d finished campaigns", len(campaign_ids))

    traces: list[dict[str, Any]] = []
    for campaign_id in campaign_ids:
        trace = await extract_campaign_trace_async(
            campaign_id, db_path, reward_config, page_size=page_size
        )
        if trace is not None and trace["n_rounds"] >= min_rounds:
            traces.append(trace)

    logger.info(
        "Extracted %d valid campaign traces (min_rounds=%d)", len(traces), min_rounds
    )
    return traces


def collect_historical_campaigns(
    db_path: str = "helios.db",
    min_rounds: int = 3,
    reward_config: RewardConfig | None = None,
) -> list[dict[str, Any]]:
    """Collect training data from all completed campaigns (backward-compatible sync API).

    Materializes one campaign at a time instead of accumulating every round of
    every campaign before processing, so peak memory scales with the largest
    single campaign rather than the whole corpus.
    """
    return _run_async(
        collect_historical_campaigns_async(db_path, min_rounds, reward_config)
    )


# ---------------------------------------------------------------------------
# Save/Load training dataset
# ---------------------------------------------------------------------------

def _snapshot_to_dict(s: CampaignSnapshot) -> dict[str, Any]:
    return {
        "round_number": s.round_number,
        "max_rounds": s.max_rounds,
        "n_observations": s.n_observations,
        "n_dimensions": s.n_dimensions,
        "has_categorical": s.has_categorical,
        "has_log_scale": s.has_log_scale,
        "kpi_history": list(s.kpi_history),
        "direction": s.direction,
        "last_batch_kpis": list(s.last_batch_kpis),
        "last_batch_params": list(s.last_batch_params),
        "best_kpi_so_far": s.best_kpi_so_far,
        "all_params": list(s.all_params),
        "all_kpis": list(s.all_kpis),
    }


def _dict_to_snapshot(s: dict[str, Any]) -> CampaignSnapshot:
    return CampaignSnapshot(
        round_number=s["round_number"],
        max_rounds=s["max_rounds"],
        n_observations=s["n_observations"],
        n_dimensions=s["n_dimensions"],
        has_categorical=s["has_categorical"],
        has_log_scale=s["has_log_scale"],
        kpi_history=tuple(s["kpi_history"]),
        direction=s["direction"],
        last_batch_kpis=tuple(s["last_batch_kpis"]),
        last_batch_params=tuple(s["last_batch_params"]),
        best_kpi_so_far=s["best_kpi_so_far"],
        all_params=tuple(s["all_params"]),
        all_kpis=tuple(s["all_kpis"]),
    )


def save_training_dataset(
    traces: list[dict[str, Any]],
    output_path: str = "models/rl_training_data.json",
) -> None:
    """Save training dataset to JSON.

    Snapshots are serialized to dicts for JSON compatibility.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    serialized_traces = []
    for trace in traces:
        serialized = dict(trace)
        serialized["snapshots"] = [_snapshot_to_dict(s) for s in trace["snapshots"]]
        serialized_traces.append(serialized)

    with open(output, "w") as f:
        json.dump(serialized_traces, f, indent=2)

    logger.info("Saved %d campaign traces to %s", len(traces), output)


def load_training_dataset(
    input_path: str = "models/rl_training_data.json",
) -> list[dict[str, Any]]:
    """Load training dataset from JSON.

    Deserializes snapshot dicts back to CampaignSnapshot objects.
    """
    with open(input_path) as f:
        serialized_traces = json.load(f)

    traces = []
    for serialized in serialized_traces:
        trace = dict(serialized)
        trace["snapshots"] = [_dict_to_snapshot(s) for s in serialized["snapshots"]]
        traces.append(trace)

    logger.info("Loaded %d campaign traces from %s", len(traces), input_path)
    return traces


async def stream_training_dataset(
    input_path: str = "models/rl_training_data.json",
) -> AsyncIterator[dict[str, Any]]:
    """Yield persisted traces one at a time (file read offloaded to a thread).

    Convenience helper so warm-start / replay-buffer loaders can iterate the
    on-disk dataset without holding every trace in memory simultaneously.
    """
    def _load() -> list[dict[str, Any]]:
        with open(input_path) as f:
            return json.load(f)

    serialized_traces = await asyncio.to_thread(_load)
    for serialized in serialized_traces:
        trace = dict(serialized)
        trace["snapshots"] = [_dict_to_snapshot(s) for s in serialized["snapshots"]]
        yield trace
