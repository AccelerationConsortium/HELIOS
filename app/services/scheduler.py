from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from app.core.config import get_settings
from app.services.run_service import claim_schedulable_runs, mark_run_failed_if_running, trigger_due_campaigns

logger = logging.getLogger(__name__)


class OrchestratorScheduler:
    """Scheduler that dispatches runs as in-process threads.

    Changed from subprocess isolation to ``asyncio.to_thread`` so that
    hardware adapters holding persistent TCP / serial connections can be
    used across steps within a single run.

    Admission control
    -----------------
    Worker spawning is bounded by an :class:`asyncio.Semaphore` sized to
    ``settings.max_concurrent_workers``. The dispatch loop only *claims* as
    many runs as there is free capacity for. This matters because
    ``claim_schedulable_runs`` transitions runs from SCHEDULED to RUNNING in
    the database: claiming more than we can spawn would strand runs in a
    RUNNING state with no worker behind them. By gating the claim count on the
    semaphore's free permits, excess runs stay SCHEDULED and are naturally
    re-claimed on a later poll once workers complete — providing backpressure
    instead of unbounded thread growth.
    """

    def __init__(self, max_concurrent_workers: int | None = None) -> None:
        self._settings = get_settings()
        self._max_concurrent_workers = (
            max_concurrent_workers
            if max_concurrent_workers is not None
            else self._settings.max_concurrent_workers
        )
        if self._max_concurrent_workers < 1:
            raise ValueError("max_concurrent_workers must be >= 1")

        self._tasks: list[asyncio.Task[Any]] = []
        self._active_workers: dict[str, asyncio.Task[Any]] = {}
        self._orchestrator_tasks: dict[str, asyncio.Task[Any]] = {}
        self._stopped = asyncio.Event()

        # Admission gate. One permit per allowed concurrent worker.
        self._admission = asyncio.Semaphore(self._max_concurrent_workers)
        # Tracks which run_ids currently hold a permit so reaping releases
        # exactly once even if a worker is cancelled mid-flight.
        self._permits_held: set[str] = set()

    @property
    def available_capacity(self) -> int:
        """Number of additional workers that may be spawned right now."""
        return max(0, self._max_concurrent_workers - len(self._permits_held))

    async def start(self) -> None:
        self._stopped.clear()
        logger.info(
            "scheduler.start",
            extra={"max_concurrent_workers": self._max_concurrent_workers},
        )
        self._tasks = [
            asyncio.create_task(self._campaign_loop(), name="campaign-loop"),
            asyncio.create_task(self._dispatch_loop(), name="dispatch-loop"),
            asyncio.create_task(self._reap_loop(), name="reap-loop"),
            asyncio.create_task(self._orchestrator_reap_loop(), name="orchestrator-reap-loop"),
        ]

    async def stop(self) -> None:
        self._stopped.set()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()

        # Cancel active worker tasks
        for run_id, task in list(self._active_workers.items()):
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            mark_run_failed_if_running(run_id, "worker terminated during scheduler shutdown")
            self._release_permit(run_id)
        self._active_workers.clear()

        # Cancel active orchestrator tasks
        for _campaign_id, task in list(self._orchestrator_tasks.items()):
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._orchestrator_tasks.clear()

    async def _campaign_loop(self) -> None:
        while not self._stopped.is_set():
            await asyncio.to_thread(trigger_due_campaigns)
            await asyncio.sleep(self._settings.campaign_poll_seconds)

    async def _dispatch_loop(self) -> None:
        while not self._stopped.is_set():
            capacity = self.available_capacity
            if capacity <= 0:
                # Fully saturated: do not claim runs we cannot service, else
                # they would be marked RUNNING in the DB with no worker.
                logger.debug(
                    "scheduler.dispatch.saturated",
                    extra={
                        "active_workers": len(self._permits_held),
                        "max_concurrent_workers": self._max_concurrent_workers,
                    },
                )
                await asyncio.sleep(self._settings.scheduler_poll_seconds)
                continue

            run_ids = await asyncio.to_thread(claim_schedulable_runs, capacity)
            for run_id in run_ids:
                await self._spawn_worker(run_id)
            await asyncio.sleep(self._settings.scheduler_poll_seconds)

    async def _reap_loop(self) -> None:
        while not self._stopped.is_set():
            await self._reap_workers()
            await asyncio.sleep(1)

    async def _spawn_worker(self, run_id: str) -> None:
        if run_id in self._active_workers:
            return

        # Acquire an admission permit. Because the dispatch loop already
        # bounded the claim count to ``available_capacity``, this should not
        # block; the explicit acquire is the authoritative gate that keeps
        # ``_permits_held`` and the semaphore in lockstep.
        await self._admission.acquire()
        self._permits_held.add(run_id)

        # Import here to avoid circular imports
        from app.worker import execute_run

        task = asyncio.create_task(
            asyncio.to_thread(execute_run, run_id),
            name=f"worker-{run_id}",
        )
        self._active_workers[run_id] = task
        logger.info(
            "scheduler.worker.spawned",
            extra={
                "run_id": run_id,
                "active_workers": len(self._permits_held),
                "max_concurrent_workers": self._max_concurrent_workers,
            },
        )

    def _release_permit(self, run_id: str) -> None:
        """Release the admission permit held by ``run_id`` exactly once."""
        if run_id in self._permits_held:
            self._permits_held.discard(run_id)
            self._admission.release()

    async def _reap_workers(self) -> None:
        for run_id, task in list(self._active_workers.items()):
            if not task.done():
                continue

            try:
                returncode = task.result()
                if returncode != 0:
                    mark_run_failed_if_running(run_id, f"worker exited with code {returncode}")
            except asyncio.CancelledError:
                mark_run_failed_if_running(run_id, "worker task was cancelled")
            except Exception as exc:
                mark_run_failed_if_running(run_id, str(exc))
            finally:
                self._active_workers.pop(run_id, None)
                self._release_permit(run_id)
                logger.info(
                    "scheduler.worker.reaped",
                    extra={
                        "run_id": run_id,
                        "active_workers": len(self._permits_held),
                        "available_capacity": self.available_capacity,
                    },
                )

    async def _orchestrator_reap_loop(self) -> None:
        """Periodically clean up completed orchestrator campaign tasks."""
        while not self._stopped.is_set():
            await self._reap_orchestrator_tasks()
            await asyncio.sleep(5)

    async def _reap_orchestrator_tasks(self) -> None:
        """Remove completed orchestrator tasks from the tracking dict."""
        for campaign_id, task in list(self._orchestrator_tasks.items()):
            if not task.done():
                continue

            try:
                task.result()
                logger.info("Orchestrator campaign %s completed", campaign_id)
            except asyncio.CancelledError:
                logger.info("Orchestrator campaign %s was cancelled", campaign_id)
            except Exception as exc:
                logger.error("Orchestrator campaign %s failed: %s", campaign_id, exc)

            self._orchestrator_tasks.pop(campaign_id, None)
