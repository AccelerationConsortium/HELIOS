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
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._tasks: list[asyncio.Task[Any]] = []
        self._active_workers: dict[str, asyncio.Task[Any]] = {}
        self._stopped = asyncio.Event()

    async def start(self) -> None:
        self._stopped.clear()
        self._tasks = [
            asyncio.create_task(self._campaign_loop(), name="campaign-loop"),
            asyncio.create_task(self._dispatch_loop(), name="dispatch-loop"),
            asyncio.create_task(self._reap_loop(), name="reap-loop"),
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
        self._active_workers.clear()

    async def _campaign_loop(self) -> None:
        while not self._stopped.is_set():
            await asyncio.to_thread(trigger_due_campaigns)
            await asyncio.sleep(self._settings.campaign_poll_seconds)

    async def _dispatch_loop(self) -> None:
        while not self._stopped.is_set():
            run_ids = await asyncio.to_thread(claim_schedulable_runs)
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

        # Import here to avoid circular imports
        from app.worker import execute_run

        task = asyncio.create_task(
            asyncio.to_thread(execute_run, run_id),
            name=f"worker-{run_id}",
        )
        self._active_workers[run_id] = task

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

            self._active_workers.pop(run_id, None)
