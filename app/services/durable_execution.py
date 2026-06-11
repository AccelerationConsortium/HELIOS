"""Durable execution boundary for campaign orchestration.

HELIOS currently runs orchestrator campaigns in-process, with SQLite
checkpointing handled inside the orchestrator. This module makes that choice
explicit and provides an adapter seam for LangGraph, Temporal, or Pydantic AI
without coupling core agents to any one workflow engine.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from app.agents.orchestrator import OrchestratorAgent, OrchestratorInput, OrchestratorOutput

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DurableRunHandle:
    campaign_id: str
    backend: str
    status: str


@dataclass(frozen=True)
class DurableRunStatus:
    campaign_id: str
    backend: str
    status: str
    result: dict[str, Any] | None = None
    error: str | None = None


@dataclass(frozen=True)
class DurableRunEvent:
    campaign_id: str
    type: str
    payload: dict[str, Any]
    created_at: str


@dataclass
class _InProcessRun:
    campaign_id: str
    task: asyncio.Task[OrchestratorOutput]
    status: str = "running"
    result: OrchestratorOutput | None = None
    error: str | None = None
    events: list[DurableRunEvent] | None = None


class DurableExecutionBackend(Protocol):
    name: str

    async def start_campaign(
        self,
        input_data: OrchestratorInput,
        *,
        resume_from_round: int | None = None,
        restored_state: dict[str, Any] | None = None,
    ) -> DurableRunHandle: ...

    async def get_status(self, campaign_id: str) -> DurableRunStatus | None: ...

    async def get_result(self, campaign_id: str) -> OrchestratorOutput | None: ...

    async def cancel_campaign(self, campaign_id: str) -> DurableRunStatus | None: ...

    async def list_events(self, campaign_id: str) -> list[DurableRunEvent]: ...


class CampaignAlreadyRunningError(RuntimeError):
    """Raised when starting a campaign whose id is already running."""

    def __init__(self, campaign_id: str) -> None:
        super().__init__(f"Campaign '{campaign_id}' is already running")
        self.campaign_id = campaign_id


class InProcessDurableBackend:
    """Current durable backend: in-process task plus orchestrator checkpoints."""

    name = "in_process_sqlite"

    # Finished runs kept in memory for status queries before falling back to DB.
    MAX_FINISHED_RUNS = 200

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[OrchestratorOutput]] = {}
        self._runs: dict[str, _InProcessRun] = {}

    async def start_campaign(
        self,
        input_data: OrchestratorInput,
        *,
        resume_from_round: int | None = None,
        restored_state: dict[str, Any] | None = None,
    ) -> DurableRunHandle:
        agent = OrchestratorAgent()
        effective_campaign_id = input_data.campaign_id or f"orch-{uuid.uuid4().hex[:12]}"
        input_data.campaign_id = effective_campaign_id

        # Guard: never silently overwrite a live task for the same campaign id.
        # Overwriting would orphan the old task and let its done-callback
        # finalize the *new* run with the *old* result.
        existing = self._tasks.get(effective_campaign_id)
        if existing is not None and not existing.done():
            raise CampaignAlreadyRunningError(effective_campaign_id)

        self._ensure_campaign_state(effective_campaign_id, input_data)
        task = asyncio.create_task(
            agent.process(
                input_data,
                resume_from_round=resume_from_round,
                restored_state=restored_state,
            ),
            name=f"orchestrator-{effective_campaign_id}",
        )
        self._tasks[effective_campaign_id] = task
        run = _InProcessRun(
            campaign_id=effective_campaign_id,
            task=task,
            events=[],
        )
        self._runs[effective_campaign_id] = run
        self._record_event(
            effective_campaign_id,
            "durable_run_started",
            {
                "backend": self.name,
                "resume_from_round": resume_from_round,
                "restored_state_count": len(restored_state or {}),
            },
        )
        task.add_done_callback(
            lambda done: self._finalize_task(
                effective_campaign_id,
                done,
            )
        )
        return DurableRunHandle(
            campaign_id=effective_campaign_id,
            backend=self.name,
            status="running",
        )

    def get_task(self, campaign_id: str) -> asyncio.Task[OrchestratorOutput] | None:
        return self._tasks.get(campaign_id)

    async def get_status(self, campaign_id: str) -> DurableRunStatus | None:
        run = self._runs.get(campaign_id)
        if run is not None:
            self._refresh_run(run)
            return self._run_status(run)

        db_status = self._load_db_status(campaign_id)
        if db_status is not None:
            return db_status
        return None

    async def get_result(self, campaign_id: str) -> OrchestratorOutput | None:
        run = self._runs.get(campaign_id)
        if run is None:
            return None
        self._refresh_run(run)
        return run.result

    async def cancel_campaign(self, campaign_id: str) -> DurableRunStatus | None:
        run = self._runs.get(campaign_id)
        if run is None:
            return await self.get_status(campaign_id)
        if not run.task.done():
            run.task.cancel()
            run.status = "cancelling"
            run.error = "Campaign cancellation requested"
            self._record_event(
                campaign_id,
                "durable_run_cancel_requested",
                {"backend": self.name},
            )
            await asyncio.sleep(0)
        self._refresh_run(run)
        return self._run_status(run)

    async def list_events(self, campaign_id: str) -> list[DurableRunEvent]:
        run = self._runs.get(campaign_id)
        if run is None or run.events is None:
            return []
        return list(run.events)

    def active_campaigns(self) -> list[str]:
        return [
            campaign_id
            for campaign_id, task in self._tasks.items()
            if not task.done()
        ]

    def _refresh_run(self, run: _InProcessRun) -> None:
        if run.task.done() and run.status in {"running", "cancelling"}:
            self._finalize_task(run.campaign_id, run.task)

    def _finalize_task(
        self,
        campaign_id: str,
        done: asyncio.Task[OrchestratorOutput],
    ) -> None:
        run = self._runs.get(campaign_id)
        if run is None:
            return
        if run.task is not done:
            # Stale callback from a superseded task (e.g. a resumed campaign):
            # never let an old task finalize the current run.
            return
        if run.status not in {"running", "cancelling"}:
            return
        logger.info(
            "durable.campaign.finished",
            extra={
                "campaign_id": campaign_id,
                "backend": self.name,
                "cancelled": done.cancelled(),
            },
        )
        try:
            run.result = done.result()
            run.status = "completed"
            self._record_event(
                campaign_id,
                "durable_run_completed",
                {
                    "backend": self.name,
                    "status": run.result.status,
                    "rounds_completed": run.result.rounds_completed,
                    "best_kpi": run.result.best_kpi,
                },
            )
        except asyncio.CancelledError:
            run.status = "cancelled"
            run.error = "Campaign cancelled"
            self._record_event(campaign_id, "durable_run_cancelled", {"backend": self.name})
        except Exception as exc:
            run.status = "failed"
            run.error = str(exc)
            self._record_event(
                campaign_id,
                "durable_run_failed",
                {"backend": self.name, "error": str(exc)},
            )
        finally:
            # The task handle is no longer needed once finished; campaign
            # status/results remain available via self._runs and the DB.
            self._tasks.pop(campaign_id, None)
            self._evict_finished_runs()

    def _evict_finished_runs(self) -> None:
        """Cap in-memory finished runs; status queries fall back to the DB.

        Without this, every campaign ever started stays in memory for the
        lifetime of the server (task result, error string, event list).
        """
        finished = [
            cid
            for cid, run in self._runs.items()
            if run.status not in {"running", "cancelling"}
        ]
        overflow = len(finished) - self.MAX_FINISHED_RUNS
        for cid in finished[:max(0, overflow)]:
            self._runs.pop(cid, None)

    def _run_status(self, run: _InProcessRun) -> DurableRunStatus:
        return DurableRunStatus(
            campaign_id=run.campaign_id,
            backend=self.name,
            status=run.status,
            result=run.result.model_dump() if run.result is not None else None,
            error=run.error,
        )

    def _record_event(
        self,
        campaign_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        event = DurableRunEvent(
            campaign_id=campaign_id,
            type=event_type,
            payload=payload,
            created_at=datetime.now(UTC).isoformat(),
        )
        run = self._runs.get(campaign_id)
        if run is not None:
            if run.events is None:
                run.events = []
            run.events.append(event)
        try:
            from app.services.campaign_events import log_event

            log_event(
                campaign_id,
                event_type,
                {
                    "type": event_type,
                    **payload,
                },
            )
        except Exception:
            logger.debug("Durable event persistence failed", exc_info=True)

    def _ensure_campaign_state(
        self,
        campaign_id: str,
        input_data: OrchestratorInput,
    ) -> None:
        try:
            from app.services.campaign_state import create_campaign

            create_campaign(
                campaign_id,
                input_data.model_dump(mode="json"),
                direction=input_data.direction,
            )
        except Exception:
            logger.debug("Durable campaign state initialization failed", exc_info=True)

    def _load_db_status(self, campaign_id: str) -> DurableRunStatus | None:
        try:
            from app.services.campaign_state import load_campaign

            db_state = load_campaign(campaign_id)
        except Exception:
            logger.debug("Durable DB status fallback failed", exc_info=True)
            return None
        if db_state is None:
            return None
        status = db_state["status"]
        result = {
            "best_kpi": db_state.get("best_kpi"),
            "current_round": db_state.get("current_round"),
            "total_rounds": db_state.get("total_rounds"),
            "total_runs": db_state.get("total_runs"),
            "stop_reason": db_state.get("stop_reason"),
        }
        if status not in {"completed", "failed", "cancelled"}:
            status = "resumable"
        return DurableRunStatus(
            campaign_id=campaign_id,
            backend=self.name,
            status=status,
            result=result,
            error=db_state.get("error"),
        )


_backend: DurableExecutionBackend = InProcessDurableBackend()


def get_durable_backend() -> DurableExecutionBackend:
    return _backend


def set_durable_backend(backend: DurableExecutionBackend) -> None:
    """Install a workflow-engine-backed implementation in tests or startup."""
    global _backend
    _backend = backend
