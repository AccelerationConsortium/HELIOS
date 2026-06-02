"""Durable execution boundary for campaign orchestration.

HELIOS currently runs orchestrator campaigns in-process, with SQLite
checkpointing handled inside the orchestrator. This module makes that choice
explicit and provides an adapter seam for LangGraph, Temporal, or Pydantic AI
without coupling core agents to any one workflow engine.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from app.agents.orchestrator import OrchestratorAgent, OrchestratorInput, OrchestratorOutput

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DurableRunHandle:
    campaign_id: str
    backend: str
    status: str


class DurableExecutionBackend(Protocol):
    name: str

    async def start_campaign(
        self,
        input_data: OrchestratorInput,
        *,
        resume_from_round: int | None = None,
        restored_state: dict[str, Any] | None = None,
    ) -> DurableRunHandle: ...


class InProcessDurableBackend:
    """Current durable backend: in-process task plus orchestrator checkpoints."""

    name = "in_process_sqlite"

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[OrchestratorOutput]] = {}

    async def start_campaign(
        self,
        input_data: OrchestratorInput,
        *,
        resume_from_round: int | None = None,
        restored_state: dict[str, Any] | None = None,
    ) -> DurableRunHandle:
        agent = OrchestratorAgent()
        campaign_id = input_data.campaign_id
        task = asyncio.create_task(
            agent.process(
                input_data,
                resume_from_round=resume_from_round,
                restored_state=restored_state,
            ),
            name=f"orchestrator-{campaign_id or 'new'}",
        )
        effective_campaign_id = campaign_id or input_data.campaign_id or "pending"
        self._tasks[effective_campaign_id] = task
        task.add_done_callback(
            lambda done: logger.info(
                "durable.campaign.finished",
                extra={
                    "campaign_id": effective_campaign_id,
                    "backend": self.name,
                    "cancelled": done.cancelled(),
                },
            )
        )
        return DurableRunHandle(
            campaign_id=effective_campaign_id,
            backend=self.name,
            status="running",
        )

    def get_task(self, campaign_id: str) -> asyncio.Task[OrchestratorOutput] | None:
        return self._tasks.get(campaign_id)

    def active_campaigns(self) -> list[str]:
        return [
            campaign_id
            for campaign_id, task in self._tasks.items()
            if not task.done()
        ]


_backend: DurableExecutionBackend = InProcessDurableBackend()


def get_durable_backend() -> DurableExecutionBackend:
    return _backend


def set_durable_backend(backend: DurableExecutionBackend) -> None:
    """Install a workflow-engine-backed implementation in tests or startup."""
    global _backend
    _backend = backend
