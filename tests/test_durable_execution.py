"""Regression tests for the in-process durable execution backend.

Covers:
- duplicate-start protection (no silent task overwrite)
- stale done-callback protection (a superseded task must not finalize
  the current run)
- task-handle cleanup and bounded retention of finished runs
- cancellation status transitions
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.agents.orchestrator import OrchestratorInput, OrchestratorOutput
from app.services import durable_execution
from app.services.durable_execution import (
    CampaignAlreadyRunningError,
    InProcessDurableBackend,
)


def _make_input(campaign_id: str | None = None) -> OrchestratorInput:
    return OrchestratorInput(
        contract_id="c-test",
        objective_kpi="kpi",
        direction="minimize",
        max_rounds=1,
        batch_size=1,
        dimensions=[],
        protocol_template={"steps": []},
        campaign_id=campaign_id,
    )


class _StubAgent:
    """Replaces OrchestratorAgent: completes when its gate event is set."""

    gates: dict[str, asyncio.Event] = {}
    results: dict[str, Any] = {}

    async def process(self, input_data: OrchestratorInput, **_kwargs) -> OrchestratorOutput:
        cid = input_data.campaign_id or "?"
        gate = self.gates.get(cid)
        if gate is not None:
            await gate.wait()
        result = self.results.get(cid)
        if isinstance(result, Exception):
            raise result
        return result or OrchestratorOutput(campaign_id=cid, status="completed")


@pytest.fixture()
def backend(monkeypatch) -> InProcessDurableBackend:
    _StubAgent.gates = {}
    _StubAgent.results = {}
    monkeypatch.setattr(durable_execution, "OrchestratorAgent", _StubAgent)
    return InProcessDurableBackend()


async def test_duplicate_start_rejected(backend: InProcessDurableBackend):
    """Starting a campaign id that is already running must raise, not
    silently orphan the original task."""
    gate = asyncio.Event()
    _StubAgent.gates["c1"] = gate

    await backend.start_campaign(_make_input("c1"))
    with pytest.raises(CampaignAlreadyRunningError):
        await backend.start_campaign(_make_input("c1"))

    gate.set()
    await asyncio.sleep(0.05)
    status = await backend.get_status("c1")
    assert status is not None and status.status == "completed"


async def test_restart_after_completion_allowed(backend: InProcessDurableBackend):
    """Once finished, the same campaign id may be started again (resume)."""
    await backend.start_campaign(_make_input("c2"))
    await asyncio.sleep(0.05)

    gate = asyncio.Event()
    _StubAgent.gates["c2"] = gate
    handle = await backend.start_campaign(_make_input("c2"))
    assert handle.status == "running"
    status = await backend.get_status("c2")
    assert status is not None and status.status == "running"
    gate.set()
    await asyncio.sleep(0.05)


async def test_stale_callback_does_not_finalize_new_run(backend: InProcessDurableBackend):
    """A done-callback from a superseded task must not write its result
    into the run object of a newer task with the same campaign id."""
    # First run completes with rounds_completed=1
    _StubAgent.results["c3"] = OrchestratorOutput(
        campaign_id="c3", status="completed", rounds_completed=1
    )
    await backend.start_campaign(_make_input("c3"))
    await asyncio.sleep(0.05)
    first_task = None  # finished; handle already evicted from _tasks

    # Second run (resume) is still in flight
    gate = asyncio.Event()
    _StubAgent.gates["c3"] = gate
    _StubAgent.results["c3"] = OrchestratorOutput(
        campaign_id="c3", status="completed", rounds_completed=2
    )
    await backend.start_campaign(_make_input("c3"))

    # Simulate the first (stale) task's callback firing late
    stale = asyncio.get_event_loop().create_future()
    stale_task = asyncio.create_task(asyncio.sleep(0))
    await stale_task
    backend._finalize_task("c3", stale_task)  # must be a no-op

    status = await backend.get_status("c3")
    assert status is not None and status.status == "running"
    del first_task, stale

    gate.set()
    await asyncio.sleep(0.05)
    status = await backend.get_status("c3")
    assert status is not None
    assert status.status == "completed"
    assert status.result is not None and status.result["rounds_completed"] == 2


async def test_finished_task_handles_released(backend: InProcessDurableBackend):
    """Task handles must be dropped once a campaign finishes."""
    await backend.start_campaign(_make_input("c4"))
    await asyncio.sleep(0.05)
    assert backend.get_task("c4") is None
    assert backend.active_campaigns() == []
    # Status remains queryable from the retained run record
    status = await backend.get_status("c4")
    assert status is not None and status.status == "completed"


async def test_finished_runs_bounded(backend: InProcessDurableBackend, monkeypatch):
    """In-memory finished-run history must not grow without bound."""
    monkeypatch.setattr(InProcessDurableBackend, "MAX_FINISHED_RUNS", 5)
    for i in range(12):
        await backend.start_campaign(_make_input(f"bulk-{i}"))
    await asyncio.sleep(0.1)
    finished = [
        r for r in backend._runs.values() if r.status not in {"running", "cancelling"}
    ]
    assert len(finished) <= 5
    assert len(backend._tasks) == 0


async def test_cancel_running_campaign(backend: InProcessDurableBackend):
    gate = asyncio.Event()
    _StubAgent.gates["c5"] = gate
    await backend.start_campaign(_make_input("c5"))

    status = await backend.cancel_campaign("c5")
    assert status is not None and status.status in {"cancelling", "cancelled"}

    await asyncio.sleep(0.05)
    status = await backend.get_status("c5")
    assert status is not None and status.status == "cancelled"


async def test_failed_campaign_reports_error(backend: InProcessDurableBackend):
    _StubAgent.results["c6"] = RuntimeError("boom")
    await backend.start_campaign(_make_input("c6"))
    await asyncio.sleep(0.05)
    status = await backend.get_status("c6")
    assert status is not None
    assert status.status == "failed"
    assert status.error == "boom"
