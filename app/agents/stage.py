"""First-class agent stage execution primitives."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from app.agents.base import AgentResult


@dataclass(frozen=True)
class AgentStageCall:
    """One agent invocation inside a stage."""

    agent_name: str
    input_data: BaseModel


@dataclass
class AgentStageResult:
    """Aggregated result for a ControlPlane-backed stage."""

    stage_name: str
    success: bool
    agent_results: list[AgentResult[Any]] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    trace_id: str = ""


class AgentStageRunner:
    """Run a group of agent calls as a named orchestration stage.

    This is the small bridge between today's direct stage blocks and a fuller
    supervisor/swarms workflow. The runner keeps the stage boundary explicit
    while still reusing ControlPlane leases, context injection, tracing, and
    reputation updates for each constituent call.
    """

    def __init__(self, control_plane: Any) -> None:
        self._control_plane = control_plane

    async def run_parallel(
        self,
        stage_name: str,
        calls: list[AgentStageCall],
        *,
        caller: str = "orchestrator",
        trace_id: str | None = None,
        timeout_s: float = 300.0,
    ) -> AgentStageResult:
        trace_id = trace_id or f"stage-{uuid.uuid4().hex[:12]}"
        start = time.monotonic()
        results = await self._control_plane.call_parallel(
            [(call.agent_name, call.input_data) for call in calls],
            caller=f"{caller}.{stage_name}",
            trace_id=trace_id,
            timeout_s=timeout_s,
        )
        outputs: dict[str, Any] = {}
        errors: list[str] = []
        success = True
        for result in results:
            if not result.success:
                success = False
                errors.extend(result.errors)
            if result.output is not None:
                outputs[result.agent_name] = result.output.model_dump(mode="json")

        return AgentStageResult(
            stage_name=stage_name,
            success=success,
            agent_results=results,
            outputs=outputs,
            errors=errors,
            duration_ms=(time.monotonic() - start) * 1000,
            trace_id=trace_id,
        )
