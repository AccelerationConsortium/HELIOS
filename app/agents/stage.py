"""First-class agent stage execution primitives."""
from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from app.agents.base import AgentResult


@dataclass(frozen=True)
class AgentStageCall:
    """One agent invocation inside a stage."""

    agent_name: str
    input_data: BaseModel


@dataclass(frozen=True)
class AgentStageNode:
    """A named stage node in an orchestration graph."""

    name: str
    agent_names: tuple[str, ...]
    depends_on: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "agent_names": list(self.agent_names),
            "depends_on": list(self.depends_on),
        }


@dataclass(frozen=True)
class AgentStageGraph:
    """Declarative graph of orchestration stages."""

    name: str
    nodes: tuple[AgentStageNode, ...]

    @classmethod
    def linear(
        cls,
        name: str,
        stages: list[tuple[str, tuple[str, ...]]],
    ) -> AgentStageGraph:
        nodes: list[AgentStageNode] = []
        previous = ""
        for stage_name, agent_names in stages:
            depends_on = (previous,) if previous else ()
            nodes.append(
                AgentStageNode(
                    name=stage_name,
                    agent_names=agent_names,
                    depends_on=depends_on,
                )
            )
            previous = stage_name
        return cls(name=name, nodes=tuple(nodes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "nodes": [node.to_dict() for node in self.nodes],
        }


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

    async def run_single(
        self,
        stage_name: str,
        agent_name: str,
        input_data: BaseModel,
        *,
        caller: str = "orchestrator",
        trace_id: str | None = None,
        timeout_s: float = 300.0,
        emit: Callable[[dict[str, Any]], None] | None = None,
    ) -> AgentResult[Any]:
        result = await self.run_parallel(
            stage_name,
            [AgentStageCall(agent_name, input_data)],
            caller=caller,
            trace_id=trace_id,
            timeout_s=timeout_s,
            emit=emit,
        )
        if result.agent_results:
            return result.agent_results[0]
        return AgentResult(
            success=False,
            errors=[f"Stage '{stage_name}' produced no agent result"],
            agent_name=agent_name,
            trace_id=result.trace_id,
            duration_ms=result.duration_ms,
        )

    async def run_parallel(
        self,
        stage_name: str,
        calls: list[AgentStageCall],
        *,
        caller: str = "orchestrator",
        trace_id: str | None = None,
        timeout_s: float = 300.0,
        emit: Callable[[dict[str, Any]], None] | None = None,
    ) -> AgentStageResult:
        trace_id = trace_id or f"stage-{uuid.uuid4().hex[:12]}"
        if emit is not None:
            emit(
                {
                    "type": "agent_stage_start",
                    "stage": stage_name,
                    "agents": [call.agent_name for call in calls],
                    "trace_id": trace_id,
                }
            )
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

        stage_result = AgentStageResult(
            stage_name=stage_name,
            success=success,
            agent_results=results,
            outputs=outputs,
            errors=errors,
            duration_ms=(time.monotonic() - start) * 1000,
            trace_id=trace_id,
        )
        if emit is not None:
            emit(
                {
                    "type": "agent_stage_end",
                    "stage": stage_name,
                    "agents": [call.agent_name for call in calls],
                    "trace_id": trace_id,
                    "success": stage_result.success,
                    "duration_ms": stage_result.duration_ms,
                    "errors": stage_result.errors,
                }
            )
        return stage_result
