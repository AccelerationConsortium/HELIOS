"""Base Agent protocol for the OTbot multi-agent orchestrator.

All agents implement the same interface. Agents are Python classes,
not microservices. Communication is via typed Pydantic models (contracts).
"""
from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


@dataclass(frozen=True)
class DecisionNode:
    """A single node in an agent's decision tree.

    Captures one decision point: what options were considered, which was
    selected, and why.  Nodes can be nested via ``children`` to represent
    dependent sub-decisions (e.g. a violation breakdown under a verdict node).
    """

    id: str                                    # unique within tree, e.g. "strategy_mode"
    label: str                                 # human label, e.g. "Choose exploration strategy"
    options: list[str]                         # all options considered
    selected: str                              # the option that was taken
    reason: str                                # why this option was chosen
    outcome: str = ""                          # observable consequence of this choice
    children: tuple[DecisionNode, ...] = ()   # dependent sub-decisions

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "options": self.options,
            "selected": self.selected,
            "reason": self.reason,
            "outcome": self.outcome,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class AgentResult(Generic[OutputT]):
    """Wrapper for agent execution results."""
    success: bool
    output: OutputT | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    agent_name: str = ""
    trace_id: str = ""


class BaseAgent(ABC, Generic[InputT, OutputT]):
    """Base class for all OTbot agents.

    Agents are stateless processors: they take typed input,
    validate it, process it, and return typed output.
    """

    name: str = "base_agent"
    description: str = ""
    layer: str = ""  # L0, L1, L2, L3, or "cross-cutting"

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"agent.{self.name}")

    @abstractmethod
    def validate_input(self, input_data: InputT) -> list[str]:
        """Validate input before processing. Returns list of error strings."""
        ...

    @abstractmethod
    async def process(self, input_data: InputT) -> OutputT:
        """Core processing logic. Override in subclasses."""
        ...

    async def run(self, input_data: InputT, trace_id: str | None = None) -> AgentResult[OutputT]:
        """Execute the agent with validation, timing, and error handling.

        This is the main entry point. Don't override this -- override process().
        """
        if trace_id is None:
            trace_id = uuid.uuid4().hex[:12]

        start = time.monotonic()

        # Validate
        errors = self.validate_input(input_data)
        if errors:
            return AgentResult(
                success=False,
                errors=errors,
                agent_name=self.name,
                trace_id=trace_id,
                duration_ms=(time.monotonic() - start) * 1000,
            )

        # Process
        try:
            output = await self.process(input_data)
            duration = (time.monotonic() - start) * 1000
            self.logger.info(
                "%s completed in %.1fms (trace=%s)",
                self.name, duration, trace_id,
            )
            return AgentResult(
                success=True,
                output=output,
                agent_name=self.name,
                trace_id=trace_id,
                duration_ms=duration,
            )
        except Exception as exc:
            duration = (time.monotonic() - start) * 1000
            self.logger.error(
                "%s failed in %.1fms: %s (trace=%s)",
                self.name, duration, exc, trace_id,
                exc_info=True,
            )
            return AgentResult(
                success=False,
                errors=[str(exc)],
                agent_name=self.name,
                trace_id=trace_id,
                duration_ms=duration,
            )
