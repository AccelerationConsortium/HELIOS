"""Base Agent protocol for the OTbot multi-agent orchestrator.

All agents implement the same interface. Agents are Python classes,
not microservices. Communication is via typed Pydantic models (contracts).

v2 additions (AgentField-inspired):
- ``request_pause()`` — agent-initiated human-in-the-loop gate
- ``assess_granularity()`` — agent decides own execution granularity
- ``_call()`` — cross-agent routing via ControlPlane
"""
from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Generic, TypeVar

from pydantic import BaseModel

from app.agents.pause import (
    Granularity,
    PauseHandler,
    PauseRequest,
    PauseResult,
    RiskAssessment,
    auto_approve_handler,
)

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
    # v2: track granularity and pause decisions made during this run
    granularity_used: Granularity | None = None
    pause_decisions: list[dict[str, Any]] = field(default_factory=list)


class AgentPauseRejected(Exception):
    """Raised when an operator rejects a pause request."""
    pass


class BaseAgent(ABC, Generic[InputT, OutputT]):
    """Base class for all OTbot agents.

    Agents are stateless processors: they take typed input,
    validate it, process it, and return typed output.

    v2 capabilities (injected by ControlPlane):
    - ``_pause_handler``: async callable for human-in-the-loop pausing
    - ``_call``: async callable for cross-agent routing
    """

    name: str = "base_agent"
    description: str = ""
    layer: str = ""  # L0, L1, L2, L3, or "cross-cutting"

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"agent.{self.name}")
        # Injected by ControlPlane (defaults for standalone / test use)
        self._pause_handler: PauseHandler = auto_approve_handler
        self._call: Callable[..., Awaitable[AgentResult]] | None = None
        # Accumulated during a single run() invocation
        self._run_pause_decisions: list[dict[str, Any]] = []

    @abstractmethod
    def validate_input(self, input_data: InputT) -> list[str]:
        """Validate input before processing. Returns list of error strings."""
        ...

    @abstractmethod
    async def process(self, input_data: InputT) -> OutputT:
        """Core processing logic. Override in subclasses."""
        ...

    # ------------------------------------------------------------------
    # Granularity assessment (override in subclasses for domain logic)
    # ------------------------------------------------------------------

    async def assess_granularity(
        self,
        input_data: InputT,
        context: dict[str, Any] | None = None,
    ) -> Granularity:
        """Decide execution granularity for this invocation.

        Default implementation returns COARSE.  Subclasses should override
        to incorporate domain-specific risk signals.

        Parameters
        ----------
        input_data : InputT
            The validated input for this run.
        context : dict | None
            Optional runtime context (campaign history, instrument state, etc.).
        """
        return Granularity.COARSE

    # ------------------------------------------------------------------
    # Pause protocol
    # ------------------------------------------------------------------

    async def request_pause(self, request: PauseRequest) -> PauseResult:
        """Request human-in-the-loop approval.

        Called by agent subclasses inside ``process()`` when they determine
        human oversight is needed.  The actual waiting is handled by the
        ``_pause_handler`` injected by ControlPlane / Orchestrator.

        Returns
        -------
        PauseResult
            The operator's decision (approved / rejected / modified / timeout).

        Raises
        ------
        AgentPauseRejected
            NOT raised automatically — the caller should check
            ``result.decision`` and raise if desired.
        """
        self.logger.info(
            "%s requesting pause: %s (risk_score=%.2f)",
            self.name,
            request.reason,
            max(request.risk_factors.values()) if request.risk_factors else 0.0,
        )
        result = await self._pause_handler(self.name, request)

        # Track for AgentResult
        self._run_pause_decisions.append({
            "pause_id": request.pause_id,
            "reason": request.reason,
            "risk_factors": request.risk_factors,
            "decision": result.decision,
            "decided_by": result.decided_by,
        })

        self.logger.info(
            "%s pause resolved: %s (by %s)",
            self.name, result.decision, result.decided_by or "auto",
        )
        return result

    # ------------------------------------------------------------------
    # Cross-agent call (convenience wrapper)
    # ------------------------------------------------------------------

    async def call_agent(
        self,
        target: str,
        input_data: BaseModel,
        **kwargs: Any,
    ) -> AgentResult:
        """Call another agent via the ControlPlane.

        Only available when registered with a ControlPlane.
        Falls back to error if no ControlPlane is connected.
        """
        if self._call is None:
            return AgentResult(
                success=False,
                errors=[
                    f"Agent '{self.name}' not registered with ControlPlane — "
                    f"cannot call '{target}'"
                ],
            )
        kwargs.setdefault("caller", self.name)
        return await self._call(target, input_data, **kwargs)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self, input_data: InputT, trace_id: str | None = None) -> AgentResult[OutputT]:
        """Execute the agent with validation, timing, and error handling.

        This is the main entry point. Don't override this -- override process().
        """
        if trace_id is None:
            trace_id = uuid.uuid4().hex[:12]

        # Reset per-run tracking
        self._run_pause_decisions = []

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
                pause_decisions=list(self._run_pause_decisions),
            )
        except AgentPauseRejected as exc:
            duration = (time.monotonic() - start) * 1000
            self.logger.warning(
                "%s paused and rejected in %.1fms: %s (trace=%s)",
                self.name, duration, exc, trace_id,
            )
            return AgentResult(
                success=False,
                errors=[f"Pause rejected: {exc}"],
                agent_name=self.name,
                trace_id=trace_id,
                duration_ms=duration,
                pause_decisions=list(self._run_pause_decisions),
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
                pause_decisions=list(self._run_pause_decisions),
            )
