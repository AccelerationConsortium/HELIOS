"""Pause protocol — agent-initiated human-in-the-loop mechanism.

Instead of a global ``require_manual_confirmation`` flag, individual agents
call ``request_pause()`` at runtime when *they* decide human oversight is
needed.  The Orchestrator registers a ``PauseHandler`` that persists the
request, emits SSE, and polls for the operator's decision.

Lifecycle::

    Agent.process() → self.request_pause(PauseRequest(...))
        → PauseHandler (injected by Orchestrator / ControlPlane)
            → persist to DB
            → emit SSE "pause_requested"
            → poll DB for operator decision
        → PauseResult returned to agent
        → agent continues / aborts / modifies based on decision
"""
from __future__ import annotations

import asyncio
import enum
import logging
import uuid
from typing import Any, Callable, Awaitable

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Granularity enum
# ---------------------------------------------------------------------------


class Granularity(str, enum.Enum):
    """Execution granularity level chosen by an agent at runtime."""

    FINE = "fine"          # Step-by-step with confirmations
    COARSE = "coarse"      # Batch execute, report at end
    ADAPTIVE = "adaptive"  # Start coarse, switch to fine on anomalies


# ---------------------------------------------------------------------------
# Pause request / result models
# ---------------------------------------------------------------------------


class PauseRequest(BaseModel):
    """Agent-initiated request to pause execution and await human decision."""

    pause_id: str = Field(default_factory=lambda: f"pause-{uuid.uuid4().hex[:10]}")
    reason: str
    risk_factors: dict[str, float] = Field(default_factory=dict)
    suggested_action: str = "approve"   # approve | modify | abort
    expires_in_s: float = 600.0         # 10 minutes default
    checkpoint: dict[str, Any] = Field(
        default_factory=dict,
        description="Serialisable agent state for resume after approval",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class PauseResult(BaseModel):
    """Operator response to a pause request."""

    decision: str = "approved"  # approved | rejected | modified | timeout
    modifications: dict[str, Any] = Field(default_factory=dict)
    decided_by: str = ""
    decided_at: str = ""


# ---------------------------------------------------------------------------
# Risk assessment helpers
# ---------------------------------------------------------------------------


class RiskAssessment(BaseModel):
    """Structured risk evaluation produced by an agent."""

    should_pause: bool = False
    risk_score: float = 0.0          # 0.0 = safe, 1.0 = dangerous
    reason: str = ""
    factors: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Type alias for the pause handler callable
# ---------------------------------------------------------------------------

PauseHandler = Callable[[str, PauseRequest], Awaitable[PauseResult]]


# ---------------------------------------------------------------------------
# Auto-approve handler (used in tests / simulation mode)
# ---------------------------------------------------------------------------


async def auto_approve_handler(_agent_name: str, _request: PauseRequest) -> PauseResult:
    """Default handler that approves everything (simulation / test mode)."""
    return PauseResult(decision="approved", decided_by="auto")
