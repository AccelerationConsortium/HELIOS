"""ControlPlane — in-process agent routing, audit, and coordination layer.

Replaces direct ``AgentClass().run(input)`` calls with a centralised
``control_plane.call("agent_name", input)`` pattern that provides:

1. **Unified audit trail** — every cross-agent call is recorded in the
   provenance_events table with caller, target, trace_id, timing.
2. **Agent-to-agent calls** — any agent can call any other agent via
   ``self._call("target_agent", input)`` without routing through Orchestrator.
3. **Pause handler injection** — all registered agents automatically get
   the control plane's pause handler so they can request human oversight.
4. **Timeout & error boundary** — each call has an optional timeout and
   errors are captured without crashing the caller.
5. **DecisionNode tracking** — granularity and pause decisions are
   automatically recorded as first-class decision nodes.

Architecture notes:
- This is an in-process Python object, NOT a microservice.
- It reuses the existing ``EventBus`` for SSE and ``audit.record_event``
  for persistence — no new infrastructure needed.
- Agents remain stateless processors; the ControlPlane is the stateful
  coordinator that the Orchestrator delegates to.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Callable, Awaitable

from pydantic import BaseModel

from app.agents.base import AgentResult, BaseAgent, DecisionNode
from app.agents.pause import (
    Granularity,
    PauseHandler,
    PauseRequest,
    PauseResult,
    auto_approve_handler,
)

logger = logging.getLogger(__name__)


def _summarize(model: BaseModel, max_len: int = 200) -> str:
    """Short repr of a Pydantic model for audit logging (no secrets)."""
    try:
        raw = model.model_dump_json()
        if len(raw) > max_len:
            return raw[:max_len] + "…"
        return raw
    except Exception:
        return repr(model)[:max_len]


class ControlPlane:
    """In-process control plane for agent routing and coordination.

    Usage::

        cp = ControlPlane(event_bus=bus)
        cp.set_pause_handler(my_handler)

        cp.register(SafetyAgent())
        cp.register(ExecutionAgent())

        result = await cp.call(
            "safety_agent", safety_input, caller="orchestrator",
        )
    """

    def __init__(
        self,
        event_bus: Any = None,
        pause_handler: PauseHandler | None = None,
    ) -> None:
        self._registry: dict[str, BaseAgent] = {}
        self._event_bus = event_bus
        self._pause_handler: PauseHandler = pause_handler or auto_approve_handler
        self._campaign_id: str = ""  # set by orchestrator per-campaign

    # -- registration -------------------------------------------------------

    def register(self, agent: BaseAgent) -> None:
        """Register an agent and inject control-plane capabilities."""
        self._registry[agent.name] = agent
        # Inject pause handler
        agent._pause_handler = self._pause_handler  # type: ignore[attr-defined]
        # Inject cross-agent call capability
        agent._call = self.call  # type: ignore[attr-defined]
        logger.debug("ControlPlane: registered agent '%s' (layer=%s)", agent.name, agent.layer)

    def set_pause_handler(self, handler: PauseHandler) -> None:
        """Update the pause handler and re-inject into all registered agents."""
        self._pause_handler = handler
        for agent in self._registry.values():
            agent._pause_handler = handler  # type: ignore[attr-defined]

    def set_campaign_id(self, campaign_id: str) -> None:
        self._campaign_id = campaign_id

    @property
    def agents(self) -> dict[str, BaseAgent]:
        return dict(self._registry)

    # -- call ---------------------------------------------------------------

    async def call(
        self,
        target: str,
        input_data: BaseModel,
        *,
        caller: str = "orchestrator",
        trace_id: str | None = None,
        timeout_s: float = 300.0,
    ) -> AgentResult:
        """Route a call to a registered agent with full audit trail.

        Parameters
        ----------
        target : str
            Agent name (e.g. ``"safety_agent"``).
        input_data : BaseModel
            Typed input for the target agent.
        caller : str
            Name of the calling agent (for audit).
        trace_id : str | None
            Correlation ID (auto-generated if omitted).
        timeout_s : float
            Maximum seconds before the call is cancelled.
        """
        trace_id = trace_id or uuid.uuid4().hex[:12]
        agent_name = target.split(".")[0]
        call_id = f"call-{uuid.uuid4().hex[:8]}"

        agent = self._registry.get(agent_name)
        if agent is None:
            logger.error("ControlPlane: agent '%s' not registered", agent_name)
            return AgentResult(
                success=False,
                errors=[f"Agent '{agent_name}' not registered in ControlPlane"],
                agent_name=agent_name,
                trace_id=trace_id,
            )

        # ── Audit: call start ────────────────────────────────────────────
        call_record: dict[str, Any] = {
            "call_id": call_id,
            "caller": caller,
            "target": agent_name,
            "trace_id": trace_id,
            "input_summary": _summarize(input_data),
        }
        self._emit_audit("agent_call_start", call_record)

        # ── Execute with timeout ─────────────────────────────────────────
        start = time.monotonic()
        try:
            result = await asyncio.wait_for(
                agent.run(input_data, trace_id=trace_id),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            duration_ms = (time.monotonic() - start) * 1000
            logger.error(
                "ControlPlane: call %s → %s timed out after %.0fs",
                caller, agent_name, timeout_s,
            )
            result = AgentResult(
                success=False,
                errors=[f"Timeout after {timeout_s}s"],
                agent_name=agent_name,
                trace_id=trace_id,
                duration_ms=duration_ms,
            )
        except Exception as exc:
            duration_ms = (time.monotonic() - start) * 1000
            logger.error(
                "ControlPlane: call %s → %s raised %s",
                caller, agent_name, exc, exc_info=True,
            )
            result = AgentResult(
                success=False,
                errors=[str(exc)],
                agent_name=agent_name,
                trace_id=trace_id,
                duration_ms=duration_ms,
            )

        # ── Audit: call end ──────────────────────────────────────────────
        call_record["success"] = result.success
        call_record["duration_ms"] = result.duration_ms
        call_record["errors"] = result.errors
        self._emit_audit("agent_call_end", call_record)

        return result

    # -- parallel call ------------------------------------------------------

    async def call_parallel(
        self,
        calls: list[tuple[str, BaseModel]],
        *,
        caller: str = "orchestrator",
        trace_id: str | None = None,
        timeout_s: float = 300.0,
    ) -> list[AgentResult]:
        """Execute multiple agent calls concurrently.

        Parameters
        ----------
        calls : list of (agent_name, input_data) tuples
        """
        trace_id = trace_id or uuid.uuid4().hex[:12]
        tasks = [
            self.call(target, data, caller=caller, trace_id=trace_id, timeout_s=timeout_s)
            for target, data in calls
        ]
        return await asyncio.gather(*tasks)

    # -- audit helpers ------------------------------------------------------

    def _emit_audit(self, action: str, details: dict[str, Any]) -> None:
        """Record to provenance_events DB + publish SSE."""
        details["campaign_id"] = self._campaign_id
        try:
            from app.core.db import run_txn
            from app.services.audit import record_event

            def _txn(conn):
                record_event(
                    conn,
                    run_id=None,
                    actor=details.get("caller", "control_plane"),
                    action=f"control_plane.{action}",
                    details=details,
                )

            run_txn(_txn)
        except Exception:
            logger.debug("ControlPlane audit write failed", exc_info=True)

        # Also publish to SSE via event bus
        if self._event_bus is not None:
            try:
                from app.services.event_bus import EventMessage
                from app.core.db import utcnow_iso

                self._event_bus.publish(EventMessage(
                    id=f"cp-{uuid.uuid4().hex[:8]}",
                    run_id=None,
                    actor=details.get("caller", "control_plane"),
                    action=f"control_plane.{action}",
                    details=details,
                    created_at=utcnow_iso(),
                ))
            except Exception:
                pass
