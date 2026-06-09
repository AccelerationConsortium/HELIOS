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
6. **Agent lease pool (v3)** — a single shared instance of each agent is
   registered once. Concurrent callers (orchestrator + swarms) acquire a
   *lease* via :meth:`get_agent_lease` before invoking the agent, which
   serialises access through a per-agent reentrant async lock. This
   eliminates the race condition where the orchestrator and a swarm both
   spawn / mutate the same logical agent concurrently and interleave writes
   to shared databases or caches.

Architecture notes:
- This is an in-process Python object, NOT a microservice.
- It reuses the existing ``EventBus`` for SSE and ``audit.record_event``
  for persistence — no new infrastructure needed.
- Agents remain stateless processors; the ControlPlane is the stateful
  coordinator that the Orchestrator delegates to.
- The lock is *reentrant per asyncio task*: an agent that, while holding a
  lease, calls back into itself (directly or transitively) will not
  deadlock, because the owning task is recognised and the depth is counted.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from app.agents.base import AgentResult, BaseAgent
from app.agents.pause import (
    PauseHandler,
    auto_approve_handler,
)
from app.services.agent_context import (
    build_agent_context,
    harvest_agent_result,
    persist_agent_context_snapshot,
    reset_current_agent_context,
    set_current_agent_context,
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


class AgentLeaseTimeout(Exception):
    """Raised when a lease cannot be acquired within the requested timeout."""


class _ReentrantAgentLock:
    """An asyncio lock that is reentrant for the *same* owning task.

    ``asyncio.Lock`` is not reentrant: a task that already holds the lock and
    tries to acquire it again will deadlock. Agents in HELIOS can legitimately
    call back into themselves (e.g. RecoveryAgent invoking its own sub-step
    via the control plane), so we need reentrancy keyed on the running task.

    Concurrency model:
    - At most one *task lineage* holds the lock at a time.
    - The owning task may re-acquire any number of times; an equal number of
      releases is required before the lock is freed for other tasks.
    - Reentrancy is keyed on the currently-running ``asyncio.Task`` identity.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._owner: asyncio.Task[Any] | None = None
        self._depth: int = 0

    @property
    def locked(self) -> bool:
        return self._lock.locked()

    @property
    def depth(self) -> int:
        return self._depth

    async def acquire(self, timeout: float | None = None) -> bool:
        """Acquire the lock, blocking up to ``timeout`` seconds.

        Returns True on success. Raises :class:`AgentLeaseTimeout` if the
        timeout elapses before the lock is obtained.
        """
        current = asyncio.current_task()
        if current is not None and self._owner is current:
            # Reentrant acquisition by the owning task — no waiting.
            self._depth += 1
            return True

        if timeout is None:
            await self._lock.acquire()
        else:
            try:
                await asyncio.wait_for(self._lock.acquire(), timeout=timeout)
            except TimeoutError as exc:
                raise AgentLeaseTimeout(
                    f"Timed out after {timeout}s waiting for agent lock"
                ) from exc

        self._owner = current
        self._depth = 1
        return True

    def release(self) -> None:
        current = asyncio.current_task()
        if self._owner is not current:
            raise RuntimeError(
                "Agent lock released by a task that does not own it "
                f"(owner={self._owner!r}, releaser={current!r})"
            )
        self._depth -= 1
        if self._depth == 0:
            self._owner = None
            self._lock.release()


class AgentLease:
    """A scoped, exclusive borrow of a shared agent instance.

    Acquired via :meth:`ControlPlane.get_agent_lease`. While held, the lease
    guarantees the caller has serialised access to the underlying agent
    relative to other lease holders. Use as an async context manager::

        async with cp.get_agent_lease("design_agent", timeout=30) as lease:
            result = await lease.run(design_input, trace_id=tid)

    The lease auto-releases on context exit (success or exception).
    """

    def __init__(
        self,
        agent: BaseAgent,
        lock: _ReentrantAgentLock,
        lease_id: str,
        holder: str,
    ) -> None:
        self._agent = agent
        self._lock = lock
        self.lease_id = lease_id
        self.holder = holder
        self._released = False

    @property
    def agent(self) -> BaseAgent:
        return self._agent

    @property
    def name(self) -> str:
        return self._agent.name

    async def run(self, input_data: BaseModel, **kwargs: Any) -> AgentResult:
        """Convenience: invoke the leased agent's ``run`` directly."""
        return await self._agent.run(input_data, **kwargs)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._lock.release()


@dataclass
class AgentReputation:
    success_count: int = 0
    failure_count: int = 0
    total_latency_ms: float = 0.0
    consecutive_failures: int = 0
    last_failure_reason: str = ""

    @property
    def score(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5
        sr = self.success_count / total
        avg_lat = self.total_latency_ms / max(total, 1)
        latency_score = max(0.0, 1.0 - avg_lat / 500.0)
        return round(0.7 * sr + 0.3 * latency_score, 4)

    @property
    def is_degraded(self) -> bool:
        return self.consecutive_failures >= 3


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

        # Swarms / concurrent callers borrow the shared instance:
        async with cp.get_agent_lease("design_agent", timeout=30) as lease:
            result = await lease.run(design_input, trace_id=tid)
    """

    def __init__(
        self,
        event_bus: Any = None,
        pause_handler: PauseHandler | None = None,
        default_lease_timeout_s: float = 60.0,
    ) -> None:
        self._registry: dict[str, BaseAgent] = {}
        self._locks: dict[str, _ReentrantAgentLock] = {}
        self._reputations: dict[str, AgentReputation] = {}
        self._event_bus = event_bus
        self._pause_handler: PauseHandler = pause_handler or auto_approve_handler
        self._campaign_id: str = ""  # set by orchestrator per-campaign
        self._default_lease_timeout_s = default_lease_timeout_s
        # Guards mutation of the registry / lock maps themselves.
        self._registry_lock = asyncio.Lock()

    # -- registration -------------------------------------------------------

    def register(self, agent: BaseAgent) -> None:
        """Register an agent and inject control-plane capabilities.

        Idempotent: registering an agent whose name already exists replaces
        the instance but preserves the existing per-agent lock so any
        in-flight leases remain valid. Register each logical agent **once**
        at orchestrator init and share it; do not create per-round instances.
        """
        self._registry[agent.name] = agent
        self._locks.setdefault(agent.name, _ReentrantAgentLock())
        # Inject pause handler
        agent._pause_handler = self._pause_handler  # type: ignore[attr-defined]
        # Inject cross-agent call capability
        agent._call = self.call  # type: ignore[attr-defined]
        logger.debug(
            "ControlPlane: registered agent '%s' (layer=%s)", agent.name, agent.layer
        )

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

    @property
    def reputations(self) -> dict[str, AgentReputation]:
        """Return a snapshot copy of per-agent reputation records."""
        return dict(self._reputations)

    def is_registered(self, name: str) -> bool:
        return name.split(".")[0] in self._registry

    # -- reputation ---------------------------------------------------------

    def _update_reputation(self, name: str, result: AgentResult) -> None:
        """Update the reputation record for ``name`` from a call result."""
        agent_name = name.split(".")[0]
        rep = self._reputations.setdefault(agent_name, AgentReputation())

        latency_ms = result.duration_ms or 0.0
        rep.total_latency_ms += latency_ms

        if result.success:
            rep.success_count += 1
            rep.consecutive_failures = 0
        else:
            rep.failure_count += 1
            rep.consecutive_failures += 1
            rep.last_failure_reason = "; ".join(result.errors) if result.errors else ""

        if rep.is_degraded:
            logger.warning(
                "ControlPlane: agent '%s' is degraded "
                "(consecutive_failures=%d, score=%.4f, last_failure=%r)",
                agent_name, rep.consecutive_failures, rep.score, rep.last_failure_reason,
            )

        self._emit_audit(
            "agent_reputation_update",
            {
                "agent": agent_name,
                "score": rep.score,
                "success_count": rep.success_count,
                "failure_count": rep.failure_count,
                "consecutive_failures": rep.consecutive_failures,
                "is_degraded": rep.is_degraded,
                "last_failure_reason": rep.last_failure_reason,
            },
        )

    # -- agent lease pool ---------------------------------------------------

    @asynccontextmanager
    async def get_agent_lease(
        self,
        name: str,
        *,
        timeout: float | None = None,
        holder: str = "unknown",
    ) -> AsyncIterator[AgentLease]:
        """Borrow the shared instance of ``name`` with serialised access.

        This is the race-condition fix: instead of orchestrator and swarms
        each constructing their own ``DesignAgent()`` and mutating shared
        state concurrently, all callers acquire a lease on the single
        registered instance. The per-agent reentrant lock guarantees that
        only one task lineage operates on the agent at a time.

        Parameters
        ----------
        name : str
            Registered agent name (``"design_agent"``). A dotted form
            (``"design_agent.optimise"``) is reduced to the base name.
        timeout : float | None
            Max seconds to wait for the lease. Defaults to the control
            plane's ``default_lease_timeout_s``. Raises
            :class:`AgentLeaseTimeout` if exceeded.
        holder : str
            Identifier of the borrower (e.g. ``"hypothesis_swarm"``) for audit.

        Yields
        ------
        AgentLease
            A scoped handle that auto-releases on context exit.
        """
        agent_name = name.split(".")[0]
        agent = self._registry.get(agent_name)
        lock = self._locks.get(agent_name)
        if agent is None or lock is None:
            raise KeyError(
                f"Agent '{agent_name}' not registered in ControlPlane; "
                "register it once at orchestrator init before leasing."
            )

        effective_timeout = (
            self._default_lease_timeout_s if timeout is None else timeout
        )
        lease_id = f"lease-{uuid.uuid4().hex[:8]}"

        wait_start = time.monotonic()
        await lock.acquire(timeout=effective_timeout)
        wait_ms = (time.monotonic() - wait_start) * 1000

        lease = AgentLease(agent, lock, lease_id, holder)
        self._emit_audit(
            "agent_lease_acquired",
            {
                "lease_id": lease_id,
                "agent": agent_name,
                "holder": holder,
                "wait_ms": round(wait_ms, 2),
                "reentrant_depth": lock.depth,
            },
        )
        logger.debug(
            "ControlPlane: lease %s on '%s' acquired by %s (waited %.1fms, depth=%d)",
            lease_id, agent_name, holder, wait_ms, lock.depth,
        )
        try:
            yield lease
        finally:
            lease.release()
            self._emit_audit(
                "agent_lease_released",
                {"lease_id": lease_id, "agent": agent_name, "holder": holder},
            )
            logger.debug(
                "ControlPlane: lease %s on '%s' released by %s",
                lease_id, agent_name, holder,
            )

    # -- call ---------------------------------------------------------------

    async def call(
        self,
        target: str,
        input_data: BaseModel,
        *,
        caller: str = "orchestrator",
        trace_id: str | None = None,
        timeout_s: float = 300.0,
        lease_timeout_s: float | None = None,
    ) -> AgentResult:
        """Route a call to a registered agent with full audit trail.

        The call acquires a lease on the target agent for the duration of the
        invocation, so concurrent ``call`` / ``get_agent_lease`` consumers of
        the same agent are serialised and cannot interleave state mutations.

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
            Maximum seconds before the agent invocation is cancelled.
        lease_timeout_s : float | None
            Maximum seconds to wait to acquire the agent lease before
            failing fast. Defaults to the control plane default.
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

        # ── Audit: call start ─────────────────────────────────────
        call_record: dict[str, Any] = {
            "call_id": call_id,
            "caller": caller,
            "target": agent_name,
            "trace_id": trace_id,
            "input_summary": _summarize(input_data),
        }
        self._emit_audit("agent_call_start", call_record)

        # ── Acquire lease (serialised access) + execute with timeout ──────
        start = time.monotonic()
        runtime_context = await build_agent_context(
            campaign_id=self._campaign_id,
            agent_name=agent_name,
            caller=caller,
            trace_id=trace_id,
            input_data=input_data,
        )
        persist_agent_context_snapshot(runtime_context)
        context_token = set_current_agent_context(runtime_context)
        try:
            async with self.get_agent_lease(
                agent_name, timeout=lease_timeout_s, holder=caller
            ) as lease:
                result = await asyncio.wait_for(
                    lease.run(input_data, trace_id=trace_id),
                    timeout=timeout_s,
                )
        except AgentLeaseTimeout as exc:
            duration_ms = (time.monotonic() - start) * 1000
            logger.error(
                "ControlPlane: call %s -> %s could not acquire lease: %s",
                caller, agent_name, exc,
            )
            result = AgentResult(
                success=False,
                errors=[f"Lease acquisition failed: {exc}"],
                agent_name=agent_name,
                trace_id=trace_id,
                duration_ms=duration_ms,
            )
        except TimeoutError:
            duration_ms = (time.monotonic() - start) * 1000
            logger.error(
                "ControlPlane: call %s -> %s timed out after %.0fs",
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
                "ControlPlane: call %s -> %s raised %s",
                caller, agent_name, exc, exc_info=True,
            )
            result = AgentResult(
                success=False,
                errors=[str(exc)],
                agent_name=agent_name,
                trace_id=trace_id,
                duration_ms=duration_ms,
            )
        finally:
            reset_current_agent_context(context_token)

        # ── Update reputation from the obtained result ───────────
        self._update_reputation(agent_name, result)
        await harvest_agent_result(context=runtime_context, result=result)
        self._emit_trace_span(
            call_id=call_id,
            caller=caller,
            agent_name=agent_name,
            trace_id=trace_id,
            result=result,
            runtime_context=runtime_context,
        )

        # ── Audit: call end ─────────────────────────────────────
        call_record["success"] = result.success
        call_record["duration_ms"] = result.duration_ms
        call_record["errors"] = result.errors
        self._emit_audit("agent_call_end", call_record)

        return result

    # -- reputation-based routing -------------------------------------------

    async def call_best(
        self,
        candidates: list[str],
        input_data: BaseModel,
        *,
        caller: str = "orchestrator",
        trace_id: str | None = None,
        timeout_s: float = 300.0,
    ) -> AgentResult:
        """Call the highest-reputation registered candidate from ``candidates``.

        Candidates are ranked by their current reputation score (descending),
        considering only those registered in the control plane. The top-ranked
        agent is invoked via :meth:`call`. If none of the candidates are
        registered, the first candidate is used as a fallback (which surfaces a
        clear "not registered" error through ``call``).

        Parameters
        ----------
        candidates : list[str]
            Ordered list of candidate agent names to choose between.
        input_data : BaseModel
            Typed input for the chosen agent.
        caller : str
            Name of the calling agent (for audit).
        trace_id : str | None
            Correlation ID (auto-generated if omitted).
        timeout_s : float
            Maximum seconds before the agent invocation is cancelled.
        """
        registered = [c for c in candidates if self.is_registered(c)]

        if not registered:
            # Fall back to the first candidate so ``call`` reports a clear error.
            chosen = candidates[0]
        else:
            chosen = max(
                registered,
                key=lambda c: self._reputations.get(
                    c.split(".")[0], AgentReputation()
                ).score,
            )

        return await self.call(
            chosen,
            input_data,
            caller=caller,
            trace_id=trace_id,
            timeout_s=timeout_s,
        )

    # -- parallel call ------------------------------------------------------

    async def call_parallel(
        self,
        calls: list[tuple[str, BaseModel]],
        *,
        caller: str = "orchestrator",
        trace_id: str | None = None,
        timeout_s: float = 300.0,
        lease_timeout_s: float | None = None,
    ) -> list[AgentResult]:
        """Execute multiple agent calls concurrently.

        Calls targeting *distinct* agents run truly in parallel. Calls
        targeting the *same* agent are automatically serialised by that
        agent's lease lock.

        Parameters
        ----------
        calls : list of (agent_name, input_data) tuples
        """
        trace_id = trace_id or uuid.uuid4().hex[:12]
        tasks = [
            self.call(
                target,
                data,
                caller=caller,
                trace_id=trace_id,
                timeout_s=timeout_s,
                lease_timeout_s=lease_timeout_s,
            )
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
                    actor=details.get("caller", details.get("holder", "control_plane")),
                    action=f"control_plane.{action}",
                    details=details,
                )

            run_txn(_txn)
        except Exception:
            logger.debug("ControlPlane audit write failed", exc_info=True)

        if self._event_bus is not None:
            try:
                from app.core.db import utcnow_iso
                from app.services.event_bus import EventMessage

                self._event_bus.publish(EventMessage(
                    id=f"cp-{uuid.uuid4().hex[:8]}",
                    run_id=None,
                    actor=details.get("caller", details.get("holder", "control_plane")),
                    action=f"control_plane.{action}",
                    details=details,
                    created_at=utcnow_iso(),
                ))
            except Exception:
                pass

    def _emit_trace_span(
        self,
        *,
        call_id: str,
        caller: str,
        agent_name: str,
        trace_id: str,
        result: AgentResult,
        runtime_context: Any,
    ) -> None:
        """Persist a structured trace event for agent-native observability."""
        if not self._campaign_id:
            return
        context_snapshot = runtime_context.snapshot() if runtime_context else {}
        payload = {
            "type": "agent_trace_span",
            "call_id": call_id,
            "caller": caller,
            "agent": agent_name,
            "trace_id": trace_id,
            "success": result.success,
            "duration_ms": result.duration_ms,
            "errors": result.errors[:3],
            "warning_count": len(result.warnings),
            "pause_count": len(result.pause_decisions),
            "granularity_used": (
                str(result.granularity_used) if result.granularity_used is not None else ""
            ),
            "round_number": context_snapshot.get("round_number"),
            "candidate_index": context_snapshot.get("candidate_index"),
            "knowledge_event_count": context_snapshot.get("knowledge_event_count", 0),
            "blackboard_entry_count": context_snapshot.get("blackboard_entry_count", 0),
        }
        try:
            from app.services.campaign_events import log_event

            log_event(self._campaign_id, "agent_trace_span", payload)
        except Exception:
            logger.debug("ControlPlane trace persistence failed", exc_info=True)
