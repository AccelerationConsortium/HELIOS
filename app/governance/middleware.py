"""Governance Layer — GovernanceMiddleware (dual-track gate + async side-car).

Dual-track design
-----------------
Track 1 — Sync gate (``gate()``):
    Called inline by agents *before* they proceed.  Evaluates hard safety
    constraints via CircuitBreaker.  Returns GovernanceDecision immediately.
    A ``blocked_hard`` verdict means the caller must trigger saga rollback.

Track 2 — Async side-car (``start_governance_listener()``):
    Subscribes to the EventBus globally.  For every event that carries
    ``emitted_claims`` in its ``details``, it records claims to the lineage
    DAG and runs the statistical / amplification checks non-blocking.
    If a violation is detected it publishes a ``governance.veto`` event so
    the *next* round of the pipeline is aware of the issue without blocking
    the current one.

Also exposes thin wrappers around SagaCoordinator so callers only need to
import GovernanceMiddleware.

Lifecycle (from FastAPI lifespan)
----------------------------------
    gov_sub = await start_governance_listener(event_bus)
    ...
    await stop_governance_listener(gov_sub, event_bus)
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from app.core.db import utcnow_iso
from app.governance.circuit_breaker import CircuitBreaker
from app.governance.claim_tracker import ClaimTracker
from app.governance.saga_coordinator import SagaCoordinator
from app.governance.schemas import (
    Claim,
    CircuitBreakerState,
    GovernanceDecision,
    RevisionRequest,
    SagaStep,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level listener task (mirrors evolution.py pattern)
# ---------------------------------------------------------------------------

_listener_task: asyncio.Task[None] | None = None


# ---------------------------------------------------------------------------
# GovernanceMiddleware
# ---------------------------------------------------------------------------


class GovernanceMiddleware:
    """Central access point for all governance operations.

    One instance is sufficient for the lifetime of the application; it is
    stateless beyond holding references to the sub-components.
    """

    def __init__(self) -> None:
        self._breaker = CircuitBreaker()
        self._tracker = ClaimTracker()
        self._coordinator = SagaCoordinator()

    # ----------------------------------------------------------------- gate

    def gate(
        self,
        campaign_id: str,
        round_number: int,
        claims: list[Claim],
        safety_envelope: dict[str, Any],
        dim_defs: list[dict[str, Any]],
        kpi_history: list[float],
    ) -> GovernanceDecision:
        """Synchronous hard-veto gate — call inline before an agent proceeds.

        1. Records every claim to the lineage DAG.
        2. Evaluates through the circuit breaker (three-tier checks).
        3. Persists the decision.
        4. Returns the decision; caller checks ``decision.is_blocked``.

        A ``blocked_hard`` verdict requires the caller to trigger
        ``rollback_saga()`` and re-submit to InverseDesignAgent (L3).
        """
        # Step 1: persist claims to lineage DAG
        for claim in claims:
            try:
                self._tracker.record(campaign_id, round_number, claim)
            except Exception:
                logger.warning(
                    "Failed to record claim %s (saga continues)", claim.id, exc_info=True
                )

        # Step 2: circuit breaker evaluation
        decision = self._breaker.check(
            campaign_id=campaign_id,
            round_number=round_number,
            claims=claims,
            safety_envelope=safety_envelope,
            dim_defs=dim_defs,
            kpi_history=kpi_history,
        )

        # Step 3: persist decision
        try:
            self._tracker.record_decision(campaign_id, round_number, decision)
        except Exception:
            logger.warning(
                "Failed to persist governance decision %s", decision.id, exc_info=True
            )

        logger.info(
            "Governance gate: campaign=%s round=%d verdict=%s claims=%d",
            campaign_id,
            round_number,
            decision.verdict,
            len(claims),
        )
        return decision

    # ---------------------------------------------------------- circuit breaker

    def get_circuit_state(self, campaign_id: str) -> CircuitBreakerState:
        """Return the current circuit breaker state for *campaign_id*."""
        return self._breaker.get_state(campaign_id)

    def reset_circuit(self, campaign_id: str) -> None:
        """Manually reset the circuit breaker (use after human review)."""
        self._breaker.reset(campaign_id)

    def half_open_circuit(self, campaign_id: str) -> None:
        """Move circuit breaker to half-open for probe request."""
        self._breaker.half_open(campaign_id)

    # -------------------------------------------------------------- sagas

    def begin_saga(self, campaign_id: str, round_number: int) -> str:
        """Begin a new compensating transaction saga.

        Returns saga_id; pass to register/commit/rollback calls.
        """
        return self._coordinator.begin_saga(campaign_id, round_number)

    def register_saga_step(
        self,
        saga_id: str,
        campaign_id: str,
        round_number: int,
        layer: str,
        agent_name: str,
        compensation_payload: dict[str, Any] | None = None,
    ) -> SagaStep:
        """Register one pipeline step under this saga."""
        return self._coordinator.register_step(
            saga_id=saga_id,
            campaign_id=campaign_id,
            round_number=round_number,
            layer=layer,
            agent_name=agent_name,
            compensation_payload=compensation_payload,
        )

    def commit_saga_step(self, saga_id: str, step_id: str) -> None:
        """Mark a saga step as committed (its effects are in place)."""
        self._coordinator.commit_step(saga_id, step_id)

    def rollback_saga(
        self,
        saga_id: str,
        from_layer: str,
        reason: str,
        failed_claims: list[Claim],
    ) -> RevisionRequest:
        """Compensate all steps from *from_layer* back to L3.

        Returns RevisionRequest to pass to InverseDesignAgent for re-design.
        """
        return self._coordinator.rollback(
            saga_id=saga_id,
            from_layer=from_layer,
            reason=reason,
            failed_claims=failed_claims,
        )

    # ---------------------------------------------------------- lineage

    def get_lineage(self, claim: Claim, campaign_id: str):  # type: ignore[return]
        """Return the full ClaimLineage for a claim."""
        return self._tracker.get_lineage(claim, campaign_id)


# ---------------------------------------------------------------------------
# Async side-car listener (advisory, non-blocking)
# ---------------------------------------------------------------------------

# Module-level singleton used by the listener
_middleware: GovernanceMiddleware | None = None


def _get_middleware() -> GovernanceMiddleware:
    global _middleware
    if _middleware is None:
        _middleware = GovernanceMiddleware()
    return _middleware


def _extract_claims_from_event(details: dict[str, Any]) -> list[Claim]:
    """Parse ``emitted_claims`` from an event details dict.

    Agents should populate ``details["emitted_claims"]`` with a list of
    ``Claim.to_dict()`` payloads.  Any entry that cannot be parsed is
    skipped with a warning.
    """
    raw_claims: list[dict[str, Any]] = details.get("emitted_claims") or []
    claims: list[Claim] = []
    for raw in raw_claims:
        try:
            claims.append(
                Claim(
                    id=raw.get("id", f"cl-{uuid.uuid4().hex[:10]}"),
                    param_name=raw["param_name"],
                    source_type=raw["source_type"],
                    emitting_agent=raw["emitting_agent"],
                    trace_id=raw["trace_id"],
                    confidence=float(raw.get("confidence", 1.0)),
                    source_ref=raw.get("source_ref", ""),
                    param_value=raw.get("param_value"),
                    param_value_str=raw.get("param_value_str"),
                    parent_trace_ids=tuple(raw.get("parent_trace_ids", [])),
                    created_at=raw.get("created_at", utcnow_iso()),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Could not parse emitted_claim entry: %s — %s", raw, exc)
    return claims


async def _handle_governance_event(event: Any, bus: Any) -> None:
    """Process one EventBus event in the governance side-car.

    Looks for ``emitted_claims`` in event.details.  If found:
    1. Records claims to lineage DAG.
    2. Runs circuit breaker statistical checks.
    3. If blocked, publishes ``governance.veto`` event for next-round handling.

    Advisory — all errors are caught and logged, never re-raised.
    """
    details: dict[str, Any] = event.details or {}
    claims = _extract_claims_from_event(details)
    if not claims:
        return

    campaign_id: str | None = details.get("campaign_id") or details.get("campaign")
    round_number_raw = details.get("round_number", 0)
    if campaign_id is None:
        # Cannot index without campaign_id; skip silently
        return

    try:
        round_number = int(round_number_raw)
    except (TypeError, ValueError):
        round_number = 0

    middleware = _get_middleware()

    # Record claims to lineage DAG (non-blocking, advisory)
    for claim in claims:
        try:
            middleware._tracker.record(campaign_id, round_number, claim)
        except Exception:
            logger.debug("Side-car: failed to record claim %s", claim.id, exc_info=True)

    # Run statistical / amplification checks (side-car only; no hard block here)
    safety_envelope: dict[str, Any] = details.get("safety_envelope") or {}
    dim_defs: list[dict[str, Any]] = details.get("dim_defs") or []
    kpi_history: list[float] = details.get("kpi_history") or []

    try:
        decision = middleware._breaker.check(
            campaign_id=campaign_id,
            round_number=round_number,
            claims=claims,
            safety_envelope=safety_envelope,
            dim_defs=dim_defs,
            kpi_history=kpi_history,
        )
        middleware._tracker.record_decision(campaign_id, round_number, decision)

        if decision.is_blocked:
            logger.warning(
                "Governance side-car veto: campaign=%s round=%d verdict=%s",
                campaign_id,
                round_number,
                decision.verdict,
            )
            # Publish governance.veto so next-round handlers can react
            if bus is not None:
                from app.services.event_bus import EventMessage

                veto_event = EventMessage(
                    id=f"gov-{uuid.uuid4().hex[:8]}",
                    run_id=event.run_id,
                    actor="governance",
                    action="governance.veto",
                    details={
                        "campaign_id": campaign_id,
                        "round_number": round_number,
                        "verdict": decision.verdict,
                        "decision_id": decision.id,
                        "violations": [v.to_dict() for v in decision.violations],
                    },
                    created_at=utcnow_iso(),
                )
                bus.publish(veto_event)

    except Exception:
        logger.warning(
            "Governance side-car check failed for campaign=%s", campaign_id, exc_info=True
        )


async def start_governance_listener(bus: Any) -> Any:
    """Subscribe to the global EventBus stream and start the side-car task.

    Mirrors the ``start_*_listener`` convention used across services.

    Returns the Subscription handle required by ``stop_governance_listener``.
    """
    global _listener_task

    sub = await bus.subscribe(run_id=None)  # global subscription

    async def _listen() -> None:
        async for event in sub:
            # Only handle events that carry emitted_claims
            if "emitted_claims" in (event.details or {}):
                await _handle_governance_event(event, bus)

    _listener_task = asyncio.create_task(_listen())
    logger.info("Governance side-car listener started")
    return sub


async def stop_governance_listener(sub: Any, bus: Any) -> None:
    """Cancel the governance side-car listener and unsubscribe."""
    global _listener_task

    sub.cancel()
    await bus.unsubscribe(sub)

    if _listener_task is not None:
        _listener_task.cancel()
        try:
            await _listener_task
        except asyncio.CancelledError:
            pass
        _listener_task = None

    logger.info("Governance side-car listener stopped")
