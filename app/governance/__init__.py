"""Governance Layer — public API.

Distributed-systems patterns for the NeverOT multi-layer agent pipeline:

  Event Sourcing   → ClaimTracker        (immutable lineage DAG)
  Circuit Breaker  → CircuitBreaker      (cascade failure prevention)
  Claim Verifier   → ClaimVerifier       (three-domain policy checks)
  Saga Pattern     → SagaCoordinator     (compensating transactions)
  Dual-track Gate  → GovernanceMiddleware (sync hard-veto + async side-car)

Typical usage
-------------
1. At startup (FastAPI lifespan Phase 2):
       from app.governance.claim_tracker import init_governance_schema
       init_governance_schema()

2. Wiring the async side-car (Phase 4, after event bus is running):
       from app.governance.middleware import start_governance_listener
       gov_sub = await start_governance_listener(event_bus)

3. Per-round inline gate (from pipeline orchestrator):
       from app.governance.middleware import GovernanceMiddleware
       middleware = GovernanceMiddleware()
       decision = middleware.gate(campaign_id, round_number, claims, ...)
       if decision.verdict == "blocked_hard":
           revision = middleware.rollback_saga(saga_id, from_layer, reason, claims)
           # pass revision to InverseDesignAgent
"""
from app.governance.claim_tracker import ClaimTracker, init_governance_schema
from app.governance.claim_verifier import ClaimVerifier
from app.governance.circuit_breaker import CircuitBreaker
from app.governance.middleware import (
    GovernanceMiddleware,
    start_governance_listener,
    stop_governance_listener,
)
from app.governance.saga_coordinator import SagaCoordinator
from app.governance.schemas import (
    BreakCondition,
    Claim,
    ClaimLineage,
    CircuitBreakerState,
    GovernanceDecision,
    GovernanceVerdict,
    PolicyViolation,
    RevisionRequest,
    SagaStep,
    SagaStepStatus,
    ViolationSeverity,
)

__all__ = [
    # Initialisation
    "init_governance_schema",
    # Core components
    "ClaimTracker",
    "ClaimVerifier",
    "CircuitBreaker",
    "SagaCoordinator",
    "GovernanceMiddleware",
    # Listener lifecycle
    "start_governance_listener",
    "stop_governance_listener",
    # Schemas
    "BreakCondition",
    "Claim",
    "ClaimLineage",
    "CircuitBreakerState",
    "GovernanceDecision",
    "GovernanceVerdict",
    "PolicyViolation",
    "RevisionRequest",
    "SagaStep",
    "SagaStepStatus",
    "ViolationSeverity",
]
