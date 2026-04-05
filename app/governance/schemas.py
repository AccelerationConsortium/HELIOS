"""Governance Layer — core data types.

All schemas used across the governance subsystem:
  Claim            : a structured assertion emitted by an agent
  ClaimLineage     : DAG node tracking causal ancestry
  PolicyViolation  : a single rule violation with severity
  GovernanceDecision: verdict + evidence from any governance check
  BreakCondition   : reason a circuit breaker was tripped
  SagaStep         : one compensatable step in a multi-layer transaction
  RevisionRequest  : signal sent from Saga back to L3 for re-design
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from app.core.db import utcnow_iso

# ---------------------------------------------------------------------------
# Type aliases (string literal unions — no enums per project rules)
# ---------------------------------------------------------------------------

ClaimSourceType = Literal[
    "literature",
    "prior_experiment",
    "model_prediction",
    "user_input",
    "llm_inference",
]

GovernanceVerdict = Literal[
    "approved",
    "approved_with_warning",
    "blocked_soft",   # pause + human review required
    "blocked_hard",   # immediate stop + saga rollback
]

CircuitBreakerState = Literal["closed", "open", "half_open"]

SagaStepStatus = Literal[
    "pending",
    "committed",
    "compensating",
    "compensated",
    "failed",
]

ViolationSeverity = Literal["warning", "soft_block", "hard_block"]


# ---------------------------------------------------------------------------
# Claim — the atomic unit of governance
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Claim:
    """A structured assertion emitted by an agent about a parameter value.

    Agents should emit claims explicitly via ``emitted_claims`` output fields
    rather than having governance parse free-text rationales post-hoc.
    """

    param_name: str
    source_type: ClaimSourceType
    emitting_agent: str
    trace_id: str
    confidence: float                        # [0.0, 1.0]
    source_ref: str = ""                     # DOI, run_id, model_name, etc.
    param_value: float | None = None         # numeric parameters
    param_value_str: str | None = None       # categorical parameters
    parent_trace_ids: tuple[str, ...] = ()   # causal parents (vector-clock edges)
    id: str = field(
        default_factory=lambda: f"cl-{uuid.uuid4().hex[:10]}"
    )
    created_at: str = field(default_factory=utcnow_iso)

    def display_value(self) -> str:
        """Human-readable value for logging."""
        if self.param_value is not None:
            return str(self.param_value)
        return self.param_value_str or "<unset>"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "param_name": self.param_name,
            "param_value": self.param_value,
            "param_value_str": self.param_value_str,
            "source_type": self.source_type,
            "source_ref": self.source_ref,
            "emitting_agent": self.emitting_agent,
            "trace_id": self.trace_id,
            "confidence": self.confidence,
            "parent_trace_ids": list(self.parent_trace_ids),
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# ClaimLineage — causal DAG node
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClaimLineage:
    """Causal ancestry of a single claim.

    Captures how many agents downstream referenced this claim and how much
    its value deviates from historical norms — together these form the
    amplification_factor used by the circuit breaker.
    """

    claim_id: str
    claim: Claim
    ancestor_claims: tuple[Claim, ...]  # ordered root → immediate parent
    causal_depth: int                   # 0 = no parents (root claim)
    downstream_ref_count: int           # how many later claims cite this one
    amplification_factor: float         # downstream_ref_count × σ_deviation

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim": self.claim.to_dict(),
            "ancestor_claims": [c.to_dict() for c in self.ancestor_claims],
            "causal_depth": self.causal_depth,
            "downstream_ref_count": self.downstream_ref_count,
            "amplification_factor": self.amplification_factor,
        }


# ---------------------------------------------------------------------------
# PolicyViolation — a single rule breach
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolicyViolation:
    """One policy rule violation with severity and evidence."""

    rule_id: str
    message: str
    severity: ViolationSeverity
    param_name: str | None = None
    observed: float | None = None
    threshold: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "message": self.message,
            "severity": self.severity,
            "param_name": self.param_name,
            "observed": self.observed,
            "threshold": self.threshold,
        }


# ---------------------------------------------------------------------------
# GovernanceDecision — full verdict record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GovernanceDecision:
    """Result of a governance gate evaluation.

    The verdict field is the authoritative outcome:
      approved              → proceed normally
      approved_with_warning → proceed, but log warnings
      blocked_soft          → pause, emit human-review event
      blocked_hard          → stop immediately, trigger saga rollback
    """

    verdict: GovernanceVerdict
    triggered_by: str    # "circuit_breaker" | "claim_verifier" | "saga_rollback" | "open_breaker"
    violations: tuple[PolicyViolation, ...]
    warnings: tuple[PolicyViolation, ...]
    claim: Claim | None = None
    amplification_factor: float | None = None
    sigma_deviation: float | None = None
    id: str = field(
        default_factory=lambda: f"gd-{uuid.uuid4().hex[:10]}"
    )
    created_at: str = field(default_factory=utcnow_iso)

    @property
    def is_blocked(self) -> bool:
        return self.verdict in ("blocked_soft", "blocked_hard")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "verdict": self.verdict,
            "triggered_by": self.triggered_by,
            "violations": [v.to_dict() for v in self.violations],
            "warnings": [w.to_dict() for w in self.warnings],
            "claim": self.claim.to_dict() if self.claim else None,
            "amplification_factor": self.amplification_factor,
            "sigma_deviation": self.sigma_deviation,
            "created_at": self.created_at,
        }


def _max_verdict(decisions: list[GovernanceDecision]) -> GovernanceVerdict:
    """Return the most severe verdict from a list of decisions."""
    order: list[GovernanceVerdict] = [
        "approved",
        "approved_with_warning",
        "blocked_soft",
        "blocked_hard",
    ]
    best = "approved"
    for d in decisions:
        if order.index(d.verdict) > order.index(best):
            best = d.verdict
    return best  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# BreakCondition — circuit breaker trip evidence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BreakCondition:
    """Evidence record written when a circuit breaker is tripped."""

    reason: str
    amplification_factor: float
    sigma_deviation: float | None
    claim_id: str | None
    created_at: str = field(default_factory=utcnow_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "amplification_factor": self.amplification_factor,
            "sigma_deviation": self.sigma_deviation,
            "claim_id": self.claim_id,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# SagaStep — one step in a compensating transaction chain
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SagaStep:
    """One layer's step in a multi-layer saga.

    When the saga is rolled back, compensation_payload is passed to the
    layer's compensating handler so it can undo its effects.
    """

    id: str
    saga_id: str
    layer: str       # "L0" | "L1" | "L2" | "L3"
    agent_name: str
    step_order: int  # 0-based; lower = earlier; rollback goes highest first
    status: SagaStepStatus = "pending"
    compensation_payload: dict[str, Any] = field(default_factory=dict)
    committed_at: str | None = None
    compensated_at: str | None = None
    created_at: str = field(default_factory=utcnow_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "saga_id": self.saga_id,
            "layer": self.layer,
            "agent_name": self.agent_name,
            "step_order": self.step_order,
            "status": self.status,
            "compensation_payload": self.compensation_payload,
            "committed_at": self.committed_at,
            "compensated_at": self.compensated_at,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# RevisionRequest — L3 re-design signal
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RevisionRequest:
    """Sent by the SagaCoordinator back to InverseDesignAgent (L3).

    Contains the failure context so the agent can revise its candidate
    systems with tighter constraints rather than blind retry.
    """

    saga_id: str
    campaign_id: str
    round_number: int
    failed_layer: str
    failure_summary: str
    failed_claims: tuple[Claim, ...]
    suggested_constraints: dict[str, Any]  # e.g. {"temperature_c": {"max": 200}}
    created_at: str = field(default_factory=utcnow_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "saga_id": self.saga_id,
            "campaign_id": self.campaign_id,
            "round_number": self.round_number,
            "failed_layer": self.failed_layer,
            "failure_summary": self.failure_summary,
            "failed_claims": [c.to_dict() for c in self.failed_claims],
            "suggested_constraints": self.suggested_constraints,
            "created_at": self.created_at,
        }
