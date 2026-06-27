"""Optimization service entrypoint for provider-level callers.

``suggest_next`` wires the optimization-intelligence provider layer together
with a hard graceful-degradation guarantee:

    provider (Nexus)  --unavailable/raises-->  local fallback
                       |
                       v
              decision policy (HELIOS authority)
                       |
                       v
              provenance record

A Nexus outage degrades to the built-in optimizer; it never stops a campaign.
HELIOS retains the final say through the decision policy, and every round is
recorded for audit.

The production campaign loop currently routes through the adaptive strategy
selector and backend registry directly because it also has to thread
campaign-local state such as BO MCP TuRBO trust regions and failure-region
avoidance.  This service remains the stable facade for direct provider calls
and tests of the Nexus/local fallback contract.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.optimization.decision_policy import OptimizationDecisionPolicy
from app.optimization.local_fallback import LocalFallbackProvider
from app.optimization.nexus_provider import NexusOptimizationProvider
from app.optimization.provenance import ProvenanceLogger
from app.optimization.schemas import (
    CandidateSuggestion,
    DecisionResult,
    OptimizationProvider,
    OptimizationRequest,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizationOutcome:
    """Result of one optimization round."""

    suggestion: CandidateSuggestion
    decision: DecisionResult
    provenance: dict[str, Any]


def _suggest_with_fallback(
    request: OptimizationRequest,
    provider: OptimizationProvider,
    fallback: OptimizationProvider,
) -> CandidateSuggestion:
    """Try the primary provider; fall back on unavailability or any error."""
    try:
        if provider.is_available():
            return provider.suggest(request)
        logger.info("Optimization provider unavailable; using local fallback")
    except Exception:
        logger.warning(
            "Optimization provider failed; falling back to local optimizer",
            exc_info=True,
        )
    return fallback.suggest(request)


def suggest_next(
    request: OptimizationRequest,
    *,
    provider: OptimizationProvider | None = None,
    fallback: OptimizationProvider | None = None,
    policy: OptimizationDecisionPolicy | None = None,
    provenance: ProvenanceLogger | None = None,
) -> OptimizationOutcome:
    """Propose, validate, and record the next experiment(s) for a campaign."""
    provider = provider if provider is not None else NexusOptimizationProvider()
    fallback = fallback if fallback is not None else LocalFallbackProvider()
    policy = policy if policy is not None else OptimizationDecisionPolicy()
    provenance = provenance if provenance is not None else ProvenanceLogger()

    suggestion = _suggest_with_fallback(request, provider, fallback)
    decision = policy.evaluate(suggestion, request)
    record = provenance.record(request, suggestion, decision)

    return OptimizationOutcome(suggestion=suggestion, decision=decision, provenance=record)
