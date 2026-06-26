"""Optimization service entrypoint -- the one call HELIOS makes per round.

``suggest_next`` wires the optimization-intelligence layer together with a
hard graceful-degradation guarantee:

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
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.optimization.arbitration_config import ArbitrationConfig
from app.optimization.candidate_pool import CandidatePoolBuilder
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

if TYPE_CHECKING:  # pragma: no cover
    from app.services.strategy_models import StrategyDecision

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


def _nexus_top_k(
    request: OptimizationRequest,
    provider: OptimizationProvider,
    k: int,
) -> CandidateSuggestion | None:
    """Best-effort Nexus top-k; returns None on unavailability or any error."""
    try:
        if not provider.is_available():
            logger.info("Optimization provider unavailable; arbitrating local-only")
            return None
        top_k = getattr(provider, "suggest_top_k", None)
        if top_k is not None:
            return top_k(request, k)
        return provider.suggest(request)
    except Exception:
        logger.warning("Optimization provider failed; arbitrating local-only", exc_info=True)
        return None


def arbitrate_next(
    request: OptimizationRequest,
    decision: StrategyDecision,
    *,
    provider: OptimizationProvider | None = None,
    fallback: OptimizationProvider | None = None,
    policy: OptimizationDecisionPolicy | None = None,
    provenance: ProvenanceLogger | None = None,
    pool_builder: CandidatePoolBuilder | None = None,
    config: ArbitrationConfig | None = None,
) -> OptimizationOutcome:
    """Deep path: build a multi-source pool and arbitrate it under the authority.

    ``decision`` is the HELIOS authority's ``StrategyDecision`` (from
    ``select_strategy``).  This function does not recompute it -- it only builds
    the candidate pool, applies the hard gate + delegated soft scoring, and
    records provenance.  A Nexus outage degrades to the local baseline; the
    round is never empty.
    """
    provider = provider if provider is not None else NexusOptimizationProvider()
    fallback = fallback if fallback is not None else LocalFallbackProvider()
    policy = policy if policy is not None else OptimizationDecisionPolicy()
    provenance = provenance if provenance is not None else ProvenanceLogger()
    config = config if config is not None else ArbitrationConfig()
    pool_builder = pool_builder if pool_builder is not None else CandidatePoolBuilder(config)

    nexus_suggestion = _nexus_top_k(request, provider, config.k_nexus)
    local_suggestion = fallback.suggest(request)

    pool = pool_builder.build(
        request,
        decision,
        nexus_suggestion=nexus_suggestion,
        local_suggestion=local_suggestion,
    )
    result = policy.arbitrate(pool, request, decision, config=config)

    # Provenance source-of-record: the primary generator this round, plus the
    # authority's Δ2 backend-selection trace (so one record holds both axes).
    primary = nexus_suggestion if nexus_suggestion is not None else local_suggestion
    record = provenance.record(request, primary, result, strategy_decision=decision)

    return OptimizationOutcome(suggestion=primary, decision=result, provenance=record)
