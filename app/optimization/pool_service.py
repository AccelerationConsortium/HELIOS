"""CandidatePoolService -- the unified candidate-and-strategy layer (Phase B-2).

The HELIOS authority's ``arbitrate_next`` (the single decision entry) delegates
candidate construction to this service.  It gathers candidates from pluggable
*sources*, assembles them into a ``CandidatePool`` via the arbitration builder,
and applies the failure-zone penalty.  The hard constraint gate and advisor
hints are downstream: the gate lives in ``OptimizationDecisionPolicy.arbitrate``;
advisor hints ride on the authority ``StrategyDecision`` (``recommended_backends``).

Layering (user-approved):

    arbitrate_next                 # authority: the only next-step decision entry
      -> CandidatePoolService      # this module: compose sources -> pool
           ├ NexusSource           # Nexus provider top-k
           ├ LocalSource           # local baseline fallback
           ├ ArchetypeSource       # nexus archetype-scored pool
           └ BomcpSource           # real GP-BO proposals (bo-engine)
      -> policy.arbitrate          # hard gate + delegated soft scoring

Dependency direction respects AC5: this boundary module may import authority
(app.services), never the reverse.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from app.optimization.candidate_pool import CandidatePoolBuilder
from app.optimization.schemas import CandidatePool, CandidateSuggestion, OptimizationRequest

if TYPE_CHECKING:  # pragma: no cover
    from app.services.strategy_models import StrategyDecision

logger = logging.getLogger(__name__)


@runtime_checkable
class CandidateSource(Protocol):
    """A pluggable candidate generator feeding the pool."""

    name: str

    def propose(
        self, request: OptimizationRequest, decision: StrategyDecision
    ) -> CandidateSuggestion | None:
        """Return a suggestion, or None when this source has nothing to offer."""
        ...


# ---------------------------------------------------------------------------
# Concrete sources
# ---------------------------------------------------------------------------


class NexusSource:
    """Nexus provider top-k (best-effort; None on outage)."""

    name = "nexus"

    def __init__(self, provider: object | None = None, k: int = 3) -> None:
        self._provider = provider
        self._k = k

    def propose(self, request, decision):
        provider = self._provider
        if provider is None:
            from app.optimization.nexus_provider import NexusOptimizationProvider

            provider = NexusOptimizationProvider()
        try:
            if not provider.is_available():
                return None
            top_k = getattr(provider, "suggest_top_k", None)
            return top_k(request, self._k) if top_k is not None else provider.suggest(request)
        except Exception:
            logger.warning("NexusSource failed; skipping", exc_info=True)
            return None


class LocalSource:
    """Local space-filling baseline -- always available, never empty."""

    name = "local"

    def __init__(self, fallback: object | None = None) -> None:
        self._fallback = fallback

    def propose(self, request, decision):
        fallback = self._fallback
        if fallback is None:
            from app.optimization.local_fallback import LocalFallbackProvider

            fallback = LocalFallbackProvider()
        try:
            return fallback.suggest(request)
        except Exception:
            logger.warning("LocalSource failed; skipping", exc_info=True)
            return None


class ArchetypeSource:
    """Nexus archetype-scored multi-source pool (Δ3/Δ7)."""

    name = "archetype"

    def propose(self, request, decision):
        from app.optimization.archetype_pool import build_candidate_pool

        try:
            res = build_candidate_pool(
                request,
                fingerprint=request.context.get("fingerprint"),
                backend_name=decision.backend_name,
                k=max(4, request.n * 2),
                select_n=request.n,
            )
        except Exception:
            logger.warning("ArchetypeSource failed; skipping", exc_info=True)
            return None
        if not res.selected:
            return None
        return CandidateSuggestion(
            candidates=tuple(res.selected),
            algorithm="archetype_pool",
            source="archetype",
            rationale=f"archetype={res.archetype}",
        )


class BomcpSource:
    """Real GP Bayesian-optimization proposals via the bomcp backend."""

    name = "bomcp"

    def propose(self, request, decision):
        if not request.observations:
            return None  # GP needs history; defer to other sources early on
        from app.services.optimization_backends import get_backend

        try:
            backend = get_backend("bomcp")
            if backend.name != "bomcp":  # degraded (bo-engine absent) -> not this source
                return None
            cands = backend.suggest(
                request.space, request.n, list(request.observations), seed=request.seed
            )
        except Exception:
            logger.warning("BomcpSource failed; skipping", exc_info=True)
            return None
        if not cands:
            return None
        return CandidateSuggestion(
            candidates=tuple(cands), algorithm="bomcp", source="bomcp",
        )


def default_sources() -> list[CandidateSource]:
    """The production source set, in pool-preference order."""
    return [NexusSource(), LocalSource(), ArchetypeSource(), BomcpSource()]


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class CandidatePoolService:
    """Compose candidate sources into an arbitration-ready ``CandidatePool``."""

    def __init__(
        self,
        sources: list[CandidateSource] | None = None,
        *,
        builder: CandidatePoolBuilder | None = None,
    ) -> None:
        self._sources = list(sources) if sources is not None else default_sources()
        self._builder = builder or CandidatePoolBuilder()

    def build_pool(
        self, request: OptimizationRequest, decision: StrategyDecision
    ) -> CandidatePool:
        nexus_sug: CandidateSuggestion | None = None
        local_sug: CandidateSuggestion | None = None
        extra: list[CandidateSuggestion] = []

        for src in self._sources:
            sug = src.propose(request, decision)
            if sug is None or not sug.candidates:
                continue
            if src.name == "nexus":
                nexus_sug = sug
            elif src.name == "local":
                local_sug = sug
            else:
                extra.append(sug)

        pool = self._builder.build(
            request,
            decision,
            nexus_suggestion=nexus_sug,
            local_suggestion=local_sug,
            extra_suggestions=tuple(extra),
        )
        return self._apply_failure_penalty(pool, request)

    @staticmethod
    def _apply_failure_penalty(pool: CandidatePool, request: OptimizationRequest) -> CandidatePool:
        """Drop candidates inside the learned failure region; never strand empty."""
        failed = request.context.get("failed_params") or ()
        if not failed or not pool.candidates:
            return pool
        from app.services.failure_region import FailureRegionModel

        model = FailureRegionModel.fit(failed=list(failed), space=request.space)
        kept = tuple(c for c in pool.candidates if model.predicted_feasible(c.params))
        if not kept or len(kept) == len(pool.candidates):
            return pool  # all-failure-prone (keep the round alive) or nothing dropped
        dropped = len(pool.candidates) - len(kept)
        return CandidatePool(
            candidates=kept,
            sources_used=pool.sources_used,
            sources_dropped=pool.sources_dropped,
            construction_trace=pool.construction_trace
            + (f"failure-zone penalty: dropped {dropped} failure-prone candidate(s)",),
        )
