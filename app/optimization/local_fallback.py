"""Local fallback optimization provider.

Wraps HELIOS's always-available ``built_in`` backend behind the
``OptimizationProvider`` interface so that a Nexus outage (or Nexus simply not
being installed) never stops a campaign.  This *is* the "LocalFallbackOptimizer"
from the design -- it reuses existing infrastructure rather than adding a new
optimizer.
"""
from __future__ import annotations

from app.optimization.schemas import CandidateSuggestion, OptimizationRequest
from app.services.optimization_backends import get_backend


class LocalFallbackProvider:
    """``OptimizationProvider`` backed by the built-in HELIOS optimizer."""

    backend_name = "built_in"

    def is_available(self) -> bool:
        # built_in has no optional dependencies; it is always available.
        return True

    def suggest(self, request: OptimizationRequest) -> CandidateSuggestion:
        backend = get_backend(self.backend_name)
        candidates = backend.suggest(
            request.space,
            request.n,
            list(request.observations),
            seed=request.seed,
        )
        return CandidateSuggestion(
            candidates=tuple(candidates),
            algorithm=self.backend_name,
            source="local_fallback",
            confidence=0.3,
            rationale="Nexus unavailable; used HELIOS built-in optimizer.",
            seed=request.seed,
        )
