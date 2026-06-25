"""Nexus optimization-intelligence provider.

HELIOS's facade over Nexus.  It answers two questions and nothing more:

* ``diagnose(request)`` -- *what does this problem look like?*  Runs Nexus's
  ``ProblemProfiler`` (fingerprint) and ``DiagnosticEngine`` (17 signals) and
  returns plain dicts (Nexus internals never leak across the boundary).
* ``suggest(request)``  -- *what should we try next?*  Generates candidates via
  the registered ``nexus_*`` backend bridge and attaches the fingerprint,
  diagnostics, and rationale.

It does **not** decide whether to run anything, talk to PUDA, or override
HELIOS's strategy selection.  The algorithm is chosen by the caller
(``request.context["backend"]``); the fingerprint's ``simplification_hint`` is
surfaced only as a *recommendation* in the rationale.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Any

from app.optimization.nexus_backend import (
    _nexus_available,
    to_nexus_observations,
    to_nexus_specs,
)
from app.optimization.schemas import CandidateSuggestion, OptimizationRequest
from app.services.optimization_backends import get_backend

logger = logging.getLogger(__name__)

_DEFAULT_BACKEND = "nexus_gp_bo"


class NexusUnavailableError(RuntimeError):
    """Raised when a Nexus operation is attempted but Nexus is not installed."""


class NexusOptimizationProvider:
    """``OptimizationProvider`` backed by Nexus's optimization core."""

    default_backend = _DEFAULT_BACKEND

    def is_available(self) -> bool:
        return _nexus_available()

    # -- snapshot construction -------------------------------------------

    def _snapshot(self, request: OptimizationRequest) -> Any:
        """Build a Nexus ``CampaignSnapshot`` for advisory profiling.

        Uses the *real* objective with an explicit direction so diagnostic
        trends stay in natural units (no negation here).
        """
        from optimization_copilot.core.models import CampaignSnapshot

        direction = "maximize" if request.direction == "maximize" else "minimize"
        return CampaignSnapshot(
            campaign_id=request.campaign_id,
            parameter_specs=to_nexus_specs(request.space),
            observations=to_nexus_observations(
                list(request.observations), request.objective_name, negate=False
            ),
            objective_names=[request.objective_name],
            objective_directions=[direction],
            current_iteration=request.round_index,
        )

    # -- advisory: fingerprint + diagnostics -----------------------------

    def diagnose(self, request: OptimizationRequest) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return ``(fingerprint, diagnostics)`` as plain dicts."""
        if not self.is_available():
            raise NexusUnavailableError("Nexus is not installed")

        from optimization_copilot.diagnostics import DiagnosticEngine
        from optimization_copilot.profiler import ProblemProfiler

        snapshot = self._snapshot(request)
        fingerprint = ProblemProfiler().profile(snapshot).to_dict()
        diagnostics = DiagnosticEngine().compute(snapshot).to_dict()
        return fingerprint, diagnostics

    # -- candidate generation --------------------------------------------

    def suggest(self, request: OptimizationRequest) -> CandidateSuggestion:
        if not self.is_available():
            raise NexusUnavailableError("Nexus is not installed")

        backend_name = request.context.get("backend") or self.default_backend

        # Fingerprint + diagnostics are best-effort advisory metadata; a
        # failure here must not prevent candidate generation.
        fingerprint: dict[str, Any] = {}
        diagnostics: dict[str, Any] = {}
        try:
            fingerprint, diagnostics = self.diagnose(request)
        except Exception:  # pragma: no cover - defensive
            logger.warning("Nexus profiling failed; continuing without it", exc_info=True)

        backend = get_backend(backend_name)
        candidates = backend.suggest(
            request.space,
            request.n,
            list(request.observations),
            seed=request.seed,
        )

        hint = fingerprint.get("simplification_hint")
        rationale = f"Nexus generated {len(candidates)} candidate(s) via {backend_name}."
        if hint:
            rationale += f" Problem profile suggests: {hint}."

        return CandidateSuggestion(
            candidates=tuple(candidates),
            algorithm=backend_name,
            source="nexus",
            confidence=0.6,
            rationale=rationale,
            diagnostics=diagnostics,
            fingerprint=fingerprint,
            seed=request.seed,
        )

    # -- top-k portfolio (Δ1, Phase 1) -----------------------------------

    def suggest_top_k(self, request: OptimizationRequest, k: int = 3) -> CandidateSuggestion:
        """Return up to ``k`` candidates with **shared** fingerprint/diagnostics.

        Phase 1: candidates come from the chosen ``nexus_*`` backend with
        ``n=k``; the fingerprint and diagnostics describe the whole batch, and
        ``per_candidate`` is empty (so downstream deltas are 0 and candidates
        inherit their authority archetype's utility).  Per-candidate diagnostics
        are a Phase-2 enhancement (Δ4).
        """
        topk_request = dataclasses.replace(request, n=max(1, k))
        return self.suggest(topk_request)
