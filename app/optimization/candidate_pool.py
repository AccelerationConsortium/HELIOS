"""CandidatePoolBuilder -- assemble the concrete portfolio for arbitration (Δ1).

Guiding principle, made literal here:

    HELIOS authority decides the archetype.
    Nexus / backends generate candidates.
    The boundary layer builds and audits the pool.

So Nexus- and local-generated candidates inherit the **authority-selected**
action archetype (``StrategyDecision``), never the backend's implied behaviour.
The generating backend is recorded separately as ``generator_backend`` for
audit.  Local *baselines* with an intrinsic role (replicate-best, recovery,
sobol/LHS diversity) carry their own explicit archetype.

source_action precedence:
  1. explicit per-candidate archetype from the provider (if any);
  2. else the authority-selected action;
  3. (last resort) ``action_from_backend_name`` -- only when no authority action
     is available, which should not happen on the normal path.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from app.optimization.arbitration_config import ArbitrationConfig
from app.optimization.schemas import (
    CandidatePool,
    CandidateSuggestion,
    OptimizationRequest,
    PooledCandidate,
)
from app.services.candidate_gen import sample_lhs

if TYPE_CHECKING:  # pragma: no cover
    from app.services.strategy_models import StrategyDecision

_EXPLORATION_PHASES = {"exploration", "explore"}


def _authority_action(decision: StrategyDecision) -> str:
    """The archetype HELIOS authority selected this round.

    ``actions_considered`` is ranked by utility (best first), so its head is the
    selected action.  Fall back to the (normalized) phase label if empty.
    """
    if decision.actions_considered:
        return decision.actions_considered[0].name
    phase = decision.phase
    return {
        "exploration": "explore",
        "exploitation": "exploit",
        "refinement": "refine",
    }.get(phase, phase)


class CandidatePoolBuilder:
    """Build a ``CandidatePool`` from provider suggestions + authority decision."""

    def __init__(self, config: ArbitrationConfig | None = None) -> None:
        self._config = config or ArbitrationConfig()

    def build(
        self,
        request: OptimizationRequest,
        decision: StrategyDecision,
        *,
        nexus_suggestion: CandidateSuggestion | None = None,
        local_suggestion: CandidateSuggestion | None = None,
    ) -> CandidatePool:
        cfg = self._config
        authority_action = _authority_action(decision)
        candidates: list[PooledCandidate] = []
        used: list[str] = []
        dropped: list[str] = []
        trace: list[str] = []

        # --- Nexus top-k (archetype = authority, unless explicitly overridden) ---
        if nexus_suggestion is not None and nexus_suggestion.candidates:
            for i, params in enumerate(nexus_suggestion.candidates):
                action = _explicit_action(nexus_suggestion, i) or authority_action
                candidates.append(
                    PooledCandidate(
                        params=dict(params),
                        source="nexus",
                        source_action=action,
                        generator_backend=nexus_suggestion.algorithm,
                        rationale=nexus_suggestion.rationale,
                    )
                )
            used.append("nexus")
            trace.append(
                f"nexus: {len(nexus_suggestion.candidates)} candidate(s) via "
                f"{nexus_suggestion.algorithm}, archetype={authority_action}"
            )
        else:
            dropped.append("nexus")
            trace.append("nexus: no suggestion (dropped)")

        # --- Local baseline (archetype = authority) ---
        if cfg.include_local_baseline and local_suggestion is not None and local_suggestion.candidates:
            for params in local_suggestion.candidates:
                candidates.append(
                    PooledCandidate(
                        params=dict(params),
                        source="local",
                        source_action=authority_action,
                        generator_backend=local_suggestion.algorithm,
                        rationale=local_suggestion.rationale,
                    )
                )
            used.append("local")
            trace.append(f"local: {len(local_suggestion.candidates)} baseline(s)")
        elif cfg.include_local_baseline:
            dropped.append("local")
            trace.append("local: no baseline (dropped)")

        # --- Replicate-best (explicit archetype = stabilize), from stabilize_spec ---
        spec = decision.stabilize_spec
        if cfg.include_replicate_best and spec is not None and spec.points_to_replicate:
            for params in spec.points_to_replicate:
                candidates.append(
                    PooledCandidate(
                        params=dict(params),
                        source="replicate",
                        source_action="stabilize",
                        generator_backend="replicate",
                        rationale=spec.reason,
                    )
                )
            used.append("replicate")
            trace.append(
                f"replicate: {len(spec.points_to_replicate)} point(s) from stabilize_spec"
            )

        # --- Sobol/LHS diversity (explicit archetype = explore), exploration only ---
        if cfg.include_sobol_in_exploration and decision.phase in _EXPLORATION_PHASES:
            diversity = sample_lhs(request.space, max(1, request.n), seed=request.seed)
            for params in diversity:
                candidates.append(
                    PooledCandidate(
                        params=dict(params),
                        source="sobol",
                        source_action="explore",
                        generator_backend="lhs",
                        rationale="space-filling diversity (exploration phase)",
                    )
                )
            used.append("sobol")
            trace.append(f"sobol/lhs: {len(diversity)} diversity candidate(s)")

        return CandidatePool(
            candidates=tuple(candidates),
            sources_used=tuple(used),
            sources_dropped=tuple(dropped),
            construction_trace=tuple(trace),
        )


def _explicit_action(suggestion: CandidateSuggestion, index: int) -> str | None:
    """Read an explicit per-candidate archetype, if the provider supplied one."""
    if index < len(suggestion.per_candidate):
        meta = suggestion.per_candidate[index]
        action = meta.get("source_action") or meta.get("action") or meta.get("archetype")
        if action:
            return str(action)
    return None
