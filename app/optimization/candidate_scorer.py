"""Delegating candidate scorer -- the bridge to the single source of utility truth.

The HELIOS authority (``strategy_selector.select_strategy``) scores four
**abstract action archetypes** (explore / exploit / refine / stabilize) with
``utility = w_improvement·improvement + w_info_gain·info_gain − w_risk·risk``.
It never scores a concrete parameter dict.

This module projects *concrete* pooled candidates onto those archetype
utilities, using the authority's **own** ``weights_used`` vector.  Each
candidate inherits its archetype's utility as a base, then receives a delta
computed from its per-candidate diagnostics under the *same* weights.  When no
diagnostics are available, the delta is exactly 0 -- the candidate ranks purely
by the authority's archetype decision.

Design invariants (enforced by tests + CI guardrails):
  * No utility-weight constants are defined here.  Weights come from the
    ``StrategyDecision`` (or, if adaptive weights were disabled, from
    ``PhaseConfig()`` defaults -- still the authority's source, not ours).
  * No phase is recomputed here.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from app.optimization.arbitration_config import ArbitrationConfig
from app.optimization.schemas import PooledCandidate, ScoredCandidate
from app.services.strategy_models import PhaseConfig, WeightsUsed

if TYPE_CHECKING:  # pragma: no cover
    from app.services.candidate_gen import ParameterSpace
    from app.services.strategy_models import ActionCandidate, StrategyDecision

# Phase labels may leak in as ``source_action``; normalize to archetype names.
_ACTION_ALIASES = {
    "exploration": "explore",
    "exploitation": "exploit",
    "refinement": "refine",
}


def _archetype_name(source_action: str) -> str:
    return _ACTION_ALIASES.get(source_action, source_action)


def _effective_weights(decision: StrategyDecision) -> WeightsUsed:
    """The authority's weights, or PhaseConfig defaults if adaptive weights were off."""
    if decision.weights_used is not None:
        return decision.weights_used
    cfg = PhaseConfig()
    return WeightsUsed(
        w_improvement=cfg.w_improvement,
        w_info_gain=cfg.w_info_gain,
        w_risk=cfg.w_risk,
        reason="fallback: PhaseConfig defaults (adaptive weights disabled)",
    )


def _per_candidate_delta(
    cand: PooledCandidate,
    archetype: ActionCandidate,
    weights: WeightsUsed,
) -> float:
    """Adjust the archetype utility by this candidate's own diagnostics.

    Every term is guarded: a missing diagnostic contributes 0, so a candidate
    with no diagnostics has delta == 0 and inherits the archetype utility.
    """
    d_imp = (
        cand.expected_improvement - archetype.expected_improvement
        if cand.expected_improvement is not None
        else 0.0
    )
    d_info = (
        cand.info_gain - archetype.expected_info_gain
        if cand.info_gain is not None
        else 0.0
    )
    d_risk = (
        max(0.0, 1.0 - cand.constraint_margin) - archetype.risk
        if cand.constraint_margin is not None
        else 0.0
    )
    return (
        weights.w_improvement * d_imp
        + weights.w_info_gain * d_info
        - weights.w_risk * d_risk
    )


def _normalized_vec(params: dict, space: ParameterSpace) -> list[float]:
    """Numeric parameters mapped to [0, 1] by their bounds (for distance)."""
    vec: list[float] = []
    for dim in space.dimensions:
        if dim.param_type not in ("number", "integer"):
            continue
        value = params.get(dim.param_name)
        if value is None:
            continue
        lo, hi = dim.min_value, dim.max_value
        if lo is None or hi is None or hi <= lo:
            vec.append(float(value))
        else:
            vec.append((float(value) - lo) / (hi - lo))
    return vec


def _distance(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return math.inf
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=False)))


def _redundancy_penalty(
    vec: list[float],
    higher_ranked_vecs: list[list[float]],
    config: ArbitrationConfig,
) -> float:
    """Penalise a candidate that sits very close to a higher-ranked one."""
    if not higher_ranked_vecs:
        return 0.0
    nearest = min(_distance(vec, other) for other in higher_ranked_vecs)
    thr = config.redundancy_distance_threshold
    if nearest >= thr:
        return 0.0
    return round(config.redundancy_weight * (1.0 - nearest / thr), 6)


def score_pool(
    survivors: list[PooledCandidate],
    decision: StrategyDecision,
    space: ParameterSpace,
    config: ArbitrationConfig | None = None,
) -> list[ScoredCandidate]:
    """Rank concrete candidates by delegating to the authority's utilities.

    Returns ``ScoredCandidate``s sorted by final utility (descending).
    """
    config = config or ArbitrationConfig()
    weights = _effective_weights(decision)
    archetypes = {a.name: a for a in decision.actions_considered}

    # 1. base + delta (independent of ranking)
    prelim: list[tuple[PooledCandidate, float, float]] = []  # (cand, base, delta)
    for cand in survivors:
        archetype = archetypes.get(_archetype_name(cand.source_action))
        if archetype is None:
            prelim.append((cand, 0.0, 0.0))
            continue
        delta = _per_candidate_delta(cand, archetype, weights)
        prelim.append((cand, archetype.utility, round(delta, 6)))

    # 2. order by (base + delta), then apply redundancy greedily against
    #    already-ranked (higher) candidates.
    prelim.sort(key=lambda t: t[1] + t[2], reverse=True)

    scored: list[ScoredCandidate] = []
    ranked_vecs: list[list[float]] = []
    for cand, base, delta in prelim:
        vec = _normalized_vec(cand.params, space)
        redundancy = _redundancy_penalty(vec, ranked_vecs, config)
        ranked_vecs.append(vec)
        scored.append(
            ScoredCandidate(
                candidate=cand,
                utility=round(base + delta - redundancy, 6),
                base_utility=base,
                delta=delta,
                redundancy=redundancy,
            )
        )

    scored.sort(key=lambda s: s.utility, reverse=True)
    return scored
