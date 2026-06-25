"""Conservative, fingerprint-biased backend ranking (staged Δ2).

``rank_backends`` chooses one backend from an ordered preference *pool* for the
winning campaign action.  It is deliberately conservative:

* **phase policy dominates** -- the pool's preference order is the base score;
* **fingerprint recommendation is a secondary boost** -- it can flip near-ties
  but cannot overturn a clearly-preferred phase backend;
* **availability and phase-incompatibility are hard vetoes** -- only backends in
  the pool that are available can ever be selected, so a recommendation can
  never pull in an unavailable or phase-incompatible backend;
* **recent failure history penalizes and (past a threshold) vetoes**;
* **selection is deterministic** -- ties break by preference order.

With no recommendation and no failures it reduces exactly to "first available
in preference order", so it is a no-op for the existing decision path.

The function is pure (no Nexus import, no I/O) and returns a ``BackendSelection``
that doubles as the provenance record for the choice.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class BackendScore:
    """Score breakdown for a single candidate backend."""

    backend: str
    phase_score: float
    fingerprint_boost: float
    failure_penalty: float
    total: float


@dataclass(frozen=True)
class BackendSelection:
    """The ratified backend choice plus full selection provenance."""

    phase: str
    candidate_backends: tuple[str, ...]  # available pool members considered, in order
    fingerprint_recommendation: tuple[str, ...]
    score_components: tuple[BackendScore, ...]
    selected_backend: str
    reason: str
    fallback: bool


def rank_backends(
    phase: str,
    pool: tuple[str, ...],
    available: Mapping[str, bool],
    *,
    recommended: tuple[str, ...] = (),
    failure_counts: Mapping[str, int] | None = None,
    phase_weight: float = 1.0,
    fingerprint_weight: float = 0.3,
    failure_penalty: float = 0.5,
    failure_veto_threshold: int = 3,
    fallback_backend: str = "built_in",
) -> BackendSelection:
    """Rank *pool* backends and return the ratified selection with provenance."""
    failures = failure_counts or {}
    n = len(pool)

    # Hard veto 1: availability + phase-incompatibility (only pool members count).
    avail = [b for b in pool if available.get(b, False)]
    # Hard veto 2: failure history at/above the veto threshold.
    considered = [b for b in avail if failures.get(b, 0) < failure_veto_threshold]
    scoring_pool = considered if considered else avail

    if not scoring_pool:
        # Nothing in the pool is usable -> degrade to the guaranteed fallback.
        return BackendSelection(
            phase=phase,
            candidate_backends=(),
            fingerprint_recommendation=tuple(recommended),
            score_components=(),
            selected_backend=fallback_backend,
            reason=f"no pool backend available; degraded to {fallback_backend}",
            fallback=True,
        )

    rec_index = {b: i for i, b in enumerate(recommended)}
    scores: list[BackendScore] = []
    for backend in scoring_pool:
        idx = pool.index(backend)
        phase_score = (n - idx) / n  # top preference -> highest

        if recommended and backend in rec_index:
            rank = rec_index[backend]
            boost = fingerprint_weight * ((len(recommended) - rank) / len(recommended))
        else:
            boost = 0.0

        fcount = failures.get(backend, 0)
        penalty = (
            failure_penalty * min(1.0, fcount / failure_veto_threshold)
            if failure_veto_threshold
            else 0.0
        )

        total = phase_weight * phase_score + boost - penalty
        scores.append(
            BackendScore(
                backend=backend,
                phase_score=round(phase_score, 4),
                fingerprint_boost=round(boost, 4),
                failure_penalty=round(penalty, 4),
                total=round(total, 4),
            )
        )

    # Deterministic: highest total, ties broken by preference order.
    ranked = sorted(scores, key=lambda s: (-s.total, pool.index(s.backend)))
    selected = ranked[0].backend

    phase_only = scoring_pool[0]  # what phase policy alone would pick
    biased = selected != phase_only
    # A choice of the fallback backend is only a "fallback" when it was not the
    # intended top preference (e.g. richer options were unavailable).
    fallback = selected == fallback_backend and pool[0] != fallback_backend

    if biased:
        reason = f"fingerprint promoted '{selected}' over phase-default '{phase_only}'"
    elif fallback:
        reason = f"degraded to '{fallback_backend}' (no richer backend viable)"
    else:
        reason = f"phase policy selected '{selected}'"

    return BackendSelection(
        phase=phase,
        candidate_backends=tuple(scoring_pool),
        fingerprint_recommendation=tuple(recommended),
        score_components=tuple(scores),
        selected_backend=selected,
        reason=reason,
        fallback=fallback,
    )
