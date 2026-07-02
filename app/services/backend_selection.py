"""Conservative, fingerprint-biased backend ranking (staged delta 2)."""
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
    influence_delta: float = 0.0


@dataclass(frozen=True)
class BackendSelection:
    """The ratified backend choice plus full selection provenance."""

    phase: str
    candidate_backends: tuple[str, ...]
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
    influence_deltas: Mapping[str, float] | None = None,
) -> BackendSelection:
    """Rank pool backends and return the ratified selection with provenance."""
    failures = failure_counts or {}
    external_deltas = influence_deltas or {}
    n = len(pool)

    avail = [b for b in pool if available.get(b, False)]
    considered = [b for b in avail if failures.get(b, 0) < failure_veto_threshold]
    scoring_pool = considered if considered else avail

    if not scoring_pool:
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
        phase_score = (n - idx) / n

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

        influence_delta = external_deltas.get(backend, 0.0)
        total = phase_weight * phase_score + boost - penalty + influence_delta
        scores.append(
            BackendScore(
                backend=backend,
                phase_score=round(phase_score, 4),
                fingerprint_boost=round(boost, 4),
                failure_penalty=round(penalty, 4),
                influence_delta=round(influence_delta, 4),
                total=round(total, 4),
            )
        )

    ranked = sorted(scores, key=lambda s: (-s.total, pool.index(s.backend)))
    selected = ranked[0].backend

    phase_only = scoring_pool[0]
    biased = selected != phase_only
    fallback = selected == fallback_backend

    if biased and any(abs(v) > 0 for v in external_deltas.values()):
        reason = f"ranking influence promoted '{selected}' over phase-default '{phase_only}'"
    elif biased:
        reason = f"fingerprint promoted '{selected}' over phase-default '{phase_only}'"
    elif fallback:
        reason = f"using fallback backend '{fallback_backend}'"
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
