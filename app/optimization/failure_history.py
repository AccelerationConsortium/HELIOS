"""Per-backend recent-failure accumulation (Δ2 follow-up).

Each round's execution outcome is attributed to the backend that produced it.
The counter feeds ``rank_backends`` (penalty/veto), so it must capture *recent*
reliability: failures push a backend toward the veto threshold, and a
subsequent success heals one step.  This prevents a transient bad round from
permanently blacklisting an otherwise-good backend.

Pure and side-effect-free; callers thread the returned dict forward and persist
it as part of campaign state.
"""
from __future__ import annotations


def update_backend_failures(
    counts: dict[str, int],
    backend: str | None,
    *,
    round_had_failure: bool,
) -> dict[str, int]:
    """Return updated per-backend failure counts after one round.

    * ``round_had_failure`` True  -> increment ``backend``'s count.
    * ``round_had_failure`` False -> decay ``backend``'s count by one (healing),
      dropping the entry once it reaches zero.
    * ``backend`` None            -> no-op (no backend was selected this round).

    The input mapping is never mutated.
    """
    new = dict(counts)
    if not backend:
        return new

    if round_had_failure:
        new[backend] = new.get(backend, 0) + 1
    elif new.get(backend, 0) > 0:
        new[backend] -= 1
        if new[backend] <= 0:
            del new[backend]
    return new
