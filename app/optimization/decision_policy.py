"""HELIOS's decision authority over optimization suggestions.

A provider (Nexus or local) *proposes*; ``OptimizationDecisionPolicy``
*disposes*.  It sequences campaign-level validation into one auditable
verdict:

1. search-space bounds (numeric ranges, valid categories)
2. deduplication (against history and within the batch)
3. an optional safety hook (delegated, not reimplemented)

Each candidate that survives is accepted; the rest are rejected with a
human-readable reason.  When nothing is executable, the result is flagged for
human review rather than silently returning an empty batch.

This class orchestrates checks; it deliberately does NOT reimplement the
safety agent, recovery agent, or TaskContract validation -- those are injected
or applied elsewhere in the campaign loop.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from app.optimization.arbitration_config import ArbitrationConfig
from app.optimization.candidate_scorer import score_pool
from app.optimization.schemas import (
    CandidatePool,
    CandidateSuggestion,
    DecisionResult,
    OptimizationRequest,
)
from app.services.candidate_gen import ParameterSpace

if TYPE_CHECKING:  # pragma: no cover
    from app.services.strategy_models import StrategyDecision

SafetyCheck = Callable[[dict[str, Any], OptimizationRequest], bool]

_NUMERIC_TYPES = ("number", "integer")


def _signature(candidate: dict[str, Any], ndigits: int = 9) -> tuple:
    """Order-independent, tolerance-aware identity for deduplication."""
    items = []
    for key in sorted(candidate):
        value = candidate[key]
        if isinstance(value, float):
            value = round(value, ndigits)
        items.append((key, value))
    return tuple(items)


def _bounds_violation(candidate: dict[str, Any], space: ParameterSpace) -> str | None:
    """Return a human-readable reason if *candidate* is out of the space, else None."""
    for dim in space.dimensions:
        if dim.param_name not in candidate:
            return f"missing parameter '{dim.param_name}'"
        value = candidate[dim.param_name]
        if dim.param_type in ("categorical", "boolean"):
            if dim.choices is not None and value not in dim.choices:
                return f"invalid category for '{dim.param_name}': {value!r}"
        elif dim.param_type in _NUMERIC_TYPES:
            if dim.min_value is not None and value < dim.min_value:
                return f"'{dim.param_name}' below bounds ({value} < {dim.min_value})"
            if dim.max_value is not None and value > dim.max_value:
                return f"'{dim.param_name}' above bounds ({value} > {dim.max_value})"
    return None


class OptimizationDecisionPolicy:
    """Validate a suggestion's candidates against campaign-level rules."""

    def __init__(self, *, safety_check: SafetyCheck | None = None) -> None:
        self._safety_check = safety_check

    # -- shared hard gate -------------------------------------------------

    def _gate(
        self,
        params: dict[str, Any],
        seen: set[tuple],
        request: OptimizationRequest,
    ) -> str | None:
        """Return a rejection reason if *params* fail the hard gate, else None.

        On acceptance, records the signature in *seen* (within-batch dedup).
        The gate is absolute and strategy-free: bounds, dedup, safety.
        """
        violation = _bounds_violation(params, request.space)
        if violation is not None:
            return f"out of bounds: {violation}"
        sig = _signature(params)
        if sig in seen:
            return "duplicate of a prior or already-accepted point"
        if self._safety_check is not None and not self._safety_check(params, request):
            return "failed safety check"
        seen.add(sig)
        return None

    # -- legacy hard-gate-only verdict (unchanged behaviour) --------------

    def evaluate(
        self,
        suggestion: CandidateSuggestion,
        request: OptimizationRequest,
    ) -> DecisionResult:
        trace: list[str] = []
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        reasons: list[str] = []

        seen: set[tuple] = {_signature(o.params) for o in request.observations}
        trace.append(
            f"evaluating {len(suggestion.candidates)} candidate(s) from "
            f"{suggestion.source}:{suggestion.algorithm}"
        )

        for cand in suggestion.candidates:
            reason = self._gate(cand, seen, request)
            if reason is not None:
                rejected.append(cand)
                reasons.append(reason)
                trace.append(f"rejected {cand}: {reason}")
                continue
            accepted.append(cand)
            trace.append(f"accepted {cand}")

        requires_human_review = len(accepted) == 0
        if requires_human_review:
            trace.append("no executable candidate -> escalating for human review")

        return DecisionResult(
            accepted=bool(accepted),
            final_candidates=tuple(accepted),
            rejected=tuple(rejected),
            rejection_reasons=tuple(reasons),
            requires_human_review=requires_human_review,
            decision_trace=tuple(trace),
        )

    # -- two-stage verdict over a concrete candidate pool (Δ1) ------------

    def arbitrate(
        self,
        pool: CandidatePool,
        request: OptimizationRequest,
        decision: StrategyDecision,
        *,
        config: ArbitrationConfig | None = None,
    ) -> DecisionResult:
        """Hard-gate the pool, then soft-rank survivors via the authority's utilities.

        Stage 2 delegates entirely to ``score_pool`` -- no weights or phase are
        computed here.  Returns the top-``request.n`` candidates plus the full
        scored portfolio for audit.
        """
        config = config or ArbitrationConfig()
        trace: list[str] = [
            f"arbitrating pool of {len(pool.candidates)} candidate(s); "
            f"sources_used={list(pool.sources_used)} dropped={list(pool.sources_dropped)}"
        ]
        rejected: list[dict[str, Any]] = []
        reasons: list[str] = []
        survivors = []

        seen: set[tuple] = {_signature(o.params) for o in request.observations}
        for cand in pool.candidates:
            reason = self._gate(cand.params, seen, request)
            if reason is not None:
                rejected.append(dict(cand.params))
                reasons.append(reason)
                trace.append(f"rejected {cand.params} [{cand.source}]: {reason}")
                continue
            survivors.append(cand)

        scored = score_pool(survivors, decision, request.space, config)
        for s in scored:
            trace.append(
                f"scored {s.candidate.params} "
                f"[{s.candidate.source}/{s.candidate.source_action}]: "
                f"util={s.utility} (base={s.base_utility}, delta={s.delta}, "
                f"redun={s.redundancy})"
            )

        final = tuple(dict(s.candidate.params) for s in scored[: request.n])
        requires_human_review = len(scored) == 0
        if requires_human_review:
            trace.append("no executable candidate -> escalating for human review")

        return DecisionResult(
            accepted=bool(final),
            final_candidates=final,
            rejected=tuple(rejected),
            rejection_reasons=tuple(reasons),
            requires_human_review=requires_human_review,
            decision_trace=tuple(trace),
            scored_pool=tuple(scored),
        )
