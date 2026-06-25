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
from typing import Any

from app.optimization.schemas import (
    CandidateSuggestion,
    DecisionResult,
    OptimizationRequest,
)
from app.services.candidate_gen import ParameterSpace

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
            violation = _bounds_violation(cand, request.space)
            if violation is not None:
                rejected.append(cand)
                reasons.append(f"out of bounds: {violation}")
                trace.append(f"rejected {cand}: {violation}")
                continue

            sig = _signature(cand)
            if sig in seen:
                rejected.append(cand)
                reasons.append("duplicate of a prior or already-accepted point")
                trace.append(f"rejected {cand}: duplicate")
                continue

            if self._safety_check is not None and not self._safety_check(cand, request):
                rejected.append(cand)
                reasons.append("failed safety check")
                trace.append(f"rejected {cand}: failed safety check")
                continue

            seen.add(sig)
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
