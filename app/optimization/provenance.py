"""Provenance logging for optimization decisions.

Every round produces a complete, auditable record of *what was proposed, what
was decided, and why* -- the evidence that HELIOS is an auditable, recoverable,
explainable scientific agent loop (not merely a connector around an optimizer).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from app.optimization.schemas import (
    CandidateSuggestion,
    DecisionResult,
    OptimizationRequest,
)

if TYPE_CHECKING:  # pragma: no cover
    from app.services.strategy_models import StrategyDecision

ProvenanceSink = Callable[[dict[str, Any]], None]


def _serialize_backend_selection(
    strategy_decision: StrategyDecision | None,
) -> dict[str, Any]:
    """Serialize the backend-selection trace from the authority decision."""
    if strategy_decision is None:
        return {}
    return {
        "chosen_backend": strategy_decision.backend_name,
        "phase": strategy_decision.phase,
        "recommended_backends": list(strategy_decision.recommended_backends),
        "confidence": strategy_decision.confidence,
    }


def _serialize_scored_pool(
    decision: DecisionResult,
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Serialize the scored arbitration portfolio, when present."""
    pool: list[dict[str, Any]] = []
    for scored in decision.scored_pool:
        params = dict(scored.candidate.params)
        pool.append(
            {
                "params": params,
                "source": scored.candidate.source,
                "source_action": scored.candidate.source_action,
                "generator_backend": scored.candidate.generator_backend,
                "base_utility": scored.base_utility,
                "delta": scored.delta,
                "redundancy": scored.redundancy,
                "utility": scored.utility,
                "selected": params in selected,
            }
        )
    return pool


class ProvenanceLogger:
    """Build and retain provenance records; optionally forward to a sink."""

    def __init__(self, sink: ProvenanceSink | None = None) -> None:
        self.records: list[dict[str, Any]] = []
        self._sink = sink

    def build(
        self,
        request: OptimizationRequest,
        suggestion: CandidateSuggestion,
        decision: DecisionResult,
        *,
        evidence: dict[str, Any] | None = None,
        strategy_decision: StrategyDecision | None = None,
    ) -> dict[str, Any]:
        selected = [dict(c) for c in decision.final_candidates]
        rec: dict[str, Any] = {
            "campaign_id": request.campaign_id,
            "round_index": request.round_index,
            "seed": request.seed,
            "optimizer_source": suggestion.source,
            "algorithm": suggestion.algorithm,
            "confidence": suggestion.confidence,
            "rationale": suggestion.rationale,
            "problem_fingerprint": suggestion.fingerprint,
            "diagnostics": suggestion.diagnostics,
            "candidates_proposed": [dict(c) for c in suggestion.candidates],
            "candidates_accepted": selected,
            "candidates_rejected": [dict(c) for c in decision.rejected],
            "rejection_reasons": list(decision.rejection_reasons),
            "accepted": decision.accepted,
            "requires_human_review": decision.requires_human_review,
            "decision_trace": list(decision.decision_trace),
            "backend_selection": _serialize_backend_selection(strategy_decision),
            "candidate_pool": _serialize_scored_pool(decision, selected),
        }
        # Evidence is additive: attached only when memory recall produced
        # something, so the record shape is unchanged when there is no history.
        if evidence is not None:
            rec["evidence"] = evidence
        return rec

    def record(
        self,
        request: OptimizationRequest,
        suggestion: CandidateSuggestion,
        decision: DecisionResult,
        *,
        evidence: dict[str, Any] | None = None,
        strategy_decision: StrategyDecision | None = None,
    ) -> dict[str, Any]:
        rec = self.build(
            request,
            suggestion,
            decision,
            evidence=evidence,
            strategy_decision=strategy_decision,
        )
        self.records.append(rec)
        if self._sink is not None:
            self._sink(rec)
        return rec
