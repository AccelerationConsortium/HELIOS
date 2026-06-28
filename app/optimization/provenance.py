"""Provenance logging for optimization decisions.

Every round produces a complete, auditable record of *what was proposed, what
was decided, and why* -- the evidence that HELIOS is an auditable, recoverable,
explainable scientific agent loop (not merely a connector around an optimizer).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.optimization.schemas import (
    CandidateSuggestion,
    DecisionResult,
    OptimizationRequest,
)

ProvenanceSink = Callable[[dict[str, Any]], None]


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
        evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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
            "candidates_accepted": [dict(c) for c in decision.final_candidates],
            "candidates_rejected": [dict(c) for c in decision.rejected],
            "rejection_reasons": list(decision.rejection_reasons),
            "accepted": decision.accepted,
            "requires_human_review": decision.requires_human_review,
            "decision_trace": list(decision.decision_trace),
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
        evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        rec = self.build(request, suggestion, decision, evidence=evidence)
        self.records.append(rec)
        if self._sink is not None:
            self._sink(rec)
        return rec
