"""Tests for the optimization service entrypoint and provenance logging.

``suggest_next`` is the single call HELIOS makes per round.  It must:
  * use Nexus when available,
  * fall back to the local optimizer when Nexus is unavailable OR raises
    (a Nexus failure must never stop a campaign), and
  * record a full provenance trail of what was proposed, decided, and why.
"""
from __future__ import annotations

import pytest

from app.optimization.nexus_provider import NexusOptimizationProvider
from app.optimization.provenance import ProvenanceLogger
from app.optimization.schemas import OptimizationRequest
from app.optimization.service import OptimizationOutcome, suggest_next
from app.services.candidate_gen import ParameterSpace, SearchDimension


def _request(seed: int = 4) -> OptimizationRequest:
    space = ParameterSpace(
        dimensions=(SearchDimension("x", "number", 0.0, 10.0),),
        protocol_template={},
    )
    return OptimizationRequest(campaign_id="camp", space=space, n=1, seed=seed, round_index=7)


class _UnavailableProvider:
    def is_available(self) -> bool:
        return False

    def suggest(self, request):  # pragma: no cover - must not be called
        raise AssertionError("unavailable provider must not be asked to suggest")


class _RaisingProvider:
    def is_available(self) -> bool:
        return True

    def suggest(self, request):
        raise RuntimeError("nexus exploded")


def test_falls_back_when_provider_unavailable():
    outcome = suggest_next(_request(), provider=_UnavailableProvider())
    assert isinstance(outcome, OptimizationOutcome)
    assert outcome.suggestion.source == "local_fallback"


def test_falls_back_when_provider_raises():
    outcome = suggest_next(_request(), provider=_RaisingProvider())
    assert outcome.suggestion.source == "local_fallback"
    assert outcome.decision is not None  # campaign still progresses


def test_uses_nexus_when_available():
    # Real NexusOptimizationProvider (default); Nexus is installed in CI/dev env.
    provider = NexusOptimizationProvider()
    if not provider.is_available():
        pytest.skip("Nexus optimization core is not installed in this environment")
    outcome = suggest_next(_request())
    assert outcome.suggestion.source == "nexus"


def test_provenance_record_captures_decision_and_source():
    logger = ProvenanceLogger()
    outcome = suggest_next(_request(), provider=_UnavailableProvider(), provenance=logger)
    rec = outcome.provenance
    assert rec["campaign_id"] == "camp"
    assert rec["round_index"] == 7
    assert rec["optimizer_source"] == "local_fallback"
    assert "decision_trace" in rec
    assert "accepted" in rec
    assert len(logger.records) == 1


def test_provenance_sink_is_called():
    captured = []
    logger = ProvenanceLogger(sink=captured.append)
    suggest_next(_request(), provider=_UnavailableProvider(), provenance=logger)
    assert len(captured) == 1
    assert captured[0]["algorithm"] == "built_in"
