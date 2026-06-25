"""Tests for the local fallback optimization provider.

When Nexus is unavailable, HELIOS must still make progress.  The local
fallback wraps HELIOS's always-available ``built_in`` backend behind the
``OptimizationProvider`` interface.
"""
from __future__ import annotations

from app.optimization.local_fallback import LocalFallbackProvider
from app.optimization.schemas import CandidateSuggestion, OptimizationRequest
from app.services.candidate_gen import ParameterSpace, SearchDimension


def _request(n: int = 2, seed: int = 3) -> OptimizationRequest:
    space = ParameterSpace(
        dimensions=(SearchDimension("x", "number", 0.0, 10.0),),
        protocol_template={},
    )
    return OptimizationRequest(campaign_id="c1", space=space, n=n, seed=seed)


def test_local_fallback_is_always_available():
    assert LocalFallbackProvider().is_available() is True


def test_local_fallback_returns_n_in_bounds_candidates():
    sug = LocalFallbackProvider().suggest(_request(n=2))
    assert isinstance(sug, CandidateSuggestion)
    assert sug.source == "local_fallback"
    assert len(sug.candidates) == 2
    for c in sug.candidates:
        assert 0.0 <= c["x"] <= 10.0


def test_local_fallback_is_deterministic_with_seed():
    a = LocalFallbackProvider().suggest(_request(seed=11))
    b = LocalFallbackProvider().suggest(_request(seed=11))
    assert a.candidates == b.candidates
