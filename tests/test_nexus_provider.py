"""Tests for NexusOptimizationProvider.

The provider is HELIOS's facade over Nexus's optimization intelligence:
problem fingerprinting + diagnostics (advisory) and candidate generation
(via the backend bridge).  It must surface Nexus's reasoning without ever
taking execution authority.
"""
from __future__ import annotations

import pytest

pytest.importorskip("optimization_copilot")

from app.optimization.nexus_provider import NexusOptimizationProvider  # noqa: E402
from app.optimization.schemas import CandidateSuggestion, OptimizationRequest  # noqa: E402
from app.services.candidate_gen import ParameterSpace, SearchDimension  # noqa: E402
from app.services.optimization_backends import Observation  # noqa: E402


def _request(n: int = 2, seed: int = 5, backend: str | None = None) -> OptimizationRequest:
    space = ParameterSpace(
        dimensions=(SearchDimension("x", "number", 0.0, 10.0),),
        protocol_template={},
    )
    obs = (
        Observation(params={"x": 1.0}, objective=2.0),
        Observation(params={"x": 5.0}, objective=8.0),
        Observation(params={"x": 9.0}, objective=4.0),
    )
    context = {"backend": backend} if backend else {}
    return OptimizationRequest(
        campaign_id="camp-1", space=space, observations=obs, n=n, seed=seed, context=context
    )


def test_provider_is_available_when_nexus_installed():
    assert NexusOptimizationProvider().is_available() is True


def test_suggest_returns_nexus_sourced_candidates_in_bounds():
    sug = NexusOptimizationProvider().suggest(_request(n=2))
    assert isinstance(sug, CandidateSuggestion)
    assert sug.source == "nexus"
    assert sug.algorithm == "nexus_gp_bo"  # default backend
    assert len(sug.candidates) == 2
    for c in sug.candidates:
        assert 0.0 <= c["x"] <= 10.0


def test_suggest_attaches_fingerprint_and_diagnostics():
    sug = NexusOptimizationProvider().suggest(_request())
    assert "noise_regime" in sug.fingerprint
    assert "data_scale" in sug.fingerprint
    assert "exploration_coverage" in sug.diagnostics


def test_suggest_honours_requested_backend():
    sug = NexusOptimizationProvider().suggest(_request(backend="nexus_random"))
    assert sug.algorithm == "nexus_random"


def test_suggest_is_deterministic_with_seed():
    a = NexusOptimizationProvider().suggest(_request(seed=13))
    b = NexusOptimizationProvider().suggest(_request(seed=13))
    assert a.candidates == b.candidates


def test_diagnose_returns_fingerprint_and_diagnostics_dicts():
    fp, diag = NexusOptimizationProvider().diagnose(_request())
    assert isinstance(fp, dict) and "noise_regime" in fp
    assert isinstance(diag, dict) and "exploration_coverage" in diag
