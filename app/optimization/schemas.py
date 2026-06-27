"""Stable HELIOS-side schemas for the optimization-intelligence layer.

These dataclasses are the *contract* between HELIOS and any optimization
provider (Nexus or local).  They deliberately reference only HELIOS's own
types (``ParameterSpace``, ``Observation``) so Nexus's internal objects never
leak across the boundary -- the two codebases stay decoupled and the provider
implementation can change without rippling through HELIOS.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from app.services.candidate_gen import ParameterSpace
from app.services.optimization_backends import Observation


@dataclass(frozen=True)
class OptimizationRequest:
    """Everything a provider needs to propose the next experiment(s)."""

    campaign_id: str
    space: ParameterSpace
    observations: tuple[Observation, ...] = ()
    objective_name: str = "objective"
    direction: str = "maximize"  # informational; HELIOS pre-flips to higher=better
    n: int = 1
    seed: int = 42
    round_index: int = 0
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateSuggestion:
    """A provider's answer: candidate(s) plus the reasoning behind them."""

    candidates: tuple[dict[str, Any], ...]
    algorithm: str
    source: str  # "nexus" | "local_fallback"
    confidence: float = 0.5
    rationale: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)
    fingerprint: dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    # Δ1: per-candidate diagnostics, aligned 1:1 with ``candidates`` when present.
    # Empty in Nexus suggest_top_k Phase 1 (shared diagnostics only).
    per_candidate: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class PooledCandidate:
    """A concrete candidate in the arbitration pool, tagged with its origin.

    ``source_action`` is the campaign **archetype** the candidate is scored
    against -- decided by HELIOS authority, *not* by the generating backend.
    ``generator_backend`` records *how* it was produced (audit only); it never
    overrides the archetype.  Per-candidate diagnostics are all optional; when
    absent, the candidate inherits its archetype's utility verbatim (delta 0).
    """

    params: dict[str, Any]
    source: str  # "nexus" | "local" | "replicate" | "recovery" | "sobol"
    source_action: str  # "explore" | "exploit" | "refine" | "stabilize"
    generator_backend: str = ""  # audit: which backend produced it
    expected_improvement: float | None = None
    uncertainty: float | None = None
    novelty: float | None = None
    constraint_margin: float | None = None
    info_gain: float | None = None
    rationale: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredCandidate:
    """A pooled candidate plus its decomposed utility (for audit + selection)."""

    candidate: PooledCandidate
    utility: float
    base_utility: float  # inherited from the archetype's authority utility
    delta: float  # per-candidate adjustment under the same weight vector
    redundancy: float  # diversity penalty vs higher-ranked candidates


@dataclass(frozen=True)
class CandidatePool:
    """The concrete portfolio assembled by the boundary layer for one round."""

    candidates: tuple[PooledCandidate, ...]
    sources_used: tuple[str, ...] = ()
    sources_dropped: tuple[str, ...] = ()
    construction_trace: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecisionResult:
    """HELIOS's verdict on a suggestion after applying campaign-level checks."""

    accepted: bool
    final_candidates: tuple[dict[str, Any], ...] = ()
    rejected: tuple[dict[str, Any], ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    requires_human_review: bool = False
    decision_trace: tuple[str, ...] = ()
    # Δ1: full scored portfolio behind the verdict (for provenance + audit).
    scored_pool: tuple[ScoredCandidate, ...] = ()


@runtime_checkable
class OptimizationProvider(Protocol):
    """Interface implemented by Nexus and local optimization providers."""

    def is_available(self) -> bool: ...

    def suggest(self, request: OptimizationRequest) -> CandidateSuggestion: ...
