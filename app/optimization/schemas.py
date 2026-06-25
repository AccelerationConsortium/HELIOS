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


@dataclass(frozen=True)
class DecisionResult:
    """HELIOS's verdict on a suggestion after applying campaign-level checks."""

    accepted: bool
    final_candidates: tuple[dict[str, Any], ...] = ()
    rejected: tuple[dict[str, Any], ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    requires_human_review: bool = False
    decision_trace: tuple[str, ...] = ()


@runtime_checkable
class OptimizationProvider(Protocol):
    """Interface implemented by Nexus and local optimization providers."""

    def is_available(self) -> bool: ...

    def suggest(self, request: OptimizationRequest) -> CandidateSuggestion: ...
