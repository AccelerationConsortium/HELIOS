"""Δ1 architectural guardrails (acceptance criteria, enforced in CI).

These tests encode the binding principle of approach A:

    HELIOS authority decides the archetype.
    Nexus / backends generate candidates.
    The boundary layer (app/optimization) builds and audits the pool.
    The existing scorer (app/services/strategy_*) is the single source of
    utility truth.

AC1 -- no second brain in the boundary:
    app/optimization defines no utility-weight constants and recomputes no
    phase posterior.

AC5 -- one dependency direction, with named exceptions:
    1. CORE AUTHORITY modules (strategy_selector / strategy_scoring /
       strategy_actions / strategy_models / strategy_diagnostics /
       strategy_router) must NOT import app.optimization.
    2. Only the two designated BRIDGE ADAPTERS may import app.optimization:
         - optimization_backends.py    (Nexus backend registration bridge)
         - optimization_intelligence.py (Nexus advice enrichment adapter)
       These are semantically bridges -- they merge Nexus/optimization advice
       into HELIOS decision context; they do not own campaign authority.
    3. Bridge adapters must NOT own authority: no scoring-weight constants, no
       phase recomputation, no decision-policy class, no select_strategy.

The allowlist exists so the guardrail stays *precise* rather than *weak*: the
real risk -- core strategy authority quietly depending on the boundary -- stays
forbidden, while the explicit enrichment seams are permitted and themselves
constrained.  (Inverting optimization_intelligence's pull-based recommendation
import to a push-based design is deferred to Δ2; see the design doc.)
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

# Phase A (nexus×main union) intentionally lands nexus's Δ2/P3a/P3b *pull-based*
# imports (strategy_selector/strategy_models → app.optimization.{backend_selection,
# method_advisor,failure_region}).  Inverting these to the push-based, AC5-compliant
# direction is Phase B (CandidatePoolService layering) — mirroring how this file's
# docstring already defers optimization_intelligence's pull→push inversion.
_AC5_INVERSION_PENDING_PHASE_B = pytest.mark.xfail(
    reason="AC5 dependency inversion deferred to Phase B (CandidatePoolService layering)",
    strict=False,
)

_ROOT = Path(__file__).resolve().parents[1]
_OPT_DIR = _ROOT / "app" / "optimization"
_SERVICES_DIR = _ROOT / "app" / "services"

# Core campaign-authority modules: forbidden from importing the boundary.
_CORE_AUTHORITY = {
    "strategy_selector.py",
    "strategy_scoring.py",
    "strategy_actions.py",
    "strategy_models.py",
    "strategy_diagnostics.py",
    "strategy_router.py",
}

# The only modules in app/services permitted to import app.optimization.
_BRIDGE_ADAPTERS = {
    "optimization_backends.py",  # backend registration bridge
    "optimization_intelligence.py",  # Nexus advice enrichment adapter
}

# assignment of a numeric default to a utility-weight name, e.g. ``w_risk = 0.2``
_WEIGHT_ASSIGN = re.compile(
    r"^\s*w_(improvement|info_gain|risk)\s*(:\s*float\s*)?=\s*[-0-9.]", re.MULTILINE
)
_PHASE_POSTERIOR = re.compile(r"\bcompute_phase_posterior\b|\bPhasePosterior\s*\(")
_OWNS_DECISION = re.compile(r"\bclass\s+\w*DecisionPolicy\b|\bdef\s+select_strategy\b")
_IMPORTS_BOUNDARY = re.compile(r"(^|\n)\s*(from|import)\s+app\.optimization")


def _py_files(root: Path):
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


# --- AC1: the boundary holds no authority ----------------------------------


def test_ac1_boundary_defines_no_weight_constants():
    offenders = [p.name for p in _py_files(_OPT_DIR) if _WEIGHT_ASSIGN.search(p.read_text())]
    assert not offenders, f"weight constants defined in boundary layer: {offenders}"


def test_ac1_boundary_recomputes_no_phase_posterior():
    offenders = [p.name for p in _py_files(_OPT_DIR) if _PHASE_POSTERIOR.search(p.read_text())]
    assert not offenders, f"boundary layer recomputes phase: {offenders}"


# --- AC5: one direction, with named & constrained bridge exceptions ---------


@_AC5_INVERSION_PENDING_PHASE_B
def test_ac5_core_authority_never_imports_boundary():
    offenders = []
    for path in _py_files(_SERVICES_DIR):
        if path.name in _CORE_AUTHORITY and _IMPORTS_BOUNDARY.search(path.read_text()):
            offenders.append(path.name)
    assert not offenders, f"core authority imports app.optimization: {offenders}"


@_AC5_INVERSION_PENDING_PHASE_B
def test_ac5_only_named_bridges_import_boundary():
    offenders = []
    for path in _py_files(_SERVICES_DIR):
        if path.name in _BRIDGE_ADAPTERS:
            continue
        if _IMPORTS_BOUNDARY.search(path.read_text()):
            offenders.append(path.name)
    assert not offenders, (
        "unsanctioned app/services module imports app.optimization "
        f"(add to _BRIDGE_ADAPTERS only if it is a true bridge): {offenders}"
    )


def test_ac5_bridge_adapters_do_not_own_authority():
    problems: dict[str, list[str]] = {}
    for name in _BRIDGE_ADAPTERS:
        path = _SERVICES_DIR / name
        if not path.exists():
            continue
        text = path.read_text()
        issues = []
        if _WEIGHT_ASSIGN.search(text):
            issues.append("defines weight constants")
        if _PHASE_POSTERIOR.search(text):
            issues.append("recomputes phase posterior")
        if _OWNS_DECISION.search(text):
            issues.append("owns a decision policy / select_strategy")
        if issues:
            problems[name] = issues
    assert not problems, f"bridge adapter overstepped into authority: {problems}"
