"""P3a -- live method advisor: problem structure -> recommended backends.

The offline method-comparison benchmark (``benchmarks/methods/``) derives a
generalizable *decision table* mapping a problem's structural tags (dims,
modality, noise) to the methods that perform best.  This module is the
production-side consumer of that table: it derives the same coarse profile from
a live ``CampaignSnapshot`` + ``DiagnosticSignals`` and returns a ranked list of
backends, which ``select_strategy`` feeds into ``rank_backends``'s recommendation
channel (a soft boost -- it can flip near-ties, never override phase policy or
pull in an out-of-pool/unavailable backend).

Dependency direction is deliberate: production code does **not** import the
benchmark harness.  ``DEFAULT_DECISION_TABLE`` is an expert prior consistent with
the benchmark's structure; a benchmark run can later regenerate it offline and
update this table.  Bucketing thresholds mirror ``benchmarks/methods/recommend``
so the offline table and the live lookup agree.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from app.services.strategy_models import CampaignSnapshot, DiagnosticSignals

# Bucket thresholds (mirror benchmarks/methods/recommend._dims_class/_noise_class).
_DIMS_LOW_MAX = 3
_SMOOTHNESS_MULTIMODAL_MAX = 0.25  # local_smoothness below this => rugged/multimodal
_NOISE_HIGH_MIN = 0.15  # noise_ratio at/above this => noisy regime

# (dims, modality, noise) -> ranked backends, best first.  Names may exceed the
# action pool; rank_backends only ever boosts pool members, so extras are inert.
DEFAULT_DECISION_TABLE: dict[tuple[str, str, str], tuple[str, ...]] = {
    # Smooth, low-dim: GP-BO is in its element.
    ("low", "unimodal", "low"): ("bomcp", "built_in", "optuna_tpe", "pymoo_nsga2"),
    # Rugged low-dim: global/evolutionary search beats local GP exploitation.
    ("low", "multimodal", "low"): ("optuna_cmaes", "scipy_de", "bomcp", "pymoo_nsga2"),
    # High-dim smooth: scalable model-based / TPE; NSGA2 as a global option.
    ("high", "unimodal", "low"): ("bomcp", "optuna_tpe", "pymoo_nsga2", "built_in"),
    # High-dim rugged: evolutionary methods scale where GP struggles.
    ("high", "multimodal", "low"): ("pymoo_nsga2", "optuna_cmaes", "optuna_tpe", "bomcp"),
    # Noisy regimes: prefer methods that tolerate/model noise (bomcp models GP
    # noise; TPE/CMA-ES are robust) over a noise-naive surrogate.
    ("low", "unimodal", "high"): ("bomcp", "optuna_tpe", "optuna_cmaes", "built_in"),
    ("low", "multimodal", "high"): ("optuna_cmaes", "scipy_de", "optuna_tpe", "bomcp"),
    ("high", "unimodal", "high"): ("optuna_tpe", "bomcp", "pymoo_nsga2", "optuna_cmaes"),
    ("high", "multimodal", "high"): ("pymoo_nsga2", "optuna_cmaes", "optuna_tpe", "bomcp"),
}

_FALLBACK_RANKING: tuple[str, ...] = ("bomcp", "optuna_tpe", "built_in")


def problem_profile(
    snapshot: CampaignSnapshot,
    diag: DiagnosticSignals | None = None,
) -> tuple[str, str, str]:
    """Derive the coarse ``(dims, modality, noise)`` bucket for the campaign."""
    dims = "low" if snapshot.n_dimensions <= _DIMS_LOW_MAX else "high"

    modality = "unimodal"
    if diag is not None and diag.local_smoothness is not None:
        if diag.local_smoothness < _SMOOTHNESS_MULTIMODAL_MAX:
            modality = "multimodal"

    noise = "low"
    if diag is not None and diag.noise_ratio is not None:
        if diag.noise_ratio >= _NOISE_HIGH_MIN:
            noise = "high"

    return (dims, modality, noise)


def recommend_backends(
    snapshot: CampaignSnapshot,
    diag: DiagnosticSignals | None = None,
) -> tuple[str, ...]:
    """Return backends ranked best-first for the campaign's problem class.

    Categorical/mixed spaces promote TPE (strong on categorical encodings) to the
    front, since the (dims, modality, noise) bucket does not capture variable type.
    """
    profile = problem_profile(snapshot, diag)
    ranking = DEFAULT_DECISION_TABLE.get(profile, _FALLBACK_RANKING)

    if snapshot.has_categorical:
        ranking = ("optuna_tpe", *[b for b in ranking if b != "optuna_tpe"])

    return ranking
