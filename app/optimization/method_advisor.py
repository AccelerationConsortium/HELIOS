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

import json
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from app.services.strategy_models import CampaignSnapshot, DiagnosticSignals

logger = logging.getLogger(__name__)

# A benchmark run (benchmarks/methods) can emit a generated decision table here;
# when present it overrides DEFAULT_DECISION_TABLE per bucket (gaps fall back to
# the default).  Overridable via the HELIOS_DECISION_TABLE env var.
DECISION_TABLE_PATH = os.environ.get(
    "HELIOS_DECISION_TABLE",
    os.path.join(os.path.dirname(__file__), "method_decision_table.json"),
)

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


def load_decision_table(
    path: str | None = None,
) -> dict[tuple[str, str, str], tuple[str, ...]]:
    """Return the decision table: DEFAULT overlaid with a generated artifact.

    The artifact (if present and readable) is a JSON list of
    ``{"dims", "modality", "noise", "methods": [...]}`` entries; each overrides
    its bucket.  Missing buckets keep the expert-prior default.  Any read/parse
    error degrades silently to the default -- the live path never depends on the
    benchmark artifact existing.
    """
    table = dict(DEFAULT_DECISION_TABLE)
    artifact_path = path if path is not None else DECISION_TABLE_PATH
    try:
        with open(artifact_path) as fh:
            entries = json.load(fh)
        for e in entries:
            key = (e["dims"], e["modality"], e["noise"])
            methods = tuple(e["methods"])
            if methods:
                table[key] = methods
    except FileNotFoundError:
        pass
    except Exception:
        logger.debug("Decision-table artifact unreadable; using default", exc_info=True)
    return table


def recommend_backends(
    snapshot: CampaignSnapshot,
    diag: DiagnosticSignals | None = None,
) -> tuple[str, ...]:
    """Return backends ranked best-first for the campaign's problem class.

    Categorical/mixed spaces promote TPE (strong on categorical encodings) to the
    front, since the (dims, modality, noise) bucket does not capture variable type.
    """
    profile = problem_profile(snapshot, diag)
    ranking = load_decision_table().get(profile, _FALLBACK_RANKING)

    if snapshot.has_categorical:
        ranking = ("optuna_tpe", *[b for b in ranking if b != "optuna_tpe"])

    return ranking
