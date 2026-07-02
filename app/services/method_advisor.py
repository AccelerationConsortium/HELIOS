"""Problem-structure method advisor for live strategy selection."""
from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from app.services.strategy_models import CampaignSnapshot, DiagnosticSignals

logger = logging.getLogger(__name__)

DECISION_TABLE_PATH = os.environ.get(
    "HELIOS_DECISION_TABLE",
    os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "optimization",
        "method_decision_table.json",
    ),
)

_DIMS_LOW_MAX = 3
_SMOOTHNESS_MULTIMODAL_MAX = 0.25
_NOISE_HIGH_MIN = 0.15

DEFAULT_DECISION_TABLE: dict[tuple[str, str, str], tuple[str, ...]] = {
    ("low", "unimodal", "low"): ("bomcp", "built_in", "optuna_tpe", "pymoo_nsga2"),
    ("low", "multimodal", "low"): ("optuna_cmaes", "scipy_de", "bomcp", "pymoo_nsga2"),
    ("high", "unimodal", "low"): ("bomcp", "optuna_tpe", "pymoo_nsga2", "built_in"),
    ("high", "multimodal", "low"): ("pymoo_nsga2", "optuna_cmaes", "optuna_tpe", "bomcp"),
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
    """Return the decision table: DEFAULT overlaid with a generated artifact."""
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
    *,
    table_path: str | None = None,
) -> tuple[str, ...]:
    """Return backends ranked best-first for the campaign's problem class."""
    profile = problem_profile(snapshot, diag)
    ranking = load_decision_table(table_path).get(profile, _FALLBACK_RANKING)

    if snapshot.has_categorical:
        ranking = ("optuna_tpe", *[b for b in ranking if b != "optuna_tpe"])

    return ranking
