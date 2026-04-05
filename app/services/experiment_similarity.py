"""Embedding-based similar experiment retrieval.

Enables agents to find "the most similar past experiment" for a given
parameter configuration, and use its outcomes to calibrate confidence.

Architecture:
- **ExperimentFingerprint**: normalized vector of parameter values for a
  candidate or historical experiment.
- **In-memory index**: built from ``campaign_candidates`` table at query
  time (no external vector DB dependency).
- **Cosine similarity**: O(n) scan — fast enough for <100k experiments,
  no ANN index needed yet.

Usage::

    from app.services.experiment_similarity import (
        find_similar_experiments,
        SimilarExperiment,
    )

    results = find_similar_experiments(
        query_params={"temp_c": 80, "time_s": 300, "volume_ul": 100},
        campaign_id="camp-abc123",  # scope to this campaign, or None for all
        top_k=5,
    )
    for r in results:
        print(f"similarity={r.similarity:.2f}, kpi={r.kpi_value}, params={r.params}")

Design principles:
- No external dependencies (numpy only, already in deps).
- Advisory — all errors swallowed, never blocks.
- Campaign-scoped or global search.
- Works with heterogeneous parameter spaces (missing params → 0).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from app.core.db import connection, parse_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimilarExperiment:
    """A historical experiment similar to the query."""

    campaign_id: str
    round_number: int
    candidate_index: int
    params: dict[str, Any]
    kpi_value: float | None
    status: str
    similarity: float  # 0.0–1.0, cosine similarity
    run_id: str | None = None


@dataclass(frozen=True)
class SimilarityReport:
    """Summary of similar experiment search results."""

    query_params: dict[str, Any]
    matches: list[SimilarExperiment]
    confidence_estimate: float  # 0.0–1.0 based on match quality
    avg_kpi: float | None       # average KPI of top matches
    kpi_stddev: float | None    # stddev of top matches
    explanation: str = ""


# ---------------------------------------------------------------------------
# Fingerprint: normalize params to a fixed-dim vector
# ---------------------------------------------------------------------------


def _build_fingerprint(
    params: dict[str, Any],
    dimensions: list[str],
    normalizers: dict[str, tuple[float, float]],  # param_name → (min, max)
) -> list[float]:
    """Convert params dict to a normalized float vector.

    Numeric params are min-max normalized to [0, 1].
    Missing params get 0.0 (neutral in cosine similarity).
    Non-numeric params are hashed to a float in [0, 1].
    """
    vec: list[float] = []
    for dim in dimensions:
        raw = params.get(dim)
        if raw is None:
            vec.append(0.0)
        elif isinstance(raw, (int, float)) and not isinstance(raw, bool):
            mn, mx = normalizers.get(dim, (0.0, 1.0))
            span = mx - mn
            if span > 0:
                vec.append((float(raw) - mn) / span)
            else:
                vec.append(0.5)
        elif isinstance(raw, str):
            # Deterministic hash for categorical params
            vec.append((hash(raw) % 10000) / 10000.0)
        else:
            vec.append(0.0)
    return vec


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two float vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Index builder: read historical experiments from DB
# ---------------------------------------------------------------------------


def _load_historical_experiments(
    campaign_id: str | None = None,
) -> list[dict[str, Any]]:
    """Load completed experiments from campaign_candidates table."""
    try:
        with connection() as conn:
            if campaign_id:
                rows = conn.execute(
                    "SELECT campaign_id, round_number, candidate_index, "
                    "       params_json, kpi_value, status, run_id "
                    "FROM campaign_candidates "
                    "WHERE campaign_id = ? AND status IN ('completed', 'succeeded') "
                    "ORDER BY round_number ASC, candidate_index ASC",
                    (campaign_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT campaign_id, round_number, candidate_index, "
                    "       params_json, kpi_value, status, run_id "
                    "FROM campaign_candidates "
                    "WHERE status IN ('completed', 'succeeded') "
                    "ORDER BY campaign_id, round_number ASC, candidate_index ASC "
                    "LIMIT 10000",
                ).fetchall()
            return [
                {
                    "campaign_id": r["campaign_id"],
                    "round_number": r["round_number"],
                    "candidate_index": r["candidate_index"],
                    "params": parse_json(r["params_json"], {}),
                    "kpi_value": r["kpi_value"],
                    "status": r["status"],
                    "run_id": r["run_id"],
                }
                for r in rows
            ]
    except Exception:
        logger.debug("Failed to load historical experiments", exc_info=True)
        return []


def _compute_normalizers(
    experiments: list[dict[str, Any]],
    dimensions: list[str],
) -> dict[str, tuple[float, float]]:
    """Compute min/max for each numeric dimension."""
    mins: dict[str, float] = {}
    maxs: dict[str, float] = {}

    for exp in experiments:
        params = exp["params"]
        for dim in dimensions:
            val = params.get(dim)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                fval = float(val)
                if dim not in mins or fval < mins[dim]:
                    mins[dim] = fval
                if dim not in maxs or fval > maxs[dim]:
                    maxs[dim] = fval

    return {
        dim: (mins.get(dim, 0.0), maxs.get(dim, 1.0))
        for dim in dimensions
    }


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


def find_similar_experiments(
    query_params: dict[str, Any],
    campaign_id: str | None = None,
    top_k: int = 5,
) -> list[SimilarExperiment]:
    """Find the top-k most similar historical experiments.

    Parameters
    ----------
    query_params : dict
        Parameter values for the query experiment.
    campaign_id : str | None
        Scope search to this campaign. None = search all.
    top_k : int
        Number of results to return.

    Returns
    -------
    list[SimilarExperiment]
        Sorted by similarity (highest first).
    """
    experiments = _load_historical_experiments(campaign_id)
    if not experiments:
        return []

    # Collect all dimension names across all experiments + query
    all_dims: set[str] = set(query_params.keys())
    for exp in experiments:
        all_dims.update(exp["params"].keys())
    dimensions = sorted(all_dims)

    # Build normalizers from historical data
    normalizers = _compute_normalizers(experiments, dimensions)

    # Fingerprint the query
    query_fp = _build_fingerprint(query_params, dimensions, normalizers)

    # Score all historical experiments
    scored: list[tuple[float, dict[str, Any]]] = []
    for exp in experiments:
        exp_fp = _build_fingerprint(exp["params"], dimensions, normalizers)
        sim = _cosine_similarity(query_fp, exp_fp)
        scored.append((sim, exp))

    # Sort by similarity descending
    scored.sort(key=lambda x: x[0], reverse=True)

    return [
        SimilarExperiment(
            campaign_id=exp["campaign_id"],
            round_number=exp["round_number"],
            candidate_index=exp["candidate_index"],
            params=exp["params"],
            kpi_value=exp["kpi_value"],
            status=exp["status"],
            similarity=sim,
            run_id=exp.get("run_id"),
        )
        for sim, exp in scored[:top_k]
    ]


def build_similarity_report(
    query_params: dict[str, Any],
    campaign_id: str | None = None,
    top_k: int = 5,
) -> SimilarityReport:
    """Find similar experiments and build a confidence-calibrated report.

    The confidence_estimate is based on:
    - How similar the top match is (similarity > 0.95 = high confidence)
    - How consistent the top matches' KPIs are (low stddev = high confidence)
    - How many matches were found (more = higher confidence)

    This report is intended for DesignAgent to calibrate how much to trust
    its current parameter choices, and for ExecutionAgent to decide
    granularity.
    """
    matches = find_similar_experiments(query_params, campaign_id, top_k)

    if not matches:
        return SimilarityReport(
            query_params=query_params,
            matches=[],
            confidence_estimate=0.0,
            avg_kpi=None,
            kpi_stddev=None,
            explanation="No historical experiments found",
        )

    # Compute KPI statistics from matches with valid KPIs
    kpi_vals = [m.kpi_value for m in matches if m.kpi_value is not None]
    avg_kpi = sum(kpi_vals) / len(kpi_vals) if kpi_vals else None
    kpi_stddev = None
    if len(kpi_vals) >= 2 and avg_kpi is not None:
        variance = sum((v - avg_kpi) ** 2 for v in kpi_vals) / (len(kpi_vals) - 1)
        kpi_stddev = math.sqrt(variance)

    # Confidence estimation
    best_sim = matches[0].similarity
    n_good = sum(1 for m in matches if m.similarity > 0.8)

    # Three signals → confidence
    sim_conf = min(1.0, best_sim)  # How similar is the best match?
    coverage_conf = min(1.0, n_good / max(top_k, 1))  # How many good matches?
    consistency_conf = 1.0
    if kpi_stddev is not None and avg_kpi is not None and avg_kpi != 0:
        cv = kpi_stddev / abs(avg_kpi)  # coefficient of variation
        consistency_conf = max(0.0, 1.0 - cv)  # low CV = high consistency

    confidence = (sim_conf * 0.4 + coverage_conf * 0.3 + consistency_conf * 0.3)

    explanation_parts = [
        f"best_similarity={best_sim:.2f}",
        f"n_good_matches={n_good}/{top_k}",
    ]
    if avg_kpi is not None:
        explanation_parts.append(f"avg_kpi={avg_kpi:.4f}")
    if kpi_stddev is not None:
        explanation_parts.append(f"kpi_stddev={kpi_stddev:.4f}")

    return SimilarityReport(
        query_params=query_params,
        matches=matches,
        confidence_estimate=confidence,
        avg_kpi=avg_kpi,
        kpi_stddev=kpi_stddev,
        explanation=" | ".join(explanation_parts),
    )
