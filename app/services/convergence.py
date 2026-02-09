"""Convergence Detector for experiment optimization campaigns.

Analyzes KPI time-series to determine whether an optimization campaign has
plateaued, is still improving, or is diverging.  Three detection methods are
combined via weighted vote:

1. **Rolling improvement rate** -- average relative improvement over a sliding window
2. **Best-KPI slope** -- linear regression on the cumulative-best series
3. **Variance collapse** -- ratio of recent variance to total variance

All operations are advisory -- wrapped in try/except, never block
run completion.  Pure Python stdlib only (no numpy / scipy).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConvergenceStatus:
    """Result of convergence analysis."""

    status: str  # "improving" | "plateau" | "diverging" | "insufficient_data"
    confidence: float  # 0.0 - 1.0
    details: dict[str, Any]  # per-method breakdown

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "confidence": self.confidence,
            "details": self.details,
        }


@dataclass(frozen=True)
class ConvergenceConfig:
    """Configuration for convergence detection."""

    window_size: int = 5  # rolling window size
    plateau_threshold: float = 0.01  # improvement rate below this = plateau
    divergence_threshold: float = -0.05  # negative improvement below this = diverging
    min_observations: int = 5  # minimum data points needed
    variance_collapse_ratio: float = 0.3  # var(recent) / var(all) below this = converged


# ---------------------------------------------------------------------------
# Pure Python stats helpers
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    """Arithmetic mean."""
    return sum(values) / len(values) if values else 0.0


def _variance(values: list[float]) -> float:
    """Population variance."""
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return sum((x - m) ** 2 for x in values) / len(values)


def _linear_regression_slope(ys: list[float]) -> float:
    """Slope of simple linear regression y = a + b*x where x = 0,1,...,n-1.

    Uses the formula:
        b = (n * sum(x*y) - sum(x) * sum(y)) / (n * sum(x^2) - sum(x)^2)
    """
    n = len(ys)
    if n < 2:
        return 0.0

    sum_x = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    sum_x2 = 0.0

    for i, y in enumerate(ys):
        x = float(i)
        sum_x += x
        sum_y += y
        sum_xy += x * y
        sum_x2 += x * x

    denominator = n * sum_x2 - sum_x * sum_x
    if abs(denominator) < 1e-15:
        return 0.0

    return (n * sum_xy - sum_x * sum_y) / denominator


# ---------------------------------------------------------------------------
# Core detection methods
# ---------------------------------------------------------------------------


def rolling_improvement_rate(
    values: list[float], window: int = 5
) -> float | None:
    """Compute average relative improvement over rolling window.

    For each consecutive pair in the last ``window`` values::

        improvement_i = (v[i] - v[i-1]) / max(abs(v[i-1]), 1e-10)

    Returns mean improvement rate.  Positive = improving, negative = degrading.
    Returns ``None`` if insufficient data (need at least 2 values in window).
    """
    if len(values) < 2:
        return None

    tail = values[-window:] if len(values) >= window else values
    if len(tail) < 2:
        return None

    improvements: list[float] = []
    for i in range(1, len(tail)):
        denom = max(abs(tail[i - 1]), 1e-10)
        improvements.append((tail[i] - tail[i - 1]) / denom)

    return _mean(improvements)


def best_kpi_slope(
    values: list[float], window: int = 5, maximize: bool = True
) -> float | None:
    """Linear regression slope of the cumulative-best KPI over last ``window`` points.

    Computes the cumulative best (running maximum for *maximize*, running
    minimum for *minimize*), then fits a simple linear regression on the
    cumulative-best values within the window.

    Returns slope.  Near-zero = plateau, positive = improving (for maximize).
    Returns ``None`` if insufficient data.
    """
    if len(values) < 2:
        return None

    # Build cumulative best series over ALL values
    cum_best: list[float] = []
    current_best = values[0]
    for v in values:
        if maximize:
            current_best = max(current_best, v)
        else:
            current_best = min(current_best, v)
        cum_best.append(current_best)

    # Take the last `window` points of the cumulative best
    tail = cum_best[-window:] if len(cum_best) >= window else cum_best
    if len(tail) < 2:
        return None

    slope = _linear_regression_slope(tail)

    # For minimize objectives, negate slope so positive = improving
    if not maximize:
        slope = -slope

    return slope


def variance_collapse(
    values: list[float], recent_window: int = 5
) -> float | None:
    """Ratio of variance in recent window to total variance.

    ``variance_collapse_ratio = var(last recent_window values) / var(all values)``

    Low ratio = convergence (recent values are tightly clustered relative to
    overall spread).  Returns ``None`` if insufficient data or zero total
    variance.
    """
    if len(values) < recent_window or len(values) < 2:
        return None

    total_var = _variance(values)
    if total_var < 1e-15:
        # All values identical -- perfectly converged
        return 0.0

    recent = values[-recent_window:]
    recent_var = _variance(recent)
    return recent_var / total_var


# ---------------------------------------------------------------------------
# Main convergence detector
# ---------------------------------------------------------------------------

# Weights for the three methods in the weighted vote
_W_IMPROVEMENT = 0.4
_W_SLOPE = 0.3
_W_VARIANCE = 0.3


def detect_convergence(
    values: list[float],
    config: ConvergenceConfig | None = None,
    maximize: bool = True,
) -> ConvergenceStatus:
    """Analyze a KPI time-series for convergence.

    Combines all three methods into a single verdict:

    1. Compute rolling improvement rate
    2. Compute best-KPI slope
    3. Compute variance collapse ratio
    4. Weighted vote: improvement_rate(0.4) + slope(0.3) + variance(0.3)

    Each method votes ``"plateau"``, ``"improving"``, or ``"diverging"`` with
    a confidence score.  The final status is the majority vote and the final
    confidence is the weighted average.

    Args:
        values: KPI values in chronological order (one per run / batch).
        config: Detection thresholds.
        maximize: If ``True``, higher values are better.

    Returns:
        :class:`ConvergenceStatus` with status, confidence, and per-method
        details.
    """
    if config is None:
        config = ConvergenceConfig()

    # Guard: insufficient data
    if len(values) < config.min_observations:
        return ConvergenceStatus(
            status="insufficient_data",
            confidence=0.0,
            details={"reason": f"need >= {config.min_observations} observations, got {len(values)}"},
        )

    # --- Method 1: Rolling improvement rate ---
    imp_rate = rolling_improvement_rate(values, window=config.window_size)
    # For minimize objectives, negate so positive = improving
    if imp_rate is not None and not maximize:
        imp_rate = -imp_rate

    imp_vote, imp_conf = _vote_from_rate(imp_rate, config)

    # --- Method 2: Best-KPI slope ---
    slope = best_kpi_slope(values, window=config.window_size, maximize=maximize)
    slope_vote, slope_conf = _vote_from_slope(slope, config)

    # --- Method 3: Variance collapse ---
    vc_ratio = variance_collapse(values, recent_window=config.window_size)
    vc_vote, vc_conf = _vote_from_variance(vc_ratio, config)

    # --- Weighted vote ---
    votes: dict[str, float] = {"improving": 0.0, "plateau": 0.0, "diverging": 0.0}

    if imp_vote is not None:
        votes[imp_vote] += _W_IMPROVEMENT * imp_conf
    if slope_vote is not None:
        votes[slope_vote] += _W_SLOPE * slope_conf
    if vc_vote is not None:
        votes[vc_vote] += _W_VARIANCE * vc_conf

    total_weight = sum(votes.values())
    if total_weight < 1e-10:
        final_status = "plateau"
        final_confidence = 0.0
    else:
        final_status = max(votes, key=votes.get)  # type: ignore[arg-type]
        final_confidence = votes[final_status] / total_weight

    # Clamp confidence to [0, 1]
    final_confidence = max(0.0, min(1.0, round(final_confidence, 4)))

    details = {
        "rolling_improvement": {
            "rate": round(imp_rate, 6) if imp_rate is not None else None,
            "vote": imp_vote,
            "confidence": round(imp_conf, 4),
        },
        "best_kpi_slope": {
            "slope": round(slope, 6) if slope is not None else None,
            "vote": slope_vote,
            "confidence": round(slope_conf, 4),
        },
        "variance_collapse": {
            "ratio": round(vc_ratio, 6) if vc_ratio is not None else None,
            "vote": vc_vote,
            "confidence": round(vc_conf, 4),
        },
        "vote_weights": {
            "improving": round(votes["improving"], 4),
            "plateau": round(votes["plateau"], 4),
            "diverging": round(votes["diverging"], 4),
        },
        "n_observations": len(values),
    }

    return ConvergenceStatus(
        status=final_status,
        confidence=final_confidence,
        details=details,
    )


# ---------------------------------------------------------------------------
# Per-method voting helpers
# ---------------------------------------------------------------------------


def _vote_from_rate(
    rate: float | None, config: ConvergenceConfig
) -> tuple[str | None, float]:
    """Convert improvement rate into a vote + confidence."""
    if rate is None:
        return None, 0.0
    if rate > config.plateau_threshold:
        # Improving -- confidence scales with how far above threshold
        conf = min(1.0, rate / (config.plateau_threshold * 10))
        return "improving", max(0.1, conf)
    if rate < config.divergence_threshold:
        conf = min(1.0, abs(rate) / abs(config.divergence_threshold * 5))
        return "diverging", max(0.1, conf)
    # Between divergence and plateau thresholds
    conf = 1.0 - abs(rate) / config.plateau_threshold if config.plateau_threshold else 0.5
    return "plateau", max(0.1, min(1.0, conf))


def _vote_from_slope(
    slope: float | None, config: ConvergenceConfig
) -> tuple[str | None, float]:
    """Convert best-KPI slope into a vote + confidence."""
    if slope is None:
        return None, 0.0
    if slope > config.plateau_threshold:
        conf = min(1.0, slope / (config.plateau_threshold * 5))
        return "improving", max(0.1, conf)
    if slope < config.divergence_threshold:
        # cumulative best slope shouldn't go negative, but handle defensively
        conf = min(1.0, abs(slope) / abs(config.divergence_threshold * 3))
        return "diverging", max(0.1, conf)
    return "plateau", min(1.0, max(0.5, 1.0 - abs(slope) / config.plateau_threshold))


def _vote_from_variance(
    ratio: float | None, config: ConvergenceConfig
) -> tuple[str | None, float]:
    """Convert variance collapse ratio into a vote + confidence."""
    if ratio is None:
        return None, 0.0
    if ratio < config.variance_collapse_ratio:
        # Low ratio = converged = plateau
        conf = 1.0 - (ratio / config.variance_collapse_ratio)
        return "plateau", max(0.1, min(1.0, conf))
    if ratio > 1.5:
        # Recent variance exceeds total variance -- diverging
        conf = min(1.0, (ratio - 1.0) / 2.0)
        return "diverging", max(0.1, conf)
    # Moderate ratio -- still improving / exploring
    conf = ratio / 2.0
    return "improving", max(0.1, min(1.0, conf))


# ---------------------------------------------------------------------------
# DB integration
# ---------------------------------------------------------------------------


def detect_campaign_convergence(
    campaign_id: str | None = None,
    kpi_name: str = "run_success_rate",
    config: ConvergenceConfig | None = None,
    maximize: bool = True,
) -> ConvergenceStatus:
    """Load KPI history from DB and detect convergence.

    Uses :func:`~app.services.metrics.get_kpi_summary` to load KPI values,
    then calls :func:`detect_convergence`.

    Args:
        campaign_id: Optional campaign filter (currently unused in
            ``get_kpi_summary`` -- included for forward compatibility).
        kpi_name: Which KPI to analyze.
        config: Detection thresholds.
        maximize: Whether higher KPI values are better.

    Returns:
        :class:`ConvergenceStatus` with the convergence verdict.
    """
    try:
        from app.services.metrics import get_kpi_summary

        rows = get_kpi_summary(kpi_name)

        # Rows come back newest-first; reverse to chronological order
        rows.reverse()

        # Extract numeric KPI values, skipping nulls
        values: list[float] = []
        for row in rows:
            v = row.get("kpi_value")
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                values.append(float(v))

        return detect_convergence(values, config=config, maximize=maximize)

    except Exception:
        logger.warning(
            "Failed to detect campaign convergence for kpi=%s",
            kpi_name,
            exc_info=True,
        )
        return ConvergenceStatus(
            status="insufficient_data",
            confidence=0.0,
            details={"error": "failed to load KPI data"},
        )
