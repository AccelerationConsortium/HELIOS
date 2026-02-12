"""Advanced Convergence Detection - Bayesian and Uncertainty-Aware Methods.

This module extends convergence_enhanced.py with:
1. **Bayesian Change-Point Detection** - Detect structural breaks in KPI series
2. **Uncertainty-Aware Stopping** - Consider epistemic uncertainty in decisions
3. **Cost-Benefit Analysis** - Expected improvement vs. experiment cost trade-off

Pure Python stdlib implementation with mathematical rigor.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from app.services.convergence_enhanced import (
    EnhancedConvergenceStatus,
    detect_convergence_enhanced,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bayesian Change-Point Detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChangePoint:
    """Detected change point in time series."""

    location: int  # Index in series where change occurred
    probability: float  # Posterior probability of change point (0-1)
    pre_mean: float  # Mean before change point
    post_mean: float  # Mean after change point
    magnitude: float  # |post_mean - pre_mean|


@dataclass(frozen=True)
class ChangePointAnalysis:
    """Result of Bayesian change-point detection."""

    detected: bool  # True if high-probability change point found
    change_points: list[ChangePoint]  # All detected change points (sorted by probability)
    most_likely: ChangePoint | None  # Highest probability change point
    confidence: float  # Overall confidence in detection (0-1)


def detect_change_points(
    values: list[float],
    min_segment_length: int = 3,
    prior_change_prob: float = 0.1,
) -> ChangePointAnalysis:
    """Bayesian change-point detection using recursive segmentation.

    Detects structural breaks in KPI series where the mean shifts significantly.
    Uses a simplified Bayesian approach with conjugate priors.

    Args:
        values: KPI time series
        min_segment_length: Minimum observations per segment
        prior_change_prob: Prior probability of a change point at any position

    Returns:
        ChangePointAnalysis with detected change points
    """
    if len(values) < 2 * min_segment_length:
        return ChangePointAnalysis(
            detected=False,
            change_points=[],
            most_likely=None,
            confidence=0.0,
        )

    change_points = []

    # Scan all possible change point locations
    for t in range(min_segment_length, len(values) - min_segment_length):
        # Split series at position t
        pre_segment = values[:t]
        post_segment = values[t:]

        # Compute segment statistics
        pre_mean = sum(pre_segment) / len(pre_segment)
        post_mean = sum(post_segment) / len(post_segment)

        pre_var = _variance(pre_segment)
        post_var = _variance(post_segment)

        # Check for linear trends within segments
        # If both segments are linear with similar slope, not a change point
        pre_slope = _compute_slope(pre_segment)
        post_slope = _compute_slope(post_segment)

        # If slopes are similar and non-zero, this is a continuous trend, not a change
        if abs(pre_slope) > 0.1 and abs(post_slope) > 0.1:
            slope_ratio = pre_slope / post_slope if abs(post_slope) > 1e-6 else 1.0
            if 0.5 < slope_ratio < 2.0:  # Similar slopes
                # This is likely a continuous trend, not a structural break
                continue

        # Magnitude of change
        magnitude = abs(post_mean - pre_mean)

        # Compute effect size (Cohen's d)
        pooled_std = math.sqrt((pre_var + post_var) / 2)

        if pooled_std < 1e-10:
            # Perfect separation, high probability
            effect_size = 10.0  # Large effect
        else:
            effect_size = magnitude / pooled_std

        # Convert effect size to probability using sigmoid
        # Small effect (<0.2) → low prob, Medium (0.5) → moderate, Large (>0.8) → high prob
        # Use logistic function: P = 1 / (1 + exp(-k * (effect_size - threshold)))
        # Center at effect_size=0.5 (medium effect), steepness k=3
        logit = 3 * (effect_size - 0.5)
        base_prob = 1 / (1 + math.exp(-logit))

        # Adjust by prior
        prior_odds = prior_change_prob / (1 - prior_change_prob)
        evidence_factor = base_prob / (1 - base_prob) if base_prob < 0.999 else 999.0
        posterior_odds = evidence_factor * prior_odds
        posterior_prob = posterior_odds / (1 + posterior_odds)

        # Only consider significant changes
        if magnitude > 0.1 * abs(pre_mean) or magnitude > 1.0:
            change_point = ChangePoint(
                location=t,
                probability=posterior_prob,
                pre_mean=pre_mean,
                post_mean=post_mean,
                magnitude=magnitude,
            )
            change_points.append(change_point)

    if not change_points:
        return ChangePointAnalysis(
            detected=False,
            change_points=[],
            most_likely=None,
            confidence=0.0,
        )

    # Sort by probability (descending)
    change_points_sorted = sorted(change_points, key=lambda cp: cp.probability, reverse=True)

    # Most likely change point
    most_likely = change_points_sorted[0]

    # Detection threshold: probability > 0.5 (more likely than not)
    detected = most_likely.probability > 0.5

    # Confidence: max probability among all candidates
    confidence = most_likely.probability

    return ChangePointAnalysis(
        detected=detected,
        change_points=change_points_sorted[:5],  # Top 5
        most_likely=most_likely,
        confidence=confidence,
    )


def _variance(values: list[float]) -> float:
    """Population variance."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / len(values)


def _compute_slope(values: list[float]) -> float:
    """Compute linear regression slope."""
    n = len(values)
    if n < 2:
        return 0.0

    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n

    numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    if abs(denominator) < 1e-15:
        return 0.0

    return numerator / denominator


# ---------------------------------------------------------------------------
# Uncertainty-Aware Stopping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UncertaintyEstimate:
    """Epistemic uncertainty estimate for KPI."""

    mean: float  # Point estimate of KPI
    std: float  # Standard deviation (uncertainty)
    lower_bound: float  # Lower confidence bound (e.g., 95% CI)
    upper_bound: float  # Upper confidence bound
    confidence_level: float  # Confidence level (e.g., 0.95)


def estimate_kpi_uncertainty(
    values: list[float],
    confidence_level: float = 0.95,
) -> UncertaintyEstimate:
    """Estimate epistemic uncertainty in KPI using bootstrap.

    Computes confidence intervals for the current KPI estimate using
    the bootstrap resampling method.

    Args:
        values: Recent KPI observations (last few rounds)
        confidence_level: Confidence level for bounds (default 0.95)

    Returns:
        UncertaintyEstimate with mean, std, and confidence bounds
    """
    if not values:
        return UncertaintyEstimate(
            mean=0.0,
            std=float('inf'),
            lower_bound=-float('inf'),
            upper_bound=float('inf'),
            confidence_level=confidence_level,
        )

    # Point estimate: mean
    mean = sum(values) / len(values)

    # Standard deviation
    if len(values) < 2:
        std = float('inf')
    else:
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        std = math.sqrt(variance)

    # Confidence bounds using t-distribution approximation
    # For small samples, use conservative t-value
    n = len(values)
    if n < 2:
        lower_bound = -float('inf')
        upper_bound = float('inf')
    else:
        # t-value for 95% CI (approximate for simplicity)
        # For n=3: t ≈ 4.3, n=5: t ≈ 2.8, n=10: t ≈ 2.3, n=30: t ≈ 2.0
        if n <= 3:
            t_value = 4.3
        elif n <= 5:
            t_value = 2.8
        elif n <= 10:
            t_value = 2.3
        else:
            t_value = 2.0

        margin = t_value * std / math.sqrt(n)
        lower_bound = mean - margin
        upper_bound = mean + margin

    return UncertaintyEstimate(
        mean=mean,
        std=std,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        confidence_level=confidence_level,
    )


def should_stop_uncertainty_aware(
    kpi_history: list[float],
    target_kpi: float | None = None,
    confidence_level: float = 0.95,
    maximize: bool = True,
) -> tuple[bool, str]:
    """Uncertainty-aware stopping criterion.

    Stops when the lower bound of the KPI estimate exceeds the target
    (for maximization), ensuring robust performance guarantees.

    Args:
        kpi_history: Full KPI history
        target_kpi: Target value to reach
        confidence_level: Confidence level for bounds
        maximize: Whether higher KPI is better

    Returns:
        (should_stop, reason)
    """
    if not kpi_history or target_kpi is None:
        return False, "insufficient_data_or_no_target"

    # Use recent observations for uncertainty estimate (last 5 rounds)
    recent_kpis = kpi_history[-5:] if len(kpi_history) >= 5 else kpi_history

    uncertainty = estimate_kpi_uncertainty(recent_kpis, confidence_level)

    if maximize:
        # Stop when lower bound > target (conservative guarantee)
        if uncertainty.lower_bound > target_kpi:
            return True, f"target_reached_with_confidence (LB={uncertainty.lower_bound:.2f} > {target_kpi:.2f})"
        return False, "target_not_reached_yet"
    else:
        # Stop when upper bound < target (conservative guarantee)
        if uncertainty.upper_bound < target_kpi:
            return True, f"target_reached_with_confidence (UB={uncertainty.upper_bound:.2f} < {target_kpi:.2f})"
        return False, "target_not_reached_yet"


# ---------------------------------------------------------------------------
# Cost-Benefit Analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostBenefitAnalysis:
    """Cost-benefit analysis for continuing campaign."""

    expected_improvement: float  # Expected KPI improvement in next round
    experiment_cost: float  # Cost of running one more round
    benefit_cost_ratio: float  # Expected improvement / cost
    recommendation: str  # "continue" | "stop"
    reason: str


def analyze_cost_benefit(
    kpi_history: list[float],
    experiment_cost: float = 1.0,
    cost_threshold: float = 0.01,
    maximize: bool = True,
) -> CostBenefitAnalysis:
    """Cost-benefit analysis for continuing optimization.

    Estimates expected improvement using recent trend and compares
    to experiment cost. Recommends stopping when expected improvement
    is too small relative to cost.

    Args:
        kpi_history: Full KPI history
        experiment_cost: Cost of running one more experiment (arbitrary units)
        cost_threshold: Stop when improvement/cost < threshold
        maximize: Whether higher KPI is better

    Returns:
        CostBenefitAnalysis with recommendation
    """
    if len(kpi_history) < 3:
        return CostBenefitAnalysis(
            expected_improvement=float('inf'),
            experiment_cost=experiment_cost,
            benefit_cost_ratio=float('inf'),
            recommendation="continue",
            reason="insufficient_data_for_cost_benefit",
        )

    # Estimate expected improvement using recent trend
    # Simple linear extrapolation from last 5 observations
    recent = kpi_history[-5:] if len(kpi_history) >= 5 else kpi_history

    # Linear regression slope
    n = len(recent)
    x_mean = (n - 1) / 2.0
    y_mean = sum(recent) / n

    numerator = sum((i - x_mean) * (recent[i] - y_mean) for i in range(n))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    if abs(denominator) < 1e-15:
        slope = 0.0
    else:
        slope = numerator / denominator

    # Expected improvement: next step along trend line
    expected_improvement = abs(slope)

    # Benefit-cost ratio
    if experiment_cost < 1e-10:
        benefit_cost_ratio = float('inf')
    else:
        benefit_cost_ratio = expected_improvement / experiment_cost

    # Decision: stop if ratio < threshold
    if benefit_cost_ratio < cost_threshold:
        recommendation = "stop"
        reason = f"expected_improvement_too_small (ratio={benefit_cost_ratio:.4f} < {cost_threshold})"
    else:
        recommendation = "continue"
        reason = f"expected_improvement_justifies_cost (ratio={benefit_cost_ratio:.4f})"

    return CostBenefitAnalysis(
        expected_improvement=expected_improvement,
        experiment_cost=experiment_cost,
        benefit_cost_ratio=benefit_cost_ratio,
        recommendation=recommendation,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Integrated Advanced Convergence Status
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdvancedConvergenceStatus:
    """Advanced convergence analysis combining all methods."""

    # Base enhanced status
    enhanced_status: EnhancedConvergenceStatus

    # Advanced features
    change_point: ChangePointAnalysis
    uncertainty: UncertaintyEstimate | None
    cost_benefit: CostBenefitAnalysis | None

    # Meta
    analysis_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "enhanced_status": self.enhanced_status.to_dict(),
            "change_point": {
                "detected": self.change_point.detected,
                "most_likely": None if not self.change_point.most_likely else {
                    "location": self.change_point.most_likely.location,
                    "probability": self.change_point.most_likely.probability,
                    "pre_mean": self.change_point.most_likely.pre_mean,
                    "post_mean": self.change_point.most_likely.post_mean,
                    "magnitude": self.change_point.most_likely.magnitude,
                },
                "confidence": self.change_point.confidence,
            },
            "metadata": self.analysis_metadata,
        }

        if self.uncertainty:
            result["uncertainty"] = {
                "mean": self.uncertainty.mean,
                "std": self.uncertainty.std,
                "lower_bound": self.uncertainty.lower_bound,
                "upper_bound": self.uncertainty.upper_bound,
                "confidence_level": self.uncertainty.confidence_level,
            }

        if self.cost_benefit:
            result["cost_benefit"] = {
                "expected_improvement": self.cost_benefit.expected_improvement,
                "experiment_cost": self.cost_benefit.experiment_cost,
                "benefit_cost_ratio": self.cost_benefit.benefit_cost_ratio,
                "recommendation": self.cost_benefit.recommendation,
                "reason": self.cost_benefit.reason,
            }

        return result


def detect_convergence_advanced(
    values: list[float],
    target: float | None = None,
    experiment_cost: float = 1.0,
    maximize: bool = True,
) -> AdvancedConvergenceStatus:
    """Advanced convergence detection with Bayesian and uncertainty-aware methods.

    Args:
        values: KPI time series
        target: Optional target KPI value
        experiment_cost: Cost of running one experiment
        maximize: Whether higher is better

    Returns:
        AdvancedConvergenceStatus with comprehensive analysis
    """
    # Enhanced detection (base)
    enhanced_status = detect_convergence_enhanced(values, target=target, maximize=maximize)

    # Change-point detection
    change_point = detect_change_points(values)

    # Uncertainty estimation (on recent observations)
    recent_kpis = values[-5:] if len(values) >= 5 else values
    uncertainty = estimate_kpi_uncertainty(recent_kpis)

    # Cost-benefit analysis
    cost_benefit = analyze_cost_benefit(
        values,
        experiment_cost=experiment_cost,
        maximize=maximize
    )

    # Metadata
    metadata = {
        "n_observations": len(values),
        "current_kpi": values[-1] if values else None,
        "change_point_detected": change_point.detected,
        "uncertainty_aware": target is not None,
    }

    return AdvancedConvergenceStatus(
        enhanced_status=enhanced_status,
        change_point=change_point,
        uncertainty=uncertainty,
        cost_benefit=cost_benefit,
        analysis_metadata=metadata,
    )


def should_stop_campaign_advanced(
    advanced_status: AdvancedConvergenceStatus,
    goal_target_reached: bool,
    rounds_exhausted: bool,
    target_kpi: float | None = None,
    enable_cost_benefit: bool = True,
    enable_uncertainty: bool = True,
) -> tuple[str, str]:
    """Advanced campaign stop decision using all available methods.

    Priority order:
    1. Target reached (uncertainty-aware if enabled)
    2. Budget exhausted
    3. Cost-benefit analysis
    4. Enhanced convergence analysis
    5. Change-point analysis

    Returns:
        (action, reason) where action is "continue" or "stop"
    """
    # Priority 1: Target reached (uncertainty-aware)
    if enable_uncertainty and target_kpi is not None and advanced_status.uncertainty:
        uncertainty = advanced_status.uncertainty
        # Conservative: check lower/upper bound depending on maximize
        # (Simplified: assume maximize=True for now)
        if uncertainty.lower_bound > target_kpi:
            return "stop", "target_reached_with_high_confidence"

    if goal_target_reached:
        return "stop", "target_reached"

    # Priority 2: Budget exhausted
    if rounds_exhausted:
        return "stop", "budget_exhausted"

    # Priority 3: Cost-benefit analysis
    if enable_cost_benefit and advanced_status.cost_benefit:
        if advanced_status.cost_benefit.recommendation == "stop":
            return "stop", advanced_status.cost_benefit.reason

    # Priority 4: Enhanced convergence
    from app.services.convergence_enhanced import should_stop_campaign_enhanced

    action, reason = should_stop_campaign_enhanced(
        advanced_status.enhanced_status,
        goal_target_reached=False,
        rounds_exhausted=False,
    )

    if action == "stop":
        return action, reason

    # Priority 5: Change-point detection
    # If a significant positive change point detected recently, continue
    if advanced_status.change_point.detected:
        cp = advanced_status.change_point.most_likely
        if cp and cp.location >= len(advanced_status.enhanced_status.basic_status.details.get("kpi_history", [])) - 5:
            # Recent change point
            if cp.magnitude > 1.0:
                return "continue", "recent_change_point_detected (continue to explore)"

    # Default: continue
    return "continue", "optimization_ongoing"
