"""Enhanced Convergence Detection with Advanced Patterns.

Extends the basic convergence detector with:
1. **Oscillation detection** - Identifies cyclical patterns
2. **Noise characterization** - Distinguishes signal from noise
3. **Adaptive thresholds** - Data-driven threshold selection
4. **Multi-scale analysis** - Short-term vs long-term trends
5. **Confidence calibration** - Better uncertainty quantification
6. **Pattern recognition** - Identifies common convergence patterns

All pure Python stdlib, no external dependencies.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from app.services.convergence import (
    ConvergenceConfig,
    ConvergenceStatus,
    detect_convergence as basic_detect_convergence,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enhanced detection patterns
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OscillationPattern:
    """Detected oscillation pattern in KPI series."""

    detected: bool
    period: int | None  # Estimated period (number of observations)
    amplitude: float  # Oscillation amplitude (normalized)
    confidence: float  # 0-1


@dataclass(frozen=True)
class NoiseCharacterization:
    """Noise analysis of KPI series."""

    signal_to_noise_ratio: float  # SNR (higher = cleaner signal)
    noise_level: float  # Estimated noise std dev
    is_noisy: bool  # True if SNR < threshold
    confidence: float


@dataclass(frozen=True)
class EnhancedConvergenceStatus:
    """Enhanced convergence analysis with additional patterns."""

    # Basic status (from original detector)
    basic_status: ConvergenceStatus

    # Enhanced patterns
    oscillation: OscillationPattern
    noise: NoiseCharacterization

    # Multi-scale analysis
    short_term_trend: str  # "improving" | "plateau" | "diverging" (last 3-5 obs)
    long_term_trend: str   # Based on all data

    # Adaptive insights
    adaptive_threshold: float | None  # Data-driven convergence threshold
    estimated_convergence_round: int | None  # Predicted round to convergence

    # Meta
    analysis_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "basic_status": self.basic_status.to_dict(),
            "oscillation": {
                "detected": self.oscillation.detected,
                "period": self.oscillation.period,
                "amplitude": self.oscillation.amplitude,
                "confidence": self.oscillation.confidence,
            },
            "noise": {
                "snr": self.noise.signal_to_noise_ratio,
                "noise_level": self.noise.noise_level,
                "is_noisy": self.noise.is_noisy,
                "confidence": self.noise.confidence,
            },
            "short_term_trend": self.short_term_trend,
            "long_term_trend": self.long_term_trend,
            "adaptive_threshold": self.adaptive_threshold,
            "estimated_convergence_round": self.estimated_convergence_round,
            "metadata": self.analysis_metadata,
        }


# ---------------------------------------------------------------------------
# Oscillation detection
# ---------------------------------------------------------------------------


def detect_oscillation(
    values: list[float],
    min_period: int = 2,
    max_period: int = 10,
    confidence_threshold: float = 0.6,
) -> OscillationPattern:
    """Detect cyclical oscillation patterns using autocorrelation.

    Args:
        values: KPI time series
        min_period: Minimum period to check
        max_period: Maximum period to check
        confidence_threshold: Minimum confidence to declare oscillation

    Returns:
        OscillationPattern with detection results
    """
    if len(values) < 2 * min_period:
        return OscillationPattern(
            detected=False,
            period=None,
            amplitude=0.0,
            confidence=0.0,
        )

    # Detrend: remove linear trend
    detrended = _detrend_linear(values)

    # Compute autocorrelation for different lags
    autocorrs = []
    for lag in range(min_period, min(max_period + 1, len(values) // 2)):
        acf = _autocorrelation(detrended, lag)
        autocorrs.append((lag, acf))

    if not autocorrs:
        return OscillationPattern(detected=False, period=None, amplitude=0.0, confidence=0.0)

    # Find strongest periodic signal
    best_lag, best_acf = max(autocorrs, key=lambda x: x[1])

    # Estimate amplitude (std dev of detrended series)
    amplitude = _std_dev(detrended)

    # Normalize amplitude by mean absolute value
    mean_abs = sum(abs(v) for v in values) / len(values) if values else 1.0
    normalized_amplitude = amplitude / mean_abs if mean_abs > 1e-10 else 0.0

    # Confidence based on autocorrelation strength
    confidence = max(0.0, min(1.0, best_acf))

    detected = best_acf > confidence_threshold

    return OscillationPattern(
        detected=detected,
        period=best_lag if detected else None,
        amplitude=normalized_amplitude,
        confidence=confidence,
    )


def _detrend_linear(values: list[float]) -> list[float]:
    """Remove linear trend from series."""
    from app.services.convergence import _linear_regression_slope, _mean

    if len(values) < 2:
        return values

    slope = _linear_regression_slope(values)
    mean_val = _mean(values)

    # Compute intercept: mean_y = intercept + slope * mean_x
    mean_x = (len(values) - 1) / 2.0
    intercept = mean_val - slope * mean_x

    # Subtract trend
    detrended = []
    for i, y in enumerate(values):
        trend = intercept + slope * i
        detrended.append(y - trend)

    return detrended


def _autocorrelation(values: list[float], lag: int) -> float:
    """Compute autocorrelation at given lag."""
    n = len(values)
    if lag >= n or lag < 1:
        return 0.0

    # Center the series
    from app.services.convergence import _mean
    mean_val = _mean(values)
    centered = [v - mean_val for v in values]

    # Compute autocorrelation
    numerator = sum(centered[i] * centered[i + lag] for i in range(n - lag))
    denominator = sum(v * v for v in centered)

    if abs(denominator) < 1e-15:
        return 0.0

    return numerator / denominator


def _std_dev(values: list[float]) -> float:
    """Standard deviation."""
    from app.services.convergence import _variance
    return math.sqrt(_variance(values))


# ---------------------------------------------------------------------------
# Noise characterization
# ---------------------------------------------------------------------------


def characterize_noise(
    values: list[float],
    snr_threshold: float = 3.0,
) -> NoiseCharacterization:
    """Characterize noise level in KPI series.

    Uses differencing method to estimate noise:
    - Signal: smoothed trend
    - Noise: residuals from trend

    Args:
        values: KPI time series
        snr_threshold: SNR below this = noisy

    Returns:
        NoiseCharacterization with SNR and noise level
    """
    if len(values) < 3:
        return NoiseCharacterization(
            signal_to_noise_ratio=float('inf'),
            noise_level=0.0,
            is_noisy=False,
            confidence=0.0,
        )

    # Smooth with moving average (window=3)
    smoothed = _moving_average(values, window=3)

    # Compute residuals (noise)
    residuals = [values[i] - smoothed[i] for i in range(len(values))]

    # Signal strength: std dev of smoothed series
    signal_std = _std_dev(smoothed)

    # Noise level: std dev of residuals
    noise_std = _std_dev(residuals)

    # SNR (signal-to-noise ratio)
    if noise_std < 1e-10:
        snr = float('inf')
    else:
        snr = signal_std / noise_std

    is_noisy = snr < snr_threshold

    # Confidence: higher for more data
    confidence = min(1.0, len(values) / 20.0)

    return NoiseCharacterization(
        signal_to_noise_ratio=snr,
        noise_level=noise_std,
        is_noisy=is_noisy,
        confidence=confidence,
    )


def _moving_average(values: list[float], window: int) -> list[float]:
    """Simple moving average."""
    if len(values) < window:
        window = max(1, len(values))

    smoothed = []
    for i in range(len(values)):
        # Use symmetric window when possible
        if i < window // 2:
            # Start: use available data
            win_values = values[:window]
        elif i >= len(values) - window // 2:
            # End: use available data
            win_values = values[-window:]
        else:
            # Middle: centered window
            start = i - window // 2
            end = start + window
            win_values = values[start:end]

        from app.services.convergence import _mean
        smoothed.append(_mean(win_values))

    return smoothed


# ---------------------------------------------------------------------------
# Multi-scale trend analysis
# ---------------------------------------------------------------------------


def analyze_multi_scale_trends(
    values: list[float],
    short_window: int = 5,
    config: ConvergenceConfig | None = None,
    maximize: bool = True,
) -> tuple[str, str]:
    """Analyze short-term vs long-term trends.

    Returns:
        (short_term_trend, long_term_trend) where each is "improving"/"plateau"/"diverging"
    """
    if not values:
        return "insufficient_data", "insufficient_data"

    # Long-term: use full history
    long_term_status = basic_detect_convergence(values, config=config, maximize=maximize)
    long_term = long_term_status.status

    # Short-term: use recent window
    if len(values) < short_window:
        short_term = long_term
    else:
        recent_values = values[-short_window:]

        # Create relaxed config for short-term analysis with lower min_observations
        short_term_config = config or ConvergenceConfig()
        # For short windows, reduce min_observations requirement to allow analysis
        short_term_config = ConvergenceConfig(
            window_size=min(short_term_config.window_size, len(recent_values)),
            plateau_threshold=short_term_config.plateau_threshold,
            divergence_threshold=short_term_config.divergence_threshold,
            min_observations=max(3, min(len(recent_values), short_window)),
            variance_collapse_ratio=short_term_config.variance_collapse_ratio,
        )

        short_term_status = basic_detect_convergence(
            recent_values,
            config=short_term_config,
            maximize=maximize
        )
        short_term = short_term_status.status

    return short_term, long_term


# ---------------------------------------------------------------------------
# Adaptive threshold selection
# ---------------------------------------------------------------------------


def compute_adaptive_threshold(
    values: list[float],
    target_confidence: float = 0.95,
) -> float | None:
    """Compute data-driven convergence threshold.

    Uses percentile-based method: convergence = improvement < X-th percentile of improvements

    Args:
        values: KPI time series
        target_confidence: Confidence level (0-1)

    Returns:
        Adaptive threshold or None if insufficient data
    """
    if len(values) < 5:
        return None

    # Compute absolute improvements
    improvements = []
    for i in range(1, len(values)):
        imp = abs(values[i] - values[i-1])
        improvements.append(imp)

    if not improvements:
        return None

    # Filter out near-zero improvements (noise threshold)
    non_zero_improvements = [imp for imp in improvements if imp > 1e-6]

    # If all improvements are ~0, return small default threshold
    if not non_zero_improvements:
        return 1e-3

    # Sort non-zero improvements
    sorted_imps = sorted(non_zero_improvements)

    # Find percentile corresponding to target confidence
    # Lower percentile = more stringent threshold
    percentile_idx = int((1.0 - target_confidence) * len(sorted_imps))
    percentile_idx = max(0, min(len(sorted_imps) - 1, percentile_idx))

    threshold = sorted_imps[percentile_idx]

    # Ensure threshold is meaningful (not too small)
    # Use median of non-zero improvements as fallback if percentile is too low
    if threshold < 1e-6:
        median_idx = len(sorted_imps) // 2
        threshold = sorted_imps[median_idx]

    return threshold


# ---------------------------------------------------------------------------
# Convergence prediction
# ---------------------------------------------------------------------------


def estimate_convergence_round(
    values: list[float],
    target: float | None = None,
    maximize: bool = True,
) -> int | None:
    """Estimate round number when convergence will be reached.

    Uses linear extrapolation of recent trend.

    Args:
        values: KPI time series
        target: Target KPI value (optional)
        maximize: Whether higher is better

    Returns:
        Estimated round number or None if unpredictable
    """
    if len(values) < 3:
        return None

    # Use recent trend (last 5 observations)
    recent = values[-5:] if len(values) >= 5 else values

    from app.services.convergence import _linear_regression_slope, _mean

    slope = _linear_regression_slope(recent)

    # If slope is near zero or wrong direction, can't predict
    if abs(slope) < 1e-6:
        return None

    current_val = values[-1]

    # If no target, estimate based on slope decay
    if target is None:
        # Assume convergence when slope < 1% of current improvement rate
        if abs(slope) < 0.01 * abs(current_val - values[0]) / len(values):
            return len(values)  # Already converged

        # Extrapolate: how many rounds until slope decays?
        # Simple model: slope decays exponentially with decay rate 0.9
        decay_rate = 0.9
        rounds_to_flat = math.log(0.01) / math.log(decay_rate)
        return len(values) + int(rounds_to_flat)

    # With target: extrapolate linearly
    gap = target - current_val

    # Check if moving in right direction
    if maximize and slope < 0:
        return None  # Moving away from target
    if not maximize and slope > 0:
        return None

    # Estimate rounds needed
    if abs(slope) < 1e-10:
        return None

    rounds_needed = abs(gap / slope)

    # Clamp to reasonable range (max 100 rounds)
    if rounds_needed > 100:
        return None

    return len(values) + int(math.ceil(rounds_needed))


# ---------------------------------------------------------------------------
# Main enhanced detector
# ---------------------------------------------------------------------------


def detect_convergence_enhanced(
    values: list[float],
    config: ConvergenceConfig | None = None,
    maximize: bool = True,
    target: float | None = None,
) -> EnhancedConvergenceStatus:
    """Enhanced convergence detection with pattern recognition.

    Args:
        values: KPI time series
        config: Detection configuration
        maximize: Whether higher is better
        target: Optional target KPI value

    Returns:
        EnhancedConvergenceStatus with comprehensive analysis
    """
    # Basic detection
    basic_status = basic_detect_convergence(values, config=config, maximize=maximize)

    # Oscillation detection
    oscillation = detect_oscillation(values)

    # Noise characterization
    noise = characterize_noise(values)

    # Multi-scale trends
    short_term, long_term = analyze_multi_scale_trends(values, config=config, maximize=maximize)

    # Adaptive threshold
    adaptive_threshold = compute_adaptive_threshold(values)

    # Convergence prediction
    estimated_round = estimate_convergence_round(values, target=target, maximize=maximize)

    # Metadata
    metadata = {
        "n_observations": len(values),
        "current_kpi": values[-1] if values else None,
        "best_kpi": max(values) if maximize and values else (min(values) if values else None),
    }

    return EnhancedConvergenceStatus(
        basic_status=basic_status,
        oscillation=oscillation,
        noise=noise,
        short_term_trend=short_term,
        long_term_trend=long_term,
        adaptive_threshold=adaptive_threshold,
        estimated_convergence_round=estimated_round,
        analysis_metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Integration with campaign loop
# ---------------------------------------------------------------------------


def should_stop_campaign_enhanced(
    enhanced_status: EnhancedConvergenceStatus,
    goal_target_reached: bool,
    rounds_exhausted: bool,
    plateau_confidence_threshold: float = 0.7,
    diverging_confidence_threshold: float = 0.8,
) -> tuple[str, str]:
    """Decide whether to stop campaign using enhanced convergence analysis.

    Returns:
        (action, reason) where action is "continue" or "stop"
        and reason explains the decision
    """
    # Priority 1: Target reached
    if goal_target_reached:
        return "stop", "target_reached"

    # Priority 2: Budget exhausted
    if rounds_exhausted:
        return "stop", "budget_exhausted"

    basic = enhanced_status.basic_status

    # Priority 3: Diverging with high confidence
    if basic.status == "diverging" and basic.confidence > diverging_confidence_threshold:
        # But check if it's just noise
        if enhanced_status.noise.is_noisy and enhanced_status.noise.confidence > 0.7:
            return "continue", "diverging_but_noisy (continue)"
        return "stop", "diverging"

    # Priority 4: Plateau with high confidence
    if basic.status == "plateau" and basic.confidence > plateau_confidence_threshold:
        # Check for oscillation (might break out of plateau)
        if enhanced_status.oscillation.detected and enhanced_status.oscillation.confidence > 0.7:
            return "continue", "plateau_with_oscillation (continue)"

        # Check short-term trend (recent improvement?)
        if enhanced_status.short_term_trend == "improving":
            return "continue", "long_term_plateau_but_short_term_improving (continue)"

        return "stop", "converged"

    # Default: continue
    return "continue", "optimization_ongoing"
