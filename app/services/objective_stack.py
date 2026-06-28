"""Rule-based ObjectiveStack proxy-gap assessment.

The analyzer is pure and deterministic: it does not call external services,
read databases, or mutate the input stack.
"""
from __future__ import annotations

from app.services.objective_models import (
    ObjectiveMetricLevel,
    ObjectiveStack,
    ProxyGapAssessment,
    ProxyGapLevel,
)

__all__ = ["ObjectiveStackAnalyzer", "assess_objective_proxy_gap"]


_BASE_GAP_BY_LEVEL: dict[ObjectiveMetricLevel, float] = {
    ObjectiveMetricLevel.CAMPAIGN_GOAL: 0.0,
    ObjectiveMetricLevel.DEVICE_PERFORMANCE: 0.1,
    ObjectiveMetricLevel.FUNCTIONAL_PROXY: 0.35,
    ObjectiveMetricLevel.MATERIAL_PROPERTY: 0.65,
    ObjectiveMetricLevel.RAW_MEASUREMENT: 0.8,
}


class ObjectiveStackAnalyzer:
    """Assess how directly active objectives measure functional performance."""

    def assess_proxy_gap(self, stack: ObjectiveStack) -> ProxyGapAssessment:
        """Return a deterministic proxy-gap assessment for an objective stack."""
        active_metrics = stack.active_metrics()
        nearest_functional_metric_names = [
            metric.name
            for metric in stack.metrics
            if metric.level
            in {
                ObjectiveMetricLevel.DEVICE_PERFORMANCE,
                ObjectiveMetricLevel.CAMPAIGN_GOAL,
            }
        ]

        if not stack.metrics or not stack.active_metric_names or not active_metrics:
            return ProxyGapAssessment(
                score=1.0,
                level=ProxyGapLevel.UNKNOWN,
                active_metric_names=[metric.name for metric in active_metrics],
                nearest_functional_metric_names=nearest_functional_metric_names,
                rationale=(
                    "Insufficient objective stack data to assess functional proxy gap."
                ),
                evidence=[
                    "Objective stack has no metrics, no active metric names, "
                    "or no resolvable active metrics."
                ],
            )

        scored_metrics = [
            (
                metric,
                _metric_proxy_gap_score(metric.level, metric.proxy_risk, metric.functional_relevance),
            )
            for metric in active_metrics
        ]
        total_weight = sum(metric.weight for metric, _score in scored_metrics)
        if total_weight <= 0.0:
            score = sum(score for _metric, score in scored_metrics) / len(scored_metrics)
            weighting_evidence = "Active metric weights sum to zero; used equal weighting."
        else:
            score = (
                sum(metric.weight * score for metric, score in scored_metrics)
                / total_weight
            )
            weighting_evidence = "Used active metric weights for weighted average."

        score = _clamp(score)
        level = _level_for_score(score)
        active_metric_names = [metric.name for metric in active_metrics]
        evidence = [
            (
                f"{metric.name}: level={metric.level.value}, "
                f"weight={metric.weight:.3g}, proxy_risk={metric.proxy_risk:.3g}, "
                f"functional_relevance={metric.functional_relevance:.3g}, "
                f"score={metric_score:.3g}"
            )
            for metric, metric_score in scored_metrics
        ]
        evidence.append(weighting_evidence)

        return ProxyGapAssessment(
            score=score,
            level=level,
            active_metric_names=active_metric_names,
            nearest_functional_metric_names=nearest_functional_metric_names,
            rationale=_rationale_for_level(level),
            evidence=evidence,
            metadata={
                "method": "deterministic_level_proxy_risk_functional_relevance",
                "active_metric_count": len(active_metrics),
            },
        )


def assess_objective_proxy_gap(stack: ObjectiveStack) -> ProxyGapAssessment:
    """Assess proxy gap with the default ObjectiveStackAnalyzer."""
    return ObjectiveStackAnalyzer().assess_proxy_gap(stack)


def _metric_proxy_gap_score(
    level: ObjectiveMetricLevel,
    proxy_risk: float,
    functional_relevance: float,
) -> float:
    base = _BASE_GAP_BY_LEVEL[level]
    return _clamp(base + 0.25 * proxy_risk - 0.25 * functional_relevance)


def _level_for_score(score: float) -> ProxyGapLevel:
    if score < 0.25:
        return ProxyGapLevel.LOW
    if score < 0.6:
        return ProxyGapLevel.MEDIUM
    return ProxyGapLevel.HIGH


def _rationale_for_level(level: ProxyGapLevel) -> str:
    if level == ProxyGapLevel.LOW:
        return "Active objectives are close to functional scientific performance."
    if level == ProxyGapLevel.MEDIUM:
        return "Active objectives are useful proxies but still carry functional gap risk."
    if level == ProxyGapLevel.HIGH:
        return "Active objectives are distant from functional scientific performance."
    return "Objective stack data is insufficient to assess proxy gap."


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
