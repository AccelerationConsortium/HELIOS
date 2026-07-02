from __future__ import annotations

from app.services.objective_models import (
    MetricNode,
    ObjectiveDirection,
    ObjectiveMetricLevel,
    ObjectiveStack,
    ProxyGapLevel,
)
from app.services.objective_stack import (
    ObjectiveStackAnalyzer,
    assess_objective_proxy_gap,
)


def _metric(
    name: str,
    level: ObjectiveMetricLevel,
    *,
    weight: float = 1.0,
    proxy_risk: float = 0.5,
    functional_relevance: float = 0.5,
) -> MetricNode:
    return MetricNode(
        name=name,
        level=level,
        direction=ObjectiveDirection.MAXIMIZE,
        weight=weight,
        proxy_risk=proxy_risk,
        functional_relevance=functional_relevance,
    )


def test_empty_stack_returns_unknown_gap_and_serializes():
    stack = ObjectiveStack(campaign_goal="maximize device efficiency")

    assessment = assess_objective_proxy_gap(stack)

    assert assessment.level in {ProxyGapLevel.UNKNOWN, ProxyGapLevel.HIGH}
    assert 0.0 <= assessment.score <= 1.0
    assert assessment.model_dump(mode="json")["level"] == "unknown"
    assert '"score":1.0' in assessment.model_dump_json()


def test_device_level_active_metric_gives_low_gap():
    stack = ObjectiveStack(
        campaign_goal="maximize device efficiency",
        metrics=[
            _metric(
                "device_efficiency",
                ObjectiveMetricLevel.DEVICE_PERFORMANCE,
                proxy_risk=0.05,
                functional_relevance=0.95,
            )
        ],
        active_metric_names=["device_efficiency"],
    )

    assessment = ObjectiveStackAnalyzer().assess_proxy_gap(stack)

    assert assessment.level == ProxyGapLevel.LOW
    assert assessment.nearest_functional_metric_names == ["device_efficiency"]


def test_raw_measurement_only_active_metric_gives_high_gap():
    stack = ObjectiveStack(
        campaign_goal="maximize device efficiency",
        metrics=[
            _metric(
                "raw_peak_area",
                ObjectiveMetricLevel.RAW_MEASUREMENT,
                proxy_risk=0.95,
                functional_relevance=0.05,
            )
        ],
        active_metric_names=["raw_peak_area"],
    )

    assessment = assess_objective_proxy_gap(stack)

    assert assessment.level == ProxyGapLevel.HIGH
    assert assessment.score >= 0.8


def test_functional_proxy_gives_medium_gap():
    stack = ObjectiveStack(
        campaign_goal="maximize device efficiency",
        metrics=[
            _metric(
                "predicted_lifetime",
                ObjectiveMetricLevel.FUNCTIONAL_PROXY,
                proxy_risk=0.5,
                functional_relevance=0.5,
            )
        ],
        active_metric_names=["predicted_lifetime"],
    )

    assessment = assess_objective_proxy_gap(stack)

    assert assessment.level == ProxyGapLevel.MEDIUM


def test_weighted_average_respects_metric_weights():
    raw_metric = _metric(
        "raw_peak_area",
        ObjectiveMetricLevel.RAW_MEASUREMENT,
        proxy_risk=0.95,
        functional_relevance=0.05,
    )
    device_metric = _metric(
        "device_efficiency",
        ObjectiveMetricLevel.DEVICE_PERFORMANCE,
        proxy_risk=0.05,
        functional_relevance=0.95,
    )

    device_weighted = assess_objective_proxy_gap(
        ObjectiveStack(
            campaign_goal="maximize device efficiency",
            metrics=[
                raw_metric.model_copy(update={"weight": 1.0}),
                device_metric.model_copy(update={"weight": 9.0}),
            ],
            active_metric_names=["raw_peak_area", "device_efficiency"],
        )
    )
    raw_weighted = assess_objective_proxy_gap(
        ObjectiveStack(
            campaign_goal="maximize device efficiency",
            metrics=[
                raw_metric.model_copy(update={"weight": 9.0}),
                device_metric.model_copy(update={"weight": 1.0}),
            ],
            active_metric_names=["raw_peak_area", "device_efficiency"],
        )
    )

    assert device_weighted.score < raw_weighted.score


def test_active_and_validation_metric_helpers_work():
    stack = ObjectiveStack(
        campaign_goal="maximize device efficiency",
        metrics=[
            _metric("raw_peak_area", ObjectiveMetricLevel.RAW_MEASUREMENT),
            _metric("device_efficiency", ObjectiveMetricLevel.DEVICE_PERFORMANCE),
            _metric("stability", ObjectiveMetricLevel.FUNCTIONAL_PROXY),
        ],
        active_metric_names=["device_efficiency", "missing"],
        validation_metric_names=["stability", "raw_peak_area"],
    )

    assert stack.metric_by_name("device_efficiency").level == (
        ObjectiveMetricLevel.DEVICE_PERFORMANCE
    )
    assert [metric.name for metric in stack.active_metrics()] == ["device_efficiency"]
    assert [metric.name for metric in stack.validation_metrics()] == [
        "stability",
        "raw_peak_area",
    ]


def test_inputs_are_not_mutated():
    stack = ObjectiveStack(
        campaign_goal="maximize device efficiency",
        metrics=[
            MetricNode(
                name="device_efficiency",
                level=ObjectiveMetricLevel.DEVICE_PERFORMANCE,
                direction=ObjectiveDirection.MAXIMIZE,
                evidence_sources=["lab-measurement"],
                metadata={"source": {"name": "initial"}},
                proxy_risk=0.1,
                functional_relevance=0.9,
            )
        ],
        active_metric_names=["device_efficiency"],
        metadata={"campaign": {"name": "initial"}},
    )
    before = stack.model_dump(mode="json")

    assessment = assess_objective_proxy_gap(stack)
    assessment.metadata["changed"] = True
    assessment.evidence.append("changed")

    assert stack.model_dump(mode="json") == before


def test_json_serialization():
    stack = ObjectiveStack(
        campaign_goal="maximize device efficiency",
        metrics=[
            _metric(
                "device_efficiency",
                ObjectiveMetricLevel.DEVICE_PERFORMANCE,
                proxy_risk=0.1,
                functional_relevance=0.9,
            )
        ],
        active_metric_names=["device_efficiency"],
        validation_metric_names=["device_efficiency"],
    )
    assessment = assess_objective_proxy_gap(stack)

    dumped_stack = stack.model_dump(mode="json")
    dumped_assessment = assessment.model_dump(mode="json")
    stack_json = stack.model_dump_json()
    assessment_json = assessment.model_dump_json()

    assert dumped_stack["metrics"][0]["level"] == "device_performance"
    assert dumped_assessment["level"] == "low"
    assert '"campaign_goal":"maximize device efficiency"' in stack_json
    assert '"level":"low"' in assessment_json


def test_import_smoke():
    import app.services.decision_layer  # noqa: F401
    import app.services.objective_models  # noqa: F401
    import app.services.objective_stack  # noqa: F401
    import app.services.strategy_selector  # noqa: F401
