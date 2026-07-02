"""Objective stack models for contextual campaign decision assessment.

These models are intentionally independent from the dynamic strategy selector.
They describe how optimization metrics relate to functional scientific
performance without changing live campaign behavior.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

__all__ = [
    "MetricNode",
    "ObjectiveDirection",
    "ObjectiveMetricLevel",
    "ObjectiveStack",
    "ProxyGapAssessment",
    "ProxyGapLevel",
]


class ObjectiveMetricLevel(StrEnum):
    """Functional level represented by an objective metric."""

    RAW_MEASUREMENT = "raw_measurement"
    MATERIAL_PROPERTY = "material_property"
    FUNCTIONAL_PROXY = "functional_proxy"
    DEVICE_PERFORMANCE = "device_performance"
    CAMPAIGN_GOAL = "campaign_goal"


class ObjectiveDirection(StrEnum):
    """Optimization direction for a metric."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    TARGET = "target"


class MetricNode(BaseModel):
    """One metric in an objective stack."""

    name: str
    level: ObjectiveMetricLevel
    direction: ObjectiveDirection
    weight: float = Field(default=1.0, ge=0.0)
    target_value: float | None = None
    current_value: float | None = None
    uncertainty: float | None = None
    measurement_source: str | None = None
    parent_metrics: list[str] = Field(default_factory=list)
    evidence_sources: list[str] = Field(default_factory=list)
    proxy_risk: float = Field(default=0.5, ge=0.0, le=1.0)
    functional_relevance: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ObjectiveStack(BaseModel):
    """Campaign objective hierarchy used for proxy-gap assessment."""

    campaign_goal: str
    metrics: list[MetricNode] = Field(default_factory=list)
    active_metric_names: list[str] = Field(default_factory=list)
    validation_metric_names: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def metric_by_name(self, name: str) -> MetricNode | None:
        """Return the first metric with ``name`` if present."""
        for metric in self.metrics:
            if metric.name == name:
                return metric
        return None

    def active_metrics(self) -> list[MetricNode]:
        """Return active metrics in ``active_metric_names`` order."""
        return [
            metric
            for name in self.active_metric_names
            if (metric := self.metric_by_name(name)) is not None
        ]

    def validation_metrics(self) -> list[MetricNode]:
        """Return validation metrics in ``validation_metric_names`` order."""
        return [
            metric
            for name in self.validation_metric_names
            if (metric := self.metric_by_name(name)) is not None
        ]


class ProxyGapLevel(StrEnum):
    """Coarse proxy-gap severity."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class ProxyGapAssessment(BaseModel):
    """Assessment of how far active metrics are from functional performance."""

    score: float = Field(ge=0.0, le=1.0)
    level: ProxyGapLevel
    active_metric_names: list[str] = Field(default_factory=list)
    nearest_functional_metric_names: list[str] = Field(default_factory=list)
    rationale: str
    evidence: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
