"""Compatibility re-export for failure-region utilities."""
from __future__ import annotations

from app.services.failure_region import (
    FailureRegionModel,
    avoid_failure_region,
    build_feasibility_observations,
    failure_outcome_constraint,
    filter_failure_prone,
)

__all__ = [
    "FailureRegionModel",
    "avoid_failure_region",
    "build_feasibility_observations",
    "failure_outcome_constraint",
    "filter_failure_prone",
]
