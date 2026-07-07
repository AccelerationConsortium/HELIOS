"""B2: structured scientist feedback folds into a v0.3 adaptive rubric."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.services.rubric import STATIC_RUBRIC, rescore
from app.services.scientist_feedback import (
    ScientistFeedback,
    apply_feedback_to_rubric,
)
from app.services.verifiable_reward import verify_validation


def test_feedback_raises_affected_signal_weight():
    fb = [
        ScientistFeedback(
            feedback_type="validation_need",
            decision_quality_signal="positive",
            confidence=0.8,
        )
    ]
    adapted = apply_feedback_to_rubric(STATIC_RUBRIC, fb)
    assert adapted.version == "v0.3_feedback_adaptive"
    assert adapted.weight("validation") == 1.0 + 0.8 * 0.5  # 1.4
    assert adapted.weight("execution") == 1.0  # untouched


def test_adapted_rubric_changes_rescore():
    v = [verify_validation(True)]  # +0.2
    static = rescore(v, STATIC_RUBRIC)
    fb = [
        ScientistFeedback(
            feedback_type="validation_need",
            decision_quality_signal="positive",
            confidence=1.0,
        )
    ]
    adapted = rescore(v, apply_feedback_to_rubric(STATIC_RUBRIC, fb))
    assert adapted.total > static.total  # validation weighted up


def test_feedback_without_signal_is_ignored_for_weighting():
    fb = [
        ScientistFeedback(
            feedback_type="resource_cost_concern",
            decision_quality_signal="negative",
            confidence=1.0,
        )
    ]
    adapted = apply_feedback_to_rubric(STATIC_RUBRIC, fb)
    assert adapted.weights == {}  # no signal correspondence → no weight change


def test_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        ScientistFeedback(
            feedback_type="safety_concern",
            decision_quality_signal="negative",
            confidence=1.5,
        )
