"""Structured scientist feedback → rubric adaptation (Phase B / B2).

Scientists judge decisions in ways a fixed reward cannot capture ("this didn't
improve the objective but it tested a key mechanism"; "this looked safe but the
prep window is too narrow"). Rather than let an LLM free-interpret such notes,
B2 constrains them to a small taxonomy, maps each to the reward signal it bears
on, and folds a batch of feedback into a derived ``v0.3_feedback_adaptive``
rubric. This closes the inner loop: human supervision reshapes what counts as a
good decision (growloop.md §2).

    ScientistFeedback[] ──► apply_feedback_to_rubric(base) ──► adapted Rubric

Feedback raises the *attention* (weight) the evaluator pays to the affected
signal, scaled by the scientist's confidence — whether the note is positive or
negative, the lesson is "weigh this signal more here".
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.services.rubric import Rubric

__all__ = [
    "FeedbackType",
    "ScientistFeedback",
    "FEEDBACK_SIGNAL_MAP",
    "apply_feedback_to_rubric",
]

FeedbackType = Literal[
    "mechanistic_value",
    "safety_concern",
    "feasibility_concern",
    "novelty_value",
    "validation_need",
    "proxy_mismatch",
    "resource_cost_concern",
]

# Which reward signal each feedback type bears on. ``None`` means the taxonomy
# entry has no current signal correspondence (recorded, but no weight change)
# — a placeholder for signals that arrive with later phases (e.g. resource cost).
FEEDBACK_SIGNAL_MAP: dict[str, str | None] = {
    "mechanistic_value": "context",
    "safety_concern": "safety",
    "feasibility_concern": "failure",
    "novelty_value": "context",
    "validation_need": "validation",
    "proxy_mismatch": "proxy_gap",
    "resource_cost_concern": None,
}

# How strongly one unit of confidence shifts a signal's weight.
_ATTENTION_STEP = 0.5


class ScientistFeedback(BaseModel):
    """One structured scientist judgment about a decision."""

    feedback_type: FeedbackType
    decision_quality_signal: Literal["positive", "negative"]
    confidence: float = Field(ge=0.0, le=1.0)
    affected_metric: str | None = None
    note: str | None = None


def apply_feedback_to_rubric(
    base: Rubric, feedback: list[ScientistFeedback]
) -> Rubric:
    """Fold scientist feedback into a derived ``v0.3_feedback_adaptive`` rubric.

    Each feedback item raises the weight of its mapped signal by
    ``confidence * step``. Feedback whose type has no signal correspondence is
    ignored for weighting (still meaningful upstream as a labelled record).
    """
    weights = dict(base.weights)
    for item in feedback:
        signal = FEEDBACK_SIGNAL_MAP.get(item.feedback_type)
        if signal is None:
            continue
        current = weights.get(signal, 1.0)
        weights[signal] = round(current + item.confidence * _ATTENTION_STEP, 10)
    return Rubric(version="v0.3_feedback_adaptive", weights=weights)
