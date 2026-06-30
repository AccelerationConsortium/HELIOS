from __future__ import annotations

import json
from datetime import UTC, datetime

from app.services.failure_attribution import (
    FailureAttributionCategory,
    FailureAttributionDistribution,
    FailureAttributionModel,
    attribute_failure,
)
from app.services.failure_signatures import classify_failure

_NOW = datetime(2026, 6, 29, 12, 0, 0, tzinfo=UTC)


def _sig(error: str, primitive: str = ""):
    return classify_failure(step_key="s1", primitive=primitive, error_message=error)


def test_distribution_normalizes_to_one():
    sig = _sig("no tips available", primitive="robot.pick_up_tip")

    dist = attribute_failure(sig, now=_NOW)

    total = sum(dist.probabilities.values())
    assert abs(total - 1.0) < 1e-9
    assert all(0.0 <= p <= 1.0 for p in dist.probabilities.values())
    assert set(dist.probabilities) == set(FailureAttributionCategory)


def test_dominant_category_is_argmax():
    instrument_sig = _sig("temp overshoot exceeded", primitive="heat")
    execution_sig = _sig("no tips available", primitive="robot.pick_up_tip")

    instrument_dist = attribute_failure(instrument_sig, now=_NOW)
    execution_dist = attribute_failure(execution_sig, now=_NOW)

    assert instrument_dist.dominant_category == FailureAttributionCategory.INSTRUMENT
    assert execution_dist.dominant_category == FailureAttributionCategory.EXECUTION
    # Dominant probability is the maximum of the distribution.
    assert instrument_dist.dominant_probability == max(
        instrument_dist.probabilities.values()
    )


def test_unknown_signature_falls_back_to_external_context_missing():
    sig = _sig("something nobody has a rule for")

    dist = attribute_failure(sig, now=_NOW)

    assert sig.failure_type == "unknown"
    assert dist.dominant_category == FailureAttributionCategory.EXTERNAL_CONTEXT_MISSING
    # Existing attribution confidence is preserved verbatim from the signature.
    assert dist.confidence == sig.confidence == 0.2


def test_preserves_source_likely_cause_and_confidence():
    sig = _sig("temp overshoot exceeded", primitive="heat")

    dist = attribute_failure(sig, now=_NOW)

    assert dist.source_likely_cause == sig.likely_cause
    assert dist.source_failure_type == sig.failure_type
    assert dist.confidence == sig.confidence


def test_evidence_is_populated_and_references_source():
    sig = _sig("impedance anomaly out of range", primitive="squidstat")

    dist = attribute_failure(sig, now=_NOW)

    assert dist.evidence
    joined = " ".join(dist.evidence)
    assert sig.failure_type in joined
    assert sig.likely_cause in joined


def test_input_signature_is_not_mutated():
    sig = _sig("connection lost", primitive="")
    before = sig.to_dict()

    attribute_failure(sig, now=_NOW)

    assert sig.to_dict() == before


def test_backward_compatible_round_trip_and_json_safe():
    sig = _sig("temp overshoot exceeded", primitive="heat")
    dist = attribute_failure(sig, now=_NOW)

    dumped = dist.to_dict()
    # JSON-serializable (no datetime / enum leakage).
    json.dumps(dumped)
    restored = FailureAttributionDistribution.from_dict(dumped)

    assert restored.to_dict() == dumped
    assert restored.dominant_category == dist.dominant_category
    assert restored.probabilities == dist.probabilities


def test_created_at_is_injectable_for_determinism():
    sig = _sig("temp overshoot exceeded", primitive="heat")

    first = attribute_failure(sig, now=_NOW)
    second = FailureAttributionModel().attribute(sig, now=_NOW)

    assert first.created_at == _NOW
    assert first.to_dict() == second.to_dict()


def test_import_smoke():
    import app.services.failure_attribution  # noqa: F401
