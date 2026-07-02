"""Distributional failure attribution — additive layer over FailureSignature.

The rule-based classifier in ``failure_signatures`` produces a single
``likely_cause`` string and a scalar ``confidence``. This module adds a
*distributional* view on top of it: a deterministic probability distribution
over the scientific attribution categories from the planning runtime design
(execution, instrument, sample, objective mismatch, model, external context
missing).

This layer is strictly additive and shadow-only:

* It does not modify ``FailureSignature`` or any classifier behavior.
* The original ``likely_cause`` and scalar ``confidence`` are preserved
  verbatim on the distribution (``source_likely_cause`` / ``confidence``), so
  existing consumers that read ``attribution_confidence`` are unaffected.
* It performs no routing, candidate selection, or strategy changes.
* It does not introduce instrument-belief state or external telemetry yet.

Every distribution carries provenance (source signature fields, evidence, and
an injectable ``created_at``) and is replayable via ``to_dict`` / ``from_dict``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from app.services.failure_signatures import FailureSignature

__all__ = [
    "FailureAttributionCategory",
    "FailureAttributionDistribution",
    "FailureAttributionModel",
    "attribute_failure",
]


class FailureAttributionCategory(StrEnum):
    """Scientific attribution categories for an observed failure."""

    EXECUTION = "execution_failure"
    INSTRUMENT = "instrument_failure"
    SAMPLE = "sample_failure"
    OBJECTIVE_MISMATCH = "objective_mismatch"
    MODEL = "model_failure"
    EXTERNAL_CONTEXT_MISSING = "external_context_missing"


#: Deterministic primary category by failure_type. Unmapped types fall back to
#: EXTERNAL_CONTEXT_MISSING (we lack enough signal to attribute confidently).
#: These assignments are initial domain heuristics, intentionally additive.
_PRIMARY_BY_FAILURE_TYPE: dict[str, FailureAttributionCategory] = {
    "volume_delivery_failure": FailureAttributionCategory.EXECUTION,
    "tip_shortage": FailureAttributionCategory.EXECUTION,
    "deck_conflict": FailureAttributionCategory.EXECUTION,
    "file_missing": FailureAttributionCategory.EXECUTION,
    "protocol_sequence_error": FailureAttributionCategory.EXECUTION,
    "safety_limit_exceeded": FailureAttributionCategory.EXECUTION,
    "temperature_deviation": FailureAttributionCategory.INSTRUMENT,
    "temperature_overshoot": FailureAttributionCategory.INSTRUMENT,
    "impedance_anomaly": FailureAttributionCategory.INSTRUMENT,
    "electrode_degradation": FailureAttributionCategory.INSTRUMENT,
    "instrument_disconnection": FailureAttributionCategory.INSTRUMENT,
    "instrument_timeout": FailureAttributionCategory.INSTRUMENT,
    "sensor_drift": FailureAttributionCategory.INSTRUMENT,
    "electrolyte_contamination": FailureAttributionCategory.SAMPLE,
    "liquid_insufficient": FailureAttributionCategory.SAMPLE,
    "unknown": FailureAttributionCategory.EXTERNAL_CONTEXT_MISSING,
}

#: Cause-level overrides take precedence over the failure_type mapping when the
#: root cause carries a stronger attribution signal than the failure type.
_PRIMARY_BY_LIKELY_CAUSE: dict[str, FailureAttributionCategory] = {
    "parameter_out_of_range": FailureAttributionCategory.OBJECTIVE_MISMATCH,
}

#: Uniform base mass on every category before adding the confidence-scaled peak.
_BASE_WEIGHT = 1.0
#: Constant bump so the primary category wins even at zero confidence.
_PRIMARY_BONUS = 0.5
#: How sharply classifier confidence concentrates mass on the primary category.
_CONFIDENCE_SCALE = 20.0


@dataclass(frozen=True)
class FailureAttributionDistribution:
    """A normalized probability distribution over attribution categories.

    ``confidence`` and ``source_likely_cause`` are preserved verbatim from the
    originating ``FailureSignature`` for backward compatibility.
    """

    probabilities: dict[FailureAttributionCategory, float]
    dominant_category: FailureAttributionCategory
    dominant_probability: float
    confidence: float
    source_failure_type: str | None
    source_likely_cause: str | None
    step_key: str | None
    primitive: str | None
    evidence: tuple[str, ...]
    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "probabilities": {
                category.value: probability
                for category, probability in self.probabilities.items()
            },
            "dominant_category": self.dominant_category.value,
            "dominant_probability": self.dominant_probability,
            "confidence": self.confidence,
            "source_failure_type": self.source_failure_type,
            "source_likely_cause": self.source_likely_cause,
            "step_key": self.step_key,
            "primitive": self.primitive,
            "evidence": list(self.evidence),
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FailureAttributionDistribution:
        """Deserialize from a dict produced by ``to_dict``."""
        probabilities = {
            FailureAttributionCategory(key): float(value)
            for key, value in dict(data.get("probabilities", {})).items()
        }
        return cls(
            probabilities=probabilities,
            dominant_category=FailureAttributionCategory(data["dominant_category"]),
            dominant_probability=float(data.get("dominant_probability", 0.0)),
            confidence=float(data.get("confidence", 0.0)),
            source_failure_type=_opt_str(data.get("source_failure_type")),
            source_likely_cause=_opt_str(data.get("source_likely_cause")),
            step_key=_opt_str(data.get("step_key")),
            primitive=_opt_str(data.get("primitive")),
            evidence=tuple(str(item) for item in data.get("evidence", ())),
            created_at=datetime.fromisoformat(data["created_at"]),
        )


class FailureAttributionModel:
    """Deterministically turn a FailureSignature into an attribution distribution."""

    def attribute(
        self,
        signature: FailureSignature,
        *,
        now: datetime | None = None,
    ) -> FailureAttributionDistribution:
        timestamp = now or datetime.now(UTC)
        primary = _primary_category(signature)
        confidence = _clamp_unit(signature.confidence)

        weights = {category: _BASE_WEIGHT for category in FailureAttributionCategory}
        weights[primary] += _PRIMARY_BONUS + _CONFIDENCE_SCALE * confidence
        total = sum(weights.values())
        probabilities = {
            category: _round(weight / total)
            for category, weight in weights.items()
        }

        dominant_category, dominant_probability = _argmax(probabilities)
        evidence = _build_evidence(signature, primary, confidence)

        return FailureAttributionDistribution(
            probabilities=probabilities,
            dominant_category=dominant_category,
            dominant_probability=dominant_probability,
            confidence=signature.confidence,
            source_failure_type=signature.failure_type,
            source_likely_cause=signature.likely_cause,
            step_key=signature.step_key or None,
            primitive=signature.primitive or None,
            evidence=evidence,
            created_at=timestamp,
        )


def attribute_failure(
    signature: FailureSignature,
    *,
    now: datetime | None = None,
) -> FailureAttributionDistribution:
    """Attribute a failure with the default FailureAttributionModel."""
    return FailureAttributionModel().attribute(signature, now=now)


def _primary_category(signature: FailureSignature) -> FailureAttributionCategory:
    cause_override = _PRIMARY_BY_LIKELY_CAUSE.get(signature.likely_cause)
    if cause_override is not None:
        return cause_override
    return _PRIMARY_BY_FAILURE_TYPE.get(
        signature.failure_type, FailureAttributionCategory.EXTERNAL_CONTEXT_MISSING
    )


def _argmax(
    probabilities: dict[FailureAttributionCategory, float],
) -> tuple[FailureAttributionCategory, float]:
    # Deterministic: iterate enum declaration order so ties resolve stably.
    best_category = next(iter(FailureAttributionCategory))
    best_probability = probabilities[best_category]
    for category in FailureAttributionCategory:
        probability = probabilities[category]
        if probability > best_probability:
            best_category = category
            best_probability = probability
    return best_category, best_probability


def _build_evidence(
    signature: FailureSignature,
    primary: FailureAttributionCategory,
    confidence: float,
) -> tuple[str, ...]:
    return (
        f"source failure_type={signature.failure_type}",
        f"source likely_cause={signature.likely_cause}",
        f"primary attribution={primary.value} (from "
        f"{'likely_cause' if signature.likely_cause in _PRIMARY_BY_LIKELY_CAUSE else 'failure_type'})",
        f"classifier confidence={confidence:.3g} scaled the primary peak",
    )


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def _round(value: float) -> float:
    return round(value, 10)
