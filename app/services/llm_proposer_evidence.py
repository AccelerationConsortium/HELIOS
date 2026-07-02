"""Offline evidence report for the LLM candidate proposer (shadow).

Aggregates recorded ``LLMProposerShadow`` artifacts against the rounds' actually
selected candidates to answer the calibration questions:

1. validity rate — how often LLM proposals survive the validation gate,
2. overlap with the current selection,
3. novel-but-valid candidates the current strategy did not pick,
4. what rejections are mostly caused by (schema / failure_zone / safety),
5. whether the failure-zone / safety rejectors actually fire,
6. whether the LLM beats a random baseline (else it is an expensive random
   point generator) — via ``random_overlap_rate`` vs the LLM's overlap.

Pure and deterministic; read-only aggregation over shadow artifacts.
"""
from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.services.candidate_gen import ParameterSpace
from app.services.llm_candidate_proposer import (
    LLMProposerShadow,
    count_point_overlaps,
)

__all__ = ["LLMProposerEvidence", "build_llm_proposer_evidence"]

_REJECTION_CATEGORIES = ("schema", "failure_zone", "safety")


class LLMProposerEvidence(BaseModel):
    """Aggregate evidence about LLM proposer behavior across rounds."""

    rounds: int = Field(ge=0)
    proposed: int = Field(ge=0)
    accepted: int = Field(ge=0)
    validity_rate: float = Field(ge=0.0, le=1.0)
    rejection_histogram: dict[str, int] = Field(default_factory=dict)
    rejector_fired: dict[str, int] = Field(default_factory=dict)
    selected_total: int = Field(ge=0)
    overlap_count: int = Field(ge=0)
    overlap_rate: float = Field(ge=0.0, le=1.0)
    novel_valid: int = Field(ge=0)
    random_overlap_rate: float | None = None
    notes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _created_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("created_at must be timezone-aware")
        return value


def build_llm_proposer_evidence(
    *,
    shadows: list[LLMProposerShadow],
    selections_by_round: list[list[dict[str, Any]]],
    space: ParameterSpace,
    random_points_by_round: list[list[dict[str, Any]]] | None = None,
    now: datetime | None = None,
) -> LLMProposerEvidence:
    """Aggregate shadow artifacts + selections into an evidence report."""
    timestamp = now or datetime.now(UTC)

    proposed = accepted = selected_total = overlap_count = novel_valid = 0
    rejection_histogram: Counter[str] = Counter()
    rejector_fired: Counter[str] = Counter()

    for index, shadow in enumerate(shadows):
        validation = shadow.validation
        selection = selections_by_round[index] if index < len(selections_by_round) else []
        accepted_points = validation.accepted_points

        proposed += len(validation.validations)
        accepted += len(accepted_points)
        selected_total += len(selection)

        for point_validation in validation.validations:
            for reason in point_validation.rejections:
                category = _classify_rejection(reason)
                rejection_histogram[category] += 1
                if category in _REJECTION_CATEGORIES:
                    rejector_fired[category] += 1

        matched = count_point_overlaps(accepted_points, selection, space=space)
        overlap_count += matched
        novel_valid += max(0, len(accepted_points) - matched)

    random_overlap_rate = _random_overlap_rate(
        random_points_by_round, selections_by_round, space
    )

    return LLMProposerEvidence(
        rounds=len(shadows),
        proposed=proposed,
        accepted=accepted,
        validity_rate=(accepted / proposed) if proposed else 0.0,
        rejection_histogram=dict(rejection_histogram),
        rejector_fired={category: rejector_fired.get(category, 0) for category in _REJECTION_CATEGORIES},
        selected_total=selected_total,
        overlap_count=overlap_count,
        overlap_rate=(overlap_count / accepted) if accepted else 0.0,
        novel_valid=novel_valid,
        random_overlap_rate=random_overlap_rate,
        notes=_notes(accepted, overlap_count, novel_valid, random_overlap_rate),
        created_at=timestamp,
    )


def _classify_rejection(reason: str) -> str:
    prefix = reason.split(":", 1)[0].strip().lower()
    return prefix if prefix in _REJECTION_CATEGORIES else "other"


def _random_overlap_rate(
    random_points_by_round: list[list[dict[str, Any]]] | None,
    selections_by_round: list[list[dict[str, Any]]],
    space: ParameterSpace,
) -> float | None:
    if not random_points_by_round:
        return None
    total = matched = 0
    for index, points in enumerate(random_points_by_round):
        selection = selections_by_round[index] if index < len(selections_by_round) else []
        total += len(points)
        matched += count_point_overlaps(points, selection, space=space)
    return (matched / total) if total else None


def _notes(
    accepted: int,
    overlap_count: int,
    novel_valid: int,
    random_overlap_rate: float | None,
) -> list[str]:
    notes: list[str] = []
    if accepted == 0:
        notes.append("No LLM proposals survived the gate; nothing to compare.")
        return notes
    llm_overlap_rate = overlap_count / accepted
    if random_overlap_rate is not None and llm_overlap_rate <= random_overlap_rate:
        notes.append(
            "LLM overlap does not exceed the random baseline; check whether it is "
            "beating an expensive random generator."
        )
    if novel_valid > 0:
        notes.append(
            f"{novel_valid} valid candidate(s) proposed that the current selection "
            "did not include."
        )
    return notes
