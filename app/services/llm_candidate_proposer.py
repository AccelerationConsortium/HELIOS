"""LLM candidate proposer (shadow-first).

An LLM that proposes candidate parameter points — one more plug-in *proposer*,
like a Nexus backend, with no special authority. HELIOS's existing backend
credit / bandit decides whether to adopt it (later increments); this module only
produces proposals and runs them through a deterministic validation gate.

Design (see docs/plans/2026-07-02-llm-candidate-proposer-design.md):
- Shadow-first: proposals are advisory; nothing here changes candidate selection.
- Trigger: invoked only on plateau / high epistemic uncertainty (cost control).
- Validation gate (deterministic, anti-hallucination): schema / space legality
  and failure-zone rejection are built in; safety / hard-constraint checks plug
  in as `extra_rejectors` (wired in a later increment).
- Fail-open: any LLM/parse error yields an empty proposal; the classical path is
  unaffected.

The LLM provider is injectable so the module is fully testable with a mock; the
per-point value estimates a surrogate would supply are NOT produced here.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.services.candidate_gen import ParameterSpace
from app.services.llm_gateway import (
    LLMError,
    LLMMessage,
    LLMProvider,
    get_llm_provider,
)

__all__ = [
    "LLMCandidateProposer",
    "LLMProposal",
    "LLMProposedPoint",
    "LLMProposerShadow",
    "LLMSelectionComparison",
    "PointValidation",
    "ValidatedProposal",
    "compare_llm_proposal_to_selection",
    "parse_llm_proposer_shadow_log_line",
    "should_invoke_llm_proposer",
    "validate_proposal",
]

#: Epistemic uncertainty at or above this triggers the proposer.
_UNCERTAINTY_THRESHOLD = 0.7
#: Normalized distance at or below this to a known failed point => rejected.
_FAILURE_ZONE_TOL = 0.05

#: A rejector maps a candidate params dict to a rejection reason, or None to pass.
Rejector = Callable[[dict[str, Any]], str | None]


class LLMProposedPoint(BaseModel):
    """A single candidate point proposed by the LLM."""

    params: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class LLMProposal(BaseModel):
    """A shadow-only batch of LLM-proposed candidate points."""

    campaign_id: str
    round_index: int = Field(ge=0)
    points: list[LLMProposedPoint] = Field(default_factory=list)
    objective_kpi: str | None = None
    direction: str | None = None
    trigger_reason: str
    model: str | None = None
    shadow_only: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _created_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("created_at must be timezone-aware")
        return value


class PointValidation(BaseModel):
    """Validation outcome for one proposed point."""

    params: dict[str, Any] = Field(default_factory=dict)
    accepted: bool
    rejections: list[str] = Field(default_factory=list)


class ValidatedProposal(BaseModel):
    """Deterministic validation of an LLM proposal (advisory)."""

    campaign_id: str
    round_index: int = Field(ge=0)
    validations: list[PointValidation] = Field(default_factory=list)
    accepted_points: list[dict[str, Any]] = Field(default_factory=list)
    shadow_only: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _created_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("created_at must be timezone-aware")
        return value


class LLMProposerShadow(BaseModel):
    """Shadow artifact bundling a proposal with its deterministic validation."""

    proposal: LLMProposal
    validation: ValidatedProposal
    shadow_only: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _created_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("created_at must be timezone-aware")
        return value


class LLMSelectionComparison(BaseModel):
    """Offline comparison of LLM proposals vs what HELIOS actually selected."""

    campaign_id: str
    round_index: int = Field(ge=0)
    n_proposed: int = Field(ge=0)
    n_accepted: int = Field(ge=0)
    validity_rate: float = Field(ge=0.0, le=1.0)
    n_selected: int = Field(ge=0)
    overlap_count: int = Field(ge=0)
    overlap_rate: float = Field(ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _created_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("created_at must be timezone-aware")
        return value


def compare_llm_proposal_to_selection(
    validation: ValidatedProposal,
    *,
    selected_candidates: list[dict[str, Any]],
    space: ParameterSpace,
    tol: float = _FAILURE_ZONE_TOL,
    now: datetime | None = None,
) -> LLMSelectionComparison:
    """Compare validated LLM points to the round's actually selected candidates.

    Advisory / offline only. ``overlap`` counts selected candidates that have an
    accepted LLM point within ``tol`` normalized distance.
    """
    timestamp = now or datetime.now(UTC)
    n_proposed = len(validation.validations)
    accepted = validation.accepted_points
    n_accepted = len(accepted)
    n_selected = len(selected_candidates)

    overlap = sum(
        1
        for chosen in selected_candidates
        if any(_normalized_distance(point, chosen, space) <= tol for point in accepted)
    )

    return LLMSelectionComparison(
        campaign_id=validation.campaign_id,
        round_index=validation.round_index,
        n_proposed=n_proposed,
        n_accepted=n_accepted,
        validity_rate=(n_accepted / n_proposed) if n_proposed else 0.0,
        n_selected=n_selected,
        overlap_count=overlap,
        overlap_rate=(overlap / n_selected) if n_selected else 0.0,
        created_at=timestamp,
    )


def parse_llm_proposer_shadow_log_line(line: str) -> LLMProposerShadow | None:
    """Reconstruct an LLMProposerShadow from its shadow log line, or None."""
    marker = "llm_proposer_shadow "
    index = line.find(marker)
    if index < 0:
        return None
    payload = line[index + len(marker):].strip()
    try:
        return LLMProposerShadow.model_validate(json.loads(payload))
    except Exception:
        return None


def should_invoke_llm_proposer(
    *,
    plateau: bool,
    epistemic_uncertainty: float | None,
    uncertainty_threshold: float = _UNCERTAINTY_THRESHOLD,
) -> bool:
    """Trigger the proposer only when it is most needed (BORA-style)."""
    if plateau:
        return True
    if epistemic_uncertainty is not None and epistemic_uncertainty >= uncertainty_threshold:
        return True
    return False


class LLMCandidateProposer:
    """Produce shadow-only candidate proposals from an injectable LLM provider."""

    def __init__(self, provider: LLMProvider | None = None) -> None:
        self._provider = provider

    async def propose(
        self,
        *,
        campaign_id: str,
        round_index: int,
        space: ParameterSpace,
        objective_kpi: str,
        direction: str,
        best_so_far: dict[str, Any] | None = None,
        recent_observations: list[dict[str, Any]] | None = None,
        trigger_reason: str,
        model: str | None = None,
        now: datetime | None = None,
    ) -> LLMProposal:
        timestamp = now or datetime.now(UTC)
        provider = self._provider or get_llm_provider()
        system = _system_prompt(objective_kpi, direction)
        user = _user_prompt(space, best_so_far, recent_observations, trigger_reason)

        try:
            response = await provider.complete(
                messages=[LLMMessage(role="user", content=user)],
                system=system,
                model=model,
            )
            points = _parse_points(response.content)
            return LLMProposal(
                campaign_id=campaign_id,
                round_index=round_index,
                points=points,
                objective_kpi=objective_kpi,
                direction=direction,
                trigger_reason=trigger_reason,
                model=response.model,
                created_at=timestamp,
            )
        except (LLMError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            return LLMProposal(
                campaign_id=campaign_id,
                round_index=round_index,
                points=[],
                objective_kpi=objective_kpi,
                direction=direction,
                trigger_reason=trigger_reason,
                model=model,
                created_at=timestamp,
                metadata={"failed": True, "error": type(exc).__name__},
            )


def validate_proposal(
    proposal: LLMProposal,
    *,
    space: ParameterSpace,
    failure_zones: list[dict[str, Any]] | None = None,
    failure_zone_tol: float = _FAILURE_ZONE_TOL,
    extra_rejectors: list[Rejector] | None = None,
    now: datetime | None = None,
) -> ValidatedProposal:
    """Run each proposed point through the deterministic validation gate."""
    timestamp = now or datetime.now(UTC)
    zones = failure_zones or []
    rejectors = extra_rejectors or []

    validations: list[PointValidation] = []
    for point in proposal.points:
        rejections = _validate_point(point.params, space, zones, failure_zone_tol, rejectors)
        validations.append(
            PointValidation(
                params=dict(point.params),
                accepted=not rejections,
                rejections=rejections,
            )
        )

    accepted = [v.params for v in validations if v.accepted]
    return ValidatedProposal(
        campaign_id=proposal.campaign_id,
        round_index=proposal.round_index,
        validations=validations,
        accepted_points=accepted,
        shadow_only=True,
        created_at=timestamp,
        metadata={"proposed": len(proposal.points), "accepted": len(accepted)},
    )


def _validate_point(
    params: dict[str, Any],
    space: ParameterSpace,
    failure_zones: list[dict[str, Any]],
    failure_zone_tol: float,
    rejectors: list[Rejector],
) -> list[str]:
    rejections: list[str] = []
    dim_names = {dim.param_name for dim in space.dimensions}

    for extra in set(params) - dim_names:
        rejections.append(f"schema: unknown parameter '{extra}'")

    for dim in space.dimensions:
        if dim.param_name not in params:
            rejections.append(f"schema: missing parameter '{dim.param_name}'")
            continue
        rejections.extend(_check_dim(dim, params[dim.param_name]))

    if not rejections and _within_failure_zone(params, failure_zones, space, failure_zone_tol):
        rejections.append("failure_zone: too close to a known failed point")

    for rejector in rejectors:
        reason = rejector(dict(params))
        if reason:
            rejections.append(reason)

    return rejections


def _check_dim(dim: Any, value: Any) -> list[str]:
    if dim.param_type in {"number", "integer"}:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return [f"schema: '{dim.param_name}' must be numeric"]
        if dim.min_value is not None and value < dim.min_value:
            return [f"schema: '{dim.param_name}'={value} below min {dim.min_value}"]
        if dim.max_value is not None and value > dim.max_value:
            return [f"schema: '{dim.param_name}'={value} above max {dim.max_value}"]
        return []
    if dim.param_type in {"categorical", "boolean"}:
        choices = dim.choices or ()
        if choices and value not in choices:
            return [f"schema: '{dim.param_name}'={value!r} not in choices {list(choices)}"]
        return []
    return []


def _within_failure_zone(
    params: dict[str, Any],
    failure_zones: list[dict[str, Any]],
    space: ParameterSpace,
    tol: float,
) -> bool:
    return any(
        _normalized_distance(params, zone, space) <= tol for zone in failure_zones
    )


def _normalized_distance(
    a: dict[str, Any],
    b: dict[str, Any],
    space: ParameterSpace,
) -> float:
    worst = 0.0
    for dim in space.dimensions:
        if dim.param_name not in a or dim.param_name not in b:
            continue
        va, vb = a[dim.param_name], b[dim.param_name]
        if dim.param_type in {"number", "integer"}:
            span = (dim.max_value - dim.min_value) if (
                dim.min_value is not None and dim.max_value is not None
            ) else None
            if not span:
                worst = max(worst, 0.0 if va == vb else 1.0)
            else:
                worst = max(worst, abs(float(va) - float(vb)) / span)
        else:
            worst = max(worst, 0.0 if va == vb else 1.0)
    return worst


def _system_prompt(objective_kpi: str, direction: str) -> str:
    return (
        "You are a scientific optimization assistant proposing candidate "
        f"experiments to {direction} the objective '{objective_kpi}'. "
        "Return ONLY JSON of the form "
        '{"proposals": [{"params": {..}, "reason": ".."}]}. '
        "Every params object must set exactly the given parameters, within bounds."
    )


def _user_prompt(
    space: ParameterSpace,
    best_so_far: dict[str, Any] | None,
    recent_observations: list[dict[str, Any]] | None,
    trigger_reason: str,
) -> str:
    dims = [
        {
            "param_name": dim.param_name,
            "param_type": dim.param_type,
            "min_value": dim.min_value,
            "max_value": dim.max_value,
            "choices": list(dim.choices) if dim.choices else None,
        }
        for dim in space.dimensions
    ]
    payload = {
        "trigger_reason": trigger_reason,
        "parameter_space": dims,
        "best_so_far": best_so_far,
        "recent_observations": list(recent_observations or []),
    }
    return json.dumps(payload)


def _parse_points(content: str) -> list[LLMProposedPoint]:
    data = json.loads(content)
    raw = data.get("proposals", data) if isinstance(data, dict) else data
    if not isinstance(raw, list):
        raise ValueError("LLM response did not contain a proposals list")
    points: list[LLMProposedPoint] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        params = item.get("params", {})
        if not isinstance(params, dict):
            continue
        points.append(LLMProposedPoint(params=params, reason=str(item.get("reason", ""))))
    return points
