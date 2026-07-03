"""Retrospective decision evaluation + offline policy evaluation (Phase B / B3+B4).

Both operate over the append-only ``decision_trajectories`` recorded in Phase A.
Because every trajectory carries its per-signal verifier report, a decision can
be re-scored under any rubric long after it was made — no campaign rerun.

B3 — retrospective audit (growloop.md §5): a decision that scored low at the
time (no objective gain) can be high-value in hindsight (it ruled out a
mechanism). Re-scoring stored verifications under a hindsight rubric surfaces
decisions the immediate reward under- or over-estimated.

    trajectories ──► rescore(verifications, rubric) ──► RetrospectiveRecord[]

B4 — offline policy evaluation: compare rubrics (proxy for policies) over the
accumulated trajectories. Gated on a minimum trajectory count so the comparison
harness is not built on noise (CLAUDE.md rule 15 — no premature abstraction).

Trigger for B3 is explicit (campaign-end batch or a manual endpoint), never an
implicit scheduler.
"""
from __future__ import annotations

from pydantic import BaseModel

from app.services.decision_trajectory import load_trajectories
from app.services.rubric import STATIC_RUBRIC, Rubric, rescore
from app.services.verifiable_reward import RewardVerification, round_component

__all__ = [
    "RETROSPECTIVE_DIVERGENCE_THRESHOLD",
    "MIN_TRAJECTORIES_FOR_OFFLINE_EVAL",
    "RetrospectiveRecord",
    "PolicyEvaluation",
    "OfflinePolicyEvaluation",
    "retrospective_audit",
    "offline_policy_evaluation",
]

# A decision is flagged when hindsight and immediate reward diverge by more
# than this absolute amount.
RETROSPECTIVE_DIVERGENCE_THRESHOLD = 0.1

# Don't run the policy-comparison harness until there's enough signal.
MIN_TRAJECTORIES_FOR_OFFLINE_EVAL = 50


class RetrospectiveRecord(BaseModel):
    """One decision re-scored in hindsight."""

    decision_id: str
    immediate_reward: float
    retrospective_reward: float
    delta: float
    verdict: str  # "underestimated" | "overestimated" | "stable"
    rubric_version: str


class PolicyEvaluation(BaseModel):
    """Aggregate score for one rubric (policy) over the trajectory set."""

    rubric_version: str
    mean_reward: float
    trajectory_count: int


class OfflinePolicyEvaluation(BaseModel):
    """Result of comparing rubrics offline (or why it was skipped)."""

    ran: bool
    trajectory_count: int
    reason: str | None = None
    policies: list[PolicyEvaluation] = []


def _verifications(row: dict) -> list[RewardVerification]:
    return [RewardVerification(**v) for v in row.get("verifier_report", [])]


def _verdict(delta: float) -> str:
    if delta > RETROSPECTIVE_DIVERGENCE_THRESHOLD:
        return "underestimated"
    if delta < -RETROSPECTIVE_DIVERGENCE_THRESHOLD:
        return "overestimated"
    return "stable"


def retrospective_audit(
    campaign_id: str | None = None, rubric: Rubric = STATIC_RUBRIC
) -> list[RetrospectiveRecord]:
    """Re-score stored decisions under *rubric* and flag divergence from the
    immediate reward. Explicit trigger only (campaign-end batch / manual)."""
    records: list[RetrospectiveRecord] = []
    for row in load_trajectories(campaign_id):
        immediate = row["reward"]
        retro = rescore(_verifications(row), rubric).total
        delta = round_component(retro - immediate)
        records.append(
            RetrospectiveRecord(
                decision_id=row.get("trace_id") or row["id"],
                immediate_reward=immediate,
                retrospective_reward=retro,
                delta=delta,
                verdict=_verdict(delta),
                rubric_version=rubric.version,
            )
        )
    return records


def offline_policy_evaluation(
    rubrics: list[Rubric],
    campaign_id: str | None = None,
    *,
    min_trajectories: int = MIN_TRAJECTORIES_FOR_OFFLINE_EVAL,
) -> OfflinePolicyEvaluation:
    """Compare rubrics over accumulated trajectories, gated on volume."""
    rows = load_trajectories(campaign_id)
    n = len(rows)
    if n < min_trajectories:
        return OfflinePolicyEvaluation(
            ran=False,
            trajectory_count=n,
            reason=(
                f"insufficient trajectories: {n} < {min_trajectories} "
                "(offline policy evaluation gated to avoid noise)"
            ),
        )
    verifs = [_verifications(row) for row in rows]
    policies = [
        PolicyEvaluation(
            rubric_version=rubric.version,
            mean_reward=round_component(
                sum(rescore(v, rubric).total for v in verifs) / n
            ),
            trajectory_count=n,
        )
        for rubric in rubrics
    ]
    return OfflinePolicyEvaluation(ran=True, trajectory_count=n, policies=policies)
