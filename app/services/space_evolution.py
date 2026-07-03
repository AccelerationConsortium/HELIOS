"""Space/Objective evolution advisor + group-relative ranking (Phase C / C1+C4).

C1 — the advisory agent that proposes breaking a fixed objective/boundary. Given
a Nexus problem fingerprint (``ProblemProfiler`` output) plus the base contract,
it emits ``SpaceOverlay`` proposals: widen a stalled dimension, or add a
secondary KPI when the proxy is mismatched. Deterministic and mock-safe — the
intelligence is in the rules, not in prompt glue, so the loop still closes under
``LLM_PROVIDER=mock``. An LLM can later enrich the reasoning, but never gates it.

C4 — group-relative ranking (GRPO-style): within one round's candidate/proposal
set, score each by its advantage over the group mean. Absolute reward is hard to
calibrate in science; "which of these is better, here" is easier and is the
signal used to decide which proposal is worth expensive robot time.

    fingerprint + contract ──► SpaceEvolutionAdvisor.propose() ──► SpaceOverlay[]
    [(id, reward), ...]     ──► group_relative_rank()          ──► ranked[]
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from app.contracts.task_contract import TaskContract
from app.services.space_overlay import (
    BoundaryOverlay,
    ObjectiveOverlay,
    SpaceOverlay,
)

__all__ = [
    "SpaceEvolutionAdvisor",
    "RankedItem",
    "group_relative_rank",
]

_DEFAULT_CONFIDENCE = 0.75
_WIDEN_FRACTION = 0.5  # widen a stalled dimension by 50% on each side


class SpaceEvolutionAdvisor:
    """Propose objective reframes / boundary expansions from a fingerprint."""

    def propose(
        self, fingerprint: dict[str, Any], contract: TaskContract
    ) -> list[SpaceOverlay]:
        proposals: list[SpaceOverlay] = []
        confidence = float(fingerprint.get("confidence", _DEFAULT_CONFIDENCE))

        # 1) Plateau → widen the first numeric dimension with finite bounds.
        if self._is_plateaued(fingerprint):
            dim = self._first_numeric_dimension(contract)
            if dim is not None:
                span = dim.max_value - dim.min_value
                pad = span * _WIDEN_FRACTION
                proposals.append(
                    SpaceOverlay(
                        proposal_id=f"widen-{dim.param_name}",
                        reason=(
                            f"objective plateaued; widen '{dim.param_name}' bounds "
                            f"by {int(_WIDEN_FRACTION * 100)}% to escape the local basin"
                        ),
                        confidence=confidence,
                        expected_gain="new optima outside the current box",
                        boundary_overlays=[
                            BoundaryOverlay(
                                param_name=dim.param_name,
                                new_min=dim.min_value - pad,
                                new_max=dim.max_value + pad,
                            )
                        ],
                    )
                )

        # 2) Proxy mismatch → add a secondary KPI to reframe the objective.
        if self._proxy_mismatched(fingerprint):
            kpi = fingerprint.get("suggested_secondary_kpi") or (
                f"{contract.objective.primary_kpi}_robustness"
            )
            proposals.append(
                SpaceOverlay(
                    proposal_id=f"reframe-{kpi}",
                    reason=(
                        "proxy mismatch: the primary KPI diverges from true value; "
                        f"add secondary KPI '{kpi}' to reframe the objective"
                    ),
                    confidence=confidence,
                    expected_gain="objective better tracks real scientific value",
                    objective_overlay=ObjectiveOverlay(add_secondary_kpis=[kpi]),
                )
            )

        return proposals

    @staticmethod
    def _is_plateaued(fingerprint: dict[str, Any]) -> bool:
        return bool(
            fingerprint.get("plateaued")
            or fingerprint.get("improvement_stalled")
            or fingerprint.get("regime") == "plateau"
        )

    @staticmethod
    def _proxy_mismatched(fingerprint: dict[str, Any]) -> bool:
        return bool(
            fingerprint.get("proxy_mismatch")
            or fingerprint.get("proxy_gap") == "high"
        )

    @staticmethod
    def _first_numeric_dimension(contract: TaskContract):
        for dim in contract.exploration_space.dimensions:
            if (
                dim.param_type in ("number", "integer")
                and dim.min_value is not None
                and dim.max_value is not None
                and dim.max_value > dim.min_value
            ):
                return dim
        return None


class RankedItem(BaseModel):
    """One item scored relative to its group."""

    id: str
    reward: float
    advantage: float  # reward - group_mean
    rank: int  # 1 = best


def group_relative_rank(items: list[dict[str, Any]]) -> list[RankedItem]:
    """Rank items by advantage over the group mean (GRPO-style).

    ``items`` is a list of ``{"id": str, "reward": float}``. Returns them sorted
    best-first with a 1-based rank and each item's advantage. Empty in → empty out.
    """
    if not items:
        return []
    mean = sum(float(it["reward"]) for it in items) / len(items)
    scored = [
        RankedItem(
            id=str(it["id"]),
            reward=float(it["reward"]),
            advantage=round(float(it["reward"]) - mean, 10),
            rank=0,
        )
        for it in items
    ]
    scored.sort(key=lambda r: r.reward, reverse=True)
    for i, item in enumerate(scored, start=1):
        item.rank = i
    return scored
