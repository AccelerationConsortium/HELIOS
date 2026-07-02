"""Builder for campaign round decision-layer context.

This module normalizes existing runtime summaries into the Phase 2
``CampaignRoundContext`` model. It is intentionally pure: no database reads,
no external services, and no live strategy selector calls.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import Any

from app.services.decision_models import CampaignRoundContext

__all__ = ["CampaignRoundContextBuilder", "build_campaign_round_context"]


class CampaignRoundContextBuilder:
    """Normalize dict-compatible summaries into a CampaignRoundContext."""

    def build(
        self,
        *,
        campaign_id: str,
        round_index: int,
        strategy_selection_result: dict[str, Any] | None = None,
        stop_requested: bool = False,
        failure_summary: dict[str, Any] | None = None,
        safety_summary: dict[str, Any] | None = None,
        objective_summary: dict[str, Any] | None = None,
        constraint_summary: dict[str, Any] | None = None,
        nexus_diagnostics: dict[str, Any] | None = None,
        backend_memory_summary: dict[str, Any] | None = None,
        bo_mcp_summary: dict[str, Any] | None = None,
        learning_policy_summary: dict[str, Any] | None = None,
        validation_summary: dict[str, Any] | None = None,
        human_observations: list[str] | None = None,
        literature_summary: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CampaignRoundContext:
        return CampaignRoundContext(
            campaign_id=campaign_id,
            round_index=round_index,
            stop_requested=stop_requested,
            failure_summary=_dict_or_empty(failure_summary),
            safety_summary=_dict_or_empty(safety_summary),
            objective_summary=_dict_or_empty(objective_summary),
            constraint_summary=_dict_or_empty(constraint_summary),
            nexus_diagnostics=_dict_or_empty(nexus_diagnostics),
            backend_memory_summary=_dict_or_empty(backend_memory_summary),
            bo_mcp_summary=_dict_or_empty(bo_mcp_summary),
            learning_policy_summary=_dict_or_empty(learning_policy_summary),
            validation_summary=_dict_or_empty(validation_summary),
            human_observations=_list_or_empty(human_observations),
            literature_summary=_dict_or_empty(literature_summary),
            strategy_selection_result=_dict_or_empty(strategy_selection_result),
            metadata=_dict_or_empty(metadata),
        )


def build_campaign_round_context(**kwargs: Any) -> CampaignRoundContext:
    """Build a CampaignRoundContext with the default builder."""
    return CampaignRoundContextBuilder().build(**kwargs)


def _dict_or_empty(value: dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if is_dataclass(value):
        return deepcopy(asdict(value))
    return deepcopy(dict(value))


def _list_or_empty(value: list[str] | None) -> list[str]:
    if value is None:
        return []
    return deepcopy(list(value))
