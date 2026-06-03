"""Null / no-op LLM advisor.

Used by default so the rest of the system doesn't need feature flags.
"""

from __future__ import annotations

from ..core.types import Action, Decision, DeviceState, HardwareError
from .types import LLMDecisionProposal


class NullLLMAdvisor:
    def propose_recovery(
        self,
        *,
        state: DeviceState,
        error: HardwareError,
        history: list[DeviceState],
        retry_counts: dict[str, int],
        last_action: Action | None,
        stage: str | None,
        baseline_decision: Decision,
    ) -> LLMDecisionProposal | None:
        return None
