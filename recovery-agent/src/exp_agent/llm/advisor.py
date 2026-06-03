"""LLM advisor interface.

Phase 2 design: the LLM is an *advisor* that proposes decisions.
The policy engine and guarded executor remain the gatekeepers.
"""

from __future__ import annotations

from typing import Protocol

from ..core.types import Action, Decision, DeviceState, HardwareError
from .types import LLMDecisionProposal


class LLMAdvisor(Protocol):
    """Interface for an LLM-backed advisor.

    Implementations should be side-effect free and fast-fail if unavailable.
    """

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
        ...
