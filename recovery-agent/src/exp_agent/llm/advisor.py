"""LLM advisor interface.

Phase 2 design: the LLM is an *advisor* that proposes decisions.
The policy engine and guarded executor remain the gatekeepers.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Dict

from ..core.types import DeviceState, HardwareError, Action, Decision
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
        history: List[DeviceState],
        retry_counts: Dict[str, int],
        last_action: Optional[Action],
        stage: Optional[str],
        baseline_decision: Decision,
    ) -> Optional[LLMDecisionProposal]:
        ...
