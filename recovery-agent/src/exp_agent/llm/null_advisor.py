"""Null / no-op LLM advisor.

Used by default so the rest of the system doesn't need feature flags.
"""

from __future__ import annotations

from typing import List, Optional, Dict

from ..core.types import DeviceState, HardwareError, Action, Decision
from .advisor import LLMAdvisor
from .types import LLMDecisionProposal


class NullLLMAdvisor:
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
        return None
