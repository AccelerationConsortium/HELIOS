"""Stub advisor for wiring tests.

This does NOT call any real model. It simply echoes the baseline decision as
an LLM proposal so you can validate the plumbing/UI.
"""

from __future__ import annotations

from typing import List, Optional, Dict

from ..core.types import DeviceState, HardwareError, Action, Decision
from .types import LLMDecisionProposal


class EchoBaselineAdvisor:
    def __init__(self, model: str = "stub/echo"):
        self.model = model

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
        return LLMDecisionProposal(
            kind=baseline_decision.kind,
            rationale=f"(LLM stub) {baseline_decision.rationale}",
            actions=list(baseline_decision.actions),
            confidence=1.0,
            model=self.model,
            provider="stub",
            notes={
                "echo": True,
                "error_type": error.type,
                "device": error.device,
            },
        )
