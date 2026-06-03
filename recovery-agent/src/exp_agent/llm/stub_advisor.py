"""Stub advisor for wiring tests.

This does NOT call any real model. It simply echoes the baseline decision as
an LLM proposal so you can validate the plumbing/UI.
"""

from __future__ import annotations

from ..core.types import Action, Decision, DeviceState, HardwareError
from .types import LLMDecisionProposal


class EchoBaselineAdvisor:
    def __init__(self, model: str = "stub/echo"):
        self.model = model

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
