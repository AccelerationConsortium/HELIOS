"""Anomaly packaging helpers.

Phase 3 groundwork: represent anomalous runs in a consistent, portable format
so other domain-specialist agents can learn from them.

This file is intentionally lightweight: it focuses on data shape, not policy.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..core.types import Decision, DeviceState, SignatureResult
from ..llm.types import LLMDecisionProposal


class AnomalyPacket(BaseModel):
    """Portable record of an anomaly + what we did about it."""

    model_config = ConfigDict(frozen=False)

    packet_id: str

    # What happened
    error: dict[str, Any]
    signature: SignatureResult | None = None

    # What we decided
    baseline_decision: Decision
    llm_proposal: LLMDecisionProposal | None = None

    # Evidence
    telemetry_window: list[DeviceState] = Field(default_factory=list)

    # Notes for future agents
    tags: list[str] = Field(default_factory=list)
    notes: dict[str, Any] = Field(default_factory=dict)
