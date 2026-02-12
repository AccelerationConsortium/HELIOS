"""Anomaly packaging helpers.

Phase 3 groundwork: represent anomalous runs in a consistent, portable format
so other domain-specialist agents can learn from them.

This file is intentionally lightweight: it focuses on data shape, not policy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict

from ..core.types import DeviceState, Decision, SignatureResult, Action
from ..llm.types import LLMDecisionProposal


class AnomalyPacket(BaseModel):
    """Portable record of an anomaly + what we did about it."""

    model_config = ConfigDict(frozen=False)

    packet_id: str

    # What happened
    error: Dict[str, Any]
    signature: Optional[SignatureResult] = None

    # What we decided
    baseline_decision: Decision
    llm_proposal: Optional[LLMDecisionProposal] = None

    # Evidence
    telemetry_window: List[DeviceState] = Field(default_factory=list)

    # Notes for future agents
    tags: List[str] = Field(default_factory=list)
    notes: Dict[str, Any] = Field(default_factory=dict)
