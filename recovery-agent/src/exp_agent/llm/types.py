"""Types for LLM proposals.

We keep these separate from core Decision so:
- policy remains deterministic and testable
- LLM output can be logged/audited without affecting execution
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..core.types import Action, DecisionType


class LLMDecisionProposal(BaseModel):
    """A *proposal* from an LLM.

    The orchestrator/policy layer may accept, modify, or ignore this.
    """

    model_config = ConfigDict(frozen=False)

    kind: DecisionType
    rationale: str
    actions: list[Action] = Field(default_factory=list)

    # Optional metadata for debugging/auditing
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    model: str | None = None
    provider: str | None = None
    notes: dict[str, Any] = Field(default_factory=dict)

    source: Literal["llm"] = "llm"
