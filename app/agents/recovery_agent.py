"""Recovery Agent -- cross-cutting error handling and retry logic.

Wraps the recovery-agent package to provide error recovery strategies
with veto power over execution continuation decisions.

Architecture Position:
- Layer: Cross-cutting (alongside SafetyAgent)
- Trigger: After execution failures
- Authority: Veto power to abort, retry, or degrade operations
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.agents.base import BaseAgent

# Lazy import recovery-agent package
_recovery_agent_available = False
_RecoveryAgent = None
_DeviceState = None
_HardwareError = None
_Action = None

RecoveryPhase = Literal[
    "observe",
    "diagnose",
    "plan_attempt",
    "evaluate",
    "revise",
    "exit",
]


def _import_recovery_agent():
    """Lazy import of recovery-agent package."""
    global _recovery_agent_available, _RecoveryAgent, _DeviceState, _HardwareError, _Action

    if _recovery_agent_available:
        return True

    try:
        # Add recovery-agent to path if not already present
        recovery_path = Path(__file__).parent.parent.parent / "recovery-agent" / "src"
        if recovery_path.exists() and str(recovery_path) not in sys.path:
            sys.path.insert(0, str(recovery_path))

        from exp_agent.core.types import Action as _A
        from exp_agent.core.types import DeviceState as _DS
        from exp_agent.core.types import HardwareError as _HE
        from exp_agent.recovery.recovery_agent import RecoveryAgent as _RA

        _RecoveryAgent = _RA
        _DeviceState = _DS
        _HardwareError = _HE
        _Action = _A
        _recovery_agent_available = True
        return True
    except ImportError:
        return False


class RecoveryInput(BaseModel):
    """Input for recovery decision-making."""
    error_type: str
    error_message: str
    device_name: str
    device_status: str = "error"  # Will be normalized to recovery-agent's DeviceStatus
    error_severity: str = "medium"  # "low", "medium", "high"
    telemetry: dict[str, Any] = Field(default_factory=dict)
    history: list[dict[str, Any]] = Field(default_factory=list)
    last_action: str | None = None
    stage: str | None = None
    retry_count: int = 0
    safety_packet: dict[str, Any] | None = None
    episode: RecoveryEpisode | None = None
    last_attempt_result: dict[str, Any] | None = None


class RecoveryAttempt(BaseModel):
    """One concrete recovery attempt inside an error episode."""

    attempt_no: int
    hypothesis: str
    action: str
    params: dict[str, Any] = Field(default_factory=dict)
    result: str | None = None
    error_after_attempt: str | None = None


class RecoveryEpisode(BaseModel):
    """Stateful trace for observe -> diagnose -> attempt -> revise recovery."""

    episode_id: str = Field(default_factory=lambda: f"rec-{uuid.uuid4().hex[:12]}")
    run_id: str | None = None
    step_key: str | None = None
    phase: RecoveryPhase = "observe"
    original_error_type: str
    current_error_type: str
    observations: list[dict[str, Any]] = Field(default_factory=list)
    attempts: list[RecoveryAttempt] = Field(default_factory=list)
    rejected_hypotheses: list[str] = Field(default_factory=list)
    max_attempts: int = 3


class RecoveryOutput(BaseModel):
    """Output from recovery decision."""
    decision: str  # "retry", "degrade", "abort", "skip"
    rationale: str
    actions: list[dict[str, Any]] = Field(default_factory=list)
    retry_delay_seconds: float = 0.0
    max_retries: int = 3
    chemical_safety_event: bool = False
    phase: RecoveryPhase = "plan_attempt"
    episode: RecoveryEpisode
    next_action: dict[str, Any] | None = None
    terminal: bool = False


class RecoveryAgent(BaseAgent[RecoveryInput, RecoveryOutput]):
    """Recovery Agent for error handling and retry strategies.

    Provides intelligent error recovery with:
    - Policy-driven retry strategies
    - Chemical safety event escalation
    - SafetyAgent veto power integration
    - Fault signature analysis

    Architecture:
    - Layer: Cross-cutting
    - Authority: Veto power over execution continuation
    - Integration: Coordinates with SafetyAgent for chemical safety events
    """

    name = "recovery_agent"
    description = "Error recovery and retry strategy coordinator"
    layer = "cross-cutting"

    def __init__(self):
        super().__init__()
        self._agent = None
        self._available = _import_recovery_agent()

        if self._available:
            self._agent = _RecoveryAgent()

    def validate_input(self, input_data: RecoveryInput) -> list[str]:
        """Validate recovery input."""
        errors: list[str] = []

        if not input_data.error_type:
            errors.append("error_type is required")

        if not input_data.device_name:
            errors.append("device_name is required")

        return errors

    async def process(self, input_data: RecoveryInput) -> RecoveryOutput:
        """Process error and determine recovery strategy.

        Returns:
            RecoveryOutput with decision (retry/degrade/abort) and actions
        """
        if not self._available or self._agent is None:
            # Fallback: simple retry logic
            return self._fallback_recovery(input_data)

        # Convert HELIOS input to recovery-agent types
        state = self._build_device_state(input_data)
        error = self._build_hardware_error(input_data)
        history = self._build_history(input_data)

        # Get recovery decision from agent
        decision = self._agent.decide(
            state=state,
            error=error,
            history=history,
            last_action=self._build_action(input_data.last_action) if input_data.last_action else None,
            stage=input_data.stage,
        )

        # Convert decision back to HELIOS format
        episode = self._update_episode(input_data, decision.kind)
        next_action = self._build_next_action(input_data, decision.kind, decision.actions)
        output = RecoveryOutput(
            decision=decision.kind,
            rationale=decision.rationale,
            actions=[self._serialize_action(a) for a in decision.actions],
            phase=episode.phase,
            episode=episode,
            next_action=next_action,
            terminal=decision.kind != "retry",
        )

        # Check if this is a chemical safety event
        if "CHEMICAL SAFETY EVENT" in decision.rationale:
            output.chemical_safety_event = True
            self.logger.warning(
                "Chemical safety event detected: %s",
                input_data.error_type
            )

        # Set retry parameters based on decision
        if decision.kind == "retry":
            output.retry_delay_seconds = 2.0
            output.max_retries = 3

        self.logger.info(
            "Recovery decision for %s: %s (rationale: %s)",
            input_data.error_type,
            decision.kind,
            decision.rationale[:100],
        )

        return output

    def _update_episode(self, input_data: RecoveryInput, decision: str) -> RecoveryEpisode:
        """Advance the recovery episode with the newest observation and plan."""
        episode = input_data.episode or RecoveryEpisode(
            run_id=input_data.telemetry.get("run_id"),
            step_key=input_data.telemetry.get("step_key"),
            original_error_type=input_data.error_type,
            current_error_type=input_data.error_type,
            max_attempts=3,
        )

        if input_data.last_attempt_result and episode.attempts:
            last = episode.attempts[-1]
            if last.result is None:
                last.result = input_data.last_attempt_result.get("result", "failed")
                last.error_after_attempt = input_data.last_attempt_result.get("error_type")

        observation = {
            "phase": "observe",
            "error_type": input_data.error_type,
            "error_message": input_data.error_message,
            "device_name": input_data.device_name,
            "retry_count": input_data.retry_count,
            "telemetry": input_data.telemetry,
        }
        episode.observations.append(observation)
        episode.current_error_type = input_data.error_type

        if input_data.retry_count > 0:
            previous = episode.attempts[-1].hypothesis if episode.attempts else input_data.error_type
            if previous not in episode.rejected_hypotheses:
                episode.rejected_hypotheses.append(previous)
            episode.phase = "revise"

        if decision == "retry":
            episode.phase = "plan_attempt"
            action_name = self._action_name_for(input_data, len(episode.attempts))
            episode.attempts.append(
                RecoveryAttempt(
                    attempt_no=len(episode.attempts) + 1,
                    hypothesis=self._hypothesis_for(input_data),
                    action=action_name,
                    params={
                        "retry_count": input_data.retry_count,
                        "error_type": input_data.error_type,
                    },
                )
            )
        else:
            episode.phase = "exit"

        return episode

    def _hypothesis_for(self, input_data: RecoveryInput) -> str:
        """Name the current recovery hypothesis in operator-readable terms."""
        if input_data.error_type in {"timeout", "communication_error"}:
            if input_data.retry_count == 0:
                return "transient communication failure"
            if input_data.retry_count == 1:
                return "stale connection or slow device response"
            return "persistent device communication fault"
        if input_data.error_type == "postcondition_failed":
            return "operation completed but system state has not stabilized"
        return f"recoverable {input_data.error_type}"

    def _action_name_for(self, input_data: RecoveryInput, prior_attempts: int) -> str:
        """Choose a progressively stronger recovery action label."""
        if input_data.error_type in {"timeout", "communication_error"}:
            actions = ["retry_original", "wait_and_retry", "reset_connection_then_retry"]
            return actions[min(prior_attempts, len(actions) - 1)]
        if input_data.error_type == "postcondition_failed":
            actions = ["retry_original", "wait_for_stabilization", "degrade_target_then_retry"]
            return actions[min(prior_attempts, len(actions) - 1)]
        return "retry_original"

    def _build_next_action(
        self,
        input_data: RecoveryInput,
        decision: str,
        actions: list[Any],
    ) -> dict[str, Any] | None:
        """Expose the next planned action without expanding executor authority."""
        if decision != "retry":
            return None
        if input_data.episode and input_data.episode.attempts:
            action_name = input_data.episode.attempts[-1].action
        else:
            action_name = self._action_name_for(input_data, input_data.retry_count)
        if actions:
            serialized = self._serialize_action(actions[0])
            serialized["planned_name"] = action_name
            return serialized
        return {
            "name": action_name,
            "effect": "control",
            "device": input_data.device_name,
            "params": {
                "retry_count": input_data.retry_count,
                "error_type": input_data.error_type,
            },
        }

    def _build_device_state(self, input_data: RecoveryInput):
        """Convert HELIOS input to recovery-agent DeviceState."""
        # Normalize status to recovery-agent's DeviceStatus literals
        status_map = {
            "idle": "idle",
            "running": "running",
            "error": "error",
            "emergency": "error",  # Emergency maps to error
            "warning": "error",  # Warning maps to error
            "ok": "idle",  # OK maps to idle
        }
        status = status_map.get(input_data.device_status.lower(), "error")

        return _DeviceState(
            name=input_data.device_name,
            status=status,
            telemetry=input_data.telemetry,
        )

    def _build_hardware_error(self, input_data: RecoveryInput):
        """Convert HELIOS input to recovery-agent HardwareError."""
        # Normalize severity to recovery-agent's Severity literals
        severity_map = {
            "low": "low",
            "medium": "medium",
            "high": "high",
            "critical": "high",  # Critical maps to high
            "warning": "low",  # Warning maps to low
        }
        severity = severity_map.get(input_data.error_severity.lower(), "medium")

        return _HardwareError(
            device=input_data.device_name,
            type=input_data.error_type,
            severity=severity,
            message=input_data.error_message,
        )

    def _build_history(self, input_data: RecoveryInput) -> list:
        """Convert HELIOS history to recovery-agent DeviceState list."""
        # Status normalization map
        status_map = {
            "idle": "idle",
            "running": "running",
            "error": "error",
            "emergency": "error",
            "warning": "error",
            "ok": "idle",
            "unknown": "error",
        }

        history = []
        for h in input_data.history:
            status = h.get("status", "unknown")
            normalized_status = status_map.get(status.lower(), "error")

            history.append(_DeviceState(
                name=h.get("device_name", input_data.device_name),
                status=normalized_status,
                telemetry=h.get("telemetry", {}),
            ))
        return history

    def _build_action(self, action_name: str):
        """Build recovery-agent Action from name."""
        if action_name and _Action:
            return _Action(
                name=action_name,
                effect="write",
                device="unknown",
            )
        return None

    def _serialize_action(self, action) -> dict[str, Any]:
        """Serialize recovery-agent Action to dict."""
        return {
            "name": action.name,
            "effect": action.effect,
            "device": action.device,
            "params": getattr(action, "params", {}),
            "preconditions": getattr(action, "preconditions", []),
            "postconditions": getattr(action, "postconditions", []),
        }

    def _fallback_recovery(self, input_data: RecoveryInput) -> RecoveryOutput:
        """Fallback recovery logic when recovery-agent is not available."""
        # Simple retry strategy
        if input_data.retry_count < 3:
            episode = self._update_episode(input_data, "retry")
            return RecoveryOutput(
                decision="retry",
                rationale=f"Fallback: retry {input_data.error_type} (attempt {input_data.retry_count + 1}/3)",
                retry_delay_seconds=2.0,
                max_retries=3,
                phase=episode.phase,
                episode=episode,
                next_action=self._build_next_action(input_data, "retry", []),
            )
        else:
            episode = self._update_episode(input_data, "abort")
            return RecoveryOutput(
                decision="abort",
                rationale=f"Fallback: max retries exceeded for {input_data.error_type}",
                phase=episode.phase,
                episode=episode,
                terminal=True,
            )
