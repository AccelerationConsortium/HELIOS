"""Persistent goal harness primitives for autonomous HELIOS operation.

This module is intentionally pure: it normalizes observations, proposes the
next tool action, records reflection notes, and marks bad paths. It does not
write to the database, call PUDA, execute tools, or promote learned policies.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "GoalHarness",
    "GoalHarnessDecision",
    "GoalHarnessState",
    "GoalHarnessStepResult",
    "ObservationEnvelope",
    "ReflectionNote",
    "ToolAction",
    "ToolDescriptor",
    "advance_goal",
]

GoalStatus = Literal[
    "active",
    "waiting_external",
    "waiting_human",
    "completed",
    "failed",
    "aborted",
]
ObservationSource = Literal[
    "api",
    "artifact",
    "human",
    "log",
    "puda",
    "strategy",
    "system",
    "vision",
]
SignalSeverity = Literal["info", "warning", "critical"]
ToolRisk = Literal["low", "medium", "high"]
ActionIntent = Literal[
    "observe",
    "plan",
    "execute",
    "recover",
    "reflect",
    "pause",
    "abort",
    "request_human",
]
ActionAuthority = Literal["shadow", "advisory", "canary", "live"]
ActionStatus = Literal["proposed", "approved", "executed", "failed", "skipped"]
ReflectionKind = Literal[
    "observation",
    "decision",
    "failure",
    "path_kill",
    "learning",
    "blocker",
    "completion",
]


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12]}"


def _ensure_aware_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError("timestamp must be timezone-aware")
    return value


class ObservationEnvelope(BaseModel):
    """Normalized signal consumed by the goal harness."""

    observation_id: str = Field(default_factory=lambda: _new_id("obs"))
    goal_id: str
    source: ObservationSource
    signal_type: str
    content: dict[str, Any] = Field(default_factory=dict)
    severity: SignalSeverity = "info"
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    received_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("received_at")
    @classmethod
    def _received_at_must_be_aware(cls, value: datetime) -> datetime:
        return _ensure_aware_datetime(value)


class ToolDescriptor(BaseModel):
    """A tool the autonomous harness may propose but not execute."""

    tool_name: str
    capability: str
    risk: ToolRisk = "low"
    input_schema: dict[str, Any] = Field(default_factory=dict)
    required_permissions: tuple[str, ...] = ()
    rollback_available: bool = False
    timeout_seconds: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolAction(BaseModel):
    """A proposed or recorded tool action."""

    action_id: str = Field(default_factory=lambda: _new_id("act"))
    tool_name: str
    intent: ActionIntent
    params: dict[str, Any] = Field(default_factory=dict)
    rationale: str
    authority: ActionAuthority = "advisory"
    required_permissions: tuple[str, ...] = ()
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    status: ActionStatus = "proposed"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReflectionNote(BaseModel):
    """Structured note the next planning step can read."""

    note_id: str = Field(default_factory=lambda: _new_id("note"))
    kind: ReflectionKind
    summary: str
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    recommendation: str | None = None
    applies_to: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _created_at_must_be_aware(cls, value: datetime) -> datetime:
        return _ensure_aware_datetime(value)


class GoalHarnessState(BaseModel):
    """Durable state owned by the goal-oriented autonomous harness."""

    goal_id: str = Field(default_factory=lambda: _new_id("goal"))
    objective: str
    status: GoalStatus = "active"
    round_index: int = Field(default=0, ge=0)
    budget_remaining: int | None = Field(default=None, ge=0)
    observations: list[ObservationEnvelope] = Field(default_factory=list)
    tool_registry: list[ToolDescriptor] = Field(default_factory=list)
    proposed_actions: list[ToolAction] = Field(default_factory=list)
    reflection_notes: list[ReflectionNote] = Field(default_factory=list)
    killed_paths: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def terminal(self) -> bool:
        return self.status in {"completed", "failed", "aborted"}

    def tool_names(self) -> set[str]:
        return {tool.tool_name for tool in self.tool_registry}


class GoalHarnessDecision(BaseModel):
    """Decision emitted by one harness step."""

    status: GoalStatus
    action: ToolAction | None = None
    notes: list[ReflectionNote] = Field(default_factory=list)
    killed_paths: tuple[str, ...] = ()
    rationale: str
    should_stop: bool = False
    requires_human: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class GoalHarnessStepResult(BaseModel):
    """State transition output for one harness step."""

    state: GoalHarnessState
    decision: GoalHarnessDecision


class GoalHarness:
    """Deterministic goal-oriented state machine.

    The harness is deliberately conservative. It can propose a tool action and
    write a reflection note, but it cannot execute the action. Runtime callers
    can later persist the returned state or pass the proposed action to a
    separate approval/execution layer.
    """

    _TERMINAL_RATIONALE = "Goal is already terminal; no further action proposed."

    def advance(
        self,
        state: GoalHarnessState,
        *,
        observations: list[ObservationEnvelope] | tuple[ObservationEnvelope, ...] = (),
        tools: list[ToolDescriptor] | tuple[ToolDescriptor, ...] = (),
    ) -> GoalHarnessStepResult:
        next_state = state.model_copy(deep=True)
        incoming_observations = [obs.model_copy(deep=True) for obs in observations]
        incoming_tools = [tool.model_copy(deep=True) for tool in tools]

        self._merge_tools(next_state, incoming_tools)
        self._append_observations(next_state, incoming_observations)

        if next_state.terminal:
            decision = GoalHarnessDecision(
                status=next_state.status,
                rationale=self._TERMINAL_RATIONALE,
                should_stop=True,
            )
            return GoalHarnessStepResult(state=next_state, decision=decision)

        if next_state.budget_remaining == 0:
            note = ReflectionNote(
                kind="completion",
                summary="Goal budget is exhausted.",
                recommendation="Stop autonomous execution and summarize the campaign.",
            )
            next_state.status = "completed"
            next_state.reflection_notes.append(note)
            decision = GoalHarnessDecision(
                status="completed",
                notes=[note],
                rationale="Budget exhausted before another action could be proposed.",
                should_stop=True,
            )
            return GoalHarnessStepResult(state=next_state, decision=decision)

        path_notes = self._mark_bad_paths(next_state, incoming_observations)
        critical = [obs for obs in incoming_observations if obs.severity == "critical"]
        if critical:
            action = ToolAction(
                tool_name="human_supervisor",
                intent="request_human",
                params={"observation_ids": [obs.observation_id for obs in critical]},
                rationale="Critical observation requires human review before continuing.",
                authority="advisory",
                required_permissions=("human_review",),
                risk_score=0.0,
            )
            note = ReflectionNote(
                kind="blocker",
                summary="Critical observation blocked autonomous progress.",
                evidence=[obs.model_dump(mode="json") for obs in critical],
                recommendation="Review safety, data quality, and execution state before resuming.",
            )
            next_state.status = "waiting_human"
            next_state.blockers.append(note.summary)
            next_state.proposed_actions.append(action)
            next_state.reflection_notes.extend([*path_notes, note])
            decision = GoalHarnessDecision(
                status="waiting_human",
                action=action,
                notes=[*path_notes, note],
                killed_paths=tuple(note.applies_to for note in path_notes if note.applies_to),
                rationale=action.rationale,
                requires_human=True,
            )
            return GoalHarnessStepResult(state=next_state, decision=decision)

        action = self._propose_next_action(next_state, incoming_observations)
        decision_note = ReflectionNote(
            kind="decision",
            summary=f"Proposed {action.intent} via {action.tool_name}.",
            evidence=[
                {
                    "action_id": action.action_id,
                    "tool_name": action.tool_name,
                    "intent": action.intent,
                    "risk_score": action.risk_score,
                }
            ],
            recommendation="Route the proposed action through approval/execution gates.",
            applies_to=action.action_id,
        )
        next_state.status = "active" if action.intent != "observe" else "waiting_external"
        if next_state.status == "waiting_external":
            next_state.blockers.append("Waiting for additional observation before execution.")
        next_state.proposed_actions.append(action)
        next_state.reflection_notes.extend([*path_notes, decision_note])

        decision = GoalHarnessDecision(
            status=next_state.status,
            action=action,
            notes=[*path_notes, decision_note],
            killed_paths=tuple(note.applies_to for note in path_notes if note.applies_to),
            rationale=action.rationale,
        )
        return GoalHarnessStepResult(state=next_state, decision=decision)

    def _merge_tools(
        self, state: GoalHarnessState, tools: list[ToolDescriptor]
    ) -> None:
        by_name = {tool.tool_name: tool for tool in state.tool_registry}
        for tool in tools:
            by_name[tool.tool_name] = tool
        state.tool_registry = [by_name[name] for name in sorted(by_name)]

    def _append_observations(
        self, state: GoalHarnessState, observations: list[ObservationEnvelope]
    ) -> None:
        for observation in observations:
            if observation.goal_id != state.goal_id:
                raise ValueError(
                    f"observation {observation.observation_id} belongs to "
                    f"{observation.goal_id}, not {state.goal_id}"
                )
            state.observations.append(observation)

    def _mark_bad_paths(
        self,
        state: GoalHarnessState,
        observations: list[ObservationEnvelope],
    ) -> list[ReflectionNote]:
        notes: list[ReflectionNote] = []
        killed = set(state.killed_paths)
        for observation in observations:
            path_id = self._path_to_kill(observation)
            if path_id is None or path_id in killed:
                continue
            killed.add(path_id)
            state.killed_paths.append(path_id)
            notes.append(
                ReflectionNote(
                    kind="path_kill",
                    summary=f"Marked path '{path_id}' as unsuitable for this goal.",
                    evidence=[observation.model_dump(mode="json")],
                    recommendation="Exclude this path from future autonomous proposals unless a human re-enables it.",
                    applies_to=path_id,
                )
            )
        return notes

    def _path_to_kill(self, observation: ObservationEnvelope) -> str | None:
        if observation.signal_type not in {
            "failure",
            "negative_result",
            "path_failure",
            "safety",
        }:
            return None
        recommendation = str(observation.content.get("recommendation", "")).lower()
        if observation.severity == "info" and recommendation != "kill_path":
            return None
        path_id = observation.content.get("path_id") or observation.content.get(
            "candidate_id"
        )
        return str(path_id) if path_id else None

    def _propose_next_action(
        self,
        state: GoalHarnessState,
        observations: list[ObservationEnvelope],
    ) -> ToolAction:
        tool = self._select_tool(state.tool_registry)
        if tool is None:
            return ToolAction(
                tool_name="tool_registry",
                intent="observe",
                params={"missing": "no_registered_tools"},
                rationale=(
                    "No registered tools are available, so the harness must wait "
                    "for tool registration."
                ),
                authority="advisory",
                required_permissions=("tool_registration",),
                risk_score=0.0,
            )

        latest_signal = observations[-1].signal_type if observations else "none"
        intent: ActionIntent = "execute"
        if "observe" in tool.capability or "sensor" in tool.capability:
            intent = "observe"
        elif "plan" in tool.capability:
            intent = "plan"
        elif "recover" in tool.capability:
            intent = "recover"

        return ToolAction(
            tool_name=tool.tool_name,
            intent=intent,
            params={
                "goal_id": state.goal_id,
                "objective": state.objective,
                "round_index": state.round_index,
                "latest_signal": latest_signal,
                "killed_paths": deepcopy(state.killed_paths),
            },
            rationale=(
                f"Use {tool.tool_name} to advance objective '{state.objective}' "
                f"after {len(state.observations)} accumulated observations."
            ),
            authority="advisory",
            required_permissions=tool.required_permissions,
            risk_score={"low": 0.1, "medium": 0.35, "high": 0.7}[tool.risk],
            metadata={"capability": tool.capability},
        )

    def _select_tool(self, tools: list[ToolDescriptor]) -> ToolDescriptor | None:
        if not tools:
            return None
        priority = ("puda", "execute", "plan", "recover", "observe", "sensor")
        for token in priority:
            for tool in tools:
                haystack = f"{tool.tool_name} {tool.capability}".lower()
                if token in haystack:
                    return tool
        return tools[0]


def advance_goal(**kwargs: Any) -> GoalHarnessStepResult:
    """Convenience wrapper around :class:`GoalHarness`."""

    return GoalHarness().advance(**kwargs)
