"""Runtime context injection for agent-native orchestration.

The context layer is deliberately advisory: it enriches agent/LLM calls with
peer knowledge, round blackboard state, and routing metadata without making
campaign execution depend on those services being available.
"""
from __future__ import annotations

import contextvars
import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from app.agents.base import AgentResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentRuntimeContext:
    campaign_id: str
    agent_name: str
    caller: str
    trace_id: str
    round_number: int | None = None
    candidate_index: int | None = None
    prompt_block: str = ""
    knowledge_events: list[dict[str, Any]] = field(default_factory=list)
    blackboard_entries: dict[str, dict[str, Any]] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        """Return a compact, serializable snapshot for replay/debug."""
        return {
            "campaign_id": self.campaign_id,
            "agent_name": self.agent_name,
            "caller": self.caller,
            "trace_id": self.trace_id,
            "round_number": self.round_number,
            "candidate_index": self.candidate_index,
            "knowledge_event_count": len(self.knowledge_events),
            "blackboard_entry_count": len(self.blackboard_entries),
            "knowledge_events": self.knowledge_events,
            "blackboard_entries": self.blackboard_entries,
            "prompt_preview": self.prompt_block[:2000],
        }


_current_context: contextvars.ContextVar[AgentRuntimeContext | None] = (
    contextvars.ContextVar("helios_agent_runtime_context", default=None)
)


def get_current_agent_context() -> AgentRuntimeContext | None:
    """Return the context for the currently executing agent call, if any."""
    return _current_context.get()


def set_current_agent_context(
    context: AgentRuntimeContext | None,
) -> contextvars.Token[AgentRuntimeContext | None]:
    """Set the contextvar and return the reset token."""
    return _current_context.set(context)


def reset_current_agent_context(
    token: contextvars.Token[AgentRuntimeContext | None],
) -> None:
    _current_context.reset(token)


async def build_agent_context(
    *,
    campaign_id: str,
    agent_name: str,
    caller: str,
    trace_id: str,
    input_data: BaseModel,
) -> AgentRuntimeContext | None:
    """Build advisory runtime context for a ControlPlane call.

    Failures are swallowed by design. Context should improve agent behavior,
    never block safety-critical campaign execution.
    """
    if not campaign_id:
        return None

    round_number = _extract_int(
        input_data,
        "round_number",
        "round",
        "current_round",
    )
    candidate_index = _extract_int(
        input_data,
        "candidate_index",
        "candidate",
        "candidate_idx",
    )

    prompt_parts: list[str] = []
    knowledge_events: list[dict[str, Any]] = []
    blackboard_entries: dict[str, dict[str, Any]] = {}

    if round_number is not None:
        try:
            from app.services.knowledge_bus import KnowledgeBus, get_bus

            bus = get_bus(campaign_id)
            events = await bus.drain_for_agent(agent_name, round_number)
            prompt = KnowledgeBus.format_for_prompt(events)
            if prompt:
                prompt_parts.append(prompt)
            knowledge_events = [
                {
                    "source_agent": ev.source_agent,
                    "key": ev.key,
                    "delta_text": ev.delta_text,
                    "confidence": ev.confidence,
                    "round_id": ev.round_id,
                }
                for ev in events
            ]
        except Exception:
            logger.debug("Agent context knowledge drain failed", exc_info=True)

        try:
            from app.agents.blackboard import manager

            board = manager.get_existing(str(round_number), campaign_id)
            entries = board.read_all() if board is not None else {}
            if entries:
                lines = ["## Round Blackboard (latest peer observations)\n"]
                for key, entry in sorted(entries.items()):
                    lines.append(
                        f"- **[{key}]** {entry.value!r} "
                        f"(type {entry.entry_type}, confidence "
                        f"{entry.confidence:.0%}, from {entry.author})"
                    )
                    blackboard_entries[key] = entry.to_dict()
                prompt_parts.append("\n".join(lines) + "\n")
        except Exception:
            logger.debug("Agent context blackboard read failed", exc_info=True)

    prompt_block = "\n".join(part for part in prompt_parts if part).strip()
    return AgentRuntimeContext(
        campaign_id=campaign_id,
        agent_name=agent_name,
        caller=caller,
        trace_id=trace_id,
        round_number=round_number,
        candidate_index=candidate_index,
        prompt_block=prompt_block,
        knowledge_events=knowledge_events,
        blackboard_entries=blackboard_entries,
    )


def persist_agent_context_snapshot(context: AgentRuntimeContext | None) -> None:
    """Persist an advisory context snapshot through the campaign event log."""
    if context is None:
        return
    try:
        from app.services.campaign_events import log_event

        log_event(
            context.campaign_id,
            "agent_context_snapshot",
            {
                "type": "agent_context_snapshot",
                **context.snapshot(),
            },
        )
    except Exception:
        logger.debug("Agent context snapshot persistence failed", exc_info=True)


async def harvest_agent_result(
    *,
    context: AgentRuntimeContext | None,
    result: AgentResult,
) -> None:
    """Publish useful post-call observations to collaboration memory."""
    if context is None or context.round_number is None or not result.success:
        return
    output = result.output
    if output is None:
        return

    observations = _extract_output_observations(output)
    if not observations:
        return

    try:
        from app.services.knowledge_bus import KnowledgeEvent, get_bus

        bus = get_bus(context.campaign_id)
        for observation in observations:
            bus.publish(
                KnowledgeEvent(
                    source_agent=context.agent_name,
                    key=observation.key,
                    delta_text=observation.prompt_text,
                    confidence=observation.confidence,
                    round_id=str(context.round_number),
                    ttl_rounds=3,
                )
            )
    except Exception:
        logger.debug("Agent context result harvest failed", exc_info=True)

    try:
        from app.agents.blackboard import manager

        board = manager.get_or_create(str(context.round_number), context.campaign_id)
        written_entries = []
        for observation in observations:
            entry = await board.write(
                observation.key,
                observation.value,
                author=context.agent_name,
                confidence=observation.confidence,
                entry_type=observation.entry_type,
                tags=observation.tags,
                metadata=observation.metadata,
            )
            written_entries.append(entry.to_dict())
        _persist_blackboard_entries(context, written_entries)
    except Exception:
        logger.debug("Agent blackboard result harvest failed", exc_info=True)


def _extract_int(model: BaseModel, *names: str) -> int | None:
    for name in names:
        value = getattr(model, name, None)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


@dataclass(frozen=True)
class _OutputObservation:
    key: str
    value: Any
    prompt_text: str
    confidence: float
    entry_type: str
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def _extract_output_observations(output: Any) -> list[_OutputObservation]:
    items: list[_OutputObservation] = []

    confidence = 0.75
    for attr in ("confidence", "convergence_confidence"):
        value = getattr(output, attr, None)
        if value is not None:
            try:
                confidence = max(0.0, min(1.0, float(value)))
                break
            except (TypeError, ValueError):
                pass

    for attr in ("narrative", "recommendation", "summary", "notes", "chat_message"):
        text = getattr(output, attr, None)
        if isinstance(text, str) and text.strip():
            trimmed = text.strip()[:500]
            items.append(
                _OutputObservation(
                    key=f"agent.{output.__class__.__name__}.{attr}",
                    value=trimmed,
                    prompt_text=trimmed,
                    confidence=confidence,
                    entry_type="observation",
                    tags=("agent_output", attr),
                    metadata={
                        "output_model": output.__class__.__name__,
                        "field": attr,
                    },
                )
            )

    decision_nodes = getattr(output, "decision_nodes", None)
    if decision_nodes:
        safe_nodes = _json_safe(decision_nodes)
        count = len(decision_nodes)
        items.append(
            _OutputObservation(
                key=f"agent.{output.__class__.__name__}.decision_nodes",
                value=safe_nodes,
                prompt_text=f"Produced {count} decision node(s).",
                confidence=confidence,
                entry_type="decision",
                tags=("agent_output", "decision_nodes"),
                metadata={
                    "output_model": output.__class__.__name__,
                    "field": "decision_nodes",
                    "count": count,
                },
            )
        )

    return items


def _persist_blackboard_entries(
    context: AgentRuntimeContext,
    entries: list[dict[str, Any]],
) -> None:
    if not entries:
        return
    try:
        from app.services.campaign_events import log_event

        log_event(
            context.campaign_id,
            "blackboard_entries",
            {
                "type": "blackboard_entries",
                "campaign_id": context.campaign_id,
                "round_number": context.round_number,
                "agent": context.agent_name,
                "trace_id": context.trace_id,
                "entries": _json_safe(entries),
            },
        )
    except Exception:
        logger.debug("Blackboard event persistence failed", exc_info=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
