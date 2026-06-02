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

            board = manager.get_or_create(str(round_number), campaign_id)
            entries = board.read_all()
            if entries:
                lines = ["## Round Blackboard (latest peer observations)\n"]
                for key, entry in sorted(entries.items()):
                    lines.append(
                        f"- **[{key}]** {entry.value!r} "
                        f"(confidence {entry.confidence:.0%}, from {entry.author})"
                    )
                    blackboard_entries[key] = {
                        "value": entry.value,
                        "author": entry.author,
                        "confidence": entry.confidence,
                        "timestamp": entry.timestamp,
                    }
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


def harvest_agent_result(
    *,
    context: AgentRuntimeContext | None,
    result: AgentResult,
) -> None:
    """Publish useful post-call observations back to the KnowledgeBus."""
    if context is None or context.round_number is None or not result.success:
        return
    output = result.output
    if output is None:
        return

    summaries = _extract_output_summaries(output)
    if not summaries:
        return

    try:
        from app.services.knowledge_bus import KnowledgeEvent, get_bus

        bus = get_bus(context.campaign_id)
        for key, text, confidence in summaries:
            bus.publish(
                KnowledgeEvent(
                    source_agent=context.agent_name,
                    key=key,
                    delta_text=text,
                    confidence=confidence,
                    round_id=str(context.round_number),
                    ttl_rounds=3,
                )
            )
    except Exception:
        logger.debug("Agent context result harvest failed", exc_info=True)


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


def _extract_output_summaries(output: Any) -> list[tuple[str, str, float]]:
    items: list[tuple[str, str, float]] = []

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
            items.append((
                f"agent.{output.__class__.__name__}.{attr}",
                text.strip()[:500],
                confidence,
            ))

    decision_nodes = getattr(output, "decision_nodes", None)
    if decision_nodes:
        items.append((
            f"agent.{output.__class__.__name__}.decision_nodes",
            f"Produced {len(decision_nodes)} decision node(s).",
            confidence,
        ))

    return items
