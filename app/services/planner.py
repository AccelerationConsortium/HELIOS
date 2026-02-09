"""LLM-based experiment planner — converts natural language intent to structured plans.

Pipeline:
1. ``build_system_prompt()`` assembles context from SOUL.md + capabilities registry
2. ``plan_from_intent()`` calls the LLM provider with the intent
3. ``parse_plan_response()`` extracts structured JSON from the LLM output

The resulting ``PlanResult`` feeds into ``plan_grounding.py`` for deterministic
validation and protocol JSON generation.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.services.llm_gateway import LLMError, LLMMessage, LLMProvider, get_llm_provider
from app.services.primitives_registry import get_registry


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PlanStep:
    """A single step in the LLM-generated plan."""

    id: str
    primitive: str
    params: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)


@dataclass
class PlanResult:
    """Output of the LLM planning stage."""

    steps: list[PlanStep]
    raw_response: str  # full LLM text (for audit trail)
    model: str
    reasoning: str | None = None  # optional LLM explanation


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PlanParseError(ValueError):
    """Raised when the LLM response cannot be parsed into a valid plan."""


# ---------------------------------------------------------------------------
# System prompt assembly
# ---------------------------------------------------------------------------

_AGENT_DIR = Path(__file__).resolve().parents[2] / "agent"

_OUTPUT_FORMAT_INSTRUCTIONS = """\

## Output Format

You MUST respond with a JSON object containing a "steps" array.
Each step has:
- "id": unique string identifier (e.g. "step-1", "home", "aspirate-sample")
- "primitive": exact primitive name from the capabilities list above
- "params": object with parameter values matching the primitive's parameter spec
- "depends_on": array of step ids that must complete before this step (empty array if none)

Optionally include a "reasoning" field (string) explaining your experimental design.

Example:
```json
{
  "reasoning": "First home the robot, then load pipettes before any liquid handling.",
  "steps": [
    {"id": "step-1", "primitive": "robot.home", "params": {}, "depends_on": []},
    {"id": "step-2", "primitive": "robot.load_pipettes", "params": {"left": "p300_single"}, "depends_on": ["step-1"]}
  ]
}
```

Respond ONLY with the JSON object. Do not include any other text outside the JSON.
"""


def build_system_prompt() -> str:
    """Assemble the system prompt from SOUL.md + capabilities registry."""
    parts: list[str] = []

    # 1. Agent identity / safety boundaries
    soul_path = _AGENT_DIR / "SOUL.md"
    if soul_path.exists():
        parts.append(soul_path.read_text(encoding="utf-8").strip())
    else:
        parts.append("You are a laboratory experiment planner. Safety is non-negotiable.")

    parts.append("")  # blank line separator

    # 2. Available primitives from the registry
    registry = get_registry()
    parts.append(registry.summary_for_llm())

    # 3. Output format instructions
    parts.append(_OUTPUT_FORMAT_INSTRUCTIONS)

    # 4. Memory context (advisory — never blocks planning)
    try:
        from app.services.memory import format_memory_for_prompt

        memory_context = format_memory_for_prompt()
        if memory_context:
            parts.append(memory_context)
    except Exception:
        pass  # Memory is advisory — never block planning

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

# Match ```json ... ``` or ``` ... ``` code blocks
_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def parse_plan_response(raw: str, *, model: str = "unknown") -> PlanResult:
    """Extract a ``PlanResult`` from raw LLM text.

    Supports:
    - Raw JSON object at the top level
    - JSON wrapped in a ```json ... ``` code block
    """
    text = raw.strip()

    # Try to extract from code block first
    match = _CODE_BLOCK_RE.search(text)
    json_text = match.group(1).strip() if match else text

    # If it doesn't start with '{', try to find the first '{' ... last '}'
    if not json_text.startswith("{"):
        brace_start = json_text.find("{")
        brace_end = json_text.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            json_text = json_text[brace_start : brace_end + 1]
        else:
            raise PlanParseError(f"No JSON object found in LLM response: {text[:200]}")

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise PlanParseError(f"Invalid JSON in LLM response: {exc}") from exc

    if not isinstance(data, dict):
        raise PlanParseError(f"Expected JSON object, got {type(data).__name__}")

    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise PlanParseError("LLM response must contain a non-empty 'steps' array")

    steps: list[PlanStep] = []
    for i, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, dict):
            raise PlanParseError(f"Step {i}: expected object, got {type(raw_step).__name__}")
        step_id = str(raw_step.get("id", f"step-{i + 1}"))
        primitive = raw_step.get("primitive")
        if not primitive:
            raise PlanParseError(f"Step {step_id}: 'primitive' is required")
        steps.append(
            PlanStep(
                id=step_id,
                primitive=str(primitive),
                params=raw_step.get("params", {}),
                depends_on=raw_step.get("depends_on", []),
            )
        )

    return PlanResult(
        steps=steps,
        raw_response=raw,
        model=model,
        reasoning=data.get("reasoning"),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def plan_from_intent(
    intent: str,
    *,
    provider: LLMProvider | None = None,
) -> PlanResult:
    """Convert a natural language experiment intent into a structured plan.

    Parameters
    ----------
    intent:
        Natural language description of the desired experiment.
    provider:
        Optional LLM provider override (default: ``get_llm_provider()``).

    Returns
    -------
    PlanResult
        Structured plan with steps, raw response, and metadata.

    Raises
    ------
    LLMError
        If the LLM call fails.
    PlanParseError
        If the LLM response cannot be parsed.
    """
    llm = provider or get_llm_provider()
    system = build_system_prompt()

    response = await llm.complete(
        messages=[LLMMessage(role="user", content=intent)],
        system=system,
    )

    return parse_plan_response(response.content, model=response.model)
