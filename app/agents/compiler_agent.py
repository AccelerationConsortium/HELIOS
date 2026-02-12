"""Protocol Compiler Agent -- L1 compilation layer.

Wraps existing compiler.py to compile protocol + params into
an executable DAG with graph hash.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.agents.base import BaseAgent


class CompileInput(BaseModel):
    """Input for protocol compilation."""
    protocol: dict[str, Any]
    inputs: dict[str, Any] = Field(default_factory=dict)
    policy_snapshot: dict[str, Any] = Field(default_factory=dict)


class CompileOutput(BaseModel):
    """Output from protocol compilation."""
    compiled_graph: dict[str, Any]
    graph_hash: str
    n_steps: int
    step_keys: list[str]
    deck_plan: dict[str, Any] = Field(default_factory=dict)
    layout_warnings: list[str] = Field(default_factory=list)


class CompilerAgent(BaseAgent[CompileInput, CompileOutput]):
    name = "compiler_agent"
    description = "Protocol -> DAG compilation"
    layer = "L1"

    def validate_input(self, input_data: CompileInput) -> list[str]:
        errors: list[str] = []
        if "steps" not in input_data.protocol:
            errors.append("protocol must contain 'steps'")
        return errors

    async def process(self, input_data: CompileInput) -> CompileOutput:
        from app.services.compiler import compile_protocol
        from app.services.deck_layout import plan_deck_layout, validate_deck_layout

        compiled, graph_hash = compile_protocol(
            protocol=input_data.protocol,
            inputs=input_data.inputs,
            policy_snapshot=input_data.policy_snapshot,
        )

        steps = compiled.get("steps", [])

        # Deck layout planning and validation
        batch_size = input_data.inputs.get("batch_size", 1)
        deck_plan = plan_deck_layout(steps, batch_size=batch_size)
        layout_validation = validate_deck_layout(
            deck_plan, steps, input_data.policy_snapshot, batch_size,
        )

        layout_warnings = list(layout_validation.warnings)
        if layout_validation.errors:
            layout_warnings.extend(
                f"[ERROR] {e}" for e in layout_validation.errors
            )

        return CompileOutput(
            compiled_graph=compiled,
            graph_hash=graph_hash,
            n_steps=len(steps),
            step_keys=[s["step_key"] for s in steps],
            deck_plan=deck_plan.to_dict(),
            layout_warnings=layout_warnings,
        )
