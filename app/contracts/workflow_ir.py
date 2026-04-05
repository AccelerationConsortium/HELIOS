"""Workflow Intermediate Representation (IR).

WorkflowGraph is the contract between PlannerAgent (L2) and ExecutionAgent (L1).
It captures scientist intent as an abstract step DAG, decoupled from any specific
execution backend (Opentrons MCP, Python API, simulation).

Layer flow:
    PlannerAgent  →  WorkflowGraph  →  ExecutionAgent  →  RunBundle
"""
from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── String-literal type aliases ──────────────────────────────────────────────

AbstractPrimitive = Literal[
    "liquid_transfer",
    "mix",
    "aspirate",
    "dispense",
    "pick_up_tip",
    "drop_tip",
    "incubate",
    "heat",
    "cool",
    "shake",
    "spin",
    "seal_plate",
    "unseal_plate",
    "image_plate",
    "read_absorbance",
    "read_fluorescence",
    "measure_ph",
    "wait",
    "home",
    "custom",
]

ExecutionBackend = Literal[
    "opentrons_mcp",
    "python_api",
    "simulation",
]

WorkflowStatus = Literal[
    "draft",
    "validated",
    "queued",
    "running",
    "completed",
    "failed",
    "cancelled",
]


# ── Step model ────────────────────────────────────────────────────────────────

class WorkflowStep(BaseModel):
    """One abstract, executable step in a workflow graph.

    Keeps intent and parameters separate from execution details — the
    ExecutionAgent is responsible for translating these to backend commands.
    """

    step_id: str = Field(description="Unique ID within the graph, e.g. 's-0001'")
    intent: str = Field(description="Human-readable description of what this step does")
    abstract_primitive: AbstractPrimitive
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution parameters, e.g. {volume_ul: 100, source: 'A1', dest: 'B1'}",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Execution hints, e.g. ['post_mix', 'air_gap', 'slow_aspirate']",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="step_id list of steps that must complete before this one starts",
    )
    estimated_duration_s: float | None = None
    labware_refs: list[str] = Field(
        default_factory=list,
        description="Labware slot keys referenced by this step (e.g. 'slot_1', 'slot_4')",
    )


# ── Graph model ───────────────────────────────────────────────────────────────

class WorkflowGraph(BaseModel):
    """Abstract workflow DAG emitted by PlannerAgent, consumed by ExecutionAgent.

    This IR decouples scientific intent from the concrete execution backend.
    A WorkflowGraph can be compiled to any supported ExecutionBackend without
    changing the steps themselves.
    """

    graph_id: str = Field(default_factory=lambda: f"wg-{uuid.uuid4().hex[:12]}")
    campaign_id: str
    round_number: int = Field(ge=0)
    steps: list[WorkflowStep] = Field(default_factory=list)
    preferred_backend: ExecutionBackend = "opentrons_mcp"
    status: WorkflowStatus = "draft"
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ── helpers ───────────────────────────────────────────────────────────

    def step_by_id(self, step_id: str) -> WorkflowStep | None:
        """Return the step with the given ID, or None."""
        for s in self.steps:
            if s.step_id == step_id:
                return s
        return None

    def topological_order(self) -> list[WorkflowStep]:
        """Return steps in dependency-safe execution order via Kahn's algorithm.

        Raises:
            ValueError: If the graph contains a cycle.
        """
        in_degree: dict[str, int] = {s.step_id: 0 for s in self.steps}
        dependents: dict[str, list[str]] = {s.step_id: [] for s in self.steps}

        for step in self.steps:
            for dep in step.dependencies:
                in_degree[step.step_id] = in_degree.get(step.step_id, 0) + 1
                dependents.setdefault(dep, []).append(step.step_id)

        queue = [sid for sid, deg in in_degree.items() if deg == 0]
        result: list[WorkflowStep] = []

        while queue:
            current = queue.pop(0)
            node = self.step_by_id(current)
            if node:
                result.append(node)
            for child in dependents.get(current, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(result) != len(self.steps):
            raise ValueError(
                f"WorkflowGraph {self.graph_id} contains a cycle; "
                f"only {len(result)}/{len(self.steps)} steps are reachable."
            )
        return result

    def primitives_used(self) -> set[str]:
        """Return the set of abstract primitives used across all steps."""
        return {s.abstract_primitive for s in self.steps}

    def estimated_total_duration_s(self) -> float | None:
        """Sum of estimated durations; None if any step has no estimate."""
        total = 0.0
        for s in self.steps:
            if s.estimated_duration_s is None:
                return None
            total += s.estimated_duration_s
        return total


# ── ID factory ────────────────────────────────────────────────────────────────

def new_workflow_graph_id() -> str:
    """Generate a fresh WorkflowGraph ID."""
    return f"wg-{uuid.uuid4().hex[:12]}"
