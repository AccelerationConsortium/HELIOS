from __future__ import annotations

import hashlib
from typing import Any

from app.core.db import json_dumps


class CompileError(ValueError):
    pass


def compile_protocol(
    *, protocol: dict[str, Any], inputs: dict[str, Any], policy_snapshot: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    if "steps" not in protocol or not isinstance(protocol["steps"], list):
        raise CompileError("protocol.steps must be a list")

    raw_steps = protocol["steps"]
    graph_steps: list[dict[str, Any]] = []
    seen = set()

    for index, raw in enumerate(raw_steps):
        step_key = str(raw.get("step_key", raw.get("id", f"step-{index + 1}")))
        if step_key in seen:
            raise CompileError(f"duplicate step id: {step_key}")
        seen.add(step_key)

        primitive = raw.get("primitive")
        if not primitive:
            raise CompileError(f"step {step_key}: primitive is required")

        depends_on = raw.get("depends_on", [])
        if not isinstance(depends_on, list):
            raise CompileError(f"step {step_key}: depends_on must be a list")

        resources = raw.get("resources", [])
        if not isinstance(resources, list):
            raise CompileError(f"step {step_key}: resources must be a list")

        graph_steps.append(
            {
                "step_key": step_key,
                "primitive": primitive,
                "params": raw.get("params", {}),
                "depends_on": depends_on,
                "resources": resources,
            }
        )

    all_step_keys = {step["step_key"] for step in graph_steps}
    for step in graph_steps:
        for dep in step["depends_on"]:
            if dep not in all_step_keys:
                raise CompileError(
                    f"step {step['step_key']}: dependency {dep} does not exist"
                )

    # --- cycle detection via Kahn's topological sort ---
    adj: dict[str, list[str]] = {s["step_key"]: list(s["depends_on"]) for s in graph_steps}
    in_degree: dict[str, int] = {k: 0 for k in all_step_keys}
    for step in graph_steps:
        for dep in step["depends_on"]:
            in_degree[step["step_key"]] += 1

    queue = [k for k, d in in_degree.items() if d == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for step in graph_steps:
            if node in step["depends_on"]:
                in_degree[step["step_key"]] -= 1
                if in_degree[step["step_key"]] == 0:
                    queue.append(step["step_key"])

    if visited != len(all_step_keys):
        raise CompileError("circular dependency detected in protocol steps")

    compiled = {
        "kind": "linearized_dag",
        "version": 1,
        "steps": sorted(graph_steps, key=lambda s: s["step_key"]),
        "input_bindings": inputs,
        "policy_snapshot": policy_snapshot,
    }

    canonical = json_dumps(
        {
            "protocol": protocol,
            "inputs": inputs,
            "policy_snapshot": policy_snapshot,
            "compiled": compiled,
        }
    )
    graph_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return compiled, graph_hash
