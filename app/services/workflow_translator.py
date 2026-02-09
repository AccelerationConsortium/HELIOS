"""Translate refactored_battery phase-based workflow JSON into OTbot protocol format.

The battery-lab workflow uses a phase/parallel_threads structure:

    {"phases": [
        {"phase_name": "setup", "steps": [...]},
        {"phase_name": "prep", "parallel_threads": [
            {"thread_name": "t1", "steps": [...]},
            {"thread_name": "t2", "steps": [...]},
        ]},
    ]}

OTbot's compiler expects a flat list of steps with explicit ``depends_on``:

    {"steps": [
        {"step_key": "setup_001", "primitive": "robot.home", "params": {}, "depends_on": []},
        ...
    ]}

This module converts between the two formats preserving execution semantics:
  - Steps within the same sequential list depend on the previous step.
  - The first step(s) of a phase depend on the last step(s) of the previous phase.
  - Within ``parallel_threads``, threads are independent but each thread's
    steps are sequentially chained.
  - The first step(s) of the next phase depend on *all* last steps of the
    previous phase's threads (join barrier).

Resources are auto-mapped from the primitive prefix:
  ``robot.*`` → ``ot2-robot``
  ``plc.*``   → ``plc-controller``
  ``relay.*`` → ``relay-controller``
  ``squidstat.*`` → ``squidstat``
  Otherwise   → ``lab-controller``
"""
from __future__ import annotations

from typing import Any


class TranslationError(ValueError):
    """Raised when the workflow JSON cannot be translated."""


def _resource_for_primitive(primitive: str) -> list[str]:
    """Derive resource list from the dotted-prefix naming convention."""
    prefix = primitive.split(".")[0] if "." in primitive else ""
    mapping: dict[str, str] = {
        "robot": "ot2-robot",
        "plc": "plc-controller",
        "relay": "relay-controller",
        "squidstat": "squidstat",
        "cleanup": "ot2-robot",
        "sample": "ot2-robot",
        "ssh": "ssh-streamer",
    }
    resource = mapping.get(prefix, "lab-controller")
    return [resource]


def translate_battery_workflow(workflow: dict[str, Any]) -> dict[str, Any]:
    """Convert a phase-based battery-lab workflow into an OTbot protocol dict.

    Args:
        workflow: The raw JSON object (parsed) with ``phases`` key.

    Returns:
        A dict compatible with ``compile_protocol(protocol=...)``, i.e.
        ``{"steps": [...]}``.

    Raises:
        TranslationError: If the workflow structure is invalid.
    """
    phases = workflow.get("phases")
    if not phases or not isinstance(phases, list):
        raise TranslationError("workflow must contain a non-empty 'phases' list")

    all_steps: list[dict[str, Any]] = []
    # Track the last step_key(s) from the previous phase for cross-phase deps
    prev_phase_tail_keys: list[str] = []

    for phase in phases:
        phase_name = phase.get("phase_name", "unnamed")

        if "parallel_threads" in phase:
            tail_keys = _translate_parallel_phase(
                phase, prev_phase_tail_keys, all_steps,
            )
        elif "steps" in phase:
            tail_keys = _translate_sequential_phase(
                phase, prev_phase_tail_keys, all_steps,
            )
        else:
            raise TranslationError(
                f"phase '{phase_name}' has neither 'steps' nor 'parallel_threads'"
            )

        prev_phase_tail_keys = tail_keys

    return {"steps": all_steps}


def _translate_sequential_phase(
    phase: dict[str, Any],
    prev_tail_keys: list[str],
    out: list[dict[str, Any]],
) -> list[str]:
    """Translate a phase with sequential ``steps`` list.

    Returns the list of tail step_keys (will be length 1 for sequential).
    """
    steps = phase.get("steps", [])
    if not steps:
        return prev_tail_keys  # empty phase, carry forward

    prev_key: str | None = None

    for i, raw_step in enumerate(steps):
        step_key = raw_step.get("step_id") or f"{phase['phase_name']}_{i:03d}"
        primitive = raw_step.get("action")
        if not primitive:
            raise TranslationError(f"step '{step_key}' is missing 'action'")

        depends_on: list[str] = []
        if prev_key is not None:
            depends_on = [prev_key]
        elif prev_tail_keys:
            depends_on = list(prev_tail_keys)

        translated: dict[str, Any] = {
            "step_key": step_key,
            "primitive": primitive,
            "params": dict(raw_step.get("params", {})),
            "depends_on": depends_on,
            "resources": _resource_for_primitive(primitive),
        }
        out.append(translated)
        prev_key = step_key

    # last step is the tail
    return [prev_key] if prev_key else prev_tail_keys


def _translate_parallel_phase(
    phase: dict[str, Any],
    prev_tail_keys: list[str],
    out: list[dict[str, Any]],
) -> list[str]:
    """Translate a phase with ``parallel_threads``.

    - Each thread's steps are sequentially chained.
    - The first step of each thread depends on prev_tail_keys (fork barrier).
    - Returns all thread-tail keys (join barrier for next phase).
    """
    threads = phase.get("parallel_threads", [])
    if not threads:
        return prev_tail_keys

    tail_keys: list[str] = []

    for thread in threads:
        thread_name = thread.get("thread_name", "thread")
        steps = thread.get("steps", [])
        prev_key: str | None = None

        for i, raw_step in enumerate(steps):
            step_key = raw_step.get("step_id") or f"{thread_name}_{i:03d}"
            primitive = raw_step.get("action")
            if not primitive:
                raise TranslationError(f"step '{step_key}' missing 'action'")

            depends_on: list[str] = []
            if prev_key is not None:
                depends_on = [prev_key]
            elif prev_tail_keys:
                # First step of this thread depends on previous phase's tails
                depends_on = list(prev_tail_keys)

            translated: dict[str, Any] = {
                "step_key": step_key,
                "primitive": primitive,
                "params": dict(raw_step.get("params", {})),
                "depends_on": depends_on,
                "resources": _resource_for_primitive(primitive),
            }
            out.append(translated)
            prev_key = step_key

        if prev_key is not None:
            tail_keys.append(prev_key)

    return tail_keys if tail_keys else prev_tail_keys
