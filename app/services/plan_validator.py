"""Static analysis for LLM-generated plans — checks beyond compiler/safety.

Produces warnings and informational notes (never blocks run creation).
Checks include:
- Unreachable steps (depends_on references non-existent step ids)
- No-root detection (all steps have dependencies — possible cycle hint)
- Safety classification summary (HAZARDOUS / CAREFUL steps listed)
- Redundant operation warnings (consecutive identical primitives)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.services.primitives_registry import get_registry


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Result of static plan validation."""

    warnings: list[str] = field(default_factory=list)  # potential issues
    info: list[str] = field(default_factory=list)  # informational (safety summary, etc.)

    @property
    def ok(self) -> bool:
        """Always True — warnings never block execution."""
        return True


# ---------------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------------


def validate_plan(protocol: dict[str, Any]) -> ValidationResult:
    """Run static analysis on a grounded protocol dict.

    Parameters
    ----------
    protocol:
        A protocol dict with a ``"steps"`` list, as produced by
        ``plan_grounding.ground_plan()``.

    Returns
    -------
    ValidationResult
        Warnings and informational notes.
    """
    result = ValidationResult()
    steps = protocol.get("steps", [])
    if not steps:
        result.warnings.append("protocol has no steps")
        return result

    step_ids = {s.get("id", s.get("step_key", f"step-{i}")) for i, s in enumerate(steps)}

    # --- 1. Unreachable step detection ---
    for step in steps:
        step_id = step.get("id", step.get("step_key", "?"))
        for dep in step.get("depends_on", []):
            if dep not in step_ids:
                result.warnings.append(
                    f"step '{step_id}': depends on '{dep}' which does not exist"
                )

    # --- 2. No-root detection ---
    roots = [
        s for s in steps if not s.get("depends_on")
    ]
    if not roots and len(steps) > 1:
        result.warnings.append(
            "no root step found (all steps have dependencies) — possible cycle"
        )

    # --- 3. Safety classification summary ---
    registry = get_registry()
    hazardous: list[str] = []
    careful: list[str] = []

    for step in steps:
        primitive = step.get("primitive", "")
        spec = registry.get_primitive(primitive)
        if spec is None:
            continue
        safety_name = spec.safety_class.name
        step_id = step.get("id", step.get("step_key", "?"))
        if safety_name == "HAZARDOUS":
            hazardous.append(f"{step_id} ({primitive})")
        elif safety_name == "CAREFUL":
            careful.append(f"{step_id} ({primitive})")

    if hazardous:
        result.info.append(f"HAZARDOUS steps requiring extra caution: {', '.join(hazardous)}")
    if careful:
        result.info.append(f"CAREFUL steps: {', '.join(careful)}")

    # --- 4. Redundant operation warning ---
    prev_primitive: str | None = None
    for step in steps:
        primitive = step.get("primitive", "")
        step_id = step.get("id", step.get("step_key", "?"))
        if primitive == prev_primitive:
            result.warnings.append(
                f"step '{step_id}': consecutive '{primitive}' — verify this is intentional"
            )
        prev_primitive = primitive

    return result
