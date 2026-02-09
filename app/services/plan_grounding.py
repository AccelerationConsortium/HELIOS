"""Deterministic grounding — converts an LLM PlanResult into protocol JSON.

Responsibilities:
1. Validate each step's primitive exists in the capabilities registry
2. Coerce parameter types (LLM often returns numbers as strings)
3. Auto-map resources from the registry (``PrimitiveSpec.resource_id``)
4. Check required parameters are present
5. Produce a ``protocol`` dict ready for ``compile_protocol()``
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

from app.services.planner import PlanResult
from app.services.primitives_registry import PrimitiveParam, PrimitiveSpec, get_registry


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GroundingResult:
    """Outcome of grounding a plan."""

    protocol: dict[str, Any] = field(default_factory=dict)  # ready for compile_protocol
    warnings: list[str] = field(default_factory=list)  # non-fatal (type coercions, etc.)
    errors: list[str] = field(default_factory=list)  # fatal (unknown primitive, etc.)

    @property
    def ok(self) -> bool:
        """True when there are no fatal errors."""
        return not self.errors


# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------


def _coerce_param(value: Any, target_type: str, param_name: str) -> tuple[Any, str | None]:
    """Attempt to coerce *value* to *target_type*.

    Returns ``(coerced_value, warning_or_None)``.
    """
    if value is None:
        return value, None

    try:
        if target_type == "number":
            if isinstance(value, (int, float)):
                return float(value), None
            return float(value), f"coerced '{param_name}' from {type(value).__name__} to number"
        if target_type == "integer":
            if isinstance(value, int) and not isinstance(value, bool):
                return value, None
            return int(value), f"coerced '{param_name}' from {type(value).__name__} to integer"
        if target_type == "boolean":
            if isinstance(value, bool):
                return value, None
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes"), (
                    f"coerced '{param_name}' from string to boolean"
                )
            return bool(value), f"coerced '{param_name}' from {type(value).__name__} to boolean"
        if target_type == "array":
            if isinstance(value, list):
                return value, None
            return [value], f"coerced '{param_name}' from {type(value).__name__} to array"
        # "string" or unknown type — keep as-is
        return value, None
    except (ValueError, TypeError):
        return value, f"could not coerce '{param_name}' to {target_type}, keeping original"


# ---------------------------------------------------------------------------
# Main grounding function
# ---------------------------------------------------------------------------


def ground_plan(plan: PlanResult) -> GroundingResult:
    """Convert a ``PlanResult`` into a protocol dict suitable for ``compile_protocol``.

    Steps:
    1. Look up each primitive in the registry
    2. Validate and coerce parameters
    3. Auto-fill resources from the registry
    4. Build the final protocol JSON
    """
    registry = get_registry()
    result = GroundingResult()
    grounded_steps: list[dict[str, Any]] = []

    for step in plan.steps:
        spec: PrimitiveSpec | None = registry.get_primitive(step.primitive)

        if spec is None:
            result.errors.append(f"step '{step.id}': unknown primitive '{step.primitive}'")
            continue

        # --- Parameter validation and coercion ---
        coerced_params: dict[str, Any] = {}
        spec_params: dict[str, PrimitiveParam] = {p.name: p for p in spec.params}

        for param_name, param_value in step.params.items():
            param_spec = spec_params.get(param_name)
            if param_spec is not None:
                coerced, warn = _coerce_param(param_value, param_spec.type, param_name)
                coerced_params[param_name] = coerced
                if warn:
                    result.warnings.append(f"step '{step.id}': {warn}")
            else:
                # Unknown param — keep it (the compiler/safety layer will handle it)
                coerced_params[param_name] = param_value
                result.warnings.append(
                    f"step '{step.id}': unknown parameter '{param_name}' for '{step.primitive}'"
                )

        # Check required parameters are present
        for pname, pspec in spec_params.items():
            if not pspec.optional and pname not in coerced_params:
                result.errors.append(
                    f"step '{step.id}': required parameter '{pname}' missing "
                    f"for '{step.primitive}'"
                )

        # --- Fill missing optional params from memory priors ---
        try:
            from app.services.memory import get_param_priors

            for pname, pspec in spec_params.items():
                if pspec.optional and pname not in coerced_params:
                    prior = get_param_priors(step.primitive, pname)
                    if prior is not None and prior.sample_count >= 3:
                        coerced_params[pname] = prior.mean
                        result.warnings.append(
                            f"step '{step.id}': filled '{pname}' from memory prior "
                            f"(mean={prior.mean:.2f}, n={prior.sample_count})"
                        )
        except Exception:
            pass  # Memory is advisory — never block grounding

        # --- Auto-map resources ---
        resources: list[str] = []
        if spec.resource_id:
            resources.append(spec.resource_id)

        grounded_steps.append(
            {
                "id": step.id,
                "primitive": step.primitive,
                "params": coerced_params,
                "depends_on": step.depends_on,
                "resources": resources,
            }
        )

    result.protocol = {"steps": grounded_steps}
    return result
