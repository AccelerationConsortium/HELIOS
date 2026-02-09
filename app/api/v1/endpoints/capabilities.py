"""Capabilities API — exposes the primitives registry for LLM and UI consumption."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from app.services.primitives_registry import get_registry

router = APIRouter(tags=["capabilities"])


def _primitive_to_dict(p: Any) -> dict[str, Any]:
    """Serialize a PrimitiveSpec to a dict with contract and safety_class fields."""
    result: dict[str, Any] = {
        "name": p.name,
        "description": p.description,
        "error_class": p.error_class,
        "safety_class": p.safety_class.name,
        "instrument": p.instrument,
        "resource_id": p.resource_id,
        "skill": p.skill_name,
        "params": [
            {
                "name": pp.name,
                "type": pp.type,
                "description": pp.description,
                "optional": pp.optional,
                **({"default": pp.default} if pp.default is not None else {}),
            }
            for pp in p.params
        ],
    }
    # Include contract details when available
    if p.contract is not None:
        result["contract"] = {
            "preconditions": [pc.predicate for pc in p.contract.preconditions],
            "effects": [e.operation for e in p.contract.effects],
            "timeout": {
                "seconds": p.contract.timeout.seconds,
                "retries": p.contract.timeout.retries,
            },
        }
    else:
        result["contract"] = None
    return result


@router.get("/capabilities")
async def list_capabilities() -> dict[str, Any]:
    """Return the full primitives catalogue with all skills and parameters."""
    registry = get_registry()
    return registry.to_dict()


@router.get("/capabilities/primitives")
async def list_primitives(
    instrument: str | None = Query(None, description="Filter by instrument"),
    error_class: str | None = Query(None, description="Filter by error class (CRITICAL or BYPASS)"),
    safety_class: str | None = Query(None, description="Filter by safety class (INFORMATIONAL, REVERSIBLE, CAREFUL, HAZARDOUS)"),
) -> dict[str, Any]:
    """List all primitives with optional filtering."""
    registry = get_registry()

    if instrument:
        prims = registry.primitives_by_instrument(instrument)
    elif error_class:
        prims = registry.primitives_by_error_class(error_class.upper())
    elif safety_class:
        prims = registry.primitives_by_safety_class(safety_class.upper())
    else:
        prims = registry.list_primitives()

    return {
        "primitives": [_primitive_to_dict(p) for p in prims],
        "count": len(prims),
    }


@router.get("/capabilities/primitives/{primitive_name:path}")
async def get_primitive(primitive_name: str) -> dict[str, Any]:
    """Get detailed information about a specific primitive."""
    registry = get_registry()
    p = registry.get_primitive(primitive_name)
    if p is None:
        return {"error": f"Primitive '{primitive_name}' not found", "found": False}
    return {
        "found": True,
        "primitive": _primitive_to_dict(p),
    }


@router.get("/capabilities/summary")
async def capabilities_summary() -> dict[str, str]:
    """Return an LLM-friendly text summary of all capabilities."""
    registry = get_registry()
    return {"summary": registry.summary_for_llm()}
