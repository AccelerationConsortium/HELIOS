"""
Intermediate Representation (IR) layer for OT2 NLP Agent.

Three-layer IR architecture:
1. Intent - User's high-level goal
2. UnitOperation - Domain semantic modules
3. Primitive - Atomic device-agnostic actions
"""

from .intent import Intent, MissingInfo, PlanningContext
from .unit_operations import UnitOperation, UOType, Placeholder
from .primitives import Primitive, ActionType, DeviceAction

__all__ = [
    # Intent layer
    "Intent",
    "MissingInfo",
    "PlanningContext",
    # UO layer
    "UnitOperation",
    "UOType",
    "Placeholder",
    # Primitive layer
    "Primitive",
    "ActionType",
    "DeviceAction",
]
