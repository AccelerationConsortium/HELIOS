"""
OT-2 NLP Agent - Natural Language Interface for Opentrons OT-2 Robot

Allows non-programmers to design liquid handling workflows using natural language.
Supports both English and Chinese instructions.

Example (Traditional - Direct Instructions):
    from ot2_agent import OT2Agent

    agent = OT2Agent()
    protocol = agent.create_protocol("My Protocol")
    agent.add_labware(protocol, "96孔板", slot=1)
    agent.add_instruction(protocol, "从A1孔吸取100微升")
    code = agent.generate(protocol)

Example (New - Planner/Compiler):
    from ot2_agent import OT2Agent

    agent = OT2Agent()

    # Step 1: Plan - describe what you want
    output = agent.plan("我想做OER测量，用的是NiFe催化剂")

    # Step 2: Review candidates and fill parameters
    draft = output.candidates[0]
    params = {'electrode_area_cm2': 0.196, ...}

    # Step 3: Compile to executable code
    from ot2_agent.planner import ConfirmedWorkflow
    confirmed = ConfirmedWorkflow(draft=draft, filled_parameters=params)
    result = agent.compile(confirmed)
    print(result.python_code)
"""

from .agent import OT2Agent
from .parser import NLParser
from .operations import OperationMapper
from .protocol import ProtocolGenerator
from .validator import ProtocolValidator
from .custom_labware import (
    CustomLabwareDefinition,
    CustomLabwareManager,
    get_labware_manager,
)

# New Planner/Compiler exports
from .planner import Planner, PlannerOutput, WorkflowDraft, ConfirmedWorkflow
from .compiler import Compiler, CompilerOutput

# IR exports
from .ir import Intent, UnitOperation, Primitive, UOType, ActionType

__version__ = "0.2.0"
__all__ = [
    # Core
    "OT2Agent",
    "NLParser",
    "OperationMapper",
    "ProtocolGenerator",
    "ProtocolValidator",
    "CustomLabwareDefinition",
    "CustomLabwareManager",
    "get_labware_manager",
    # Planner/Compiler
    "Planner",
    "PlannerOutput",
    "WorkflowDraft",
    "ConfirmedWorkflow",
    "Compiler",
    "CompilerOutput",
    # IR
    "Intent",
    "UnitOperation",
    "Primitive",
    "UOType",
    "ActionType",
]
