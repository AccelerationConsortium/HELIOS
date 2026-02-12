"""
Main Compiler class - UO workflows to executable code.

The Compiler takes a confirmed workflow and generates
executable code and JSON.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..ir import UnitOperation, Primitive, DeviceAction
from ..protocol import Protocol, ProtocolGenerator
from ..validator import ValidationResult, ProtocolValidator
from ..validation.workflow_validator import WorkflowValidator, EnhancedValidationResult
from ..planner import ConfirmedWorkflow

from .uo_expander import UOExpander
from .device_mapper import DeviceMapper, DeviceRegistry


@dataclass
class CompilerOutput:
    """
    Output from the Compiler.

    Contains all generated artifacts.
    """
    protocol: Protocol
    python_code: str
    workflow_json: str
    validation_result: EnhancedValidationResult
    device_actions: List[DeviceAction] = field(default_factory=list)
    primitives: List[Primitive] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "protocol": self.protocol.to_dict(),
            "python_code": self.python_code,
            "workflow_json": self.workflow_json,
            "validation": self.validation_result.to_dict(),
            "device_actions": [a.to_dict() for a in self.device_actions],
            "primitives": [p.to_dict() for p in self.primitives],
        }


class Compiler:
    """
    Main Compiler class.

    Compiles confirmed UO workflows to executable code/JSON.
    """

    def __init__(self, device_registry: DeviceRegistry = None):
        """
        Initialize the Compiler.

        Args:
            device_registry: Registry of available devices.
                           Creates default if None.
        """
        self.device_registry = device_registry or DeviceRegistry()
        self.uo_expander = UOExpander()
        self.device_mapper = DeviceMapper(self.device_registry)
        self.generator = ProtocolGenerator()
        self.validator = WorkflowValidator()

    def compile(self, workflow: ConfirmedWorkflow) -> CompilerOutput:
        """
        Main compilation method.

        Args:
            workflow: User-confirmed workflow with filled parameters

        Returns:
            CompilerOutput with code, JSON, and validation
        """
        # Step 1: Get UOs with filled parameters
        filled_uos = workflow.get_filled_unit_operations()

        # Step 2: Expand UOs to primitives
        primitives = self.uo_expander.expand(filled_uos)

        # Step 3: Map primitives to device actions
        device_actions = self.device_mapper.map(primitives)

        # Step 4: Build Protocol object
        protocol = self._build_protocol(workflow, device_actions)

        # Step 5: Validate with enhanced validation (includes checkpoints)
        validation_result = self.validator.validate(
            protocol=protocol,
            unit_operations=filled_uos,
            device_actions=device_actions
        )

        # Step 6: Generate code and JSON
        python_code = self.generator.generate(protocol)
        workflow_json = self.generator.to_json(protocol)

        return CompilerOutput(
            protocol=protocol,
            python_code=python_code,
            workflow_json=workflow_json,
            validation_result=validation_result,
            device_actions=device_actions,
            primitives=primitives,
        )

    def _build_protocol(
        self,
        workflow: ConfirmedWorkflow,
        device_actions: List[DeviceAction]
    ) -> Protocol:
        """
        Build a Protocol object from device actions.

        This bridges the new system with the existing Protocol format.
        """
        # Create protocol
        protocol = Protocol(
            name=workflow.draft.name,
            description=workflow.draft.description,
        )

        # Add default labware and pipette for OT-2 operations
        # (In a full implementation, this would be derived from device_actions)
        has_liquid_handling = any(
            a.device_type == "liquid_handler" for a in device_actions
        )

        if has_liquid_handling:
            protocol.add_labware("plate", "corning_96_wellplate_360ul_flat", 1)
            protocol.add_labware("tips", "opentrons_96_tiprack_300ul", 2)
            protocol.add_pipette("pipette", "p300_single_gen2", "left", "tips")

        # Convert device actions to operations
        operations = self.device_mapper.to_operations(device_actions)
        for op in operations:
            if op:  # Skip None operations
                protocol.add_operation(op)

        # Store original workflow info in metadata
        protocol.metadata["workflow_draft"] = workflow.draft.name
        protocol.metadata["filled_parameters"] = workflow.filled_parameters
        protocol.metadata["uo_count"] = len(workflow.draft.unit_operations)
        protocol.metadata["device_action_count"] = len(device_actions)

        # Store original instructions for documentation
        protocol.original_instructions = [
            uo.get_description() for uo in workflow.draft.unit_operations
        ]

        return protocol

    def compile_from_dict(self, workflow_dict: Dict[str, Any]) -> CompilerOutput:
        """
        Compile from a dictionary representation.

        Convenience method for API usage.
        """
        workflow = ConfirmedWorkflow.from_dict(workflow_dict)
        return self.compile(workflow)
