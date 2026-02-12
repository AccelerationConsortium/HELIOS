"""
OT-2 NLP Agent - Main interface for natural language protocol creation.
"""

from typing import Any, Dict, List, Optional, Union

from .operations import Operation, OperationMapper, OperationType
from .parser import NLParser, ParsedIntent
from .protocol import LabwareConfig, PipetteConfig, Protocol, ProtocolGenerator
from .validator import ProtocolValidator, ValidationResult
from .custom_labware import CustomLabwareDefinition, CustomLabwareManager, get_labware_manager

# New Planner/Compiler imports
from .planner import Planner, PlannerOutput, WorkflowDraft, ConfirmedWorkflow
from .compiler import Compiler, CompilerOutput


class OT2Agent:
    """
    Natural Language Interface for Opentrons OT-2 Robot.

    Allows users to create liquid handling protocols using natural language
    in English or Chinese.

    Example:
        agent = OT2Agent()

        # Create a new protocol
        protocol = agent.create_protocol("My Transfer Protocol")

        # Add labware
        agent.add_labware(protocol, "96孔板", slot=1, name="plate")
        agent.add_labware(protocol, "tip rack", slot=2, name="tips")

        # Add pipette
        agent.add_pipette(protocol, "p300", mount="left", tip_rack="tips")

        # Parse and add operations from natural language
        agent.add_instructions(protocol, [
            "取枪头",
            "从A1孔吸取100微升",
            "分配到B1-B4孔，每孔25微升",
            "丢弃枪头"
        ])

        # Validate
        result = agent.validate(protocol)
        print(result)

        # Preview
        print(agent.preview(protocol))

        # Generate code
        code = agent.generate(protocol)

        # Save to file
        agent.save(protocol, "my_protocol.py")
    """

    def __init__(self, api_level: str = "2.13"):
        """
        Initialize the OT-2 NLP Agent.

        Args:
            api_level: Opentrons API level (default: 2.13)
        """
        self.api_level = api_level
        self.parser = NLParser()
        self.mapper = OperationMapper()
        self.generator = ProtocolGenerator()
        self.validator = ProtocolValidator()

        # New Planner/Compiler components
        self.planner = Planner()
        self.compiler = Compiler()

    # =========================================================================
    # NEW: Planner/Compiler Methods (Intent -> Workflow -> Code)
    # =========================================================================

    def plan(
        self,
        intent: str,
        context: Dict[str, Any] = None
    ) -> PlannerOutput:
        """
        Generate candidate workflows from user intent.

        This is the first step of the new Planner/Compiler architecture.
        The user describes what they want to do, and the system generates
        candidate workflows for review.

        Args:
            intent: Natural language description of the experiment
                   Example: "我想做OER测量，用的是NiFe催化剂"
            context: Optional dict with known conditions:
                    - devices: List of available devices
                    - materials: Dict of known materials
                    - constraints: Dict of constraints

        Returns:
            PlannerOutput containing:
            - intent: Parsed intent
            - candidates: List of WorkflowDraft candidates
            - recommended_idx: Index of recommended candidate

        Example:
            output = agent.plan("我想做OER测量")
            for i, candidate in enumerate(output.candidates):
                print(f"{i+1}. {candidate.name}")
                print(f"   Missing params: {len(candidate.missing_info)}")
        """
        return self.planner.plan(intent, context)

    def compile(self, workflow: ConfirmedWorkflow) -> CompilerOutput:
        """
        Compile a confirmed workflow to executable code.

        This is the second step after planning. The user selects a
        workflow draft and fills in missing parameters, then this
        method generates the executable code.

        Args:
            workflow: ConfirmedWorkflow with:
                     - draft: The selected WorkflowDraft
                     - filled_parameters: Dict of filled parameter values

        Returns:
            CompilerOutput containing:
            - protocol: Protocol object
            - python_code: Generated Python code
            - workflow_json: Generated JSON
            - validation_result: Validation result
            - device_actions: List of device actions
            - primitives: List of primitives

        Example:
            # After user selects a draft and fills parameters
            confirmed = ConfirmedWorkflow(
                draft=output.candidates[0],
                filled_parameters={'electrode_area_cm2': 0.196, ...}
            )
            result = agent.compile(confirmed)
            print(result.python_code)
        """
        return self.compiler.compile(workflow)

    def plan_and_compile(
        self,
        intent: str,
        parameters: Dict[str, Any],
        context: Dict[str, Any] = None,
        candidate_idx: int = 0
    ) -> CompilerOutput:
        """
        Convenience method: Plan and compile in one step.

        Useful when parameters are already known.

        Args:
            intent: Natural language experiment description
            parameters: Dict of parameter values
            context: Optional planning context
            candidate_idx: Which candidate to use (default: 0, recommended)

        Returns:
            CompilerOutput with generated code

        Example:
            result = agent.plan_and_compile(
                "做OER测量",
                parameters={
                    'electrode_area_cm2': 0.196,
                    'electrode_material': 'NiFe',
                    ...
                }
            )
            print(result.python_code)
        """
        # Plan
        planner_output = self.plan(intent, context)

        # Select candidate
        if candidate_idx >= len(planner_output.candidates):
            candidate_idx = 0
        draft = planner_output.candidates[candidate_idx]

        # Create confirmed workflow
        confirmed = ConfirmedWorkflow(
            draft=draft,
            filled_parameters=parameters
        )

        # Compile
        return self.compile(confirmed)

    def get_missing_parameters(
        self,
        intent: str,
        context: Dict[str, Any] = None,
        candidate_idx: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get list of parameters needed for an intent.

        Useful for building UI forms.

        Args:
            intent: Natural language experiment description
            context: Optional planning context
            candidate_idx: Which candidate to check

        Returns:
            List of parameter info dicts with:
            - parameter: Parameter name
            - question: Question (English)
            - question_zh: Question (Chinese)
            - required: Whether required
            - default: Default value
            - options: Valid options (if any)
            - unit: Unit string
        """
        planner_output = self.plan(intent, context)

        if candidate_idx >= len(planner_output.candidates):
            candidate_idx = 0
        draft = planner_output.candidates[candidate_idx]

        return [mi.to_dict() for mi in draft.missing_info]

    def create_protocol(
        self,
        name: str,
        description: str = "",
        author: str = "OT-2 NLP Agent"
    ) -> Protocol:
        """
        Create a new protocol.

        Args:
            name: Protocol name
            description: Protocol description
            author: Protocol author

        Returns:
            New Protocol object
        """
        return Protocol(
            name=name,
            description=description,
            author=author,
            api_level=self.api_level
        )

    def add_labware(
        self,
        protocol: Protocol,
        labware_type: str,
        slot: int,
        name: str = None
    ):
        """
        Add labware to the protocol.

        Supports aliases like "96孔板", "tip rack", etc.

        Args:
            protocol: Protocol to add labware to
            labware_type: Labware type or alias
            slot: Deck slot (1-11)
            name: Variable name for the labware
        """
        # Resolve alias to actual labware name
        resolved_type = self.mapper.resolve_labware(labware_type)

        # Generate name if not provided
        if not name:
            name = f"labware_{slot}"

        protocol.add_labware(name, resolved_type, slot)

    def add_custom_labware(
        self,
        protocol: Protocol,
        labware: Union[str, CustomLabwareDefinition],
        slot: int,
        name: str = None,
        json_path: str = None
    ):
        """
        Add custom/3D-printed labware to the protocol.

        Args:
            protocol: Protocol to add labware to
            labware: Custom labware name (template) or CustomLabwareDefinition object
            slot: Deck slot (1-11)
            name: Variable name for the labware
            json_path: Path to save the labware JSON (optional)

        Example:
            # Use built-in template
            agent.add_custom_labware(protocol, "battery_holder_4x6", slot=1)

            # Create custom labware
            manager = get_labware_manager()
            my_labware = manager.create_grid_labware(
                name="my_holder",
                rows=3, columns=4,
                well_depth=20, well_diameter=15,
                well_volume=3000
            )
            agent.add_custom_labware(protocol, my_labware, slot=1)
        """
        manager = get_labware_manager()

        if isinstance(labware, str):
            # Look up template or loaded labware
            labware_def = manager.get_labware(labware)
            if not labware_def:
                raise ValueError(
                    f"Unknown custom labware: {labware}. "
                    f"Available templates: {manager.list_templates()}"
                )
        else:
            labware_def = labware

        # Generate name if not provided
        if not name:
            name = labware_def.name.replace("-", "_").replace(" ", "_")

        # Store custom labware definition in protocol metadata
        if 'custom_labware' not in protocol.metadata:
            protocol.metadata['custom_labware'] = {}
        protocol.metadata['custom_labware'][name] = labware_def.to_opentrons_json()

        # Add to protocol
        protocol.add_labware(name, labware_def.name, slot)

        # Optionally save JSON
        if json_path:
            labware_def.save_json(json_path)

    def list_custom_labware_templates(self) -> List[str]:
        """List available custom labware templates."""
        manager = get_labware_manager()
        return manager.list_templates()

    def create_custom_labware(
        self,
        name: str,
        rows: int,
        columns: int,
        well_depth: float,
        well_diameter: float = None,
        well_volume: float = 1000,
        row_spacing: float = 9.0,
        column_spacing: float = 9.0,
        description: str = ""
    ) -> CustomLabwareDefinition:
        """
        Create a custom labware definition.

        Args:
            name: Unique labware name
            rows: Number of rows
            columns: Number of columns
            well_depth: Well depth in mm
            well_diameter: Well diameter in mm (for circular wells)
            well_volume: Well volume in µL
            row_spacing: Spacing between rows in mm
            column_spacing: Spacing between columns in mm
            description: Description

        Returns:
            CustomLabwareDefinition object
        """
        manager = get_labware_manager()
        return manager.create_grid_labware(
            name=name,
            rows=rows,
            columns=columns,
            well_depth=well_depth,
            well_diameter=well_diameter,
            well_volume=well_volume,
            row_spacing=row_spacing,
            column_spacing=column_spacing,
            description=description
        )

    def add_pipette(
        self,
        protocol: Protocol,
        pipette_type: str,
        mount: str,
        tip_rack: str = None,
        name: str = "pipette"
    ):
        """
        Add a pipette to the protocol.

        Supports aliases like "p300", "单道移液器", etc.

        Args:
            protocol: Protocol to add pipette to
            pipette_type: Pipette type or alias
            mount: Mount position ('left' or 'right')
            tip_rack: Name of tip rack labware
            name: Variable name for the pipette
        """
        # Resolve alias
        resolved_type = self.mapper.resolve_pipette(pipette_type)
        protocol.add_pipette(name, resolved_type, mount, tip_rack)

    def parse(self, instruction: str) -> ParsedIntent:
        """
        Parse a single natural language instruction.

        Args:
            instruction: Natural language instruction

        Returns:
            ParsedIntent with operation type and parameters
        """
        return self.parser.parse(instruction)

    def parse_multi(self, instruction: str) -> List[ParsedIntent]:
        """
        Parse a multi-step instruction.

        Handles instructions with step markers like "第一步...第二步..."

        Args:
            instruction: Natural language instruction with multiple steps

        Returns:
            List of ParsedIntent objects
        """
        return self.parser.parse_multi_step(instruction)

    def add_instruction(
        self,
        protocol: Protocol,
        instruction: str,
        auto_tips: bool = False
    ) -> List[Operation]:
        """
        Add an instruction to the protocol.

        Parses the instruction and adds resulting operations.

        Args:
            protocol: Protocol to add to
            instruction: Natural language instruction
            auto_tips: Automatically add tip operations if needed

        Returns:
            List of operations added
        """
        # Store original instruction
        protocol.original_instructions.append(instruction)

        # Parse instruction(s)
        intents = self.parser.parse_multi_step(instruction)
        operations = []

        for intent in intents:
            if intent.operation_type:
                op = Operation(
                    type=intent.operation_type,
                    params=intent.params,
                    description=intent.original_text
                )
                protocol.add_operation(op)
                operations.append(op)

        return operations

    def add_instructions(
        self,
        protocol: Protocol,
        instructions: List[str],
        auto_tips: bool = False
    ) -> List[Operation]:
        """
        Add multiple instructions to the protocol.

        Args:
            protocol: Protocol to add to
            instructions: List of natural language instructions
            auto_tips: Automatically add tip operations if needed

        Returns:
            List of all operations added
        """
        all_ops = []
        for instruction in instructions:
            ops = self.add_instruction(protocol, instruction, auto_tips)
            all_ops.extend(ops)
        return all_ops

    def validate(self, protocol: Protocol) -> ValidationResult:
        """
        Validate the protocol.

        Checks for errors and warnings before execution.

        Args:
            protocol: Protocol to validate

        Returns:
            ValidationResult with issues found
        """
        return self.validator.validate(protocol)

    def preview(self, protocol: Protocol) -> str:
        """
        Generate a human-readable preview.

        Useful for user confirmation before generating code.

        Args:
            protocol: Protocol to preview

        Returns:
            Formatted preview string
        """
        return self.generator.generate_preview(protocol)

    def generate(self, protocol: Protocol) -> str:
        """
        Generate Python protocol code.

        Args:
            protocol: Protocol to generate code for

        Returns:
            Python code string
        """
        return self.generator.generate(protocol)

    def save(self, protocol: Protocol, filepath: str):
        """
        Save protocol to a Python file.

        Args:
            protocol: Protocol to save
            filepath: Path to save to
        """
        self.generator.save(protocol, filepath)

    def to_json(self, protocol: Protocol) -> str:
        """
        Generate workflow JSON string.

        Args:
            protocol: Protocol to convert

        Returns:
            JSON string in workflow format
        """
        return self.generator.to_json(protocol)

    def save_json(self, protocol: Protocol, filepath: str, version: str = "1.0"):
        """
        Save protocol as workflow JSON file.

        Args:
            protocol: Protocol to save
            filepath: Path to save to
            version: Workflow version string
        """
        self.generator.save_workflow_json(protocol, filepath, version)

    # =========================================================================
    # Convenience Methods for Common Operations
    # =========================================================================

    def quick_transfer(
        self,
        protocol: Protocol,
        source: str,
        destination: Union[str, List[str]],
        volume: float
    ):
        """
        Add a quick transfer operation.

        Args:
            protocol: Protocol to add to
            source: Source well (e.g., "A1")
            destination: Destination well(s)
            volume: Volume in µL
        """
        op = Operation(
            type=OperationType.TRANSFER,
            params={
                'volume': volume,
                'source': source,
                'destination': destination,
            },
            description=f"Transfer {volume}µL from {source} to {destination}"
        )
        protocol.add_operation(op)

    def quick_distribute(
        self,
        protocol: Protocol,
        source: str,
        destinations: List[str],
        volume_per_well: float
    ):
        """
        Distribute from one source to multiple destinations.

        Args:
            protocol: Protocol to add to
            source: Source well
            destinations: List of destination wells
            volume_per_well: Volume per destination in µL
        """
        for dest in destinations:
            op = Operation(
                type=OperationType.TRANSFER,
                params={
                    'volume': volume_per_well,
                    'source': source,
                    'destination': dest,
                },
                description=f"Distribute {volume_per_well}µL from {source} to {dest}"
            )
            protocol.add_operation(op)

    def quick_serial_dilution(
        self,
        protocol: Protocol,
        wells: List[str],
        initial_volume: float,
        dilution_factor: float = 2.0,
        mix_reps: int = 3
    ):
        """
        Perform serial dilution across wells.

        Args:
            protocol: Protocol to add to
            wells: List of wells in dilution order
            initial_volume: Starting volume in first well
            dilution_factor: Dilution factor (default 2x)
            mix_reps: Number of mixing repetitions
        """
        transfer_volume = initial_volume / dilution_factor

        for i in range(len(wells) - 1):
            src = wells[i]
            dst = wells[i + 1]

            # Mix source
            protocol.add_operation(Operation(
                type=OperationType.MIX,
                params={
                    'repetitions': mix_reps,
                    'volume': transfer_volume,
                    'location': src,
                },
                description=f"Mix {src}"
            ))

            # Transfer to next well
            protocol.add_operation(Operation(
                type=OperationType.TRANSFER,
                params={
                    'volume': transfer_volume,
                    'source': src,
                    'destination': dst,
                },
                description=f"Serial dilution: {src} → {dst}"
            ))

        # Final mix
        protocol.add_operation(Operation(
            type=OperationType.MIX,
            params={
                'repetitions': mix_reps,
                'volume': transfer_volume,
                'location': wells[-1],
            },
            description=f"Final mix {wells[-1]}"
        ))
