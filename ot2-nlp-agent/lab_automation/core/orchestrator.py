"""
Lab Automation Orchestrator

The main agent that coordinates multiple instrument plugins to
parse natural language instructions and generate unified workflows.
"""

from typing import Any, Dict, List, Optional, Tuple
from .plugin_base import PluginBase
from .workflow import Workflow, Phase, Step, DeviceConfig


class LabAutomationAgent:
    """
    Main orchestrator for lab automation workflows.

    Manages multiple instrument plugins and routes natural language
    instructions to the appropriate plugin for parsing.

    Example:
        agent = LabAutomationAgent()

        # Register plugins
        agent.register_plugin(LiquidHandlerPlugin())
        agent.register_plugin(PotentiostatPlugin())
        agent.register_plugin(PumpControllerPlugin())

        # Create workflow from instructions
        workflow = agent.create_workflow(
            name="My Experiment",
            instructions=[
                "transfer 100ul from A1 to B1",
                "run EIS from 10kHz to 0.1Hz",
                "pump 5ml of water",
            ]
        )

        # Export
        workflow.save_json("my_experiment.json")
    """

    def __init__(self):
        self._plugins: Dict[str, PluginBase] = {}
        self._device_type_to_plugin: Dict[str, str] = {}

    def register_plugin(self, plugin: PluginBase):
        """
        Register an instrument plugin.

        Args:
            plugin: Plugin instance to register
        """
        self._plugins[plugin.name] = plugin
        self._device_type_to_plugin[plugin.device_type] = plugin.name

    def get_plugin(self, name: str) -> Optional[PluginBase]:
        """Get a plugin by name."""
        return self._plugins.get(name)

    def get_plugin_by_device_type(self, device_type: str) -> Optional[PluginBase]:
        """Get a plugin by device type."""
        plugin_name = self._device_type_to_plugin.get(device_type)
        if plugin_name:
            return self._plugins.get(plugin_name)
        return None

    def list_plugins(self) -> List[str]:
        """List all registered plugin names."""
        return list(self._plugins.keys())

    def list_device_types(self) -> List[str]:
        """List all supported device types."""
        return list(self._device_type_to_plugin.keys())

    def parse_instruction(self, instruction: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Parse an instruction and route to the appropriate plugin.

        Args:
            instruction: Natural language instruction

        Returns:
            Tuple of (plugin_name, parsed_result)
            Returns (None, {}) if no plugin can handle the instruction
        """
        best_plugin = None
        best_confidence = 0.0
        best_result = {}

        for name, plugin in self._plugins.items():
            can_handle, confidence = plugin.can_handle(instruction)
            if can_handle and confidence > best_confidence:
                best_plugin = name
                best_confidence = confidence
                best_result = plugin.parse(instruction)

        return best_plugin, best_result

    def create_workflow(
        self,
        name: str,
        instructions: List[str],
        description: str = "",
        version: str = "1.0",
        phase_name: str = "execution",
    ) -> Workflow:
        """
        Create a workflow from a list of natural language instructions.

        Args:
            name: Workflow name
            instructions: List of natural language instructions
            description: Workflow description
            version: Workflow version
            phase_name: Name for the main execution phase

        Returns:
            Workflow object
        """
        workflow = Workflow(
            workflow_name=name,
            version=version,
            description=description or f"Workflow created from {len(instructions)} instructions",
        )
        workflow.original_instructions = instructions.copy()

        # Create execution phase
        phase = Phase(phase_name=phase_name, description="Main execution steps")
        workflow.add_phase(phase)

        # Track which device types are used
        used_devices = set()

        # Parse each instruction
        step_counter = 1
        for instruction in instructions:
            instruction = instruction.strip()
            if not instruction:
                continue

            plugin_name, parsed = self.parse_instruction(instruction)

            if plugin_name and parsed.get('action'):
                plugin = self._plugins[plugin_name]
                step = Step(
                    step_id=f"{phase_name}_{step_counter:03d}",
                    device_type=plugin.device_type,
                    action=parsed['action'],
                    params=parsed.get('params', {}),
                    description=parsed.get('description', instruction),
                )
                phase.add_step(step)
                used_devices.add(plugin.device_type)
                step_counter += 1
            else:
                # Handle unknown instruction - add as comment/note
                step = Step(
                    step_id=f"{phase_name}_{step_counter:03d}",
                    device_type="unknown",
                    action="unknown",
                    params={"original_text": instruction},
                    description=f"[UNRECOGNIZED] {instruction}",
                )
                phase.add_step(step)
                step_counter += 1

        # Add device configurations for used devices
        for device_type in used_devices:
            plugin = self.get_plugin_by_device_type(device_type)
            if plugin:
                adapters = plugin.list_adapters()
                default_adapter = adapters[0] if adapters else "generic"
                workflow.add_device(DeviceConfig(
                    device_type=device_type,
                    adapter=default_adapter,
                    name=f"{device_type}_1",
                ))

        return workflow

    def create_multi_phase_workflow(
        self,
        name: str,
        phases_config: List[Dict[str, Any]],
        description: str = "",
        version: str = "1.0",
    ) -> Workflow:
        """
        Create a workflow with multiple phases.

        Args:
            name: Workflow name
            phases_config: List of phase configurations, each with:
                - phase_name: str
                - description: str (optional)
                - instructions: List[str]
                - parallel: bool (optional, default False)
            description: Workflow description
            version: Workflow version

        Returns:
            Workflow object
        """
        workflow = Workflow(
            workflow_name=name,
            version=version,
            description=description,
        )

        used_devices = set()
        all_instructions = []

        for phase_config in phases_config:
            phase_name = phase_config['phase_name']
            phase_desc = phase_config.get('description', '')
            instructions = phase_config.get('instructions', [])
            is_parallel = phase_config.get('parallel', False)

            all_instructions.extend(instructions)

            phase = Phase(phase_name=phase_name, description=phase_desc)
            workflow.add_phase(phase)

            step_counter = 1
            for instruction in instructions:
                instruction = instruction.strip()
                if not instruction:
                    continue

                plugin_name, parsed = self.parse_instruction(instruction)

                if plugin_name and parsed.get('action'):
                    plugin = self._plugins[plugin_name]
                    step = Step(
                        step_id=f"{phase_name}_{step_counter:03d}",
                        device_type=plugin.device_type,
                        action=parsed['action'],
                        params=parsed.get('params', {}),
                        description=parsed.get('description', instruction),
                    )
                    phase.add_step(step)
                    used_devices.add(plugin.device_type)
                    step_counter += 1

        workflow.original_instructions = all_instructions

        # Add device configurations
        for device_type in used_devices:
            plugin = self.get_plugin_by_device_type(device_type)
            if plugin:
                adapters = plugin.list_adapters()
                default_adapter = adapters[0] if adapters else "generic"
                workflow.add_device(DeviceConfig(
                    device_type=device_type,
                    adapter=default_adapter,
                    name=f"{device_type}_1",
                ))

        return workflow

    def get_all_operations(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all operations from all registered plugins.

        Returns:
            Dict mapping device_type to list of operation definitions
        """
        result = {}
        for name, plugin in self._plugins.items():
            ops = []
            for op_name, op_def in plugin.get_operations().items():
                ops.append({
                    "name": op_name,
                    "action": op_def.action,
                    "keywords": op_def.keywords,
                    "description": op_def.description,
                })
            result[plugin.device_type] = ops
        return result

    def get_help_text(self) -> str:
        """Generate help text describing available operations."""
        lines = ["# Lab Automation Agent - Available Operations\n"]

        for name, plugin in self._plugins.items():
            lines.append(f"## {plugin.device_type.replace('_', ' ').title()}")
            lines.append(f"*{plugin.description}*\n")

            for op_name, op_def in plugin.get_operations().items():
                keywords_en = ", ".join(op_def.keywords.get("en", [])[:3])
                lines.append(f"- **{op_name}**: {op_def.description}")
                lines.append(f"  Keywords: {keywords_en}")

            lines.append("")

        return "\n".join(lines)
