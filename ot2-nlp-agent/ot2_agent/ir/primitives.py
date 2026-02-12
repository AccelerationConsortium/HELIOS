"""
Primitive layer - Atomic device-agnostic actions.

This is the lowest layer of the IR before device mapping.
Primitives are parameterized atomic actions that can be
mapped to specific device commands.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ActionType(Enum):
    """Types of primitive actions."""
    # Liquid handling
    LIQUID_TRANSFER = "liquid_transfer"     # Move liquid from A to B
    ASPIRATE = "aspirate"                   # Draw liquid
    DISPENSE = "dispense"                   # Release liquid
    MIXING = "mixing"                       # Mix liquid
    DILUTION = "dilution"                   # Dilute sample

    # Temperature control
    HEAT = "heat"                           # Heat to temperature
    COOL = "cool"                           # Cool to temperature
    INCUBATE = "incubate"                   # Maintain temperature for duration

    # Electrochemistry
    POTENTIOSTAT_METHOD = "potentiostat_method"  # Run electrochemical method
    IMPEDANCE_SCAN = "impedance_scan"            # EIS measurement
    OPEN_CIRCUIT = "open_circuit"                # OCP measurement

    # Mechanical
    STIR = "stir"                           # Stir solution
    SHAKE = "shake"                         # Shake plate
    CENTRIFUGE = "centrifuge"               # Centrifuge samples

    # Data
    DATA_LOGGING = "data_logging"           # Log data
    CAPTURE_IMAGE = "capture_image"         # Take photo
    READ_SENSOR = "read_sensor"             # Read sensor value

    # Control flow
    WAIT = "wait"                           # Wait for duration
    USER_CHECKPOINT = "user_checkpoint"     # Pause for user action
    CONDITIONAL = "conditional"             # Branch based on condition

    # Setup/cleanup
    SETUP_DEVICE = "setup_device"           # Initialize device
    CLEANUP = "cleanup"                     # Clean up resources

    # Movement
    MOVE = "move"                           # Move to position
    HOME = "home"                           # Return to home position


@dataclass
class Primitive:
    """
    A primitive action - the lowest level before device mapping.

    Primitives are atomic, parameterized actions that specify
    WHAT to do without specifying WHICH device does it.

    Example:
        Primitive(
            name="transfer_electrolyte",
            action_type=ActionType.LIQUID_TRANSFER,
            params={
                "volume_ul": 1000,
                "source": "reservoir_A1",
                "destination": "cell_electrolyte_chamber",
            },
            device_type="liquid_handler",
            description="Transfer 1mL electrolyte to cell"
        )
    """
    name: str                               # Primitive name/identifier
    action_type: ActionType                 # Action category
    params: Dict[str, Any] = field(default_factory=dict)  # Action parameters

    # Device requirements
    device_type: str = "any"                # Required device type
    device_constraints: Dict[str, Any] = field(default_factory=dict)  # Device constraints

    # Documentation
    description: str = ""                   # What this primitive does

    # Timing
    estimated_duration_s: Optional[float] = None

    # Dependencies
    depends_on: List[str] = field(default_factory=list)  # Primitive names this depends on

    # Source UO (for traceability)
    source_uo: Optional[str] = None         # Name of UO this came from

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "action_type": self.action_type.value,
            "params": self.params,
            "device_type": self.device_type,
            "device_constraints": self.device_constraints,
            "description": self.description,
            "estimated_duration_s": self.estimated_duration_s,
            "depends_on": self.depends_on,
            "source_uo": self.source_uo,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Primitive":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            action_type=ActionType(data["action_type"]),
            params=data.get("params", {}),
            device_type=data.get("device_type", "any"),
            device_constraints=data.get("device_constraints", {}),
            description=data.get("description", ""),
            estimated_duration_s=data.get("estimated_duration_s"),
            depends_on=data.get("depends_on", []),
            source_uo=data.get("source_uo"),
        )


@dataclass
class DeviceAction:
    """
    A device-specific action - the result of mapping a Primitive to a device.

    This is what gets translated to actual device commands/code.

    Example:
        DeviceAction(
            name="transfer_electrolyte",
            device_id="ot2_robot",
            device_type="ot2",
            command="transfer",
            params={
                "volume": 1000,
                "source": "reservoir['A1']",
                "dest": "cell['A1']",
                "new_tip": "always",
            },
            description="OT-2: Transfer 1mL electrolyte"
        )
    """
    name: str                               # Action name
    device_id: str                          # Specific device identifier
    device_type: str                        # Device type
    command: str                            # Device command/method
    params: Dict[str, Any] = field(default_factory=dict)  # Command parameters

    # Documentation
    description: str = ""

    # Timing
    estimated_duration_s: Optional[float] = None

    # Source
    source_primitive: Optional[str] = None  # Primitive name this came from

    # Execution
    requires_confirmation: bool = False     # Need user confirmation?
    is_reversible: bool = False             # Can this be undone?

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "device_id": self.device_id,
            "device_type": self.device_type,
            "command": self.command,
            "params": self.params,
            "description": self.description,
            "estimated_duration_s": self.estimated_duration_s,
            "source_primitive": self.source_primitive,
            "requires_confirmation": self.requires_confirmation,
            "is_reversible": self.is_reversible,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeviceAction":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            device_id=data["device_id"],
            device_type=data["device_type"],
            command=data["command"],
            params=data.get("params", {}),
            description=data.get("description", ""),
            estimated_duration_s=data.get("estimated_duration_s"),
            source_primitive=data.get("source_primitive"),
            requires_confirmation=data.get("requires_confirmation", False),
            is_reversible=data.get("is_reversible", False),
        )

    def to_python_code(self, indent: int = 0) -> str:
        """
        Generate Python code for this action.

        This is a simple default implementation. Device-specific
        code generators may override this.
        """
        prefix = "    " * indent
        lines = []

        if self.description:
            lines.append(f"{prefix}# {self.description}")

        # Format parameters
        param_strs = [f"{k}={repr(v)}" for k, v in self.params.items()]
        params_str = ", ".join(param_strs)

        lines.append(f"{prefix}{self.device_id}.{self.command}({params_str})")

        return "\n".join(lines)
