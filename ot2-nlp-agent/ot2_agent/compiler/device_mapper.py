"""
Device Mapper - Map Primitives to Device Actions.

This module maps device-agnostic primitives to specific
device commands.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from ..ir import Primitive, ActionType, DeviceAction
from ..operations import Operation, OperationType


@dataclass
class DeviceCapability:
    """Describes what a device can do."""
    action_types: List[ActionType]
    constraints: Dict[str, Any] = field(default_factory=dict)


class DeviceRegistry:
    """
    Registry of available devices and their capabilities.
    """

    def __init__(self):
        """Initialize with default devices."""
        self.devices: Dict[str, Dict[str, Any]] = {}
        self._register_default_devices()

    def _register_default_devices(self):
        """Register default devices."""
        # OT-2 liquid handler
        self.register_device(
            device_id="ot2",
            device_type="liquid_handler",
            capabilities=DeviceCapability(
                action_types=[
                    ActionType.LIQUID_TRANSFER,
                    ActionType.ASPIRATE,
                    ActionType.DISPENSE,
                    ActionType.MIXING,
                ],
                constraints={
                    "volume_range_ul": (1, 1000),
                    "tip_required": True,
                },
            ),
        )

        # Potentiostat
        self.register_device(
            device_id="potentiostat",
            device_type="potentiostat",
            capabilities=DeviceCapability(
                action_types=[
                    ActionType.POTENTIOSTAT_METHOD,
                    ActionType.IMPEDANCE_SCAN,
                    ActionType.OPEN_CIRCUIT,
                ],
                constraints={
                    "voltage_range_V": (-10, 10),
                    "current_range_A": (1e-9, 1),
                },
            ),
        )

        # Temperature module
        self.register_device(
            device_id="temperature_module",
            device_type="temperature_module",
            capabilities=DeviceCapability(
                action_types=[
                    ActionType.HEAT,
                    ActionType.COOL,
                    ActionType.INCUBATE,
                ],
                constraints={
                    "temperature_range_C": (4, 95),
                },
            ),
        )

        # Data system (always available)
        self.register_device(
            device_id="data_system",
            device_type="data_system",
            capabilities=DeviceCapability(
                action_types=[
                    ActionType.DATA_LOGGING,
                ],
            ),
        )

        # User (for checkpoints)
        self.register_device(
            device_id="user",
            device_type="user",
            capabilities=DeviceCapability(
                action_types=[
                    ActionType.USER_CHECKPOINT,
                ],
            ),
        )

    def register_device(
        self,
        device_id: str,
        device_type: str,
        capabilities: DeviceCapability
    ):
        """Register a device."""
        self.devices[device_id] = {
            "device_type": device_type,
            "capabilities": capabilities,
        }

    def find_device(self, action_type: ActionType, device_type: str = None) -> Optional[str]:
        """Find a device that can perform the given action."""
        for device_id, device_info in self.devices.items():
            caps = device_info["capabilities"]
            if action_type in caps.action_types:
                if device_type is None or device_info["device_type"] == device_type:
                    return device_id
        return None

    def get_device(self, device_id: str) -> Optional[Dict]:
        """Get device info by ID."""
        return self.devices.get(device_id)


class DeviceMapper:
    """
    Maps Primitives to DeviceActions.

    Takes device-agnostic primitives and converts them to
    specific device commands.
    """

    def __init__(self, registry: DeviceRegistry = None):
        """
        Initialize the device mapper.

        Args:
            registry: Device registry to use. Creates default if None.
        """
        self.registry = registry or DeviceRegistry()

    def map(self, primitives: List[Primitive]) -> List[DeviceAction]:
        """
        Map primitives to device actions.

        Args:
            primitives: List of primitives to map

        Returns:
            List of DeviceAction objects
        """
        actions = []

        for primitive in primitives:
            device_action = self._map_primitive(primitive)
            if device_action:
                actions.append(device_action)

        return actions

    def _map_primitive(self, primitive: Primitive) -> Optional[DeviceAction]:
        """Map a single primitive to a device action."""
        # Find appropriate device
        device_id = self.registry.find_device(
            primitive.action_type,
            primitive.device_type if primitive.device_type != "any" else None
        )

        if not device_id:
            # No device found, create a placeholder
            return self._create_placeholder_action(primitive)

        # Get device info
        device_info = self.registry.get_device(device_id)
        device_type = device_info["device_type"]

        # Map based on action type and device type
        mapper_method = self._get_mapper_method(primitive.action_type, device_type)
        return mapper_method(primitive, device_id, device_type)

    def _get_mapper_method(self, action_type: ActionType, device_type: str):
        """Get the appropriate mapper method."""
        # Device-specific mappers
        mappers = {
            ("liquid_handler", ActionType.LIQUID_TRANSFER): self._map_ot2_transfer,
            ("liquid_handler", ActionType.ASPIRATE): self._map_ot2_aspirate,
            ("liquid_handler", ActionType.DISPENSE): self._map_ot2_dispense,
            ("liquid_handler", ActionType.MIXING): self._map_ot2_mix,
            ("potentiostat", ActionType.POTENTIOSTAT_METHOD): self._map_potentiostat_method,
            ("potentiostat", ActionType.IMPEDANCE_SCAN): self._map_potentiostat_eis,
            ("temperature_module", ActionType.HEAT): self._map_temperature,
            ("data_system", ActionType.DATA_LOGGING): self._map_data_logging,
            ("user", ActionType.USER_CHECKPOINT): self._map_user_checkpoint,
        }

        key = (device_type, action_type)
        return mappers.get(key, self._map_generic)

    def _map_ot2_transfer(self, primitive: Primitive, device_id: str, device_type: str) -> DeviceAction:
        """Map liquid transfer to OT-2 command."""
        params = primitive.params
        return DeviceAction(
            name=primitive.name,
            device_id=device_id,
            device_type=device_type,
            command="transfer",
            params={
                "volume": params.get("volume_ul"),
                "source": params.get("source"),
                "dest": params.get("destination"),
                "new_tip": "always",
            },
            description=primitive.description,
            source_primitive=primitive.name,
        )

    def _map_ot2_aspirate(self, primitive: Primitive, device_id: str, device_type: str) -> DeviceAction:
        """Map aspirate to OT-2 command."""
        params = primitive.params
        return DeviceAction(
            name=primitive.name,
            device_id=device_id,
            device_type=device_type,
            command="aspirate",
            params={
                "volume": params.get("volume_ul"),
                "location": params.get("source"),
            },
            description=primitive.description,
            source_primitive=primitive.name,
        )

    def _map_ot2_dispense(self, primitive: Primitive, device_id: str, device_type: str) -> DeviceAction:
        """Map dispense to OT-2 command."""
        params = primitive.params
        return DeviceAction(
            name=primitive.name,
            device_id=device_id,
            device_type=device_type,
            command="dispense",
            params={
                "volume": params.get("volume_ul"),
                "location": params.get("destination"),
            },
            description=primitive.description,
            source_primitive=primitive.name,
        )

    def _map_ot2_mix(self, primitive: Primitive, device_id: str, device_type: str) -> DeviceAction:
        """Map mixing to OT-2 command."""
        params = primitive.params
        return DeviceAction(
            name=primitive.name,
            device_id=device_id,
            device_type=device_type,
            command="mix",
            params={
                "repetitions": params.get("repetitions", 3),
                "volume": params.get("volume_ul"),
                "location": params.get("location"),
            },
            description=primitive.description,
            source_primitive=primitive.name,
        )

    def _map_potentiostat_method(self, primitive: Primitive, device_id: str, device_type: str) -> DeviceAction:
        """Map potentiostat method to device action."""
        params = primitive.params
        method = params.get("method", "LSV")

        return DeviceAction(
            name=primitive.name,
            device_id=device_id,
            device_type=device_type,
            command=f"run_{method.lower()}",
            params=params,
            description=primitive.description,
            source_primitive=primitive.name,
            estimated_duration_s=primitive.estimated_duration_s,
        )

    def _map_potentiostat_eis(self, primitive: Primitive, device_id: str, device_type: str) -> DeviceAction:
        """Map EIS to potentiostat command."""
        return DeviceAction(
            name=primitive.name,
            device_id=device_id,
            device_type=device_type,
            command="run_eis",
            params=primitive.params,
            description=primitive.description,
            source_primitive=primitive.name,
            estimated_duration_s=primitive.estimated_duration_s,
        )

    def _map_temperature(self, primitive: Primitive, device_id: str, device_type: str) -> DeviceAction:
        """Map temperature control to device action."""
        params = primitive.params
        return DeviceAction(
            name=primitive.name,
            device_id=device_id,
            device_type=device_type,
            command="set_temperature",
            params={
                "temperature": params.get("temperature_C"),
                "hold": params.get("hold", True),
            },
            description=primitive.description,
            source_primitive=primitive.name,
        )

    def _map_data_logging(self, primitive: Primitive, device_id: str, device_type: str) -> DeviceAction:
        """Map data logging to device action."""
        return DeviceAction(
            name=primitive.name,
            device_id=device_id,
            device_type=device_type,
            command="log_data",
            params=primitive.params,
            description=primitive.description,
            source_primitive=primitive.name,
        )

    def _map_user_checkpoint(self, primitive: Primitive, device_id: str, device_type: str) -> DeviceAction:
        """Map user checkpoint to action."""
        return DeviceAction(
            name=primitive.name,
            device_id=device_id,
            device_type=device_type,
            command="user_checkpoint",
            params=primitive.params,
            description=primitive.description,
            source_primitive=primitive.name,
            requires_confirmation=True,
        )

    def _map_generic(self, primitive: Primitive, device_id: str, device_type: str) -> DeviceAction:
        """Generic mapping for unhandled combinations."""
        return DeviceAction(
            name=primitive.name,
            device_id=device_id,
            device_type=device_type,
            command=primitive.action_type.value,
            params=primitive.params,
            description=primitive.description,
            source_primitive=primitive.name,
        )

    def _create_placeholder_action(self, primitive: Primitive) -> DeviceAction:
        """Create a placeholder action when no device is found."""
        return DeviceAction(
            name=primitive.name,
            device_id="unassigned",
            device_type=primitive.device_type,
            command="placeholder",
            params=primitive.params,
            description=f"[UNASSIGNED] {primitive.description}",
            source_primitive=primitive.name,
            requires_confirmation=True,
        )

    def to_operations(self, device_actions: List[DeviceAction]) -> List[Operation]:
        """
        Convert DeviceActions to existing Operation format for compatibility.

        This bridges the new system with the existing Protocol/Generator.
        """
        operations = []

        for action in device_actions:
            # Map device action to OperationType
            op_type = self._action_to_operation_type(action)

            if op_type:
                op = Operation(
                    type=op_type,
                    params=action.params,
                    description=action.description,
                )
                operations.append(op)

        return operations

    def _action_to_operation_type(self, action: DeviceAction) -> Optional[OperationType]:
        """Map DeviceAction to OperationType."""
        command_map = {
            "transfer": OperationType.TRANSFER,
            "aspirate": OperationType.ASPIRATE,
            "dispense": OperationType.DISPENSE,
            "mix": OperationType.MIX,
            "user_checkpoint": OperationType.PAUSE,
            "log_data": OperationType.COMMENT,
        }
        return command_map.get(action.command)
