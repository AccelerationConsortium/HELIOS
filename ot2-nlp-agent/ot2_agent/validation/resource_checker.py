"""
Resource Checker - Detect resource conflicts in workflows.

Checks for issues like:
- Device concurrency conflicts
- Consumable availability
- Volume/capacity limits
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional

from ..ir import DeviceAction, Primitive


@dataclass
class ResourceConflict:
    """Represents a resource conflict."""
    severity: str  # "error", "warning"
    resource_type: str  # "device", "consumable", "capacity"
    resource_id: str
    message: str
    affected_actions: List[str] = field(default_factory=list)
    suggestion: str = ""


class ResourceChecker:
    """
    Checks for resource conflicts in workflows.
    """

    def __init__(self):
        """Initialize resource checker."""
        # Track resource usage
        self.device_usage: Dict[str, List[str]] = {}  # device_id -> [action_names]
        self.consumables: Dict[str, float] = {}  # consumable -> remaining

    def check(
        self,
        device_actions: List[DeviceAction],
        available_resources: Dict[str, Any] = None
    ) -> List[ResourceConflict]:
        """
        Check for resource conflicts.

        Args:
            device_actions: List of device actions to check
            available_resources: Optional dict of available resources

        Returns:
            List of ResourceConflict objects
        """
        conflicts = []
        available = available_resources or {}

        # Reset tracking
        self.device_usage = {}
        self.consumables = {}

        # Check each action
        for action in device_actions:
            # Check device availability
            device_conflicts = self._check_device_availability(action, available)
            conflicts.extend(device_conflicts)

            # Check consumable usage
            consumable_conflicts = self._check_consumables(action, available)
            conflicts.extend(consumable_conflicts)

            # Check capacity limits
            capacity_conflicts = self._check_capacity(action, available)
            conflicts.extend(capacity_conflicts)

        return conflicts

    def _check_device_availability(
        self,
        action: DeviceAction,
        available: Dict
    ) -> List[ResourceConflict]:
        """Check if required device is available."""
        conflicts = []
        device_id = action.device_id

        # Skip unassigned and user actions
        if device_id in ["unassigned", "user", "data_system"]:
            return conflicts

        # Track device usage
        if device_id not in self.device_usage:
            self.device_usage[device_id] = []
        self.device_usage[device_id].append(action.name)

        # Check if device exists in available resources
        available_devices = available.get("devices", [])
        if available_devices and device_id not in available_devices:
            conflicts.append(ResourceConflict(
                severity="warning",
                resource_type="device",
                resource_id=device_id,
                message=f"Device '{device_id}' may not be available",
                affected_actions=[action.name],
                suggestion=f"Ensure {device_id} is connected and configured",
            ))

        return conflicts

    def _check_consumables(
        self,
        action: DeviceAction,
        available: Dict
    ) -> List[ResourceConflict]:
        """Check consumable availability."""
        conflicts = []
        params = action.params

        # Check tips for liquid handling
        if action.device_type == "liquid_handler":
            if params.get("new_tip") == "always":
                tips_key = "tips"
                if tips_key not in self.consumables:
                    self.consumables[tips_key] = available.get("tips", 96)

                self.consumables[tips_key] -= 1

                if self.consumables[tips_key] < 0:
                    conflicts.append(ResourceConflict(
                        severity="error",
                        resource_type="consumable",
                        resource_id="tips",
                        message="Insufficient tips for workflow",
                        affected_actions=[action.name],
                        suggestion="Add more tip racks or reduce tip changes",
                    ))

        return conflicts

    def _check_capacity(
        self,
        action: DeviceAction,
        available: Dict
    ) -> List[ResourceConflict]:
        """Check capacity limits."""
        conflicts = []
        params = action.params

        # Check volume limits for liquid handling
        if action.device_type == "liquid_handler":
            volume = params.get("volume")
            if volume:
                # Default pipette range
                min_vol = available.get("min_volume_ul", 1)
                max_vol = available.get("max_volume_ul", 1000)

                if volume < min_vol:
                    conflicts.append(ResourceConflict(
                        severity="error",
                        resource_type="capacity",
                        resource_id="volume",
                        message=f"Volume {volume}µL below minimum {min_vol}µL",
                        affected_actions=[action.name],
                        suggestion="Use a smaller pipette or increase volume",
                    ))

                if volume > max_vol:
                    conflicts.append(ResourceConflict(
                        severity="error",
                        resource_type="capacity",
                        resource_id="volume",
                        message=f"Volume {volume}µL exceeds maximum {max_vol}µL",
                        affected_actions=[action.name],
                        suggestion="Split into multiple transfers or use larger pipette",
                    ))

        return conflicts

    def get_resource_summary(self) -> Dict[str, Any]:
        """Get summary of resource usage."""
        return {
            "devices_used": list(self.device_usage.keys()),
            "actions_per_device": {k: len(v) for k, v in self.device_usage.items()},
            "consumables_remaining": self.consumables.copy(),
        }
