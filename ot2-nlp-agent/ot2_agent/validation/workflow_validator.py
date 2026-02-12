"""
Workflow Validator - Comprehensive workflow validation.

Combines schema, resource, and topology validation into
a single comprehensive validation result.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..ir import UnitOperation, DeviceAction
from ..protocol import Protocol
from ..validator import ProtocolValidator, ValidationResult, ValidationIssue

from .resource_checker import ResourceChecker, ResourceConflict
from .topology_checker import TopologyChecker, TopologyIssue


@dataclass
class Checkpoint:
    """A point where human confirmation is required."""
    step_index: int
    step_name: str
    message: str
    message_zh: str = ""
    checklist: List[str] = field(default_factory=list)
    is_critical: bool = False


@dataclass
class EnhancedValidationResult:
    """
    Enhanced validation result with detailed information.

    Extends the basic ValidationResult with resource and topology info.
    """
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    resource_conflicts: List[ResourceConflict] = field(default_factory=list)
    topology_issues: List[TopologyIssue] = field(default_factory=list)
    checkpoints: List[Checkpoint] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        """Count of error-level issues."""
        count = sum(1 for i in self.issues if i.severity == "ERROR")
        count += sum(1 for c in self.resource_conflicts if c.severity == "error")
        count += sum(1 for t in self.topology_issues if t.severity == "error")
        return count

    @property
    def warning_count(self) -> int:
        """Count of warning-level issues."""
        count = sum(1 for i in self.issues if i.severity == "WARNING")
        count += sum(1 for c in self.resource_conflicts if c.severity == "warning")
        count += sum(1 for t in self.topology_issues if t.severity == "warning")
        return count

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [
                {"severity": i.severity, "message": i.message, "step": i.step}
                for i in self.issues
            ],
            "resource_conflicts": [
                {
                    "severity": c.severity,
                    "type": c.resource_type,
                    "resource": c.resource_id,
                    "message": c.message,
                }
                for c in self.resource_conflicts
            ],
            "topology_issues": [
                {
                    "severity": t.severity,
                    "type": t.issue_type,
                    "message": t.message,
                    "step": t.affected_step,
                }
                for t in self.topology_issues
            ],
            "checkpoints": [
                {
                    "step": c.step_index,
                    "name": c.step_name,
                    "message": c.message,
                    "is_critical": c.is_critical,
                }
                for c in self.checkpoints
            ],
        }


class WorkflowValidator:
    """
    Comprehensive workflow validator.

    Combines multiple validation strategies:
    - Protocol validation (existing)
    - Resource checking
    - Topology verification
    - Checkpoint identification
    """

    def __init__(self):
        """Initialize the workflow validator."""
        self.protocol_validator = ProtocolValidator()
        self.resource_checker = ResourceChecker()
        self.topology_checker = TopologyChecker()

    def validate(
        self,
        protocol: Protocol,
        unit_operations: List[UnitOperation] = None,
        device_actions: List[DeviceAction] = None,
        available_resources: Dict[str, Any] = None
    ) -> EnhancedValidationResult:
        """
        Perform comprehensive validation.

        Args:
            protocol: Protocol to validate
            unit_operations: Optional list of UOs for topology check
            device_actions: Optional list of device actions for resource check
            available_resources: Optional dict of available resources

        Returns:
            EnhancedValidationResult with all validation info
        """
        # Run protocol validation
        protocol_result = self.protocol_validator.validate(protocol)

        # Initialize result
        result = EnhancedValidationResult(
            is_valid=protocol_result.is_valid,
            issues=protocol_result.issues.copy(),
        )

        # Run resource checking if device actions provided
        if device_actions:
            resource_conflicts = self.resource_checker.check(
                device_actions,
                available_resources or {}
            )
            result.resource_conflicts = resource_conflicts

            # Update validity
            if any(c.severity == "error" for c in resource_conflicts):
                result.is_valid = False

        # Run topology checking if UOs provided
        if unit_operations:
            topology_issues = self.topology_checker.check(unit_operations)
            result.topology_issues = topology_issues

            # Also check device actions if available
            if device_actions:
                action_issues = self.topology_checker.check_device_actions(device_actions)
                result.topology_issues.extend(action_issues)

            # Topology issues are warnings, don't affect validity

        # Identify checkpoints
        if device_actions:
            checkpoints = self._identify_checkpoints(device_actions)
            result.checkpoints = checkpoints

        return result

    def _identify_checkpoints(self, device_actions: List[DeviceAction]) -> List[Checkpoint]:
        """Identify required human checkpoints."""
        checkpoints = []

        for i, action in enumerate(device_actions):
            # Actions marked as requiring confirmation
            if action.requires_confirmation:
                params = action.params

                checkpoint = Checkpoint(
                    step_index=i,
                    step_name=action.name,
                    message=params.get("message", f"Confirm: {action.description}"),
                    message_zh=params.get("message_zh", ""),
                    checklist=params.get("checklist", []),
                    is_critical=action.device_type == "user",
                )
                checkpoints.append(checkpoint)

        return checkpoints

    def validate_simple(self, protocol: Protocol) -> ValidationResult:
        """
        Simple validation (backward compatible).

        Returns the same ValidationResult as the original validator.
        """
        return self.protocol_validator.validate(protocol)
