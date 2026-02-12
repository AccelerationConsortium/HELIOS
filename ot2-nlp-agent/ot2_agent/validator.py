"""
Protocol Validator for OT-2 robot.
Validates protocols before execution to catch errors early.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Set

from .operations import Operation, OperationType
from .protocol import Protocol


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"      # Must be fixed before execution
    WARNING = "warning"  # Should be reviewed
    INFO = "info"        # Informational only


@dataclass
class ValidationIssue:
    """A single validation issue."""
    severity: ValidationSeverity
    message: str
    step: Optional[int] = None
    suggestion: Optional[str] = None

    def __str__(self):
        step_str = f"Step {self.step}: " if self.step else ""
        severity_icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}[self.severity.value]
        msg = f"{severity_icon} {step_str}{self.message}"
        if self.suggestion:
            msg += f"\n   💡 Suggestion: {self.suggestion}"
        return msg


@dataclass
class ValidationResult:
    """Result of protocol validation."""
    is_valid: bool
    issues: List[ValidationIssue]
    warnings_count: int
    errors_count: int

    def __str__(self):
        if self.is_valid and not self.issues:
            return "✅ Protocol is valid!"

        lines = []
        if not self.is_valid:
            lines.append(f"❌ Protocol has {self.errors_count} error(s)")
        if self.warnings_count > 0:
            lines.append(f"⚠️ Protocol has {self.warnings_count} warning(s)")

        lines.append("")
        for issue in self.issues:
            lines.append(str(issue))

        return "\n".join(lines)


class ProtocolValidator:
    """
    Validates OT-2 protocols for common issues.

    Checks for:
    - Missing labware/pipette configuration
    - Invalid slot numbers
    - Volume limits
    - Tip management issues
    - Well coordinate validity
    - Safety constraints
    """

    # OT-2 constraints
    MAX_SLOTS = 11
    VALID_SLOTS = set(range(1, 12))  # 1-11

    # Volume limits by pipette type (in µL)
    VOLUME_LIMITS = {
        'p20': (1, 20),
        'p300': (20, 300),
        'p1000': (100, 1000),
    }

    # Valid well rows and columns
    VALID_ROWS = set('ABCDEFGH')
    VALID_COLS_96 = set(range(1, 13))  # 1-12
    VALID_COLS_24 = set(range(1, 7))   # 1-6

    def validate(self, protocol: Protocol) -> ValidationResult:
        """
        Validate a complete protocol.

        Args:
            protocol: Protocol to validate

        Returns:
            ValidationResult with all issues found
        """
        issues = []

        # Check configuration
        issues.extend(self._validate_labware(protocol))
        issues.extend(self._validate_pipettes(protocol))

        # Check operations
        issues.extend(self._validate_operations(protocol))

        # Check tip management
        issues.extend(self._validate_tip_management(protocol))

        # Count by severity
        errors = sum(1 for i in issues if i.severity == ValidationSeverity.ERROR)
        warnings = sum(1 for i in issues if i.severity == ValidationSeverity.WARNING)

        return ValidationResult(
            is_valid=(errors == 0),
            issues=issues,
            warnings_count=warnings,
            errors_count=errors,
        )

    def _validate_labware(self, protocol: Protocol) -> List[ValidationIssue]:
        """Validate labware configuration."""
        issues = []
        used_slots: Set[int] = set()

        if not protocol.labware:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="No labware configured",
                suggestion="Add labware using protocol.add_labware()"
            ))

        for lw in protocol.labware:
            # Check slot validity
            if lw.slot not in self.VALID_SLOTS:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid slot {lw.slot} for labware '{lw.name}'",
                    suggestion=f"Use slot numbers 1-11"
                ))

            # Check for slot conflicts
            if lw.slot in used_slots:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Slot {lw.slot} is already occupied",
                    suggestion="Each labware must be in a unique slot"
                ))
            used_slots.add(lw.slot)

        return issues

    def _validate_pipettes(self, protocol: Protocol) -> List[ValidationIssue]:
        """Validate pipette configuration."""
        issues = []
        used_mounts: Set[str] = set()

        if not protocol.pipettes:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="No pipettes configured",
                suggestion="Add a pipette using protocol.add_pipette()"
            ))

        for pip in protocol.pipettes:
            # Check mount validity
            if pip.mount not in ('left', 'right'):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid mount '{pip.mount}' for pipette '{pip.name}'",
                    suggestion="Use 'left' or 'right'"
                ))

            # Check for mount conflicts
            if pip.mount in used_mounts:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Mount '{pip.mount}' already has a pipette",
                    suggestion="Each mount can only have one pipette"
                ))
            used_mounts.add(pip.mount)

        return issues

    def _validate_operations(self, protocol: Protocol) -> List[ValidationIssue]:
        """Validate individual operations."""
        issues = []

        for i, op in enumerate(protocol.operations, 1):
            # Validate required parameters
            op_errors = op.validate()
            for error in op_errors:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=error,
                    step=i
                ))

            # Validate volume limits
            if 'volume' in op.params:
                volume_issues = self._validate_volume(op.params['volume'], i)
                issues.extend(volume_issues)

            # Validate well coordinates
            for key in ['location', 'source', 'destination']:
                if key in op.params:
                    well_val = op.params[key]
                    if isinstance(well_val, str):
                        well_issues = self._validate_well(well_val, i)
                        issues.extend(well_issues)
                    elif isinstance(well_val, list):
                        for well in well_val:
                            issues.extend(self._validate_well(well, i))

        return issues

    def _validate_volume(self, volume: float, step: int) -> List[ValidationIssue]:
        """Validate volume is within reasonable limits."""
        issues = []

        if volume <= 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Volume must be positive, got {volume}",
                step=step
            ))
        elif volume < 1:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"Very small volume ({volume} µL) may be inaccurate",
                step=step,
                suggestion="Consider using a smaller pipette for better accuracy"
            ))
        elif volume > 1000:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Volume {volume} µL exceeds maximum pipette capacity",
                step=step,
                suggestion="Split into multiple transfers or use a different method"
            ))

        return issues

    def _validate_well(self, well: str, step: int) -> List[ValidationIssue]:
        """Validate well coordinate format."""
        issues = []

        if not well or len(well) < 2:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Invalid well format: '{well}'",
                step=step,
                suggestion="Use format like 'A1', 'B12', etc."
            ))
            return issues

        row = well[0].upper()
        try:
            col = int(well[1:])
        except ValueError:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Invalid well column in '{well}'",
                step=step
            ))
            return issues

        if row not in self.VALID_ROWS:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Invalid well row '{row}' in '{well}'",
                step=step,
                suggestion="Use rows A-H"
            ))

        if col < 1 or col > 12:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Invalid well column {col} in '{well}'",
                step=step,
                suggestion="Use columns 1-12"
            ))

        return issues

    def _validate_tip_management(self, protocol: Protocol) -> List[ValidationIssue]:
        """Validate tip pick-up and drop patterns."""
        issues = []
        has_tip = False

        for i, op in enumerate(protocol.operations, 1):
            if op.type == OperationType.PICK_UP_TIP:
                if has_tip:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message="Picking up tip while already holding one",
                        step=i,
                        suggestion="Drop the current tip first"
                    ))
                has_tip = True

            elif op.type == OperationType.DROP_TIP:
                if not has_tip:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message="Dropping tip when none is attached",
                        step=i
                    ))
                has_tip = False

            elif op.type in (OperationType.ASPIRATE, OperationType.DISPENSE,
                           OperationType.MIX, OperationType.BLOWOUT):
                if not has_tip:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"{op.type.value} operation without a tip",
                        step=i,
                        suggestion="Pick up a tip before liquid handling operations"
                    ))

        # Check if tip is still attached at end
        if has_tip:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Protocol ends with tip still attached",
                suggestion="Consider dropping the tip at the end"
            ))

        return issues
