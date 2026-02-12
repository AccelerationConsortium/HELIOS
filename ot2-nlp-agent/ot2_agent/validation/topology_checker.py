"""
Topology Checker - Verify logical ordering of workflow steps.

Checks for issues like:
- Prerequisites not met
- Incorrect step ordering
- Missing required steps
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional

from ..ir import UnitOperation, UOType, DeviceAction


@dataclass
class TopologyIssue:
    """Represents a topology/ordering issue."""
    severity: str  # "error", "warning"
    issue_type: str  # "missing_prerequisite", "wrong_order", "missing_step"
    message: str
    affected_step: str
    suggestion: str = ""


class TopologyChecker:
    """
    Checks workflow topology and step ordering.
    """

    # Standard OER workflow order
    OER_STANDARD_ORDER = [
        UOType.ELECTRODE_PREPARATION,
        UOType.ELECTROLYTE_PREPARATION,
        UOType.CELL_ASSEMBLY,
        UOType.CALIBRATION,
        UOType.MEASUREMENT,
        UOType.DATA_ANALYSIS,
        UOType.DATA_LOGGING,
        UOType.CLEANUP,
    ]

    # Required preconditions for each UO type
    PRECONDITIONS = {
        UOType.CELL_ASSEMBLY: {UOType.ELECTRODE_PREPARATION, UOType.ELECTROLYTE_PREPARATION},
        UOType.CALIBRATION: {UOType.CELL_ASSEMBLY},
        UOType.MEASUREMENT: {UOType.CELL_ASSEMBLY},
        UOType.STABILITY_TEST: {UOType.MEASUREMENT},
        UOType.DATA_ANALYSIS: {UOType.MEASUREMENT},
        UOType.CLEANUP: {UOType.MEASUREMENT},
    }

    def __init__(self):
        """Initialize topology checker."""
        pass

    def check(self, unit_operations: List[UnitOperation]) -> List[TopologyIssue]:
        """
        Check workflow topology.

        Args:
            unit_operations: List of UOs to check

        Returns:
            List of TopologyIssue objects
        """
        issues = []

        # Get UO types in order
        uo_types = [uo.uo_type for uo in unit_operations]
        seen_types: Set[UOType] = set()

        for i, uo in enumerate(unit_operations):
            # Check preconditions
            precond_issues = self._check_preconditions(uo, seen_types)
            issues.extend(precond_issues)

            # Check ordering against standard flow
            order_issues = self._check_ordering(uo, i, uo_types)
            issues.extend(order_issues)

            # Track seen types
            seen_types.add(uo.uo_type)

        # Check for missing essential steps
        missing_issues = self._check_missing_steps(uo_types)
        issues.extend(missing_issues)

        return issues

    def _check_preconditions(
        self,
        uo: UnitOperation,
        seen_types: Set[UOType]
    ) -> List[TopologyIssue]:
        """Check if preconditions are met."""
        issues = []

        required = self.PRECONDITIONS.get(uo.uo_type, set())
        missing = required - seen_types

        for missing_type in missing:
            issues.append(TopologyIssue(
                severity="warning",
                issue_type="missing_prerequisite",
                message=f"'{uo.name}' may require '{missing_type.value}' to be completed first",
                affected_step=uo.name,
                suggestion=f"Consider adding {missing_type.value} before {uo.name}",
            ))

        return issues

    def _check_ordering(
        self,
        uo: UnitOperation,
        index: int,
        all_types: List[UOType]
    ) -> List[TopologyIssue]:
        """Check if UO is in expected position."""
        issues = []

        # Get expected position in standard order
        try:
            expected_pos = self.OER_STANDARD_ORDER.index(uo.uo_type)
        except ValueError:
            # Not in standard order, skip
            return issues

        # Check if any later-ordered UO appears before this one
        for j, other_type in enumerate(all_types[:index]):
            try:
                other_pos = self.OER_STANDARD_ORDER.index(other_type)
                if other_pos > expected_pos:
                    issues.append(TopologyIssue(
                        severity="warning",
                        issue_type="wrong_order",
                        message=f"'{uo.name}' ({uo.uo_type.value}) appears after a later-stage step",
                        affected_step=uo.name,
                        suggestion="Consider reordering workflow steps",
                    ))
                    break
            except ValueError:
                continue

        return issues

    def _check_missing_steps(self, uo_types: List[UOType]) -> List[TopologyIssue]:
        """Check for missing essential steps."""
        issues = []
        type_set = set(uo_types)

        # Essential steps for OER
        if UOType.MEASUREMENT in type_set:
            # If doing measurement, should have calibration
            if UOType.CALIBRATION not in type_set:
                issues.append(TopologyIssue(
                    severity="warning",
                    issue_type="missing_step",
                    message="Measurement without calibration may give incorrect results",
                    affected_step="workflow",
                    suggestion="Add reference electrode calibration step",
                ))

            # If doing measurement, should have cell assembly
            if UOType.CELL_ASSEMBLY not in type_set:
                issues.append(TopologyIssue(
                    severity="warning",
                    issue_type="missing_step",
                    message="Measurement requires cell assembly",
                    affected_step="workflow",
                    suggestion="Add cell assembly step before measurement",
                ))

        return issues

    def check_device_actions(self, actions: List[DeviceAction]) -> List[TopologyIssue]:
        """
        Check device action ordering (lower level check).

        Args:
            actions: List of device actions

        Returns:
            List of TopologyIssue objects
        """
        issues = []

        # Track state
        has_pick_up_tip = False
        has_current_tip = False

        for action in actions:
            # OT-2 specific checks
            if action.device_type == "liquid_handler":
                cmd = action.command

                if cmd in ["aspirate", "dispense", "transfer", "mix"]:
                    if not has_current_tip and action.params.get("new_tip") != "always":
                        issues.append(TopologyIssue(
                            severity="warning",
                            issue_type="missing_prerequisite",
                            message=f"'{action.name}' may need tip pickup first",
                            affected_step=action.name,
                            suggestion="Add pick_up_tip before liquid handling",
                        ))

                if cmd == "pick_up_tip":
                    has_current_tip = True
                    has_pick_up_tip = True

                if cmd == "drop_tip":
                    has_current_tip = False

        return issues
