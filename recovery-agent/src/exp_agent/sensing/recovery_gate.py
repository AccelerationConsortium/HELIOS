"""
RecoveryGate - Enforce sensor-aware recovery rules.

Key rule: No blind recovery.

Any recovery action involving heating/reagent/pressure MUST satisfy:
1. Related sensors health == OK
2. Recent valid update exists (within N seconds)

Otherwise, RecoveryAgent can ONLY choose:
- SAFE_SHUTDOWN
- ASK_HUMAN

This is the "gold rule" that prevents accidents.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, Any, Set
from enum import Enum

from exp_agent.sensing.safety_state import (
    SafetyStateMachine,
    SafetyState,
    SafetyStateUpdate,
    InterlockClass,
    RecommendedAction,
)
from exp_agent.sensing.protocol.snapshot import SystemSnapshot, SensorSnapshot
from exp_agent.sensing.protocol.sensor_event import SensorType
from exp_agent.sensing.protocol.health_event import HealthStatus


class RecoveryAction(str, Enum):
    """Recovery actions that RecoveryAgent might propose."""

    # Safe actions (always allowed)
    SAFE_SHUTDOWN = "safe_shutdown"
    ASK_HUMAN = "ask_human"
    WAIT = "wait"
    LOG_ONLY = "log_only"

    # Low-risk actions (require basic sensor health)
    RETRY = "retry"
    SKIP_STEP = "skip_step"

    # Medium-risk actions (require sensor verification)
    DEGRADE = "degrade"
    REHOME = "rehome"
    FLUSH = "flush"
    DRAIN = "drain"

    # High-risk actions (require full sensor health + recent updates)
    INCREASE_HEAT = "increase_heat"
    START_HEAT = "start_heat"
    ADD_REAGENT = "add_reagent"
    INCREASE_PRESSURE = "increase_pressure"
    VENT = "vent"
    QUENCH = "quench"


class ActionRiskLevel(str, Enum):
    """Risk classification for recovery actions."""

    ALWAYS_SAFE = "always_safe"      # Always allowed
    LOW = "low"                       # Basic sensor check
    MEDIUM = "medium"                 # Sensor health check
    HIGH = "high"                     # Full verification required


@dataclass
class SensorRequirement:
    """Sensor requirement for an action."""

    sensor_type: SensorType
    max_age_seconds: float = 10.0     # Maximum age of last reading
    required_health: HealthStatus = HealthStatus.HEALTHY


@dataclass
class GateDecision:
    """Decision from RecoveryGate."""

    action: RecoveryAction
    allowed: bool
    reason: str
    alternative_actions: list[RecoveryAction] = field(default_factory=list)
    sensor_issues: list[str] = field(default_factory=list)
    requires_human: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "allowed": self.allowed,
            "reason": self.reason,
            "alternative_actions": [a.value for a in self.alternative_actions],
            "sensor_issues": self.sensor_issues,
            "requires_human": self.requires_human,
        }


class RecoveryGate:
    """
    Gate that enforces sensor-aware recovery rules.

    The fundamental rule: No heating/reagent/pressure actions
    without verified sensor state.

    Usage:
        gate = RecoveryGate(state_machine)
        decision = gate.check_action(RecoveryAction.START_HEAT, snapshot)
        if not decision.allowed:
            # Must use alternative (usually SAFE_SHUTDOWN or ASK_HUMAN)
    """

    # Action risk classification
    ACTION_RISK: dict[RecoveryAction, ActionRiskLevel] = {
        # Always safe
        RecoveryAction.SAFE_SHUTDOWN: ActionRiskLevel.ALWAYS_SAFE,
        RecoveryAction.ASK_HUMAN: ActionRiskLevel.ALWAYS_SAFE,
        RecoveryAction.WAIT: ActionRiskLevel.ALWAYS_SAFE,
        RecoveryAction.LOG_ONLY: ActionRiskLevel.ALWAYS_SAFE,
        # Low risk
        RecoveryAction.RETRY: ActionRiskLevel.LOW,
        RecoveryAction.SKIP_STEP: ActionRiskLevel.LOW,
        # Medium risk
        RecoveryAction.DEGRADE: ActionRiskLevel.MEDIUM,
        RecoveryAction.REHOME: ActionRiskLevel.MEDIUM,
        RecoveryAction.FLUSH: ActionRiskLevel.MEDIUM,
        RecoveryAction.DRAIN: ActionRiskLevel.MEDIUM,
        # High risk
        RecoveryAction.INCREASE_HEAT: ActionRiskLevel.HIGH,
        RecoveryAction.START_HEAT: ActionRiskLevel.HIGH,
        RecoveryAction.ADD_REAGENT: ActionRiskLevel.HIGH,
        RecoveryAction.INCREASE_PRESSURE: ActionRiskLevel.HIGH,
        RecoveryAction.VENT: ActionRiskLevel.HIGH,
        RecoveryAction.QUENCH: ActionRiskLevel.HIGH,
    }

    # Sensor requirements for high-risk actions
    HIGH_RISK_REQUIREMENTS: dict[RecoveryAction, list[SensorRequirement]] = {
        RecoveryAction.INCREASE_HEAT: [
            SensorRequirement(SensorType.TEMPERATURE, max_age_seconds=5.0),
        ],
        RecoveryAction.START_HEAT: [
            SensorRequirement(SensorType.TEMPERATURE, max_age_seconds=5.0),
            SensorRequirement(SensorType.AIRFLOW, max_age_seconds=10.0),
        ],
        RecoveryAction.ADD_REAGENT: [
            SensorRequirement(SensorType.TEMPERATURE, max_age_seconds=5.0),
            SensorRequirement(SensorType.PRESSURE, max_age_seconds=10.0),
            SensorRequirement(SensorType.AIRFLOW, max_age_seconds=10.0),
        ],
        RecoveryAction.INCREASE_PRESSURE: [
            SensorRequirement(SensorType.PRESSURE, max_age_seconds=5.0),
            SensorRequirement(SensorType.TEMPERATURE, max_age_seconds=10.0),
        ],
        RecoveryAction.VENT: [
            SensorRequirement(SensorType.PRESSURE, max_age_seconds=5.0),
        ],
        RecoveryAction.QUENCH: [
            SensorRequirement(SensorType.TEMPERATURE, max_age_seconds=5.0),
        ],
    }

    def __init__(
        self,
        state_machine: Optional[SafetyStateMachine] = None,
        max_sensor_age_seconds: float = 10.0,
    ):
        self._state_machine = state_machine
        self._max_sensor_age = max_sensor_age_seconds

    def check_action(
        self,
        action: RecoveryAction,
        snapshot: SystemSnapshot,
        current_state: Optional[SafetyStateUpdate] = None,
    ) -> GateDecision:
        """
        Check if a recovery action is allowed.

        Args:
            action: The recovery action to check
            snapshot: Current sensor snapshot
            current_state: Current safety state (if available)

        Returns:
            GateDecision indicating if action is allowed
        """
        risk_level = self.ACTION_RISK.get(action, ActionRiskLevel.HIGH)

        # Always-safe actions are always allowed
        if risk_level == ActionRiskLevel.ALWAYS_SAFE:
            return GateDecision(
                action=action,
                allowed=True,
                reason="Action is always safe",
            )

        # Check safety state first
        if current_state:
            # EMERGENCY state: only safe actions allowed
            if current_state.state == SafetyState.EMERGENCY:
                return GateDecision(
                    action=action,
                    allowed=False,
                    reason=f"EMERGENCY state: {current_state.reason.value}",
                    alternative_actions=[RecoveryAction.SAFE_SHUTDOWN, RecoveryAction.ASK_HUMAN],
                    requires_human=True,
                )

            # Hard interlock: only safe actions allowed
            if current_state.has_hard_interlock:
                return GateDecision(
                    action=action,
                    allowed=False,
                    reason="Hard interlock active - human confirmation required",
                    alternative_actions=[RecoveryAction.ASK_HUMAN],
                    requires_human=True,
                )

            # INTERLOCKED state: high-risk actions blocked
            if current_state.state == SafetyState.INTERLOCKED and risk_level == ActionRiskLevel.HIGH:
                return GateDecision(
                    action=action,
                    allowed=False,
                    reason=f"INTERLOCKED state: {current_state.reason.value}",
                    alternative_actions=[RecoveryAction.SAFE_SHUTDOWN, RecoveryAction.ASK_HUMAN, RecoveryAction.WAIT],
                )

        # Check sensor requirements for high-risk actions
        if risk_level == ActionRiskLevel.HIGH:
            sensor_issues = self._check_sensor_requirements(action, snapshot)
            if sensor_issues:
                return GateDecision(
                    action=action,
                    allowed=False,
                    reason="Sensor requirements not met",
                    alternative_actions=[RecoveryAction.SAFE_SHUTDOWN, RecoveryAction.ASK_HUMAN],
                    sensor_issues=sensor_issues,
                )

        # Check basic sensor health for medium-risk
        if risk_level == ActionRiskLevel.MEDIUM:
            # Medium-risk actions require at least some sensor data
            if not snapshot.sensors:
                return GateDecision(
                    action=action,
                    allowed=False,
                    reason="No sensor data available for medium-risk action",
                    alternative_actions=[RecoveryAction.WAIT, RecoveryAction.ASK_HUMAN],
                    sensor_issues=["No sensors available"],
                )
            if not snapshot.critical_sensors_ok:
                return GateDecision(
                    action=action,
                    allowed=False,
                    reason=f"Critical sensor issues: {snapshot.critical_sensor_issues}",
                    alternative_actions=[RecoveryAction.WAIT, RecoveryAction.ASK_HUMAN],
                    sensor_issues=snapshot.critical_sensor_issues,
                )

        # Action is allowed
        return GateDecision(
            action=action,
            allowed=True,
            reason="All checks passed",
        )

    def _check_sensor_requirements(
        self,
        action: RecoveryAction,
        snapshot: SystemSnapshot,
    ) -> list[str]:
        """Check sensor requirements for high-risk actions."""
        issues = []
        now = datetime.now(timezone.utc)

        requirements = self.HIGH_RISK_REQUIREMENTS.get(action, [])

        for req in requirements:
            # Find sensors of required type
            matching_sensors = snapshot.get_sensors_by_type(req.sensor_type)

            if not matching_sensors:
                issues.append(f"No {req.sensor_type.value} sensor available")
                continue

            # Check at least one sensor meets requirements
            sensor_ok = False
            for sensor in matching_sensors:
                # Check health
                if sensor.health_status.value < req.required_health.value:
                    continue

                # Check age
                if sensor.age_seconds is None:
                    continue
                if sensor.age_seconds > req.max_age_seconds:
                    continue

                sensor_ok = True
                break

            if not sensor_ok:
                issues.append(
                    f"{req.sensor_type.value}: no healthy sensor with reading < {req.max_age_seconds}s old"
                )

        return issues

    def get_allowed_actions(
        self,
        snapshot: SystemSnapshot,
        current_state: Optional[SafetyStateUpdate] = None,
    ) -> list[RecoveryAction]:
        """
        Get list of all currently allowed recovery actions.

        Useful for RecoveryAgent to know what options are available.
        """
        allowed = []
        for action in RecoveryAction:
            decision = self.check_action(action, snapshot, current_state)
            if decision.allowed:
                allowed.append(action)
        return allowed

    def get_safest_action(
        self,
        proposed_actions: list[RecoveryAction],
        snapshot: SystemSnapshot,
        current_state: Optional[SafetyStateUpdate] = None,
    ) -> RecoveryAction:
        """
        From a list of proposed actions, return the safest allowed one.

        Falls back to SAFE_SHUTDOWN if no proposed actions are allowed.
        """
        for action in proposed_actions:
            decision = self.check_action(action, snapshot, current_state)
            if decision.allowed:
                return action

        # Default to safe shutdown
        return RecoveryAction.SAFE_SHUTDOWN


# Convenience function for RecoveryAgent integration
def create_sensor_aware_recovery_check(
    state_machine: Optional[SafetyStateMachine] = None,
) -> RecoveryGate:
    """
    Create a RecoveryGate for sensor-aware recovery decisions.

    This is the main entry point for RecoveryAgent integration.
    """
    return RecoveryGate(state_machine=state_machine)
