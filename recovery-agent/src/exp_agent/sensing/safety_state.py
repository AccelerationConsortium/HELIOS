"""
SafetyStateMachine - Unified safety state abstraction for decision layers.

Replaces ad-hoc veto logic with a stable abstraction that:
1. Outputs a well-defined state (SAFE/DEGRADED/INTERLOCKED/EMERGENCY)
2. Provides evidence chain (snapshot_id, trigger_events, window_ref)
3. Distinguishes soft vs hard interlocks
4. Supports hysteresis to prevent state oscillation

Based on Phase 3 requirements:
- RecoveryAgent consumes SafetyStateUpdate instead of building its own logic
- Every state change has auditable evidence
- Hard interlocks force ASK_HUMAN, prevent auto-retry
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, IntEnum
from typing import Optional, Any, Callable
import hashlib
import json

from exp_agent.sensing.protocol.sensor_event import SensorEvent, SensorType
from exp_agent.sensing.protocol.snapshot import SystemSnapshot
from exp_agent.sensing.protocol.health_event import HealthStatus


class SafetyState(IntEnum):
    """
    Safety state hierarchy (higher = more severe).

    SAFE: Normal operation, all actions allowed
    DEGRADED: High-risk actions blocked, normal ops continue
    INTERLOCKED: Force stop/safe mode, limited recovery allowed
    EMERGENCY: Evacuate/human takeover required
    """

    SAFE = 0
    DEGRADED = 1
    INTERLOCKED = 2
    EMERGENCY = 3


class InterlockClass(str, Enum):
    """Classification of interlock severity."""

    SOFT = "soft"           # Software-enforceable, auto-recovery possible
    HARD_REQUIRED = "hard"  # Requires human confirmation, no auto-recovery


class InterlockReason(str, Enum):
    """Enumerated reasons for interlock (not free text)."""

    # Airflow
    HOOD_AIRFLOW_LOW = "hood_airflow_low"
    HOOD_AIRFLOW_ZERO = "hood_airflow_zero"

    # Temperature
    TEMPERATURE_HIGH = "temperature_high"
    TEMPERATURE_RUNAWAY = "temperature_runaway"  # Slope too high
    TEMPERATURE_CRITICAL = "temperature_critical"

    # Pressure
    PRESSURE_HIGH = "pressure_high"
    PRESSURE_CRITICAL = "pressure_critical"

    # Sensor health
    SENSOR_STALE = "sensor_stale"
    SENSOR_STUCK = "sensor_stuck"
    SENSOR_OFFLINE = "sensor_offline"
    CRITICAL_SENSOR_FAILURE = "critical_sensor_failure"

    # Emergency
    ESTOP_TRIGGERED = "estop_triggered"
    POWER_FAILURE = "power_failure"
    SPILL_DETECTED = "spill_detected"
    FIRE_DETECTED = "fire_detected"

    # Multiple
    MULTIPLE_INTERLOCKS = "multiple_interlocks"


class RecommendedAction(str, Enum):
    """Actions that can be recommended by safety state."""

    CONTINUE = "continue"           # Normal operation
    BLOCK_HIGH_RISK = "block_high_risk"  # Block heating/pressure/reagent
    STOP_HEATING = "stop_heating"
    STOP_REAGENT_ADD = "stop_reagent_add"
    EXECUTE_VENT = "execute_vent"
    ENTER_SAFE_MODE = "enter_safe_mode"
    SAFE_SHUTDOWN = "safe_shutdown"
    EVACUATE = "evacuate"
    ASK_HUMAN = "ask_human"
    WAIT_SENSOR_RECOVERY = "wait_sensor_recovery"


@dataclass(frozen=True)
class EvidenceChain:
    """
    Auditable evidence for a safety state change.

    Every veto/degrade must have traceable evidence for post-incident analysis.
    """

    # Snapshot at time of decision
    snapshot_id: str                   # Hash of snapshot
    snapshot_ts: datetime

    # Time window of events considered
    window_start: datetime
    window_end: datetime

    # Events that triggered this state
    trigger_event_ids: tuple[str, ...]  # Immutable for hashing

    # Summary metrics at trigger time
    trigger_values: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "snapshot_ts": self.snapshot_ts.isoformat(),
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "trigger_event_ids": list(self.trigger_event_ids),
            "trigger_values": self.trigger_values,
        }

    @staticmethod
    def compute_snapshot_id(snapshot: SystemSnapshot) -> str:
        """Compute deterministic hash of snapshot state."""
        # Hash key sensor values for reproducibility
        data = {
            "ts": snapshot.ts.isoformat(),
            "status": snapshot.system_status.value,
            "sensors": {
                sid: {
                    "value": s.latest_value,
                    "health": s.health_status.value,
                }
                for sid, s in snapshot.sensors.items()
            },
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]


@dataclass
class Interlock:
    """A single interlock condition."""

    reason: InterlockReason
    interlock_class: InterlockClass
    sensor_id: str
    current_value: Optional[float]
    threshold: Optional[float]
    message: str
    triggered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason.value,
            "class": self.interlock_class.value,
            "sensor_id": self.sensor_id,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "message": self.message,
            "triggered_at": self.triggered_at.isoformat(),
        }


@dataclass
class SafetyStateUpdate:
    """
    Complete safety state update consumed by RecoveryAgent.

    This is the stable abstraction that decision layers depend on.
    RecoveryAgent should not build its own veto logic - it consumes this.
    """

    # Current state
    state: SafetyState
    previous_state: Optional[SafetyState] = None

    # Primary reason (use MULTIPLE_INTERLOCKS if >1)
    reason: InterlockReason = InterlockReason.HOOD_AIRFLOW_LOW

    # All active interlocks
    interlocks: list[Interlock] = field(default_factory=list)

    # Evidence for audit
    evidence: Optional[EvidenceChain] = None

    # Recommended actions (ordered by priority)
    recommended_actions: list[RecommendedAction] = field(default_factory=list)

    # Timestamp
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Optional explanation from SafetyAdvisor
    explanation: Optional[str] = None

    @property
    def is_state_change(self) -> bool:
        return self.previous_state is not None and self.previous_state != self.state

    @property
    def has_hard_interlock(self) -> bool:
        return any(i.interlock_class == InterlockClass.HARD_REQUIRED for i in self.interlocks)

    @property
    def allows_auto_recovery(self) -> bool:
        """Check if automatic recovery is allowed."""
        if self.state >= SafetyState.EMERGENCY:
            return False
        if self.has_hard_interlock:
            return False
        return True

    @property
    def requires_human(self) -> bool:
        """Check if human intervention is required."""
        return self.state >= SafetyState.EMERGENCY or self.has_hard_interlock

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.name,
            "previous_state": self.previous_state.name if self.previous_state else None,
            "reason": self.reason.value,
            "interlocks": [i.to_dict() for i in self.interlocks],
            "evidence": self.evidence.to_dict() if self.evidence else None,
            "recommended_actions": [a.value for a in self.recommended_actions],
            "ts": self.ts.isoformat(),
            "explanation": self.explanation,
            "is_state_change": self.is_state_change,
            "has_hard_interlock": self.has_hard_interlock,
            "allows_auto_recovery": self.allows_auto_recovery,
            "requires_human": self.requires_human,
        }


@dataclass
class HysteresisConfig:
    """Configuration for state transition hysteresis."""

    # Minimum time in state before allowing transition down
    min_hold_time_ms: float = 5000.0

    # Transition delays (state -> state)
    # INTERLOCKED -> DEGRADED requires this many clean readings
    recovery_threshold_readings: int = 3

    # DEGRADED -> SAFE requires this duration of clean state
    safe_recovery_delay_ms: float = 10000.0


class SafetyStateMachine:
    """
    State machine for safety state transitions.

    Features:
    - Hysteresis to prevent oscillation
    - Priority-based state calculation
    - Evidence chain generation
    - Interlock aggregation
    """

    def __init__(
        self,
        hysteresis: Optional[HysteresisConfig] = None,
    ):
        self.hysteresis = hysteresis or HysteresisConfig()

        # Current state
        self._state = SafetyState.SAFE
        self._state_entered_at = datetime.now(timezone.utc)
        self._clean_readings_count = 0
        self._last_clean_at: Optional[datetime] = None

        # Active interlocks
        self._active_interlocks: dict[str, Interlock] = {}

        # Evidence tracking
        self._trigger_events: list[str] = []
        self._trigger_values: dict[str, float] = {}

        # Callbacks
        self._state_callbacks: list[Callable[[SafetyStateUpdate], None]] = []

        # Interlock definitions
        self._interlock_defs = self._build_interlock_definitions()

    def _build_interlock_definitions(self) -> dict:
        """Build interlock condition definitions."""
        return {
            # Hood airflow interlocks
            "hood_airflow_low": {
                "sensor_type": SensorType.AIRFLOW,
                "check": lambda v, cfg: v < cfg.get("min_airflow", 0.3),
                "reason": InterlockReason.HOOD_AIRFLOW_LOW,
                "class": InterlockClass.SOFT,
                "state": SafetyState.INTERLOCKED,
                "actions": [RecommendedAction.STOP_REAGENT_ADD, RecommendedAction.ENTER_SAFE_MODE],
            },
            "hood_airflow_zero": {
                "sensor_type": SensorType.AIRFLOW,
                "check": lambda v, cfg: v < 0.1,
                "reason": InterlockReason.HOOD_AIRFLOW_ZERO,
                "class": InterlockClass.HARD_REQUIRED,
                "state": SafetyState.EMERGENCY,
                "actions": [RecommendedAction.SAFE_SHUTDOWN, RecommendedAction.ASK_HUMAN],
            },
            # Temperature interlocks
            "temperature_high": {
                "sensor_type": SensorType.TEMPERATURE,
                "check": lambda v, cfg: v > cfg.get("max_temp", 130),
                "reason": InterlockReason.TEMPERATURE_HIGH,
                "class": InterlockClass.SOFT,
                "state": SafetyState.INTERLOCKED,
                "actions": [RecommendedAction.STOP_HEATING],
            },
            "temperature_critical": {
                "sensor_type": SensorType.TEMPERATURE,
                "check": lambda v, cfg: v > cfg.get("critical_temp", 150),
                "reason": InterlockReason.TEMPERATURE_CRITICAL,
                "class": InterlockClass.HARD_REQUIRED,
                "state": SafetyState.EMERGENCY,
                "actions": [RecommendedAction.SAFE_SHUTDOWN, RecommendedAction.EVACUATE],
            },
            # Pressure interlocks
            "pressure_high": {
                "sensor_type": SensorType.PRESSURE,
                "check": lambda v, cfg: v > cfg.get("max_pressure", 200),
                "reason": InterlockReason.PRESSURE_HIGH,
                "class": InterlockClass.SOFT,
                "state": SafetyState.INTERLOCKED,
                "actions": [RecommendedAction.STOP_HEATING, RecommendedAction.EXECUTE_VENT],
            },
            "pressure_critical": {
                "sensor_type": SensorType.PRESSURE,
                "check": lambda v, cfg: v > cfg.get("critical_pressure", 300),
                "reason": InterlockReason.PRESSURE_CRITICAL,
                "class": InterlockClass.HARD_REQUIRED,
                "state": SafetyState.EMERGENCY,
                "actions": [RecommendedAction.EXECUTE_VENT, RecommendedAction.EVACUATE],
            },
            # E-stop
            "estop_triggered": {
                "sensor_type": SensorType.ESTOP,
                "check": lambda v, cfg: v > 0.5,
                "reason": InterlockReason.ESTOP_TRIGGERED,
                "class": InterlockClass.HARD_REQUIRED,
                "state": SafetyState.EMERGENCY,
                "actions": [RecommendedAction.SAFE_SHUTDOWN, RecommendedAction.ASK_HUMAN],
            },
        }

    def set_state_callback(self, callback: Callable[[SafetyStateUpdate], None]) -> None:
        """Add callback for state changes."""
        self._state_callbacks.append(callback)

    def process_snapshot(
        self,
        snapshot: SystemSnapshot,
        config: Optional[dict[str, float]] = None,
        window_events: Optional[list[SensorEvent]] = None,
    ) -> SafetyStateUpdate:
        """
        Process a system snapshot and return current safety state.

        Args:
            snapshot: Current system snapshot
            config: Threshold configuration overrides
            window_events: Recent events for evidence chain

        Returns:
            SafetyStateUpdate with state, interlocks, evidence, and actions
        """
        config = config or {}
        now = datetime.now(timezone.utc)
        previous_state = self._state

        # Clear trigger tracking for this evaluation
        self._trigger_events = []
        self._trigger_values = {}

        # Evaluate all interlock conditions
        new_interlocks: list[Interlock] = []

        for sensor_id, sensor in snapshot.sensors.items():
            # Check sensor health first
            if sensor.health_status == HealthStatus.OFFLINE:
                new_interlocks.append(Interlock(
                    reason=InterlockReason.SENSOR_OFFLINE,
                    interlock_class=InterlockClass.SOFT,
                    sensor_id=sensor_id,
                    current_value=None,
                    threshold=None,
                    message=f"Sensor {sensor_id} is offline",
                ))
                self._trigger_events.append(sensor.latest_event.event_id if sensor.latest_event else "")

            elif sensor.health_status == HealthStatus.UNHEALTHY:
                # Check if critical sensor
                if sensor.sensor_type in (SensorType.TEMPERATURE, SensorType.PRESSURE, SensorType.AIRFLOW):
                    new_interlocks.append(Interlock(
                        reason=InterlockReason.CRITICAL_SENSOR_FAILURE,
                        interlock_class=InterlockClass.HARD_REQUIRED,
                        sensor_id=sensor_id,
                        current_value=sensor.latest_value,
                        threshold=None,
                        message=f"Critical sensor {sensor_id} is unhealthy",
                    ))

            # Check value-based interlocks
            if sensor.latest_value is not None:
                self._trigger_values[sensor_id] = sensor.latest_value

                for def_key, idef in self._interlock_defs.items():
                    if idef["sensor_type"] == sensor.sensor_type:
                        if idef["check"](sensor.latest_value, config):
                            new_interlocks.append(Interlock(
                                reason=idef["reason"],
                                interlock_class=idef["class"],
                                sensor_id=sensor_id,
                                current_value=sensor.latest_value,
                                threshold=config.get(def_key.split("_")[0] + "_threshold"),
                                message=f"{idef['reason'].value}: {sensor.latest_value}",
                            ))
                            if sensor.latest_event:
                                self._trigger_events.append(sensor.latest_event.event_id)

        # Calculate new state based on worst interlock
        calculated_state = SafetyState.SAFE
        for interlock in new_interlocks:
            for def_key, idef in self._interlock_defs.items():
                if idef["reason"] == interlock.reason:
                    if idef["state"] > calculated_state:
                        calculated_state = idef["state"]
                    break
            else:
                # Sensor health interlocks
                if interlock.reason == InterlockReason.SENSOR_OFFLINE:
                    calculated_state = max(calculated_state, SafetyState.DEGRADED)
                elif interlock.reason == InterlockReason.CRITICAL_SENSOR_FAILURE:
                    calculated_state = max(calculated_state, SafetyState.INTERLOCKED)

        # Apply hysteresis for downward transitions
        new_state = self._apply_hysteresis(calculated_state, len(new_interlocks) == 0, now)

        # Update internal state
        if new_state != self._state:
            self._state = new_state
            self._state_entered_at = now

        # Build evidence chain
        evidence = EvidenceChain(
            snapshot_id=EvidenceChain.compute_snapshot_id(snapshot),
            snapshot_ts=snapshot.ts,
            window_start=snapshot.ts - timedelta(seconds=60),
            window_end=snapshot.ts,
            trigger_event_ids=tuple(self._trigger_events),
            trigger_values=dict(self._trigger_values),
        )

        # Determine primary reason
        if len(new_interlocks) == 0:
            primary_reason = InterlockReason.HOOD_AIRFLOW_LOW  # Placeholder
        elif len(new_interlocks) == 1:
            primary_reason = new_interlocks[0].reason
        else:
            # Find highest severity
            primary_reason = InterlockReason.MULTIPLE_INTERLOCKS

        # Collect recommended actions
        actions = self._collect_actions(new_interlocks, new_state)

        # Build update
        update = SafetyStateUpdate(
            state=new_state,
            previous_state=previous_state if new_state != previous_state else None,
            reason=primary_reason,
            interlocks=new_interlocks,
            evidence=evidence,
            recommended_actions=actions,
            ts=now,
        )

        # Store active interlocks
        self._active_interlocks = {i.sensor_id: i for i in new_interlocks}

        # Notify callbacks on state change
        if update.is_state_change:
            for callback in self._state_callbacks:
                try:
                    callback(update)
                except Exception as e:
                    print(f"Error in state callback: {e}")

        return update

    def _apply_hysteresis(
        self,
        calculated_state: SafetyState,
        is_clean: bool,
        now: datetime,
    ) -> SafetyState:
        """Apply hysteresis to prevent state oscillation."""
        # Upward transitions are immediate
        if calculated_state > self._state:
            self._clean_readings_count = 0
            self._last_clean_at = None
            return calculated_state

        # Downward transitions require hysteresis
        if calculated_state < self._state:
            if is_clean:
                self._clean_readings_count += 1
                if self._last_clean_at is None:
                    self._last_clean_at = now

                # Check if we've been clean long enough
                time_since_enter = (now - self._state_entered_at).total_seconds() * 1000
                if time_since_enter < self.hysteresis.min_hold_time_ms:
                    return self._state  # Hold current state

                if self._clean_readings_count >= self.hysteresis.recovery_threshold_readings:
                    time_clean = (now - self._last_clean_at).total_seconds() * 1000
                    if time_clean >= self.hysteresis.safe_recovery_delay_ms:
                        # Allow transition down
                        self._clean_readings_count = 0
                        self._last_clean_at = None
                        return calculated_state
            else:
                self._clean_readings_count = 0
                self._last_clean_at = None

            return self._state  # Hold current state

        return calculated_state

    def _collect_actions(
        self,
        interlocks: list[Interlock],
        state: SafetyState,
    ) -> list[RecommendedAction]:
        """Collect and prioritize recommended actions."""
        actions: set[RecommendedAction] = set()

        # State-based actions
        if state == SafetyState.SAFE:
            actions.add(RecommendedAction.CONTINUE)
        elif state == SafetyState.DEGRADED:
            actions.add(RecommendedAction.BLOCK_HIGH_RISK)
        elif state == SafetyState.INTERLOCKED:
            actions.add(RecommendedAction.ENTER_SAFE_MODE)
        elif state == SafetyState.EMERGENCY:
            actions.add(RecommendedAction.SAFE_SHUTDOWN)
            actions.add(RecommendedAction.ASK_HUMAN)

        # Interlock-specific actions
        for interlock in interlocks:
            for def_key, idef in self._interlock_defs.items():
                if idef["reason"] == interlock.reason:
                    actions.update(idef["actions"])
                    break

        # Hard interlocks always require human
        if any(i.interlock_class == InterlockClass.HARD_REQUIRED for i in interlocks):
            actions.add(RecommendedAction.ASK_HUMAN)

        # Prioritize actions
        priority = [
            RecommendedAction.EVACUATE,
            RecommendedAction.SAFE_SHUTDOWN,
            RecommendedAction.ASK_HUMAN,
            RecommendedAction.EXECUTE_VENT,
            RecommendedAction.STOP_HEATING,
            RecommendedAction.STOP_REAGENT_ADD,
            RecommendedAction.ENTER_SAFE_MODE,
            RecommendedAction.WAIT_SENSOR_RECOVERY,
            RecommendedAction.BLOCK_HIGH_RISK,
            RecommendedAction.CONTINUE,
        ]

        return [a for a in priority if a in actions]

    @property
    def current_state(self) -> SafetyState:
        """Get current safety state."""
        return self._state

    @property
    def active_interlocks(self) -> list[Interlock]:
        """Get list of active interlocks."""
        return list(self._active_interlocks.values())

    def get_status(self) -> dict[str, Any]:
        """Get current status summary."""
        return {
            "state": self._state.name,
            "state_entered_at": self._state_entered_at.isoformat(),
            "active_interlocks": [i.to_dict() for i in self._active_interlocks.values()],
            "clean_readings_count": self._clean_readings_count,
            "allows_auto_recovery": self._state < SafetyState.EMERGENCY and not any(
                i.interlock_class == InterlockClass.HARD_REQUIRED
                for i in self._active_interlocks.values()
            ),
        }
