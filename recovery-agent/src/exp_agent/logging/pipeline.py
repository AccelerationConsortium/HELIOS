"""
Log-Decision-Recovery Pipeline - Structured event logging with decision tracing.

This module provides:
1. Structured logging for all experiment events
2. Decision audit trail with full context
3. Recovery action tracking with outcomes
4. Pipeline that connects: Event → Log → Decision → Recovery → Outcome
"""
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional
from collections import deque

from ..core.types import DeviceState, HardwareError, Decision, Action
from ..llm.types import LLMDecisionProposal


# ============================================================================
# Event Types
# ============================================================================

class EventType(str, Enum):
    """Categories of loggable events in the pipeline."""
    # Observation events
    DEVICE_READ = "device.read"
    DEVICE_TICK = "device.tick"
    TELEMETRY_UPDATE = "telemetry.update"

    # Execution events
    ACTION_PROPOSED = "action.proposed"
    ACTION_STARTED = "action.started"
    ACTION_COMPLETED = "action.completed"
    ACTION_FAILED = "action.failed"

    # Error events
    ERROR_DETECTED = "error.detected"
    ERROR_CLASSIFIED = "error.classified"

    # Decision events
    SIGNATURE_ANALYZED = "decision.signature_analyzed"
    DECISION_MADE = "decision.made"
    DECISION_EXECUTED = "decision.executed"

    # LLM events (Phase 2)
    LLM_PROPOSAL = "llm.proposal"

    # Anomaly packaging (Phase 3 groundwork)
    ANOMALY_PACKET = "anomaly.packet"

    # Recovery events
    RECOVERY_STARTED = "recovery.started"
    RECOVERY_ACTION = "recovery.action"
    RECOVERY_COMPLETED = "recovery.completed"
    RECOVERY_FAILED = "recovery.failed"

    # Lifecycle events
    EXPERIMENT_STARTED = "experiment.started"
    EXPERIMENT_COMPLETED = "experiment.completed"
    SHUTDOWN_INITIATED = "shutdown.initiated"
    SHUTDOWN_COMPLETED = "shutdown.completed"


class LogLevel(str, Enum):
    """Log severity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# ============================================================================
# Structured Log Event
# ============================================================================

@dataclass
class LogEvent:
    """
    Structured log event with full context for audit trail.

    Every event in the pipeline produces one of these, enabling:
    - Full replay of experiment execution
    - Decision audit trails
    - Error forensics
    - Performance analysis
    """
    event_id: str
    timestamp: str  # ISO format
    event_type: EventType
    level: LogLevel
    message: str

    # Context
    device_name: Optional[str] = None
    step_number: Optional[int] = None
    correlation_id: Optional[str] = None  # Links related events

    # Payload (event-specific data)
    payload: Dict[str, Any] = field(default_factory=dict)

    # Telemetry snapshot at event time
    telemetry_snapshot: Optional[Dict[str, Any]] = None

    # Error context (if applicable)
    error_type: Optional[str] = None
    error_severity: Optional[str] = None

    # Decision context (if applicable)
    decision_kind: Optional[str] = None
    decision_rationale: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), default=str)


# ============================================================================
# Log Storage Backend
# ============================================================================

class LogBackend:
    """Abstract base for log storage."""
    def write(self, event: LogEvent) -> None:
        raise NotImplementedError

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class MemoryBackend(LogBackend):
    """In-memory storage for testing and analysis."""

    def __init__(self, max_events: int = 10000):
        self.events: deque[LogEvent] = deque(maxlen=max_events)

    def write(self, event: LogEvent) -> None:
        self.events.append(event)

    def query(
        self,
        event_type: Optional[EventType] = None,
        level: Optional[LogLevel] = None,
        device: Optional[str] = None,
        correlation_id: Optional[str] = None,
        since: Optional[str] = None,
    ) -> List[LogEvent]:
        """Query events with filters."""
        results = []
        for e in self.events:
            if event_type and e.event_type != event_type:
                continue
            if level and e.level != level:
                continue
            if device and e.device_name != device:
                continue
            if correlation_id and e.correlation_id != correlation_id:
                continue
            if since and e.timestamp < since:
                continue
            results.append(e)
        return results

    def get_decision_trail(self, correlation_id: str) -> List[LogEvent]:
        """Get all events related to a decision."""
        return self.query(correlation_id=correlation_id)


class FileBackend(LogBackend):
    """JSON-lines file storage for persistence."""

    def __init__(self, log_dir: Path, experiment_id: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{experiment_id}.jsonl"
        self._handle = open(self.log_file, "a")

    def write(self, event: LogEvent) -> None:
        self._handle.write(event.to_json() + "\n")

    def flush(self) -> None:
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


class ConsoleBackend(LogBackend):
    """Pretty-printed console output."""

    LEVEL_COLORS = {
        LogLevel.DEBUG: "\033[90m",     # Gray
        LogLevel.INFO: "\033[0m",       # Default
        LogLevel.WARNING: "\033[93m",   # Yellow
        LogLevel.ERROR: "\033[91m",     # Red
        LogLevel.CRITICAL: "\033[95m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, min_level: LogLevel = LogLevel.INFO, use_color: bool = True):
        self.min_level = min_level
        self.use_color = use_color
        self._level_order = list(LogLevel)

    def write(self, event: LogEvent) -> None:
        if self._level_order.index(event.level) < self._level_order.index(self.min_level):
            return

        color = self.LEVEL_COLORS.get(event.level, "") if self.use_color else ""
        reset = self.RESET if self.use_color else ""

        # Format: [LEVEL] [EventType] Message (device) {key=value}
        parts = [
            f"{color}[{event.level.value:8}]{reset}",
            f"[{event.event_type.value:25}]",
            event.message,
        ]

        if event.device_name:
            parts.append(f"({event.device_name})")

        # Add key payload fields inline
        if event.payload:
            kv = " ".join(f"{k}={v}" for k, v in list(event.payload.items())[:3])
            parts.append(f"{{{kv}}}")

        print(" ".join(parts))


# ============================================================================
# Pipeline Logger
# ============================================================================

class PipelineLogger:
    """
    Central logger for the log-decision-recovery pipeline.

    Usage:
        logger = PipelineLogger(experiment_id="exp_001")
        logger.add_backend(ConsoleBackend())
        logger.add_backend(FileBackend(Path("logs"), "exp_001"))

        # Log events
        logger.log_device_read(device_state)
        logger.log_error(error, device_state)
        logger.log_decision(decision, error, signature)
    """

    def __init__(self, experiment_id: str = None):
        self.experiment_id = experiment_id or f"exp_{int(time.time())}"
        self.backends: List[LogBackend] = []
        self._step_number = 0
        self._correlation_stack: List[str] = []

    def add_backend(self, backend: LogBackend) -> None:
        self.backends.append(backend)

    def set_step(self, step: int) -> None:
        self._step_number = step

    def start_correlation(self, context: str = "operation") -> str:
        """Start a new correlation context (e.g., for a recovery operation)."""
        corr_id = f"{context}_{uuid.uuid4().hex[:8]}"
        self._correlation_stack.append(corr_id)
        return corr_id

    def end_correlation(self) -> None:
        if self._correlation_stack:
            self._correlation_stack.pop()

    @property
    def current_correlation(self) -> Optional[str]:
        return self._correlation_stack[-1] if self._correlation_stack else None

    def _emit(self, event: LogEvent) -> None:
        """Emit event to all backends."""
        for backend in self.backends:
            backend.write(event)

    def _create_event(
        self,
        event_type: EventType,
        level: LogLevel,
        message: str,
        device_name: str = None,
        payload: Dict[str, Any] = None,
        telemetry: Dict[str, Any] = None,
        **kwargs
    ) -> LogEvent:
        return LogEvent(
            event_id=uuid.uuid4().hex[:12],
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            level=level,
            message=message,
            device_name=device_name,
            step_number=self._step_number,
            correlation_id=self.current_correlation,
            payload=payload or {},
            telemetry_snapshot=telemetry,
            **kwargs
        )

    # ========================================================================
    # Observation Events
    # ========================================================================

    def log_device_read(self, state: DeviceState) -> None:
        event = self._create_event(
            EventType.DEVICE_READ,
            LogLevel.DEBUG,
            f"Read state from {state.name}: status={state.status}",
            device_name=state.name,
            telemetry=state.telemetry,
            payload={"status": state.status}
        )
        self._emit(event)

    def log_telemetry_update(self, state: DeviceState, history_len: int) -> None:
        temp = state.telemetry.get("temperature", "N/A")
        target = state.telemetry.get("target", "N/A")
        event = self._create_event(
            EventType.TELEMETRY_UPDATE,
            LogLevel.INFO,
            f"Telemetry: temp={temp:.1f}°C target={target}°C",
            device_name=state.name,
            telemetry=state.telemetry,
            payload={"history_length": history_len}
        )
        self._emit(event)

    # ========================================================================
    # Execution Events
    # ========================================================================

    def log_action_proposed(self, action: Action) -> None:
        event = self._create_event(
            EventType.ACTION_PROPOSED,
            LogLevel.INFO,
            f"Proposed: {action.name}",
            device_name=action.device,
            payload={
                "action_name": action.name,
                "params": action.params,
                "postconditions": action.postconditions[:2] if action.postconditions else []
            }
        )
        self._emit(event)

    def log_action_started(self, action: Action) -> None:
        event = self._create_event(
            EventType.ACTION_STARTED,
            LogLevel.INFO,
            f"Executing: {action.name}",
            device_name=action.device,
            payload={"action_name": action.name, "params": action.params}
        )
        self._emit(event)

    def log_action_completed(self, action: Action, duration_ms: float) -> None:
        event = self._create_event(
            EventType.ACTION_COMPLETED,
            LogLevel.INFO,
            f"Completed: {action.name} ({duration_ms:.0f}ms)",
            device_name=action.device,
            payload={"action_name": action.name, "duration_ms": duration_ms}
        )
        self._emit(event)

    def log_action_failed(self, action: Action, error: HardwareError) -> None:
        event = self._create_event(
            EventType.ACTION_FAILED,
            LogLevel.ERROR,
            f"Failed: {action.name} - {error.message}",
            device_name=action.device,
            payload={"action_name": action.name},
            error_type=error.type,
            error_severity=error.severity
        )
        self._emit(event)

    # ========================================================================
    # Error Events
    # ========================================================================

    def log_error_detected(
        self,
        error: HardwareError,
        state: Optional[DeviceState] = None
    ) -> str:
        """Log error detection. Returns correlation_id for tracking recovery."""
        corr_id = self.start_correlation("error")

        event = self._create_event(
            EventType.ERROR_DETECTED,
            LogLevel.ERROR,
            f"Error detected: {error.type} - {error.message}",
            device_name=error.device,
            telemetry=state.telemetry if state else None,
            payload={"context": error.context},
            error_type=error.type,
            error_severity=error.severity
        )
        self._emit(event)
        return corr_id

    def log_error_classified(
        self,
        error: HardwareError,
        unsafe: bool,
        recoverable: bool,
        strategy: str
    ) -> None:
        level = LogLevel.CRITICAL if unsafe else LogLevel.WARNING
        event = self._create_event(
            EventType.ERROR_CLASSIFIED,
            level,
            f"Classified: unsafe={unsafe}, recoverable={recoverable}, strategy={strategy}",
            device_name=error.device,
            payload={
                "unsafe": unsafe,
                "recoverable": recoverable,
                "default_strategy": strategy
            },
            error_type=error.type
        )
        self._emit(event)

    # ========================================================================
    # Decision Events
    # ========================================================================

    def log_signature_analyzed(
        self,
        mode: str,
        confidence: float,
        features: Dict[str, Any]
    ) -> None:
        event = self._create_event(
            EventType.SIGNATURE_ANALYZED,
            LogLevel.INFO,
            f"Signature: {mode} (confidence={confidence:.2f})",
            payload={
                "mode": mode,
                "confidence": confidence,
                "avg_slope": features.get("avg_slope"),
                "variance": features.get("variance"),
                "sign_changes": features.get("sign_changes")
            }
        )
        self._emit(event)

    def log_decision_made(
        self,
        decision: Decision,
        error: HardwareError,
        retry_count: int,
    ) -> None:
        level = LogLevel.WARNING if decision.kind in ["abort", "degrade"] else LogLevel.INFO

        event = self._create_event(
            EventType.DECISION_MADE,
            level,
            f"Decision: {decision.kind.upper()} - {decision.rationale}",
            device_name=error.device,
            payload={
                "retry_count": retry_count,
                "action_count": len(decision.actions),
                "actions": [a.name for a in decision.actions],
            },
            error_type=error.type,
            decision_kind=decision.kind,
            decision_rationale=decision.rationale
        )
        self._emit(event)

    # ========================================================================
    # LLM Events (Phase 2)
    # ========================================================================

    def log_llm_proposal(
        self,
        proposal: LLMDecisionProposal,
        error: HardwareError,
    ) -> None:
        """Log an LLM proposal (advisory only)."""
        event = self._create_event(
            EventType.LLM_PROPOSAL,
            LogLevel.INFO,
            f"LLM proposal: {proposal.kind.upper()} - {proposal.rationale}",
            device_name=error.device,
            payload={
                "kind": proposal.kind,
                "rationale": proposal.rationale,
                "action_count": len(proposal.actions),
                "actions": [a.name for a in proposal.actions],
                "confidence": proposal.confidence,
                "model": proposal.model,
                "provider": proposal.provider,
                "notes": proposal.notes,
            },
            error_type=error.type,
        )
        self._emit(event)

    # ========================================================================
    # Anomaly Packaging (Phase 3 groundwork)
    # ========================================================================

    def log_anomaly_packet(self, packet: Dict[str, Any], device_name: Optional[str] = None) -> None:
        """Persist a compact anomaly packet as a log event payload."""
        event = self._create_event(
            EventType.ANOMALY_PACKET,
            LogLevel.WARNING,
            "Anomaly packet created",
            device_name=device_name,
            payload=packet,
        )
        self._emit(event)

    # ========================================================================
    # Recovery Events
    # ========================================================================

    def log_recovery_started(self, decision: Decision) -> None:
        event = self._create_event(
            EventType.RECOVERY_STARTED,
            LogLevel.WARNING,
            f"Starting recovery: {decision.kind} with {len(decision.actions)} actions",
            payload={
                "decision_kind": decision.kind,
                "action_count": len(decision.actions),
                "actions": [a.name for a in decision.actions]
            },
            decision_kind=decision.kind
        )
        self._emit(event)

    def log_recovery_action(
        self,
        action: Action,
        index: int,
        total: int,
        success: bool,
        duration_ms: float = 0
    ) -> None:
        level = LogLevel.INFO if success else LogLevel.ERROR
        status = "OK" if success else "FAILED"

        event = self._create_event(
            EventType.RECOVERY_ACTION,
            level,
            f"Recovery [{index+1}/{total}]: {action.name} - {status}",
            device_name=action.device,
            payload={
                "action_name": action.name,
                "index": index,
                "total": total,
                "success": success,
                "duration_ms": duration_ms
            }
        )
        self._emit(event)

    def log_recovery_completed(
        self,
        decision: Decision,
        success: bool,
        total_duration_ms: float
    ) -> None:
        level = LogLevel.INFO if success else LogLevel.ERROR
        status = "SUCCESS" if success else "FAILED"

        event = self._create_event(
            EventType.RECOVERY_COMPLETED,
            level,
            f"Recovery {status}: {decision.kind} ({total_duration_ms:.0f}ms)",
            payload={
                "decision_kind": decision.kind,
                "success": success,
                "duration_ms": total_duration_ms
            },
            decision_kind=decision.kind
        )
        self._emit(event)

        # End correlation context
        self.end_correlation()

    # ========================================================================
    # Lifecycle Events
    # ========================================================================

    def log_experiment_started(
        self,
        target_temp: float,
        fault_mode: str,
        config: Dict[str, Any] = None
    ) -> None:
        event = self._create_event(
            EventType.EXPERIMENT_STARTED,
            LogLevel.INFO,
            f"Experiment started: target={target_temp}°C, fault_mode={fault_mode}",
            payload={
                "experiment_id": self.experiment_id,
                "target_temp": target_temp,
                "fault_mode": fault_mode,
                "config": config or {}
            }
        )
        self._emit(event)

    def log_experiment_completed(
        self,
        success: bool,
        total_steps: int,
        final_temp: float = None
    ) -> None:
        level = LogLevel.INFO if success else LogLevel.WARNING
        status = "SUCCESS" if success else "INCOMPLETE"

        event = self._create_event(
            EventType.EXPERIMENT_COMPLETED,
            level,
            f"Experiment {status}: {total_steps} steps, final_temp={final_temp}°C",
            payload={
                "success": success,
                "total_steps": total_steps,
                "final_temp": final_temp
            }
        )
        self._emit(event)

    def log_shutdown(self, reason: str, safe: bool) -> None:
        level = LogLevel.INFO if safe else LogLevel.CRITICAL
        event = self._create_event(
            EventType.SHUTDOWN_INITIATED,
            level,
            f"Shutdown: {reason}",
            payload={"reason": reason, "safe": safe}
        )
        self._emit(event)

    def flush(self) -> None:
        for backend in self.backends:
            backend.flush()

    def close(self) -> None:
        for backend in self.backends:
            backend.close()


# ============================================================================
# Decision Trail Analyzer
# ============================================================================

@dataclass
class DecisionTrail:
    """Analysis of a complete decision-recovery sequence."""
    correlation_id: str
    error_type: str
    error_severity: str
    signature_mode: str
    signature_confidence: float
    decision_kind: str
    decision_rationale: str
    recovery_actions: List[str]
    recovery_success: bool
    total_duration_ms: float
    events: List[LogEvent]


class TrailAnalyzer:
    """Analyzes decision trails from logged events."""

    def __init__(self, backend: MemoryBackend):
        self.backend = backend

    def get_trail(self, correlation_id: str) -> Optional[DecisionTrail]:
        """Reconstruct a decision trail from its correlation_id."""
        events = self.backend.get_decision_trail(correlation_id)
        if not events:
            return None

        # Extract key information
        error_event = next(
            (e for e in events if e.event_type == EventType.ERROR_DETECTED),
            None
        )
        signature_event = next(
            (e for e in events if e.event_type == EventType.SIGNATURE_ANALYZED),
            None
        )
        decision_event = next(
            (e for e in events if e.event_type == EventType.DECISION_MADE),
            None
        )
        completion_event = next(
            (e for e in events if e.event_type == EventType.RECOVERY_COMPLETED),
            None
        )
        recovery_events = [
            e for e in events if e.event_type == EventType.RECOVERY_ACTION
        ]

        return DecisionTrail(
            correlation_id=correlation_id,
            error_type=error_event.error_type if error_event else "unknown",
            error_severity=error_event.error_severity if error_event else "unknown",
            signature_mode=signature_event.payload.get("mode", "unknown") if signature_event else "unknown",
            signature_confidence=signature_event.payload.get("confidence", 0) if signature_event else 0,
            decision_kind=decision_event.decision_kind if decision_event else "unknown",
            decision_rationale=decision_event.decision_rationale if decision_event else "",
            recovery_actions=[e.payload.get("action_name", "") for e in recovery_events],
            recovery_success=completion_event.payload.get("success", False) if completion_event else False,
            total_duration_ms=completion_event.payload.get("duration_ms", 0) if completion_event else 0,
            events=events
        )

    def get_all_trails(self) -> List[DecisionTrail]:
        """Get all decision trails in the log."""
        # Find unique correlation IDs from error events
        error_events = self.backend.query(event_type=EventType.ERROR_DETECTED)
        trails = []
        for event in error_events:
            if event.correlation_id:
                trail = self.get_trail(event.correlation_id)
                if trail:
                    trails.append(trail)
        return trails

    def summarize_decisions(self) -> Dict[str, Any]:
        """Generate summary statistics of all decisions."""
        trails = self.get_all_trails()

        if not trails:
            return {"total_decisions": 0}

        return {
            "total_decisions": len(trails),
            "by_kind": self._count_by(trails, lambda t: t.decision_kind),
            "by_error_type": self._count_by(trails, lambda t: t.error_type),
            "by_signature": self._count_by(trails, lambda t: t.signature_mode),
            "success_rate": sum(1 for t in trails if t.recovery_success) / len(trails),
            "avg_recovery_duration_ms": sum(t.total_duration_ms for t in trails) / len(trails)
        }

    def _count_by(self, trails: List[DecisionTrail], key_fn: Callable) -> Dict[str, int]:
        counts = {}
        for trail in trails:
            k = key_fn(trail)
            counts[k] = counts.get(k, 0) + 1
        return counts
