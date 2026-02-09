"""
PhaseResult - Encapsulates the result of a workflow phase execution

This class allows phases (especially prep phases) to report failures
without raising exceptions, enabling graceful error recovery and loop continuation.

Enhanced for parallel thread execution with structured failure reporting.
"""
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class ThreadFailure:
    """
    Structured information about a thread failure in parallel execution

    Enables intelligent recovery decisions based on failure type and context.
    """

    thread_name: str
    exception: Exception
    traceback: str
    failure_class: str  # OT2_ALARM, INPUT_DATA, AUX_STREAM, NETWORK, UNKNOWN
    recoverable: bool  # Policy decision - can this be recovered from?
    suggested_recovery: str  # e.g., "SKIP_LOOP_AND_HOME", "DEGRADED_COMPLETE"
    step_id: Optional[str] = None  # Step where failure occurred

    def __str__(self) -> str:
        return (f"ThreadFailure(thread={self.thread_name}, "
               f"class={self.failure_class}, "
               f"recoverable={self.recoverable}, "
               f"exception={type(self.exception).__name__})")


@dataclass
class PhaseResult:
    """
    Result of a workflow phase execution

    Enhanced to support parallel thread execution with detailed failure tracking.
    """

    phase_name: str
    success: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    error_details: Dict[str, Any] = field(default_factory=dict)
    skipped: bool = False

    # Parallel thread execution results
    thread_failures: List[ThreadFailure] = field(default_factory=list)
    cancelled: bool = False  # Was execution cancelled?
    timed_out: bool = False  # Did execution timeout?

    def __str__(self) -> str:
        """String representation of phase result"""
        if self.success:
            return f"PhaseResult(phase={self.phase_name}, success=True)"
        elif self.skipped:
            return f"PhaseResult(phase={self.phase_name}, skipped=True)"
        elif self.thread_failures:
            return (f"PhaseResult(phase={self.phase_name}, success=False, "
                   f"thread_failures={len(self.thread_failures)}, "
                   f"cancelled={self.cancelled}, timed_out={self.timed_out})")
        else:
            return (f"PhaseResult(phase={self.phase_name}, success=False, "
                   f"error_type={self.error_type}, error={self.error_message})")

    @classmethod
    def success_result(cls, phase_name: str) -> 'PhaseResult':
        """Create a successful phase result"""
        return cls(phase_name=phase_name, success=True)

    @classmethod
    def failure_result(cls, phase_name: str, error: Exception,
                      error_details: Optional[Dict[str, Any]] = None) -> 'PhaseResult':
        """Create a failed phase result from an exception"""
        return cls(
            phase_name=phase_name,
            success=False,
            error_type=type(error).__name__,
            error_message=str(error),
            error_details=error_details or {}
        )

    @classmethod
    def skipped_result(cls, phase_name: str, reason: str) -> 'PhaseResult':
        """Create a skipped phase result"""
        return cls(
            phase_name=phase_name,
            success=False,
            skipped=True,
            error_message=reason
        )

    @classmethod
    def parallel_failure_result(cls, phase_name: str,
                               thread_failures: List[ThreadFailure],
                               cancelled: bool = False,
                               timed_out: bool = False) -> 'PhaseResult':
        """Create a failed phase result from parallel thread failures"""
        # Aggregate error information
        error_types = set(f.failure_class for f in thread_failures)
        thread_names = [f.thread_name for f in thread_failures]

        return cls(
            phase_name=phase_name,
            success=False,
            error_type=f"ParallelFailure({','.join(error_types)})",
            error_message=f"Thread(s) failed: {', '.join(thread_names)}",
            error_details={'thread_count': len(thread_failures)},
            thread_failures=thread_failures,
            cancelled=cancelled,
            timed_out=timed_out
        )
