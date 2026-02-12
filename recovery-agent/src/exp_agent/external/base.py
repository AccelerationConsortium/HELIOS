"""
Base classes for external execution.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import time


class ExecutionStatus(str, Enum):
    """Status of an external execution."""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ExecutionResult:
    """Result of an external execution.

    Attributes:
        status: Execution status
        output: Standard output or response body
        error: Error message if failed
        exit_code: Process exit code (for CLI/script)
        duration_ms: Execution duration in milliseconds
        metadata: Additional execution metadata
    """
    status: ExecutionStatus
    output: Any = None
    error: Optional[str] = None
    exit_code: Optional[int] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == ExecutionStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "exit_code": self.exit_code,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


class ExecutionError(Exception):
    """Error during external execution."""

    def __init__(
        self,
        message: str,
        result: Optional[ExecutionResult] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.result = result
        self.cause = cause


@dataclass
class BaseAction:
    """Base class for external actions."""
    name: str
    timeout_seconds: float = 30.0
    retries: int = 0
    retry_delay_seconds: float = 1.0

    def validate(self) -> None:
        """Validate action parameters. Override in subclasses."""
        if not self.name:
            raise ValueError("Action name is required")
        if self.timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")
        if self.retries < 0:
            raise ValueError("Retries must be non-negative")


class ExternalExecutor(ABC):
    """Base class for external executors.

    Provides common functionality:
    - Retry logic with exponential backoff
    - Timeout handling
    - Result normalization
    - Error handling
    """

    def __init__(self, name: str = "executor"):
        self.name = name
        self._execution_count = 0

    @abstractmethod
    async def _execute_impl(self, action: BaseAction) -> ExecutionResult:
        """Implementation-specific execution logic."""
        pass

    async def execute(self, action: BaseAction) -> ExecutionResult:
        """Execute an action with retry logic.

        Args:
            action: The action to execute

        Returns:
            ExecutionResult with status and output

        Raises:
            ExecutionError: If all retries fail
        """
        action.validate()

        last_error: Optional[Exception] = None
        last_result: Optional[ExecutionResult] = None

        for attempt in range(action.retries + 1):
            self._execution_count += 1

            try:
                start_time = time.time()
                result = await self._execute_impl(action)
                result.duration_ms = (time.time() - start_time) * 1000
                result.metadata["attempt"] = attempt + 1
                result.metadata["executor"] = self.name

                if result.success:
                    return result

                last_result = result
                last_error = ExecutionError(
                    f"Execution failed: {result.error}",
                    result=result
                )

            except Exception as e:
                last_error = e
                last_result = ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    error=str(e),
                    metadata={"attempt": attempt + 1, "executor": self.name}
                )

            # Wait before retry (with exponential backoff)
            if attempt < action.retries:
                delay = action.retry_delay_seconds * (2 ** attempt)
                await self._sleep(delay)

        # All retries exhausted
        raise ExecutionError(
            f"All {action.retries + 1} attempts failed for {action.name}",
            result=last_result,
            cause=last_error if isinstance(last_error, Exception) else None
        )

    async def _sleep(self, seconds: float) -> None:
        """Sleep for retry delay. Can be overridden for testing."""
        import asyncio
        await asyncio.sleep(seconds)

    @property
    def execution_count(self) -> int:
        """Total number of execution attempts."""
        return self._execution_count
