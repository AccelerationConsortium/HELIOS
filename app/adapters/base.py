"""Abstract adapter interface for instrument execution."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class InstrumentAdapter(Protocol):
    """Protocol that every instrument adapter must implement."""

    def execute_primitive(
        self, *, instrument_id: str, primitive: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a single primitive action on the instrument.

        Args:
            instrument_id: Logical instrument identifier.
            primitive: Action name (e.g. "robot.aspirate", "wait").
            params: Action-specific parameters.

        Returns:
            Result dict with at least {"ok": bool}.
        """
        ...

    def connect(self) -> None:
        """Establish connections to hardware (or validate dry-run mode)."""
        ...

    def disconnect(self) -> None:
        """Release all hardware connections gracefully."""
        ...

    def health_check(self) -> dict[str, Any]:
        """Return health status of connected hardware."""
        ...
