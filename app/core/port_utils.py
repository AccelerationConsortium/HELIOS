"""Cross-platform serial port detection utilities.

Supports Linux, macOS, Windows, and Docker/WSL2 environments.
When 'auto' is specified, attempts to find an appropriate port
for the given device type.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def detect_serial_port(
    device_hint: str = "usb",
    fallback: str = "",
) -> str:
    """Auto-detect a serial port for the given device type.

    Args:
        device_hint: Type hint — 'usb', 'serial', 'relay', 'potentiostat'.
        fallback: Value to return if no port is found.

    Returns:
        Port path string, or *fallback* if nothing found.
    """
    try:
        from serial.tools.list_ports import comports
        ports = list(comports())
        if not ports:
            logger.debug("No serial ports detected.")
            return fallback
        # 优先返回第一个匹配 hint 的端口
        hint_lower = device_hint.lower()
        for p in ports:
            desc = (p.description or "").lower()
            if hint_lower in desc or hint_lower in (p.device or "").lower():
                logger.info("Auto-detected port %s for '%s'", p.device, device_hint)
                return p.device
        # 没有精确匹配就返回第一个
        first = ports[0].device
        logger.info("No exact match for '%s', using first port: %s", device_hint, first)
        return first
    except ImportError:
        logger.debug("pyserial not installed — cannot auto-detect ports.")
        return fallback
    except Exception as exc:  # noqa: BLE001
        logger.warning("Port detection failed: %s", exc)
        return fallback


def default_port(comm_type: str) -> str:
    """Return a sensible default port string based on OS.

    Args:
        comm_type: One of 'usb', 'serial', 'relay', 'potentiostat'.

    Returns:
        Platform-appropriate default port path.
    """
    if sys.platform == "win32":
        return "COM3"
    if sys.platform == "darwin":
        # macOS: USB-serial adapters appear as /dev/cu.usbserial-*
        return "/dev/cu.usbserial-0001"
    # Linux / Docker / WSL2
    if comm_type in ("usb", "relay", "potentiostat"):
        return "/dev/ttyUSB0"
    return "/dev/ttyS0"


def resolve_port(configured: str, comm_type: str = "usb") -> str:
    """Resolve a port value from configuration.

    Handles the special value 'auto' by attempting detection,
    falling back to platform defaults.

    Args:
        configured: Value from env / config (may be 'auto', '', or a real path).
        comm_type: Device type hint for auto-detection.

    Returns:
        Resolved port string.
    """
    if not configured or configured.lower() == "auto":
        detected = detect_serial_port(device_hint=comm_type)
        if detected:
            return detected
        return default_port(comm_type)
    return configured
