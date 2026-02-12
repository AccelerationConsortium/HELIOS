"""
Real hardware device implementations for the Experiment Agent.

This module provides concrete implementations for actual laboratory equipment,
replacing the simulated versions with real hardware communication.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import time

from ..base import Device
from ...core.types import DeviceState, Action, HardwareError


class CommunicationInterface(ABC):
    """Abstract interface for device communication protocols."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to device."""
        pass

    @abstractmethod
    def disconnect(self):
        """Close connection to device."""
        pass

    @abstractmethod
    def send_command(
        self, command: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Send command to device and get response."""
        pass

    @abstractmethod
    def read_data(self) -> Dict[str, Any]:
        """Read current device state/data."""
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if device is connected."""
        pass


class SerialCommunication(CommunicationInterface):
    """Serial port communication interface."""

    def __init__(self, port: str, baudrate: int = 9600, timeout: float = 1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._serial = None

    def connect(self) -> bool:
        try:
            import serial

            self._serial = serial.Serial(
                port=self.port, baudrate=self.baudrate, timeout=self.timeout
            )
            return True
        except ImportError:
            raise HardwareError(
                device="serial_interface",
                type="missing_dependency",
                severity="high",
                message="pyserial not installed",
            )
        except Exception as e:
            raise HardwareError(
                device="serial_interface",
                type="connection_failed",
                severity="high",
                message=f"Serial connection failed: {e}",
            )

    def disconnect(self):
        if self._serial:
            self._serial.close()

    def send_command(
        self, command: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self._serial:
            raise HardwareError(
                device="serial_interface",
                type="not_connected",
                severity="high",
                message="Serial port not connected",
            )

        # Format command (device-specific protocol)
        cmd_str = f"{command}"
        if params:
            param_str = ",".join(f"{k}={v}" for k, v in params.items())
            cmd_str += f" {param_str}"

        cmd_str += "\n"

        try:
            self._serial.write(cmd_str.encode())
            response = self._serial.readline().decode().strip()

            # Parse response (simplified - device specific)
            if response.startswith("OK"):
                return {"status": "success", "data": response[3:]}
            elif response.startswith("ERROR"):
                raise HardwareError(
                    device="serial_device",
                    type="command_failed",
                    severity="medium",
                    message=f"Device command failed: {response}",
                )
            else:
                return {"status": "data", "data": response}

        except Exception as e:
            raise HardwareError(
                device="serial_interface",
                type="communication_error",
                severity="high",
                message=f"Serial communication error: {e}",
            )

    def read_data(self) -> Dict[str, Any]:
        return self.send_command("READ")

    @property
    def is_connected(self) -> bool:
        return self._serial is not None and self._serial.is_open


class NetworkCommunication(CommunicationInterface):
    """Network/TCP communication interface."""

    def __init__(self, host: str, port: int, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._socket = None

    def connect(self) -> bool:
        try:
            import socket

            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self.timeout)
            self._socket.connect((self.host, self.port))
            return True
        except Exception as e:
            raise HardwareError(
                device="network_interface",
                type="connection_failed",
                severity="high",
                message=f"Network connection failed: {e}",
            )

    def disconnect(self):
        if self._socket:
            self._socket.close()

    def send_command(
        self, command: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self._socket:
            raise HardwareError(
                device="network_interface",
                type="not_connected",
                severity="high",
                message="Network not connected",
            )

        # Format command as JSON
        import json

        message = {"command": command}
        if params:
            message["params"] = params

        try:
            data = json.dumps(message).encode() + b"\n"
            self._socket.sendall(data)

            response_data = self._socket.recv(1024)
            response = json.loads(response_data.decode())

            if response.get("status") == "error":
                raise HardwareError(
                    device="network_device",
                    type="command_failed",
                    severity="medium",
                    message=f"Device command failed: {response.get('message', 'Unknown error')}",
                )

            return response

        except json.JSONDecodeError:
            raise HardwareError(
                device="network_interface",
                type="protocol_error",
                severity="high",
                message="Invalid JSON response from device",
            )
        except Exception as e:
            raise HardwareError(
                device="network_interface",
                type="communication_error",
                severity="high",
                message=f"Network communication error: {e}",
            )

    def read_data(self) -> Dict[str, Any]:
        return self.send_command("read_state")

    @property
    def is_connected(self) -> bool:
        return self._socket is not None


class RealDevice(Device):
    """Base class for real hardware devices."""

    def __init__(self, name: str, comm_interface: CommunicationInterface):
        super().__init__(name)
        self.comm = comm_interface
        self._connected = False
        self._last_health_check = 0
        self._health_check_interval = 30  # seconds

    def connect(self) -> bool:
        """Establish connection to the device."""
        if not self._connected:
            self._connected = self.comm.connect()
        return self._connected

    def disconnect(self):
        """Disconnect from the device."""
        self.comm.disconnect()
        self._connected = False

    def _ensure_connected(self):
        """Ensure device is connected before operations."""
        if not self._connected:
            if not self.connect():
                raise HardwareError(
                    device=self.name,
                    type="connection_failed",
                    severity="high",
                    message=f"Failed to connect to device {self.name}",
                )

    def health(self) -> bool:
        """Check device health with caching."""
        current_time = time.time()
        if current_time - self._last_health_check > self._health_check_interval:
            try:
                self._ensure_connected()
                # Send health check command
                response = self.comm.send_command("health_check")
                self._last_health_check = current_time
                return response.get("status") == "healthy"
            except Exception:
                return False
        return True

    def read_state(self) -> DeviceState:
        """Read device state."""
        self._ensure_connected()
        try:
            data = self.comm.read_data()
            return self._parse_device_state(data)
        except HardwareError:
            raise
        except Exception as e:
            raise HardwareError(
                device=self.name,
                type="read_error",
                severity="high",
                message=f"Failed to read device state: {e}",
            )

    @abstractmethod
    def _parse_device_state(self, raw_data: Dict[str, Any]) -> DeviceState:
        """Parse raw device data into DeviceState."""
        pass
