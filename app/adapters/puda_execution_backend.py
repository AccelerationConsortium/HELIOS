"""PUDA execution backend for HELIOS worker steps."""
from __future__ import annotations

import asyncio
import importlib
from typing import Any

from app.core.config import get_settings
from app.core.db import run_txn
from app.services.audit import record_event


class PudaCommandError(RuntimeError):
    """Raised when PUDA reports a command-level failure."""

    def __init__(self, *, code: str | None, message: str | None, data: dict[str, Any] | None) -> None:
        self.code = code
        self.message = message or "PUDA command failed"
        self.data = data or {}
        detail = self.message if code is None else f"{code}: {self.message}"
        super().__init__(detail)


class PudaExecutionBackend:
    """Translate compiled HELIOS steps into PUDA CommandRequest messages."""

    def __init__(
        self,
        *,
        servers: list[str] | None = None,
        user_id: str | None = None,
        username: str | None = None,
        default_machine_id: str | None = None,
        machine_map: dict[str, str] | None = None,
        timeout_seconds: int | None = None,
        command_service_cls: type[Any] | None = None,
        command_request_cls: type[Any] | None = None,
    ) -> None:
        settings = get_settings()
        self.servers = servers or settings.puda_nats_servers
        self.user_id = user_id or settings.puda_user_id
        self.username = username or settings.puda_username
        self.default_machine_id = default_machine_id or settings.puda_default_machine_id
        self.machine_map = dict(machine_map if machine_map is not None else settings.puda_machine_map)
        self.timeout_seconds = timeout_seconds or settings.puda_command_timeout_seconds
        self._command_service_cls = command_service_cls
        self._command_request_cls = command_request_cls
        self._service: Any | None = None
        self._connected = False

    def connect(self) -> None:
        self._load_puda_types()
        self._service = self._command_service_cls(self.servers)
        connected = asyncio.run(self._service.connect())
        if not connected:
            raise RuntimeError(f"failed to connect to PUDA NATS servers: {self.servers}")
        self._connected = True

    def disconnect(self) -> None:
        if self._service is not None and self._connected:
            asyncio.run(self._service.disconnect())
        self._connected = False
        self._service = None

    def health_check(self) -> dict[str, Any]:
        return {
            "adapter": "puda",
            "connected": self._connected,
            "servers": self.servers,
        }

    def execute_primitive(
        self, *, instrument_id: str, primitive: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        step = {
            "step_key": "adhoc",
            "primitive": primitive,
            "params": params,
            "resources": [instrument_id],
            "_step_number": 0,
        }
        return self.execute_step(run_id="adhoc", step=step, instrument_id=instrument_id)

    def execute_step(
        self,
        *,
        run_id: str,
        step: dict[str, Any],
        instrument_id: str,
    ) -> dict[str, Any]:
        if self._service is None or not self._connected:
            raise RuntimeError("PUDA backend is not connected")

        primitive = str(step["primitive"])
        params = step.get("params") or {}
        resources = [str(r) for r in step.get("resources", [])]
        step_number = int(step.get("_step_number", step.get("step_number", 0)))
        machine_id = self.resolve_machine(resources, primitive, instrument_id)
        request = self._command_request_cls(
            name=primitive,
            machine_id=machine_id,
            params=params,
            step_number=step_number,
        )

        self._record_puda_event(
            run_id=run_id,
            action="puda.command_request.created",
            details={
                "step_key": step.get("step_key"),
                "primitive": primitive,
                "machine_id": machine_id,
                "resources": resources,
                "step_number": step_number,
            },
        )

        message = asyncio.run(
            self._service.send_queue_command(
                request=request,
                run_id=run_id,
                user_id=self.user_id,
                username=self.username,
                timeout=self.timeout_seconds,
            )
        )
        if message is None:
            raise TimeoutError(
                f"PUDA command timed out after {self.timeout_seconds}s: "
                f"{primitive} step_number={step_number} machine_id={machine_id}"
            )

        response = getattr(message, "response", None)
        if response is None:
            raise RuntimeError("PUDA response message did not include response payload")

        raw_status = getattr(response, "status", "")
        response_status = str(getattr(raw_status, "value", raw_status)).lower()
        response_data = getattr(response, "data", None) or {}
        response_code = getattr(response, "code", None)
        code_value = getattr(response_code, "value", response_code)
        message_text = getattr(response, "message", None)
        completed_at = getattr(response, "completed_at", None)

        self._record_puda_event(
            run_id=run_id,
            action="puda.command_response.received",
            details={
                "step_key": step.get("step_key"),
                "primitive": primitive,
                "machine_id": machine_id,
                "step_number": step_number,
                "status": response_status,
                "code": code_value,
                "message": message_text,
                "completed_at": completed_at,
            },
        )

        if response_status.endswith("error"):
            raise PudaCommandError(code=code_value, message=message_text, data=response_data)

        return {
            "ok": True,
            "backend": "puda",
            "instrument_id": instrument_id,
            "machine_id": machine_id,
            "primitive": primitive,
            "step_key": step.get("step_key"),
            "step_number": step_number,
            "completed_at": completed_at,
            "observation": response_data,
            "puda": {
                "status": response_status,
                "code": code_value,
                "message": message_text,
                "data": response_data,
            },
        }

    def resolve_machine(
        self,
        resources: list[str],
        primitive: str,
        instrument_id: str | None = None,
    ) -> str:
        if primitive in self.machine_map:
            return self.machine_map[primitive]

        primitive_prefix = primitive.split(".", 1)[0]
        if primitive_prefix in self.machine_map:
            return self.machine_map[primitive_prefix]

        for resource in resources:
            if resource in self.machine_map:
                return self.machine_map[resource]

        if resources:
            return resources[0]
        return instrument_id or self.default_machine_id

    def _load_puda_types(self) -> None:
        if self._command_service_cls is not None and self._command_request_cls is not None:
            return
        try:
            command_service = importlib.import_module("puda.command_service")
            models = importlib.import_module("puda.models")
        except ImportError as exc:
            raise RuntimeError(
                "EXECUTION_BACKEND=puda requires the PUDA Python SDK to be installed "
                "and importable as 'puda'."
            ) from exc

        self._command_service_cls = self._command_service_cls or command_service.CommandService
        self._command_request_cls = self._command_request_cls or models.CommandRequest

    @staticmethod
    def _record_puda_event(*, run_id: str, action: str, details: dict[str, Any]) -> None:
        def _txn(conn):
            record_event(conn, run_id=run_id, actor="puda-backend", action=action, details=details)

        if run_id != "adhoc":
            run_txn(_txn)
