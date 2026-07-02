from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from app.adapters.puda_execution_backend import PudaCommandError, PudaExecutionBackend


@dataclass
class FakeCommandRequest:
    name: str
    machine_id: str
    params: dict[str, Any]
    step_number: int


class FakeStatus:
    def __init__(self, value: str) -> None:
        self.value = value


@dataclass
class FakeResponse:
    status: FakeStatus
    completed_at: str = "2026-06-22T00:00:00Z"
    code: str | None = None
    message: str | None = None
    data: dict[str, Any] | None = None


@dataclass
class FakeMessage:
    response: FakeResponse


class FakeCommandService:
    last_instance: FakeCommandService | None = None

    def __init__(self, servers: list[str]) -> None:
        self.servers = servers
        self.connected = False
        self.disconnected = False
        self.sent: list[dict[str, Any]] = []
        self.next_message: FakeMessage | None = FakeMessage(
            response=FakeResponse(status=FakeStatus("success"), data={"voltage_v": 1.23})
        )
        FakeCommandService.last_instance = self

    async def connect(self) -> bool:
        self.connected = True
        return True

    async def disconnect(self) -> None:
        self.disconnected = True

    async def send_queue_command(
        self,
        *,
        request: FakeCommandRequest,
        run_id: str,
        user_id: str,
        username: str,
        timeout: int,
    ) -> FakeMessage | None:
        self.sent.append(
            {
                "request": request,
                "run_id": run_id,
                "user_id": user_id,
                "username": username,
                "timeout": timeout,
            }
        )
        return self.next_message


def _backend() -> PudaExecutionBackend:
    return PudaExecutionBackend(
        servers=["nats://test:4222"],
        user_id="helios-user",
        username="HELIOS Test",
        default_machine_id="fallback-machine",
        machine_map={"robot": "ot2-a", "squidstat.run_experiment": "squidstat-a"},
        timeout_seconds=7,
        command_service_cls=FakeCommandService,
        command_request_cls=FakeCommandRequest,
    )


def test_execute_step_translates_compiled_step_to_puda_command_request():
    backend = _backend()
    backend.connect()

    result = backend.execute_step(
        run_id="adhoc",
        instrument_id="sim-instrument-1",
        step={
            "step_key": "s1",
            "primitive": "robot.aspirate",
            "params": {"volume_ul": 50},
            "resources": ["deck-slot-1"],
            "_step_number": 3,
        },
    )

    service = FakeCommandService.last_instance
    assert service is not None
    sent = service.sent[0]
    request = sent["request"]
    assert request == FakeCommandRequest(
        name="robot.aspirate",
        machine_id="ot2-a",
        params={"volume_ul": 50},
        step_number=3,
    )
    assert sent["run_id"] == "adhoc"
    assert sent["user_id"] == "helios-user"
    assert sent["username"] == "HELIOS Test"
    assert sent["timeout"] == 7
    assert result["ok"] is True
    assert result["backend"] == "puda"
    assert result["observation"] == {"voltage_v": 1.23}
    assert result["puda"]["status"] == "success"


def test_execute_step_raises_command_error_for_puda_error_response():
    backend = _backend()
    backend.connect()
    service = FakeCommandService.last_instance
    assert service is not None
    service.next_message = FakeMessage(
        response=FakeResponse(
            status=FakeStatus("error"),
            code="EXECUTION_ERROR",
            message="pump failed",
            data={"axis": "pump-1"},
        )
    )

    with pytest.raises(PudaCommandError) as exc_info:
        backend.execute_step(
            run_id="adhoc",
            instrument_id="sim-instrument-1",
            step={
                "step_key": "s2",
                "primitive": "plc.dispense_ml",
                "params": {"volume_ml": 1},
                "resources": ["plc-a"],
                "_step_number": 4,
            },
        )

    assert exc_info.value.code == "EXECUTION_ERROR"
    assert exc_info.value.data == {"axis": "pump-1"}


def test_resolve_machine_prefers_exact_primitive_then_prefix_then_resource():
    backend = _backend()

    assert backend.resolve_machine(["robot-b"], "squidstat.run_experiment") == "squidstat-a"
    assert backend.resolve_machine(["robot-b"], "robot.dispense") == "ot2-a"
    assert backend.resolve_machine(["relay-a"], "relay.turn_on") == "relay-a"
    assert backend.resolve_machine([], "custom.unknown", "instrument-a") == "instrument-a"
