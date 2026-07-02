from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.services.telegram_bot import TelegramBotService


class FakeTelegramClient:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def post(self, path: str, json: dict[str, Any]) -> httpx.Response:
        self.sent.append({"path": path, "json": json})
        return httpx.Response(200, json={"ok": True})


@pytest.fixture
def isolated_db(monkeypatch, tmp_path):
    from app.core.config import get_settings
    from app.core.db import init_db

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "data" / "orchestrator.db"))
    monkeypatch.setenv("OBJECT_STORE_DIR", str(tmp_path / "objects"))
    get_settings.cache_clear()
    init_db()


def _update(chat_id: int, text: str) -> dict[str, Any]:
    return {
        "update_id": 1,
        "message": {
            "chat": {"id": chat_id},
            "text": text,
        },
    }


async def test_whoami_works_without_allowlist():
    client = FakeTelegramClient()
    bot = TelegramBotService(
        token="token",
        allowed_chat_ids=set(),
        client=client,  # type: ignore[arg-type]
    )

    await bot.handle_update(_update(1234, "/whoami"))

    assert client.sent[-1]["json"]["text"] == "chat_id: 1234"


async def test_run_command_requires_allowed_chat():
    client = FakeTelegramClient()
    bot = TelegramBotService(
        token="token",
        allowed_chat_ids={999},
        client=client,  # type: ignore[arg-type]
    )

    await bot.handle_update(_update(1234, '/run {"steps":[]}'))

    assert "not allowed" in client.sent[-1]["json"]["text"]


async def test_run_command_creates_helios_run(isolated_db):
    client = FakeTelegramClient()
    bot = TelegramBotService(
        token="token",
        allowed_chat_ids={1234},
        default_instrument_id="telegram-instrument",
        client=client,  # type: ignore[arg-type]
    )
    payload = {
        "steps": [
            {
                "step_key": "s1",
                "primitive": "log",
                "params": {"message": "hello"},
            }
        ]
    }

    await bot.handle_update(_update(1234, f"/run {json.dumps(payload)}"))

    from app.services.run_service import list_runs

    runs = list_runs()
    assert len(runs) == 1
    assert runs[0]["trigger_type"] == "telegram"
    assert runs[0]["inputs"]["instrument_id"] == "telegram-instrument"
    assert bot._chat_last_run[1234] == runs[0]["id"]
    assert "Run created" in client.sent[-1]["json"]["text"]


async def test_run_command_accepts_bot_mention_and_keeps_inputs_out_of_protocol(isolated_db):
    client = FakeTelegramClient()
    bot = TelegramBotService(
        token="token",
        allowed_chat_ids={1234},
        client=client,  # type: ignore[arg-type]
    )
    payload = {
        "steps": [{"step_key": "s1", "primitive": "log", "params": {}}],
        "inputs": {"instrument_id": "mapped-instrument"},
    }

    await bot.handle_update(_update(1234, f"/run@helios_bot {json.dumps(payload)}"))

    from app.services.run_service import list_runs

    runs = list_runs()
    assert len(runs) == 1
    assert runs[0]["protocol"] == {"steps": payload["steps"]}
    assert runs[0]["inputs"]["instrument_id"] == "mapped-instrument"


async def test_status_uses_last_run_for_chat(isolated_db):
    client = FakeTelegramClient()
    bot = TelegramBotService(
        token="token",
        allowed_chat_ids={1234},
        client=client,  # type: ignore[arg-type]
    )

    await bot.handle_update(_update(1234, '/run {"steps":[]}'))
    await bot.handle_update(_update(1234, "/status"))

    assert "run_id:" in client.sent[-1]["json"]["text"]
    assert "status:" in client.sent[-1]["json"]["text"]
