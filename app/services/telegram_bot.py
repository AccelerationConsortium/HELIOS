"""Telegram control-plane entrypoint for running HELIOS without the web UI."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from app.core.config import get_settings
from app.core.constants import TERMINAL_RUN_STATUSES
from app.services.run_service import (
    DomainError,
    create_run_from_trigger,
    get_run,
    list_events,
)

logger = logging.getLogger(__name__)


HELP_TEXT = """HELIOS Telegram commands:
/whoami - show your Telegram chat id
/run <json> - create a HELIOS run from protocol JSON
/status [run_id] - show run and step status
/events [run_id] - show recent run events

/run accepts either {"steps":[...]} or {"protocol":{"steps":[...]},"inputs":{...}}.
Commands that touch runs require TELEGRAM_ALLOWED_CHAT_IDS."""


class TelegramBotService:
    """Small Telegram long-polling adapter over the existing run service."""

    def __init__(
        self,
        *,
        token: str,
        allowed_chat_ids: set[int],
        poll_timeout_seconds: int = 25,
        default_instrument_id: str = "sim-instrument-1",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.token = token
        self.allowed_chat_ids = allowed_chat_ids
        self.poll_timeout_seconds = poll_timeout_seconds
        self.default_instrument_id = default_instrument_id
        self._client = client
        self._owns_client = client is None
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._offset = 0
        self._chat_last_run: dict[int, str] = {}

    async def start(self) -> None:
        if self._task is not None:
            return
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._base_url(), timeout=self.poll_timeout_seconds + 10)
        self._task = asyncio.create_task(self._poll_loop(), name="telegram-bot")
        logger.info("Telegram control-plane started")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._client is not None and self._owns_client:
            await self._client.aclose()
        self._client = None
        logger.info("Telegram control-plane stopped")

    async def handle_update(self, update: dict[str, Any]) -> None:
        message = update.get("message") or update.get("edited_message") or {}
        text = str(message.get("text") or "").strip()
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        if not text or not isinstance(chat_id, int):
            return

        command, _, command_args = text.partition(" ")
        base_command = command.split("@", 1)[0]

        if base_command == "/whoami":
            await self._send_message(chat_id, f"chat_id: {chat_id}")
            return

        if base_command in {"/help", "/start"}:
            await self._send_message(chat_id, HELP_TEXT)
            return

        if not self._is_allowed(chat_id):
            await self._send_message(
                chat_id,
                "This chat is not allowed to control HELIOS. "
                f"Set TELEGRAM_ALLOWED_CHAT_IDS={chat_id} on the HELIOS host.",
            )
            return

        try:
            if base_command == "/run":
                await self._handle_run(chat_id, command_args.strip())
            elif text.startswith("{"):
                await self._handle_run(chat_id, text)
            elif base_command == "/status":
                await self._handle_status(chat_id, command_args.strip())
            elif base_command == "/events":
                await self._handle_events(chat_id, command_args.strip())
            else:
                await self._send_message(chat_id, HELP_TEXT)
        except DomainError as exc:
            await self._send_message(chat_id, f"HELIOS rejected the request: {exc}")
        except json.JSONDecodeError as exc:
            await self._send_message(chat_id, f"Invalid JSON: {exc.msg}")
        except Exception:
            logger.exception("Telegram command failed")
            await self._send_message(chat_id, "Command failed; check HELIOS logs for details.")

    async def _handle_run(self, chat_id: int, raw_payload: str) -> None:
        if not raw_payload:
            await self._send_message(chat_id, "Usage: /run {\"steps\":[...]}")
            return

        payload = json.loads(_strip_json_fence(raw_payload))
        if not isinstance(payload, dict):
            raise DomainError("/run payload must be a JSON object")

        if "protocol" in payload:
            protocol = payload.get("protocol")
        elif "steps" in payload:
            protocol = {"steps": payload.get("steps")}
        else:
            protocol = payload
        if not isinstance(protocol, dict):
            raise DomainError("protocol must be a JSON object")

        inputs = payload.get("inputs") or {}
        if not isinstance(inputs, dict):
            raise DomainError("inputs must be a JSON object")
        inputs = {"instrument_id": self.default_instrument_id, **inputs}

        run = await asyncio.to_thread(
            create_run_from_trigger,
            trigger_type="telegram",
            trigger_payload={"chat_id": chat_id},
            campaign_id=payload.get("campaign_id"),
            protocol=protocol,
            inputs=inputs,
            policy_snapshot=payload.get("policy_snapshot"),
            actor="telegram",
            session_key=str(chat_id),
        )
        self._chat_last_run[chat_id] = run["id"]
        await self._send_message(
            chat_id,
            _format_run_created(run),
        )

    async def _handle_status(self, chat_id: int, raw_run_id: str) -> None:
        run_id = raw_run_id.strip() or self._chat_last_run.get(chat_id)
        if not run_id:
            await self._send_message(chat_id, "No run_id provided and no recent Telegram run in memory.")
            return

        run = await asyncio.to_thread(get_run, run_id)
        if run is None:
            await self._send_message(chat_id, f"Run not found: {run_id}")
            return
        await self._send_message(chat_id, _format_run_status(run))

    async def _handle_events(self, chat_id: int, raw_run_id: str) -> None:
        run_id = raw_run_id.strip() or self._chat_last_run.get(chat_id)
        if not run_id:
            await self._send_message(chat_id, "No run_id provided and no recent Telegram run in memory.")
            return

        events = await asyncio.to_thread(list_events, run_id)
        if not events:
            await self._send_message(chat_id, f"No events found for run {run_id}.")
            return
        await self._send_message(chat_id, _format_recent_events(run_id, events[-8:]))

    async def _poll_loop(self) -> None:
        assert self._client is not None
        while not self._stop_event.is_set():
            try:
                response = await self._client.get(
                    "/getUpdates",
                    params={
                        "offset": self._offset,
                        "timeout": self.poll_timeout_seconds,
                        "allowed_updates": json.dumps(["message", "edited_message"]),
                    },
                )
                response.raise_for_status()
                body = response.json()
                for update in body.get("result", []):
                    update_id = update.get("update_id")
                    if isinstance(update_id, int):
                        self._offset = max(self._offset, update_id + 1)
                    await self.handle_update(update)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Telegram polling failed")
                await asyncio.sleep(5)

    async def _send_message(self, chat_id: int, text: str) -> None:
        assert self._client is not None
        await self._client.post(
            "/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text[:3900],
                "disable_web_page_preview": True,
            },
        )

    def _base_url(self) -> str:
        return f"https://api.telegram.org/bot{self.token}"

    def _is_allowed(self, chat_id: int) -> bool:
        return chat_id in self.allowed_chat_ids


_service: TelegramBotService | None = None


async def start_telegram_bot() -> TelegramBotService | None:
    global _service
    settings = get_settings()
    if not settings.telegram_bot_token:
        return None
    if _service is not None:
        return _service

    _service = TelegramBotService(
        token=settings.telegram_bot_token,
        allowed_chat_ids=settings.telegram_allowed_chat_ids,
        poll_timeout_seconds=settings.telegram_poll_timeout_seconds,
        default_instrument_id=settings.telegram_default_instrument_id,
    )
    await _service.start()
    return _service


async def stop_telegram_bot() -> None:
    global _service
    if _service is not None:
        await _service.stop()
        _service = None


def _strip_json_fence(raw: str) -> str:
    text = raw.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _format_run_created(run: dict[str, Any]) -> str:
    status = run["status"]
    text = [
        "Run created.",
        f"run_id: {run['id']}",
        f"status: {status}",
        f"steps: {len(run.get('steps', []))}",
    ]
    if status in TERMINAL_RUN_STATUSES and run.get("rejection_reason"):
        text.append(f"reason: {run['rejection_reason']}")
    return "\n".join(text)


def _format_run_status(run: dict[str, Any]) -> str:
    lines = [
        f"run_id: {run['id']}",
        f"status: {run['status']}",
    ]
    if run.get("rejection_reason"):
        lines.append(f"reason: {run['rejection_reason']}")
    for step in run.get("steps", []):
        suffix = f" error={step['error']}" if step.get("error") else ""
        lines.append(f"- {step['step_key']}: {step['status']} ({step['primitive']}){suffix}")
    return "\n".join(lines)


def _format_recent_events(run_id: str, events: list[dict[str, Any]]) -> str:
    lines = [f"recent events for {run_id}:"]
    for event in events:
        lines.append(f"- {event['created_at']} {event['actor']} {event['action']}")
    return "\n".join(lines)
