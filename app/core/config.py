from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path


class Settings:
    def __init__(self) -> None:
        root = Path(__file__).resolve().parents[2]
        self.workspace_root = Path(os.getenv("WORKSPACE_ROOT", str(root)))
        self.data_dir = Path(os.getenv("DATA_DIR", str(self.workspace_root / "data")))
        self.db_path = Path(os.getenv("DB_PATH", str(self.data_dir / "orchestrator.db")))
        self.object_store_dir = Path(
            os.getenv("OBJECT_STORE_DIR", str(self.data_dir / "object_store"))
        )
        self.scheduler_poll_seconds = float(os.getenv("SCHEDULER_POLL_SECONDS", "2"))
        # Upper bound on concurrently-executing worker threads. Defaults to the
        # CPU count clamped to [2, 8] so a busy queue cannot spawn unbounded
        # threads and exhaust hardware resources.
        self.max_concurrent_workers = int(
            os.getenv("MAX_CONCURRENT_WORKERS", str(max(2, min(8, os.cpu_count() or 4))))
        )
        self.campaign_poll_seconds = float(os.getenv("CAMPAIGN_POLL_SECONDS", "5"))
        self.lock_ttl_seconds = int(os.getenv("LOCK_TTL_SECONDS", "90"))
        self.default_firmware_version = os.getenv("DEFAULT_FIRMWARE_VERSION", "sim-fw-1.0.0")
        self.default_calibration_id = os.getenv("DEFAULT_CALIBRATION_ID", "sim-cal-2026-01")

        # ---- Hardware adapter settings ----
        self.adapter_mode: str = os.getenv("ADAPTER_MODE", "simulated")  # simulated | battery_lab
        self.adapter_dry_run: bool = os.getenv("ADAPTER_DRY_RUN", "true").lower() in ("true", "1", "yes")
        self.execution_backend: str = os.getenv("EXECUTION_BACKEND", "local").lower()  # local | puda

        # ---- PUDA execution backend settings ----
        self.puda_nats_servers: list[str] = [
            s.strip()
            for s in os.getenv("PUDA_NATS_SERVERS", "nats://localhost:4222").split(",")
            if s.strip()
        ]
        self.puda_user_id: str = os.getenv("PUDA_USER_ID", "helios")
        self.puda_username: str = os.getenv("PUDA_USERNAME", "HELIOS")
        self.puda_default_machine_id: str = os.getenv("PUDA_DEFAULT_MACHINE_ID", "default")
        self.puda_command_timeout_seconds: int = int(os.getenv("PUDA_COMMAND_TIMEOUT_SECONDS", "120"))
        self.puda_machine_map: dict[str, str] = self._load_puda_machine_map(
            os.getenv("PUDA_MACHINE_MAP", "{}")
        )

        # ---- Telegram control-plane settings ----
        self.telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.telegram_allowed_chat_ids: set[int] = self._load_int_set(
            os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", "")
        )
        self.telegram_poll_timeout_seconds: int = int(os.getenv("TELEGRAM_POLL_TIMEOUT_SECONDS", "25"))
        self.telegram_default_instrument_id: str = os.getenv(
            "TELEGRAM_DEFAULT_INSTRUMENT_ID", "sim-instrument-1"
        )

        # Robot type: "ot2" | "flex"
        self.robot_type: str = os.getenv("ROBOT_TYPE", "ot2")

        # Robot IP
        self.robot_ip: str = os.getenv("ROBOT_IP", "localhost")

        # PLC (Modbus TCP) — address configured inside OT_PLC_Client_Edit
        # No separate setting needed; PLC auto-discovers or uses its own config.

        # USB Relay — use "auto" for cross-platform detection
        self.relay_port: str = os.getenv("RELAY_PORT", "auto")

        # Squidstat — use "auto" for cross-platform detection
        self.squidstat_port: str = os.getenv("SQUIDSTAT_PORT", "auto")

        # ---- LLM settings ----
        self.llm_provider: str = os.getenv("LLM_PROVIDER", "mock")  # "anthropic" | "openai" | "mock"
        self.llm_api_key: str = os.getenv("LLM_API_KEY", "")
        self.llm_model: str = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
        self.llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.anthropic.com")
        self.llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))

        # ---- Optional Nexus REST advisor settings ----
        # The per-round optimization path is in-process.  REST advisory calls
        # are opt-in so a missing Nexus server cannot add latency to normal runs.
        self.nexus_advisor_enabled: bool = os.getenv(
            "NEXUS_ADVISOR_ENABLED", "false"
        ).lower() in ("true", "1", "yes")

    @staticmethod
    def _load_puda_machine_map(raw: str) -> dict[str, str]:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("PUDA_MACHINE_MAP must be a JSON object") from exc
        if not isinstance(parsed, dict):
            raise ValueError("PUDA_MACHINE_MAP must be a JSON object")
        return {str(k): str(v) for k, v in parsed.items()}

    @staticmethod
    def _load_int_set(raw: str) -> set[int]:
        values: set[int] = set()
        for item in raw.split(","):
            value = item.strip()
            if not value:
                continue
            values.add(int(value))
        return values


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
