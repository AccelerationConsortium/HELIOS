from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import os


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
        self.campaign_poll_seconds = float(os.getenv("CAMPAIGN_POLL_SECONDS", "5"))
        self.lock_ttl_seconds = int(os.getenv("LOCK_TTL_SECONDS", "90"))
        self.default_firmware_version = os.getenv("DEFAULT_FIRMWARE_VERSION", "sim-fw-1.0.0")
        self.default_calibration_id = os.getenv("DEFAULT_CALIBRATION_ID", "sim-cal-2026-01")

        # ---- Hardware adapter settings ----
        self.adapter_mode: str = os.getenv("ADAPTER_MODE", "simulated")  # simulated | battery_lab
        self.adapter_dry_run: bool = os.getenv("ADAPTER_DRY_RUN", "true").lower() in ("true", "1", "yes")

        # Robot type: "ot2" | "flex"
        self.robot_type: str = os.getenv("ROBOT_TYPE", "ot2")

        # Robot IP
        self.robot_ip: str = os.getenv("ROBOT_IP", "localhost")

        # PLC (Modbus TCP) — address configured inside OT_PLC_Client_Edit
        # No separate setting needed; PLC auto-discovers or uses its own config.

        # USB Relay
        self.relay_port: str = os.getenv("RELAY_PORT", "COM11")

        # Squidstat — not yet integrated; placeholder
        self.squidstat_port: str = os.getenv("SQUIDSTAT_PORT", "")

        # ---- LLM settings ----
        self.llm_provider: str = os.getenv("LLM_PROVIDER", "mock")  # "anthropic" | "mock"
        self.llm_api_key: str = os.getenv("LLM_API_KEY", "")
        self.llm_model: str = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
        self.llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.anthropic.com")
        self.llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
