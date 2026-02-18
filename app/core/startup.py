"""Startup validation framework.

Runs ordered checks before the application accepts traffic.
Each check returns a :class:`CheckResult`; any *required* check
failure prevents the application from starting.
"""
from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single startup check."""

    name: str
    ok: bool
    message: str
    elapsed_ms: float = 0.0
    details: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_environment_vars() -> CheckResult:
    """Validate critical environment variables and their constraints."""
    t0 = time.monotonic()
    settings = get_settings()
    warnings: list[str] = []

    # ADAPTER_MODE must be one of the known values
    valid_modes = {"simulated", "battery_lab"}
    if settings.adapter_mode not in valid_modes:
        return CheckResult(
            name="environment_vars",
            ok=False,
            message=f"ADAPTER_MODE='{settings.adapter_mode}' invalid; "
            f"expected one of {valid_modes}",
            elapsed_ms=_elapsed(t0),
        )

    # LLM_PROVIDER must be known
    valid_providers = {"mock", "anthropic", "openai"}
    if settings.llm_provider not in valid_providers:
        warnings.append(
            f"LLM_PROVIDER='{settings.llm_provider}' not in {valid_providers}"
        )

    # If LLM_PROVIDER is not mock, API key should be present
    if settings.llm_provider != "mock" and not settings.llm_api_key:
        return CheckResult(
            name="environment_vars",
            ok=False,
            message=f"LLM_PROVIDER='{settings.llm_provider}' requires "
            "LLM_API_KEY to be set",
            elapsed_ms=_elapsed(t0),
        )

    msg = "All environment variables valid"
    if warnings:
        msg += f" (warnings: {'; '.join(warnings)})"

    return CheckResult(
        name="environment_vars",
        ok=True,
        message=msg,
        elapsed_ms=_elapsed(t0),
        details={"adapter_mode": settings.adapter_mode, "llm_provider": settings.llm_provider},
    )


def check_data_directories() -> CheckResult:
    """Verify data/object-store directories are writable."""
    t0 = time.monotonic()
    settings = get_settings()
    issues: list[str] = []

    for label, dirpath in [
        ("data_dir", settings.data_dir),
        ("object_store_dir", settings.object_store_dir),
    ]:
        try:
            dirpath.mkdir(parents=True, exist_ok=True)
            # Verify write access via a temp file
            probe = dirpath / ".startup_probe"
            probe.write_text("ok")
            probe.unlink()
        except OSError as exc:
            issues.append(f"{label} ({dirpath}): {exc}")

    if issues:
        return CheckResult(
            name="data_directories",
            ok=False,
            message=f"Directory issues: {'; '.join(issues)}",
            elapsed_ms=_elapsed(t0),
        )

    return CheckResult(
        name="data_directories",
        ok=True,
        message="Data directories writable",
        elapsed_ms=_elapsed(t0),
        details={
            "data_dir": str(settings.data_dir),
            "object_store_dir": str(settings.object_store_dir),
        },
    )


def check_disk_space(min_mb: int = 100) -> CheckResult:
    """Ensure sufficient free disk space for data_dir."""
    t0 = time.monotonic()
    settings = get_settings()
    try:
        usage = shutil.disk_usage(settings.data_dir)
        free_mb = usage.free / (1024 * 1024)
        if free_mb < min_mb:
            return CheckResult(
                name="disk_space",
                ok=False,
                message=f"Insufficient disk space: {free_mb:.0f} MB free "
                f"(need {min_mb} MB)",
                elapsed_ms=_elapsed(t0),
                details={"free_mb": round(free_mb, 1), "required_mb": min_mb},
            )
        return CheckResult(
            name="disk_space",
            ok=True,
            message=f"Disk space OK: {free_mb:.0f} MB free",
            elapsed_ms=_elapsed(t0),
            details={"free_mb": round(free_mb, 1)},
        )
    except OSError as exc:
        return CheckResult(
            name="disk_space",
            ok=False,
            message=f"Cannot check disk space: {exc}",
            elapsed_ms=_elapsed(t0),
        )


def check_database() -> CheckResult:
    """Test DB connectivity: open, table count, write/read cycle."""
    t0 = time.monotonic()
    try:
        from app.core.db import connection

        with connection() as conn:
            # Count tables
            tables = conn.execute(
                "SELECT count(*) FROM sqlite_master WHERE type='table'"
            ).fetchone()[0]
            # Quick write/read probe
            conn.execute(
                "CREATE TABLE IF NOT EXISTS _startup_probe (v TEXT)"
            )
            conn.execute("INSERT INTO _startup_probe VALUES ('ok')")
            row = conn.execute(
                "SELECT v FROM _startup_probe LIMIT 1"
            ).fetchone()
            conn.execute("DROP TABLE IF EXISTS _startup_probe")
            conn.commit()

            if row is None or row[0] != "ok":
                return CheckResult(
                    name="database",
                    ok=False,
                    message="DB write/read verification failed",
                    elapsed_ms=_elapsed(t0),
                )

        return CheckResult(
            name="database",
            ok=True,
            message=f"Database OK ({tables} tables)",
            elapsed_ms=_elapsed(t0),
            details={"table_count": tables},
        )
    except Exception as exc:
        return CheckResult(
            name="database",
            ok=False,
            message=f"Database check failed: {exc}",
            elapsed_ms=_elapsed(t0),
        )


def check_llm_provider() -> CheckResult:
    """Verify LLM provider configuration is usable.

    For ``mock`` mode, always passes.  For real providers,
    validates that the API key has the expected prefix.
    """
    t0 = time.monotonic()
    settings = get_settings()

    if settings.llm_provider == "mock":
        return CheckResult(
            name="llm_provider",
            ok=True,
            message="LLM provider: mock (no API key required)",
            elapsed_ms=_elapsed(t0),
        )

    key = settings.llm_api_key
    if not key:
        return CheckResult(
            name="llm_provider",
            ok=False,
            message=f"LLM provider '{settings.llm_provider}' requires API key",
            elapsed_ms=_elapsed(t0),
        )

    # Basic prefix validation (non-exhaustive)
    prefix_map = {"anthropic": "sk-ant-", "openai": "sk-"}
    expected = prefix_map.get(settings.llm_provider)
    if expected and not key.startswith(expected):
        return CheckResult(
            name="llm_provider",
            ok=True,  # warning, not a blocker
            message=f"LLM API key doesn't start with expected '{expected}' "
            f"(provider: {settings.llm_provider})",
            elapsed_ms=_elapsed(t0),
            details={"warning": "key_prefix_mismatch"},
        )

    return CheckResult(
        name="llm_provider",
        ok=True,
        message=f"LLM provider '{settings.llm_provider}' configured",
        elapsed_ms=_elapsed(t0),
        details={"provider": settings.llm_provider, "model": settings.llm_model},
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


# Ordered list of (check_fn, required).
# Required=True means the app must not start if this check fails.
STARTUP_CHECKS: list[tuple[Any, bool]] = [
    (check_environment_vars, True),
    (check_data_directories, True),
    (check_disk_space, False),
    (check_database, True),
    (check_llm_provider, False),
]


def run_startup_checks() -> list[CheckResult]:
    """Execute all startup checks in order.

    Returns
    -------
    list[CheckResult]
        Results for every check.

    Raises
    ------
    RuntimeError
        If any *required* check fails.
    """
    results: list[CheckResult] = []
    failures: list[str] = []
    total_t0 = time.monotonic()

    logger.info("=" * 60)
    logger.info("OTbot startup validation")
    logger.info("=" * 60)

    for check_fn, required in STARTUP_CHECKS:
        result = check_fn()
        results.append(result)

        status = "OK" if result.ok else "FAIL"
        req_tag = " [REQUIRED]" if required else ""
        logger.info(
            "  [%s] %s — %s (%.1f ms)%s",
            status,
            result.name,
            result.message,
            result.elapsed_ms,
            req_tag,
        )

        if not result.ok and required:
            failures.append(f"{result.name}: {result.message}")

    total_ms = _elapsed(total_t0)
    logger.info("-" * 60)

    if failures:
        msg = (
            f"Startup validation failed ({len(failures)} required "
            f"check(s)):\n" + "\n".join(f"  • {f}" for f in failures)
        )
        logger.error(msg)
        raise RuntimeError(msg)

    logger.info(
        "All startup checks passed in %.0f ms (%d checks)",
        total_ms,
        len(results),
    )
    logger.info("=" * 60)
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _elapsed(t0: float) -> float:
    """Milliseconds elapsed since *t0*."""
    return (time.monotonic() - t0) * 1000
