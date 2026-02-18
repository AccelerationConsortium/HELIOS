"""Health-check endpoints.

Three tiers:
- ``GET /health``       — Liveness (always 200 if process is up)
- ``GET /health/ready`` — Readiness (checks DB + event bus)
- ``GET /health/detail``— Full diagnostic status for debugging
"""
from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter

router = APIRouter(tags=["health"])


# ---------------------------------------------------------------------------
# Liveness — is the process alive?
# ---------------------------------------------------------------------------


@router.get("/health")
async def health() -> dict[str, Any]:
    """Liveness probe for Docker / k8s / load balancer."""
    return {"ok": True, "service": "otbot"}


# ---------------------------------------------------------------------------
# Readiness — is the app ready to serve traffic?
# ---------------------------------------------------------------------------


@router.get("/health/ready")
async def health_ready() -> dict[str, Any]:
    """Readiness probe: DB reachable and event bus running."""
    checks: dict[str, bool] = {}

    # DB connectivity
    try:
        from app.core.db import connection

        with connection() as conn:
            conn.execute("SELECT 1").fetchone()
        checks["database"] = True
    except Exception:
        checks["database"] = False

    # Event bus loop captured
    try:
        from app.main import event_bus

        checks["event_bus"] = event_bus._loop is not None
    except Exception:
        checks["event_bus"] = False

    all_ok = all(checks.values())
    return {"ok": all_ok, "checks": checks}


# ---------------------------------------------------------------------------
# Detailed status — full diagnostic view
# ---------------------------------------------------------------------------


@router.get("/health/detail")
async def health_detail() -> dict[str, Any]:
    """Comprehensive health detail for debugging & monitoring dashboards."""
    t0 = time.monotonic()
    components: dict[str, dict[str, Any]] = {}

    # 1. Database
    components["database"] = _check_database()

    # 2. Event bus
    components["event_bus"] = _check_event_bus()

    # 3. Scheduler
    components["scheduler"] = _check_scheduler()

    # 4. Disk space
    components["disk"] = _check_disk()

    # 5. Configuration summary (no secrets)
    components["config"] = _check_config()

    all_ok = all(c.get("ok", False) for c in components.values())
    elapsed_ms = (time.monotonic() - t0) * 1000

    return {
        "ok": all_ok,
        "service": "otbot",
        "elapsed_ms": round(elapsed_ms, 1),
        "components": components,
    }


# ---------------------------------------------------------------------------
# Component checkers (internal)
# ---------------------------------------------------------------------------


def _check_database() -> dict[str, Any]:
    """Test DB connectivity and report table count."""
    try:
        from app.core.db import connection

        with connection() as conn:
            tables = conn.execute(
                "SELECT count(*) FROM sqlite_master WHERE type='table'"
            ).fetchone()[0]
            views = conn.execute(
                "SELECT count(*) FROM sqlite_master WHERE type='view'"
            ).fetchone()[0]
        return {"ok": True, "tables": tables, "views": views}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _check_event_bus() -> dict[str, Any]:
    """Check event bus state."""
    try:
        from app.main import event_bus

        loop_active = event_bus._loop is not None
        subscriber_count = len(event_bus._subscribers)
        return {
            "ok": loop_active,
            "loop_active": loop_active,
            "subscribers": subscriber_count,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _check_scheduler() -> dict[str, Any]:
    """Check scheduler state."""
    try:
        from app.main import scheduler

        task_count = len(scheduler._tasks)
        active_workers = len(scheduler._active_workers)
        orchestrator_tasks = len(scheduler._orchestrator_tasks)
        return {
            "ok": task_count > 0,
            "loop_tasks": task_count,
            "active_workers": active_workers,
            "orchestrator_tasks": orchestrator_tasks,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _check_disk() -> dict[str, Any]:
    """Report free disk space for data_dir."""
    try:
        import shutil

        from app.core.config import get_settings

        settings = get_settings()
        usage = shutil.disk_usage(settings.data_dir)
        free_mb = round(usage.free / (1024 * 1024), 1)
        total_mb = round(usage.total / (1024 * 1024), 1)
        return {
            "ok": free_mb > 100,
            "free_mb": free_mb,
            "total_mb": total_mb,
            "data_dir": str(settings.data_dir),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _check_config() -> dict[str, Any]:
    """Return non-sensitive configuration summary."""
    try:
        from app.core.config import get_settings

        settings = get_settings()
        return {
            "ok": True,
            "adapter_mode": settings.adapter_mode,
            "adapter_dry_run": settings.adapter_dry_run,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "scheduler_poll_seconds": settings.scheduler_poll_seconds,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
