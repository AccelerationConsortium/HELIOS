from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import api_router
from app.core.db import init_db
from app.core.startup import run_startup_checks
from app.services.audit import set_event_bus
from app.services.event_bus import EventBus
from app.services.memory import seed_initial_recipes, start_memory_listener, stop_memory_listener
from app.services.metrics import start_metrics_listener, stop_metrics_listener
from app.services.campaign_metrics import (
    start_campaign_metrics_listener,
    stop_campaign_metrics_listener,
)
from app.services.evolution import start_evolution_listener, stop_evolution_listener
from app.services.reviewer import start_reviewer_listener, stop_reviewer_listener
from app.services.scheduler import OrchestratorScheduler

logger = logging.getLogger(__name__)

scheduler = OrchestratorScheduler()
event_bus = EventBus()


@asynccontextmanager
async def lifespan(_: FastAPI):
    t0 = time.monotonic()

    # ---- Phase 1: Pre-flight validation ----
    run_startup_checks()  # raises RuntimeError on required-check failure

    # ---- Phase 2: Database ----
    try:
        init_db()
        logger.info("Database initialised")
    except Exception:
        logger.exception("Database initialisation failed")
        raise

    # ---- Phase 3: Seed data ----
    try:
        seed_initial_recipes()
        logger.info("Initial recipes seeded")
    except Exception:
        logger.exception("Recipe seeding failed")
        raise

    # ---- Phase 4: Event bus + listeners ----
    await event_bus.start()
    set_event_bus(event_bus)
    logger.info("Event bus started")

    mem_sub = await start_memory_listener(event_bus)
    metrics_sub = await start_metrics_listener(event_bus)
    reviewer_sub = await start_reviewer_listener(event_bus)
    evolution_sub = await start_evolution_listener(event_bus)
    campaign_metrics_sub = await start_campaign_metrics_listener(event_bus)
    logger.info("All event listeners registered")

    # ---- Phase 5: Scheduler ----
    await scheduler.start()

    elapsed = (time.monotonic() - t0) * 1000
    logger.info("OTbot startup complete in %.0f ms", elapsed)

    try:
        yield
    finally:
        logger.info("OTbot shutting down …")
        await scheduler.stop()
        set_event_bus(None)
        await stop_campaign_metrics_listener(campaign_metrics_sub, event_bus)
        await stop_evolution_listener(evolution_sub, event_bus)
        await stop_reviewer_listener(reviewer_sub, event_bus)
        await stop_metrics_listener(metrics_sub, event_bus)
        await stop_memory_listener(mem_sub, event_bus)
        await event_bus.stop()
        logger.info("OTbot shutdown complete")


app = FastAPI(
    title="Lab Orchestrator MVP",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(api_router, prefix="/api/v1")

# Static files for the initialization UI
_static_dir = Path(__file__).resolve().parent / "static"
if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/health", tags=["health"])
async def root_health() -> dict:
    """Root-level liveness probe for Docker / load balancer.

    The detailed endpoints live at ``/api/v1/health/ready`` and
    ``/api/v1/health/detail``.
    """
    return {"ok": True, "service": "otbot"}


@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/static/init.html")


@app.get("/lab", include_in_schema=False)
async def lab_redirect():
    """New single-input agent UI."""
    return RedirectResponse(url="/static/lab.html")
