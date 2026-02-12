from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import api_router
from app.core.db import init_db
from app.services.audit import set_event_bus
from app.services.event_bus import EventBus
from app.services.memory import seed_initial_recipes, start_memory_listener, stop_memory_listener
from app.services.metrics import start_metrics_listener, stop_metrics_listener
from app.services.evolution import start_evolution_listener, stop_evolution_listener
from app.services.reviewer import start_reviewer_listener, stop_reviewer_listener
from app.services.scheduler import OrchestratorScheduler

scheduler = OrchestratorScheduler()
event_bus = EventBus()


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    seed_initial_recipes()
    await event_bus.start()
    set_event_bus(event_bus)
    mem_sub = await start_memory_listener(event_bus)
    metrics_sub = await start_metrics_listener(event_bus)
    reviewer_sub = await start_reviewer_listener(event_bus)
    evolution_sub = await start_evolution_listener(event_bus)
    await scheduler.start()
    try:
        yield
    finally:
        await scheduler.stop()
        set_event_bus(None)
        await stop_evolution_listener(evolution_sub, event_bus)
        await stop_reviewer_listener(reviewer_sub, event_bus)
        await stop_metrics_listener(metrics_sub, event_bus)
        await stop_memory_listener(mem_sub, event_bus)
        await event_bus.stop()


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


@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/static/init.html")


@app.get("/lab", include_in_schema=False)
async def lab_redirect():
    """New single-input agent UI."""
    return RedirectResponse(url="/static/lab.html")
