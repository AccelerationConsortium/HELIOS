from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.endpoints.agent import router as agent_router
from app.api.v1.endpoints.batches import router as batches_router
from app.api.v1.endpoints.campaigns import router as campaigns_router
from app.api.v1.endpoints.capabilities import router as capabilities_router
from app.api.v1.endpoints.evolution import router as evolution_router
from app.api.v1.endpoints.events_stream import router as events_stream_router
from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.metrics import router as metrics_router
from app.api.v1.endpoints.reviews import router as reviews_router
from app.api.v1.endpoints.runs import router as runs_router
from app.api.v1.endpoints.triggers import router as triggers_router
from app.api.v1.endpoints.init import router as init_router
from app.api.v1.endpoints.orchestrate import router as orchestrate_router
from app.api.v1.endpoints.orchestrate_events import router as orchestrate_events_router
from app.api.v1.endpoints.nl_parse import router as nl_parse_router
from app.api.v1.endpoints.onboarding import router as onboarding_router
from app.api.v1.endpoints.workflows import router as workflows_router
from app.api.v1.endpoints.orchestrate_demo import router as orchestrate_demo_router

api_router = APIRouter()
api_router.include_router(agent_router)
api_router.include_router(batches_router)
api_router.include_router(health_router)
api_router.include_router(campaigns_router)
api_router.include_router(capabilities_router)
api_router.include_router(evolution_router)
api_router.include_router(events_stream_router)
api_router.include_router(triggers_router)
api_router.include_router(metrics_router)
api_router.include_router(reviews_router)
api_router.include_router(runs_router)
api_router.include_router(workflows_router)
api_router.include_router(init_router)
api_router.include_router(orchestrate_router)
api_router.include_router(orchestrate_events_router)
api_router.include_router(orchestrate_demo_router)
api_router.include_router(nl_parse_router)
api_router.include_router(onboarding_router)
