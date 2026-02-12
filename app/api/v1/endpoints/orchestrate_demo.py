"""Demo endpoint for frontend UI visualization

发送详细的agent执行事件用于前端展示
"""
from __future__ import annotations

import asyncio
import logging
import uuid

from fastapi import APIRouter
from pydantic import BaseModel

from app.services.detailed_event_emitter import (
    DetailedEventEmitter,
    emit_detailed_round_execution,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/orchestrate", tags=["orchestrate"])


class DemoRequest(BaseModel):
    """Demo campaign request"""
    objective_kpi: str = "overpotential_eta10"
    max_rounds: int = 2  # Demo只运行2轮


class DemoResponse(BaseModel):
    """Demo campaign response"""
    campaign_id: str
    status: str = "started"


async def _run_demo_campaign(campaign_id: str, max_rounds: int):
    """运行demo campaign，发送详细的执行事件"""

    # Import orchestrate_events to get the emit function
    from app.api.v1.endpoints.orchestrate_events import publish_campaign_event

    # Create detailed emitter
    emitter = DetailedEventEmitter(campaign_id, publish_campaign_event)

    try:
        # Campaign start
        publish_campaign_event(campaign_id, {
            "type": "campaign_start",
            "campaign_id": campaign_id,
            "phase": "demo",
            "message": "Starting demo campaign with detailed execution trace...",
        })

        await asyncio.sleep(0.5)

        # Phase 1: Planning
        publish_campaign_event(campaign_id, {
            "type": "agent_thinking",
            "agent": "planner",
            "message": "Analyzing parameter space and generating campaign plan...",
        })

        await asyncio.sleep(1.0)

        publish_campaign_event(campaign_id, {
            "type": "agent_result",
            "agent": "planner",
            "success": True,
            "message": f"Plan generated: {max_rounds} rounds, 14D search space",
        })

        await asyncio.sleep(0.5)

        # Execute rounds
        for round_num in range(1, max_rounds + 1):
            # Round start
            strategy = "lhs" if round_num == 1 else "bayesian_knn"

            publish_campaign_event(campaign_id, {
                "type": "round_start",
                "round": round_num,
                "total_rounds": max_rounds,
                "strategy": strategy,
                "message": f"Starting round {round_num}/{max_rounds} (strategy: {strategy})",
            })

            await asyncio.sleep(0.5)

            # Detailed round execution
            candidate_params = {
                "stock_1_fraction": 0.15,
                "stock_2_fraction": 0.08,
                "stock_3_fraction": 0.22,
                "stock_4_fraction": 0.05,
                "stock_5_fraction": 0.12,
                "stock_6_fraction": 0.18,
                "stock_7_fraction": 0.03,
                "stock_8_fraction": 0.09,
                "stock_9_fraction": 0.06,
                "stock_10_fraction": 0.02,
                "total_volume_ml": 2.5,
                "deposition_current_density_ma_cm2": 10.0,
                "deposition_time_seconds": 45.0,
                "temperature_c": 35.0,
            }

            # Emit detailed execution with simulated delays
            await emit_detailed_round_execution(
                emitter,
                round_num,
                strategy,
                candidate_params,
                simulate=True
            )

            # Round complete
            eta10 = 127.3 if round_num == 1 else 89.7
            improvement = 0.0 if round_num == 1 else 29.5

            publish_campaign_event(campaign_id, {
                "type": "round_complete",
                "round": round_num,
                "eta10": eta10,
                "improvement_pct": improvement,
                "message": f"Round {round_num} complete: η10 = {eta10} mV",
            })

            await asyncio.sleep(0.5)

            # Convergence analysis
            if round_num == max_rounds:
                publish_campaign_event(campaign_id, {
                    "type": "agent_thinking",
                    "agent": "stop",
                    "message": "Analyzing convergence...",
                })

                await asyncio.sleep(1.0)

                publish_campaign_event(campaign_id, {
                    "type": "agent_result",
                    "agent": "stop",
                    "success": True,
                    "message": "Status: IMPROVING - Continue optimization recommended",
                })

                await asyncio.sleep(0.5)

        # Campaign complete
        publish_campaign_event(campaign_id, {
            "type": "campaign_complete",
            "campaign_id": campaign_id,
            "status": "completed",
            "total_rounds": max_rounds,
            "best_eta10": 89.7,
            "message": f"Demo campaign completed: {max_rounds} rounds executed",
        })

    except Exception as exc:
        logger.exception("Demo campaign failed")
        publish_campaign_event(campaign_id, {
            "type": "campaign_complete",
            "campaign_id": campaign_id,
            "status": "failed",
            "error": str(exc),
            "message": f"Demo campaign failed: {exc}",
        })


@router.post("/demo", response_model=DemoResponse)
async def orchestrate_demo(payload: DemoRequest) -> DemoResponse:
    """
    启动demo campaign，展示详细的agent执行过程

    用于前端UI录屏演示，发送详细的execution tree事件
    """
    campaign_id = f"demo-{uuid.uuid4().hex[:12]}"

    # 后台运行demo
    task = asyncio.create_task(
        _run_demo_campaign(campaign_id, payload.max_rounds),
        name=f"demo-{campaign_id}",
    )

    # Store task reference (for tracking/stopping if needed)
    from app.api.v1.endpoints.orchestrate import _running_campaigns
    _running_campaigns[campaign_id] = task

    return DemoResponse(campaign_id=campaign_id, status="started")
