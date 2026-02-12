"""
Sensing API - HTTP endpoints for sensor data access.

Provides:
- GET /sensors/snapshot - Current state panel
- GET /sensors/health - Health status
- GET /sensors/{id}/history - Historical data
"""

from exp_agent.sensing.api.sensing_api import SensingAPI, create_sensing_router

__all__ = [
    "SensingAPI",
    "create_sensing_router",
]
