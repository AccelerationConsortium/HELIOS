"""
SensingAPI - HTTP endpoints for sensor data access.

Provides a REST API for decision layers to query sensor state.
Can be used standalone or integrated into an existing FastAPI app.
"""

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, Any

from exp_agent.sensing.hub.sensor_hub import SensorHub
from exp_agent.sensing.health.health_monitor import HealthMonitor
from exp_agent.sensing.protocol.sensor_event import SensorType
from exp_agent.sensing.protocol.health_event import HealthStatus


@dataclass
class SensingAPI:
    """
    API interface for sensor data.

    Can be used directly or wrapped in HTTP endpoints.

    Example usage:
        api = SensingAPI(hub, monitor)
        snapshot = api.get_snapshot()
        health = api.get_health("temp_1")
    """

    hub: SensorHub
    health_monitor: Optional[HealthMonitor] = None

    def get_snapshot(self) -> dict[str, Any]:
        """
        Get current state panel for all sensors.

        Returns:
            {
                "ts": "2026-02-05T14:20:11Z",
                "system_status": "NOMINAL|DEGRADED|CRITICAL",
                "sensors": {
                    "sensor_id": {
                        "latest_value": 25.3,
                        "latest_unit": "C",
                        "age_seconds": 1.2,
                        "health_status": "HEALTHY",
                        "trend_slope": 0.1,
                        ...
                    }
                }
            }
        """
        snapshot = self.hub.get_snapshot()
        return snapshot.to_dict()

    def get_sensor(self, sensor_id: str) -> Optional[dict[str, Any]]:
        """Get snapshot for a specific sensor."""
        sensor_snapshot = self.hub.get_sensor_snapshot(sensor_id)
        if sensor_snapshot is None:
            return None
        return sensor_snapshot.to_dict()

    def get_value(self, sensor_id: str) -> Optional[float]:
        """Get latest value for a sensor."""
        return self.hub.get_latest_value(sensor_id)

    def get_health(self, sensor_id: str) -> dict[str, Any]:
        """Get health status for a sensor."""
        if self.health_monitor:
            status = self.health_monitor.check_health(sensor_id)
            metrics = self.health_monitor.get_metrics(sensor_id)
            return {
                "sensor_id": sensor_id,
                "status": status.value,
                "metrics": metrics.to_dict() if metrics else None,
            }
        else:
            # Fallback to hub snapshot
            snapshot = self.hub.get_sensor_snapshot(sensor_id)
            if snapshot:
                return {
                    "sensor_id": sensor_id,
                    "status": snapshot.health_status.value,
                    "is_healthy": snapshot.is_healthy,
                }
            return {"sensor_id": sensor_id, "status": "UNKNOWN"}

    def get_all_health(self) -> dict[str, Any]:
        """Get health summary for all sensors."""
        if self.health_monitor:
            return self.health_monitor.get_summary()
        else:
            snapshot = self.hub.get_snapshot()
            return {
                "system_status": snapshot.system_status.value,
                "healthy_count": snapshot.healthy_count,
                "degraded_count": snapshot.degraded_count,
                "unhealthy_count": snapshot.unhealthy_count,
                "offline_count": snapshot.offline_count,
            }

    def get_history(
        self,
        sensor_id: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get historical events."""
        events = self.hub.get_history(
            sensor_id=sensor_id,
            since=since,
            limit=limit,
        )
        return [e.to_dict() for e in events]

    def get_sensors_by_type(self, sensor_type: str) -> list[dict[str, Any]]:
        """Get all sensors of a specific type."""
        try:
            stype = SensorType(sensor_type)
        except ValueError:
            return []

        snapshot = self.hub.get_snapshot()
        sensors = snapshot.get_sensors_by_type(stype)
        return [s.to_dict() for s in sensors]

    def get_unhealthy_sensors(self) -> list[str]:
        """Get list of unhealthy sensor IDs."""
        if self.health_monitor:
            return self.health_monitor.get_unhealthy_sensors()
        else:
            snapshot = self.hub.get_snapshot()
            return [
                sid for sid, s in snapshot.sensors.items()
                if s.health_status in (HealthStatus.UNHEALTHY, HealthStatus.OFFLINE)
            ]

    def get_stats(self) -> dict[str, Any]:
        """Get hub statistics."""
        return self.hub.get_stats()

    def get_driver_status(self) -> dict[str, dict]:
        """Get status of all drivers."""
        return self.hub.get_driver_status()

    # Convenience methods for safety-critical checks

    def is_safe_to_proceed(self) -> tuple[bool, list[str]]:
        """
        Check if it's safe to proceed with operations.

        Returns:
            (is_safe, list of issues)
        """
        issues = []

        snapshot = self.hub.get_snapshot()

        # Check for critical sensor issues
        if not snapshot.critical_sensors_ok:
            issues.extend([
                f"Critical sensor issue: {sid}"
                for sid in snapshot.critical_sensor_issues
            ])

        # Check for unhealthy sensors
        unhealthy = self.get_unhealthy_sensors()
        if unhealthy:
            issues.extend([f"Unhealthy sensor: {sid}" for sid in unhealthy])

        return (len(issues) == 0, issues)

    def check_threshold(
        self,
        sensor_id: str,
        max_value: Optional[float] = None,
        min_value: Optional[float] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if a sensor value is within thresholds.

        Returns:
            (is_ok, violation_message)
        """
        value = self.get_value(sensor_id)
        if value is None:
            return (False, f"Sensor {sensor_id} has no value")

        if max_value is not None and value > max_value:
            return (False, f"{sensor_id} value {value} exceeds max {max_value}")

        if min_value is not None and value < min_value:
            return (False, f"{sensor_id} value {value} below min {min_value}")

        return (True, None)

    def check_airflow(
        self,
        sensor_id: str,
        min_airflow: float = 0.3,
    ) -> tuple[bool, Optional[str]]:
        """Check fume hood airflow is adequate."""
        return self.check_threshold(sensor_id, min_value=min_airflow)

    def check_temperature(
        self,
        sensor_id: str,
        max_temp: float,
    ) -> tuple[bool, Optional[str]]:
        """Check temperature is below limit."""
        return self.check_threshold(sensor_id, max_value=max_temp)

    def check_pressure(
        self,
        sensor_id: str,
        max_pressure: float,
    ) -> tuple[bool, Optional[str]]:
        """Check pressure is below limit."""
        return self.check_threshold(sensor_id, max_value=max_pressure)


def create_sensing_router(api: SensingAPI):
    """
    Create a FastAPI router for the sensing API.

    Usage:
        from fastapi import FastAPI
        app = FastAPI()
        api = SensingAPI(hub, monitor)
        app.include_router(create_sensing_router(api), prefix="/sensors")
    """
    try:
        from fastapi import APIRouter, HTTPException, Query
    except ImportError:
        raise ImportError("FastAPI is required for HTTP API. Install with: pip install fastapi")

    router = APIRouter(tags=["sensors"])

    @router.get("/snapshot")
    def get_snapshot():
        """Get current state panel for all sensors."""
        return api.get_snapshot()

    @router.get("/health")
    def get_all_health():
        """Get health summary for all sensors."""
        return api.get_all_health()

    @router.get("/health/{sensor_id}")
    def get_sensor_health(sensor_id: str):
        """Get health status for a specific sensor."""
        return api.get_health(sensor_id)

    @router.get("/{sensor_id}")
    def get_sensor(sensor_id: str):
        """Get snapshot for a specific sensor."""
        result = api.get_sensor(sensor_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Sensor {sensor_id} not found")
        return result

    @router.get("/{sensor_id}/value")
    def get_sensor_value(sensor_id: str):
        """Get latest value for a sensor."""
        value = api.get_value(sensor_id)
        if value is None:
            raise HTTPException(status_code=404, detail=f"Sensor {sensor_id} not found")
        return {"sensor_id": sensor_id, "value": value}

    @router.get("/{sensor_id}/history")
    def get_sensor_history(
        sensor_id: str,
        limit: int = Query(default=100, le=1000),
        since_minutes: Optional[int] = Query(default=None),
    ):
        """Get historical events for a sensor."""
        since = None
        if since_minutes:
            since = datetime.now(timezone.utc) - timedelta(minutes=since_minutes)
        return api.get_history(sensor_id=sensor_id, since=since, limit=limit)

    @router.get("/type/{sensor_type}")
    def get_sensors_by_type(sensor_type: str):
        """Get all sensors of a specific type."""
        return api.get_sensors_by_type(sensor_type)

    @router.get("/status/unhealthy")
    def get_unhealthy_sensors():
        """Get list of unhealthy sensor IDs."""
        return {"unhealthy": api.get_unhealthy_sensors()}

    @router.get("/status/safe")
    def check_safe_to_proceed():
        """Check if it's safe to proceed with operations."""
        is_safe, issues = api.is_safe_to_proceed()
        return {"is_safe": is_safe, "issues": issues}

    @router.get("/stats")
    def get_stats():
        """Get hub statistics."""
        return api.get_stats()

    @router.get("/drivers")
    def get_driver_status():
        """Get status of all drivers."""
        return api.get_driver_status()

    return router
