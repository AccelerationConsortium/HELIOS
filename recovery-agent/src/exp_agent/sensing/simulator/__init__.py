"""
Simulator - Fault injection and replay for testing.

Components:
- FaultInjector: Inject various faults into mock sensors
- Replay: Replay historical events from logs
"""

from exp_agent.sensing.simulator.fault_injector import (
    FaultInjector,
    FaultType,
    FaultConfig,
)
from exp_agent.sensing.simulator.replay import (
    ReplayDriver,
    ReplayConfig,
)

__all__ = [
    "FaultInjector",
    "FaultType",
    "FaultConfig",
    "ReplayDriver",
    "ReplayConfig",
]
