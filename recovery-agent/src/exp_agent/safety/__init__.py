"""Safety module for chemical safety integration.

This module provides the interface between the disaster recovery agent
and external chemical safety agents (e.g., Safety SDL Agent).

Key components:
- SafetyAgent: Protocol for safety agent implementations
- MockSafetyAgent: Testing implementation
- safety_check: Runtime action validation against SafetyPacket
"""

from exp_agent.safety.agent import SafetyAgent
from exp_agent.safety.mock_agent import MockSafetyAgent
from exp_agent.safety.checker import check_action_safety

__all__ = [
    "SafetyAgent",
    "MockSafetyAgent",
    "check_action_safety",
]
