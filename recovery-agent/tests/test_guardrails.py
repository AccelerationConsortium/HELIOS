import pytest
from exp_agent.core.types import HardwareError, Action, ExecutionState
from exp_agent.executor.guarded_executor import GuardedExecutor

def test_guardrails_safety_check():
    executor = GuardedExecutor()
    state = ExecutionState()
    
    # Safe action
    safe_action = Action(name="set_temperature", effect="write", params={"temperature": 100.0})
    executor.check_safety(state, safe_action) # Should not raise

    # Unsafe action
    unsafe_action = Action(name="set_temperature", effect="write", params={"temperature": 150.0})
    
    with pytest.raises(HardwareError) as excinfo:
        executor.check_safety(state, unsafe_action)
    
    error = excinfo.value
    assert error.type == "safety_violation"
    assert "exceeds safety limit" in error.message
