# Recovery Agent Integration Guide

## Overview

RecoveryAgent provides intelligent error recovery strategies with cross-cutting veto power in the OTbot architecture. It wraps the `recovery-agent` package to provide policy-driven retry/abort decisions, chemical safety escalation, and fault signature analysis.

## Architecture Position

```
┌──────────────────────────────────────────────────┐
│            OTbot Agent Layers                    │
├──────────────────────────────────────────────────┤
│  L3: Task Entry (User Intent)                   │
│  L2: Planning (Campaign Strategy)               │
│  L1: Compilation (Protocol Generation)          │
│  L0: Execution (Hardware Control)               │
├──────────────────────────────────────────────────┤
│  Cross-Cutting (Veto Power):                    │
│    • SafetyAgent (Pre-execution validation)     │
│    • RecoveryAgent (Post-error decision) ⭐      │
│    • SensingAgent (Post-execution QC)           │
└──────────────────────────────────────────────────┘
```

**Key Characteristics**:
- **Layer**: Cross-cutting (same as SafetyAgent)
- **Trigger**: After execution failures
- **Authority**: Veto power to decide retry/abort/degrade strategies
- **Safety Integration**: Escalates chemical safety events to SafetyAgent

## Installation

The RecoveryAgent wrapper is already integrated. To use the full recovery-agent capabilities:

```bash
# Recovery-agent should be in your project root
ls recovery-agent/src/exp_agent/recovery/recovery_agent.py

# If not present, clone it:
# git clone <recovery-agent-repo-url> recovery-agent
```

The agent will automatically use fallback logic if recovery-agent package is unavailable.

## Basic Usage

### 1. Simple Error Recovery

```python
from app.agents import RecoveryAgent, RecoveryInput

agent = RecoveryAgent()

# Handle a timeout error
input_data = RecoveryInput(
    error_type="timeout",
    error_message="Connection timeout after 30s",
    device_name="opentrons_ot2",
    device_status="error",
    error_severity="low",
    retry_count=0,
)

result = await agent.run(input_data)

if result.success:
    decision = result.output.decision  # "retry", "abort", "degrade", "skip"
    print(f"Recovery decision: {decision}")
    print(f"Rationale: {result.output.rationale}")

    if decision == "retry":
        print(f"Retry after {result.output.retry_delay_seconds}s")
```

### 2. Chemical Safety Escalation

```python
# Chemical safety event - automatically escalates
input_data = RecoveryInput(
    error_type="spill_detected",
    error_message="Liquid spill detected in workspace",
    device_name="opentrons_ot2",
    device_status="error",
    error_severity="high",
    telemetry={
        "spill_detected": True,
        "temperature": 25.0,
    },
)

result = await agent.run(input_data)

if result.success and result.output.chemical_safety_event:
    # Chemical safety events always result in abort
    assert result.output.decision == "abort"
    print("⚠️ CHEMICAL SAFETY EVENT - SafetyAgent veto active")
```

### 3. With Telemetry History

```python
# Provide history for fault signature analysis
input_data = RecoveryInput(
    error_type="sensor_drift",
    error_message="Temperature sensor drift detected",
    device_name="plc_controller",
    telemetry={"temperature": 45.0},
    history=[
        {"device_name": "plc_controller", "status": "idle", "telemetry": {"temperature": 25.0}},
        {"device_name": "plc_controller", "status": "running", "telemetry": {"temperature": 30.0}},
        {"device_name": "plc_controller", "status": "error", "telemetry": {"temperature": 40.0}},
        {"device_name": "plc_controller", "status": "error", "telemetry": {"temperature": 45.0}},
    ],
    retry_count=0,
)

result = await agent.run(input_data)
# Agent analyzes drift pattern and suggests appropriate strategy
```

## Integration with Orchestrator

### Option 1: Add Recovery to Execution Loop

Modify `app/agents/orchestrator.py` to call RecoveryAgent when execution fails:

```python
from app.agents import RecoveryAgent, RecoveryInput

class OrchestratorAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.recovery = RecoveryAgent()

    async def _execute_real_run(self, ...):
        """Execute run with recovery logic."""
        max_retries = 3
        retry_count = 0

        while retry_count <= max_retries:
            try:
                # Attempt execution
                result = await self._do_execute(...)
                return result

            except Exception as exc:
                # Build recovery input
                recovery_input = RecoveryInput(
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    device_name="campaign_execution",
                    device_status="error",
                    error_severity="medium",
                    retry_count=retry_count,
                )

                # Get recovery decision
                recovery_result = await self.recovery.run(recovery_input)

                if not recovery_result.success:
                    raise exc  # Recovery agent failed, propagate

                decision = recovery_result.output.decision

                if decision == "retry":
                    retry_count += 1
                    if recovery_result.output.retry_delay_seconds > 0:
                        await asyncio.sleep(recovery_result.output.retry_delay_seconds)
                    continue

                elif decision == "abort":
                    raise exc  # Abort execution

                elif decision == "degrade":
                    # Implement degraded mode (e.g., skip QC, reduce precision)
                    result = await self._do_execute(..., degraded=True)
                    return result

                else:  # skip
                    return None, {}  # Skip this candidate

        raise Exception(f"Max retries ({max_retries}) exceeded")
```

### Option 2: Event-Driven Integration

```python
# Emit recovery events for monitoring/UI
self._emit(campaign_id, {
    "type": "recovery_decision",
    "error_type": error_type,
    "decision": decision,
    "rationale": rationale,
    "retry_count": retry_count,
    "chemical_safety_event": recovery_result.output.chemical_safety_event,
})
```

## Error Types and Severities

### Supported Error Types

Recovery-agent recognizes these error types:
- **Transient**: `timeout`, `connection_lost`, `sensor_drift`
- **Hardware**: `overshoot`, `sensor_fail`, `actuator_jam`
- **Safety**: `safety_violation`, `postcondition_failed`
- **Chemical Safety**: `spill_detected`, `fire_detected`, `exposure_detected`, `thermal_runaway`

### Severity Levels

- **low**: Minor issues, high retry likelihood
- **medium**: Moderate issues, conditional retry
- **high**: Serious issues, likely abort or degrade

## Recovery Decisions

### `retry`
- Attempt operation again after optional delay
- Use `output.retry_delay_seconds` for backoff
- Check `output.max_retries` for limit

### `degrade`
- Continue with reduced functionality
- Example: Skip QC checks, reduce precision, use backup sensor

### `abort`
- Stop execution completely
- Always used for chemical safety events
- Use for unrecoverable errors

### `skip`
- Skip this specific operation, continue campaign
- Use for non-critical failures

## Safety Integration

RecoveryAgent coordinates with SafetyAgent:

1. **Chemical Safety Events**: Detected errors trigger SafetyAgent veto power
2. **Allowed Actions**: Only SAFE_SHUTDOWN, EVACUATE, or ASK_HUMAN
3. **SafetyPacket**: Optional safety constraints check before recovery actions

```python
# Check for chemical safety escalation
if result.output.chemical_safety_event:
    # SafetyAgent has veto power
    # Only abort/shutdown actions allowed
    # Normal retry strategies blocked
    print("🚨 Chemical safety event - SafetyAgent veto active")
```

## Fallback Mode

If recovery-agent package is unavailable, RecoveryAgent uses simple fallback logic:
- Retry up to 3 times with 2-second delay
- Abort after max retries exceeded
- No advanced fault signature analysis
- No chemical safety escalation

To check if full recovery-agent is available:

```python
agent = RecoveryAgent()
if agent._available:
    print("✅ Full recovery-agent capabilities active")
else:
    print("⚠️ Using fallback recovery logic")
```

## Testing

Run the test suite:

```bash
python3 -m pytest tests/test_recovery_agent.py -v
```

Test coverage:
- ✅ Basic error recovery
- ✅ Retry logic with increasing counts
- ✅ Chemical safety event escalation
- ✅ Input validation
- ✅ Telemetry history analysis
- ✅ Fallback mode

## Best Practices

1. **Always provide telemetry**: More context = better decisions
2. **Track retry counts**: Pass increasing retry_count to prevent infinite loops
3. **Respect chemical safety decisions**: Never override abort on safety events
4. **Use appropriate severity**: Match severity to actual risk
5. **Monitor recovery metrics**: Log decisions for campaign analysis

## Future Enhancements

- [ ] LLM advisor integration for complex recovery scenarios
- [ ] Orchestrator integration in `_execute_real_run()`
- [ ] Recovery metrics dashboard in Lab Agent UI
- [ ] Custom recovery policies per experiment type
- [ ] Integration with OpenTrons API error codes

## References

- Recovery-agent repository: `recovery-agent/`
- Base agent interface: `app/agents/base.py`
- Safety agent integration: `app/agents/safety_agent.py`
- Orchestrator: `app/agents/orchestrator.py`
