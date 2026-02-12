# Scenario: Timeout (device did not reach expected state in time)

## Signals required
- `current_temp`
- `target_temp`
- `device_heartbeat` (or comms health)
- `heater_power`

## Trigger
- Expected progress not observed for > `[TIMEOUT_SECONDS]`
  - e.g., `abs(current_temp - target_temp)` not decreasing

## Allowed decisions (by capability)
- **L0**: ESCALATE only
- **L1**: RETRY allowed if comms/heartbeat ok AND state appears safe; otherwise ABORT
- **L2**: RETRY preferred; optionally RESET_DEVICE if retries exhausted; DEGRADE only if safe and verifiable

## Intents (ordered)
1) `RETRY_COMMUNICATION` (if heartbeat/comms suspect)
2) `RETRY` (repeat the last safe action)
3) `RESET_DEVICE` (L2 only; bounded)
4) `DEGRADE` (L2 only; e.g., lower target, slower ramp)
5) `ENTER_SAFE_STATE` if progress remains absent
6) `NOTIFY_HUMAN` if budgets exceeded

## Verification (must pass)
- After retry/reset: progress resumes OR state transitions to safe state

## Escalation criteria
- Heartbeat missing
- Repeated timeouts beyond retry budget
- Unable to verify device state

## Notes
- Timeouts are often transient; autonomy here can be high as long as safety signals are reliable.
