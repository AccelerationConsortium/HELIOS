# Scenario: Overshoot (temperature above target)

## Signals required
- `current_temp`
- `target_temp`
- `max_temp` (or policy constant)
- `sensor_health` (or reliability proxy)

## Trigger
- `current_temp > target_temp + [DELTA]` for > `[WINDOW_SECONDS]`

## Allowed decisions (by capability)
- **L0**: ESCALATE only (cannot guarantee safe state transitions)
- **L1**: ABORT (+ ESCALATE) — enter safe state via cooling/stop
- **L2**:
  - If `current_temp < max_temp` AND `sensor_health == ok`: DEGRADE allowed
  - Else: ABORT (+ ESCALATE)

## Intents (ordered)
1) `REDUCE_RISK_HEAT`
2) `LOWER_TARGET` (L2 only)
3) `ENTER_SAFE_STATE` (if risk remains)
4) `NOTIFY_HUMAN` (if budgets exceeded or uncertainty)

## Verification (must pass)
- Heating is off OR temperature is trending down within `[X]` seconds
- If degraded: `target_temp` updated AND overshoot no longer increasing

## Escalation criteria
- `sensor_health != ok` or missing
- `current_temp >= max_temp - [SAFETY_MARGIN]`
- Unable to verify cooldown/stop
- Repeated overshoot after DEGRADE

## Notes
- DEGRADE should be a **single-step** reduction (e.g., `new_target = target - step`), bounded by `max_degradations`.
