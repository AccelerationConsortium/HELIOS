# Scenario: Sensor Fail (unreliable or impossible readings)

## Signals required
- `sensor_health` OR raw sensor readings + validation rules
- `current_temp` (may be invalid)
- `heater_power`

## Trigger
Any of:
- Sensor returns sentinel values (e.g., `-999`, `NaN`, empty)
- Sudden impossible jump (> `[JUMP_DELTA]` within `[SECONDS]`)
- Frozen value (no change) while actuator state implies change
- Explicit `sensor_health == bad`

## Allowed decisions (by capability)
- **L0**: ESCALATE only (treat as UNSAFE)
- **L1**: ABORT (+ ESCALATE) — enter safe state; do not attempt DEGRADE
- **L2**:
  - If an independent verification exists (secondary sensor, self-test): attempt `RESTORE_SENSOR_TRUST` then decide
  - Otherwise: ABORT (+ ESCALATE)

## Intents (ordered)
1) `ENTER_SAFE_STATE`
2) `RESTORE_SENSOR_TRUST` (L2 only: self-test / reset / reinit)
3) `RUN_SELF_TEST` (if available)
4) `NOTIFY_HUMAN`

## Verification (must pass)
- Heater is off (`heater_power == 0` or equivalent)
- If attempting restore: sensor becomes consistent for `[WINDOW]` samples

## Escalation criteria
- Always escalate after entering safe state (this is a serious fault)
- Self-test fails or sensor remains unreliable

## Notes
- In this scenario, “continue operating” is usually not acceptable unless you have independent verification.
